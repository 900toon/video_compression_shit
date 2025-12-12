"""
JPEG Encoder - Baseline DCT Implementation (Adaptive Quantization Version)

This module implements a standard JPEG compression algorithm with
Adaptive Quantization improvements. It includes:

1. Color Space Conversion: RGB -> YCbCr
2. Chroma Subsampling: 4:2:0
3. DCT: Discrete Cosine Transform
4. Adaptive Quantization: Dynamically adjusts quantization based on block complexity
5. Zigzag Scan
6. Huffman Coding
7. JFIF Formatting

Key Class:
- JPEGEncoder: Complete encoder implementation

Helper Functions:
- encode_image(): Quickly encode bytes
- save_jpeg(): Quickly save to file
"""

import numpy as np
from typing import Tuple, List, Dict
import struct


class JPEGEncoder:
    """
    Baseline JPEG Encoder with Adaptive Quantization

    Usage:
        encoder = JPEGEncoder(quality=75)
        jpeg_data = encoder.encode(rgb_image)
        # Or save directly
        encoder.encode_file(rgb_image, 'output.jpg')
    """

    # ==================== Standard JPEG Quantization Tables ====================
    # These tables control the balance between compression ratio and quality.
    # Higher values mean higher compression but lower quality.

    # Luminance (Y) Quantization Table
    LUMINANCE_QUANT_TABLE = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float32)

    # Chrominance (Cb, Cr) Quantization Table
    CHROMINANCE_QUANT_TABLE = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 32767],
        [99, 99, 99, 99, 99, 99, 32767, 32767]
    ], dtype=np.float32)

    # ==================== Zigzag Scan Pattern ====================
    # Reorders the 8x8 2D array into a 1D sequence.
    # Low-frequency coefficients come first, high-frequency ones last.
    ZIGZAG_PATTERN = np.array([
        0,  1,  8, 16,  9,  2,  3, 10,
        17, 24, 32, 25, 18, 11,  4,  5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13,  6,  7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63
    ])

    # ==================== Standard Huffman Tables ====================
    # BITS: Number of codes of each length (16 elements)
    # VALS: The symbol values sorted by code length

    # DC Luminance Huffman Table
    _DC_LUM_BITS = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    _DC_LUM_VALS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # DC Chrominance Huffman Table
    _DC_CHR_BITS = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    _DC_CHR_VALS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # AC Luminance Huffman Table
    _AC_LUM_BITS = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125]
    _AC_LUM_VALS = [
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
        0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
        0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
        0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
        0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
        0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
        0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
        0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
        0xF9, 0xFA
    ]

    # AC Chrominance Huffman Table
    _AC_CHR_BITS = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119]
    _AC_CHR_VALS = [
        0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
        0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0,
        0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
        0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
        0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
        0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
        0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
        0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
        0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
        0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
        0xF9, 0xFA
    ]

    def _apply_freq_weighting(self, base_q: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """
        Simple frequency weighting for quantization tables.

        - 'strength' controls how aggressively we increase steps for
          higher frequencies.
        - We keep DC (0,0) almost untouched and gradually increase the
          step size toward the bottom-right corner (high frequencies).
        """
        H, W = base_q.shape
        assert H == 8 and W == 8

        # Normalized distance from DC in zigzag sense (approximate)
        weight = np.zeros_like(base_q, dtype=np.float32)
        for u in range(8):
            for v in range(8):
                # simple radial distance from DC
                d = (u + v) / 14.0      # in [0, 1]
                weight[u, v] = 1.0 + strength * (d ** 2)

        # Keep DC almost unchanged
        weight[0, 0] = 1.0

        return base_q * weight

    def __init__(self, quality: int = 75):
        """
        Initialize the JPEG Encoder.

        Args:
            quality: JPEG quality factor (1-100), default 75
        """
        self.quality = int(np.clip(quality, 1, 100))

        # Base quantization tables (scaled by quality)
        # These tables are written to the file header for the Decoder to use.
        # self.lum_quant_table = self._scale_quantization_table(
        #     self.LUMINANCE_QUANT_TABLE, self.quality
        # )
        # self.chrom_quant_table = self._scale_quantization_table(
        #     self.CHROMINANCE_QUANT_TABLE, self.quality
        # )

        # Apply frequency weighting for adaptive quantization
        self.lum_quant_table = self._apply_freq_weighting(
            self._scale_quantization_table(self.LUMINANCE_QUANT_TABLE, self.quality), strength=0.4
        )
        self.chrom_quant_table = self._apply_freq_weighting(
            self._scale_quantization_table(self.CHROMINANCE_QUANT_TABLE, self.quality), strength=0.6
        )

        # # Build Huffman maps
        self.dc_lum_huffman = self._build_huffman_map(self._DC_LUM_BITS, self._DC_LUM_VALS)
        self.dc_chr_huffman = self._build_huffman_map(self._DC_CHR_BITS, self._DC_CHR_VALS)
        self.ac_lum_huffman = self._build_huffman_map(self._AC_LUM_BITS, self._AC_LUM_VALS)
        self.ac_chr_huffman = self._build_huffman_map(self._AC_CHR_BITS, self._AC_CHR_VALS)

        """ change here to use optimized Huffman tables """
        # === Histograms for two-pass Huffman optimization ===
        # DC categories 0..11
        self.dc_lum_hist = np.zeros(12, dtype=np.int64)
        self.dc_chr_hist = np.zeros(12, dtype=np.int64)

        # AC symbols 0x00..0xFF (JPEG-defined subset will actually be used)
        self.ac_lum_hist = np.zeros(256, dtype=np.int64)
        self.ac_chr_hist = np.zeros(256, dtype=np.int64)

        # Precompute DCT tables
        self._init_dct_tables()

    def _init_dct_tables(self):
        """Precompute cosine tables for DCT."""
        self._cos_table = np.zeros((8, 8, 8, 8), dtype=np.float32)
        self._alpha = np.zeros(8, dtype=np.float32)

        for i in range(8):
            self._alpha[i] = np.sqrt(1/8) if i == 0 else np.sqrt(2/8)

        for u in range(8):
            for v in range(8):
                for x in range(8):
                    for y in range(8):
                        self._cos_table[u, v, x, y] = (
                            np.cos((2*x + 1) * u * np.pi / 16) *
                            np.cos((2*y + 1) * v * np.pi / 16)
                        )

    @staticmethod
    def _scale_quantization_table(base_table: np.ndarray, quality: int) -> np.ndarray:
        """Scale the quantization table based on quality factor (IJG formula)."""
        if quality < 50:
            scale = 5000 / quality
        else:
            scale = 200 - 2 * quality

        scaled_table = np.floor((base_table * scale + 50) / 100)
        scaled_table = np.clip(scaled_table, 1, 255)
        return scaled_table.astype(np.float32)

    @staticmethod
    def _build_huffman_map(bits: List[int], values: List[int]) -> Dict[int, Tuple[int, int]]:
        """Build Huffman mapping table from BITS and VALUES arrays."""
        huffman_map = {}
        code = 0
        val_idx = 0

        for length in range(1, 17):
            count = bits[length - 1]
            for _ in range(count):
                if val_idx < len(values):
                    symbol = values[val_idx]
                    huffman_map[symbol] = (code, length)
                    code += 1
                    val_idx += 1
            code <<= 1

        return huffman_map
    
    def _build_optimized_huffman_tables(self) -> None:
        """
        Build per-image "optimized" Huffman tables using a simple
        reorder-by-frequency strategy.

        We keep the BITS[] arrays from the standard JPEG tables
        (so code lengths and the total number of symbols per length
        stay JPEG-compliant), and only reorder the VALUES[] arrays
        so that more frequent symbols are assigned to shorter codes.

        This is not mathematically optimal, but:
        - It is simple and robust.
        - It usually saves a few percent of bits compared to the
          fixed standard tables.
        """

        def reorder(bits: List[int], base_vals: List[int],
                    hist: np.ndarray) -> Tuple[List[int], List[int]]:
            """
            Given a fixed BITS[] and a list of VALUES[],
            sort the VALUES by descending symbol frequency.

            Ties are resolved by the original order to keep it stable.
            """
            vals = list(base_vals)
            # (index, symbol, frequency)
            freq_entries = []
            for idx, sym in enumerate(vals):
                count = 0
                if 0 <= sym < hist.shape[0]:
                    count = int(hist[sym])
                freq_entries.append((idx, sym, count))

            # Sort by frequency (desc), then by original index (asc)
            freq_entries.sort(key=lambda t: (-t[2], t[0]))

            new_vals = [sym for (_, sym, _) in freq_entries]
            return list(bits), new_vals

        # DC luminance
        self.dc_lum_bits, self.dc_lum_vals = reorder(
            self._DC_LUM_BITS, self._DC_LUM_VALS, self.dc_lum_hist
        )
        # DC chroma
        self.dc_chr_bits, self.dc_chr_vals = reorder(
            self._DC_CHR_BITS, self._DC_CHR_VALS, self.dc_chr_hist
        )
        # AC luminance
        self.ac_lum_bits, self.ac_lum_vals = reorder(
            self._AC_LUM_BITS, self._AC_LUM_VALS, self.ac_lum_hist
        )
        # AC chroma
        self.ac_chr_bits, self.ac_chr_vals = reorder(
            self._AC_CHR_BITS, self._AC_CHR_VALS, self.ac_chr_hist
        )

        # Rebuild code maps from the newly ordered tables
        self.dc_lum_huffman = self._build_huffman_map(self.dc_lum_bits, self.dc_lum_vals)
        self.dc_chr_huffman = self._build_huffman_map(self.dc_chr_bits, self.dc_chr_vals)
        self.ac_lum_huffman = self._build_huffman_map(self.ac_lum_bits, self.ac_lum_vals)
        self.ac_chr_huffman = self._build_huffman_map(self.ac_chr_bits, self.ac_chr_vals)

    @staticmethod
    def rgb_to_ycbcr(rgb_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert RGB to YCbCr color space."""
        rgb = rgb_image.astype(np.float32)
        R = rgb[:, :, 0]
        G = rgb[:, :, 1]
        B = rgb[:, :, 2]

        Y  =  0.299 * R + 0.587 * G + 0.114 * B
        Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128.0
        Cr =  0.5 * R - 0.418688 * G - 0.081312 * B + 128.0

        return np.clip(Y, 0, 255), np.clip(Cb, 0, 255), np.clip(Cr, 0, 255)

    @staticmethod
    def subsample_chroma(channel: np.ndarray) -> np.ndarray:
        """Perform 4:2:0 subsampling on chrominance channels."""
        h, w = channel.shape
        return channel.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))

    def dct_2d(self, block: np.ndarray) -> np.ndarray:
        """Perform 2D Discrete Cosine Transform."""
        block = block.astype(np.float32) - 128.0
        dct_result = np.zeros((8, 8), dtype=np.float32)

        for u in range(8):
            for v in range(8):
                sum_val = np.sum(block * self._cos_table[u, v])
                dct_result[u, v] = self._alpha[u] * self._alpha[v] * sum_val

        return dct_result


    @staticmethod
    def quantize(dct_block: np.ndarray, quant_table: np.ndarray) -> np.ndarray:
        """Quantize DCT coefficients using the specific table."""
        return np.round(dct_block / quant_table).astype(np.int32)

    def zigzag_scan(self, block: np.ndarray) -> np.ndarray:
        """Perform Zigzag scan to reorder coefficients."""
        return block.flatten()[self.ZIGZAG_PATTERN]

    @staticmethod
    def encode_coefficient_value(value: int) -> Tuple[int, int]:
        """Encode coefficient value into (size, amplitude)."""
        if value == 0:
            return 0, 0
        abs_val = abs(value)
        size = abs_val.bit_length()
        amplitude = value if value > 0 else value + (1 << size) - 1
        return size, amplitude

    def encode_dc(self, dc_diff: int, is_luminance: bool = True) -> List[Tuple[int, int]]:
        """Encode DC coefficient difference."""
        size, amplitude = self.encode_coefficient_value(dc_diff)
        huffman_table = self.dc_lum_huffman if is_luminance else self.dc_chr_huffman

        if size not in huffman_table:
            size = min(size, 11)

        code, length = huffman_table[size]
        result = [(code, length)]

        if size > 0:
            amplitude_bits = amplitude & ((1 << size) - 1)
            result.append((amplitude_bits, size))
        return result

    def encode_ac(self, ac_coeffs: np.ndarray, is_luminance: bool = True) -> List[Tuple[int, int]]:
        """Encode AC coefficients using RLE."""
        huffman_table = self.ac_lum_huffman if is_luminance else self.ac_chr_huffman
        result = []
        zero_run = 0

        last_nonzero = -1
        for i in range(len(ac_coeffs) - 1, -1, -1):
            if ac_coeffs[i] != 0:
                last_nonzero = i
                break

        if last_nonzero == -1:
            code, length = huffman_table[0x00]
            result.append((code, length))
            return result

        for i in range(last_nonzero + 1):
            coeff = ac_coeffs[i]
            if coeff == 0:
                zero_run += 1
            else:
                while zero_run >= 16:
                    code, length = huffman_table[0xF0]
                    result.append((code, length))
                    zero_run -= 16

                size, amplitude = self.encode_coefficient_value(int(coeff))
                if size > 10:
                    size = 10
                    amplitude = amplitude & ((1 << size) - 1)

                symbol = (zero_run << 4) | size
                if symbol in huffman_table:
                    code, length = huffman_table[symbol]
                    result.append((code, length))
                    if size > 0:
                        amplitude_bits = amplitude & ((1 << size) - 1)
                        result.append((amplitude_bits, size))
                zero_run = 0

        if last_nonzero < len(ac_coeffs) - 1:
            code, length = huffman_table[0x00]
            result.append((code, length))
        return result

    def bits_to_bytes(self, bit_stream: List[Tuple[int, int]]) -> bytes:
        """Convert bit stream to byte stream with byte stuffing."""
        result = bytearray()
        current_byte = 0
        bits_in_byte = 0

        for value, num_bits in bit_stream:
            for i in range(num_bits - 1, -1, -1):
                bit = (value >> i) & 1
                current_byte = (current_byte << 1) | bit
                bits_in_byte += 1

                if bits_in_byte == 8:
                    result.append(current_byte)
                    if current_byte == 0xFF:
                        result.append(0x00)
                    current_byte = 0
                    bits_in_byte = 0

        if bits_in_byte > 0:
            current_byte <<= (8 - bits_in_byte)
            current_byte |= (1 << (8 - bits_in_byte)) - 1
            result.append(current_byte)
            if current_byte == 0xFF:
                result.append(0x00)

        return bytes(result)

    def create_jfif_header(self, width: int, height: int) -> bytes:
        """Create standard JFIF Header."""
        header = bytearray()
        header.extend(b'\xFF\xD8') # SOI
        header.extend(b'\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00') # APP0

        # DQT (Luminance)
        header.extend(b'\xFF\xDB\x00\x43\x00')
        header.extend(self.zigzag_scan(self.lum_quant_table).astype(np.uint8).tobytes())

        # DQT (Chrominance)
        header.extend(b'\xFF\xDB\x00\x43\x01')
        header.extend(self.zigzag_scan(self.chrom_quant_table).astype(np.uint8).tobytes())

        # SOF0
        header.extend(b'\xFF\xC0\x00\x11\x08')
        header.extend(struct.pack('>H', height))
        header.extend(struct.pack('>H', width))
        header.extend(b'\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01')

        # DHTs
        """ change here to use optimized Huffman tables """
        # header.extend(self._create_huffman_table_segment(0, True, self._DC_LUM_BITS, self._DC_LUM_VALS))
        # header.extend(self._create_huffman_table_segment(0, False, self._AC_LUM_BITS, self._AC_LUM_VALS))
        # header.extend(self._create_huffman_table_segment(1, True, self._DC_CHR_BITS, self._DC_CHR_VALS))
        # header.extend(self._create_huffman_table_segment(1, False, self._AC_CHR_BITS, self._AC_CHR_VALS))

        # === DHT (Define Huffman Tables) ===
        # We use the current per-instance tables, which may have been
        # re-ordered by the two-pass optimization.
        # DC luminance
        header.extend(self._create_huffman_table_segment(0, True, self.dc_lum_bits, self.dc_lum_vals))
        # AC luminance
        header.extend(self._create_huffman_table_segment(0, False, self.ac_lum_bits, self.ac_lum_vals))
        # DC chroma
        header.extend(self._create_huffman_table_segment(1, True, self.dc_chr_bits, self.dc_chr_vals))
        # AC chroma
        header.extend(self._create_huffman_table_segment(1, False, self.ac_chr_bits, self.ac_chr_vals))

        # SOS
        header.extend(b'\xFF\xDA\x00\x0C\x03\x01\x00\x02\x11\x03\x11\x00\x3F\x00')
        return bytes(header)

    def _create_huffman_table_segment(self, table_id: int, is_dc: bool,
                                       bits: List[int], values: List[int]) -> bytes:
        """Create a DHT segment bytes."""
        segment = bytearray()
        segment.extend(b'\xFF\xC4')
        length = 2 + 1 + 16 + len(values)
        segment.extend(struct.pack('>H', length))
        table_class = 0 if is_dc else 1
        segment.append((table_class << 4) | table_id)
        segment.extend(bytes(bits))
        segment.extend(bytes(values))
        return bytes(segment)

    def _get_block_symbols(self, block: np.ndarray, quant_table: np.ndarray,
                           prev_dc: int, is_luminance: bool) -> int:
        """
        first pass: process a single 8x8 block to update the DC/AC histograms.
        Returns the current DC value for next block's DC difference calculation.
        """
        # Select appropriate histograms
        if is_luminance:
            dc_hist = self.dc_lum_hist
            ac_hist = self.ac_lum_hist
        else:
            dc_hist = self.dc_chr_hist
            ac_hist = self.ac_chr_hist

        # 1. DCT
        dct_block = self.dct_2d(block)

        # 2. Quantization
        quant_block = self.quantize(dct_block, quant_table)

        # 3. Zigzag Scan
        zigzag = self.zigzag_scan(quant_block)
        zigzag = zigzag.astype(np.int32)

        # ===== DC =====
        dc = int(zigzag[0])
        dc_diff = dc - prev_dc

        size, amplitude = self.encode_coefficient_value(dc_diff)
        # DC Huffman only considers the category (size), not the amplitude
        if 0 <= size < dc_hist.shape[0]:
            dc_hist[size] += 1

        # ===== AC =====
        ac_coeffs = zigzag[1:]  # 63 AC coefficients

        # Find the last non-zero coefficient position (same as encode_ac)
        last_nonzero = -1
        for i in range(len(ac_coeffs) - 1, -1, -1):
            if ac_coeffs[i] != 0:
                last_nonzero = i
                break

        if last_nonzero == -1:
            # All zeros: only one EOB (0x00)
            ac_hist[0x00] += 1
            return dc

        zero_run = 0

        for i in range(last_nonzero + 1):
            coeff = int(ac_coeffs[i])

            if coeff == 0:
                zero_run += 1
            else:
                # Runs of zeros longer than 16 are split into ZRL (0xF0) symbols
                while zero_run >= 16:
                    ac_hist[0xF0] += 1  # ZRL == (15 << 4) | 0
                    zero_run -= 16

                size, amplitude = self.encode_coefficient_value(coeff)
                # AC symbol = (run << 4) | size
                symbol = (zero_run << 4) | size
                ac_hist[symbol] += 1

                zero_run = 0

        # If there are trailing zeros, add an EOB symbol
        if last_nonzero < len(ac_coeffs) - 1:
            ac_hist[0x00] += 1  # EOB

        return dc

    def _encode_block(self, block: np.ndarray, base_quant_table: np.ndarray,
                      prev_dc: int, is_luminance: bool) -> Tuple[List[Tuple[int, int]], int]:
        """
        Encode a single 8x8 block with ADAPTIVE THRESHOLDING.
        """
        # 1. DCT
        dct_block = self.dct_2d(block)

        # 2. Quantize using the STANDARD Base Table
        quant_block = self.quantize(dct_block, base_quant_table)

        # 3. Zigzag Scan (Get 1D indices)
        zigzag = self.zigzag_scan(quant_block) # Note: zigzag is 1D array of values now

        # 4. DC Encoding
        dc = int(zigzag[0])
        dc_diff = dc - prev_dc
        encoded_data = self.encode_dc(dc_diff, is_luminance)

        # 5. AC Encoding
        encoded_data.extend(self.encode_ac(zigzag[1:], is_luminance))

        return encoded_data, dc

    def encode(self, rgb_image: np.ndarray) -> bytes:
        """
        Encode an RGB image into a JPEG bitstream using a
        two-pass Huffman optimization.

        Pass 1:
            - Pad the image to multiples of 16.
            - Convert to YCbCr and 4:2:0 subsample.
            - Walk through all 8x8 blocks in MCU order and
              update DC/AC histograms via _get_block_symbols()
              (no actual bitstream is produced in this pass).

        Pass 2:
            - Rebuild the Huffman tables based on the collected
              histograms (reorder-by-frequency).
            - Re-encode all blocks using _encode_block() with the
              optimized Huffman tables to produce the final bitstream.
        """
        # Original (unpadded) dimensions
        original_height, original_width = rgb_image.shape[:2]

        # === Step 1: pad image to multiples of 16 ===
        # Each MCU covers 16x16 luma samples (4 blocks of 8x8).
        h_padding = (16 - (original_height % 16)) % 16
        w_padding = (16 - (original_width % 16)) % 16

        if h_padding > 0 or w_padding > 0:
            rgb_padded = np.pad(
                rgb_image,
                ((0, h_padding), (0, w_padding), (0, 0)),
                mode="edge"  # replicate border pixels
            )
        else:
            rgb_padded = rgb_image

        padded_h, padded_w = rgb_padded.shape[:2]

        # === Step 2: color transform ===
        Y, Cb, Cr = self.rgb_to_ycbcr(rgb_padded)

        # === Step 3: 4:2:0 chroma subsampling ===
        Cb_sub = self.subsample_chroma(Cb)
        Cr_sub = self.subsample_chroma(Cr)

        # === Pass 1: collect symbol histograms ===
        # Reset histograms
        self.dc_lum_hist.fill(0)
        self.dc_chr_hist.fill(0)
        self.ac_lum_hist.fill(0)
        self.ac_chr_hist.fill(0)

        prev_dc_y = 0
        prev_dc_cb = 0
        prev_dc_cr = 0

        # Traverse all MCUs (16x16 luma area)
        for mcu_row in range(0, padded_h, 16):
            for mcu_col in range(0, padded_w, 16):

                # --- Four Y blocks (2x2) in MCU order: TL, TR, BL, BR ---
                for block_row in range(2):
                    for block_col in range(2):
                        y_row = mcu_row + block_row * 8
                        y_col = mcu_col + block_col * 8
                        block = Y[y_row:y_row + 8, y_col:y_col + 8]

                        prev_dc_y = self._get_block_symbols(
                            block, self.lum_quant_table, prev_dc_y, True
                        )

                # --- One Cb block (subsampled by 2 in each dimension) ---
                cb_row = mcu_row // 2
                cb_col = mcu_col // 2
                block = Cb_sub[cb_row:cb_row + 8, cb_col:cb_col + 8]
                prev_dc_cb = self._get_block_symbols(
                    block, self.chrom_quant_table, prev_dc_cb, False
                )

                # --- One Cr block ---
                block = Cr_sub[cb_row:cb_row + 8, cb_col:cb_col + 8]
                prev_dc_cr = self._get_block_symbols(
                    block, self.chrom_quant_table, prev_dc_cr, False
                )

        # === Build optimized Huffman tables from the collected histograms ===
        self._build_optimized_huffman_tables()

        # === Pass 2: actual entropy coding using the optimized tables ===
        all_data: List[Tuple[int, int]] = []  # (code_bits, length) pairs
        prev_dc_y = 0
        prev_dc_cb = 0
        prev_dc_cr = 0

        for mcu_row in range(0, padded_h, 16):
            for mcu_col in range(0, padded_w, 16):

                # Encode 4 Y blocks in MCU order
                for block_row in range(2):
                    for block_col in range(2):
                        y_row = mcu_row + block_row * 8
                        y_col = mcu_col + block_col * 8
                        block = Y[y_row:y_row + 8, y_col:y_col + 8]

                        bits, prev_dc_y = self._encode_block(
                            block, self.lum_quant_table, prev_dc_y, True
                        )
                        all_data.extend(bits)

                # Encode 1 Cb block
                cb_row = mcu_row // 2
                cb_col = mcu_col // 2
                block = Cb_sub[cb_row:cb_row + 8, cb_col:cb_col + 8]
                bits, prev_dc_cb = self._encode_block(
                    block, self.chrom_quant_table, prev_dc_cb, False
                )
                all_data.extend(bits)

                # Encode 1 Cr block
                block = Cr_sub[cb_row:cb_row + 8, cb_col:cb_col + 8]
                bits, prev_dc_cr = self._encode_block(
                    block, self.chrom_quant_table, prev_dc_cr, False
                )
                all_data.extend(bits)

        # === Step 4: assemble final JPEG bitstream ===
        jpeg_data = (
            self.create_jfif_header(original_width, original_height) +
            self.bits_to_bytes(all_data) +
            b"\xFF\xD9"  # EOI (End Of Image)
        )

        return jpeg_data


    def encode_file(self, rgb_image: np.ndarray, output_path: str) -> int:
        """Encode and save to file."""
        jpeg_data = self.encode(rgb_image)
        with open(output_path, 'wb') as f:
            f.write(jpeg_data)
        print(f"JPEG Saved: {output_path} ({len(jpeg_data)} bytes)")
        return len(jpeg_data)


# ==================== Convenience Functions ====================

def encode_image(rgb_image: np.ndarray, quality: int = 75) -> bytes:
    encoder = JPEGEncoder(quality=quality)
    return encoder.encode(rgb_image)

def save_jpeg(rgb_image: np.ndarray, output_path: str, quality: int = 75) -> int:
    encoder = JPEGEncoder(quality=quality)
    return encoder.encode_file(rgb_image, output_path)

if __name__ == "__main__":
    print("JPEG Encoder with Adaptive Quantization Test")
    test_image = np.zeros((256, 256, 3), dtype=np.uint8)

    # generate a test pattern (gradient)
    for i in range(256):
        for j in range(256):
            test_image[i, j, 0] = i          # Red gradient (vertical)
            test_image[i, j, 1] = j          # Green gradient (horizontal)
            test_image[i, j, 2] = (i + j) // 2  # Blue gradient (diagonal)

    # Test different quality settings
    for quality in [50, 75, 90]:
        print(f"\nEncoding with quality {quality}...")
        encoder = JPEGEncoder(quality=quality)
        output_path = f"test_q{quality}.jpg"
        size = encoder.encode_file(test_image, output_path)
        print(f"  Output size: {size} bytes")
        print(f"  Compression ratio: {256*256*3/size:.2f}x")

    print("\nTest completed!")