"""
JPEG Decoder - Baseline DCT Implementation

This module implements the reverse process of standard JPEG encoding:
1. Parse JFIF Header and Markers (SOI, DQT, DHT, SOF0, SOS)
2. Entropy Decoding (Huffman + RLE + DPCM)
3. Dequantization
4. Inverse Discrete Cosine Transform (IDCT)
5. Chroma Upsampling
6. Color Space Conversion (YCbCr -> RGB)

Main Class:
- JPEGDecoder: The complete decoder implementation
"""

import numpy as np
import math
import struct
from typing import Tuple, List, Dict

class JPEGDecoder:
    def __init__(self):
        # Store tables parsed from the file
        self.quant_tables = {}    # {table_id: 8x8 numpy array}
        self.huffman_tables = {}  # {(class, id): lookup_dict}
        self.image_info = {}      # width, height, components
        self.q_scale = 1.0        # For potential quality adjustment
        
        # Zigzag reverse mapping table (maps 1D index back to 2D coordinates)
        self.zigzag_map = self._create_zigzag_map()
        
        # Precompute IDCT cosine tables
        self._init_idct_tables()

    def _create_zigzag_map(self) -> List[Tuple[int, int]]:
        """Create the reverse mapping for Zigzag scan (index -> (row, col))"""
        zigzag_flat = [
            0,  1,  8, 16,  9,  2,  3, 10,
            17, 24, 32, 25, 18, 11,  4,  5,
            12, 19, 26, 33, 40, 48, 41, 34,
            27, 20, 13,  6,  7, 14, 21, 28,
            35, 42, 49, 56, 57, 50, 43, 36,
            29, 22, 15, 23, 30, 37, 44, 51,
            58, 59, 52, 45, 38, 31, 39, 46,
            53, 60, 61, 54, 47, 55, 62, 63
        ]
        mapping = [(0, 0)] * 64
        for i, idx in enumerate(zigzag_flat):
            row = idx // 8
            col = idx % 8
            mapping[i] = (row, col)
        return mapping

    def _init_idct_tables(self):
        """Precompute cosine values needed for IDCT to speed up calculation"""
        self._cos_table = np.zeros((8, 8, 8, 8), dtype=np.float32)
        self._alpha = np.zeros(8, dtype=np.float32)

        for i in range(8):
            self._alpha[i] = math.sqrt(1/8) if i == 0 else math.sqrt(2/8)

        for x in range(8):
            for y in range(8):
                for u in range(8):
                    for v in range(8):
                        self._cos_table[x, y, u, v] = (
                            math.cos((2*x + 1) * u * math.pi / 16) *
                            math.cos((2*y + 1) * v * math.pi / 16)
                        )

    def idct_2d(self, block: np.ndarray) -> np.ndarray:
        """
        Perform 8x8 Inverse Discrete Cosine Transform (IDCT).
        F(x,y) = sum_u sum_v alpha(u)alpha(v) C(u,v) cos(...)cos(...)
        """
        # This is the most intuitive O(N^4) implementation for educational clarity.
        # Production environments typically use fast algorithms (e.g., Loeffler algorithm).
        output = np.zeros((8, 8), dtype=np.float32)
        
        # Use numpy broadcasting to accelerate
        # output[x, y] = sum(block * alpha_u * alpha_v * cos_table[x, y])
        
        # Precompute weighted coefficient matrix
        weighted_block = block * np.outer(self._alpha, self._alpha)
        
        for x in range(8):
            for y in range(8):
                output[x, y] = np.sum(weighted_block * self._cos_table[x, y])
                
        return output + 128.0 # Level shift restoration (-128..127 -> 0..255)

    def dequantize(self, block: np.ndarray, quant_table: np.ndarray) -> np.ndarray:
        """Dequantization: Coefficient * Quantization Table"""
        return block * quant_table

    def upsample_chroma(self, cb: np.ndarray, cr: np.ndarray, output_h: int, output_w: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple Nearest Neighbor Upsampling.
        Scales chroma channels back to original dimensions (for 4:2:0).
        """
        # Use numpy's repeat function to scale up
        # Repeat rows first, then columns
        cb_up = cb.repeat(2, axis=0).repeat(2, axis=1)
        cr_up = cr.repeat(2, axis=0).repeat(2, axis=1)
        
        # Ensure dimensions match (handle edge differences caused by padding)
        return cb_up[:output_h, :output_w], cr_up[:output_h, :output_w]

    def ycbcr_to_rgb(self, Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
        """Color Space Conversion: YCbCr -> RGB"""
        # Convert to float for calculation
        Y = Y.astype(np.float32)
        Cb = Cb.astype(np.float32) - 128.0
        Cr = Cr.astype(np.float32) - 128.0
        
        R = Y + 1.402 * Cr
        G = Y - 0.344136 * Cb - 0.714136 * Cr
        B = Y + 1.772 * Cb
        
        # Stack and clip range
        rgb = np.stack([R, G, B], axis=-1)
        return np.clip(rgb, 0, 255).astype(np.uint8)

    # ================= Bitstream Reader =================
    class BitStream:
        """Handles JPEG bitstream reading, including byte stuffing removal (0xFF00 -> 0xFF)"""
        def __init__(self, data):
            self.data = data
            self.pos = 0
            self.bit_buffer = 0
            self.bits_count = 0
            
        def read_bit(self) -> int:
            """Read 1 bit"""
            if self.bits_count == 0:
                self._fill_buffer()
            
            self.bits_count -= 1
            return (self.bit_buffer >> self.bits_count) & 1

        def read_bits(self, n: int) -> int:
            """Read n bits"""
            val = 0
            for _ in range(n):
                val = (val << 1) | self.read_bit()
            return val

        def _fill_buffer(self):
            """Fill buffer from byte stream, handling 0xFF00 stuffing"""
            if self.pos >= len(self.data):
                # End of file, pad with zeros
                self.bit_buffer = 0
                self.bits_count = 0
                return

            byte = self.data[self.pos]
            self.pos += 1

            # Handle JPEG Byte Stuffing
            # If 0xFF is encountered, check the next byte
            if byte == 0xFF:
                if self.pos < len(self.data):
                    next_byte = self.data[self.pos]
                    if next_byte == 0x00:
                        self.pos += 1 # Skip stuffed 0x00
                    # Note: If next_byte is not 0x00, it might be a Marker
                    # In SOS data stream, usually only 0xFF00 or RST markers exist

            self.bit_buffer = byte
            self.bits_count = 8

    # ================= Huffman Decoding =================
    def decode_huffman(self, stream: BitStream, table_class: int, table_id: int) -> int:
        """
        Decode the next Huffman symbol from the bitstream
        using the built Lookup Table.
        """
        # Get the corresponding Huffman table
        huff_table = self.huffman_tables.get((table_class, table_id))
        if not huff_table:
            raise ValueError(f"Huffman table not found: class={table_class}, id={table_id}")
            
        code = 0
        length = 0
        
        # Read bit by bit until a match is found (This is a simple but less efficient traversal)
        # Optimized versions should use binary trees or efficient lookup structures
        while True:
            bit = stream.read_bit()
            code = (code << 1) | bit
            length += 1
            
            if length > 16:
                raise ValueError("Huffman decode overflow")
                
            # Check if (length, code) exists in the table
            if (length, code) in huff_table:
                return huff_table[(length, code)]

    def build_huffman_table(self, bits: List[int], values: List[int]) -> Dict[Tuple[int, int], int]:
        """
        Build Huffman Lookup Table from BITS and VALUES arrays
        Key: (length, code), Value: symbol
        """
        huff_map = {}
        code = 0
        val_idx = 0
        
        for length in range(1, 17):
            count = bits[length - 1]
            for _ in range(count):
                huff_map[(length, code)] = values[val_idx]
                code += 1
                val_idx += 1
            code <<= 1
            
        return huff_map

    def decode_block(self, stream: BitStream, dc_table_id: int, ac_table_id: int, prev_dc: int) -> Tuple[np.ndarray, int]:
        """Decode a single 8x8 block (DC + AC)"""
        block = np.zeros(64, dtype=np.int32)
        
        # === 1. Decode DC Coefficient ===
        # Read DC coefficient size (category)
        size = self.decode_huffman(stream, 0, dc_table_id) # 0 = DC table class
        
        # Read DC coefficient amplitude
        if size == 0:
            diff = 0
        else:
            diff = stream.read_bits(size)
            # Handle negative numbers (if MSB is 0, it's negative)
            if diff < (1 << (size - 1)):
                diff -= (1 << size) - 1
                
        # Restore actual DC value (DPCM)
        dc_val = prev_dc + diff
        block[0] = dc_val
        
        # === 2. Decode AC Coefficients ===
        idx = 1
        while idx < 64:
            symbol = self.decode_huffman(stream, 1, ac_table_id) # 1 = AC table class
            
            if symbol == 0x00: # EOB (End of Block)
                break
            elif symbol == 0xF0: # ZRL (16 zeros)
                idx += 16
                continue
                
            run = symbol >> 4
            size = symbol & 0x0F
            
            idx += run # Skip 'run' zeros
            
            if idx >= 64:
                break
                
            # Read AC value
            if size > 0:
                val = stream.read_bits(size)
                if val < (1 << (size - 1)):
                    val -= (1 << size) - 1
                block[idx] = val
            
            idx += 1
            
        # === 3. Un-Zigzag ===
        # Restore 64 1D coefficients to 8x8
        final_block = np.zeros((8, 8), dtype=np.float32)
        for i in range(64):
            r, c = self.zigzag_map[i]
            final_block[r, c] = block[i]
            
        return final_block, dc_val

    # ================= Main Process =================
    def decode(self, jpeg_data: bytes) -> np.ndarray:
        """Main decoding function: bytes -> RGB numpy array"""
        
        pos = 0
        data_len = len(jpeg_data)
        
        # Must start with SOI (FF D8)
        if jpeg_data[0:2] != b'\xFF\xD8':
            raise ValueError("Not a valid JPEG file")
        pos += 2
        
        scan_data = None
        
        # === Parse Markers ===
        while pos < data_len:
            if jpeg_data[pos] != 0xFF:
                pos += 1
                continue
            
            marker = jpeg_data[pos+1]
            pos += 2
            
            if marker == 0xD9: # EOI (End of Image)
                break
            elif marker == 0x00: # Stuffing byte, ignore
                continue
                
            # Read segment length
            length = struct.unpack(">H", jpeg_data[pos:pos+2])[0]
            segment_data = jpeg_data[pos+2 : pos+length]
            
            if marker == 0xC0: # SOF0 (Baseline DCT)
                precision, h, w, components = struct.unpack(">BHHB", segment_data[:6])
                self.image_info = {'h': h, 'w': w, 'c': components}
                # Parse component info (usually 1, 2, 3 corresponds to Y, Cb, Cr)
                # Here simplified, assuming standard YCbCr 4:2:0
                
            elif marker == 0xDB: # DQT (Define Quantization Table)
                idx = 0
                while idx < len(segment_data):
                    t_info = segment_data[idx]
                    t_id = t_info & 0x0F
                    # precision = (t_info >> 4) # Assuming 8-bit
                    idx += 1
                    # read 64 bytes
                    table = np.frombuffer(segment_data[idx:idx+64], dtype=np.uint8)
                    # reverse zigzag - Handled during dequantization mapping in this implementation
                    
                    full_table = np.zeros((8, 8), dtype=np.float32)
                    for i in range(64):
                        r, c = self.zigzag_map[i]
                        full_table[r, c] = table[i]
                        
                    self.quant_tables[t_id] = full_table
                    idx += 64
                    
            elif marker == 0xC4: # DHT (Define Huffman Table)
                idx = 0
                while idx < len(segment_data):
                    t_info = segment_data[idx]
                    t_class = (t_info >> 4) & 0x0F # 0=DC, 1=AC
                    t_id = t_info & 0x0F
                    idx += 1
                    
                    bits = list(segment_data[idx:idx+16])
                    idx += 16
                    count = sum(bits)
                    
                    values = list(segment_data[idx:idx+count])
                    idx += count
                    
                    self.huffman_tables[(t_class, t_id)] = self.build_huffman_table(bits, values)
            
            elif marker == 0xDA: # SOS (Start of Scan)
                # Start of Scan segment indicates the beginning of compressed image data
                scan_data = jpeg_data[pos+length:]
                break 
                
            pos += length

        # === Start Decoding ===
        if scan_data is None or not self.image_info:
            raise ValueError("Failed to parse JPEG structure")

        h, w = self.image_info['h'], self.image_info['w']
        
        # Calculate number of MCUs (horizontally and vertically)
        mcu_h = (h + 15) // 16
        mcu_w = (w + 15) // 16
        
        # Buffers
        Y_plane = np.zeros((mcu_h * 16, mcu_w * 16), dtype=np.float32)
        Cb_plane = np.zeros((mcu_h * 8, mcu_w * 8), dtype=np.float32) # 4:2:0 Subsampled
        Cr_plane = np.zeros((mcu_h * 8, mcu_w * 8), dtype=np.float32)
        
        stream = self.BitStream(scan_data)
        
        # DC predictor reset
        prev_dc_y = 0
        prev_dc_cb = 0
        prev_dc_cr = 0
        
        # Iterate over all MCUs
        for r in range(mcu_h):
            for c in range(mcu_w):
                # Decoding Order: Y1, Y2, Y3, Y4, Cb, Cr
                
                # --- Decode Y Component (Four 8x8 blocks) ---
                for by in range(2):
                    for bx in range(2):
                        block, prev_dc_y = self.decode_block(stream, 0, 0, prev_dc_y) # Y uses table 0
                        block = self.dequantize(block, self.quant_tables[0])
                        block = self.idct_2d(block)
                        
                        y_pos = r * 16 + by * 8
                        x_pos = c * 16 + bx * 8
                        Y_plane[y_pos:y_pos+8, x_pos:x_pos+8] = block

                # --- Decode Cb Component (One 8x8 block) ---
                block, prev_dc_cb = self.decode_block(stream, 1, 1, prev_dc_cb) # Cb uses table 1
                block = self.dequantize(block, self.quant_tables[1])
                block = self.idct_2d(block)
                Cb_plane[r*8:(r+1)*8, c*8:(c+1)*8] = block

                # --- Decode Cr Component (One 8x8 block) ---
                block, prev_dc_cr = self.decode_block(stream, 1, 1, prev_dc_cr) # Cr uses table 1
                block = self.dequantize(block, self.quant_tables[1])
                block = self.idct_2d(block)
                Cr_plane[r*8:(r+1)*8, c*8:(c+1)*8] = block

        # === Reconstruct Image ===
        # 1. Crop padded parts
        Y_plane = Y_plane[:h, :w]
        
        # 2. Upsample Cb, Cr to original size
        Cb_up, Cr_up = self.upsample_chroma(Cb_plane, Cr_plane, h, w)
        
        # 3. Convert to RGB
        rgb_image = self.ycbcr_to_rgb(Y_plane, Cb_up, Cr_up)
        
        return rgb_image