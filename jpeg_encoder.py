"""
JPEG 編碼器 - Baseline DCT 實現（完整版）

本模組實現了標準的 JPEG 壓縮演算法，包含以下核心步驟：

1. 顏色空間轉換：RGB → YCbCr
2. 色度子採樣：4:2:0 子採樣以減少資料量
3. 離散餘弦變換（DCT）：將空間域轉換為頻率域
4. 量化：使用量化表壓縮資料（有損壓縮的關鍵步驟）
5. Zigzag 掃描：重新排列係數以提高壓縮效率
6. Huffman 編碼：無損壓縮編碼
7. JFIF 檔案格式：組裝成標準 JPEG 檔案

主要類別：
- JPEGEncoder: 完整的 JPEG 編碼器實現

便捷函數：
- encode_image(): 快速編碼圖像為 JPEG 位元組資料
- save_jpeg(): 快速將圖像儲存為 JPEG 檔案
"""

import numpy as np
from typing import Tuple, List, Dict
import struct


class JPEGEncoder:
    """
    Baseline JPEG 編碼器類別

    此類別實現了標準的 JPEG 壓縮演算法，可將 RGB 圖像
    壓縮為 JPEG 格式的位元組資料。

    使用方法：
        encoder = JPEGEncoder(quality=75)
        jpeg_data = encoder.encode(rgb_image)
        # 或直接儲存
        encoder.encode_file(rgb_image, 'output.jpg')
    """

    # ==================== 標準 JPEG 量化表 ====================
    # 量化表用於控制壓縮率和品質的平衡
    # 數值越大，壓縮率越高但品質越低

    # 亮度（Y）量化表 - 人眼對亮度變化較敏感，使用較小的量化值
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

    # 色度（Cb, Cr）量化表 - 人眼對色度變化較不敏感，使用較大的量化值
    CHROMINANCE_QUANT_TABLE = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ], dtype=np.float32)

    # ==================== 標準 Zigzag 掃描模式 ====================
    # Zigzag 掃描將 8x8 二維陣列轉換為一維序列
    # 將低頻（重要）係數排在前面，高頻（不重要）係數排在後面
    # 這樣可以提高後續壓縮的效率（因為高頻係數通常為 0）

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

    # ==================== 標準 Huffman 表定義 ====================
    # Huffman 表用於無損壓縮編碼
    # BITS 陣列：每個碼長的碼字數量（長度 16）
    # VALS 陣列：按碼長排序的符號值

    # DC 亮度 Huffman 表
    _DC_LUM_BITS = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    _DC_LUM_VALS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # DC 色度 Huffman 表
    _DC_CHR_BITS = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    _DC_CHR_VALS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # AC 亮度 Huffman 表
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

    # AC 色度 Huffman 表
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

    def __init__(self, quality: int = 75):
        """
        初始化 JPEG 編碼器。

        參數：
            quality: JPEG 品質因子（1-100），預設 75
                    - 1-50: 低品質，高壓縮率
                    - 51-75: 中等品質（推薦）
                    - 76-100: 高品質，低壓縮率
        """
        # 限制品質值在有效範圍內
        self.quality = int(np.clip(quality, 1, 100))

        # 根據品質參數縮放量化表
        # 品質越高，量化值越小，保留更多細節
        self.lum_quant_table = self._scale_quantization_table(
            self.LUMINANCE_QUANT_TABLE, self.quality
        )
        self.chrom_quant_table = self._scale_quantization_table(
            self.CHROMINANCE_QUANT_TABLE, self.quality
        )

        # 從標準定義自動構建 Huffman 編碼映射表
        # 這些表將符號映射到 (碼字, 碼長) 對
        self.dc_lum_huffman = self._build_huffman_map(self._DC_LUM_BITS, self._DC_LUM_VALS)
        self.dc_chr_huffman = self._build_huffman_map(self._DC_CHR_BITS, self._DC_CHR_VALS)
        self.ac_lum_huffman = self._build_huffman_map(self._AC_LUM_BITS, self._AC_LUM_VALS)
        self.ac_chr_huffman = self._build_huffman_map(self._AC_CHR_BITS, self._AC_CHR_VALS)

        # 預計算 DCT 所需的餘弦值表以加速運算
        self._init_dct_tables()

    def _init_dct_tables(self):
        """
        預計算 DCT 所需的餘弦值表。

        DCT 運算涉及大量的餘弦計算，預先計算可以顯著提升效能。
        """
        # 建立 4D 餘弦表：cos_table[u, v, x, y]
        self._cos_table = np.zeros((8, 8, 8, 8), dtype=np.float32)
        # Alpha 係數：u=0 或 v=0 時為 sqrt(1/8)，否則為 sqrt(2/8)
        self._alpha = np.zeros(8, dtype=np.float32)

        # 計算 alpha 係數
        for i in range(8):
            self._alpha[i] = np.sqrt(1/8) if i == 0 else np.sqrt(2/8)

        # 預計算所有可能的餘弦值
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
        """
        根據品質因子縮放量化表。

        使用 IJG (Independent JPEG Group) 標準縮放公式：
        - 品質 < 50: scale = 5000 / quality
        - 品質 >= 50: scale = 200 - 2 * quality

        參數：
            base_table: 基礎量化表（8x8）
            quality: 品質因子（1-100）

        返回：
            縮放後的量化表
        """
        # 根據品質計算縮放因子
        if quality < 50:
            scale = 5000 / quality
        else:
            scale = 200 - 2 * quality

        # 應用縮放並限制在有效範圍內
        scaled_table = np.floor((base_table * scale + 50) / 100)
        scaled_table = np.clip(scaled_table, 1, 255)
        return scaled_table.astype(np.float32)

    @staticmethod
    def _build_huffman_map(bits: List[int], values: List[int]) -> Dict[int, Tuple[int, int]]:
        """
        根據 BITS 和 VALUES 陣列構建 Huffman 編碼映射表。

        這是 JPEG 標準中定義的 Huffman 表構建演算法。
        從 BITS 和 VALUES 陣列生成完整的符號到碼字的映射。

        參數：
            bits: 每個碼長的碼字數量（長度 16）
            values: 按碼長排序的符號值

        返回：
            字典 {symbol: (code, length)}
            例如：{0: (0, 2), 1: (2, 2), ...}
        """
        huffman_map = {}
        code = 0  # 當前碼字
        val_idx = 0  # values 陣列索引

        # 遍歷所有可能的碼長（1-16 位元）
        for length in range(1, 17):
            count = bits[length - 1]  # 該碼長的符號數量
            # 為該碼長的所有符號分配碼字
            for _ in range(count):
                if val_idx < len(values):
                    symbol = values[val_idx]
                    huffman_map[symbol] = (code, length)
                    code += 1
                    val_idx += 1
            # 進入下一個碼長時，碼字左移一位
            code <<= 1

        return huffman_map

    @staticmethod
    def rgb_to_ycbcr(rgb_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        將 RGB 圖像轉換為 YCbCr 顏色空間。

        YCbCr 顏色空間將亮度（Y）和色度（Cb, Cr）分離，
        更符合人眼視覺特性，便於壓縮。

        使用 JPEG/JFIF 標準轉換公式：
        Y  =  0.299*R + 0.587*G + 0.114*B
        Cb = -0.168736*R - 0.331264*G + 0.5*B + 128
        Cr =  0.5*R - 0.418688*G - 0.081312*B + 128

        參數：
            rgb_image: RGB 圖像陣列（height, width, 3），值範圍 [0, 255]

        返回：
            (Y, Cb, Cr) 三個通道陣列，值範圍 [0, 255]
        """
        rgb = rgb_image.astype(np.float32)

        # 分離 RGB 三個通道
        R = rgb[:, :, 0]
        G = rgb[:, :, 1]
        B = rgb[:, :, 2]

        # 使用標準 JPEG 公式進行顏色空間轉換
        Y  =  0.299 * R + 0.587 * G + 0.114 * B          # 亮度
        Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128.0  # 藍色色度
        Cr =  0.5 * R - 0.418688 * G - 0.081312 * B + 128.0  # 紅色色度

        # 裁剪到有效範圍 [0, 255]
        Y  = np.clip(Y, 0, 255)
        Cb = np.clip(Cb, 0, 255)
        Cr = np.clip(Cr, 0, 255)

        return Y, Cb, Cr

    @staticmethod
    def subsample_chroma(channel: np.ndarray) -> np.ndarray:
        """
        對色度通道進行 2x2 子採樣（4:2:0）。

        人眼對色度變化不敏感，可以降低色度的解析度
        來減少資料量，而不會明顯影響視覺品質。

        4:2:0 子採樣：色度的水平和垂直解析度都減半

        參數：
            channel: 色度通道陣列

        返回：
            子採樣後的通道陣列（尺寸減半）
        """
        h, w = channel.shape
        # 使用 reshape 和 mean 進行高效的 2x2 區域平均
        # 將每 2x2 的像素塊平均為一個像素
        return channel.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))

    def dct_2d(self, block: np.ndarray) -> np.ndarray:
        """
        對 8x8 塊執行二維離散餘弦變換（DCT）。

        DCT 將空間域的圖像資料轉換為頻率域，
        使得能量集中在低頻係數上，便於壓縮。

        參數：
            block: 8x8 像素塊，值範圍 [0, 255]

        返回：
            8x8 DCT 係數塊
        """
        # 電平偏移：將 [0, 255] 轉換為 [-128, 127]
        # 這樣可以使 DC 係數更接近 0，便於編碼
        block = block.astype(np.float32) - 128.0

        # 使用預計算的餘弦表進行 DCT
        dct_result = np.zeros((8, 8), dtype=np.float32)

        # 對每個頻率位置 (u, v) 計算 DCT 係數
        for u in range(8):
            for v in range(8):
                # 累加所有空間位置 (x, y) 的貢獻
                sum_val = np.sum(block * self._cos_table[u, v])
                # 乘以 alpha 係數得到最終結果
                dct_result[u, v] = self._alpha[u] * self._alpha[v] * sum_val

        return dct_result

    @staticmethod
    def quantize(dct_block: np.ndarray, quant_table: np.ndarray) -> np.ndarray:
        """
        使用量化表對 DCT 係數進行量化。

        量化是 JPEG 有損壓縮的關鍵步驟，
        通過除以量化表中的值並四捨五入來丟棄不重要的資訊。

        參數：
            dct_block: 8x8 DCT 係數塊
            quant_table: 8x8 量化表

        返回：
            8x8 量化後的係數塊（整數）
        """
        # 除以量化表並四捨五入
        # 量化表中的值越大，該位置的係數壓縮越多
        return np.round(dct_block / quant_table).astype(np.int32)

    def zigzag_scan(self, block: np.ndarray) -> np.ndarray:
        """
        對 8x8 塊執行 zigzag 掃描。

        Zigzag 掃描將二維陣列重新排列為一維序列，
        使得低頻係數（重要）排在前面，高頻係數（不重要）排在後面。
        這樣可以提高後續 Huffman 編碼的效率。

        參數：
            block: 8x8 係數塊

        返回：
            64 個係數的一維陣列（按 zigzag 順序）
        """
        # 使用預定義的 zigzag 模式重新排列係數
        return block.flatten()[self.ZIGZAG_PATTERN]

    @staticmethod
    def encode_coefficient_value(value: int) -> Tuple[int, int]:
        """
        將係數值編碼為（size, amplitude）對。

        JPEG 使用特殊的編碼方式表示係數：
        - size: 表示該值所需的位元數
        - amplitude: 實際的數值
        - 正數直接使用其二進位表示
        - 負數使用其反碼（所有位元取反）

        參數：
            value: 係數值

        返回：
            (size, amplitude) 元組
        """
        if value == 0:
            return 0, 0

        abs_val = abs(value)
        # size 是表示該絕對值所需的位元數
        size = abs_val.bit_length()

        if value > 0:
            # 正數直接使用原值
            amplitude = value
        else:
            # 負數使用反碼表示
            # 例如：-3 的 size=2，amplitude = -3 + (1<<2) - 1 = -3 + 3 = 0
            amplitude = value + (1 << size) - 1

        return size, amplitude

    def encode_dc(self, dc_diff: int, is_luminance: bool = True) -> List[Tuple[int, int]]:
        """
        編碼 DC 係數差值。

        DC 係數使用差分編碼（DPCM），即編碼與前一個塊的差值。
        這是因為相鄰塊的 DC 係數通常相近，差值較小，便於壓縮。

        參數：
            dc_diff: DC 係數與前一個塊的差值
            is_luminance: True 使用亮度表，False 使用色度表

        返回：
            (code, length) 元組列表
        """
        # 將差值編碼為 (size, amplitude)
        size, amplitude = self.encode_coefficient_value(dc_diff)
        # 選擇對應的 Huffman 表
        huffman_table = self.dc_lum_huffman if is_luminance else self.dc_chr_huffman

        # 限制 size 在有效範圍內（0-11）
        if size not in huffman_table:
            size = min(size, 11)

        # 獲取 size 的 Huffman 編碼
        code, length = huffman_table[size]
        result = [(code, length)]

        # 添加實際的振幅位元
        if size > 0:
            amplitude_bits = amplitude & ((1 << size) - 1)
            result.append((amplitude_bits, size))

        return result

    def encode_ac(self, ac_coeffs: np.ndarray, is_luminance: bool = True) -> List[Tuple[int, int]]:
        """
        編碼 AC 係數序列。

        AC 係數使用遊程編碼（Run-Length Encoding）：
        - 記錄連續零的個數和下一個非零值
        - 符號格式：(zero_run, size)，其中 zero_run 是連續零的個數
        - 使用特殊符號 EOB (0x00) 表示剩餘係數全為零
        - 使用特殊符號 ZRL (0xF0) 表示連續 16 個零

        參數：
            ac_coeffs: 63 個 AC 係數（zigzag 順序，不含 DC）
            is_luminance: True 使用亮度表，False 使用色度表

        返回：
            (code, length) 元組列表
        """
        # 選擇對應的 Huffman 表
        huffman_table = self.ac_lum_huffman if is_luminance else self.ac_chr_huffman
        result = []
        zero_run = 0  # 記錄連續零的個數

        # 找到最後一個非零係數的位置
        last_nonzero = -1
        for i in range(len(ac_coeffs) - 1, -1, -1):
            if ac_coeffs[i] != 0:
                last_nonzero = i
                break

        # 如果全是零，只輸出 EOB（End of Block）
        if last_nonzero == -1:
            code, length = huffman_table[0x00]  # EOB 符號
            result.append((code, length))
            return result

        # 只編碼到最後一個非零係數
        for i in range(last_nonzero + 1):
            coeff = ac_coeffs[i]

            if coeff == 0:
                # 累計連續零的個數
                zero_run += 1
            else:
                # 遇到非零係數，先處理之前的連續零
                # 如果連續零超過 16 個，使用 ZRL 符號
                while zero_run >= 16:
                    code, length = huffman_table[0xF0]  # ZRL 符號（16 個零）
                    result.append((code, length))
                    zero_run -= 16

                # 編碼當前非零係數
                size, amplitude = self.encode_coefficient_value(int(coeff))

                # 限制 size 到有效範圍（1-10）
                if size > 10:
                    size = 10
                    amplitude = amplitude & ((1 << size) - 1)

                # 組合符號：(zero_run, size)
                symbol = (zero_run << 4) | size

                if symbol in huffman_table:
                    code, length = huffman_table[symbol]
                    result.append((code, length))

                    # 添加振幅位元
                    if size > 0:
                        amplitude_bits = amplitude & ((1 << size) - 1)
                        result.append((amplitude_bits, size))

                zero_run = 0  # 重置零計數

        # 如果最後還有尾部的零，添加 EOB
        if last_nonzero < len(ac_coeffs) - 1:
            code, length = huffman_table[0x00]  # EOB
            result.append((code, length))

        return result

    def bits_to_bytes(self, bit_stream: List[Tuple[int, int]]) -> bytes:
        """
        將位元流轉換為位元組流。

        處理 JPEG 的位元組填充規則：
        - 如果出現 0xFF，必須在後面插入 0x00 以避免與標記碼混淆
        - 不完整的位元組用 1 填充

        參數：
            bit_stream: (code, length) 元組列表

        返回：
            位元組資料
        """
        result = bytearray()
        current_byte = 0  # 當前正在組裝的位元組
        bits_in_byte = 0  # 當前位元組中已有的位元數

        # 逐個處理所有位元
        for value, num_bits in bit_stream:
            # 從高位到低位處理每一位
            for i in range(num_bits - 1, -1, -1):
                bit = (value >> i) & 1
                current_byte = (current_byte << 1) | bit
                bits_in_byte += 1

                # 當累積了 8 個位元時，輸出一個位元組
                if bits_in_byte == 8:
                    result.append(current_byte)
                    # JPEG 位元組填充規則：0xFF 後必須跟 0x00
                    if current_byte == 0xFF:
                        result.append(0x00)
                    current_byte = 0
                    bits_in_byte = 0

        # 處理最後不完整的位元組（用 1 填充）
        if bits_in_byte > 0:
            current_byte <<= (8 - bits_in_byte)
            current_byte |= (1 << (8 - bits_in_byte)) - 1  # 用 1 填充剩餘位元
            result.append(current_byte)
            if current_byte == 0xFF:
                result.append(0x00)

        return bytes(result)

    def create_jfif_header(self, width: int, height: int) -> bytes:
        """
        創建 JFIF 格式的 JPEG 檔案標頭。

        JFIF（JPEG File Interchange Format）是最常用的 JPEG 檔案格式。
        標頭包含：
        - SOI: 檔案開始標記
        - APP0: JFIF 應用標記
        - DQT: 量化表定義
        - SOF0: 訊框開始（Baseline DCT）
        - DHT: Huffman 表定義
        - SOS: 掃描開始

        參數：
            width: 圖像寬度
            height: 圖像高度

        返回：
            JPEG 檔案標頭位元組資料
        """
        header = bytearray()

        # === SOI (Start of Image) 檔案開始標記 ===
        header.extend(b'\xFF\xD8')

        # === APP0 (JFIF marker) JFIF 應用標記 ===
        header.extend(b'\xFF\xE0')
        header.extend(b'\x00\x10')  # 段長度 = 16
        header.extend(b'JFIF\x00')  # 標識符
        header.extend(b'\x01\x01')  # 版本 1.1
        header.extend(b'\x00')      # 像素密度單位：無單位
        header.extend(b'\x00\x01')  # X 密度 = 1
        header.extend(b'\x00\x01')  # Y 密度 = 1
        header.extend(b'\x00\x00')  # 無縮圖

        # === DQT (Define Quantization Table) 亮度量化表 ===
        header.extend(b'\xFF\xDB')
        header.extend(b'\x00\x43')  # 長度 = 67 (2 + 1 + 64)
        header.extend(b'\x00')      # 表 ID = 0（亮度），精度 = 8 位元
        # 量化表必須按 zigzag 順序寫入
        lum_qt_zigzag = self.zigzag_scan(self.lum_quant_table).astype(np.uint8)
        header.extend(lum_qt_zigzag.tobytes())

        # === DQT 色度量化表 ===
        header.extend(b'\xFF\xDB')
        header.extend(b'\x00\x43')  # 長度 = 67
        header.extend(b'\x01')      # 表 ID = 1（色度），精度 = 8 位元
        chrom_qt_zigzag = self.zigzag_scan(self.chrom_quant_table).astype(np.uint8)
        header.extend(chrom_qt_zigzag.tobytes())

        # === SOF0 (Start of Frame - Baseline DCT) 訊框開始 ===
        header.extend(b'\xFF\xC0')
        header.extend(b'\x00\x11')  # 長度 = 17
        header.extend(b'\x08')      # 樣本精度 = 8 位元
        header.extend(struct.pack('>H', height))  # 圖像高度（大端序）
        header.extend(struct.pack('>H', width))   # 圖像寬度（大端序）
        header.extend(b'\x03')      # 分量數 = 3 (Y, Cb, Cr)
        # Y 分量: ID=1, 採樣因子=2x2（水平2垂直2），量化表=0
        header.extend(b'\x01\x22\x00')
        # Cb 分量: ID=2, 採樣因子=1x1，量化表=1
        header.extend(b'\x02\x11\x01')
        # Cr 分量: ID=3, 採樣因子=1x1，量化表=1
        header.extend(b'\x03\x11\x01')

        # === DHT (Define Huffman Table) 定義所有 Huffman 表 ===
        # DC 亮度表
        header.extend(self._create_huffman_table_segment(0, True, self._DC_LUM_BITS, self._DC_LUM_VALS))
        # AC 亮度表
        header.extend(self._create_huffman_table_segment(0, False, self._AC_LUM_BITS, self._AC_LUM_VALS))
        # DC 色度表
        header.extend(self._create_huffman_table_segment(1, True, self._DC_CHR_BITS, self._DC_CHR_VALS))
        # AC 色度表
        header.extend(self._create_huffman_table_segment(1, False, self._AC_CHR_BITS, self._AC_CHR_VALS))

        # === SOS (Start of Scan) 掃描開始 ===
        header.extend(b'\xFF\xDA')
        header.extend(b'\x00\x0C')  # 長度 = 12
        header.extend(b'\x03')      # 分量數 = 3
        header.extend(b'\x01\x00')  # Y: DC表0, AC表0
        header.extend(b'\x02\x11')  # Cb: DC表1, AC表1
        header.extend(b'\x03\x11')  # Cr: DC表1, AC表1
        header.extend(b'\x00')      # Ss = 0（起始光譜選擇）
        header.extend(b'\x3F')      # Se = 63（結束光譜選擇）
        header.extend(b'\x00')      # Ah = 0, Al = 0（連續近似位元）

        return bytes(header)

    def _create_huffman_table_segment(self, table_id: int, is_dc: bool,
                                       bits: List[int], values: List[int]) -> bytes:
        """
        創建 DHT (Define Huffman Table) 段。

        參數：
            table_id: 表 ID（0 = 亮度，1 = 色度）
            is_dc: True = DC 表，False = AC 表
            bits: BITS 陣列（長度 16）
            values: VALUES 陣列

        返回：
            DHT 段位元組資料
        """
        segment = bytearray()
        segment.extend(b'\xFF\xC4')  # DHT 標記

        # 計算段長度
        length = 2 + 1 + 16 + len(values)
        segment.extend(struct.pack('>H', length))

        # 表類別和 ID
        # 表類別：0 = DC，1 = AC
        table_class = 0 if is_dc else 1
        segment.append((table_class << 4) | table_id)

        # BITS 陣列（16 位元組）
        segment.extend(bytes(bits))

        # VALUES 陣列
        segment.extend(bytes(values))

        return bytes(segment)

    def _encode_block(self, block: np.ndarray, quant_table: np.ndarray,
                      prev_dc: int, is_luminance: bool) -> Tuple[List[Tuple[int, int]], int]:
        """
        編碼單個 8x8 塊的完整流程。

        步驟：
        1. DCT 變換
        2. 量化
        3. Zigzag 掃描
        4. DC 差分編碼
        5. AC 遊程編碼

        參數：
            block: 8x8 像素塊
            quant_table: 量化表
            prev_dc: 前一個塊的 DC 值（用於差分編碼）
            is_luminance: 是否為亮度塊

        返回：
            (編碼資料列表, 當前 DC 值)
        """
        # 第一步：DCT 變換
        dct_block = self.dct_2d(block)

        # 第二步：量化
        quant_block = self.quantize(dct_block, quant_table)

        # 第三步：Zigzag 掃描
        zigzag = self.zigzag_scan(quant_block)

        # 第四步：DC 差分編碼
        dc = int(zigzag[0])
        dc_diff = dc - prev_dc
        encoded_data = self.encode_dc(dc_diff, is_luminance)

        # 第五步：AC 遊程編碼
        encoded_data.extend(self.encode_ac(zigzag[1:], is_luminance))

        return encoded_data, dc

    def encode(self, rgb_image: np.ndarray) -> bytes:
        """
        將 RGB 圖像編碼為 JPEG 格式。

        完整的編碼流程：
        1. 填充圖像到 16 的倍數
        2. RGB → YCbCr 轉換
        3. 色度子採樣（4:2:0）
        4. 分塊處理（MCU = 16x16）
        5. 對每個塊進行 DCT、量化、編碼
        6. 組裝 JFIF 檔案

        參數：
            rgb_image: RGB 圖像陣列（height, width, 3），值範圍 [0, 255]

        返回：
            JPEG 格式位元組資料
        """
        original_height, original_width = rgb_image.shape[:2]

        # === 第一步：填充圖像到 16 的倍數 ===
        # MCU（Minimum Coded Unit）大小為 16x16
        # 因為 4:2:0 子採樣：Y 是 16x16（4 個 8x8 塊），Cb/Cr 是 8x8（1 個塊）
        h_padding = (16 - (original_height % 16)) % 16
        w_padding = (16 - (original_width % 16)) % 16

        if h_padding > 0 or w_padding > 0:
            rgb_image = np.pad(
                rgb_image,
                ((0, h_padding), (0, w_padding), (0, 0)),
                mode='edge'  # 使用邊緣值填充
            )

        padded_h, padded_w = rgb_image.shape[:2]

        # === 第二步：顏色空間轉換 ===
        Y, Cb, Cr = self.rgb_to_ycbcr(rgb_image)

        # === 第三步：色度子採樣 ===
        Cb_sub = self.subsample_chroma(Cb)
        Cr_sub = self.subsample_chroma(Cr)

        # === 第四步：編碼所有 MCU ===
        all_data = []  # 儲存所有編碼後的資料
        prev_dc_y = 0   # Y 通道的前一個 DC 值
        prev_dc_cb = 0  # Cb 通道的前一個 DC 值
        prev_dc_cr = 0  # Cr 通道的前一個 DC 值

        # 按 MCU 順序遍歷（每個 MCU 為 16x16 像素）
        for mcu_row in range(0, padded_h, 16):
            for mcu_col in range(0, padded_w, 16):

                # 編碼 4 個 Y 塊（2x2 排列）
                # 順序：左上、右上、左下、右下
                for block_row in range(2):
                    for block_col in range(2):
                        y_row = mcu_row + block_row * 8
                        y_col = mcu_col + block_col * 8
                        block = Y[y_row:y_row + 8, y_col:y_col + 8]
                        bits, prev_dc_y = self._encode_block(
                            block, self.lum_quant_table, prev_dc_y, True
                        )
                        all_data.extend(bits)

                # 編碼 1 個 Cb 塊
                cb_row = mcu_row // 2
                cb_col = mcu_col // 2
                block = Cb_sub[cb_row:cb_row + 8, cb_col:cb_col + 8]
                bits, prev_dc_cb = self._encode_block(
                    block, self.chrom_quant_table, prev_dc_cb, False
                )
                all_data.extend(bits)

                # 編碼 1 個 Cr 塊
                block = Cr_sub[cb_row:cb_row + 8, cb_col:cb_col + 8]
                bits, prev_dc_cr = self._encode_block(
                    block, self.chrom_quant_table, prev_dc_cr, False
                )
                all_data.extend(bits)

        # === 第五步：組裝最終檔案 ===
        jpeg_data = (
            self.create_jfif_header(original_width, original_height) +  # 標頭
            self.bits_to_bytes(all_data) +                              # 壓縮資料
            b'\xFF\xD9'  # EOI (End of Image) 檔案結束標記
        )

        return jpeg_data

    def encode_file(self, rgb_image: np.ndarray, output_path: str) -> int:
        """
        將 RGB 圖像編碼並儲存為 JPEG 檔案。

        參數：
            rgb_image: RGB 圖像陣列（height, width, 3）
            output_path: 輸出檔案路徑

        返回：
            輸出檔案大小（位元組）
        """
        # 編碼為 JPEG 資料
        jpeg_data = self.encode(rgb_image)

        # 寫入檔案
        with open(output_path, 'wb') as f:
            f.write(jpeg_data)

        print(f"JPEG 已儲存：{output_path}（{len(jpeg_data)} 位元組）")
        return len(jpeg_data)


# ==================== 便捷函數 ====================

def encode_image(rgb_image: np.ndarray, quality: int = 75) -> bytes:
    """
    將 RGB 圖像編碼為 JPEG 位元組資料。

    這是一個便捷函數，簡化了編碼器的使用。

    參數：
        rgb_image: RGB 圖像陣列
        quality: JPEG 品質（1-100）

    返回：
        JPEG 位元組資料
    """
    encoder = JPEGEncoder(quality=quality)
    return encoder.encode(rgb_image)


def save_jpeg(rgb_image: np.ndarray, output_path: str, quality: int = 75) -> int:
    """
    將 RGB 圖像儲存為 JPEG 檔案。

    這是一個便捷函數，簡化了編碼器的使用。

    參數：
        rgb_image: RGB 圖像陣列
        output_path: 輸出檔案路徑
        quality: JPEG 品質（1-100）

    返回：
        輸出檔案大小（位元組）
    """
    encoder = JPEGEncoder(quality=quality)
    return encoder.encode_file(rgb_image, output_path)


# ==================== 測試程式碼 ====================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("JPEG 編碼器測試程式")
    print("=" * 60)

    # 建立測試圖像
    print("\n建立測試圖像...")
    test_image = np.zeros((256, 256, 3), dtype=np.uint8)

    # 建立彩色漸層圖案
    for i in range(256):
        for j in range(256):
            test_image[i, j, 0] = i          # 紅色漸層（垂直）
            test_image[i, j, 1] = j          # 綠色漸層（水平）
            test_image[i, j, 2] = (i + j) // 2  # 藍色漸層（對角）

    # 測試不同品質設定
    for quality in [50, 75, 90]:
        print(f"\n使用品質 {quality} 進行編碼...")
        encoder = JPEGEncoder(quality=quality)
        output_path = f"test_q{quality}.jpg"
        size = encoder.encode_file(test_image, output_path)
        print(f"  輸出大小：{size} 位元組")
        print(f"  壓縮率：{256*256*3/size:.2f}x")

    print("\n測試完成！")
