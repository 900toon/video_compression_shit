# JPEG 編碼器 - 程式碼說明文件

這是一個完全使用 Python 從零實現的 JPEG 圖像壓縮程式，包含完整的 Baseline DCT 演算法。本專案不依賴任何第三方壓縮庫，所有壓縮步驟都是自行實現的。

## 專案架構說明

本專案由三個核心 Python 檔案組成，每個檔案負責不同的功能模組：

### 1. `jpeg_encoder.py` - JPEG 編碼核心引擎

這是整個專案的核心，包含約 900 行程式碼，實現了完整的 JPEG 壓縮演算法。

#### 主要類別：`JPEGEncoder`

這個類別封裝了所有 JPEG 編碼所需的功能。初始化時會：

```python
def __init__(self, quality: int = 75)
```

- **品質參數處理**：接收 1-100 的品質值，並使用 IJG 標準公式計算縮放因子
  - 品質 < 50：`scale = 5000 / quality`
  - 品質 >= 50：`scale = 200 - 2 * quality`

- **量化表生成**：根據品質參數縮放標準量化表
  - 亮度量化表（8x8）：針對人眼對亮度敏感的特性，使用較小的量化值
  - 色度量化表（8x8）：人眼對色度不敏感，使用較大的量化值以提高壓縮率

- **Huffman 編碼表構建**：從標準 BITS 和 VALUES 陣列自動生成 4 張 Huffman 映射表
  - DC 亮度表、DC 色度表
  - AC 亮度表、AC 色度表

- **DCT 餘弦表預計算**：建立 4D 陣列 `_cos_table[u, v, x, y]`，避免重複計算三角函數

#### 核心演算法函數說明

##### 1. 顏色空間轉換 `rgb_to_ycbcr()`

```python
@staticmethod
def rgb_to_ycbcr(rgb_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

**程式邏輯**：
- 將 RGB 三通道分離為 R、G、B 陣列
- 套用 JPEG/JFIF 標準轉換公式：
  ```
  Y  =  0.299*R + 0.587*G + 0.114*B
  Cb = -0.168736*R - 0.331264*G + 0.5*B + 128
  Cr =  0.5*R - 0.418688*G - 0.081312*B + 128
  ```
- 使用 `np.clip()` 確保數值在 [0, 255] 範圍內
- 回傳分離的 Y（亮度）、Cb（藍色色度）、Cr（紅色色度）三個通道

**為什麼要轉換？** 因為人眼對亮度變化比色度變化更敏感，分離後可以對色度進行更激進的壓縮。

##### 2. 色度子採樣 `subsample_chroma()`

```python
@staticmethod
def subsample_chroma(channel: np.ndarray) -> np.ndarray
```

**程式邏輯**：
- 將通道 reshape 成 `(h//2, 2, w//2, 2)` 的 4D 陣列
- 對第 1 和第 3 軸（每個 2x2 區塊）取平均值
- 結果是原始尺寸的 1/4（寬高各減半）

**實現細節**：這是 4:2:0 子採樣的高效實現，將每 2x2 的色度像素平均成一個像素。

##### 3. 二維 DCT 變換 `dct_2d()`

```python
def dct_2d(self, block: np.ndarray) -> np.ndarray
```

**程式邏輯**：
- 先進行電平偏移：`block - 128`（將 [0, 255] 轉為 [-128, 127]）
- 對每個頻率位置 (u, v)：
  - 使用預計算的餘弦表 `_cos_table[u, v]` 與輸入塊相乘
  - 計算所有空間位置的總和
  - 乘以 alpha 係數（u=0 或 v=0 時為 √(1/8)，否則為 √(2/8)）
- 回傳 8x8 的 DCT 係數矩陣

**數學原理**：DCT 將空間域的像素資料轉換為頻率域，能量會集中在左上角的低頻係數。

##### 4. 量化 `quantize()`

```python
@staticmethod
def quantize(dct_block: np.ndarray, quant_table: np.ndarray) -> np.ndarray
```

**程式邏輯**：
- 將 DCT 係數除以對應的量化表數值
- 使用 `np.round()` 四捨五入到整數
- 這是 JPEG 有損壓縮的關鍵步驟

**效果**：高頻係數（右下角）會被大幅量化甚至變成 0，達到壓縮目的。

##### 5. Zigzag 掃描 `zigzag_scan()`

```python
def zigzag_scan(self, block: np.ndarray) -> np.ndarray
```

**程式邏輯**：
- 使用預定義的 `ZIGZAG_PATTERN` 陣列（64 個索引）
- 將 8x8 二維陣列展平，然後按 zigzag 順序重新排列
- 結果是一個 64 元素的一維陣列

**目的**：將低頻係數排在前面，高頻係數（通常是 0）排在後面，利於後續的遊程編碼。

##### 6. DC 係數編碼 `encode_dc()`

```python
def encode_dc(self, dc_diff: int, is_luminance: bool = True) -> List[Tuple[int, int]]
```

**程式邏輯**：
- 計算 DC 係數的 (size, amplitude) 編碼
  - size：表示該值需要的位元數
  - amplitude：正數用原值，負數用反碼
- 從 Huffman 表中查找 size 對應的碼字
- 回傳 `[(Huffman碼字, 碼長), (振幅位元, size)]`

**差分編碼**：輸入是與前一個塊的差值，因為相鄰塊的 DC 通常很接近。

##### 7. AC 係數編碼 `encode_ac()`

```python
def encode_ac(self, ac_coeffs: np.ndarray, is_luminance: bool = True) -> List[Tuple[int, int]]
```

**程式邏輯**：
- 使用遊程編碼（Run-Length Encoding）
- 記錄連續零的個數（zero_run）和下一個非零值
- 特殊符號處理：
  - `0x00` (EOB)：剩餘全是零
  - `0xF0` (ZRL)：連續 16 個零
- 對每個非零係數：
  - 計算 (size, amplitude)
  - 組合符號：`(zero_run << 4) | size`
  - 從 Huffman 表查找並輸出

**效率**：由於 zigzag 排列，後面通常是連續的零，遊程編碼非常有效。

##### 8. 位元流轉位元組 `bits_to_bytes()`

```python
def bits_to_bytes(self, bit_stream: List[Tuple[int, int]]) -> bytes
```

**程式邏輯**：
- 逐位組裝位元組
- 累積 8 個位元後輸出一個位元組
- **位元組填充規則**：若輸出 `0xFF`，必須插入 `0x00`（避免與 JPEG 標記碼衝突）
- 最後不足 8 位元的部分用 1 填充

##### 9. JFIF 標頭生成 `create_jfif_header()`

```python
def create_jfif_header(self, width: int, height: int) -> bytes
```

**程式邏輯**：依序生成 JPEG 檔案的各個段落：
1. **SOI** (`0xFFD8`)：檔案開始標記
2. **APP0**：JFIF 應用段，包含版本號、像素密度資訊
3. **DQT** (定義量化表)：
   - 亮度量化表（表 ID = 0）
   - 色度量化表（表 ID = 1）
   - 兩者都按 zigzag 順序寫入
4. **SOF0** (`0xFFC0`)：訊框開始（Baseline DCT）
   - 圖像寬高、精度
   - 三個分量資訊（Y, Cb, Cr）
   - Y 使用 2x2 採樣，Cb/Cr 使用 1x1（實現 4:2:0）
5. **DHT** (定義 Huffman 表)：寫入 4 張 Huffman 表
6. **SOS** (`0xFFDA`)：掃描開始

##### 10. 完整編碼流程 `encode()`

```python
def encode(self, rgb_image: np.ndarray) -> bytes
```

**程式邏輯**：
1. **圖像填充**：將圖像填充到 16 的倍數（MCU 大小）
   - 使用 `np.pad()` 和 `mode='edge'`（邊緣延伸）
2. **顏色轉換**：RGB → YCbCr
3. **子採樣**：對 Cb 和 Cr 進行 4:2:0 子採樣
4. **MCU 編碼**：以 16x16 像素為單位處理
   - 每個 MCU 包含：
     - 4 個 Y 塊（8x8 各一，排列為 2x2）
     - 1 個 Cb 塊（8x8）
     - 1 個 Cr 塊（8x8）
   - 對每個 8x8 塊執行：DCT → 量化 → zigzag → Huffman 編碼
   - 使用差分編碼：記錄各通道的前一個 DC 值
5. **組裝檔案**：
   - JFIF 標頭
   - 壓縮位元流
   - EOI (`0xFFD9`)：檔案結束標記

**MCU 處理順序**：由左到右、由上到下掃描，每個 MCU 內部按 Y0, Y1, Y2, Y3, Cb, Cr 順序編碼。

### 2. `image_utils.py` - 圖像載入與儲存工具

這個模組提供跨平台的圖像 I/O 功能，自動偵測並使用可用的圖像處理庫。

#### `load_image()` - 圖像載入函數

```python
def load_image(file_path: str) -> np.ndarray
```

**程式邏輯**：
1. **優先使用 PIL/Pillow**：
   - 使用 `Image.open()` 開啟檔案
   - 檢查圖像模式，若非 RGB 則使用 `convert('RGB')` 轉換
   - 轉換為 numpy 陣列
2. **備用方案一：OpenCV**：
   - 使用 `cv2.imread()` 讀取
   - 注意 OpenCV 預設是 BGR，需要 `cvtColor()` 轉為 RGB
3. **備用方案二：Matplotlib**：
   - 使用 `mpimg.imread()` 讀取
   - 處理浮點數格式（0.0-1.0 轉為 0-255）
   - 處理 RGBA（去除 alpha 通道）
   - 處理灰階（複製到三個通道）

**錯誤處理**：若所有庫都不可用，拋出 `ImportError` 並提示安裝指令。

#### `save_image()` - 圖像儲存函數

```python
def save_image(image: np.ndarray, file_path: str)
```

**程式邏輯**：
- 同樣按 Pillow → OpenCV → Matplotlib 的順序嘗試
- Pillow：使用 `Image.fromarray()` 轉換後 `save()`
- OpenCV：先轉 BGR 再 `imwrite()`
- Matplotlib：直接使用 `plt.imsave()`

#### `check_dependencies()` - 依賴檢查函數

```python
def check_dependencies() -> dict
```

**程式邏輯**：
- 嘗試 `import PIL`, `import cv2`, `import matplotlib`
- 捕捉 `ImportError` 判斷是否可用
- 回傳字典：`{'Pillow': True/False, 'OpenCV': True/False, 'Matplotlib': True/False}`

### 3. `encode_photo.py` - 命令列應用程式

這是使用者互動的入口點，提供友善的命令列介面。

#### `encode_photo()` - 主要編碼函數

```python
def encode_photo(input_path: str, output_path: str = None, quality: int = 75)
```

**程式邏輯**：
1. **檢查依賴**：呼叫 `check_dependencies()` 並顯示結果
2. **產生輸出路徑**：若未指定，使用 `原檔名_encoded.jpg`
3. **載入圖像**：呼叫 `load_image()` 並顯示圖像資訊（尺寸、通道數、資料型態）
4. **執行編碼**：建立 `JPEGEncoder` 實例並呼叫 `encode_file()`
5. **錯誤處理**：
   - `FileNotFoundError`：檔案不存在
   - 一般 `Exception`：顯示完整 traceback

#### `main()` - 命令列介面

```python
def main()
```

**程式邏輯**：
1. **參數解析**：
   - `sys.argv[1]`：輸入檔案（必需）
   - `sys.argv[2]`：輸出檔案（可選）
   - `sys.argv[3]`：品質參數（可選，預設 75）
2. **示範模式**（無參數時）：
   - 使用 numpy 建立 256x256 的彩色漸層測試圖像
   - 紅色通道：垂直漸層 (i)
   - 綠色通道：水平漸層 (j)
   - 藍色通道：對角漸層 ((i+j)//2)
   - 儲存為 `demo_input.png` 並編碼為 `demo_output.jpg`
3. **品質驗證**：確保品質值在 1-100 範圍內

## 演算法流程總覽

完整的 JPEG 編碼流程可以用以下步驟概括：

```
輸入 RGB 圖像 (H×W×3)
    ↓
[1] 填充到 16 的倍數 → RGB 圖像 (H'×W'×3)
    ↓
[2] RGB → YCbCr 轉換 → Y, Cb, Cr (H'×W')
    ↓
[3] 4:2:0 子採樣 → Y (H'×W'), Cb (H'/2×W'/2), Cr (H'/2×W'/2)
    ↓
[4] 分割為 MCU (16×16) → 遍歷每個 MCU
    ↓
    對每個 8×8 塊：
    [5] DCT 變換 → 頻率域係數
    [6] 量化 → 整數係數
    [7] Zigzag 掃描 → 一維序列
    [8] DC 差分編碼 + AC 遊程編碼 → 符號序列
    [9] Huffman 編碼 → 位元流
    ↓
[10] 組裝 JFIF 格式：標頭 + 位元流 + 結束標記
    ↓
輸出 JPEG 檔案
```

## 程式碼特色

1. **完全自主實現**：不使用任何第三方 JPEG 編碼庫，所有步驟都是從頭實現
2. **效能最佳化**：
   - DCT 使用預計算餘弦表，避免重複計算三角函數
   - 使用 numpy 向量化運算加速矩陣操作
3. **標準相容**：嚴格遵循 JPEG/JFIF 標準規範
4. **跨平台相容**：自動偵測並支援多種圖像處理庫
5. **完整註解**：每個函數都有詳細的中文註解說明

## 系統需求

- **Python 版本**：3.6 以上
- **必要套件**：
  - `numpy`：用於陣列運算和數學計算
- **圖像處理庫（三選一）**：
  - `Pillow`：推薦使用，相容性最佳
  - `opencv-python`：提供 OpenCV 支援
  - `matplotlib`：備用方案

## 使用範例

### 基本使用

```bash
# 使用預設品質（75）
python encode_photo.py input.png

# 指定輸出檔名
python encode_photo.py input.png output.jpg

# 指定品質參數
python encode_photo.py input.png output.jpg 90
```

### 程式化呼叫

```python
from jpeg_encoder import JPEGEncoder
from image_utils import load_image

# 載入圖像
image = load_image('photo.png')

# 建立編碼器（品質 85）
encoder = JPEGEncoder(quality=85)

# 編碼並儲存
encoder.encode_file(image, 'output.jpg')

# 或取得位元組資料
jpeg_bytes = encoder.encode(image)
```

"# video_compression_shit" 
