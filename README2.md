
# Simple JPEG Codec (Video Compression Final Project)

This repository contains a small educational JPEG codec written in Python. 
It implements a baseline (JFIF) JPEG **encoder** and **decoder**, plus a few helper
scripts for running experiments and measuring compression quality.

The code is organized so that you can either call the encoder/decoder directly from
Python, or use the provided command-line tools.

---

## Project Structure

- `jpeg_encoder_new.py` – Core JPEG encoder with adaptive quantization and two-pass Huffman optimization. 
- `jpeg_decoder.py` – Baseline JPEG decoder that reconstructs RGB images from JFIF streams. :contentReference[oaicite:1]{index=1} 
- `encode_photo.py` – CLI wrapper that loads a photo from disk, encodes it with `JPEGEncoder`, and saves a `.jpg`. :contentReference[oaicite:2]{index=2} 
- `decode_photo.py` – CLI tool that decodes a `.jpg` using `JPEGDecoder` and saves a PNG (or other format). :contentReference[oaicite:3]{index=3} 
- `compare_images.py` – Utility to compare two images using MSE, PSNR, SSIM and file size / compression ratio. :contentReference[oaicite:4]{index=4} 
- `image_utils.py` – Helper functions to load/save images and check which image libraries are available. :contentReference[oaicite:5]{index=5}

> **Note:** `encode_photo.py` expects a `JPEGEncoder` class; in this project it lives in `jpeg_encoder_new.py`. You can either rename the file to `jpeg_encoder.py` or adjust the import line accordingly.

---

## Requirements

- Python 3.8+
- Core dependencies:
  - `numpy`
- For image I/O (at least one of):
  - `Pillow`
  - `opencv-python`
  - `matplotlib`
- For quality metrics (`compare_images.py`):
  - `scikit-image`
  - `opencv-python`

Example installation:

```bash
pip install numpy pillow opencv-python matplotlib scikit-image
````

---

## Modules & Main Functions

### `jpeg_encoder_new.py`

Core encoder implementation.

* **Class `JPEGEncoder`**

  * Implements RGB → YCbCr, 4:2:0 subsampling, 8×8 DCT, (adaptive) quantization,
    zig-zag scan, Huffman coding and JFIF formatting.
  * `encode(rgb_image: np.ndarray) -> bytes`
    Encodes an RGB image array into a JPEG byte stream using a **two-pass** scheme:
    first pass builds symbol histograms, second pass rebuilds Huffman tables and
    produces the final bitstream.
  * `encode_file(rgb_image: np.ndarray, output_path: str)`
    Convenience method: runs `encode()` and writes the JPEG to disk.
* Internal helpers such as `rgb_to_ycbcr`, `subsample_chroma`, `dct_2d`,
  `quantize`, `zigzag_scan`, `encode_dc`, `encode_ac`, and `_get_block_symbols()`
  are used to implement the JPEG pipeline and Huffman optimization. 

Typical use from Python:

```python
from jpeg_encoder_new import JPEGEncoder
import image_utils

img = image_utils.load_image("lena.png")   # RGB uint8 array
encoder = JPEGEncoder(quality=75)
jpeg_bytes = encoder.encode(img)

with open("output_lena.jpg", "wb") as f:
    f.write(jpeg_bytes)
```

---

### `jpeg_decoder.py`

Baseline JPEG decoder.

* **Class `JPEGDecoder`**

  * Parses JFIF markers (`SOI`, `DQT`, `DHT`, `SOF0`, `SOS`), reads quantization and
    Huffman tables, performs entropy decoding (Huffman + RLE + DPCM), dequantization,
    8×8 IDCT, chroma upsampling, and YCbCr → RGB conversion.
  * `decode(jpeg_bytes: bytes) -> np.ndarray`
    Takes a JPEG byte stream and returns an RGB `numpy` array (`H × W × 3`, `uint8`).

---

### `encode_photo.py`

Command-line tool to encode an image file using `JPEGEncoder`.

* **Function `encode_photo(input_path, output_path=None, quality=75)`**

  * Checks available image libraries (`Pillow`, `OpenCV`, `Matplotlib`).
  * Loads the input image as RGB using `image_utils.load_image`.
  * Runs `JPEGEncoder(quality=quality).encode_file(...)`.
  * Writes the encoded JPEG to `output_path` (default: `<input>_encoded.jpg`).

* **Function `main()`**

  * CLI interface:

    * `python encode_photo.py input.png`
    * `python encode_photo.py input.png output.jpg`
    * `python encode_photo.py input.png output.jpg 90`
  * If no arguments are given, it creates a demo gradient image and encodes it for
    demonstration. 

---

### `decode_photo.py`

Command-line JPEG decoder.

* Reads a `.jpg` file from disk, decodes it using `JPEGDecoder`, and saves the
  result as PNG (or another format) via `image_utils.save_image`.

Usage:

```bash
python decode_photo.py input.jpg
# → writes input_decoded.png

python decode_photo.py input.jpg output.png
```

---

### `compare_images.py`

Image quality comparison tool.

* **Function `compare_images(original_path, compressed_path)`**

  * Loads two images using OpenCV.
  * Ensures they have the same dimensions (resizes compressed image if needed).
  * Computes:

    * MSE (mean squared error)
    * PSNR (via `skimage.metrics.peak_signal_noise_ratio`)
    * SSIM (via `skimage.metrics.structural_similarity`)
    * Original / compressed file sizes and compression ratio 
  * Prints a formatted report and returns `(psnr, ssim)`.

CLI usage:

```bash
python compare_images.py original.png output_lena.jpg
```

---

### `image_utils.py`

Shared helper functions for image I/O.

* **`load_image(file_path) -> np.ndarray`**

  * Tries `Pillow` → `OpenCV` → `matplotlib` in order.
  * Always returns an RGB `numpy` array (`uint8`, shape `(H, W, 3)`).

* **`save_image(image, file_path)`**

  * Saves an RGB `numpy` array to disk, trying `Pillow`, then `OpenCV`, then
    `matplotlib`. 

* **`check_dependencies() -> dict`**

  * Returns a dict like `{"Pillow": True, "OpenCV": False, "Matplotlib": True}`
    indicating which image libraries are installed. 

---

## Typical Workflows

### 1. Encode and Decode a Photo

```bash
# Encode
python encode_photo.py lena.png output_lena.jpg 75

# Decode using your own decoder
python decode_photo.py output_lena.jpg output_lena_decoded.png
```

### 2. Compare Against Original

```bash
python compare_images.py lena.png output_lena.jpg
python compare_images.py lena.png output_lena_decoded.png
```

---

## Notes

* This project is intended for learning and experimentation, not for production use.
* The encoder implements adaptive quantization and per-image optimized Huffman tables, so bitstreams may differ from those produced by standard JPEG libraries, but they  are still JFIF-compliant and can be opened by normal image viewers.
