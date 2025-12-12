import sys
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def compare_images(original_path, compressed_path):
    # 1. Load Images
    # Read using cv2, and convert to RGB (since cv2 defaults to BGR)
    img1 = cv2.imread(original_path)
    img2 = cv2.imread(compressed_path)

    if img1 is None:
        print(f"Error: Cannot read original image {original_path}")
        return
    if img2 is None:
        print(f"Error: Cannot read compressed image {compressed_path}")
        return

    # 2. Check Dimensions
    # JPEG compression sometimes causes minor edge differences due to padding (16x16 MCU)
    # Here we force resize to the same dimensions (based on original) or crop
    if img1.shape != img2.shape:
        print(f"Warning: Dimension mismatch! Original: {img1.shape}, Compressed: {img2.shape}")
        print("Resizing compressed image to match original size for comparison...")
        h, w = img1.shape[:2]
        img2 = cv2.resize(img2, (w, h))

    # Convert to RGB to ensure correct color space
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # 3. Calculate MSE (Mean Squared Error)
    # MSE = mean((I1 - I2)^2)
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)

    # 4. Calculate PSNR
    # data_range=255 indicates pixel range is 0-255
    score_psnr = psnr(img1, img2, data_range=255)

    # 5. Calculate SSIM
    # channel_axis=2 indicates the 3rd dimension is the color channel (RGB)
    # win_size of 7 or 11 is standard practice
    score_ssim = ssim(img1, img2, data_range=255, channel_axis=2)

    # 6. Calculate Compression Info (Optional)
    size_orig = os.path.getsize(original_path)
    size_comp = os.path.getsize(compressed_path)
    compression_ratio = size_orig / size_comp if size_comp > 0 else 0

    # Output Results
    print("="*40)
    print(f"Image Comparison Report")
    print(f"Original: {original_path}")
    print(f"Compressed: {compressed_path}")
    print("-" * 40)
    print(f"MSE  (Lower is better): {mse:.4f}")
    print(f"PSNR (Higher is better): {score_psnr:.4f} dB")
    print(f"SSIM (Closer to 1 is better): {score_ssim:.4f}")
    print("-" * 40)
    print(f"Original Size: {size_orig/1024:.2f} KB")
    print(f"Compressed Size: {size_comp/1024:.2f} KB")
    print(f"Compression Ratio: {compression_ratio:.2f}x")
    print("="*40)

    return score_psnr, score_ssim

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_images.py <original.png> <compressed.jpg>")
    else:
        compare_images(sys.argv[1], sys.argv[2])