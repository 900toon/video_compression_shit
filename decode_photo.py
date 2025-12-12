import sys
import os
import time
from jpeg_decoder import JPEGDecoder
from image_utils import save_image

def main():
    if len(sys.argv) < 2:
        print("Usage: python decode_photo.py <input.jpg> [output.png]")
        return

    input_path = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        filename, _ = os.path.splitext(input_path)
        output_path = f"{filename}_decoded.png"

    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return

    print(f"Reading {input_path}...")
    with open(input_path, 'rb') as f:
        jpeg_data = f.read()

    print("Initializing Decoder...")
    decoder = JPEGDecoder()
    
    start_time = time.time()
    
    try:
        print("Decoding...")
        rgb_image = decoder.decode(jpeg_data)
        
        end_time = time.time()
        duration = end_time - start_time
        
        h, w, _ = rgb_image.shape
        print(f"Success! Decoded image: {w}x{h}")
        print(f"Time taken: {duration:.4f} seconds")
        
        print(f"Saving to {output_path}...")
        save_image(rgb_image, output_path)
        print("Done.")
        
    except Exception as e:
        print(f"Decoding failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()