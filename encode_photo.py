"""
Photo Encoding Example Script

This script demonstrates how to load and encode real photos using the JPEG encoder.
It provides a full command-line interface, supporting batch processing and quality adjustment.

Features:
- Loads image files in various formats (PNG, BMP, JPEG, etc.)
- Checks system dependencies
- Automatically generates output filenames
- Adjustable JPEG quality parameters
- Demo mode (automatically generates a test image when no arguments are provided)
"""

import sys
from jpeg_encoder_new import JPEGEncoder
from image_utils import load_image, check_dependencies


def encode_photo(input_path: str, output_path: str = None, quality: int = 75):
    """
    Loads a photo from a file and encodes it into JPEG format.

    This function will:
    1. Check for available image processing libraries
    2. Load the input image
    3. Perform JPEG encoding with the specified quality
    4. Save the result and display information

    Args:
        input_path: Path to the input image file (supports PNG, BMP, JPEG, etc.)
        output_path: Path to the output JPEG file (default: input_filename_encoded.jpg)
        quality: JPEG quality (1-100, default: 75)
                 Higher values mean better quality but larger file size
    """
    # === Step 1: Check available image processing libraries ===
    print("Checking available image processing libraries...")
    deps = check_dependencies()
    for lib, available in deps.items():
        # Show status of each library (✓ means installed, ✗ means not installed)
        status = "✓" if available else "✗"
        print(f"  {status} {lib}")

    # If no image processing library is available, terminate the program
    if not any(deps.values()):
        print("\nError: No image loading libraries found!")
        print("Please install one of the following packages:")
        print("  pip install Pillow        (Recommended)")
        print("  pip install opencv-python")
        print("  pip install matplotlib")
        sys.exit(1)

    print()

    # === Step 2: Generate output file path ===
    if output_path is None:
        import os
        # Split filename and extension
        base_name = os.path.splitext(input_path)[0]
        # Automatically generate output filename: original_name_encoded.jpg
        output_path = f"{base_name}_encoded.jpg"

    try:
        # === Step 3: Load image ===
        print(f"Loading image: {input_path}")
        image = load_image(input_path)
        # Display image information
        print(f"  Image Size: {image.shape[1]}x{image.shape[0]} pixels")
        print(f"  Color Channels: {image.shape[2]}")
        print(f"  Data Type: {image.dtype}")
        print()

        # === Step 4: Encode to JPEG ===
        print(f"Encoding with quality {quality}...")
        encoder = JPEGEncoder(quality=quality)
        encoder.encode_file(image, output_path)
        print()

        # === Done ===
        print(f"Success! Encoded image saved to: {output_path}")

    except FileNotFoundError:
        # Input file not found
        print(f"Error: Image file not found: {input_path}")
        sys.exit(1)
    except Exception as e:
        # Other errors
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """
    Main function handling the command-line interface.

    Supports the following usage:
    1. python encode_photo.py input.png
    2. python encode_photo.py input.png output.jpg
    3. python encode_photo.py input.png output.jpg 90

    If no arguments are provided, a demo image is automatically generated.
    """
    print("=" * 70)
    print("JPEG Encoder - Photo Encoding Tool")
    print("=" * 70)
    print()

    # === Check command line arguments ===
    if len(sys.argv) < 2:
        # No input file provided, show usage instructions
        print("Usage:")
        print(f"  python {sys.argv[0]} <input_image> [output_image] [quality]")
        print()
        print("Examples:")
        print(f"  python {sys.argv[0]} photo.png")
        print(f"  python {sys.argv[0]} photo.png output.jpg")
        print(f"  python {sys.argv[0]} photo.png output.jpg 90")
        print()
        print("Arguments:")
        print("  input_image  : Input image file (PNG, BMP, JPEG, etc.)")
        print("  output_image : Output JPEG file (Optional, default: input_name_encoded.jpg)")
        print("  quality      : JPEG Quality 1-100 (Optional, default: 75)")
        print()

        # === Demo Mode: Generate Test Image ===
        print("No input file specified, creating demo image...")
        try:
            import numpy as np
            from image_utils import save_image

            # Create a 256x256 color gradient image
            demo_image = np.zeros((256, 256, 3), dtype=np.uint8)
            for i in range(256):
                for j in range(256):
                    demo_image[i, j, 0] = i              # Red channel: Vertical gradient
                    demo_image[i, j, 1] = j              # Green channel: Horizontal gradient
                    demo_image[i, j, 2] = (i + j) // 2   # Blue channel: Diagonal gradient

            # Save demo image
            demo_path = "demo_input.png"
            print(f"Saving demo image to: {demo_path}")
            save_image(demo_image, demo_path)
            print()

            # Encode demo image
            encode_photo(demo_path, "demo_output.jpg", 75)
            print()
            print("Demo complete! You can now test with your own images.")

        except Exception as e:
            print(f"Failed to create demo image: {e}")
            print("Please provide an input image file.")

        sys.exit(0)

    # === Parse command line arguments ===
    input_path = sys.argv[1]  # First argument: Input file
    output_path = sys.argv[2] if len(sys.argv) > 2 else None  # Second argument: Output file (Optional)
    quality = int(sys.argv[3]) if len(sys.argv) > 3 else 75   # Third argument: Quality (Optional)

    # === Validate quality parameter ===
    if not 1 <= quality <= 100:
        print(f"Error: Quality must be between 1 and 100 (Current value: {quality})")
        sys.exit(1)

    # === Execute encoding ===
    encode_photo(input_path, output_path, quality)


# Entry point
if __name__ == "__main__":
    main()