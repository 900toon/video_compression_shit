"""
Image Utility Functions Module

This module provides utility functions for loading and saving images, supporting multiple image formats.
It automatically selects available image processing libraries (PIL/Pillow, OpenCV, Matplotlib).

Key Features:
- Load Image: Supports common formats like PNG, BMP, JPEG, etc.
- Save Image: Saves numpy arrays as image files.
- Check Dependencies: Verifies which image processing libraries are installed on the system.
"""

import numpy as np
from typing import Optional


def load_image(file_path: str) -> np.ndarray:
    """
    Loads an image from a file and converts it into a numpy array in RGB format.

    This function attempts to use different image processing libraries in priority order:
    1. PIL/Pillow (Recommended, supports the most formats)
    2. OpenCV (If installed)
    3. Matplotlib (Fallback)

    Args:
        file_path: Path to the image file.

    Returns:
        RGB image array with shape (height, width, 3) and value range [0, 255].

    Raises:
        ImportError: If no image loading library is available.
        FileNotFoundError: If the image file does not exist.
    """
    # Method 1: Try using PIL/Pillow (Most common and best compatibility)
    try:
        from PIL import Image
        # Open image file
        img = Image.open(file_path)
        # Convert to RGB mode (Handles RGBA, Grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Convert to numpy array and return
        return np.array(img, dtype=np.uint8)
    except ImportError:
        # PIL/Pillow not installed, try other methods
        pass

    # Method 2: Try using OpenCV
    try:
        import cv2
        # Read image file
        img = cv2.imread(file_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {file_path}")
        # OpenCV loads in BGR format, convert to RGB
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except ImportError:
        # OpenCV not installed, try other methods
        pass

    # Method 3: Try using Matplotlib as a fallback
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        # Read image
        img = mpimg.imread(file_path)
        # Handle different value formats
        if img.dtype == np.float32 or img.dtype == np.float64:
            # Convert float format (0.0-1.0) to integer format (0-255)
            img = (img * 255).astype(np.uint8)
        # Handle RGBA format (remove alpha channel)
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        # Handle grayscale images (convert to RGB)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        return img
    except ImportError:
        # All methods failed, Matplotlib is also not installed
        pass

    # If no library is available, raise an error with installation instructions
    raise ImportError(
        "No available image loading library found. Please install one of the following packages:\n"
        "  - Pillow: pip install Pillow (Recommended)\n"
        "  - OpenCV: pip install opencv-python\n"
        "  - Matplotlib: pip install matplotlib"
    )


def save_image(image: np.ndarray, file_path: str):
    """
    Saves a numpy array as an image file.

    This function attempts to use different image processing libraries to save the image in priority order.

    Args:
        image: RGB image array with shape (height, width, 3) and value range [0, 255].
        file_path: Output file path.
    """
    # Method 1: Try using PIL/Pillow to save
    try:
        from PIL import Image
        # Convert numpy array to PIL Image object
        img = Image.fromarray(image.astype(np.uint8))
        # Save image
        img.save(file_path)
        return
    except ImportError:
        # PIL/Pillow not installed, try other methods
        pass

    # Method 2: Try using OpenCV to save
    try:
        import cv2
        # Convert RGB to BGR (OpenCV uses BGR format)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Write image file
        cv2.imwrite(file_path, bgr)
        return
    except ImportError:
        # OpenCV not installed, try other methods
        pass

    # Method 3: Try using Matplotlib to save
    try:
        import matplotlib.pyplot as plt
        # Save image
        plt.imsave(file_path, image)
        return
    except ImportError:
        # All methods failed
        pass

    # If no library is available, raise an error
    raise ImportError("No available image saving library found. Please install Pillow, OpenCV, or Matplotlib.")


def check_dependencies() -> dict:
    """
    Checks which image processing libraries are installed on the system.

    This function attempts to import each image processing library and reports its availability status.

    Returns:
        A dictionary containing library names and their availability status.
        Example: {'Pillow': True, 'OpenCV': False, 'Matplotlib': True}
    """
    libraries = {}

    # Check if Pillow is installed
    try:
        import PIL
        libraries['Pillow'] = True
    except ImportError:
        libraries['Pillow'] = False

    # Check if OpenCV is installed
    try:
        import cv2
        libraries['OpenCV'] = True
    except ImportError:
        libraries['OpenCV'] = False

    # Check if Matplotlib is installed
    try:
        import matplotlib
        libraries['Matplotlib'] = True
    except ImportError:
        libraries['Matplotlib'] = False

    return libraries