import cv2
import numpy as np
import os


def load_image(path, to_gray=True):
    """
    Loads an image from disk, converts to grayscale if needed,
    and normalizes to float32 in range [0, 1].

    Parameters:
        path : str   - path to image file
        to_gray : bool - convert to grayscale (recommended for deblurring)

    Returns:
        image : float32 array in [0,1], shape (H, W) if grayscale
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cv2 cannot load image: {path}")

    # Convert BGR -> RGB internally if needed (optional)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if to_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # To float32 in [0,1]
    img = img.astype(np.float32)

    if img.max() > 1.0:
        img /= 255.0

    return img


def save_image(img, path):
    """
    Saves an image to disk.
    Accepts img in range [0,1] float or uint8 in [0,255].

    Parameters:
        img  : numpy array
        path : str - where to save
    """

    # Create folder if needed
    folder = os.path.dirname(path)
    if folder != "" and not os.path.exists(folder):
        os.makedirs(folder)

    out = img.copy()

    # If float, convert to uint8
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 1)
        out = (out * 255).astype(np.uint8)

    cv2.imwrite(path, out)
    print(f"[INFO] Saved image to {path}")


def to_gray(img):
    """
    Ensures an image is grayscale float32 in [0,1].
    Useful if user feeds RGB images.

    Parameters:
        img : numpy array (H,W) or (H,W,3)

    Returns:
        gray : float32 grayscale image
    """

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0

    return img
