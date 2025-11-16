import numpy as np
import cv2

def dark_channel(image, window_size=15):
    # If the image is grayscale (H,W), skip channel min
    if image.ndim == 2:
        min_channel = image
    else:
        min_channel = np.min(image, axis=2)

    # Apply erosion (min filter)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark = cv2.erode(min_channel, kernel)

    return dark.astype(np.float32)


def bright_channel(image, window_size=15):
    # If the image is grayscale (H,W), skip channel max
    if image.ndim == 2:
        max_channel = image
    else:
        max_channel = np.max(image, axis=2)

    # Apply dilation (max filter)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    bright = cv2.dilate(max_channel, kernel)

    return bright.astype(np.float32)


#norms
def dcpl0norm(dark):
    return np.count_nonzero(dark)

def bcpl0norm(bright):
    return np.count_nonzero(bright)