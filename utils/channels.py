import numpy as np
import cv2

def dark_channel(image, window_size=15):
    # image must be uint8 or float32 in range [0,1]
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Step 1: min across RGB channels
    min_channel = np.min(image, axis=2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark = cv2.erode(min_channel, kernel)

    return dark

def bright_channel(image, window_size=15):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Step 1: max across channels
    max_channel = np.max(image, axis=2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    bright = cv2.dilate(max_channel, kernel)

    return bright

#norms
def dcpl0norm(dark):
    return np.count_nonzero(dark)

def bcpl0norm(bright):
    return np.count_nonzero(bright)