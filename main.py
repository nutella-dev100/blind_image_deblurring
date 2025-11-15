from utils.pyramid import gaussian_pyramid, upsample_kernel, upsample_l
from model import initialize_l, initialize_kernel
import numpy as np
import cv2

if __name__ == "__main__":
    # --- Load a random or sample image ---
    img = cv2.imread("test.jpeg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Build pyramid ---
    pyramid = gaussian_pyramid(img, num_levels=4)
    print("Pyramid shapes (coarse → fine):")
    for i, p in enumerate(pyramid):
        print(f" Level {i}: {p.shape}")

    # --- Initialize l at coarsest level ---
    l0 = initialize_l(pyramid[0])
    print("Initial l shape:", l0.shape)

    # --- Initialize small kernel ---
    k0 = initialize_kernel(kernel_size=21)
    print("Initial kernel shape:", k0.shape)

    # --- Upsample to next finer level ---
    next_shape = pyramid[1].shape[:2]
    l1 = upsample_l(l0, next_shape)
    k1 = upsample_kernel(k0, next_shape)

    print("Upsampled l shape:", l1.shape)
    print("Upsampled k shape:", k1.shape)

    print("\nSanity check complete. Shapes look consistent.")
