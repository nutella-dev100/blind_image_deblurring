import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Import your existing utils
# Ensure these files exist in your project structure as indicated in your upload
from utils.channels import bright_channel, dark_channel, bcpl0norm, dcpl0norm
from utils.gradient import compute_gradients
from utils.pyramid import gaussian_pyramid, upsample_kernel, upsample_l, upsample_small_kernel
from utils.threshold import threshold_gradient
from utils.io import save_image, load_image
from utils.kernel_utils import clean_kernel, pad_and_ifftshift_kernel, postprocess_kernel_spatial
from solver import update_kernel, update_l
import config

class DeblurModel:

    def __init__(self, config):
        self.config = config

    def initialize_kernel(self, kernel_size=None):
        if kernel_size is None:
            kernel_size = 31

        if isinstance(kernel_size, tuple):
            kh, kw = kernel_size
        else:
            kh = kw = int(kernel_size)

        # Initialize as a simple delta (one pixel in center)
        k = np.zeros((kh, kw), dtype=np.float32)
        cy, cx = kh // 2, kw // 2
        k[cy, cx] = 1.0
        return k

    def initialize_l(self, b, dtype=np.float32):
        if b.dtype == np.uint8:
            l = b.astype(dtype) / 255.0
        else:
            l = b.astype(dtype)
            # Basic check if normalization is needed
            if l.max() > 1.0:
                l = l / 255.0
        return l
    
    def final_restore(self, img, kernel):
        """
        Performs final deconvolution using a robust Wiener Filter.
        This is more stable than the iterative solver for the final output step.
        """
        print("Running final non-blind deconvolution (Wiener)...")
        
        from numpy.fft import fft2, ifft2, ifftshift
        
        # 1. Ensure kernel is normalized and padded
        kernel = kernel / (np.sum(kernel) + 1e-8)
        H, W = img.shape[:2]
        kh, kw = kernel.shape
        
        # Pad kernel to image size
        k_pad = np.zeros((H, W), dtype=np.float32)
        # Place in center
        y_off, x_off = (H - kh) // 2, (W - kw) // 2
        k_pad[y_off:y_off+kh, x_off:x_off+kw] = kernel
        # Shift center to origin (0,0) for correct phase
        k_pad = ifftshift(k_pad)
        
        # Precompute Kernel FFT
        # We process over (0,1) axes. If img is RGB, we broadcast K.
        K_f = fft2(k_pad)
        
        # 2. Wiener Deconvolution Function
        # Formula: F = (Conj(K) * G) / (|K|^2 + SNR_const)
        # SNR_const: Higher = smoother/blurrier. Lower = sharper/noisier.
        # 0.01 - 0.05 is a good range for visual results.
        snr_const = 0.015
        
        denom = np.abs(K_f)**2 + snr_const
        
        def apply_wiener(channel_img):
            Y_f = fft2(channel_img)
            numer = np.conj(K_f) * Y_f
            res = np.real(ifft2(numer / denom))
            return res

        # 3. Apply to channels
        if img.ndim == 3:
            final = np.zeros_like(img)
            for c in range(3):
                final[:,:,c] = apply_wiener(img[:,:,c])
        else:
            final = apply_wiener(img)
            
        return np.clip(final, 0, 1)

    def run(self, image_path, save_path="results/output.png"):
        print(f"Loading image from {image_path}...")
        b_full = load_image(image_path)
        
        # Generate Gaussian pyramid for coarse-to-fine strategy
        pyramid = gaussian_pyramid(b_full, self.config.NUM_SCALES)

        # Initialize Latent Image (l) and Kernel (k) at the coarsest scale
        l = self.initialize_l(pyramid[0])
        k = self.initialize_kernel(kernel_size=self.config.KERNEL_SIZE)

        # === Coarse-to-Fine Loop ===
        for scale_idx in range(len(pyramid)):
            print(f"\n=== Scale {scale_idx+1}/{len(pyramid)} ===")
            b_scaled = pyramid[scale_idx]
            H, W = b_scaled.shape[:2]

            # Adjust Gamma (noise suppression) per scale
            # Coarser scales need higher regularization to find the kernel structure
            if scale_idx == 0:
                self.config.GAMMA = 0.5
            elif scale_idx == 1:
                self.config.GAMMA = 0.2
            else:
                self.config.GAMMA = 0.05

            # Upsample l and k from previous scale (if not at the start)
            if scale_idx > 0:
                # Upsample latent image to current size
                l = upsample_l(l, (H, W))
                
                # Upsample kernel
                # Cap kernel size to avoid it growing too large for the image
                max_kh = min(self.config.KERNEL_MAX_SIZE, H)
                max_kw = min(self.config.KERNEL_MAX_SIZE, W)
                k = upsample_small_kernel(k, scale_factor=2, max_size=(max_kh, max_kw))
                
                # Safety check for kernel validity
                if k.sum() <= 1e-12:
                    print("WARNING: Upsampled kernel energy lost. Resetting to delta.")
                    k = self.initialize_kernel(k.shape)
                else:
                    k = k / k.sum()

            # === Inner Iteration Loop (Alternating Minimization) ===
            for it in range(self.config.MAX_ITER):
                
                # 1. Compute Weight (w_k) based on Bright Channel sparsity
                # This helps adapt the strength of the prior
                B = bright_channel(l, window_size=self.config.BCP_WINDOW)
                B_l0 = bcpl0norm(B)
                w_k = self.config.MU / (B_l0 + self.config.EPSILON)

                # 2. Compute Prior 'p' (Based on Dark Channel)
                # [CRITICAL FIX]: The prior p must be 3-channel RGB to match 'l' in the solver.
                # We create 'p' by taking 'l' and forcing dark/noisy regions to zero.
                D = dark_channel(l, window_size=self.config.DCP_WINDOW)
                
                # Threshold determines which Dark Channel values are considered "noise/blur"
                threshold_val = w_k / self.config.XI
                
                # Create a copy of l to be our target 'p'
                p = l.copy()
                
                # Logic: If Dark Channel < Threshold, it SHOULD be zero (sparsity).
                # So we force those pixels in 'p' to black. 
                # If Dark Channel > Threshold (e.g. sky/edges), we keep 'l' as is.
                mask_should_be_zero = D < threshold_val
                
                # Handle broadcasting mask (H,W) to (H,W,3)
                if l.ndim == 3:
                    mask_broadcast = np.repeat(mask_should_be_zero[:, :, np.newaxis], 3, axis=2)
                    p[mask_broadcast] = 0.0
                else:
                    p[mask_should_be_zero] = 0.0

                # 3. Compute Gradients 'g' and threshold them
                # This preserves sharp edges (large gradients) and smooths texture (small gradients)
                gh, gv = compute_gradients(l)
                g = threshold_gradient((gh, gv), self.config.THETA, self.config.LAMBDA)
                
                # Debug info
                grad_count = np.count_nonzero(g[0]) + np.count_nonzero(g[1])
                print(f"    Iter {it+1}: |Grad| kept {grad_count} px")

                # 4. Update Latent Image 'l' (FFT Solver)
                # Now passing the corrected 3-channel 'p'
                l = update_l(l=l, k=k, b=b_scaled, g=g, p=p, params=self.config)

                # 5. Update Blur Kernel 'k'
                k = update_kernel(
                    l=l,
                    b=b_scaled,
                    gamma=self.config.GAMMA,
                    image_shape=b_scaled.shape[:2],
                    prev_k=k 
                )
                
                # 6. Post-process Kernel (Center, Crop, Normalize, Threshold)
                k = clean_kernel(k)
                
                # Ensure kernel is valid
                if k.sum() <= 1e-12:
                    print("    WARNING: Kernel vanished. Resetting to delta.")
                    k = self.initialize_kernel(k.shape)

        # === End of Loops ===
        
        print("\nOptimization finished.")
        
        final_image = self.final_restore(b_full, k)

        # === Visualization ===
        # Center the kernel for display so it doesn't look split at corners
        k_display = np.fft.fftshift(k)
        # Scale kernel for visibility
        k_display = k_display / k_display.max()
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.title("Input Blur")
        plt.imshow(cv2.cvtColor(b_full, cv2.COLOR_BGR2RGB) if b_full.ndim==3 else b_full, cmap='gray')
        
        plt.subplot(1, 3, 2)
        plt.title("Estimated Kernel")
        plt.imshow(k_display, cmap='gray')
        
        plt.subplot(1, 3, 3)
        plt.title("Final Deconvolution")
        plt.imshow(np.clip(final_image, 0, 1), cmap='gray')
        plt.show()

        # Save results
        final_result_uint8 = np.clip(final_image * 255, 0, 255).astype(np.uint8)
        return final_result_uint8