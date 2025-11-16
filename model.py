import cv2
import os
import numpy as np
from utils.channels import bright_channel, dark_channel, bcpl0norm, dcpl0norm
from utils.gradient import compute_gradients
from utils.pyramid import gaussian_pyramid, upsample_kernel, upsample_l, upsample_small_kernel
from utils.threshold import threshold_dark_channel, threshold_gradient
from utils.io import save_image, load_image
from utils.kernel_utils import clean_kernel, pad_and_ifftshift_kernel, postprocess_kernel_spatial
from solver import update_kernel, update_l
import config

def initialize_kernel(kernel_size = None, image_shape = None):
    if kernel_size is None:
        kernel_size = 31

    if isinstance(kernel_size, tuple):
        kh, kw = kernel_size
    else:
        kh = kw = int(kernel_size)

    k = np.zeros((kh, kw), dtype = np.float32)
    cy, cx = kh // 2, kw // 2
    k[cy, cx] = 1.0
    return k

def initialize_l(b, dtype = np.float32):
    if b.dtype == np.uint8:
        l = b.astype(dtype) / 255.0
    else:
        l = b.astype(dtype)
        if l.max() > 1.0:
            l = l / 255
    return l

def final_deconvolution(self, b, k):
    """
    Blend hyper-Laplacian and TV deconvolution or placeholder.
    """
    pass


class DeblurModel:

    def __init__(self, config):
        self.config = config

    def run(self, image_path, save_path="results/output.png"):
        b = load_image(image_path)
        pyramid = gaussian_pyramid(b, self.config.NUM_SCALES)

        #init l and k
        l = initialize_l(pyramid[0])
        k = initialize_kernel(kernel_size=self.config.KERNEL_SIZE)

        for scale_idx in range(len(pyramid)):
            print(f"\n=== Scale {scale_idx+1}/{len(pyramid)} ===")
            b_scaled = pyramid[scale_idx]
            H, W = b_scaled.shape[:2]

            # models.py (inside scale loop before inner iterations)
            if scale_idx > 0:
                l = upsample_l(l, (H, W))
                # upsample small kernel by factor 2 but cap it not to exceed some fraction of image
                max_kh = min(self.config.KERNEL_MAX_SIZE, H) if hasattr(self.config, "KERNEL_MAX_SIZE") else H
                max_kw = min(self.config.KERNEL_MAX_SIZE, W) if hasattr(self.config, "KERNEL_MAX_SIZE") else W
                k = upsample_small_kernel(k, scale_factor=2, max_size=(max_kh, max_kw))
                print(f"AFTER K clean: shape={k.shape}, min={k.min():.6f}, max={k.max():.6f}, sum={k.sum():.6f}")



                if k.ndim != 2:
                    raise ValueError("kernel must be 2D small kernel")

                s = k.sum()
                if s <= 1e-12 or np.isnan(s):
                    print("WARNING: upsampled kernel invalid, replacing with delta")
                    k = np.zeros((3,3), dtype=np.float32); k[1,1]=1.0
                else:
                    k = k / s


            for it in range(self.config.MAX_ITER):
                #bright channel - find w_k
                B = bright_channel(l, window_size=self.config.BCP_WINDOW)
                B_l0 = bcpl0norm(B)
                w_k = self.config.MU / (B_l0 + self.config.EPSILON)

                #dark channel - threshold p
                D = dark_channel(l, window_size=self.config.DCP_WINDOW)
                p = threshold_dark_channel(D, w_k, self.config.XI)

                #gradients and threshold g
                gh, gv = compute_gradients(l)
                g = threshold_gradient((gh, gv), self.config.THETA, self.config.LAMBDA)
                print(f"    g stats: |gh|={np.mean(np.abs(gh)):.5f}, "
                        f"|gv|={np.mean(np.abs(gv)):.5f}, "
                        f"nonzero={np.count_nonzero(gh)+np.count_nonzero(gv)}")


                #update l and k
                H, W = b_scaled.shape
                l = update_l(l = l, k = k, b = b_scaled, g = g, p = p, params=self.config)
                k = update_kernel(
                    l=l,
                    b=b_scaled,
                    gamma=self.config.GAMMA,
                    image_shape=b_scaled.shape,
                    prev_k=k  # important
                    )
                
                print(f"AFTER K clean: shape={k.shape}, min={k.min():.6f}, max={k.max():.6f}, sum={k.sum():.6f}")

                
                #clean kernel ------- UPDATE : Kernel is now stable, do not touch kernel pipeline
                k = clean_kernel(k)
                print("AFTER K clean: shape=", k.shape, "min=", k.min(), "max=", k.max(), "sum=", k.sum())
                s = k.sum()
                assert not np.isnan(s), "kernel sum NaN"
                if s <= 1e-12:
                    print("WARNING: kernel sum too small, replacing with delta")
                    k = np.zeros_like(k); k[k.shape[0]//2, k.shape[1]//2] = 1.0

            
            #end loop

        #final non blind deconv - skipped for now
        print("running final non blind deconv")
        final = l
        save_image(final, save_path)    #TODO : define save path and io funcs
        return final
       