import cv2
import numpy as np
from utils import channels, gradient, pyramid

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

    def run(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR) #Input blurry image
        #pyramid = build_gaussian_pyramid(b, self.config.NUM_SCALES)

        #k = initialize_kernel(...)
        #l = initialize_image(...)

        for scale in pyramid:
            b_scaled = pyramid[scale]
            l = pyramid.upsample_l(l, scale)
            k = pyramid.upsample_kernel(k, scale)

            for iter in range(self.config.MAX_ITER):
                #compute w_k
                B = channels.bright_channel(l)
                B_l0 = channels.bcpl0norm(B)
                w_k = self.config.MU / (B_l0 + self.config.EPSILON)

                #compute D(l)
                D = channels.dark_channel(l)

                #compute gradients (grad(l))
                grad = gradient.compute_gradients(l)

                #compute g
                g = gradient.threshold_gradient(grad, self.config.THETA, self.config.LAMBDA)

                #update intermediate image(FFT solver)

                #update kernel(FFT solver)

                #normalise and clean kernel

            #inner iter ka end

        #pyramid loop ka end

        #non blind deconv for fine texture (optional), ref 46 and ref 47 ka kaam

        #save result and return

