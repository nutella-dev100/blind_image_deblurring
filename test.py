class DeblurModel:
    def __init__(self, config):
        self.config = config

    def run(self, image_path):
        # ---------------------------------------------------------
        # 1. Load blurry input image b
        # ---------------------------------------------------------
        b = load_image(...)
        
        # ---------------------------------------------------------
        # 2. Build pyramid (coarse-to-fine)
        # ---------------------------------------------------------
        pyramid = build_gaussian_pyramid(b, self.config.NUM_SCALES)

        # ---------------------------------------------------------
        # 3. Initialize kernel k and intermediate image l
        # ---------------------------------------------------------
        k = initialize_kernel(...)
        l = initialize_image(...)

        # ---------------------------------------------------------
        # 4. Loop through pyramid from coarse → fine
        # ---------------------------------------------------------
        for scale in pyramid:
            b_scaled = pyramid[scale]

            # Upsample previous level's l and k to this scale
            l = upsample_l(l, scale)
            k = upsample_kernel(k, scale)

            # ---------------------------------------------
            # 5. Inner Iterations at Current Scale
            # ---------------------------------------------
            for iter in range(self.config.MAX_ITER):

                # ======================
                # (A) Bright channel → compute w_k
                # ======================
                B = bright_channel(l)
                B_l0 = l0_norm(B)
                w_k = self.config.MU / (B_l0 + self.config.EPSILON)

                # ======================
                # (B) Dark channel → compute D(l)
                # ======================
                D = dark_channel(l)

                # ======================
                # (C) L0 thresholding on dark channel → compute p
                # ======================
                p = threshold_dark_channel(D, w_k, self.config.XI)

                # ======================
                # (D) Compute gradients ∇l
                # ======================
                grad = compute_gradients(l)

                # ======================
                # (E) L0 thresholding on gradient → compute g
                # ======================
                g = threshold_gradient(grad, self.config.THETA, self.config.LAMBDA)

                # ======================
                # (F) Update intermediate image l (FFT solver)
                # ======================
                l = solver.update_l(
                    l=l,
                    k=k,
                    b=b_scaled,
                    g=g,
                    p=p,
                    params=self.config
                )

                # ======================
                # (G) Update kernel k (FFT solver)
                # ======================
                k = solver.update_kernel(
                    l=l,
                    b=b_scaled,
                    gamma=self.config.GAMMA
                )

                # ======================
                # (H) Normalize / clean kernel
                # ======================
                k = normalize_kernel(k)
                k = threshold_small_values(k)

            # end inner iter

        # end pyramid loop

        # ---------------------------------------------------------
        # 6. Final non-blind deconvolution for fine textures
        # ---------------------------------------------------------
        latent = final_deconvolution(b, k)

        # ---------------------------------------------------------
        # 7. Return / save result
        # ---------------------------------------------------------
        save_image(latent)
        return latent
