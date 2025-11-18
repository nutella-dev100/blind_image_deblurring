# channel window sizes
DCP_WINDOW = 15
BCP_WINDOW = 15

# regularization weights (Adapted for [0.0, 1.0] float images)
# MU depends on how bcpl0norm is implemented. 
# If B_l0 is a pixel count (e.g. 1,000,000), MU needs to be large (~50-100).
# If B_l0 is normalized (0-1), MU should be small (~0.003).
# Assuming unnormalized pixel count based on typical implementations:
MU = 50.0       

# Gradient & Prior Weights
LAMBDA = 0.02   # Controls how strictly we follow the "kept" gradients
XI = 0.02       # Controls how strictly we follow the Dark Channel prior

# Thresholding
# THETA determines which edges are kept. 
# Threshold T = THETA / LAMBDA. 
# We want T ~ 0.05 (strong edges in float image).
# 0.001 / 0.02 = 0.05.
THETA = 0.001   

GAMMA = 1.0     # Kernel regularization
EPSILON = 1e-6

# pyramid
NUM_SCALES = 4
KERNEL_SIZE = 25   # Must be odd
KERNEL_MAX_SIZE = 35 # Keep small to prevent drift
MAX_ITER = 10

# path 
RESULT_PATH = "results/output.png"