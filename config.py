# channel window sizes
DCP_WINDOW = 15
BCP_WINDOW = 15

# regularization weights
MU = 0.0003
LAMBDA = 0.001
XI = 0.02
THETA = 0.01
GAMMA = 1e-3
EPSILON = 1e-6

# pyramid settings
NUM_SCALES = 4
KERNEL_SIZE = 25
KERNEL_MAX_SIZE = 151   # optional: avoid insane kernel growth
MAX_ITER = 20

#path
RESULT_PATH = "results/output.jpeg"