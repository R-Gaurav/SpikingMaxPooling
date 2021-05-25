#
# Author: Ramashish Gaurav
#

SEED = 90
MNIST = "mnist"
CIFAR10 = "cifar10"
NUM_X = 4 # Default number of elements in the MaxPooling pool matrix.

# Following variables NEURONS_LAST_SPIKED_TS, NEURONS_LATEST_ISI, MAX_POOL_MASK
# are supposed to be respectively initialized in `nengo_dl_test.py` as
# np.zeros((num_chnls, rows, cols)), np.ones((num_chnls, rows, cols))*np.inf,
# and np.ones((num_chnls, rows, cols)) / NUM_X for each MaxPooling layer.
NEURONS_LAST_SPIKED_TS = None
NEURONS_LATEST_ISI = None
MAX_POOL_MASK = None

