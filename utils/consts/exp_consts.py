#
# Author: Ramashish Gaurav
#

SEED = 0
MNIST = "mnist"
CIFAR10 = "cifar10"
NUM_X = 4 # Default number of elements in the MaxPooling pool matrix.

# Following variables NEURONS_LAST_SPIKED_TS, NEURONS_LATEST_ISI, MAX_POOL_MASK
# are supposed to be respectively initialized in `nengo_dl_test.py` as
# np.zeros((num_chnls, rows, cols)), np.ones((num_chnls, rows, cols))*np.inf,
# and np.ones((num_chnls, rows, cols)) / NUM_X for each MaxPooling layer.
NEURONS_LAST_SPIKED_TS = {}
NEURONS_LATEST_ISI = {}
MAX_POOL_MASK = {}

ISI_BASED_MP_PARAMS = {}

AVAM = "avam"
MJOP = "max_join_op"
AVGP = "avg_pooling"
