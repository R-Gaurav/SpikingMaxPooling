#
# Author: Ramashish Gaurav
#

from  collections import namedtuple

Layer = namedtuple("Layer", "num_kernels kernel_dims stride_dims")

MODEL_1 = {
  "name": "model_1",
  "layers": {
    "layer_1": Layer(num_kernels=32, kernel_dims=(3, 3), stride_dims=(1, 1)),
    "layer_2": Layer(num_kernels=64, kernel_dims=(3, 3), stride_dims=(1, 1)),
    "layer_3": Layer(num_kernels=64, kernel_dims=(3, 3), stride_dims=(2, 2)),
    "layer_4": Layer(num_kernels=96, kernel_dims=(3, 3), stride_dims=(2, 2)),
    "layer_5": Layer(num_kernels=128, kernel_dims=(3, 3), stride_dims=(2, 2))
  }
}
