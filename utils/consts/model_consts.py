#
# Author: Ramashish Gaurav
#

from  collections import namedtuple

Layer = namedtuple(
    "Layer", "name num_kernels kernel_dims stride_dims data_format",
    defaults=(None,) * 5) # 5 None values as defaults for 5 fields.

MODEL_1 = {
  "name": "model_1",
  "layers": {
    "layer_1": Layer(name="Conv", num_kernels=32, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_last"),
    "layer_2": Layer(name="Conv", num_kernels=64, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_last"),
    "layer_3": Layer(name="Conv", num_kernels=64, kernel_dims=(3, 3),
                     stride_dims=(2, 2), data_format="channels_last"),
    "layer_4": Layer(name="Conv", num_kernels=96, kernel_dims=(3, 3),
                     stride_dims=(2, 2), data_format="channels_last"),
    "layer_5": Layer(name="Conv", num_kernels=128, kernel_dims=(3, 3),
                     stride_dims=(2, 2), data_format="channels_last")
  }
}

MODEL_2 = {
  "name": "model_2",
  "layers": {
    "layer_1": Layer(name="Conv", num_kernels=16, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_2": Layer(name="MaxPool", kernel_dims=(2, 2),
                     data_format="channels_first"),
    "layer_3": Layer(name="Conv", num_kernels=24, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first")
  }
}

MODEL_3 = {
  "name": "model_3",
  "layers": {
    "layer_1": Layer(name="Conv", num_kernels=16, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_2": Layer(name="MaxPool", kernel_dims=(2, 2),
                     data_format="channels_first"),
    "layer_3": Layer(name="Conv", num_kernels=24, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_4": Layer(name="MaxPool", kernel_dims=(2, 2),
                     data_format="channels_first"),
    "layer_5": Layer(name="Conv", num_kernels=32, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first")
  }
}
