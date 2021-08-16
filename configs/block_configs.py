#
# Author: Ramashish Gaurav.
#
# Contains the block configuration for MNIST and CIFAR dataset.
#

block_shapes = {
  "channels_last": {
    "mnist": {
      "model_1": {
        "conv2d_0": (26, 26, 1),
        "conv2d_1": (11, 11, 8)
      },
      "model_2": {
        "conv2d_0": (26, 26, 1),
        "conv2d_1": (11, 11, 8),
        "conv2d_2": (3, 3, 32)
      },
    },
    #########################################################
    "cifar10": {
      "model_1": {
        "conv2d_0": (30, 30, 1),
        "conv2d_1": (13, 13, 4),
      },
      "model_2": {
        "conv2d_0": (30, 30, 1),
        "conv2d_1": (13, 13, 4),
        "conv2d_2": (5, 5, 24)
      },
      "model_7": {
        "conv2d_0": (8, 8, 16),#(1, 30, 30),
        "conv2d_1": (28, 28, 1), #(8, 8, 16), #(28, 28, 1),
        "conv2d_2": (8, 8, 8), # (8, 8, 16), (12, 12, 4),
        "conv2d_3": (8, 8, 8), #(10, 10, 4),
        "conv2d_4": (3, 3, 12),# (4, 4, 8), (3, 3, 16)
        "conv2d_5": (4, 4, 64),
      },
      "all_conv_model": {
        "conv2d_0": (30, 30, 1),
        "conv2d_1": (14, 14, 4),
        "conv2d_2": (12, 12, 4),
      }
    },
    ##########################################################
    "fashion_mnist": {
      "model_1": {
        "conv2d_0": (26, 26, 1),
        "conv2d_1": (11, 11, 8)
      },
      "model_2": {
        "conv2d_0": (26, 26, 1),
        "conv2d_1": (11, 11, 8),
        "conv2d_2": (3, 3, 32)
      },
      "model_7": {
        "conv2d_0": (8, 8, 16),#(1, 30, 30),
        "conv2d_1": (28, 28, 1), #(8, 8, 16), #(28, 28, 1),
        "conv2d_2": (8, 8, 8), # (8, 8, 16), (12, 12, 4),
        "conv2d_3": (8, 8, 8), #(10, 10, 4),
        "conv2d_4": (3, 3, 12),# (4, 4, 8), (3, 3, 16)
        "conv2d_5": (4, 4, 64),
      },
    },
  },

  ##########################################################################
  ##########################################################################
  "channels_first": {
    "mnist": {
      "model_1": {
        "conv2d_0": (1, 26, 26),
        "conv2d_1": (8, 11, 11)
      },
      "model_2": {
        "conv2d_0": (1, 26, 26),
        "conv2d_1": (8, 11, 11),
        "conv2d_2": (32, 3, 3,)
      },
    },
    #########################################################
    "cifar10": {
      "model_1": {
        "conv2d_0": (1, 30, 30),
        "conv2d_1": (4, 13, 13),
      },
      "model_2": {
        "conv2d_0": (1, 30, 30),
        "conv2d_1": (4, 13, 13),
        "conv2d_2": (24, 5, 5)
      },
      "model_7": {
        "conv2d_0": (16, 8, 8),
        "conv2d_1": (1, 28, 28),
        "conv2d_2": (8, 8, 8),
        "conv2d_3": (8, 8, 8),
        "conv2d_4": (12, 3, 3),
        "conv2d_5": (64, 4, 4),
      }
    },
    ########################################################
    "fashion_mnist": {
      "model_1": {
        "conv2d_0": (1, 26, 26),
        "conv2d_1": (8, 11, 11)
      },
      "model_2": {
        "conv2d_0": (1, 26, 26),
        "conv2d_1": (8, 11, 11),
        "conv2d_2": (32, 3, 3,)
      },
    },
  }
}


