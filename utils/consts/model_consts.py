#
# Author: Ramashish Gaurav
#

from  collections import namedtuple

Layer = namedtuple(
    "Layer", "name num_kernels kernel_dims stride_dims data_format",
    defaults=(None,) * 5) # 5 None values as defaults for 5 fields.

# MODEL_1 = {
#   "name": "model_1",
#   "layers": {
#     "layer_1": Layer(name="Conv", num_kernels=32, kernel_dims=(3, 3),
#                      stride_dims=(1, 1), data_format="channels_last"),
#     "layer_2": Layer(name="Conv", num_kernels=64, kernel_dims=(3, 3),
#                      stride_dims=(1, 1), data_format="channels_last"),
#     "layer_3": Layer(name="Conv", num_kernels=64, kernel_dims=(3, 3),
#                      stride_dims=(2, 2), data_format="channels_last"),
#     "layer_4": Layer(name="Conv", num_kernels=96, kernel_dims=(3, 3),
#                      stride_dims=(2, 2), data_format="channels_last"),
#     "layer_5": Layer(name="Conv", num_kernels=128, kernel_dims=(3, 3),
#                      stride_dims=(2, 2), data_format="channels_last")
#   }
# }

MODEL_ALL_CONV = {
  "name": "all_conv_model",
  "layers": {
    "layer_1": Layer(name="Conv", num_kernels=4, kernel_dims=(1, 1),
                     stride_dims=(1, 1)),
    "layer_2": Layer(name="Conv", num_kernels=8, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_3": Layer(name="Conv", num_kernels=8, kernel_dims=(3, 3),
                     stride_dims=(2, 2)),
    "layer_4": Layer(name="Conv", num_kernels=16, kernel_dims=(3, 3),
                     stride_dims=(1, 1))
  }
}

MODEL_1 = {
  # Total params: 125,989 (MNIST), 175,147 (CIFAR10)
  "name": "model_1",
  "layers": {
    "layer_1": Layer(name="Conv", num_kernels=4, kernel_dims=(1, 1),
                     stride_dims=(1, 1)),
    "layer_2": Layer(name="Conv", num_kernels=8, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_3": Layer(name="MaxPool", kernel_dims=(2, 2)),
    "layer_4": Layer(name="Conv", num_kernels=16, kernel_dims=(3, 3),
                     stride_dims=(1, 1))
  }
}

MODEL_1_AP = {
  # Total params: 125,989 (MNIST), 175,147 (CIFAR10)
  "name": "model_1_ap",
  "layers": {
    "layer_1": Layer(name="Conv", num_kernels=4, kernel_dims=(1, 1),
                     stride_dims=(1, 1)),
    "layer_2": Layer(name="Conv", num_kernels=8, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_3": Layer(name="AvgPool", kernel_dims=(2, 2)),
    "layer_4": Layer(name="Conv", num_kernels=16, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
  }
}

MODEL_2 = {
  # Total params: 13,605 (MNIST), 20,779 (CIFAR10): Conv num_kernels: 8, 16, 16
  # Total params: 25,125 (MINST), 39,467 (CIFAR10): Conv num_kernels: 8, 16, 32

  # Total params: 29,949 (MNIST), 44,291 (CIFAR10): Conv num_kernels: 16, 24, 32
  # => Results in Error: "nengo.exceptions.BuildError: Total synapse bits (1119744)
  # exceeded max (1048576) in LoihiBlock(<Ensemble "conv2d_3.0">) while executing
  # on Loihi.
  "name": "model_2",
  "layers": {
    "layer_1": Layer(name="Conv", num_kernels=4, kernel_dims=(1, 1),
                     stride_dims=(1, 1)),
    "layer_2": Layer(name="Conv", num_kernels=16, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_3": Layer(name="MaxPool", kernel_dims=(2, 2)),
    "layer_4": Layer(name="Conv", num_kernels=24, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_5": Layer(name="MaxPool", kernel_dims=(2, 2)),
    "layer_6": Layer(name="Conv", num_kernels=32, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
  }
}

MODEL_2_AP = {
  # Total params: 13,605 (MNIST), 20,779 (CIFAR10): Conv num_kernels: 8, 16, 16
  # Total params: 25,125 (MINST), 39,467 (CIFAR10): Conv num_kernels: 8, 16, 32

  # Total params: 29,949 (MNIST), 44,291 (CIFAR10): Conv num_kernels: 16, 24, 32
  # => Results in Error: "nengo.exceptions.BuildError: Total synapse bits (1119744)
  # exceeded max (1048576) in LoihiBlock(<Ensemble "conv2d_3.0">) while executing
  # on Loihi.
  "name": "model_2_ap",
  "layers": {
    "layer_1": Layer(name="Conv", num_kernels=4, kernel_dims=(1, 1),
                     stride_dims=(1, 1)),
    "layer_2": Layer(name="Conv", num_kernels=16, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_3": Layer(name="AvgPool", kernel_dims=(2, 2)),
    "layer_4": Layer(name="Conv", num_kernels=24, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_5": Layer(name="AvgPool", kernel_dims=(2, 2)),
    "layer_6": Layer(name="Conv", num_kernels=32, kernel_dims=(3, 3),
                     stride_dims=(1, 1))
  }
}

MODEL_3 = {
  "name": "model_3",
  "layers": {
    "layer_1": Layer(name="Conv", num_kernels=4, kernel_dims=(1, 1),
                     stride_dims=(1, 1)),
    "layer_2": Layer(name="Conv", num_kernels=8, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_3": Layer(name="Conv", num_kernels=16, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_4": Layer(name="MaxPool", kernel_dims=(2, 2)),
    "layer_5": Layer(name="Conv", num_kernels=16, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_6": Layer(name="MaxPool", kernel_dims=(2, 2)),
    "layer_7": Layer(name="Conv", num_kernels=24, kernel_dims=(3, 3),
                     stride_dims=(1, 1))
  }
}

MODEL_4 = {
  "name": "model_4",
  "layers": {
    "layer_1": Layer(name="Conv", num_kernels=4, kernel_dims=(1, 1),
                     stride_dims=(1, 1)),
    "layer_2": Layer(name="Conv", num_kernels=8, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_3": Layer(name="Conv", num_kernels=16, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_4": Layer(name="MaxPool", kernel_dims=(2, 2)),
    "layer_5": Layer(name="Conv", num_kernels=16, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_6": Layer(name="Conv", num_kernels=24, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_7": Layer(name="MaxPool", kernel_dims=(2, 2)),
    "layer_8": Layer(name="Conv", num_kernels=24, kernel_dims=(3, 3),
                     stride_dims=(1, 1))
  }
}

MODEL_5 = {
  "name": "model_5",
  "layers": {
    "layer_1": Layer(name="Conv", num_kernels=4, kernel_dims=(1, 1),
                     stride_dims=(1, 1)),
    "layer_2": Layer(name="Conv", num_kernels=8, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_3": Layer(name="Conv", num_kernels=16, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_4": Layer(name="MaxPool", kernel_dims=(2, 2)),
    "layer_5": Layer(name="Conv", num_kernels=24, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_6": Layer(name="Conv", num_kernels=24, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_7": Layer(name="MaxPool", kernel_dims=(2, 2)),
    "layer_8": Layer(name="Conv", num_kernels=32, kernel_dims=(3, 3),
                     stride_dims=(1, 1))
  }
}

MODEL_6 = {
  "name": "model_6",
  "layers": {
    "layer_1": Layer(name="Conv", num_kernels=4, kernel_dims=(1, 1),
                     stride_dims=(1, 1)),
    "layer_2": Layer(name="Conv", num_kernels=8, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_3": Layer(name="MaxPool", kernel_dims=(2, 2)),
    "layer_4": Layer(name="Conv", num_kernels=16, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_5": Layer(name="MaxPool", kernel_dims=(2, 2)),
    "layer_6": Layer(name="Conv", num_kernels=24, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_7": Layer(name="Conv", num_kernels=32, kernel_dims=(3, 3),
                     stride_dims=(1, 1))
  }
}

MODEL_7 = { # TF CIFAR10: Without Data Augmentation: 71% ACC. (32 EPOCHS)
            # TF CIFAR10: With Data Augmentation: 78% ACC. (32 EPOCHS)
  "name": "model_7",
  "layers": {
    "layer_1": Layer(name="Conv", num_kernels=4, kernel_dims=(1, 1),
                     stride_dims=(1, 1)),

    "layer_2": Layer(name="Conv", num_kernels=32, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_3": Layer(name="Conv", num_kernels=32, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_4": Layer(name="MaxPool", kernel_dims=(2, 2)),
    # num_kernels is Dropout Probability.
    "layer_5": Layer(name="Dropout", num_kernels=0.2),


    "layer_6": Layer(name="Conv", num_kernels=64, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_7": Layer(name="Conv", num_kernels=64, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_8": Layer(name="MaxPool", kernel_dims=(2, 2)),
    # num_kernels is Dropout Probability.
    "layer_9": Layer(name="Dropout", num_kernels=0.3),


    "layer_10": Layer(name="Conv", num_kernels=128, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_11": Layer(name="Conv", num_kernels=128, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_12": Layer(name="MaxPool", kernel_dims=(2, 2)),
    # num_kernels is Dropout Probability.
    "layer_13": Layer(name="Dropout", num_kernels=0.4),


    "layer_14": Layer(name="Conv", num_kernels=128, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_15": Layer(name="Conv", num_kernels=128, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_16": Layer(name="MaxPool", kernel_dims=(2, 2)),
    # num_kernels is Dropout Probability.
    "layer_17": Layer(name="Dropout", num_kernels=0.4),


    "layer_18": Layer(name="Conv", num_kernels=128, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_19": Layer(name="Conv", num_kernels=128, kernel_dims=(3, 3),
                     stride_dims=(1, 1)),
    "layer_20": Layer(name="MaxPool", kernel_dims=(2, 2)),
    # num_kernels is Dropout Probability.
    "layer_21": Layer(name="Dropout", num_kernels=0.4),
  }
}

"""
https://www.kaggle.com/amyjang/tensorflow-cifar10-cnn-tutorial

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, 3, padding='same', input_shape=x_train.shape[1:], activation='relu'),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax'),
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-06),
            loss='categorical_crossentropy', metrics=['acc'])
"""
