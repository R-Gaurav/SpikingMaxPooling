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

MODEL_1 = {
  # Total params: 125,989 (MNIST), 175,147 (CIFAR10)
  "name": "model_1",
  "layers": {
    "layer_1": Layer(name="Conv", num_kernels=4, kernel_dims=(1, 1),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_2": Layer(name="Conv", num_kernels=8, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_3": Layer(name="MaxPool", kernel_dims=(2, 2),
                     data_format="channels_first"),
    "layer_4": Layer(name="Conv", num_kernels=16, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first")
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
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_2": Layer(name="Conv", num_kernels=8, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_3": Layer(name="MaxPool", kernel_dims=(2, 2),
                     data_format="channels_first"),
    "layer_4": Layer(name="Conv", num_kernels=16, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_5": Layer(name="MaxPool", kernel_dims=(2, 2),
                     data_format="channels_first"),
    "layer_6": Layer(name="Conv", num_kernels=24, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first")
  }
}

MODEL_3 = {
  "name": "model_3",
  "layers": {
    "layer_1": Layer(name="Conv", num_kernels=4, kernel_dims=(1, 1),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_2": Layer(name="Conv", num_kernels=8, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_3": Layer(name="Conv", num_kernels=16, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_4": Layer(name="MaxPool", kernel_dims=(2, 2),
                     data_format="channels_first"),
    "layer_5": Layer(name="Conv", num_kernels=16, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_6": Layer(name="MaxPool", kernel_dims=(2, 2),
                     data_format="channels_first"),
    "layer_7": Layer(name="Conv", num_kernels=24, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first")
  }
}

MODEL_4 = {
  "name": "model_4",
  "layers": {
    "layer_1": Layer(name="Conv", num_kernels=4, kernel_dims=(1, 1),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_2": Layer(name="Conv", num_kernels=8, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_3": Layer(name="Conv", num_kernels=16, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_4": Layer(name="MaxPool", kernel_dims=(2, 2),
                     data_format="channels_first"),
    "layer_5": Layer(name="Conv", num_kernels=16, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_6": Layer(name="Conv", num_kernels=24, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_7": Layer(name="MaxPool", kernel_dims=(2, 2),
                     data_format="channels_first"),
    "layer_8": Layer(name="Conv", num_kernels=24, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first")
  }
}

MODEL_5 = {
  "name": "model_5",
  "layers": {
    "layer_1": Layer(name="Conv", num_kernels=4, kernel_dims=(1, 1),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_2": Layer(name="Conv", num_kernels=8, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_3": Layer(name="Conv", num_kernels=16, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_4": Layer(name="MaxPool", kernel_dims=(2, 2),
                     data_format="channels_first"),
    "layer_5": Layer(name="Conv", num_kernels=24, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_6": Layer(name="Conv", num_kernels=24, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_7": Layer(name="MaxPool", kernel_dims=(2, 2),
                     data_format="channels_first"),
    "layer_8": Layer(name="Conv", num_kernels=32, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first")
  }
}

MODEL_6 = {
  "name": "model_6",
  "layers": {
    "layer_1": Layer(name="Conv", num_kernels=4, kernel_dims=(1, 1),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_2": Layer(name="Conv", num_kernels=8, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_3": Layer(name="MaxPool", kernel_dims=(2, 2),
                     data_format="channels_first"),
    "layer_4": Layer(name="Conv", num_kernels=16, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_5": Layer(name="MaxPool", kernel_dims=(2, 2),
                     data_format="channels_first"),
    "layer_6": Layer(name="Conv", num_kernels=24, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_7": Layer(name="Conv", num_kernels=32, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first"),
  }
}

MODEL_7 = { # TF CIFAR10: Without Data Augmentation: 71% ACC. (32 EPOCHS)
            # TF CIFAR10: With Data Augmentation: 78% ACC. (32 EPOCHS)
  "name": "model_7",
  "layers": {
    "layer_1": Layer(name="Conv", num_kernels=4, kernel_dims=(1, 1),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_2": Layer(name="Conv", num_kernels=32, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_3": Layer(name="Conv", num_kernels=32, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_4": Layer(name="MaxPool", kernel_dims=(2, 2),
                     data_format="channels_first"),
    "layer_5": Layer(name="Conv", num_kernels=64, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_6": Layer(name="Conv", num_kernels=64, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_7": Layer(name="MaxPool", kernel_dims=(2, 2),
                     data_format="channels_first"),
    "layer_8": Layer(name="Conv", num_kernels=128, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first"),
    "layer_9": Layer(name="Conv", num_kernels=128, kernel_dims=(3, 3),
                     stride_dims=(1, 1), data_format="channels_first"),
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
