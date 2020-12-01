#
# Author: Ramashish Gaurav
#

import numpy as np
import tensorflow as tf

def get_cifar_10_data():
  """
  Returns normalized CIFAR-10 images and binarized (i.e. one-hot encoded) labels.

  Returns:
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
  """
  # Dataset download
  (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
  # Normalize the images in range [-1, 1]
  train_x = train_x.astype(np.float32) / 127.5 - 1
  test_x = test_x.astype(np.float32) / 127.5 - 1
  # Binarize the labels.
  train_y = np.eye(10, dtype=np.float32)[train_y].squeeze(axis=1)
  test_y = np.eye(10, dtype=np.float32)[test_y].squeeze(axis=1)

  return train_x, train_y, test_x, test_y
