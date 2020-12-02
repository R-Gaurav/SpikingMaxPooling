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

def get_batches_of_cifar_data(ngo_cfg, is_test=True):
  """
  Returns the batches of training or test data.

  Args:
    ngo_cfg <dict>: The Nengo-DL related configuration.
    is_test <True>: Returns the test batches if True else Train batches.

  Returns
    (np.ndarray, np.ndarray): Train/Test images, Train/Test classes.
  """
  if is_test:
    batch_size = ngo_cfg["test_batch_size"]
    _, _, imgs, clss = get_cifar_10_data()
  else:
    batch_size = ngo_cfg["train_batch_size"]
    imgs, clss, _, _ = get_cifar_10_data()

  num_instances = imgs.shape[0]
  imgs = imgs.reshape((num_instances, 1, -1))
  tiled_imgs = np.tile(imgs, (1, ngo_cfg["n_steps"], 1))

  for start in range(0, num_instances, batch_size):
    if start+batch_size > num_instances:
      continue
    yield (tiled_imgs[start:start+batch_size], clss[start:start+batch_size])
