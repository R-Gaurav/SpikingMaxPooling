#
# Author: Ramashish Gaurav
#
import numpy as np
import tensorflow as tf

from . import log

import _init_paths

from utils.consts.exp_consts import MNIST, CIFAR10, SEED
from utils.base_utils.exp_utils import get_shuffled_lists_in_unison

def get_exp_dataset(dataset, channels_first=True, start_idx=None, end_idx=None):
  """
  Returns MNIST data with first dimension as the channel dimension if
  `channels_first` = True.

  Args:
    dataset <str>: One of MNIST | CIFAR10. Both are in range 0 to 255.
    channels_first <bool>: Make the first dimension as channel dimension if True.
    start_idx <int>: The start index (inclusive) of the test dataset.
    end_idx <int>: The end index (exclusive) of the test dataset

  Returns:
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
  """
  if dataset == CIFAR10:
    log.INFO("Getting CIFAR10 datasset...")
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
  elif dataset == MNIST:
    log.INFO("Getting MNIST dataset...")
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    train_x, test_x = np.expand_dims(train_x, -1), np.expand_dims(test_x, -1)
    train_y, test_y = np.expand_dims(train_y, -1), np.expand_dims(test_y, -1)

  # TODO: Probably standardize the images?
  # Normalize the images in range [-1, 1]
  train_x = train_x.astype(np.float32) / 127.5 - 1
  test_x = test_x.astype(np.float32) / 127.5 - 1


  # Default image data format is "channels_last", change the image data to
  # "channels_first" if required.
  if channels_first:
    train_x, test_x = np.moveaxis(train_x, -1, 1), np.moveaxis(test_x, -1, 1)
  log.INFO("{0} data with shape (train_x) : {1}, (test_x): {2}".format(dataset,
           train_x.shape, test_x.shape))
  # Binarize the labels.
  train_y = np.eye(10, dtype=np.float32)[train_y].squeeze(axis=1)
  test_y = np.eye(10, dtype=np.float32)[test_y].squeeze(axis=1)

  if start_idx and end_idx:
    log.INFO("Partial test data returned. Start Index: %s, End Index: %s"
             % (start_idx, end_idx))
    return train_x, train_y, test_x[start_idx:end_idx], test_y[start_idx:end_idx]
  else:
    return train_x, train_y, test_x, test_y

def get_batches_of_exp_dataset(ndl_cfg, is_test=True, channels_first=True):
  """
  Returns the batches of training or test data.

  Args:
    ndl_cfg <dict>: The Nengo-DL related configuration.
    is_test <True>: Returns the test batches if True else Train batches.

  Returns
    (np.ndarray, np.ndarray): Train/Test images, Train/Test classes.
  """
  log.INFO("Getting Nengo-DL data for dataset: %s" % ndl_cfg["dataset"])
  if is_test:
    batch_size = ndl_cfg["test_mode"]["test_batch_size"]
    _, _, imgs, clss = get_exp_dataset(
        ndl_cfg["dataset"], channels_first=channels_first)
    num_instances = imgs.shape[0]
    imgs = imgs.reshape((num_instances, 1, -1))
    tiled_imgs = np.tile(imgs, (1, ndl_cfg["test_mode"]["n_steps"], 1))
    for start in range(0, num_instances, batch_size):
      if start+batch_size > num_instances:
        continue
      yield (tiled_imgs[start:start+batch_size], clss[start:start+batch_size])

  else:
    assert ndl_cfg["train_mode"]["n_steps"] == 1
    batch_size = ndl_cfg["train_mode"]["train_batch_size"]
    imgs, clss, _, _ = get_exp_dataset(
        ndl_cfg["dataset"], channels_first=channels_first)
    clss = clss.reshape((clss.shape[0], 1, -1))
    num_instances = imgs.shape[0]
    train_idg = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.1, height_shift_range=0.1, rotation_range=20,
        horizontal_flip=True, data_format="channels_first")
    train_idg.fit(imgs, seed=SEED)

    # nodes = ndl_model.net.all_nodes
    for start in range(0, num_instances, batch_size):
      if start+batch_size > num_instances:
        continue

      batch_imgs = imgs[start:start+batch_size]
      batch_clss = clss[start:start+batch_size]
      batch_x = np.zeros(batch_imgs.shape)
      batch_y = np.zeros(batch_clss.shape)
      batch_imgs, batch_clss = get_shuffled_lists_in_unison(batch_imgs, batch_clss)

      for i, img in enumerate(batch_imgs):
        #params = train_idg.get_random_transform(img.shape)
        #batch_x[i] = train_idg.standardize(train_idg.apply_transform(img, params))
        #batch_x[i] = train_idg.apply_transform(img, params)
        batch_x[i] = img
        batch_y[i] = batch_clss[i]

      # Flatten the `batch_x` images.
      batch_x = batch_x.reshape((batch_x.shape[0], 1, -1))
      tiled_imgs = np.tile(batch_x, (1, ndl_cfg["train_mode"]["n_steps"], 1))
      input_dict = {
        #"input_1": imgs[start:start+batch_size],
        "input_1": tiled_imgs,
        "n_steps": np.ones((batch_size, 1), dtype=np.int32),
        # Start from "conv2d" layer till the output "dense_2" layer.
        #nodes[1].label + ".0.bias": np.ones((batch_size, 32, 1), dtype=np.int32),
        #"conv2d.0.bias": np.ones((batch_size, 32, 1), dtype=np.int32),
        #nodes[2].label + ".0.bias": np.ones((batch_size, 64, 1), dtype=np.int32),
        #"conv2d_1.0.bias": np.ones((batch_size, 64, 1), dtype=np.int32),
        #nodes[3].label + ".0.bias": np.ones((batch_size, 64, 1), dtype=np.int32),
        #"conv2d_2.0.bias": np.ones((batch_size, 64, 1), dtype=np.int32),
        #nodes[4].label + ".0.bias": np.ones((batch_size, 96, 1), dtype=np.int32),
        #"conv2d_3.0.bias": np.ones((batch_size, 96, 1), dtype=np.int32),
        #nodes[5].label + ".0.bias": np.ones((batch_size, 128, 1), dtype=np.int32),
        #"conv2d_4.0.bias": np.ones((batch_size, 128, 1), dtype=np.int32),
        #nodes[6].label + ".0.bias": np.ones((batch_size, 10, 1), dtype=np.int32),
        #########################################################################
        # When "kernel_regularizer" is not included, "dense.0.bias" and
        # "dense_1.0.bias" does not appear in `converter.net.all_nodes`. When
        # the two dense layers do not have "relu" neurons (or `tf.nn.relu`), they
        # appear. Why so?
        # Next, in the case of "relu" neurons (or tf.nn.relu) present in Dense
        # layers, as "dense.0.bias" and "dense_1.0.bias" do not appear, having
        # them or not in the `input_dict` does not matter while training/test.
        #########################################################################
        #"dense.0.bias": np.ones((batch_size, 10, 1), dtype=np.int32),
        #nodes[7].label + ".0.bias": np.ones((batch_size, 10, 1), dtype=np.int32),
        "dense_1.0.bias": np.ones((batch_size, 10, 1), dtype=np.int32),
        #nodes[8].label + ".0.bias": np.ones((batch_size, 10, 1), dtype=np.int32)
        #"dense_2.0.bias": np.ones((batch_size, 10, 1), dtype=np.int32),
      }
      output_dict = {
        # ndl_mdl_probes[-1]: clss[start:start+batch_size], # Doesn't work. Why?
        "probe": batch_y, # Output or "probe".
      }
      # Include the rest probes to regularize their firing rates.
      #for i, probe in enumerate(ndl_mdl_probes[1:-1]):
      #  output_dict["probe_%s" % (i+1)] = np.ones(
      #      (batch_size, 1, probe.size_in), dtype=np.int32)

      yield (input_dict, output_dict)
