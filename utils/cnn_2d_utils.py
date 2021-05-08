#
# Author: Ramashish Gaurav
#

from utils.base_utils import log

import tensorflow as tf

def _get_2d_cnn_block(block, layer_cfg, layer_objs_lst):
  """
  Returns a conv block.

  Args:
    block <tf.Tensor>: A TF Tensor object.
    layer_cfg <namedtuple>: A named tuple of layer configuraion.
    layer_objs_lst <[]>: A list of layer objects.

  Returns:
    tf.Tensor
  """
  log.INFO("Layer name: {}".format(block.name))
  log.INFO("Layer config: {}".format(layer_cfg))
  conv = tf.keras.layers.Conv2D(
    layer_cfg.num_kernels, layer_cfg.kernel_dims, strides=layer_cfg.stride_dims,
    padding="valid", data_format=layer_cfg.data_format, activation="relu",
    kernel_initializer="he_uniform")(block)
  layer_objs_lst.append(conv)

  return conv

def _get_max_pool_block(block, layer_cfg, layer_objs_lst):
  """
  Returns a MaxPool block.

  Args:
    block <tf.Tensor>: A TF Tensor object.
    layer_cfg <namedtuple>: A named tuple of the layer configuration.
    layer_objs_lst <[]>: A list of layer objects.
  """
  log.INFO("Layer name: {}".format(block.name))
  log.INFO("Layer config: {}".format(layer_cfg))
  max_pool = tf.keras.layers.MaxPool2D(
      pool_size=layer_cfg.kernel_dims, padding="valid",
      data_format=layer_cfg.data_format)(block)
  layer_objs_lst.append(max_pool)

  return max_pool

def _get_dense_block(block, nn_dlyr, layer_objs_lst, actvn="relu"):
  """
  Returns a dense block.

  Args:
    block <tf.Tensor>: A TF Tensor.
    nn_dlyr <int>: Number of neurons in the dense layer.
    layer_objs_lst <[]>: The list of layer objects.
    actvn <str>: The activation function.

  Returns:
    tf.Tensor.
  """
  dense = tf.keras.layers.Dense(
      nn_dlyr, activation=actvn, kernel_initializer="he_uniform",)(block)
  layer_objs_lst.append(dense)

  return dense

def get_2d_cnn_model(inpt_shape, exp_cfg, num_clss=10):
  """
  Returns a 2D CNN model.

  Args:
    inpt_shape <()>: A tuple of (img_rows, img_cols, num_channels).
    exp_cfg <{}>: A dict of experimental config.
    num_clss <int>: The number of classes.

  Returns:
    training.Model
  """
  layer_objs_lst = []
  inpt_lyr = tf.keras.Input(shape=inpt_shape)
  layer_objs_lst.append(inpt_lyr)
  model = exp_cfg["tf_model"]

  ###################### Construct the model's arch. ########################
  # Add Conv and MaxPool blocks.
  x = inpt_lyr
  for _, layer in model["layers"].items(): # Dicts are ordered in Python3.7.
    if layer.name == "Conv":
      x = _get_2d_cnn_block(x, layer, layer_objs_lst)
    elif layer.name == "MaxPool":
      x = _get_max_pool_block(x, layer, layer_objs_lst)
  # Flatten
  # FIXME: Probable bug in Nengo-DL where data_format = "channels_last" in
  # Flatten layer results in garbage predictions.
  #x = tf.keras.layers.Flatten(data_format=layer.data_format)(x)
  x = tf.keras.layers.Flatten()(x)
  # Add one Dense block.
  x = _get_dense_block(x, exp_cfg["nn_dlyr"], layer_objs_lst)
  # Add the final output Dense block.
  output_lyr = _get_dense_block(x, num_clss, layer_objs_lst, actvn="softmax")

  model = tf.keras.Model(inputs=inpt_lyr, outputs=output_lyr)
  return model, layer_objs_lst
