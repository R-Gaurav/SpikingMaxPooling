#
# Author: Ramashish Gaurav
#

from utils.base_utils import log

import tensorflow as tf

def _get_2d_cnn_block(conv, layer_cfg, exp_cfg, layer_objs_lst):
  """
  Returns a conv block.

  Args:
    conv <tf.Tensor>: The convolution object.
    layer_cfg <namedtuple>: A named tuple of layer configuraion.
    exp_cfg <dict>: A dict of experimental configs.
    layer_objs_lst <[]>: A list of layer objects.

  Returns:
    tf.Tensor
  """
  log.INFO("Layer name: {}".format(conv.name))
  log.INFO("Layer config: {}".format(layer_cfg))
  conv = tf.keras.layers.Conv2D(
    layer_cfg.num_kernels, layer_cfg.kernel_dims, strides=layer_cfg.stride_dims,
    padding="same", data_format="channels_last", activation="relu",
    kernel_initializer="he_uniform",
    kernel_regularizer=tf.keras.regularizers.l2(exp_cfg["rf"]))(conv)

  layer_objs_lst.append(conv)
  return conv

def _get_dense_block(block, exp_cfg, layer_objs_lst, actvn="relu"):
  """
  Returns a dense block.

  Args:
    block <tf.Tensor>: A TF Tensor.
    exp_cfg <{}>: The experimental configuration dict.
    layer_objs_lst <[]>: The list of layer objects.
    actvn <str>: The activation function.

  Returns:
    tf.Tensor.
  """
  dense = tf.keras.layers.Dense(
    exp_cfg["nn_dlyr"], activation=actvn, kernel_initializer="he_uniform",
    kernel_regularizer=tf.keras.regularizers.l2(exp_cfg["rf"]))(block)
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

  # Construct the model's arch.
  x = inpt_lyr
  for _, layer in model["layers"].items(): # Dicts are ordered in Python3.7.
    x = _get_2d_cnn_block(x, layer, exp_cfg, layer_objs_lst)
  # Flatten
  x = tf.keras.layers.Flatten(data_format="channels_last")(x)
  x = _get_dense_block(x, exp_cfg, layer_objs_lst)
  x = _get_dense_block(x, exp_cfg, layer_objs_lst)

  exp_cfg["nn_dlyr"] = num_clss
  output_lyr = _get_dense_block(x, exp_cfg, layer_objs_lst, actvn="softmax")

  model = tf.keras.Model(inputs=inpt_lyr, outputs=output_lyr)
  return model, layer_objs_lst
