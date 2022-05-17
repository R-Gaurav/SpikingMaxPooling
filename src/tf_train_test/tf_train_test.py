#
# This file executes TF code to train and test the models over datasets.
#
# Author: Ramashish Gaurav
#

import datetime
import numpy as np
import random
import tensorflow as tf

import _init_paths

from configs.exp_configs import tf_exp_cfg as tf_cfg
from utils.base_utils import log
from utils.base_utils.data_prep_utils import get_exp_dataset
from utils.cnn_2d_utils import get_2d_cnn_model
from utils.consts.exp_consts import SEED, MNIST, CIFAR10, FMNIST

# Set the SEED.
tf.random.set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

def tf_train_test():
  """
  Trains and Tests the TF code.
  """
  log.INFO("TF EXP CONFIG: %s" % tf_cfg)
  ##############################################################################
  log.INFO("Getting the dataset: %s" % tf_cfg["dataset"])
  train_x, train_y, test_x, test_y = get_exp_dataset(
      tf_cfg["dataset"], channels_first=tf_cfg["is_channels_first"],
      is_nengo_dl_train_test=True)
  log.INFO("Augmenting the dataset: %s" % tf_cfg["dataset"])

  if tf_cfg["dataset"] == MNIST or tf_cfg["dataset"] == FMNIST:
    inpt_shape = (1, 28, 28) if tf_cfg["is_channels_first"] else (28, 28, 1)
    num_clss = 10
  elif tf_cfg["dataset"] == CIFAR10:
    inpt_shape = (3, 32, 32) if tf_cfg["is_channels_first"] else (32, 32, 3)
    num_clss = 10
  ##############################################################################

  tf_model, tf_lyr_obs = get_2d_cnn_model(inpt_shape, tf_cfg, num_clss=num_clss)
  log.INFO("Writing tf_model.summary() to file tf_model_summary.txt")
  with open(tf_cfg["tf_res_otpt_dir"]+"/tf_model_summary.txt", "w") as f:
    tf_model.summary(print_fn=lambda x: f.write(x + "\n"))

  log.INFO("Compiling and training the model...")
  tf_model.compile(
      optimizer=tf.optimizers.Adam(tf_cfg["lr"], decay=1e-4),
      loss=tf.losses.CategoricalCrossentropy(from_logits=True),
      metrics=[tf.metrics.categorical_accuracy])
  tf_model.fit(train_x, train_y,
               batch_size=tf_cfg["batch_size"], epochs=tf_cfg["epochs"])
  log.INFO("Training done. Saving the model weights...")
  tf_model.save_weights(tf_cfg["tf_wts_otpt_dir"]+"/weights")

  log.INFO("Saving weights done. Now testing/evaluating the model...")
  loss, acc = tf_model.evaluate(test_x, test_y)
  log.INFO("Model: %s performance loss: %s accuracy: %s"
            % (tf_cfg["tf_model"]["name"], loss, acc))

if __name__ == "__main__":
  log.configure_log_handler(
      "%s_%s.log" % (
      tf_cfg["tf_res_otpt_dir"] + __file__, datetime.datetime.now()))
  tf_train_test()
