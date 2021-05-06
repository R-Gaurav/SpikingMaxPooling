#
# Tests a Nengo-DL model with TF trained weights.
#
# Author: Ramashish Gaurav
#

import datetime
import nengo
import nengo_dl
import numpy as np
import random

import _init_paths

from configs.exp_configs import nengo_dl_cfg as ndl_cfg, tf_exp_cfg as tf_cfg
from utils.base_utils import log
from utils.base_utils.data_prep_utils import get_batches_of_exp_dataset
from utils.cnn_2d_utils import get_2d_cnn_model
from utils.consts.exp_consts import SEED, MNIST, CIFAR10
from utils.nengo_dl_utils import get_nengo_dl_model

# Set the SEED.
random.seed(SEED)
np.random.seed(SEED)

def _do_tensor_node_max_pooling(inpt_shape, num_clss):
  """
  Does Nengo-DL testing with TensorNode MaxPooling.

  Args:
    inpt_shape <(int, int, int)>: A tuple of Image shape with channels_first order.
    num_clss <int>: Number of test classes.
  """
  log.INFO("Getting the Nengo-DL model with loaded TF trained weights...")
  ndl_model, ngo_probes_lst = get_nengo_dl_model(
      inpt_shape, tf_cfg, ndl_cfg, mode="test", num_clss=num_clss)
  log.INFO("Getting the dataset: %s" % ndl_cfg["dataset"])
  test_batches = get_batches_of_exp_dataset(ndl_cfg, is_test=True)
  log.INFO("Start testing...")
  with nengo_dl.Simulator(
      ndl_model.net, minibatch_size=ndl_cfg["test_mode"]["test_batch_size"],
      progress_bar=False) as sim:
    acc, n_test_imgs = 0, 0
    for batch in test_batches:
      sim_data = sim.predict_on_batch({ngo_probes_lst[0]: batch[0]})
      for true_lbl, pred_lbl in zip(batch[1], sim_data[ngo_probes_lst[-1]]):
        if np.argmax(true_lbl) == np.argmax(pred_lbl[-1]):
          acc += 1
        n_test_imgs += 1
        # TODO: Collect the intermediate layers spike/synapsed output.

    log.INFO("Testing done. Writing TensorNode MP test accuracy results in log..")
    log.INFO("TensorNode MP Test Accuracy: %s" % (acc/n_test_imgs))
    log.INFO("*"*100)

def nengo_dl_test():
  """
  Loads a TF model weights and tests it in Nengo-DL. It first executes the
  MaxPooling op with TensorNode, then executes the `max_to_avg_pool` op, followed
  by the Associative-Max op.

  Args:
    tf_cfg <dict>: The TF experiment related config.
    ndl_cfg <dict>: The Nengo-DL experiment related config.
  """
  log.INFO("TF CONFIG: %s" % tf_cfg)
  log.INFO("NENGO DL CONFIG: %s" % ndl_cfg)
  assert ndl_cfg["dataset"] == tf_cfg["dataset"]

  ##############################################################################
  if ndl_cfg["dataset"] == MNIST:
    inpt_shape = (1, 28, 28)
    num_clss = 10
  ##############################################################################

  log.INFO("Testing in TensorNode MaxPooling mode...")
  _do_tensor_node_max_pooling(inpt_shape, num_clss)

if __name__ == "__main__":
  log.configure_log_handler(
      "%s_sfr_%s_n_steps_%s_synapse_%s_%s.log" % (
      ndl_cfg["test_mode"]["ndl_test_mode_res_otpt_dir"] + __file__,
      ndl_cfg["test_mode"]["sfr"], ndl_cfg["test_mode"]["n_steps"],
      ndl_cfg["test_mode"]["synapse"], datetime.datetime.now()))
  nengo_dl_test()
