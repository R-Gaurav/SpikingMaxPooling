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

from configs.exp_configs import (
    nengo_dl_cfg as ndl_cfg, tf_exp_cfg as tf_cfg, asctv_max_cfg as am_cfg)
from utils.base_utils import log
from utils.base_utils.data_prep_utils import get_batches_of_exp_dataset
from utils.base_utils.exp_utils import get_grouped_slices_2d_pooling
from utils.cnn_2d_utils import get_2d_cnn_model
from utils.consts.exp_consts import SEED, MNIST, CIFAR10
from utils.nengo_dl_utils import get_nengo_dl_model, get_max_pool_global_net

# Set the SEED.
random.seed(SEED)
np.random.seed(SEED)

def _do_nengo_dl_max_or_max_to_avg(inpt_shape, num_clss, max_to_avg_pool=False):
  """
  Does Nengo-DL testing with TensorNode MaxPooling or does max_to_avg_pooling
  if `max_to_avg_pool`=True.

  Args:
    inpt_shape <(int, int, int)>: A tuple of Image shape with channels_first order.
    num_clss <int>: Number of test classes.
    max_to_avg_pool <bool>: Do max_to_avg_pooling if True, else do MaxPooling.
  """
  log.INFO("Getting the Nengo-DL model with loaded TF trained weights...")
  ndl_model, ngo_probes_lst = get_nengo_dl_model(
      inpt_shape, tf_cfg, ndl_cfg, mode="test", num_clss=num_clss,
      max_to_avg_pool=max_to_avg_pool)
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

    log.INFO("Testing done! Writing max_to_avg_pool: %s test accuracy results "
             "in log..." % max_to_avg_pool)
    log.INFO("Nengo-DL Test Accuracy: %s" % (acc/n_test_imgs))
    #TODO: Delete the `ndl_model` to reclaim GPU memory.
    log.INFO("*"*100)

def _do_custom_associative_max_or_avg(inpt_shape, num_clss, do_max=True):
  """
  Does the custom associative max or associative avg depending on the value of
  `do_max`.

  Args:
    inpt_shape <(int, int, int)>: A tuple of Image shape with channels_first order.
    num_clss <int>: Number of test classes.
    do_max <bool>: Do associative max pooling if True, else do associative avg.
  """
  log.INFO("Getting the Nengo-DL model with loaded TF trained weights...")
  ndl_model, ngo_probes_lst = get_nengo_dl_model(
      inpt_shape, tf_cfg, ndl_cfg, mode="test", num_clss=num_clss,
      max_to_avg_pool=False)
  log.INFO("Getting the dataset: %s" % ndl_cfg["dataset"])
  test_batches = get_batches_of_exp_dataset(ndl_cfg, is_test=True)

  def _get_conv_layer_output_shape(layer_name):
    for layer in ndl_model.model.layers:
      if layer.name == layer_name.split(".")[0]:
        return layer.output.shape[1:] # (num_chnls, rows, cols)

  # Replace the MaxPool TensorNode with the custom layer of associative max nets.
  log.INFO("Replacing the TensorNode with custom layer of associative max nets.")
  with ndl_model.net:
    for i, conn in enumerate(ndl_model.net.all_connections):
      if (isinstance(conn.post_obj, nengo_dl.tensor_node.TensorNode) and
          conn.post_obj.label.startswith("max_pooling")):
        ############# GET THE CONNECTIONS ####################
        conn_from_conv_to_max = ndl_model.net.all_connections[i]
        # In Nengo-DL 3.4.0, the paired connections are at 3rd index difference.
        # Therefore cross check in the logs that you are dealing with correct
        # paired connections.
        conn_from_max_to_conv = ndl_model.net.all_connections[i+3]
        log.INFO("Found connection from prev conv to max pool: {} \n It's "
                 "transform parameter: {}, It's function parameter: {}, "
                 "It's synapse parameter: {}".format(conn_from_conv_to_max,
                 conn_from_conv_to_max.transform, conn_from_conv_to_max.function,
                 conn_from_conv_to_max.synapse))
        log.INFO("Found connection from max pool to next conv: {} \n It's "
                 "transform paramter: {}, It's function parameter: {}, "
                 "It's synapse parameter: {}".format(conn_from_max_to_conv,
                 conn_from_max_to_conv.transform, conn_from_max_to_conv.function,
                 conn_from_max_to_conv.synapse))

        ########## GET THE CONV LAYER OUTPUT SHAPE FOR MAX POOLING ###########
        conv_label = conn.pre_obj.ensemble.label
        (num_chnls, rows, cols) = _get_conv_layer_output_shape(conv_label)
        grouped_slices = get_grouped_slices_2d_pooling(
            pool_size=(2, 2), num_chnls=num_chnls, rows=rows, cols=cols)
        log.INFO("Grouped slices of prev Conv layer for MaxPooling obtained.")
        # Create the MaxPool layer of multiple smaller 2x2 sub-network.
        max_pool_layer = get_max_pool_global_net(
            num_chnls * row * cols, seed=SEED,
            max_rate=am_cfg[conv_label]["max_rate"],
            radius=am_cfg[conv_label]["radius"], sf=am_cfg[conv_label]["sf"],
            do_max=do_max, synapse=am_cfg[conv_label]["synapse"])
        log.INFO("Associative-Max MaxPool layer obtained.")

        ######### CONNECT THE PREV ENS/CONV TO ASSOCIATIVE-MAX MAXPOOL ########
        nengo.Connection(
            conn_from_conv_to_max.pre_obj[grouped_slices],
            max_pool_layer.input,
            transform=conn_from_conv_to_max.transform,
            synapse=conn_from_conv_to_max.synapse,
            function=conn_from_conv_to_max.function
        )
        ######### CONNECT THE ASSOCIATIVE-MAX MAXPOOL TO NEXT ENS/CONV ########
        nengo.Connection(
            max_pool_layer.output,
            conn_from_max_to_conv.post_obj,
            transform=conn_from_max_to_conv.transform,
            synapse=conn_from_max_to_conv.synapse,
            function=conn_from_max_to_conv.function
        )
        ########### REMOVE THE OLD CONNECTIONS AND TENSORNODES ###############
        ndl_model.net._connections.remove(conn_from_conv_to_max)
        ndl_model.net._connections.remove(conn_from_max_to_conv)
        ndl_model.net._nodes.remove(conn_from_conv_to_max.post_obj)

  log.INFO("All connections made, now checking the new connections (in log)...")
  for conn in ndl_model.net.all_connections:
    log.INFO("Connection: {} | Synapse: {}".format(conn, conn.synapse))

  log.INFO("Start testing...")
  with nengo_dl.Simulator(
      ndl_model.net, minibatch_size=ndl_cfg["test_mode"]["test_batch_size"],
      progress_bar=False) as sim:
    log.INFO("Nengo-DL model with associative-max max pooling layer compiled.")
    acc, n_test_imgs = 0, 0
    for batch in test_batches:
      sim_data = sim.predict_on_batch({ngo_probes_lst[0]: batch[0]})
      for true_lbl, pred_lbl in zip(batch[1], sim_data[ngo_probes_lst[-1]]):
        if np.argmax(true_lbl) == np.argmax(pred_lbl[-1]):
          acc += 1
        n_test_imgs += 1
        # TODO: Collect the intermediate layers spike/synapsed output.

    log.INFO("Testing done! Writing associative-max max pooling - do_max: %s "
             "test accuracy results in log..." % do_max)
    log.INFO("Nengo DL Test Accuracy: %s" % (acc/n_test_imgs))
    # TODO: Delete the `ndl_model` to reclaim GPU memory.
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
  log.INFO("ASSOCIATIVE MAX CONFIG: %s" % am_cfg)
  assert ndl_cfg["dataset"] == tf_cfg["dataset"]

  ##############################################################################
  if ndl_cfg["dataset"] == MNIST:
    inpt_shape = (1, 28, 28)
    num_clss = 10
  ##############################################################################

  log.INFO("*"*100)
  log.INFO("Testing in TensorNode MaxPooling mode...")
  _do_nengo_dl_max_or_max_to_avg(inpt_shape, num_clss, max_to_avg_pool=False)

  log.INFO("Testing in Max To Avg Pooling mode...")
  _do_nengo_dl_max_or_max_to_avg(inpt_shape, num_clss, max_to_avg_pool=True)

  #log.INFO("Testing in custom associative max mode...")
  #_do_custom_associative_max_or_avg(inpt_shape, num_clss, do_max=True)

  #log.INFO("Testing in custom associative avg mode...")
  #_do_custom_associative_max_or_avg(inpt_shape, num_clss, do_max=False)

if __name__ == "__main__":
  log.configure_log_handler(
      "%s_sfr_%s_n_steps_%s_synapse_%s_%s.log" % (
      ndl_cfg["test_mode"]["ndl_test_mode_res_otpt_dir"] + __file__,
      ndl_cfg["test_mode"]["sfr"], ndl_cfg["test_mode"]["n_steps"],
      ndl_cfg["test_mode"]["synapse"], datetime.datetime.now()))
  nengo_dl_test()
