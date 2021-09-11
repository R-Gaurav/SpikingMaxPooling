#
# Tests a Nengo-DL model with TF trained weights.
#
# Author: Ramashish Gaurav
#

import argparse
import copy
import datetime
import nengo
import nengo_dl
import numpy as np
import random

import _init_paths

#from nengo_dl.graph_optimizer import noop_planner
from collections import defaultdict

from configs.exp_configs import (
    nengo_dl_cfg as ndl_cfg, tf_exp_cfg as tf_cfg) #, asctv_max_cfg as am_cfg)
from utils.base_utils import log
from utils.base_utils.data_prep_utils import get_batches_of_exp_dataset
from utils.base_utils.exp_utils import (get_grouped_slices_2d_pooling_cf,
    get_grouped_slices_2d_pooling_cl, get_isi_based_max_pooling_params)
from utils.consts.exp_consts import SEED, MNIST, CIFAR10, FMNIST
from utils.nengo_dl_utils import (get_nengo_dl_model, get_max_pool_global_net,
                                  get_isi_based_maximally_spiking_mask)

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
      max_to_avg_pool=max_to_avg_pool, load_tf_trained_wts=ndl_cfg["load_tf_wts"],
      include_layer_probes=True)
  log.INFO("Getting the dataset: %s" % ndl_cfg["dataset"])
  test_batches = get_batches_of_exp_dataset(
      ndl_cfg, is_test=True, channels_first=tf_cfg["is_channels_first"],
      is_nengo_dl_train_test=not ndl_cfg["load_tf_wts"])
  log.INFO("Start testing...")
  with nengo_dl.Simulator(
      ndl_model.net, minibatch_size=ndl_cfg["test_mode"]["test_batch_size"],
      progress_bar=False) as sim:
    if not ndl_cfg["load_tf_wts"]:
      sim.load_params(ndl_cfg["trained_model_params"]+ "/ndl_trained_params")

    acc, n_test_imgs = 0, 0
    layer_probes_otpt = defaultdict(list)
    for batch in test_batches:
      log.INFO("RG: N Test Images Done: %s" % n_test_imgs)
      sim_data = sim.predict_on_batch({ngo_probes_lst[0]: batch[0]})
      # Collect the intermediate layers spike/synapsed output.
      for probe in ngo_probes_lst[1:-1]:
        layer_probes_otpt[probe.obj.ensemble.label].extend(sim_data[probe])
      for true_lbl, pred_lbl in zip(batch[1], sim_data[ngo_probes_lst[-1]]):
        if np.argmax(true_lbl) == np.argmax(pred_lbl[-1]):
          acc += 1
        n_test_imgs += 1
      if n_test_imgs == 100: # For quick check, comment if run for entire data.
        break

    log.INFO("Testing done! Writing max_to_avg_pool: %s test accuracy results "
             "in log..." % max_to_avg_pool)
    log.INFO("Nengo-DL Test Accuracy: %s" % (acc/n_test_imgs))
    if layer_probes_otpt:
      log.INFO("Saving intermediate probes output...")
      np.save(ndl_cfg["test_mode"]["test_mode_res_otpt_dir"]+
              "/layer_probes_otpt.npy", layer_probes_otpt)
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
  log.INFO("Testing in custom associative max mode...")
  log.INFO("Getting the Nengo-DL model with loaded TF trained weights...")
  ndl_model, ngo_probes_lst = get_nengo_dl_model(
      inpt_shape, tf_cfg, ndl_cfg, mode="test", num_clss=num_clss,
      max_to_avg_pool=False, load_tf_trained_wts=ndl_cfg["load_tf_wts"])
  with nengo_dl.Simulator(ndl_model.net, seed=SEED) as ndl_sim:
    ndl_sim.load_params(ndl_cfg["trained_model_params"]+ "/ndl_trained_params")
                        #"/attempting_TN_MP_loihineurons_8_16")
    ndl_sim.freeze_params(ndl_model.net)
  log.INFO("Getting the dataset: %s" % ndl_cfg["dataset"])
  test_batches = get_batches_of_exp_dataset(
      ndl_cfg, is_test=True, channels_first=tf_cfg["is_channels_first"],
      is_nengo_dl_train_test=not ndl_cfg["load_tf_wts"])

  def _get_conv_layer_output_shape(layer_name):
    for layer in ndl_model.model.layers:
      if layer.name == layer_name.split(".")[0]:
        return layer.output.shape[1:] # (num_chnls, rows, cols)

  # Replace the MaxPool TensorNode with the custom layer of associative max nets.
  log.INFO("Replacing all the MaxPooling TensorNodes with custom layer of "
           "associative max nets. Getting the respective connections...")
  ################# GET THE CONNECTIONS TO REPLACE ####################
  all_mp_tn_conns = [] # Stores all the to/from MaxPool TensorNode connections.
  for i, conn in enumerate(ndl_model.net.all_connections):
    if (isinstance(conn.post_obj, nengo_dl.tensor_node.TensorNode) and
        conn.post_obj.label.startswith("max_pooling")):
      # In Nengo-DL 3.4.0, the paired connections are at 3rd index difference.
      # Therefore cross check in the logs that you are dealing with correct
      # paired connections.
      #
      # Connection from previous Conv to current Max TensorNode.
      conn_from_pconv_to_max = ndl_model.net.all_connections[i]
      # Connection from current Max TensorNode to next Conv.
      conn_from_max_to_nconv = ndl_model.net.all_connections[i+1]
      log.INFO("Found connection from prev conv to max pool: {} with transform: "
               "{}, function: {}, and synapse: {}".format(conn_from_pconv_to_max,
               conn_from_pconv_to_max.transform, conn_from_pconv_to_max.function,
               conn_from_pconv_to_max.synapse))
      log.INFO("Found connection from max pool to next conv: {} with transform: "
               "{}, function: {}, and synapse: {}".format(conn_from_max_to_nconv,
               conn_from_max_to_nconv.transform, conn_from_max_to_nconv.function,
               conn_from_max_to_nconv.synapse))
      all_mp_tn_conns.append(
          (conn_from_pconv_to_max, conn_from_max_to_nconv))

  log.INFO("Connections to be replaced: {}".format(all_mp_tn_conns))

  ##################### REPLACE THE CONNECTIONS #######################
  with ndl_model.net:
    # Disable operator merging to improve compilation time.
    # nengo_dl.configure_settings(planner=noop_planner)

    for conn_tpl in all_mp_tn_conns:
      conn_from_pconv_to_max, conn_from_max_to_nconv = conn_tpl
      ########## GET THE CONV LAYER GROUPED SLICES FOR MAX POOLING ###########
      conv_label = conn_from_pconv_to_max.pre_obj.ensemble.label

      if tf_cfg["is_channels_first"]:
        (num_chnls, rows, cols) = _get_conv_layer_output_shape(conv_label)
        grouped_slices = get_grouped_slices_2d_pooling_cf(
            pool_size=(2, 2), num_chnls=num_chnls, rows=rows, cols=cols)
        output_idcs = np.arange(num_chnls * (rows//2) * (cols//2))
      else:
        (rows, cols, num_chnls) = _get_conv_layer_output_shape(conv_label)
        grouped_slices = get_grouped_slices_2d_pooling_cl(
            pool_size=(2, 2), num_chnls=num_chnls, rows=rows, cols=cols)
        output_idcs = np.arange(num_chnls * (rows//2) * (cols//2)).reshape(
            num_chnls, rows//2, cols//2)
        output_idcs = np.moveaxis(output_idcs, 0, -1)
        output_idcs = output_idcs.flatten()

      log.INFO("Grouped slices of Conv: %s for MaxPooling obtained." % conv_label)
      if rows % 2 and cols % 2:
        rows, cols = rows-1, cols-1
      num_neurons = num_chnls * rows * cols

      # Create the MaxPool layer of multiple smaller 2x2 sub-network.
      max_pool_layer = get_max_pool_global_net(
          (num_chnls, rows, cols), seed=SEED,
          max_rate=250, #am_cfg[conv_label]["max_rate"],
          #radius=3, #am_cfg[conv_label]["radius"],
          radius=ndl_cfg["test_mode"]["radius"], #am_cfg[conv_label]["radius"],
          #sf=1.2, #am_cfg[conv_label]["sf"],
          sf=1, #am_cfg[conv_label]["sf"],
          synapse=0.001, #am_cfg[conv_label]["synapse"],
          do_max=do_max
          )
      log.INFO("Associative-Max MaxPool layer obtained.")

      ######### CONNECT THE PREV ENS/CONV TO ASSOCIATIVE-MAX MAXPOOL ########
      # Set transform=None to avoid following error:
      # "nengo.exceptions.ValidationError: Connection.transform: Transform input
      # size (2904) not equal to Neurons output size (2400)" in case of odd rows/cols.
      nengo.Connection(
          conn_from_pconv_to_max.pre_obj[grouped_slices[:num_neurons]],
          max_pool_layer.input,
          transform=None, #conn_from_pconv_to_max.transform, # NoTransform
          synapse=conn_from_pconv_to_max.synapse,
          function=conn_from_pconv_to_max.function
      )
      ######### CONNECT THE ASSOCIATIVE-MAX MAXPOOL TO NEXT ENS/CONV ########
      nengo.Connection(
          max_pool_layer.output[output_idcs],
          conn_from_max_to_nconv.post_obj,
          transform=conn_from_max_to_nconv.transform,
          synapse=conn_from_max_to_nconv.synapse,
          function=conn_from_max_to_nconv.function
      )

      log.INFO("To/From connection w.r.t. %s done."
               % conn_from_pconv_to_max.post_obj.label)

  ########### REMOVE THE OLD CONNECTIONS AND TENSOR-NODES ###############
  with ndl_model.net:
    # Disable operator merging to improve compilation time.
    # nengo_dl.configure_settings(planner=noop_planner)
    for conn_tpl in all_mp_tn_conns:
      conn_from_pconv_to_max, conn_from_max_to_nconv = conn_tpl
      ndl_model.net._connections.remove(conn_from_pconv_to_max)
      ndl_model.net._connections.remove(conn_from_max_to_nconv)
      ndl_model.net._nodes.remove(conn_from_pconv_to_max.post_obj)

  log.INFO("All connections made, now checking the new connections (in log)...")
  for conn in ndl_model.net._connections:
    log.INFO("Connection: {} | Synapse: {}".format(conn, conn.synapse))

  ndl_sim_preds = []
  log.INFO("Start testing...")
  with nengo_dl.Simulator(
      ndl_model.net, minibatch_size=ndl_cfg["test_mode"]["test_batch_size"],
      progress_bar=False) as sim:
    log.INFO("Nengo-DL model with associative-max max pooling layer compiled.")
    acc, n_test_imgs = 0, 0
    for batch in test_batches:
      sim_data = sim.predict_on_batch({ngo_probes_lst[0]: batch[0]})
      for true_lbl, pred_lbl in zip(batch[1], sim_data[ngo_probes_lst[-1]]):
        ndl_sim_preds.append(pred_lbl)
        if np.argmax(true_lbl) == np.argmax(pred_lbl[-1]):
          acc += 1
        n_test_imgs += 1
        # TODO: Collect the intermediate layers spike/synapsed output.

    log.INFO("Testing done! Writing associative-max max pooling - do_max: %s "
             "test accuracy results in log..." % do_max)
    log.INFO("Nengo DL Test Accuracy: %s" % (acc/n_test_imgs))
    log.INFO("Saving the AVAM method predictions...")
    np.save((ndl_cfg["test_mode"]["test_mode_res_otpt_dir"]+
             "avam_do_max_%s_radius_%s.npy" % (
             "true" if do_max==True else "false",
             "_".join(str(ndl_cfg["test_mode"]["radius"]).split(".")))),
             np.array(ndl_sim_preds))
    # TODO: Delete the `ndl_model` to reclaim GPU memory.
    log.INFO("*"*100)

def _do_isi_based_max_pooling(inpt_shape, num_clss):
  """
  Does ISI based MaxPooling.

  Args:
    inpt_shape <(int, int, int)>: A tuple of Image shape with channels_first order.
    num_clss <int>: Number of test classes.
  """
  log.INFO("Getting the Nengo-DL model with loaded TF trained weights...")
  ndl_model, ngo_probes_lst = get_nengo_dl_model(
      inpt_shape, tf_cfg, ndl_cfg, mode="test", num_clss=num_clss,
      max_to_avg_pool=False, load_tf_trained_wts=ndl_cfg["load_tf_wts"])
  log.INFO("Getting the dataset: %s" % ndl_cfg["dataset"])
  test_batches = get_batches_of_exp_dataset(
      ndl_cfg, is_test=True, is_nengo_dl_train_test=not ndl_cfg["load_tf_wts"])

  # Populate ISI based MaxPooling Parameters.
  get_isi_based_max_pooling_params(ndl_model.model.layers)

  # Replace the MaxPool TensorNode with the ISI based MaxPooling Node.
  log.INFO("Replacing all the MaxPooling TensorNodes with custom Node for doing "
           "ISI based MaxPooling. Getting the respective connections...")
  ################# GET THE CONNECTIONS TO REPLACE ####################
  all_mp_tn_conns = [] # Stores all the to/from MaxPool TensorNode connections.
  for i, conn in enumerate(ndl_model.net.all_connections):
    if (isinstance(conn.post_obj, nengo_dl.tensor_node.TensorNode) and
        conn.post_obj.label.startswith("max_pooling")):
      # Connection from previous Conv to current Max TensorNode.
      conn_from_pconv_to_max = ndl_model.net.all_connections[i]
      # Connection from current Max TensorNode to next Conv.
      conn_from_max_to_nconv = ndl_model.net.all_connections[i+1]
      log.INFO("Found connection from prev conv to max pool: {} with transform: "
               "{}, function: {}, and synapse: {}".format(conn_from_pconv_to_max,
               conn_from_pconv_to_max.transform, conn_from_pconv_to_max.function,
               conn_from_pconv_to_max.synapse))
      # Set the connection synapse from the Conv to MaxPooling layers to None to
      # receive spikes for ISI based MaxPooling.
      conn_from_pconv_to_max.synapse = None
      log.INFO("Connection: {}, | and after setting synapse to None: {}".format(
          conn_from_pconv_to_max, conn_from_pconv_to_max.synapse))

      log.INFO("Found connection from max pool to next conv: {} with transform: "
               "{}, function: {}, and synapse: {}".format(conn_from_max_to_nconv,
               conn_from_max_to_nconv.transform, conn_from_max_to_nconv.function,
               conn_from_max_to_nconv.synapse))
      # Since the incoming spikes to the MaxPooling layers were not synpased, we
      # need to synapse the selected outgoing spikes from the MaxPooling layer to
      # the next Conv layer.
      conn_from_max_to_nconv.synapse = nengo.Lowpass(ndl_cfg["test_mode"]["synapse"])
      log.INFO("Connection: {}, and after setting Lowpass synapse: {}".format(
          conn_from_max_to_nconv, conn_from_max_to_nconv.synapse))

      all_mp_tn_conns.append(
          (conn_from_pconv_to_max, conn_from_max_to_nconv))

  log.INFO("Connections to be replaced: {}".format(all_mp_tn_conns))

  ##################### REPLACE THE CONNECTIONS #######################
  with ndl_model.net:
    for conn_tpl in all_mp_tn_conns:
      conn_from_pconv_to_max, conn_from_max_to_nconv = conn_tpl
      log.INFO("RG: Size In: %s of the Max TN: %s" % (
                conn_from_pconv_to_max.post_obj.size_in,
                conn_from_pconv_to_max.post_obj.label))
      isi_max_node = nengo.Node(
          output=get_isi_based_maximally_spiking_mask,
          size_in=conn_from_pconv_to_max.post_obj.size_in)

      ######### CONNECT THE PREV ENS/CONV TO ASSOCIATIVE-MAX MAXPOOL ########
      nengo.Connection(
          conn_from_pconv_to_max.pre_obj,
          isi_max_node,
          transform=conn_from_pconv_to_max.transform,
          synapse=conn_from_pconv_to_max.synapse,
          function=conn_from_pconv_to_max.function
      )

      ######### CONNECT THE ASSOCIATIVE-MAX MAXPOOL TO NEXT ENS/CONV ########
      nengo.Connection(
          isi_max_node,
          conn_from_max_to_nconv.post_obj,
          transform=conn_from_max_to_nconv.transform,
          synapse=conn_from_max_to_nconv.synapse,
          function=conn_from_max_to_nconv.function
      )
      log.INFO("Replacement of to/from connection w.r.t. %s done."
               % conn_from_pconv_to_max.post_obj.label)

  ########### REMOVE THE OLD CONNECTIONS AND TENSOR-NODES ###############
  with ndl_model.net:
    for conn_tpl in all_mp_tn_conns:
      conn_from_pconv_to_max, conn_from_max_to_nconv = conn_tpl
      ndl_model.net._connections.remove(conn_from_pconv_to_max)
      ndl_model.net._connections.remove(conn_from_max_to_nconv)
      ndl_model.net._nodes.remove(conn_from_pconv_to_max.post_obj)

  log.INFO("All connections made, now checking the new connections (in log)...")
  for conn in ndl_model.net._connections:
    log.INFO("Connection: {} | Synapse: {}".format(conn, conn.synapse))

  log.INFO("Start testing...")
  with nengo_dl.Simulator(
      ndl_model.net, minibatch_size=ndl_cfg["test_mode"]["test_batch_size"]
      ) as sim:
    acc, n_test_imgs = 0, 0
    for batch in test_batches:
      sim_data = sim.predict_on_batch({ngo_probes_lst[0]: batch[0]})
      for true_lbl, pred_lbl in zip(batch[1], sim_data[ngo_probes_lst[-1]]):
        if np.argmax(true_lbl) == np.argmax(pred_lbl[-1]):
          acc += 1
        n_test_imgs += 1
      if n_test_imgs == 200:
        break
  log.INFO("Testing done! Writing the ISI based MaxPooling test accuracy results "
           "in log...")
  log.INFO("Nengo DL Test Accuracy: %s" % (acc/n_test_imgs))
  # TODO: Delete the `ndl_model` to reclaim GPU memory.
  log.INFO("*"*100)

def nengo_dl_test():
  """
  Loads a TF model weights and tests it in Nengo-DL. It first executes the
  MaxPooling op with TensorNode, then executes the `max_to_avg_pool` op, followed
  by the Associative-Max op.
  """
  log.INFO("TF CONFIG: %s" % tf_cfg)
  log.INFO("NENGO DL CONFIG: %s" % ndl_cfg)
  #log.INFO("ASSOCIATIVE MAX CONFIG: %s" % am_cfg)
  assert ndl_cfg["dataset"] == tf_cfg["dataset"]

  ##############################################################################
  if ndl_cfg["dataset"] == MNIST or ndl_cfg["dataset"] == FMNIST:
    inpt_shape = (1, 28, 28) if tf_cfg["is_channels_first"] else (28, 28, 1)
    num_clss = 10
  elif ndl_cfg["dataset"] == CIFAR10:
    inpt_shape = (3, 32, 32) if tf_cfg["is_channels_first"] else (32, 32, 3)
    num_clss = 10

  ##############################################################################

  log.INFO("*"*100)

  log.INFO("Testing in TensorNode MaxPooling mode...")
  _do_nengo_dl_max_or_max_to_avg(inpt_shape, num_clss, max_to_avg_pool=False)

  """
  log.INFO("Testing in Max To Avg Pooling mode...")
  _do_nengo_dl_max_or_max_to_avg(inpt_shape, num_clss, max_to_avg_pool=True)

  """
  #log.INFO("Testing in AVAM Mode with do_max=True")
  #_do_custom_associative_max_or_avg(inpt_shape, num_clss, do_max=True)

  """
  log.INFO("Testing in custom associative avg mode...")
  _do_custom_associative_max_or_avg(inpt_shape, num_clss, do_max=False)

  log.INFO("Testing in ISI based MaxPooling mode...")
  _do_isi_based_max_pooling(inpt_shape, num_clss)
  """

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--load_tf_wts", type=str, required=True,
                      help="Loads TF weights if True else loads NDL weights.")
  args = parser.parse_args()
  load_tf_wts = True if args.load_tf_wts=="True" else False

  log.configure_log_handler(
      "%s_sfr_%s_n_steps_%s_synapse_%s_%s.log" % (
      ndl_cfg["test_mode"]["test_mode_res_otpt_dir"] + __file__,
      ndl_cfg["test_mode"]["sfr"], ndl_cfg["test_mode"]["n_steps"],
      ndl_cfg["test_mode"]["synapse"], datetime.datetime.now()))

  if load_tf_wts:
    ndl_cfg["test_mode"]["sfr"] = 100
    ndl_cfg["test_mode"]["spk_neuron"] = nengo.SpikingRectifiedLinear()
  ndl_cfg["load_tf_wts"] = load_tf_wts
  nengo_dl_test()
