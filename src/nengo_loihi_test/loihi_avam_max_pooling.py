#
# Author: Ramashish Gaurav
#
# This file implement the Loihi adapted AVAM method of MaxPooling.
#

import nengo
import nengo_dl
import nengo_loihi
import numpy as np
import random

from collections import defaultdict

import _init_paths

from configs.exp_configs import tf_exp_cfg as tf_cfg, nengo_loihi_cfg as nloihi_cfg
from utils.base_utils import log
from utils.base_utils.data_prep_utils import get_exp_dataset
from utils.base_utils.exp_utils import (
    get_grouped_slices_2d_pooling_cf, get_grouped_slices_2d_pooling_cl)
from utils.consts.exp_consts import SEED
from utils.nengo_dl_utils import get_nengo_dl_model
from utils.nengo_loihi_utils import get_loihi_adapted_max_pool_global_net

# Set the SEED.
random.seed(SEED)
np.random.seed(SEED)

def _do_nengo_loihi_AVAM_MaxPooling(inpt_shape, num_clss, start_idx, end_idx,
                                    include_avam_max_otpt_probes=False,
                                    do_max=True):
  """
  Does the Nengo Loihi adapted AVAM MaxPooling.

  Args:
      inpt_shape <(int, int, int)>: A tuple of Image shape with channels_firs/
                                    channels_last order.
      num_clss <int>: Number of test classes.
      start_idx <int>: The start index (inclusive) of the test dataset.
      end_idx <int>: The end index (exclusive) of the test dataset.
      include_avam_max_otpt_probes <bool>: Probe the outupt of the global AVAM
                                           net if True else don't.

    Returns:
      float, np.ndarray: Test Accuracy, Test class predictions (e.g. [7, 1, ..])
  """
  log.INFO("Getting the NengoDL model for AVAM based MaxPooling with do_max: %s"
           % do_max)
  ndl_model, ngo_probes_lst = get_nengo_dl_model(
      inpt_shape, tf_cfg, nloihi_cfg, mode="test", num_clss=num_clss,
      max_to_avg_pool=False, include_layer_probes=False)
  log.INFO("Getting the dataset: %s" % nloihi_cfg["dataset"])
  _, _, test_x, test_y = get_exp_dataset(
      nloihi_cfg["dataset"], channels_first=tf_cfg["is_channels_first"],
      start_idx=start_idx, end_idx=end_idx, is_nengo_dl_train_test=True)
  # Flatten `test_x`.
  print("RG: test_x shape: {}".format(test_x.shape))
  test_x = test_x.reshape((test_x.shape[0], 1, -1))
  pres_time = nloihi_cfg["test_mode"]["n_steps"] * 0.001
  log.INFO("Loading the trained Params: %s" % nloihi_cfg["trained_model_params"])
  nengo_input, nengo_output = ngo_probes_lst[0], ngo_probes_lst[-1]

  # Build the Network and load the trained weights, save to the network.
  with nengo_dl.Simulator(ndl_model.net, seed=SEED) as ndl_sim:
    ndl_sim.load_params(nloihi_cfg["trained_model_params"]+"/ndl_trained_params")
    ndl_sim.freeze_params(ndl_model.net)

  log.INFO("Configuring the network...")
  with ndl_model.net:
    nengo_input.output = nengo.processes.PresentInput(
        test_x, presentation_time=pres_time)
    nengo_loihi.add_params(ndl_model.net) # Allows on_chip to be set for Ensembles.
    # Set the first Conv `to_spikes` layer to run Off-Chip.
    ndl_model.net.config[
      ndl_model.layers[ndl_model.model.layers[1]].ensemble].on_chip = False

  log.INFO("Getting all the To/Fro connections with MaxPool TensorNodes")
  all_mp_tn_conns = []
  for i, conn in enumerate(ndl_model.net.all_connections):
    if (isinstance(conn.post_obj, nengo_dl.tensor_node.TensorNode) and
        conn.post_obj.label.startswith("max_pooling")):
      conn_from_pconv_to_max = ndl_model.net.all_connections[i]
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
  def _get_conv_layer_output_shape(layer_name):
    for layer in ndl_model.model.layers:
      if layer.name == layer_name.split(".")[0]:
        return layer.output.shape[1:]

  ###################### REPLACE THE CONNECTIONS ###############################
  avam_net_probe_lst = []
  with ndl_model.net:
    for i, conn_tpl in enumerate(all_mp_tn_conns):
      conn_from_pconv_to_max, conn_from_max_to_nconv = conn_tpl
      # Get the Conv layer grouped slices for MaxPooling.
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

      # Get the AVAM layer.
      max_pool_layer = get_loihi_adapted_max_pool_global_net(
          (num_chnls, rows, cols), seed=SEED, max_rate=250, radius=2.5, sf=1,
          do_max=do_max, synapse=0.005)
      log.INFO("AVAM MaxPool layer obtained.")

      ########## CONNECT THE PREV ENS/CONV TO ASSOCIATIVE-MAX MAXPOOL ########
      nengo.Connection(
          conn_from_pconv_to_max.pre_obj[grouped_slices[:num_neurons]],
          max_pool_layer.inputs,
          transform=None,
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

      log.INFO("To/From connection w.r.t %s done."
               % conn_from_pconv_to_max.post_obj.label)

  ############# REMOVE THE OLD CONNECTIONS #########################
  with ndl_model.net:
    for conn_tpl in all_mp_tn_conns:
      conn_from_pconv_to_max, conn_from_max_to_nconv = conn_tpl
      ndl_model.net._connections.remove(conn_from_pconv_to_max)
      ndl_model.net._connections.remove(conn_from_max_to_nconv)
      ndl_model.net._nodes.remove(conn_from_pconv_to_max.post_obj)

  log.INFO("All connections made, now checking the new connections (in log)...")
  for conn in ndl_model.net._connections:
    log.INFO("Connection: {} | Synapse: {}, Transform: {}".format(
              conn, conn.synapse, conn.transform))

  ############### SET THE BLOCK SHAPES ON LOIHI NEUROCORES #####################
  # Only the intermediate Conv layers and Dense layers run on Loihi, the Input
  # layer, first Conv layer (to convert input to spikes) run off-chip, and the
  # last Dense layer (to collect class prediction scores/logits) runs off-chip.
  bls_dict, i = nloihi_cfg["layer_blockshapes"], 0
  with ndl_model.net:
    # Exclude the Conv layer at index 1 since it runs off-chip to create spikes.
    for layer in ndl_model.model.layers[2:-1]:
      if layer.name.startswith("conv"):
        conv_shape = tuple(layer.output.shape[1:])
        if np.prod(conv_shape) <= 1024:
          continue
        ndl_model.net.config[
            ndl_model.layers[layer].ensemble].block_shape = (
            nengo_loihi.BlockShape(bls_dict["conv2d_%s" % i], conv_shape))
        i+=1
        # For the intermediate "Dense" layers, they are already small Ensembles
        # with number of neurons less than 1024, so no need to partition them.

  ############## BUILD THE NENGOLOIHI MODEL AND EXECUTE ON LOIHI ###############
  log.INFO("Start testing...")
  with nengo_loihi.Simulator(ndl_model.net, seed=SEED, target="loihi") as loihi_sim:
    loihi_sim.run(nloihi_cfg["test_mode"]["n_test"] * pres_time)

  layer_probes_otpt = defaultdict(list)
  # Get the output.
  pres_steps = int(round(pres_time / loihi_sim.dt))
  layer_probes_otpt[nengo_output.obj.label].extend(loihi_sim.data[nengo_output])
  output = loihi_sim.data[nengo_output][pres_steps-1 :: pres_steps]
  for probe in ngo_probes_lst[1:-1]:
    layer_probes_otpt[probe.obj.ensemble.label].extend(loihi_sim.data[probe])

  # Compute Loihi Accuracy.
  loihi_predictions = np.argmax(output, axis=-1)
  acc = 100 * np.mean(
      loihi_predictions == np.argmax(
      test_y[:nloihi_cfg["test_mode"]["n_test"]], axis=-1))
  log.INFO("AVAM based Loihi Accuracy with Model: %s is %s"
           % (tf_cfg["tf_model"]["name"], acc))
  log.INFO("*"*100)
  return acc, loihi_predictions, layer_probes_otpt
