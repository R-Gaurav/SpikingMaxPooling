#
# Test a NengoLoihi model on Loihi Board.
#
# Author: Ramashish Gaurav
#

import argparse
import datetime
import nengo
import nengo_dl
import nengo_loihi
import numpy as np
import random
import warnings

import _init_paths

from collections import defaultdict
from configs.exp_configs import tf_exp_cfg as tf_cfg, nengo_loihi_cfg as nloihi_cfg
from utils.base_utils import log
from utils.base_utils.data_prep_utils import get_exp_dataset
from utils.base_utils.exp_utils import (
    get_grouped_slices_2d_pooling_cf, get_grouped_slices_2d_pooling_cl)
from utils.consts.exp_consts import SEED, MNIST, CIFAR10
from utils.nengo_dl_utils import get_nengo_dl_model
from utils.nengo_loihi_utils import configure_ensemble_for_2x2_max_join_op

# Ignore NengoDL warning about no GPU.
warnings.filterwarnings("ignore", message="No GPU", module="nengo_dl")

# Set the SEED.
random.seed(SEED)
np.random.seed(SEED)

def _do_nengo_loihi_MAX_joinOP_MaxPooling(inpt_shape, num_clss,
                                          start_idx, end_idx,
                                          include_max_jop_otpt_probes=False):
  """
  Doest NengoLoihi test of models with MaxPooling implemented by MAX joinOp method.

  Args:
    inpt_shape <(int, int, int)>: A tuple of Image shape with channels_first order.
    num_clss <int>: Number of test classes.
    start_idx <int>: The start index (inclusive) of the test dataset.
    end_idx <int>: The end index (exclusive) of the test dataset.

  Returns:
    float, np.ndarray: Test Accuracy, Test class predictions (e.g. [7, 1, ..])
  """
  log.INFO("Getting the NengoDL model for MAX joinOp based MaxPooling...")
  ndl_model, ngo_probes_lst = get_nengo_dl_model(
      inpt_shape, tf_cfg, nloihi_cfg, mode="test", num_clss=num_clss,
      max_to_avg_pool=False, include_layer_probes=True)
  log.INFO("Getting the dataset: %s" % nloihi_cfg["dataset"])
  _, _, test_x, test_y = get_exp_dataset(
      nloihi_cfg["dataset"], channels_first=tf_cfg["is_channels_first"],
      start_idx=start_idx, end_idx=end_idx)
  # Flatten `test_x`.
  print("RG: test_x shape: {}".format(test_x.shape))
  test_x = test_x.reshape((test_x.shape[0], 1, -1))
  pres_time = nloihi_cfg["test_mode"]["n_steps"] * 0.001 # Convert to ms.

  log.INFO("Loading the trained Params: %s" % nloihi_cfg["trained_model_params"])
  nengo_input, nengo_output = ngo_probes_lst[0], ngo_probes_lst[-1]
  # Build the Network, load the trained weights, save to network.
  with nengo_dl.Simulator(ndl_model.net, seed=SEED) as ndl_sim:
    ndl_sim.load_params(nloihi_cfg["trained_model_params"]+ "/ndl_trained_params")
                        #"/attempting_TN_MP_loihineurons_8_16")
    ndl_sim.freeze_params(ndl_model.net)

  log.INFO("Configuring the network...")
  with ndl_model.net:
    nengo_input.output = nengo.processes.PresentInput(
        test_x, presentation_time=pres_time)

    nengo_loihi.add_params(ndl_model.net) # Allow on_chip to be set for Ensembles.
    # TODO: Check if on_chip is set True for the new MAX joinOp Ensemble?
    # In the TF model, the first Conv layer (immediately after the Input layer)
    # is responsible to converting images to spikes, therefore set it to run Off-Chip.
    #print("RG: NDL MOdel layers: ", ndl_model.model.layers[1])
    #print("RG: ", [key for key in ndl_model.layers.keys()])
    #print("RG: NDL Model net config keys: ", ndl_model.net.config)
    ndl_model.net.config[
        ndl_model.layers[ndl_model.model.layers[1]].ensemble].on_chip = False

  # Get the To/Fro connections to the MaxPool TensorNodes to be replaced.
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

  ################# REPLACE THE CONNECTIONS ##########################
  # List to store Ensembles doing MAX joinOp to be configured later in the
  # NengoLoihi Simulator.
  max_join_op_ens_list, max_join_op_ens_probe_lst = [], []
  with ndl_model.net:
    for i, conn_tpl in enumerate(all_mp_tn_conns):
      conn_from_pconv_to_max, conn_from_max_to_nconv = conn_tpl
      # Get the Conv layer grouped slices for MaxPooling.
      conv_label = conn_from_pconv_to_max.pre_obj.ensemble.label
      if tf_cfg["is_channels_first"]:
        (num_chnls, rows, cols) = _get_conv_layer_output_shape(conv_label)
        grouped_slices = get_grouped_slices_2d_pooling_cf(
            pool_size=(2, 2), num_chnls=num_chnls, rows=rows, cols=cols)
      else:
        (rows, cols, num_chnls) = _get_conv_layer_output_shape(conv_label)
        print("RG: Conv layer otpt shape: %s, %s, %s" % (rows, cols, num_chnls))
        grouped_slices = get_grouped_slices_2d_pooling_cl(
            pool_size=(2, 2), num_chnls=num_chnls, rows=rows, cols=cols)

      print("RG: Grouped Slices first 20: %s" % grouped_slices[:20])
      log.INFO("Grouped slices of Conv: %s for MaxPooling obtained." % conv_label)

      # Create the Ensemble to do the MAX joinOp.
      # Check for the feature maps (over which MaxPooling is done) if they have
      # odd number of rows and cols, if odd then discard the last row and column.
      if rows % 2 and cols % 2: # Note that in this project, rows = cols.
        rows, cols = rows-1, cols-1
      num_neurons = num_chnls * rows * cols
      max_join_op_ens = nengo.Ensemble(
        n_neurons=num_neurons, dimensions=1, gain=1000*np.ones(num_neurons),
        bias=np.zeros(num_neurons), seed=SEED,
        neuron_type=nengo_loihi.neurons.LoihiSpikingRectifiedLinear(
          amplitude=nloihi_cfg["test_mode"]["scale"]/1000,
          initial_state={"voltage": np.zeros(num_neurons)}
        ),
        label="max_join_op_ens_%s" % i
      )
      # Probing selected neurons of Max-JoinOP Ensemble results in following error:
      """
      nengo.exceptions.SimulationError: Board connection error: A source compartment can only connect to either discrete axons or population axons, but not both types. Note that spike probes require a discrete axon.
      """
      #if include_max_jop_otpt_probes:
      #  max_join_op_ens_probe_lst.append(
      #      nengo.Probe(
      #      #max_join_op_ens.neurons[[i for i in range(num_neurons) if i%4==3]],
      #      max_join_op_ens.neurons[:20],
      #      synapse=None))
      #  max_join_op_ens_list.append(max_join_op_ens)

      # Set the BlockShape of `max_join_op_ens` on Loihi Neurocore.
      if tf_cfg["is_channels_first"]:
        ndl_model.net.config[max_join_op_ens].block_shape = nengo_loihi.BlockShape(
            (1, rows, cols), (num_chnls, rows, cols))
      else:
        ndl_model.net.config[max_join_op_ens].block_shape = nengo_loihi.BlockShape(
            (rows, cols, 1), (rows, cols, num_chnls))
          #(1, rows, cols), (num_chnls, rows, cols)) # Results in 100% acc in 40 n_steps in MODEL_2.
          #(16, 8, 8), (num_chnls, rows, cols)) # Results is 95% acc in 40 and 50 n_steps in MODEL_2.

      if tf_cfg["is_channels_first"]:
        output_idcs = [i for i in range(num_neurons) if i%4==3]
      else:
        output_idcs = np.array([i for i in range(num_neurons) if i%4==3]).reshape(
            num_chnls, rows//2, cols//2)
        output_idcs = np.moveaxis(output_idcs, 0, -1)
        output_idcs = output_idcs.flatten()
      ######### CONNECT THE PREV ENS/CONV TO MAX_JOINOP_ENSEMBLE #########
      nengo.Connection(
          conn_from_pconv_to_max.pre_obj[grouped_slices[:num_neurons]],
          max_join_op_ens.neurons,
          transform=None, #conn_from_pconv_to_max.transform, # NoTransform.
          # TODO: Remove the following.
          #synapse=conn_from_pconv_to_max.synapse, # Here synapse is 0.005.
          synapse=None, # Feed Spikes to JoinOp Ens instead of filtered signal.
          function=conn_from_pconv_to_max.function # None.
      )

      ########### CONNECT THE MAX_JOINOP_ENSEMBLE TO NEXT ENS/CONV ##############
      nengo.Connection(
          # The fourth neuron (index 3, 7, ..) in each group of 4 is the root
          # node/neuron which outputs spikes corresponding to the MAX value.
          #max_join_op_ens.neurons[[i for i in range(num_neurons) if i%4==3]],
          max_join_op_ens.neurons[output_idcs],
          conn_from_max_to_nconv.post_obj,
          transform=conn_from_max_to_nconv.transform, # Convolution
          # The conn_from_max_to_nconv.synapse is None, but `Synapse` is required
          # because input to next Conv layer from MAX JoinOp Ens is spikes.
          synapse=nloihi_cfg["test_mode"]["synapse"],
          function=conn_from_max_to_nconv.function # None.
      )

      log.INFO("To/From connection w.r.t. %s done!"
               % conn_from_pconv_to_max.post_obj.label)

  ############# REMOVE THE OLD CONNECTIONS #########################
  with ndl_model.net:
    for conn_tpl in all_mp_tn_conns:
      conn_from_pconv_to_max, conn_from_max_to_nconv = conn_tpl
      ndl_model.net._connections.remove(conn_from_pconv_to_max)
      ndl_model.net._connections.remove(conn_from_max_to_nconv)
      ndl_model.net._nodes.remove(conn_from_pconv_to_max.post_obj)

  #log.INFO("All connections made. Now checking the new connections (in log)...")
  #for conn in ndl_model.net._connections:
  #  log.INFO("Connection: {} | Synapse: {}".format(conn, conn.synapse))

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
  with nengo_loihi.Simulator(ndl_model.net, seed=SEED, target="loihi") as loihi_sim:

  ########################z TODO: REMOVE LATER ###########################
  #  for ens in ndl_model.net.all_ensembles:
  #    print("$"*80)
  #    print("RG: ", ens)
  #    #ens = ndl_model.layers[ndl_model.model.layers[1]].ensemble
  #    blocks = loihi_sim.model.objs[ens]
  #    log.INFO("Number of (in and out) Blocks for Ensemble %s are: %s and %s."
  #             % (ens, len(blocks["in"]), len(blocks["out"])))
  #    for block in blocks["in"]:
  #      in_chip_idx, in_core_idx, in_block_idx, in_compartment_idxs, _ = (
  #          board.find_block(block))
  #      print("Ens: %s, block: %s, chip_idx: %s, core_idx: %s"
  #            % (ens, block, in_chip_idx, in_core_idx))
  #    print("$"*80)
  ##############################################################################
    for max_join_op_ens in max_join_op_ens_list:
      configure_ensemble_for_2x2_max_join_op(loihi_sim, max_join_op_ens)
    loihi_sim.run(nloihi_cfg["test_mode"]["n_test"] * pres_time)

  layer_probes_otpt = defaultdict(list)
  # Get the output (last timestep of each presentation period)
  pres_steps = int(round(pres_time / loihi_sim.dt))
  output = loihi_sim.data[nengo_output][pres_steps-1 :: pres_steps]
  for probe in ngo_probes_lst[1:-1]:
    layer_probes_otpt[probe.obj.ensemble.label].extend(loihi_sim.data[probe])
  for probe in max_join_op_ens_probe_lst:
    layer_probes_otpt[probe.obj.label].extend(loihi_sim.data[probe])
  # Compute Loihi Accuracy.
  loihi_predictions = np.argmax(output, axis=-1)
  print("RG: Loihi Prediction classes: {}".format(loihi_predictions))
  correct = 100 * np.mean(
      loihi_predictions == np.argmax(
      test_y[:nloihi_cfg["test_mode"]["n_test"]], axis=-1))
  log.INFO("MAX joinOp based Loihi Accuracy with Model: %s is: %s"
           % (tf_cfg["tf_model"]["name"], correct))
  log.INFO("*"*100)
  return correct, loihi_predictions, layer_probes_otpt

def _do_nengo_loihi_average_pooling(inpt_shape, num_clss, start_idx, end_idx):
  """
  Does NengoLoihi test of models with AveragePooling layers.

  Args:
    inpt_shape <(int, int, int)>: A tuple of Image shape with channels_first order.
    num_clss <int>: Number of test classes.
    start_idx <int>: The start index (inclusive) of the test dataset.
    end_idx <int>: The end index (exclusive) of the test dataset.

  """
  log.INFO("Getting the NengoDL model for AveragePooling spiking CNN...")
  ndl_model, ngo_probes_lst = get_nengo_dl_model(
      inpt_shape, tf_cfg, nloihi_cfg, mode="test", num_clss=num_clss,
      max_to_avg_pool=False)
  log.INFO("Getting the dataset: %s" % nloihi_cfg["dataset"])
  _, _, test_x, test_y = get_exp_dataset(
      nloihi_cfg["dataset"], start_idx=start_idx, end_idx=end_idx)

  # Flatten `test_x`.
  test_x = test_x.reshape((test_x.shape[0], 1, -1))
  pres_time = nloihi_cfg["test_mode"]["n_steps"] * 0.001 # Convert to ms.

  log.INFO("Loading the trained Params: %s" % nloihi_cfg["trained_model_params"])
  nengo_input, nengo_output = ngo_probes_lst[0], ngo_probes_lst[-1]
  # Build the Network, load the trained weights, save to network.
  with nengo_dl.Simulator(ndl_model.net, seed=SEED) as ndl_sim:
    ndl_sim.load_params(nloihi_cfg["trained_model_params"]+ "/ndl_trained_params")
    ndl_sim.freeze_params(ndl_model.net)

  log.INFO("Configuring the network...")
  with ndl_model.net:
    nengo_input.output = nengo.processes.PresentInput(
        test_x, presentation_time=pres_time)
    nengo_loihi.add_params(ndl_model.net) # Allow on_chip to be set for Ensembles.
    # In the TF model, the first Conv layer (immediately after the Input layer)
    # is responsible to converting images to spikes, therefore set it to run Off-Chip.
    ndl_model.net.config[
        ndl_model.layers[ndl_model.model.layers[1]].ensemble].on_chip = False

  ############### SET THE BLOCK SHAPES ON LOIHI NEUROCORES #####################
  # Only the intermediate Conv layers and Dense layers run on Loihi, the Input
  # layer, first Conv layer (to convert input to spikes) run off-chip, and the
  # last Dense layer (to collect class prediction scores/logits) runs off-chip.
  bls_dict, i = nloihi_cfg["layer_blockshapes"][tf_cfg["tf_model"]["name"]], 0
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
  #with nengo_loihi.Simulator(ndl_model.net, remove_passthrough=False) as loihi_sim:
  with nengo_loihi.Simulator(ndl_model.net, seed=SEED) as loihi_sim:
    loihi_sim.run(nloihi_cfg["test_mode"]["n_test"] * pres_time)

  # Get the output (last timestep of each presentation period)
  pres_steps = int(round(pres_time / loihi_sim.dt))
  output = loihi_sim.data[nengo_output][pres_steps-1 :: pres_steps]

  # Compute Loihi Accuracy.
  loihi_predictions = np.argmax(output, axis=-1)
  correct = 100 * np.mean(
      loihi_predictions == np.argmax(
      test_y[:nloihi_cfg["test_mode"]["n_test"]], axis=-1))
  correct = 100 * np.mean(loihi_predictions == np.argmax(test_y, axis=-1))
  log.INFO("Loihi Accuracy with Model: %s is: %s"
           % (tf_cfg["tf_model"]["name"], correct))
  log.INFO("*"*100)
  return correct, loihi_predictions

def nengo_loihi_test(start):
  """
  Does a variety of NengoLoihi test of TF models. Runs the model on Loihi Boards.

  Args:
    start <int>: The start batch number of the test dataset.
  """
  log.INFO("TF CONFIG: %s" % tf_cfg)
  log.INFO("NENGO-LOIHI CONFIG: %s" % nloihi_cfg)
  assert nloihi_cfg["dataset"] == tf_cfg["dataset"]

  ###############################################################################
  if nloihi_cfg["dataset"] == MNIST:
    inpt_shape = (1, 28, 28) if tf_cfg["is_channels_first"] else (28, 28, 1)
    num_clss = 10
    num_test_imgs = 10000
  elif nloihi_cfg["dataset"] == CIFAR10:
    inpt_shape = (3, 32, 32) if tf_cfg["is_channels_first"] else (32, 32, 3)
    num_clss = 10
    num_test_imgs = 10000
  ###############################################################################

  acc_per_batch_list = []
  log.INFO("*"*100)

  while True:
    log.INFO("Testing for start batch: %s" % start)
    start_idx = start*nloihi_cfg["test_mode"]["n_test"]
    end_idx = (start+1) * nloihi_cfg["test_mode"]["n_test"]

    if tf_cfg["tf_model"]["name"].endswith("ap"):
      log.INFO("Testing the NengoLoihi model in AveragePooling mode...")
      acc, loihi_batch_preds = _do_nengo_loihi_average_pooling(
          inpt_shape, num_clss, start_idx=start_idx, end_idx=end_idx)
    else:
      log.INFO("Testing the NengoLoihi MAX joinOp MaxPooling mode...")
      acc, loihi_batch_preds, layer_probes_otpt = (
          _do_nengo_loihi_MAX_joinOP_MaxPooling(
          inpt_shape, num_clss, start_idx=start_idx, end_idx=end_idx,
          include_max_jop_otpt_probes=True))

    acc_per_batch_list.append(acc)
    # Dump the accuracy result for the current batch.
    #np.save(nloihi_cfg["test_mode"]["test_mode_res_otpt_dir"] +
    #        "/Acc_and_preds_batch_start_%s_end_%s.npy" % (start_idx, end_idx),
    #        (acc, loihi_batch_preds))
    np.save(nloihi_cfg["test_mode"]["test_mode_res_otpt_dir"] +
            "/Layer_probes_otpt_batch_start_%s_end_%s.npy" % (start_idx, end_idx),
            layer_probes_otpt)

    log.INFO("Batch: [%s, %s) Done!" % (start_idx, end_idx))
    log.INFO("Up till batch %s, Mean Accuracy so far: %s" % (
              start, np.mean(acc_per_batch_list)))
    start += 1
    if end_idx == num_test_imgs:
      log.INFO("Infernce over all test images done.")
      break

    if start==1: # To check for a single run overs MODELs.
      break

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--start", type=int, required=True, help="Batch start #?")
  args = parser.parse_args()

  log.configure_log_handler(
    "%s_start_%s_sfr_%s_n_steps_%s_synapse_%s_timestamp_%s.log" % (
    nloihi_cfg["test_mode"]["test_mode_res_otpt_dir"] + "_nengo_loihi_test_",
    args.start, nloihi_cfg["test_mode"]["sfr"], nloihi_cfg["test_mode"]["n_steps"],
    nloihi_cfg["test_mode"]["synapse"], datetime.datetime.now()))
  nengo_loihi_test(args.start)
