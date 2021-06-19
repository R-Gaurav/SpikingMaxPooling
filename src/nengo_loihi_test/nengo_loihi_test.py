#
# Test a NengoLoihi model on Loihi Board.
#
# Author: Ramashish Gaurav
#

import datetime
import nengo
import nengo_dl
import nengo_loihi
import numpy as np
import random
import warnings

import _init_paths

from configs.exp_configs import tf_exp_cfg as tf_cfg, nengo_loihi_cfg as nloihi_cfg
from utils.base_utils import log
from utils.base_utils.data_prep_utils import get_exp_dataset
from utils.base_utils.exp_utils import get_grouped_slices_2d_pooling
from utils.consts.exp_consts import SEED, MNIST, CIFAR10
from utils.nengo_dl_utils import get_nengo_dl_model
from utils.nengo_loihi_utils import configure_ensemble_for_2x2_max_join_op

# Ignore NengoDL warning about no GPU.
warnings.filterwarnings("ignore", message="No GPU", module="nengo_dl")

# Set the SEED.
random.seed(SEED)
np.random.seed(SEED)

def _do_nengo_loihi_MAX_joinOP_MaxPooling(inpt_shape, num_clss):
  """
  Doest NengoLoihi test of models with MaxPooling implemented by MAX joinOp method.

  Args:
    inpt_shape <(int, int, int)>: A tuple of Image shape with channels_first order.
    num_clss <int>: Number of test classes.
  """
  log.INFO("Getting the NengoDL model...")
  ndl_model, ngo_probes_lst = get_nengo_dl_model(
      inpt_shape, tf_cfg, nloihi_cfg, mode="test", num_clss=num_clss,
      max_to_avg_pool=False, include_non_max_pool_probes=False)
  log.INFO("Getting the dataset: %s" % nloihi_cfg["dataset"])
  _, _, test_x, test_y = get_exp_dataset(nloihi_cfg["dataset"])
  # Flatten `test_x`: (10000, 1, 784) for MNIST.
  test_x = test_x.reshape((test_x.shape[0], 1, -1))
  pres_time = nloihi_cfg["test_mode"]["n_steps"] * 0.001 # Convert to ms.

  log.INFO("Loading the trained Params: %s" % nloihi_cfg["trained_model_params"])
  nengo_input, nengo_output = ngo_probes_lst[0], ngo_probes_lst[-1]
  # Build the Network, load the trained weights, save to network.
  with nengo_dl.Simulator(ndl_model.net) as ndl_sim:
    ndl_sim.load_params(nloihi_cfg["trained_model_params"]+ #"/ndl_trained_params")
                        "/attempting_TN_MP_loihineurons_8_16")
    ndl_sim.freeze_params(ndl_model.net)

  log.INFO("Configuring the network...")
  with ndl_model.net:
    nengo_input.output = nengo.processes.PresentInput(
        test_x, presentation_time=pres_time)

    nengo_loihi.add_params(ndl_model.net) # Allow on_chip to be set for Ensembles.
    # TODO: Check if on_chip is set True for the new MAX joinOp Ensemble?
    # In the TF model, the first Conv layer (immediately after the Input layer)
    # is responsible to converting images to spikes, therefore set it to run Off-Chip.
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
        return layer.output.shape[1:] # (num_chnls, rows, cols)

  ################# REPLACE THE CONNECTIONS ##########################
  # List to store Ensembles doing MAX joinOp to be configured later in the
  # NengoLoihi Simulator.
  max_join_op_ens_list = []
  with ndl_model.net:
    for conn_tpl in all_mp_tn_conns:
      conn_from_pconv_to_max, conn_from_max_to_nconv = conn_tpl
      # Get the Conv layer grouped slices for MaxPooling.
      conv_label = conn_from_pconv_to_max.pre_obj.ensemble.label
      (num_chnls, rows, cols) = _get_conv_layer_output_shape(conv_label)
      grouped_slices = get_grouped_slices_2d_pooling(
          pool_size=(2, 2), num_chnls=num_chnls, rows=rows, cols=cols)
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
        )
      )
      max_join_op_ens_list.append(max_join_op_ens)
      # Set the BlockShape of `max_join_op_ens` on Loihi Neurocore.
      ndl_model.net.config[max_join_op_ens].block_shape = nengo_loihi.BlockShape(
          (1, rows, cols), (num_chnls, rows, cols))

      ######### CONNECT THE PREV ENS/CONV TO MAX_JOINOP_ENSEMBLE #########
      nengo.Connection(
          conn_from_pconv_to_max.pre_obj[grouped_slices[:num_neurons]],
          max_join_op_ens.neurons,
          transform=conn_from_pconv_to_max.transform, # NoTransform.
          synapse=conn_from_pconv_to_max.synapse, # None => Feed Spikes to JoinOp Ens.
          function=conn_from_pconv_to_max.function # None.
      )

      ########### CONNECT THE MAX_JOINOP_ENSEMBLE TO NEXT ENS/CONV ##############
      nengo.Connection(
          # The fourth neuron (index 3, 7, ..) in each group of 4 is the root
          # node/neuron which outputs spikes corresponding to the MAX value.
          max_join_op_ens.neurons[[i for i in range(num_neurons) if i%4==3]],
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

  log.INFO("All connections made. Now checking the new connections (in log)...")
  for conn in ndl_model.net._connections:
    log.INFO("Connection: {} | Synapse: {}".format(conn, conn.synapse))

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
        ndl_model.net.config[
            ndl_model.layers[layer].ensemble].block_shape = (
            nengo_loihi.BlockShape(bls_dict["conv2d_%s" % i], conv_shape))
        i+=1
      # For the intermediate "Dense" layers, they are already small Ensembles
      # with number of neurons less than 1024, so no need to partition them.

  ############## BUILD THE NENGOLOIHI MODEL AND EXECUTE ON LOIHI ###############
  with nengo_loihi.Simulator(ndl_model.net, target="loihi") as loihi_sim:
    for max_join_op_ens in max_join_op_ens_list:
      configure_ensemble_for_2x2_max_join_op(loihi_sim, max_join_op_ens)
    loihi_sim.run(nloihi_cfg["test_mode"]["n_test"] * pres_time)

  # Get the output (last timestep of each presentation period)
  pres_steps = int(round(pres_time / loihi_sim.dt))
  output = loihi_sim.data[nengo_output][pres_steps-1 :: pres_steps]

  # Compute Loihi Accuracy.
  loihi_predictions = np.argmax(output, axis=-1)
  correct = 100 * np.mean(
      loihi_predictions == np.argmax(
      test_y[:nloihi_cfg["test_mode"]["n_test"]], axis=-1))
  log.INFO("Loihi Accuracy with Model: %s is: %s"
           % (tf_cfg["tf_model"]["name"], correct))
  log.INFO("*"*100)

def nengo_loihi_test():
  """
  Does a variety of NengoLoihi test of TF models. Runs the model on Loihi Boards.
  """
  log.INFO("TF CONFIG: %s" % tf_cfg)
  log.INFO("NENGO-LOIHI CONFIG: %s" % nloihi_cfg)
  assert nloihi_cfg["dataset"] == tf_cfg["dataset"]

  ###############################################################################
  if nloihi_cfg["dataset"] == MNIST:
    inpt_shape = (1, 28, 28)
    num_clss = 10
  ###############################################################################

  log.INFO("*"*100)
  log.INFO("Testing the NengoLoihi MAX joinOp MaxPooling mode...")
  _do_nengo_loihi_MAX_joinOP_MaxPooling(inpt_shape, num_clss)

if __name__ == "__main__":
  log.configure_log_handler(
    "%s_sfr_%s_n_steps_%s_synapse_%s_timestamp_%s.log" % (
    nloihi_cfg["test_mode"]["test_mode_res_otpt_dir"] + "_nengo_loihi_test_",
    nloihi_cfg["test_mode"]["sfr"], nloihi_cfg["test_mode"]["n_steps"],
    nloihi_cfg["test_mode"]["synapse"], datetime.datetime.now()))
  nengo_loihi_test()
