import nengo
import nengo_loihi
import numpy as np
import random
import warnings

import _init_paths

from configs.exp_configs import nengo_loihi_cfg as nloihi_cfg
from utils.base_utils import log
from utils.base_utils.data_prep_utils import get_exp_dataset
from utils.base_utils.exp_utils import get_grouped_slices_2d_pooling
from utils.consts.exp_consts import SEED
from utils.nengo_dl_utils import get_nengo_dl_model
from utils.nengo_loihi_utils import configure_ensemble_for_2x2_max_join_op

# Ignore NengoDL warning about no GPU.
warnings.filterwarnings("ignore", message="No GPU", module="nengo_dl")

# Set the SEED.
random.seed(SEED)
np.random.seed(SEED)

#TODO:
# Fix the get NengoDL model function, modify get 2D CNN model to train without softmax and with to_spikes layer.
# Also set use_bias=False.

def _do_nengo_loihi_MAX_joinOP_MaxPooling(inpt_shape, num_clss):
  """
  """
  log.INFO("Getting the NengoDL model...")
  ndl_model, ngo_probes_lst = get_nengo_dl_model(
      inpt_shape, tf_cfg, ndl_cfg, model="test", num_clss=num_clss,
      max_to_avg_pool=False, include_non_max_pool_probes=False)
  log.INFO("Getting the dataset: %s" % nloihi_cfg["dataset"])
  _, _, test_x, test_y = get_exp_dataset(nloihi_cfg["dataset"])
  # Flatten `test_x`: (10000, 1, 784) for MNIST.
  test_x = test_x.reshape((test_x.shape[0], 1, -1))

  log.INFO("Loading the trained Params: %s" % nloihi_cfg["trained_model_params"])
  nengo_input, nengo_output = ngo_probes_lst[0], ngo_probes_lst[-1]
  with nengo_dl.Simulator(ndl_model.net) as ndl_sim:
    ndl_sim.load_params(nloihi_cfg["trained_model_params"]+"/ndl_trained_params")
    ndl_sim.freeze_params(ndl_model.net)

  log.INFO("Configuring the network...")
  with ndl_model.net:
    nengo_loihi.add_params(ndl_model.net) # Allow on_chip to be set for Ensembles.
    # TODO: Check if on_chip is set True for the new MAX joinOp Ensemble?
    # In the TF model, the first Conv layer (immediately after the Input layer)
    # is responsible to converting images to spikes, therefore set it to run Off-Chip.
    ndl_model.net.config[
        ndl_model.layers[ndl_model.model.layers[1]].ensemble].on_chip = False
    nengo_input.output = nengo.processes.PresentInput(
        test_x, presentation_time=nloihi_cfg["presentation_time"])

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
      num_neurons = num_chnls * rows * cols
      max_join_op_ens = nengo.Ensemble(
        n_neurons = num_neurons, dimensions=1, gain=1000*np.ones(num_neurons),
        bias=np.zeros(num_neurons), seed=SEED,
        neuron_type=nengo_loihi.neurons.LoihiSpikingRectifiedLinear(
          amplitude=nloihi_cfg["scale"]/1000,
          initial_state={"voltage": np.zeros(num_neurons)}
        )
      # Set the BlockShape of `max_join_op_ens` on Loihi Neurocore.
      ndl_model.net[max_join_op_ens].block_shape = nengo_loihi.BlockShape(
          (1, rows, cols), (num_chnls, rows, cols))

      ######### CONNECT THE PREV ENS/CONV TO MAX_JOINOP_ENSEMBLE #########
      nengo.Connection(
          conn_from_pconv_to_max.pre_obj[grouped_slices],
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
          transform=conn_from_max_to_nconv.transform,
          synapse=0.005, # Synapse required because the input to Conv is spikes.
          function=conn_from_max_to_nconv.function
      )

      log.INFO("To/From connection w.r.t. % done!"
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
    log.INFO("Connection: {} | Synapse: {}".format(conn, conn.synapse)

  ############### SET THE BLOCK SHAPES ON LOIHI NEUROCORES #####################
  # Only the intermediate Conv layers and Dense layers run on Loihi, the Input
  # layer, first Conv layer (to convert input to spikes) run off-chip, and the
  # last Dense layer (to collect class prediction scores/logits) runs off-chip.
  with ndl_model.net:
    bls_dict, i = nloihi_cfg["layer_blockshapes"][tf_cfg["tf_model"]["name"]], 0
    for layer in ndl_model.model.layers[2:-1]:
      if layer.name.startswith("conv"):
        conv_shape = tuple(layer.output.shape[1:])
        ndl_model.net.config[
            ndl_model.layers[layer].ensemble].block_shape = (
            nengo_loihi.BlockShape(bls_dict["conv2d_%s" % i], conv_shape))
        i+=1
      # For the intermediate "Dense" layers, they are already small Ensembles
      # with number of neurons less than 1024, so no need to partition them.

  with nengo_loihi.Simulator(ndl_model.net, target="loihi") as loihi_sim:
    configure_ensemble_for_2x2_max_join_op(loihi_sim, max_join_op_ens) # TODO: More than 1 max_join_op_ens?
    loihi_sim.run(nloihi_cfg["n_test"] * nloihi_cfg["presentation_time"])

  # Get the output (last timestep of each presentation period)
  pres_steps = int(round(nloihi_cfg["presentation_time"] / loihi_sim.dt))
  output = loihi_sim.data[nengo_output][pres_steps-1 :: pres_steps]
  loihi_predictions = np.argmax(output, axis=-1)
  correct = 100 * np.mean(
      loihi_predictions == np.argmax(test_y[:nloihi_cfg["n_test"]], axis=-1))
  log.INFO("Loihi Accuracy with Model: %s is: %s"
           % (tf_cfg["tf_model"]["name"], correct))
