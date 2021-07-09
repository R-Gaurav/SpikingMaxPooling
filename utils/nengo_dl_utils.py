#
# Author: Ramashish Gaurav
#

import nengo
import nengo_dl
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from focal_loss import SparseCategoricalFocalLoss

from configs.exp_configs import nengo_dl_cfg as ndl_cfg
from utils.base_utils import log
from utils.cnn_2d_utils import get_2d_cnn_model
from utils.consts.exp_consts import (ISI_BASED_MP_PARAMS, SEED,
                                     NEURONS_LAST_SPIKED_TS, NEURONS_LATEST_ISI,
                                     MAX_POOL_MASK, NUM_X)

def get_nengo_dl_model(inpt_shape, tf_cfg, ngo_cfg, mode="test", num_clss=10,
                       collect_probe_history=True, max_to_avg_pool=False,
                       load_tf_trained_wts=False, include_layer_probes=False,
                       include_mp_layer_probes=False, channels_first=True):
  """
  Returns the nengo_dl model.

  Args:
    inpt_shape <()>: A tuple of input shape of 2D CNNs in channels_first order.
    tf_cfg <{}>: The experimental configuration.
    ngo_cfg <{}>: Nengo-DL/NengoLoihi model related configuration.
    model <str>: One of "test" or "train".
    num_clss <int>: Number of classes.
    collect_probe_history <bool>: Collect probes entire `n_steps` simulation time
        history if True, else don't collect spikes.
    max_to_avg_pool <bool>: Set `max_to_avg_pool` in `Converter` to True or False.
    load_tf_trained_wts <bool>: Load TF trained weights if True else don't.
    include_layer_probes <bool>: Collect the probe data of Non-MP layers if True
                                 else don't. Note: Setting it True doesn't collect
                                 MP layers output, for which, flag below is used.
    include_mp_layer_probes <bool>: Collect the probe data MP layers if True
                                    else don't.

  Returns:
    nengo.Model, [Probes]
  """
  log.INFO("TF Config: {}".format(tf_cfg))
  log.INFO("Nengo-DL Config: {}".format(ngo_cfg))
  log.INFO("Number of classes: {}".format(num_clss))
  log.INFO("Nengo DL Mode: {}".format(mode))

  # Creating the model.
  include_dropout = mode == "train"
  model, layer_objs_lst = get_2d_cnn_model(
      inpt_shape, tf_cfg, num_clss, include_dropout=include_dropout,
      channels_first=channels_first)
  log.INFO("Writing tf_model.summary() to file nengo_tf_model_summary.txt")
  file_path = (ngo_cfg["test_mode"]["test_mode_res_otpt_dir"] if mode == "test"
               else ngo_cfg["train_mode"]["ndl_train_mode_res_otpt_dir"])
  with open(file_path + "/nengo_tf_model_summary.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

  if mode == "test":
    test_cfg = ngo_cfg["test_mode"]
    if load_tf_trained_wts:
      log.INFO("Test Mode: Loading the TF trained weights in the model...")
      model.load_weights(ngo_cfg["tf_wts_inpt_dir"])

    log.INFO("Test Mode: Converting the TF model to spiking Nengo-DL model...")
    np.random.seed(SEED)
    ndl_model = nengo_dl.Converter(
        model,
        swap_activations={tf.keras.activations.relu: test_cfg["spk_neuron"]},
        scale_firing_rates=test_cfg["sfr"],
        synapse=test_cfg["synapse"],
        #inference_only=True,
        max_to_avg_pool=max_to_avg_pool
    )

    # Explicitly set the connection synapse from Conv to MaxPooling layers.
    for i, conn in enumerate(ndl_model.net.all_connections):
      if isinstance(conn.pre_obj, nengo.ensemble.Neurons):
        if (isinstance(conn.post_obj, nengo_dl.tensor_node.TensorNode) and
            conn.post_obj.label.startswith("max_pooling")):
          log.INFO(
              "Connection: {}, | and prior to explicit synapsing: {}".format(
              conn, conn.synapse))
          ndl_model.net._connections[i].synapse = nengo.Lowpass(
              test_cfg["synapse"])
          log.INFO("Connection: {}, | and after explicit synapsing: {}".format(
              conn, conn.synapse))

  else:
    train_cfg = ngo_cfg["train_mode"]
    log.INFO("Train Mode: Converting the obtained TF model to Nengo-DL model..")
    np.random.seed(SEED)
    ndl_model = nengo_dl.Converter(
        model,
        swap_activations={tf.keras.activations.relu: train_cfg["neuron"]},
        scale_firing_rates=train_cfg["sfr"]
    )

  for i in range(len(ndl_model.net.ensembles)):
    ensemble = ndl_model.net.ensembles[i]
    log.INFO("Layer: %s, Nengo Neuron Type: %s, Max Firing Rates: %s" % (
             ensemble.label, ensemble.neuron_type, ensemble.max_rates))

  # Set the probe on the Input layer of the Nengo-DL model.
  nengo_probes_obj_lst = []
  nengo_input = ndl_model.inputs[layer_objs_lst[0]]
  nengo_probes_obj_lst.append(nengo_input)
  # Set the probes on the Conv + Dense layers of the Nengo-DL model if you want
  # to probe their output.
  with ndl_model.net:
    if not collect_probe_history:
      nengo_dl.configure_settings(keep_history=False)
    nengo_dl.configure_settings(stateful=False)
    if include_layer_probes:
      for lyr_obj in layer_objs_lst[1:-1]:
        # Skip the probes for MaxPooling layers as they won't be present in the
        # Associative-Max based SNN, else execution will error out.
        # TODO: Include it for other analysis though.
        if not lyr_obj.name.startswith("max_pooling"):
          nengo_probes_obj_lst.append(nengo.Probe(ndl_model.layers[lyr_obj]))
    if include_mp_layer_probes:
      for lyr_obj in layer_objs_lst[1:-1]:
        if lyr_obj.name.startswith("max_pooling"):
          nengo_probes_obj_lst.append(nengo.Probe(ndl_model.layers[lyr_obj]))
  # Set the probes on the Output layer of the Nengo-DL model.
  nengo_output = ndl_model.outputs[layer_objs_lst[-1]]
  nengo_probes_obj_lst.append(nengo_output)

  return ndl_model, nengo_probes_obj_lst

def percentile_l2_loss_range(y_true, y, sample_weight=None, min_rate=0.0,
                             max_rate=1.0, percentile=99.9):
  """
  Compute the percentile loss for regularizing firing rate of neurons.
    https://www.frontiersin.org/articles/10.3389/fnins.2020.00662/full

  Args:
    y_true (numpy.ndarray): The expected firing rate (not used in this function).
    y (numpy.ndarray): The actual firing rate to be regularized.
    sample_weight (float): The sample_weightage of firing rate loss.
    min_rate (float): Minimum firing rate of neurons.
    max_rate (float): Maximum firing rate of neurons.
    percentile (float): The percentile of firing rate below which all rates
                        should be.

  Returns:
    float: The loss value.
  """
  assert len(y.shape) == 3 # (batch_size, time (==1), num_neurons)
  # Get 99th percentile firing rate among rates of all neurons.
  rates = tfp.stats.percentile(y, percentile, axis=(0, 1)) # Shape: `num_neurons`.
  low_error = tf.maximum(0.0, min_rate - rates)
  high_error = tf.maximum(0.0, rates - max_rate)
  loss = tf.nn.l2_loss(low_error + high_error)

  return (sample_weight * loss) if sample_weight is not None else loss

def nengo_dl_focal_loss(y_true, y_pred):
  assert len(y_true.shape) == 3 # (batch_size, 1, num_clss) i.e. binarized class.
  assert len(y_pred.shape) == 3 # (batch_size, 1, num_clss)

  y_pred = tf.squeeze(y_pred, axis=1) # -> [batch_size, num_classes]
  y_true = tf.squeeze(y_true, axis=1) # -> [batch_size, num_classes]
  y_true = tf.argmax(y_true, axis=-1)

  loss = SparseCategoricalFocalLoss(gamma=2.0)
  return loss(y_true, y_pred)

def get_network_for_2x2_max_pooling(seed=SEED, max_rate=250, radius=1, sf=1,
                                    do_max=True, synapse=None):
  """
  Returns a network for associative max pooling using |x| calculation.

  Args:
    seed <int>: Any arbitrary seed value.
    max_rates <int>: Max Firing rate of the neurons.
    radius <int>: Value at which maximum spiking rate occurs (
                  i.e. representational radius)
    sf <int>: Scale factor by which to scale the inputs.
    do_max <bool>: Do MaxPooling if True else do AvgPooling.
    synapse <float>: Synaptic time constant.

  Returns:
    nengo.Network
  """
  with nengo.Network(seed=seed) as net:
    net.input = nengo.Node(size_in=4) # 4 dimensional input for 2x2 pooling.

    def _get_ensemble():
      ens = nengo.Ensemble(
          n_neurons=2, dimensions=1, encoders = [[1], [-1]], intercepts=[0, 0],
          max_rates=[max_rate, max_rate], radius=radius,
          neuron_type=nengo.SpikingRectifiedLinear())
      return ens

    ens_12 = _get_ensemble() # Ensemble for max(a, b).
    ens_34 = _get_ensemble() # Ensemble for max(c, d).
    ens_1234 = _get_ensemble() # Ensemble for max(max(a,b), max(c, d)).

    # Intermediate passthrough Nodes for summing and outputing the result.
    node_12 = nengo.Node(size_in=1) # For max(a, b).
    node_34 = nengo.Node(size_in=1) # For max(c, d).
    net.output = nengo.Node(size_in=1) # For max(max(a, b), max(c, d)).

    ############################################################################
    # Calculate max(a, b) = (a+b)/2 + |a-b|/2.
    # Calculate (a+b)/2.
    nengo.Connection(net.input[0], node_12, synapse=None, transform=sf/2)
    nengo.Connection(net.input[1], node_12, synapse=None, transform=sf/2)

    if do_max:
      # Calculate |a-b|/2.
      nengo.Connection(net.input[0], ens_12, synapse=None, transform=sf/2)
      nengo.Connection(net.input[1], ens_12, synapse=None, transform=-sf/2)
      nengo.Connection(
          ens_12.neurons[0], node_12, synapse=synapse, transform=1*radius/max_rate)
      nengo.Connection(
          ens_12.neurons[1], node_12, synapse=synapse, transform=1*radius/max_rate)
    ############################################################################

    ############################################################################
    # Calculate max(c, d) = (c+d)/2 + |c-d|/2.
    # Calculate (c+d)/2.
    nengo.Connection(net.input[2], node_34, synapse=None, transform=sf/2)
    nengo.Connection(net.input[3], node_34, synapse=None, transform=sf/2)

    if do_max:
      # Calculate |c-d|/2.
      nengo.Connection(net.input[2], ens_34, synapse=None, transform=sf/2)
      nengo.Connection(net.input[3], ens_34, synapse=None, transform=-sf/2)
      nengo.Connection(
          ens_34.neurons[0], node_34, synapse=synapse, transform=1*radius/max_rate)
      nengo.Connection(
          ens_34.neurons[1], node_34, synapse=synapse, transform=1*radius/max_rate)
    ############################################################################

    ############################################################################
    # Calculate max(a, b, c, d) = max(max(a, b), max(c, d)).
    # Calculate (node_12 + node_34)/2.
    nengo.Connection(node_12, net.output, synapse=synapse, transform=1/2)
    nengo.Connection(node_34, net.output, synapse=synapse, transform=1/2)

    if do_max:
      # Calculate |node_12 - node_34|/2.
      nengo.Connection(node_12, ens_1234, synapse=synapse, transform=1/2)
      nengo.Connection(node_34, ens_1234, synapse=synapse, transform=-1/2)
      nengo.Connection(ens_1234.neurons[0], net.output, synapse=synapse,
                       transform=1*radius/max_rate)
      nengo.Connection(ens_1234.neurons[1], net.output, synapse=synapse,
                       transform=1*radius/max_rate)
    ############################################################################

  return net

def get_max_pool_global_net(mp_input_size, seed=SEED, max_rate=250, radius=1,
                            sf=1, do_max=True, synapse=None):
  """
  Returns the global max pool net, where there are multiple small max pool subnets
  to compute max over 4 numbers. Note that the flattened vector over which MaxPool
  has to be taken, should be orginally arranged in `channels_first` coding. Also
  make sure that the inputs to the max pool global net is grouped in slices
  accordingly as: [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15, ...]

  Args:
    mp_input_size: Global size of the input over MaxPool has to be computed. If
                   the previous Conv layer is of shape (32, 26, 26), then its
                   value should be 32 x 26 x 26 = 21632.
    seed <int>: Any arbitray seed value.
    max_rate <int>: Max Firing rate of the neurons.
    radius <int>: Value at which maximum spiking rate occurs (
                  i.e. representational radius)
    sf <int>: Scale factor by which to scale the inputs.
    do_max <bool>: Do MaxPooling if True else do AvgPooling.
    synapse <float>: Synaptic time constant.

  Returns:
    nengo.Network
  """
  num_chnls, rows, cols = mp_input_size
  with nengo.Network(label="custom_max_pool_layer", seed=seed) as net:
    net.input = nengo.Node(size_in=np.prod(mp_input_size))
    if rows % 2 and cols % 2:
      out_size = (num_chnls * (rows-1) * (cols-1))//4
    else:
      out_size = np.prod(mp_input_size)//4

    net.output = nengo.Node(size_in=out_size)

    for i in range(out_size):
      mp_subnet = get_network_for_2x2_max_pooling(
          seed, max_rate, radius, sf, do_max=do_max, synapse=synapse)
      # Connect the grouped slice of 4 numbers to the `mp_subnet`. Make sure that
      # the Connection to `net.input` is already synapsed when connecting to this
      # global max pool net, thus no need to synapse further to the subnets.
      nengo.Connection(net.input[i*4 : i*4+4], mp_subnet.input, synapse=None)
      # MaxPool is calculated over already synpased inputs, so no need to synapse.
      nengo.Connection(mp_subnet.output, net.output[i], synapse=None)

  return net

def get_isi_based_maximally_spiking_mask(t, inp):
  """
  Sets a binary mask of size NUM_X (constituting of all 0s except one 1) which
  corresponds to the maximally spiking neuron in the groups of NUM_X in `inp`.
  The binary mask is set in MAX_POOL_MASK.

  Args:
    t <float>: The simulation timestep.
    inp <numpy.array>: The incoming spiking activity of neurons of size
                       `num_chnls` x `rows` x `cols` i.e. size of the output of
                       the previous Convolutional layer.
                       e.g. [10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 10.0, ..]
                       where 10.0 is the spiking amplitude if the neuron spiked,
                       else 0.0.

  Note:
    * When two or more neurons have smallest ISI, we choose the neuron whose
    index is smallest. This can be suboptimal.
    * When this function is executed in a Nengo Node (i.e. passed as a parameter
    to the `output`) it expects exactly two arguments: time as a float and data
    as a numpy array. Although you can mention extra fixed keyword arguments in
    the function, but cannot reinitialize them when passing the function to Node.
  """
  # Based on the size of the `inp`, the `num_chnls`, `rows`, and `cols` are
  # determined from the dictionary ISI_BASED_MP_PARAMS.
  inp_size = np.shape(inp)[0]
  log.INFO("ISI Params for Input Size: %s" % inp_size)
  log.INFO("ISI_BASED_MP_PARAMS: {}".format(ISI_BASED_MP_PARAMS[inp_size]))
  log.INFO("NEURONS_LAST_SPIKED_TS: {}, value: {}".format(
           NEURONS_LAST_SPIKED_TS[inp_size].shape,
           NEURONS_LAST_SPIKED_TS[inp_size][0, 0, 0]))
  log.INFO("NEURONS_LATEST_ISI: {}, value: {}".format(
           NEURONS_LATEST_ISI[inp_size].shape,
           NEURONS_LATEST_ISI[inp_size][0, 0, 0]))
  log.INFO("MAX_POOL_MASK: {}, value: {}".format(
           MAX_POOL_MASK[inp_size].shape, MAX_POOL_MASK[inp_size][0, 0, 0]))

  num_chnls, rows, cols = ISI_BASED_MP_PARAMS[inp_size]
  inp = inp.reshape(num_chnls, rows, cols)
  max_pooled_ret = np.zeros((num_chnls, rows//2, cols//2))

  def _isi_max_pooling(t, x, chnl, r1, r2, c1, c2):
    int_t = int(t*1000.0)
    # Get the local copies of updated NEURONS_LAST_SPIKED_TS, NEURONS_LATEST_ISI,
    # and MAX_POOL_MASK for each timestep `t`.
    max_pool_mask = MAX_POOL_MASK[inp_size][chnl, r1:r2, c1:c2].flatten()
    if int_t >= ndl_cfg["test_mode"]["skip_isi_tstep"]:
      return np.dot(max_pool_mask, x)

    neurons_last_spiked_ts = (
        NEURONS_LAST_SPIKED_TS[inp_size][chnl, r1:r2, c1:c2].flatten())
    neurons_latest_isi = NEURONS_LATEST_ISI[inp_size][chnl, r1:r2, c1:c2].flatten()
    spiked_neurons_mask = np.logical_not(np.isin(x, 0))

    # Case 1: None of the neurons in the considered pooling matrix spiked in this
    # timestep, therefore leave the MACROS unchanged. Since none spiked, the
    # `x` is all 0, hence whatever is the value of MAX_POOL_MASK, the dot product
    # will be 0.
    if np.all(spiked_neurons_mask == False):
      return 0

    # One or more of the neurons have spiked in this timestep.
    # Case 2: Check if the currently spiked neurons spiked previously as well?
    if np.any(neurons_last_spiked_ts[spiked_neurons_mask]):
      # One of more of the currently spiked neurons spiked previously as well.
      # Therefore update only their's latest ISI. Take bitwise `&` operation
      # between the currently spiked_neurons_mask and neurons_last_spiked_ts_mask.
      neurons_last_spiked_ts_mask = np.logical_not(
          np.isin(neurons_last_spiked_ts, 0))
      neurons_whose_isi_to_be_updated = (
          neurons_last_spiked_ts_mask & spiked_neurons_mask)
      # Update the latest ISI of the selected neurons.
      neurons_latest_isi[neurons_whose_isi_to_be_updated] = (
          int_t - neurons_last_spiked_ts[neurons_whose_isi_to_be_updated])
      # Create a max_pool_mask.
      max_pool_mask[:] = np.zeros(NUM_X)
      # Set the mask to 1.0 for the neuron with minimum ISI.
      max_pool_mask[np.argmin(neurons_latest_isi)] = 1.0
      # Update the neurons_last_spiked_ts.
      neurons_last_spiked_ts[spiked_neurons_mask] = int_t

      NEURONS_LAST_SPIKED_TS[inp_size][chnl, r1:r2, c1:c2] = (
          neurons_last_spiked_ts.reshape(2, 2))
      NEURONS_LATEST_ISI[inp_size][chnl, r1:r2, c1:c2] = (
          neurons_latest_isi.reshape(2, 2))
      MAX_POOL_MASK[inp_size][chnl, r1:r2, c1:c2] = max_pool_mask.reshape(2, 2)
      return np.dot(max_pool_mask, x)

    # None of the currently spiked neurons spiked previously! i.e. all the
    # currently spiked neurons have spiked for the first time. Two possible
    # situations exist:
    #   FIRST: There might be few other neurons which could have spiked twice
    #   or more earlier and did not spike in this current timestep; thus
    #   the currently spiked neurons spiked for the first time after a number of
    #   previous timesteps and they don't qualify for maximally spiking neurons.
    #
    #   SECOND: There have been no other neurons which have spiked twice
    #   or more so far, thus, there could be other neurons which spiked first
    #   in earlier timesteps and did not spike in this current timestep OR the
    #   currently spiked neurons are the first ones among the pooled group of
    #   neurons to spike.
    else:
      # Check if there are neurons with smallest ISI.
      if np.min(neurons_latest_isi) != np.inf:
        # Some of the other neurons spiked twice or more in earlier timesteps.
        # This means that the currently spiked neurons (which spiked for the
        # first time) are quite late to spike and we already have neurons which
        # are frequently spiking, thus they can be candidate for min ISI.
        max_pool_mask[:] = np.zeros(NUM_X)
        max_pool_mask[np.argmin(neurons_latest_isi)] = 1.0
        neurons_last_spiked_ts[spiked_neurons_mask] = int_t
      else:
        # There are no neurons which have spiked twice or more earlier, else
        # their ISI would have been calculated. Therefore choose the index of
        # neuron which spiked first and set mask to 1.0 for it, rest 0.0.
        if np.any(neurons_last_spiked_ts):
          # Found at least one neuron which spiked earlier than the current
          # timestep.
          neurons_last_spiked_ts_mask = np.logical_not(
              np.isin(neurons_last_spiked_ts, 0))
          min_last_spiked_ts = np.min(
              neurons_last_spiked_ts[neurons_last_spiked_ts_mask])
          # Choose the first index if multiple neurons have same minimum last
          # spiked timestep. This can be suboptimal.
          earliest_spiked_neuron_index = np.where(
              neurons_last_spiked_ts == min_last_spiked_ts)[0][0]
          max_pool_mask[:] = np.zeros(NUM_X)
          max_pool_mask[earliest_spiked_neuron_index] = 1.0
          neurons_last_spiked_ts[spiked_neurons_mask] = int_t
        else:
          # None of the neurons in the considered pool group have spiked
          # previously even once! thus, currently spiked neurons are the first
          # among all to spike. Choose the index of any of the first spiking
          # neurons and set its mask to 1.0, rest 0.0. (WINNER TAKE ALL?).
          neurons_last_spiked_ts[spiked_neurons_mask] = int_t
          max_pool_mask[:] = np.zeros(NUM_X)
          # Choose the first index if multiple neurons are first spiking ones.
          # This can be suboptimal.
          max_pool_mask[np.where(neurons_last_spiked_ts)[0][0]] = 1.0

      NEURONS_LAST_SPIKED_TS[inp_size][chnl, r1:r2, c1:c2] = (
          neurons_last_spiked_ts.reshape(2, 2))
      NEURONS_LATEST_ISI[inp_size][chnl, r1:r2, c1:c2] = (
          neurons_latest_isi.reshape(2, 2))
      MAX_POOL_MASK[inp_size][chnl, r1:r2, c1:c2] = max_pool_mask.reshape(2, 2)
      return np.dot(max_pool_mask, x)

  for chnl in range(num_chnls):
    for r in range(rows//2):
      for c in range(cols//2):
        max_pooled_ret[chnl, r, c] = _isi_max_pooling(
            t, inp[chnl, r*2:r*2+2, c*2:c*2+2].flatten(), chnl, r*2, r*2+2, c*2,
            c*2+2)

  return max_pooled_ret.flatten()

#def configure_ensemble_for_2x2_max_join_op(loihi_sim, ens):
#  """
#  Configures the Ensemble `ens` neurons for MaxPooling using NxSDK's MAX joinOp
#  method.
#
#  Args:
#    loihi_sim <nengo_loihi.simulator.Simulator>: The NengoLoihi simulator object.
#    ens <nengo.ensemble.Ensemble>: The Nengo Ensmeble object whose neurons are
#                                   supposed to be configured for MaxPooling.
#
#  Note: The number of neurons in `ens` should be equal to the number of neurons
#        in the previous Convolutional ensemble, i.e. # neurons = total number
#  values over which 2 x 2 MaxPooling is supposed to be done. Although, if the
#  previous Convolutional layer has odd number of `rows` and `cols` then the actual
#  number of neurons configured in the Ensemble `ens` will be equalt to:
#  `num_chnls` x `rows-1` x `cols-1`.
#  """
#  nxsdk_board = loihi_sim.sims["loihi"].nxsdk_board
#  board = loihi_sim.sims["loihi"].board
#
#  # Get the blocks (which can be many depending on how large the Ensemble `ens`
#  # is and in how many blocks is it broken).
#  blocks = loihi_sim.model.objs[ens]
#  for block in blocks["in"]:
#    in_chip_idx, in_core_idx, in_block_idx, in_compartment_idxs, _ = (
#        board.find_block(block))
#    nxsdk_core = nxsdk_board.n2Chips[in_chip_idx].n2CoresAsList[in_core_idx]
#    # Set the cxProfileCfg in nxsdk_core. Leave vthProfileCfg unchanged.
#    #log.INFO("For Core Index: {}, vthProfileCfg is: {}".format(
#    #    in_core_idx, nxsdk_core.vthProfileCfg[0].staticCfg))
#    #log.INFO("Before setting the cxProfileCfgs...")
#    #log.INFO("For Core Index: {}, cxProfileCfg[0] is: {}".format(
#    #    in_core_idx, nxsdk_core.cxProfileCfg[0]))
#    #log.INFO("For Core Index: {}, cxProfileCfg[1] is: {}".format(
#    #    in_core_idx, nxsdk_core.cxProfileCfg[1]))
#    #log.INFO("For Core Index: {}, cxProfileCfg[2] is: {}".format(
#    #    in_core_idx, nxsdk_core.cxProfileCfg[2]))
#
#    # Set the cxProfileCfg[0] as the leaf node's profile with `stackOut=3` =>
#    # it pushes the current U to the top of the stack.
#    nxsdk_core.cxProfileCfg[0].configure(stackOut=3, bapAction=0, refractDelay=0)
#    # Set the cxProfileCfg[1] as the intermediate node's profile with `stackIn=2`
#    # => it pops the element from the stack, `joinOp=2` => it does the MAX joinOp
#    # with the popped element from stack and its current U, `stackOut=3` => it
#    # pushes the MAXed current U on the top of the stack,
#    # `decayU=nxsdk_core.cxProfileCfg[0].decayU` => the decay constant for current
#    # U is same as that of the cxProfileCfg[0]. If `decayU` is 0, the current due
#    # incoming spike never decays resulting in constant spiking of the neuron
#    # and if it is default value, then the current decays instantly.
#    nxsdk_core.cxProfileCfg[1].configure(
#        stackIn=2, joinOp=2, stackOut=3, decayU=nxsdk_core.cxProfileCfg[0].decayU)
#    # Set the root node which will output the spikes corresonding to the MAXed U.
#    nxsdk_core.cxProfileCfg[2].configure(
#        stackIn=2, joinOp=2, decayU=nxsdk_core.cxProfileCfg[0].decayU)
#
#    #log.INFO("After setting the cxProfileCfgs...")
#    #log.INFO("For Core Index: {}, cxProfileCfg[0] is: {}".format(
#    #    in_core_idx, nxsdk_core.cxProfileCfg[0]))
#    #log.INFO("For Core Index: {}, cxProfileCfg[1] is: {}".format(
#    #    in_core_idx, nxsdk_core.cxProfileCfg[1]))
#    #log.INFO("For Core Index: {}, cxProfileCfg[2] is: {}".format(
#    #    in_core_idx, nxsdk_core.cxProfileCfg[2]))
#
#    # Set the compartments now.
#    # Since the incoming connection from the previous Conv layer already as the
#    # inputs in order of grouped slices, they are simply connected to the neuron
#    # in this Ensembel `ens` from 0 index onwards.
#    # `in_compartment_idxs` has the mapping of all compartment neurons in a
#    # specific core, starting from index 0.
#
#    # Maximum number of compartment idxs = 1024.
#    for i in range(0, len(in_compartment_idxs), 4):
#      c_idx = in_compartment_idxs[i]
#      # Set a leaf node/compartment.
#      nxsdk_core.cxCfg[c_idx].configure(cxProfile=0, vthProfile=0)
#      # Set two intermediate nodes/compartments.
#      nxsdk_core.cxCfg[c_idx+1].configure(cxProfile=1, vthProfile=0)
#      nxsdk_core.cxCfg[c_idx+2].configure(cxProfile=1, vthProfile=0)
#      # Set a root node/compartment to output spikes corresponding to MAX input.
#      nxsdk_core.cxCfg[c_idx+3].configure(cxProfile=2, vthProfile=0)
