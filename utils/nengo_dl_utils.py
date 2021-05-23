#
# Author: Ramashish Gaurav
#

import nengo
import nengo_dl
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from focal_loss import SparseCategoricalFocalLoss

from utils.base_utils import log
from utils.cnn_2d_utils import get_2d_cnn_model
from utils.consts.exp_consts import SEED

def get_nengo_dl_model(inpt_shape, tf_cfg, ndl_cfg, mode="test", num_clss=10,
                       collect_probe_history=True, max_to_avg_pool=False):
  """
  Returns the nengo_dl model.

  Args:
    inpt_shape <()>: A tuple of input shape of 2D CNNs in channels_first order.
    tf_cfg <{}>: The experimental configuration.
    ndl_cfg <{}>: Nengo-DL model related configuration.
    model <str>: One of "test" or "train".
    num_clss <int>: Number of classes.
    collect_probe_history <bool>: Collect probes entire `n_steps` simulation time
        history if True, else don't collect spikes.
    max_to_avg_pool <bool>: Set `max_to_avg_pool` in `Converter` to True or False.

  Return:
    nengo.Model
  """
  log.INFO("TF Config: {}".format(tf_cfg))
  log.INFO("Nengo-DL Config: {}".format(ndl_cfg))
  log.INFO("Number of classes: {}".format(num_clss))
  log.INFO("Nengo DL Mode: {}".format(mode))

  # Creating the model.
  model, layer_objs_lst = get_2d_cnn_model(inpt_shape, tf_cfg, num_clss)
  log.INFO("Writing tf_model.summary() to file ndl_tf_model_summary.txt")
  with open(ndl_cfg["test_mode"]["ndl_test_mode_res_otpt_dir"]+
            "/ndl_tf_model_summary.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

  if mode=="test":
    test_cfg = ndl_cfg["test_mode"]
    log.INFO("Test Mode: Loading the TF trained weights in the model...")
    model.load_weights(ndl_cfg["tf_wts_inpt_dir"])
    log.INFO("Test Mode: Converting the TF model to spiking Nengo-DL model...")
    np.random.seed(SEED)
    ndl_model = nengo_dl.Converter(
        model,
        swap_activations={tf.keras.activations.relu: test_cfg["spk_neuron"]},
        scale_firing_rates=test_cfg["sfr"],
        synapse=test_cfg["synapse"],
        inference_only=True,
        max_to_avg_pool=max_to_avg_pool
    )

    # Explicitly set the connection synapse from Conv to MaxPooling layers.
    for i, conn in enumerate(ndl_model.net.all_connections):
      if isinstance(conn.pre_obj, nengo.ensemble.Neurons):
        if (isinstance(conn.post_obj, nengo_dl.tensor_node.TensorNode) and
            conn.post_obj.label.startswith("max_pooling")):
          log.INFO("Connection: {}, | and prior to explicit synapsing: {}".format(
              conn, conn.synapse))
          ndl_model.net._connections[i].synapse = nengo.Lowpass(test_cfg["synapse"])
          log.INFO("Connection: {}, | and after explicit synapsing: {}".format(
              conn, conn.synapse))

  else:
    train_cfg = ndl_cfg["train_mode"]
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
    for lyr_obj in layer_objs_lst[1:-1]:
      # Skip the probes for MaxPooling layers as they won't be present in the
      # Associative-Max based SNN, else execution will error out.
      # TODO: Include it for other analysis though.
      if not lyr_obj.name.startswith("max_pooling"):
        nengo_probes_obj_lst.append(nengo.Probe(ndl_model.layers[lyr_obj]))
  # Set the probes on the Output layer of the Nengo-DL model.
  nengo_output = ndl_model.outputs[layer_objs_lst[-1]]
  nengo_probes_obj_lst.append(nengo_output)

  return ndl_model, nengo_probes_obj_lst

def percentile_l2_loss_range(y_true, y, sample_weight=None, min_rate=0.0,
                             max_rate=150.0, percentile=99.0):
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
