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

def get_nengo_dl_model(inpt_shape, exp_cfg, nengo_cfg, mode="test", num_clss=10,
                       load_weights=True, collect_probe_history=True):
  """
  Returns the nengo_dl model.

  Args:
    inpt_shape <()>: A tuple of input shape of 2D CNNs.
    exp_cfg <{}>: The experimental configuration.
    nengo_cfg <{}>: Nengo-DL model related configuration.
    model <str>: One of "test" or "train".
    num_clss <int>: Number of classes.
    load_weights <bool>: True if TF trained weights should be loaded else False.
    collect_probe_history <bool>: Collect probes entire `n_steps` simulation time
        history if True, else don't collect spikes.

  Return:
    nengo.Model.
  """
  log.INFO("Exp Config: {}".format(exp_cfg))
  log.INFO("Nengo Config: {}".format(nengo_cfg))
  log.INFO("Number of classes: {}".format(num_clss))
  log.INFO("Mode: {}".format(mode))

  # Creating the model.
  model, layer_objs_lst = get_2d_cnn_model(inpt_shape, exp_cfg, num_clss)
  if mode=="test":
    if load_weights:
      model.load_weights(nengo_cfg["tf_wts_inpt_dir"])
    nengo_model = nengo_dl.Converter(
      model,
      swap_activations={tf.keras.activations.relu: nengo_cfg["spk_neuron"]},
      scale_firing_rates=nengo_cfg["sfr"],
      synapse=nengo_cfg["synapse"],
      #inference_only=True
      )
  else:
    nengo_model = nengo_dl.Converter(model, scale_firing_rates=250)

  for i in range(len(nengo_model.net.ensembles)):
    ensemble = nengo_model.net.ensembles[i]
    log.INFO("Layer: %s, Nengo Neuron Type: %s, Max Firing Rates: %s" % (
             ensemble.label, ensemble.neuron_type, ensemble.max_rates))

  # Set the probe on the Input layer of the Nengo-DL model.
  nengo_probes_obj_lst = []
  nengo_input = nengo_model.inputs[layer_objs_lst[0]]
  nengo_probes_obj_lst.append(nengo_input)
  # Set the probes on the Conv + Dense layers of the Nengo-DL model if you want
  # to probe their output.
  with nengo_model.net:
    if not collect_probe_history:
      nengo_dl.configure_settings(keep_history=False)
    nengo_dl.configure_settings(stateful=False)
    for lyr_obj in layer_objs_lst[1:-1]:
      nengo_probes_obj_lst.append(nengo.Probe(nengo_model.layers[lyr_obj]))
  # Set the probes on the Output layer of the Nengo-DL model.
  nengo_output = nengo_model.outputs[layer_objs_lst[-1]]
  nengo_probes_obj_lst.append(nengo_output)

  return nengo_model, nengo_probes_obj_lst

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
