#
# Author: Ramashish Gaurav
#

import nengo
import nengo_dl
import tensorflow as tf

from utils.base_utils import log
from utils.cnn_2d_utils import get_2d_cnn_model

def get_nengo_dl_model(inpt_shape, exp_cfg, nengo_cfg, mode="test", num_clss=10):
  """
  Returns the nengo_dl model.

  Args:
    inpt_shape <()>: A tuple of input shape of 2D CNNs.
    exp_cfg <{}>: The experimental configuration.
    nengo_cfg <{}>: Nengo-DL model related configuration.
    model <str>: One of "test" or "train".
    num_clss <int>: Number of classes.

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
    model.load_weights(nengo_cfg["tf_wts_inpt_dir"])
    nengo_model = nengo_dl.Converter(
      model, swap_activations={tf.keras.activations.relu: nengo_cfg["spk_neuron"]},
      scale_firing_rates=nengo_cfg["sfr"], synapse=nengo_cfg["synapse"],
      inference_only=True)
  else:
    nengo_model = nengo_dl.Converter(model)

  for i in range(len(nengo_model.net.ensembles)):
    ensemble = nengo_model.net.ensembles[i]
    log.INFO("Layer: %s, Nengo Neuron Type: %s, Max Firing Rates: %s" % (
             ensemble.label, ensemble.neuron_type, ensemble.max_rates))

  # Set the probes on the Input layer of the Nengo-DL model.
  nengo_probes_obj_lst = []
  nengo_input = nengo_model.inputs[layer_objs_lst[0]]
  nengo_probes_obj_lst.append(nengo_input)
  # Set the probes on the Conv + Dense layers of the Nengo-DL model.
  with nengo_model.net:
    for lyr_obj in layer_objs_lst[1:-1]:
      nengo_probes_obj_lst.append(nengo.Probe(nengo_model.layers[lyr_obj]))
  # Set the probes on the Output layer of the Nengo-DL model.
  nengo_output = nengo_model.outputs[layer_objs_lst[-1]]
  nengo_probes_obj_lst.append(nengo_output)

  return nengo_model, nengo_probes_obj_lst

def percentile_l2_loss_range(y_true, y):
  pass
