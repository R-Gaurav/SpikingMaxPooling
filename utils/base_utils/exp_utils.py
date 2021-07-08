#
# Author: Ramashish Gaurav
#

import matplotlib
import matplotlib.pyplot as plt
import nengo
import nengo_dl
import numpy as np
import pickle
import random

import _init_paths

#from configs.exp_configs import tf_exp_cfg as exp_cfg, nengo_dl_cfg as ngo_cfg
from utils.consts.dir_consts import EXP_OTPT_DIR
from utils.consts.exp_consts import (ISI_BASED_MP_PARAMS, SEED, NUM_X,
                                     NEURONS_LAST_SPIKED_TS, NEURONS_LATEST_ISI,
                                     MAX_POOL_MASK)
#from utils.nengo_dl_utils import get_nengo_dl_model

def collect_sim_data_spikes(probes_lst, sim_data):
  """
  Returns the collected spikes.

  Args:
    probes_lst ([Nengo.Probes]): A list of Nengo Probes.
    sim_data (collections.OrderedDict): An ordered dict of the nengo_model output.

  Returns:
    [{}]: List of model outputs, List of spike outputs
  """
  ndl_model_spks = []
  batch_size = sim_data[probes_lst[1]].shape[0]

  for i in range(batch_size):
    dikt = {}
    for lyr_p in probes_lst[1:-1]:
      dikt[lyr_p.obj.ensemble.label] = sim_data[lyr_p][i] # i^th data point.
    ndl_model_spks.append(dikt)

  return ndl_model_spks

def _probability_otpt(ax, test_point, num_clss):
  ax.set_title("True Class: %s, Pred Class: %s" % (test_point[0], test_point[1]))
  ax.plot(test_point[2]) # axis=2 has the prediction probabilities.
  ax.legend([str(j) for j in range(num_clss)], loc="upper left")
  ax.set_xlabel("Timestamp")
  ax.set_ylabel("Probability")

def plot_ndl_otpt_layer(start, otpt_lyr_vals):
  """
  Plots the `otpt_lyr_vals`.

  Args:
    start <int>: The start of the test points.
    otpt_lyr_vals <[()]>: A list of tuples of (y_true, y_pred, Nengo-DL model
                          output layer probabilities) for multiple test data
                          points.
  """
  num_clss = otpt_lyr_vals[0][2].shape[1]
  num_test_points = len(otpt_lyr_vals)
  num_rows = num_test_points // 2
  fig, axs = plt.subplots(num_rows, 2, figsize=(14, 3*num_test_points))
  fig.suptitle("Output Predictions")
  for i, test_point in enumerate(otpt_lyr_vals):
    row, col = i // 2, i % 2
    #axs[row, col].set_title(
    #  "Test Point: %s, True Class: %s, Pred Class: %s"
    #  % (str(start+i), test_point[0], test_point[1]))
    _probability_otpt(axs[row, col], test_point, num_clss)
  fig.tight_layout()
  fig.show()

def _spike_plot(ax, spikes_matrix):
  n_steps, num_neurons = spikes_matrix.shape
  color = matplotlib.cm.get_cmap('tab10')(0)
  timesteps = np.arange(n_steps)

  for i in range(num_neurons):
    # Sometime spikes are not valued at 1, rather 0.99999994 therefore != 0.
    for t in timesteps[np.where(spikes_matrix[:, i] != 0)]:
      ax.plot([t, t], [i+0.5, i+1.5], color=color)
  ax.set_ylim(0.5, num_neurons+0.5)
  ax.set_yticks(list(range(1, num_neurons+1, int(np.ceil(num_neurons/50)))))
  ax.set_xticks(list(range(1, n_steps+1, 10)))

def plot_ndl_layer_spikes(lyr_name, sfr, layer_spk_res, num_random_neurons=None):
  """
  Plots the spikes in `layer_spk_res`.

  Args:
    lyr_name <str>: Name of the layer.
    sfr <int>: Scale Firing Rates paramater.
    layer_spk_res <numpy.ndarray>: Matrix of spiking activity of a particular
                                   layer for a data point.
                                   Shape: n_steps x total # neurons in layer.
    num_random_neurons <int>: Choose `num_random_neurons` if given else plot for
                              all neurons.
  """
  # All values in spikes_matrix are either 0 or 1 (1 denoting a spike).
  spikes_matrix = layer_spk_res * sfr * 0.001 # (default dt = 0.001).
  # Shape of spikes_matrix is n_steps x num_neurons.
  fig, ax = plt.subplots(figsize=(14, 12))
  _, num_total_neurons = spikes_matrix.shape
  if num_random_neurons != None:
    np.random.seed(SEED)
    random_neurons = np.random.choice(
        num_total_neurons, num_random_neurons, replace=False)
  spikes_matrix = spikes_matrix[:, random_neurons]
  _spike_plot(ax, spikes_matrix)
  ax.set_ylabel("Neuron Index")
  ax.set_xlabel("Time in $ms$")
  ax.set_title(
      "Layer: %s, # Neurons: %s" % (lyr_name, num_total_neurons))

def plot_entire_network_info(img, layers_spks_dict, otpt_lyr_vals, sfr,
                             num_rndm_neurons=128):
  """
  Plots the entire network's spiking for a data point, including the probability
  of predicted classes.

  Args:
    img <numpy.ndarray>: A 3D matrix of the image to be plotted.
    layers_spks_dict <{}> : The dict of the spiking activity corresponding to `img`
                            in all the layers.
    otpt_lyr_vals <tuple>: A tuple of (true class, pred class, Nengo-DL model
                           output layer probabilites) for the `img`. The last
                           element of the tuple is of shape: n_steps x num_clss.
    sfr <int> : The scaled firing rates.
    num_rndm_neurons <int>: Number of random neurons to be considered.
  """
  y_true, y_pred, otpt_prob = otpt_lyr_vals
  n_steps, num_clss = otpt_prob.shape
  layer_names = list(layers_spks_dict.keys())
  num_lyrs = len(layer_names)
  num_plots = (num_lyrs + 2)
  num_rows = int(np.ceil(num_plots/2))
  fig, axs = plt.subplots(num_rows, 2, figsize=(14, 3*num_rows*2))
  fig.suptitle("Entire Network Analysis")
  # Plot the test image.
  axs[0, 0].imshow(img)
  # Plot the class probability output.
  _probability_otpt(axs[0, 1], otpt_lyr_vals, num_clss)
  # Plot the spiking activity in all the network layers.
  for i in range(2, num_plots):
    lyr_name = layer_names[i-2]
    spikes_matrix = layers_spks_dict[lyr_name] * sfr * 0.001
    _, num_total_neurons = spikes_matrix.shape
    np.random.seed(SEED)
    random_neurons = np.random.choice(
        num_total_neurons, num_rndm_neurons, replace=False)
    spikes_matrix = spikes_matrix[:, random_neurons]
    row, col = i // 2, i % 2
    _spike_plot(axs[row, col], spikes_matrix)
    axs[row, col].set_ylabel("Neuron Index")
    axs[row, col].set_xlabel("Time in $ms$")
    axs[row, col].set_title(
        "Layer: %s, Total Number of Neurons: %s" % (lyr_name, num_total_neurons))

def _scatter_plot(ax, y_values_lst, title):
  spk_vals, non_spk_vals = y_values_lst[0].reshape(-1), y_values_lst[1]
  corr = np.corrcoef(spk_vals, non_spk_vals)
  bias = np.mean(spk_vals - non_spk_vals)
  rmse = np.sqrt(np.mean((spk_vals-non_spk_vals)**2))
  covr = np.sum(
      (spk_vals - np.mean(spk_vals)) * (non_spk_vals - np.mean(non_spk_vals)))/(
      spk_vals.shape[0]-1)

  ax.set_title("Layer: %s \n, Bias: %s, Covariance: %s, RMSE: %s, Correlation: %s"
               % (title, np.round(bias, 2), np.round(covr, 2), np.round(rmse, 2),
               np.round(corr, 2)))
  ax.scatter(spk_vals, non_spk_vals, color="black")
  ax.set_xlabel("Spiking Neuron Vals")
  ax.set_ylabel("Non-Spiking Neuron Vals (last timestep)")

def plot_comparison_between_spiking_and_relu(sfr, layers_spks_res, layers_relu_res,
                                             num_rndm_neurons=512):
  """
  Plots the comparison between spiking and non-spiking relu outputs.

  Args:
    sfr <int>: The scale firing rates for spiking experiment.
    layers_spks_res <{}>: The dict of spiking activity corresponding to one data
                         point for all layers.
    layers_relu_res <{}>: The dict of the relu activity corresponding to the same
                         data point for all layers.
    num_rndm_neurons <int>: Number of random neurons to choose.
  """
  # Get the class probability outputs.
  spk_y_true, spk_y_pred, spk_prob_otpt = layers_spks_res["acc_res"]
  relu_y_true, relu_y_pred, relu_prob_otpt = layers_relu_res["acc_res"]
  assert spk_y_true == relu_y_true
  # Get the firing rate results.
  spks_fr_layers = layers_spks_res["spk_res"]
  relu_fr_layers = layers_relu_res["spk_res"]

  layers_name = list(layers_spks_res["spk_res"].keys())
  num_plots = len(layers_name) + 1 # +1 for class probability output.
  fig, axs = plt.subplots(int(np.ceil(num_plots/2)), 2, figsize=(16, 3*num_plots))
  fig.suptitle("Filtered Spiking Output - Correlation Plot, True Class: %s, "
               "ReLU Predicted Class: %s, Spiking Neuron Predicted Class: %s\n\n"
               % (relu_y_true, relu_y_pred, spk_y_pred))

  for i in range(num_plots):
    if i==0: # Plot the class probability.
      _scatter_plot(
          axs[int(i/2), int(i%2)], [spk_prob_otpt[-1], relu_prob_otpt[-1]],
          "Class Probability")
    else: # Get the spikes and firing rates.
      # For the spiking neurons.
      spikes_matrix = spks_fr_layers[layers_name[i-1]] * sfr * 0.001
      n_steps, num_total_neurons = spikes_matrix.shape
      #spks_fr_otpt = np.sum(spikes_matrix, axis=0) / (n_steps * 0.001)
      np.random.seed(SEED)
      random_neurons = np.random.choice(
          num_total_neurons, num_rndm_neurons, replace=False)
      spks_fr_otpt = []
      for randon_neuron in random_neurons:
        spks_fr_otpt.append(
            get_filtered_signal_from_spikes(
            spikes_matrix[:, randon_neuron], n_steps)[-1])
      spks_fr_otpt = np.array(spks_fr_otpt)

      # For the relu neurons.
      relu_fr_otpt = relu_fr_layers[layers_name[i-1]]
      # Consider the same random neurons as for spiking neurons.
      relu_fr_otpt = relu_fr_otpt[:, random_neurons].reshape(-1)
      _scatter_plot(
          axs[int(i/2), int(i%2)], [spks_fr_otpt, relu_fr_otpt], layers_name[i-1])

  fig.tight_layout()
  fig.show()

def get_filtered_signal_from_spikes(spike_train, n_steps, synapse=0.005):
  """
  Returns the filtered signal from the spike train.

  Args:
    spike_train <[]>: The list of spikes.
		n_steps <int>: The simulation time.
    synapse <float>: The low pass filter time constant.

  Returns:
    np.ndarray(float)
  """
  def spike_generator(t):
    if int(t*1000) < n_steps:
      return spike_train[int(t*1000)]
    else:
      print("No more spikes to generate!")

  # Obtain the filtered signal.
  with nengo.Network() as net:
    inpt = nengo.Node(output=spike_generator)
    probe = nengo.Probe(inpt, synapse=synapse)

  with nengo.Simulator(net, progress_bar=False) as sim:
    sim.run((n_steps-1)*0.001)

  return sim.data[probe]

def get_tf_non_spiking_relu_results(dataset="cifar10", model="model_1"):
  """
  Returns the Nengo-DL results tested over non-spiking: nengo.RectifiedLinear().

  Args:
    dataset
  """
  return pickle.load(
      open(EXP_OTPT_DIR + "/%s/%s/ndl_relu_results/ndl_%s_results_sfr_1_nstps_1.p"
           % (dataset, model, model), "rb"))

def plot_ndl_model_layers_info(do_plot_tuning_curves=True):
  """
  Plots the detailed spiking related info of the Nengo-DL model.

  Args:
    do_plot_tuning_curves (bool): Should the tuning curves be plotted too?
  """
  ndl_model, _ = get_nengo_dl_model((32, 32, 3), exp_cfg, ngo_cfg)
  # Get the info for each layer except the first (input) and last (output) layer.
  layers = list(ndl_model.layers.dict.values())[1:-1] # The layers are ordered.
  fig, axs = plt.subplots(len(layers), figsize=(16, 3*len(layers)))
  if do_plot_tuning_curves:
    fig1, axs1 = plt.subplots(len(layers), figsize=(16, 3*len(layers)))
  with ndl_model.net:
    nengo_dl.configure_settings(stateful=False)

  with nengo_dl.Simulator(ndl_model.net, minibatch_size=ngo_cfg["test_batch_size"],
                          progress_bar=True, seed=SEED) as sim:
    for i, layer in enumerate(layers):
      ens = layer.ensemble
      # Plot the general info.
      axs[i].set_title(
          "Layer: {0}, Neuron Type: {1}, Number of Neurons: {2}, Seed: {3}".format(
          ens.neurons, ens.neuron_type, ens.n_neurons, ens.seed))
      axs[i].plot(sim.data[ens].max_rates, 'ro', label="Max Firing Rates")
      #axs[i].plot(sim.data[ens].encoders, 'g--', label="Encoders")
      print("Layer: {0}, Neuron Type: {1}, Number of Neurons: {2}, Seed: {3}".format(
            ens.neurons, ens.neuron_type, ens.n_neurons, ens.seed))
      print("Unique Encoders: {}".format(np.unique(sim.data[ens].encoders)))
      #axs[i].plot(sim.data[ens].scaled_encoders, 'b+', label="Scaled Encoders")
      print("Unique Scaled Encoders: {}".format(
            np.unique(sim.data[ens].scaled_encoders)))
      #axs[i].plot(sim.data[ens].bias, 'g^', label="Bias")
      print("Unique bias: {}".format(np.unique(sim.data[ens].bias)))
      #axs[i].plot(sim.data[ens].gain, 'b:', label="Gain")
      print("Unique gain: {}".format(np.unique(sim.data[ens].gain)))
      #axs[i].plot(sim.data[ens].intercepts, 'r1', label="Intercepts")
      print("Unique intercepts: {}".format(np.unique(sim.data[ens].intercepts)))
      axs[i].legend()
      axs[i].set_xlabel("Neuron Indices")
      print("*"*100)

      # Plot the Tuning Curves.
      if do_plot_tuning_curves:
        axs1[i].set_title("Tuning Curves for {}".format(ens.neurons))
        x, act_mat = nengo.utils.ensemble.tuning_curves(ens, sim)
        for j in range(ens.n_neurons):
          axs1[i].plot(act_mat[:, j])
        axs1[i].set_xlabel("x - values")

    fig.tight_layout()
    fig.show()
    fig1.tight_layout()
    fig1.show()
    pass # No need to execute the simlation, as we just need the compiled model.

def get_grouped_slices_2d_pooling(**kwargs):
  """
  Creates square grouped slices based on the `pool_size`. E.g. for a flattened
  array, the indices are [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
  ...] with `pool_size` = (2, 2), `rows=4`, `cols=4`, the flattened array is
  actually:
  [[0, 1, 2, 3],
   [4, 5, 6, 7],
   [8, 9, 10, 11],
   [12, 13, 14, 15]]

  so the returned grouped slices array should be of indices:
  [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15, ...]

  Note: It expects the "Channels First" coding of Input/Conv layers. It is also
  assumed that the input `rows` and `cols` are of same size and `pool_size` also
  has same row/col dimension.

  Args:
    kwargs <dict>:
      pool_size <tuple>: (int, int) for 2D Pooling - (row, col) arrangement.
      num_chnls <int>: Number of channels in the reshaped matrix.
      rows <int>: Number of rows in the reshaped matrix.
      cols <int>: Number of columns in the reshaped matrix.

  Returns:
    <[int]>
  """
  pool_size, num_chnls, rows, cols = (
      kwargs["pool_size"], kwargs["num_chnls"], kwargs["rows"], kwargs["cols"])
  matrix = np.arange(rows * cols * num_chnls).reshape((num_chnls, rows, cols))
  grouped_slices = np.zeros(rows * cols * num_chnls, dtype=int)
  start, slice_len = 0, np.prod(pool_size)

  for chnl in range(num_chnls):
    for row in range(rows//pool_size[0]):
      for col in range(cols//pool_size[1]):
        grouped_slices[start:start+slice_len] = (
            matrix[chnl, row*pool_size[0]:row*pool_size[0]+pool_size[0],
                   col*pool_size[1]:col*pool_size[1]+pool_size[1]]).flatten()
        start += slice_len

  # Return the grouped slices of valid length in case of odd rows/cols.
  #if rows % 2 and cols % 2:
  #  return grouped_slices[: num_chnls*(rows-1)*(cols-1)]

  return grouped_slices

def get_grouped_slices_2d_pooling_cl(**kwargs):
  """
  Returns grouped slices indices for channels last ordering, with all the other
  expectations as mentioned above for channels first ordering.

  Args:
    kwargs <dict>:
      pool_size <tuple>: (int, int) for 2D Pooling - (row, col) arrangement.
      num_chnls <int>: Number of channels in the reshaped matrix.
      rows <int>: Number of rows in the reshaped matrix.
      cols <int>: Number of columns in the reshaped matrix.

  Returns:
    <[int]>
  """
  pool_size, num_chnls, rows, cols = (
        kwargs["pool_size"], kwargs["num_chnls"], kwargs["rows"], kwargs["cols"])
  matrix = np.arange(rows * cols * num_chnls).reshape((rows, cols, num_chnls))
  grouped_slices = np.zeros(rows * cols * num_chnls, dtype=int)
  start, slice_len = 0, np.prod(pool_size)

  for chnl in range(num_chnls):
    for row in range(rows//pool_size[0]):
      for col in range(cols//pool_size[1]):
        grouped_slices[start:start+slice_len] = (
            matrix[row*pool_size[0]:row*pool_size[0]+pool_size[0],
            col*pool_size[1]:col*pool_size[1]+pool_size[1], chnl]).flatten()
        start += slice_len

  return grouped_slices

def get_isi_based_max_pooling_params(layers):
  """
  Populates ISI based MaxPooling experiemt parameters. That is: `MAX_POOL_MASK`,
  `NEURONS_LAST_SPIKED_TS`, `NEURONS_LATEST_ISI`, `ISI_BASED_MP_PARAMS` dicts.

  Args:
    layers <list>: Model's layers' parameters.
  """
  for layer in layers:
    if layer.name.startswith("conv2d"):
      lyr_otp, size = layer.output.shape[1:], np.prod(layer.output.shape[1:])
      ISI_BASED_MP_PARAMS[size] = np.array(lyr_otp)
      NEURONS_LAST_SPIKED_TS[size] = np.zeros(lyr_otp)
      NEURONS_LATEST_ISI[size] = np.ones(lyr_otp)*np.inf
      MAX_POOL_MASK[size] = np.ones(lyr_otp)/NUM_X

def get_shuffled_lists_in_unison(lst_a, lst_b):
  """
  Shuffles two passed lists in unison.
  Args:
    lst_a ([]): 1st list.
    lst_b ([]): 2nd list.
  Returns:
    [], []
  """
  lst_f = list(zip(lst_a, lst_b))
  random.shuffle(lst_f)
  lst_a, lst_b = zip(*lst_f)
  return list(lst_a), list(lst_b)
