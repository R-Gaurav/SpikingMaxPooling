#
# Author: Ramashish Gaurav
#

import matplotlib
import matplotlib.pyplot as plt
import nengo
import numpy as np

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
  spikes_matrix = layer_spk_res * sfr * 0.001 # (default dt = 0.001).
  # Shape of spikes_matrix is n_steps x num_neurons.
  fig, ax = plt.subplots(figsize=(14, 12))
  _, num_total_neurons = spikes_matrix.shape
  if num_random_neurons != None:
    random_neurons = np.random.choice(num_total_neurons, num_random_neurons)
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
    random_neurons = np.random.choice(num_total_neurons, num_rndm_neurons)
    spikes_matrix = spikes_matrix[:, random_neurons]
    row, col = i // 2, i % 2
    _spike_plot(axs[row, col], spikes_matrix)
    axs[row, col].set_ylabel("Neuron Index")
    axs[row, col].set_xlabel("Time in $ms$")
    axs[row, col].set_title(
        "Layer: %s, Total Number of Neurons: %s" % (lyr_name, num_total_neurons))

def _scatter_plot(ax, y_values_lst, title, x_values=None):
  spk_vals, non_spk_vals = y_values_lst[0], y_values_lst[1]

  if x_values != None:
    ax.plot(x_values, spk_vals, 'o', color="black", label="Spiking", alpha=0.03)
    ax.plot(x_values, non_spk_vals, 'o', color="red", label="Non-Spiking", alpha=0.03)
  else:
    ax.plot(spk_vals, 'o', color="black", label="Spiking", alpha=0.03)
    ax.plot(non_spk_vals, 'o', color="red", label="Non-Spiking", alpha=0.03)

  bias = np.mean(spk_vals - non_spk_vals)
  rmse = np.sqrt(np.mean((spk_vals-non_spk_vals)**2))
  covr = np.sum(
      (spk_vals - np.mean(spk_vals)) * (non_spk_vals - np.mean(non_spk_vals)))/(
      spk_vals.shape[0]-1)

  ax.legend()
  ax.set_title("Layer: %s, Bias: %s, Covariance: %s, RMSE: %s"
               % (title, bias, covr, rmse))
  ax.set_xlabel("Neurons Indices")
  ax.set_ylabel("Firing Frequency $Hz$")

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
  fig, axs = plt.subplots(num_plots, figsize=(14, 3*num_plots*2))
  fig.suptitle("True Class: %s, ReLU Predicted Class: %s, "
               "Spiking Neuron Predicted Class: %s"
               % (relu_y_true, relu_y_pred, spk_y_pred))

  for i in range(num_plots):
    if i==0: # Plot the class probability.
      _scatter_plot(
          axs[i], [spk_prob_otpt[-1], relu_prob_otpt[-1]], "Class Probability")
    else: # Get the spikes and firing rates.
      # For the spiking neurons.
      spikes_matrix = spks_fr_layers[layers_name[i-1]] * sfr * 0.001
      n_steps, num_total_neurons = spikes_matrix.shape
      #spks_fr_otpt = np.sum(spikes_matrix, axis=0) / (n_steps * 0.001)
      random_neurons = np.random.choice(num_total_neurons, num_rndm_neurons)
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
          axs[i], [spks_fr_otpt, relu_fr_otpt], layers_name[i-1])


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
