#
# POC UTILS
#

import nengo
import nengo_loihi
import numpy as np
import random
from scipy.stats import poisson
import seaborn as sbn
import matplotlib.pyplot as plt
from matplotlib.legend import Legend

def get_groups_of_poisson_distributed_isi(mu, size, num_groups, seed=0):
  np.random.seed(seed)
  isi_poisson = poisson.rvs(mu=mu, size=size)
  ret = []
  for _ in range(num_groups):
    ret.append(random.choices(isi_poisson, k=4))

  return ret, isi_poisson

def get_groups_of_isi_from_given_isi_dist(isi_vals, isi_counts, num_groups, seed=0):
  np.random.seed(seed)
  isi_probs = np.array(isi_counts) / np.sum(isi_counts)
  ret = []
  for _ in range(num_groups):
    ret.append(random.choices(isi_vals, weights=isi_probs, k=4))

  return ret

def plot_distplot(isi_poisson, font_size, file_name):
  plt.figure(figsize=(4, 4))
  ax = sbn.distplot(isi_poisson,
                    bins=len(np.unique(isi_poisson)),
                    kde=False,
                    color='skyblue',
                    hist_kws={"rwidth":0.5,'alpha':1}
                   )
  mids = [rect.get_x() + rect.get_width() / 2 for rect in ax.patches]
  ax.set_xticks(mids)
  ax.set_xticklabels(np.unique(isi_poisson))
  ax.tick_params(labelsize=font_size)
  ax.set_xlabel('ISI', fontsize=font_size)
  ax.set_ylabel('Frequency', fontsize=font_size)
  plt.savefig(file_name, dpi=450, bbox_inches = "tight")

def plot_barplot(xs, heights, font_size, file_name):
  plt.figure(figsize=(4, 4))
  ax = plt.bar(xs, heights, color='skyblue')
  plt.xticks(fontsize=font_size)
  plt.yticks(fontsize=font_size)
  plt.xlabel('ISI', fontsize=font_size)
  plt.ylabel('Mean Frequency', fontsize=font_size)
  plt.savefig(file_name, dpi=450, bbox_inches = "tight")
"""
def plot_sctrplot(x1, x2, x3, x4, y, s1, s2, s3, font_size, lgnd_str, file_name):
  plt.figure(figsize=(4, 4))
  ax = sbn.scatterplot(x1, y, alpha=0.5, s=150, label="$%s=%s$" % (lgnd_str, s1))
  sbn.scatterplot(
      x2, y, alpha=0.5, s=200, label="$%s=%s$" % (lgnd_str, s2), marker="P")
  sbn.scatterplot(
      x3, y, alpha=0.5, s=450, label="$%s=%s$" % (lgnd_str, s3), marker="*")
  sbn.scatterplot(
      x4, y, alpha=0.5, s=450, label="Average", marker="^")
  ax.tick_params(labelsize=font_size)
  ax.set_xticks(np.round(np.arange(0, 1.4, 0.2), 1))
  ax.set_yticks(np.round(np.arange(0, 1.4, 0.2), 1))
  ax.legend(prop={"size": font_size})
  ax.set_xlabel("Estimated Max Values", fontsize=font_size)
  ax.set_ylabel("True Max Values", fontsize=font_size)
  ax.legend(handletextpad=0.1, fontsize=font_size, loc='upper left', framealpha=0.5)
  plt.savefig(file_name, dpi=450, bbox_inches = "tight")
"""
def plot_sctrplot(x1, x2, x3, x4, y, s1, s2, s3, font_size, lgnd_str, file_name):
  fig, ax = plt.subplots(figsize=(4, 4))
  ax1 = ax.scatter(x1, y, alpha=0.5, s=150, label="$%s=%s$" % (lgnd_str, s1))
  ax2 = ax.scatter(
      x2, y, alpha=0.5, s=200, label="$%s=%s$" % (lgnd_str, s2), marker="P")
  ax3 = ax.scatter(
      x3, y, alpha=0.5, s=400, label="$%s=%s$" % (lgnd_str, s3), marker="*")
  ax4 = ax.scatter(
      x4, y, alpha=0.5, s=175, label="Average", marker="^")
  plt.tick_params(labelsize=font_size)
  plt.xticks(np.round(np.arange(0, 1.4, 0.2), 1))
  plt.yticks(np.round(np.arange(0, 1.4, 0.2), 1))
  plt.xlabel("Estimated Max $U$", fontsize=font_size)
  plt.ylabel("True Max $U$", fontsize=font_size)
  plt.legend([ax4, ax1], ["Average", "$%s=%s$" % (lgnd_str, s1)],
             handletextpad=0.1, fontsize=font_size, loc='upper left',
             framealpha=0.325)
  leg = Legend(ax, [ax2, ax3],
              ["$%s=%s$" % (lgnd_str, s2), "$%s=%s$" % (lgnd_str, s3)],
              loc='lower right', handletextpad=0.1, fontsize=font_size,
              framealpha=0.325)
  ax.add_artist(leg);
  plt.savefig(file_name, dpi=450, bbox_inches = "tight")

def configure_ensemble_for_2x2_max_join_op(loihi_sim, ens):
  nxsdk_board = loihi_sim.sims["loihi"].nxsdk_board
  board = loihi_sim.sims["loihi"].board

  # Get the blocks (which can be many depending on how large the Ensemble `ens`
  # is and in how many blocks is it broken).
  blocks = loihi_sim.model.objs[ens]
  #print("Number of (in and out) Blocks for Ensemble %s are: %s and %s."
  #          % (ens, len(blocks["in"]), len(blocks["out"])))
  for block in blocks["in"]:
    in_chip_idx, in_core_idx, in_block_idx, in_compartment_idxs, _ = (
        board.find_block(block))
    nxsdk_core = nxsdk_board.n2Chips[in_chip_idx].n2CoresAsList[in_core_idx]

    # Set the cxProfileCfg[0] as the leaf node's profile with `stackOut=3` =>
    # it pushes the current U to the top of the stack.
    nxsdk_core.cxProfileCfg[0].configure(stackOut=3, bapAction=0, refractDelay=0)
    # Set the cxProfileCfg[1] as the intermediate node's profile with `stackIn=2`
    # => it pops the element from the stack, `joinOp=2` => it does the MAX joinOp
    # with the popped element from stack and its current U, `stackOut=3` => it
    # pushes the MAXed current U on the top of the stack,
    # `decayU=nxsdk_core.cxProfileCfg[0].decayU` => the decay constant for current
    # U is same as that of the cxProfileCfg[0]. If `decayU` is 0, the current due
    # incoming spike never decays resulting in constant spiking of the neuron
    # and if it is default value, then the current decays instantly.
    nxsdk_core.cxProfileCfg[1].configure(
        stackIn=2, joinOp=2, stackOut=3, decayU=nxsdk_core.cxProfileCfg[0].decayU)
    # Set the root node which will output the spikes corresonding to the MAXed U.
    nxsdk_core.cxProfileCfg[2].configure(
        stackIn=2, joinOp=2, decayU=nxsdk_core.cxProfileCfg[0].decayU)

    # Set the compartments now.
    # Since the incoming connection from the previous Conv layer already as the
    # inputs in order of grouped slices, they are simply connected to the neuron
    # in this Ensembel `ens` from 0 index onwards.
    # `in_compartment_idxs` has the mapping of all compartment neurons in a
    # specific core, starting from index 0.

    # Maximum number of compartment idxs = 1024.
    for i in range(0, len(in_compartment_idxs), 4):
      c_idx = in_compartment_idxs[i]
      # Set a leaf node/compartment.
      nxsdk_core.cxCfg[c_idx].configure(cxProfile=0, vthProfile=0)
      # Set two intermediate nodes/compartments.
      nxsdk_core.cxCfg[c_idx+1].configure(cxProfile=1, vthProfile=0)
      nxsdk_core.cxCfg[c_idx+2].configure(cxProfile=1, vthProfile=0)
      # Set a root node/compartment to output spikes corresponding to MAX input.
      nxsdk_core.cxCfg[c_idx+3].configure(cxProfile=2, vthProfile=0)

def get_avam_net_for_2x2_max_pooling(seed=0, max_rate=250, radius=1, sf=1,
                                    do_max=True, synapse=0.005):
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
    # Disable operator merging to improve compilation time.
    #nengo_dl.configure_settings(planner=noop_planner)

    net.input = nengo.Node(size_in=4) # 4 dimensional input for 2x2 pooling.

    def _get_ensemble():
      ens = nengo.Ensemble(
          n_neurons=2, dimensions=1, encoders = [[1], [-1]], intercepts=[0, 0],
          max_rates=[max_rate, max_rate], radius=radius,
          neuron_type=nengo_loihi.neurons.LoihiSpikingRectifiedLinear(
          initial_state={"voltage": np.zeros(2)}
          ))
      return ens

    if do_max:
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
