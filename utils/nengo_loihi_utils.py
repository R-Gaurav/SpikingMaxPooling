import nengo
import nengo_loihi

from utils.base_utils import log
from utils.consts.exp_consts import SEED

def configure_ensemble_for_2x2_max_join_op(loihi_sim, ens):
  """
  Configures the Ensemble `ens` neurons for MaxPooling using NxSDK's MAX joinOp
  method.

  Args:
    loihi_sim <nengo_loihi.simulator.Simulator>: The NengoLoihi simulator object.
    ens <nengo.ensemble.Ensemble>: The Nengo Ensmeble object whose neurons are
                                   supposed to be configured for MaxPooling.

  Note: The number of neurons in `ens` should be equal to the number of neurons
        in the previous Convolutional ensemble, i.e. # neurons = total number
  values over which 2 x 2 MaxPooling is supposed to be done. Although, if the
  previous Convolutional layer has odd number of `rows` and `cols` then the actual
  number of neurons configured in the Ensemble `ens` will be equalt to:
  `num_chnls` x `rows-1` x `cols-1`.
  """
  nxsdk_board = loihi_sim.sims["loihi"].nxsdk_board
  board = loihi_sim.sims["loihi"].board

  # Get the blocks (which can be many depending on how large the Ensemble `ens`
  # is and in how many blocks is it broken).
  blocks = loihi_sim.model.objs[ens]
  log.INFO("Number of (in and out) Blocks for Ensemble %s are: %s and %s."
            % (ens, len(blocks["in"]), len(blocks["out"])))
  for block in blocks["in"]:
    in_chip_idx, in_core_idx, in_block_idx, in_compartment_idxs, _ = (
        board.find_block(block))
    nxsdk_core = nxsdk_board.n2Chips[in_chip_idx].n2CoresAsList[in_core_idx]
    # Set the cxProfileCfg in nxsdk_core. Leave vthProfileCfg unchanged.
    #log.INFO("For Core Index: {}, vthProfileCfg is: {}".format(
    #    in_core_idx, nxsdk_core.vthProfileCfg[0].staticCfg))
    #log.INFO("Before setting the cxProfileCfgs...")
    #log.INFO("For Core Index: {}, cxProfileCfg[0] is: {}".format(
    #    in_core_idx, nxsdk_core.cxProfileCfg[0]))
    #log.INFO("For Core Index: {}, cxProfileCfg[1] is: {}".format(
    #    in_core_idx, nxsdk_core.cxProfileCfg[1]))
    #log.INFO("For Core Index: {}, cxProfileCfg[2] is: {}".format(
    #    in_core_idx, nxsdk_core.cxProfileCfg[2]))

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

    #log.INFO("After setting the cxProfileCfgs...")
    #log.INFO("For Core Index: {}, cxProfileCfg[0] is: {}".format(
    #    in_core_idx, nxsdk_core.cxProfileCfg[0]))
    #log.INFO("For Core Index: {}, cxProfileCfg[1] is: {}".format(
    #    in_core_idx, nxsdk_core.cxProfileCfg[1]))
    #log.INFO("For Core Index: {}, cxProfileCfg[2] is: {}".format(
    #    in_core_idx, nxsdk_core.cxProfileCfg[2]))

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

def get_loihi_adapted_avam_net_for_2x2_max_pooling(
    seed=SEED, max_rate=1000, radius=0.5, sf=1, do_max=True, synapse=None):
  """
  Returns a Loihi adapted network for absolute value based associative max pooling.

  Args:
    seed <int>: Any arbitrary seed value.
    max_rate <int>: Max Firing rate of the neurons.
    radius <int>: Value at which Maximum Firing rate occurs (
                  i.e. the representational radius)
    sf <int>: The scale factor.
    do_max <bool>: Do MaxPooling if True else do AvgPooling.
    synapse <float>: Synapic time constant.
  """
  with nengo.Network(seed=seed) as net:
    net.inputs = nengo.Node(size_in=4) # 4 dimensional input for 2x2 MaxPooling.

    def _get_ensemble():
      ens =  nengo.Ensemble(
          n_neurons=2, dimensions=1, encoders=[[1], [-1]], interceps=[0, 0],
          max_rate=[max_rate, max_rate], radius=radius,
          neuron_type=nengo_loihi.neurons.SpikingRectifiedLinear())
      return ens

    ens_12 = _get_ensemble() # Ensemble for max(a, b).
    ens_34 = _get_ensemble() # Ensemble for max(c, d).
    ens_1234 = _get_ensemble() # Ensemble for max(max(a, b), max(c, d)).

    # Intermediate passthrough nodes for summing and outputting the result.
    node_12 = nengo.Node(size_in=1) # For max(a, b).
    node_34 = nengo.Node(size_in=1) # For max(a, b).
    net.output = nengo.Node(size_in=1) # For max(max(a, b), max(c, d)).

    ############################################################################
    # Calculate max(a, b) = (a+b)/2 + |a-b|/2.
    # Calculate (a+b)/2.
    nengo.Connection(net.inputs[0], node_12, synapse=None, transform=sf/2)
    nengo.Connection(net.inputs[1], node_12, synapse=None, transform=sf/2)

    if do_max:
      # Calculate |a-b|/2.
      nengo.Connection(net.inputs[0], ens_12, synapse=None, transform=sf/2)
      nengo.Connection(net.inputs[1], ens_12, synapse=None, transform=-sf/2)
      nengo.Connection(
          ens_12.neurons[0], node_12, synapse=synapse, transform=radius/max_rate)
      nengo.Connection(
          ens_12.neurons[1], node_12, synapse=synapse, transform=radius/max_rate)
    ############################################################################

    ############################################################################
    # Calculate max(c, d) = (c+d)/2 + |c-d|/2.
    # Calculate (c+d)/2.
    nengo.Connection(net.inputs[2], node_34, synapse=None, transform=sf/2)
    nengo.Connection(net.inputs[3], node_34, synapse=None, transform=sf/2)

    if do_max:
      # Calculate |c-d|/2.
      nengo.Connection(net.inputs[2], ens_34, synapse=None, transform=sf/2)
      nengo.Connection(net.inputs[3], ens_34, synapse=None, transform=-sf/2)
      nengo.Connection(
          ens_34.neurons[0], node_34, synapse=synapse, transform=radius/max_rate)
      nengo.Connection(
          ens_34.neurons[1], node_34, synapse=synapse, transform=radius/max_rate)
    ############################################################################

    ############################################################################
    # Calculate max(a, b, c, d) = max(max(a, b), max(c, d)).
    # Calculate (node_12 + node_34)/2.
    nengo.Connection(node_12, net.output, synapse=synapse, transform=sf/2)
    nengo.Connection(node_34, net.output, synapse=synapse, transform=sf/2)

    if do_max:
      # Calculate |node_12 - node_34|/2.
      nengo.Connection(node_12, ens_1234, synapse=synapse, transform=sf/2)
      nengo.Connection(node_34, ens_1234, synapse=synapse, transform=-sf/2)
      nengo.Connection(ens_1234.neurons[0], net.output, synapse=synapse,
                       transform=radius/max_rate)
      nengo.Connection(ens_1234.neurons[1], net.output, synapse=synapse,
                       transform=radius/max_rate)
    ############################################################################

  return net
