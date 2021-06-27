import nengo
import nengo_loihi

from utils.base_utils import log

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
