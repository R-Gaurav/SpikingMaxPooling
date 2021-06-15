import _init_paths
import nengo
import nengo_loihi
import numpy as np

from utils.nengo_dl_utils import configure_ensemble_for_2x2_max_join_op
from utils.base_utils.exp_utils import get_grouped_slices_2d_pooling

#NUM_CHNLS, ROWS, COLS = 4, 32, 32
NUM_CHNLS, ROWS, COLS = 12, 26, 26
NUM_NEURONS = NUM_CHNLS * ROWS * COLS
np.random.seed(0)
input_matrix = np.random.rand(NUM_CHNLS, ROWS, COLS).flatten().tolist()
grouped_slices = get_grouped_slices_2d_pooling(
    pool_size=(2, 2), num_chnls=NUM_CHNLS, rows=ROWS, cols=COLS)
input_matrix = [input_matrix[i] for i in grouped_slices]

with nengo.Network() as net:
  inp_node = nengo.Node(input_matrix)
  otp_node = nengo.Node(size_in=NUM_NEURONS//4)
  ens = nengo.Ensemble(
      n_neurons=NUM_NEURONS,
      dimensions=1,
      gain=1000 * np.ones(NUM_NEURONS),
      bias=np.zeros(NUM_NEURONS),
      seed=0,
      neuron_type=nengo_loihi.neurons.LoihiSpikingRectifiedLinear(
          initial_state={"voltage": np.zeros(NUM_NEURONS)})
  )

  nengo.Connection(inp_node, ens.neurons, synapse=None)
  nengo.Connection(
      ens.neurons[[i for i in range(NUM_NEURONS) if i%4==3]], otp_node,
      synapse=0.005, transform=2/1000
  )
  max_probe = nengo.Probe(otp_node)

with net:
  nengo_loihi.add_params(net)
  net.config[ens].block_shape = nengo_loihi.BlockShape(
      (1, ROWS, COLS), (NUM_CHNLS, ROWS, COLS))

loihi_sim = nengo_loihi.Simulator(net)
configure_ensemble_for_2x2_max_join_op(
    loihi_sim, ens, pool_size=(2, 2), num_chnls=NUM_CHNLS, rows=ROWS, cols=COLS)

with loihi_sim:
  loihi_sim.run(0.100)

maxed_values = loihi_sim.data[max_probe]
np.save("Maxed_Values_NC_%s_R_%s_C_%s.npy" % (NUM_CHNLS, ROWS, COLS), maxed_values)
