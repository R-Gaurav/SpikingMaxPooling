import collections
import warnings
import os

import nengo
import nengo_dl
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import nengo_loihi

import _init_paths

from utils.base_utils.exp_utils import get_grouped_slices_2d_pooling_cf
from utils.nengo_dl_utils import configure_ensemble_for_2x2_max_join_op

# ignore NengoDL warning about no GPU
warnings.filterwarnings("ignore", message="No GPU", module="nengo_dl")

np.random.seed(0)
tf.random.set_seed(0)

################################################################################
# load mnist dataset
(train_images, train_labels), (
    test_images,
    test_labels,
) = tf.keras.datasets.mnist.load_data()

# flatten images and add time dimension
train_images = train_images.reshape((train_images.shape[0], 1, -1))
train_labels = train_labels.reshape((train_labels.shape[0], 1, -1))
test_images = test_images.reshape((test_images.shape[0], 1, -1))
test_labels = test_labels.reshape((test_labels.shape[0], 1, -1))

################################################################################
##################     DESIGN NETWORK     ################################
inp = tf.keras.Input(shape=(1, 28, 28), name="input")
#################### OFF-CHIP INPUT LAYER ###########################

to_spikes = tf.keras.layers.Conv2D(
  filters=3, # 3 RGB Neurons per pixel.
  kernel_size=(1, 1), strides=(1, 1), activation=tf.nn.relu,
  use_bias=False, # Default is True.
  data_format="channels_first", name="to-spikes")(inp)

################################ ON-CHIP LAYERS ##################################
conv0 = tf.keras.layers.Conv2D(
  filters=8, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu,
  use_bias=False, data_format="channels_first", name="conv0")(to_spikes)

maxp0 = tf.keras.layers.MaxPool2D(
  pool_size=(2, 2), padding="valid", data_format="channels_first",
  name="MaxPool0")(conv0)

conv1 = tf.keras.layers.Conv2D(
  filters=16, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu,
  use_bias=False, data_format="channels_first", name="conv1")(maxp0)

flatten = tf.keras.layers.Flatten(name="flatten")(conv1)
dense0 = tf.keras.layers.Dense(64, activation=tf.nn.relu, name="dense0")(flatten)

############################ OFF-CHIP OUTPUT LAYER #########################
# `Dense` layer with "softmax" results in following error:
# `nengo.exceptions.SimulationError: Cannot call TensorNode output function
# (this probably means you are trying to use a TensorNode inside a Simulator other
# than NengoDL)`. Therefore comment out the line.
# dense1 = tf.keras.layers.Dense(10, activation="softmax", name="dense1")(dense0)
dense1 = tf.keras.layers.Dense(10, name="dense1")(dense0)
model = tf.keras.Model(inputs=inp, outputs=dense1)
model.summary()

################################################################################
pres_time = 0.04  # how long to present each input, in seconds
n_test = 200  # how many images to test
SCALE = 1.1


# convert the keras model to a nengo network
nengo_converter = nengo_dl.Converter(
    model,
    scale_firing_rates=400,
    swap_activations={tf.nn.relu: nengo_loihi.neurons.LoihiSpikingRectifiedLinear()},
    synapse=0.005,
)
net = nengo_converter.net

# get input/output objects
nengo_input = nengo_converter.inputs[inp]
nengo_output = nengo_converter.outputs[dense1]

################################################################################
# build network, load in trained weights, save to network
with nengo_dl.Simulator(net) as nengo_sim:
    nengo_sim.load_params("attempting_TN_MP_loihineurons_8_16")
    #nengo_sim.load_params("ndl_trained_params")
    nengo_sim.freeze_params(net)

with net:
    nengo_input.output = nengo.processes.PresentInput(
        test_images, presentation_time=pres_time
    )
################################################################################
print(nengo_converter.model.layers)

with net:
    nengo_loihi.add_params(net)  # allow on_chip to be set
    # Both of the following lines works.
    #net.config[nengo_converter.layers[to_spikes].ensemble].on_chip = False
    net.config[nengo_converter.layers[nengo_converter.model.layers[1]].ensemble].on_chip = False

################################################################################
# Replace the TN MaxPooling with MAX joinOp based MaxPooling.
conn_from_pconv_to_max = net.all_connections[2]
conn_from_max_to_nconv = net.all_connections[3]
print("Conn from Prev Conv to MaxPooling: {}, Transform: {}, Synapse: {}, "
      "Function: {}".format(
      conn_from_pconv_to_max, conn_from_pconv_to_max.transform,
      conn_from_pconv_to_max.synapse, conn_from_pconv_to_max.function))
print("Conn from MaxPooling to Next Conv: {}, Transform: {}, Synapse: {}, "
      "Function: {}".format(
      conn_from_max_to_nconv, conn_from_max_to_nconv.transform,
      conn_from_max_to_nconv.synapse, conn_from_max_to_nconv.function))
print("Transform: {}".format(dir(conn_from_max_to_nconv.transform)))

NUM_CHNLS, ROWS, COLS = 8, 26, 26
grouped_slices = get_grouped_slices_2d_pooling_cf(
    pool_size=(2, 2), num_chnls=NUM_CHNLS, rows=ROWS, cols=COLS)
NUM_NEURONS = NUM_CHNLS * ROWS * COLS

############### CREATE ENSEMBLES ##############################
with net:
  ens = nengo.Ensemble(
      n_neurons=NUM_NEURONS,
      dimensions=1,
      gain=1000 * np.ones(NUM_NEURONS),
      bias=np.zeros(NUM_NEURONS),
      seed=0,
      neuron_type=nengo_loihi.neurons.LoihiSpikingRectifiedLinear(
          initial_state={"voltage": np.zeros(NUM_NEURONS)},
          amplitude=SCALE/1000 # Scaling is important.
      )
  )
#  passthrough_node = nengo.Node(size_in=NUM_NEURONS//4)
#  ens2 = nengo.Ensemble(
#      n_neurons=NUM_NEURONS//4,
#      dimensions=1,
#      gain=1000.0 * np.ones(NUM_NEURONS//4),
#      bias=np.zeros(NUM_NEURONS//4),
#      seed=0,
#      neuron_type=nengo_loihi.neurons.LoihiSpikingRectifiedLinear(
#          initial_state={"voltage": np.zeros(NUM_NEURONS//4)})
#  )

  ######### CONNECT THE PREV ENS/CONV TO MAX_JOINOP MAXPOOL ########
  nengo.Connection(
      conn_from_pconv_to_max.pre_obj[grouped_slices],
      ens.neurons,
      transform=None, #conn_from_pconv_to_max.transform,  #NoTransform.
      synapse=conn_from_pconv_to_max.synapse, # None.
      function=conn_from_pconv_to_max.function # None.
  )
  ######### CONNECT THE MAX_JOINOP MAXPOOL TO NEXT ENS/CONV ########
#  nengo.Connection(
#      ens.neurons[[i for i in range(NUM_NEURONS) if i%4==3]],
#      passthrough_node,
#      #ens2.neurons,
#      synapse=0.005,
#      #transform=1.1/1000
#      transform=nengo.Dense(
#          (NUM_NEURONS//4, NUM_NEURONS//4), 1.1/1000*np.eye(NUM_NEURONS//4))
#  )

  nengo.Connection(
      #passthrough_node,
      ens.neurons[[i for i in range(NUM_NEURONS) if i%4==3]],
      #ens2.neurons,
      conn_from_max_to_nconv.post_obj,
      transform=conn_from_max_to_nconv.transform,
      #synapse=conn_from_max_to_nconv.synapse,
      synapse=0.005, # Synapse required because the output from `ens` is spikes.
      #transform=1/1000,
      function=conn_from_max_to_nconv.function
  )

  ########## REMOVE THE OLD CONNECTIONS ############################
  net._connections.remove(conn_from_pconv_to_max)
  net._connections.remove(conn_from_max_to_nconv)
  net._nodes.remove(conn_from_pconv_to_max.post_obj)

#########################################################################
converter_layers = nengo_converter.model.layers
conv0_shape = tuple(converter_layers[2].output.shape[1:])
conv1_shape = tuple(converter_layers[4].output.shape[1:])
dense0_shape = tuple(converter_layers[6].output.shape[1:])

########### SET THE BLOCKSHAPES #######################
with net:
  net.config[ens].block_shape = nengo_loihi.BlockShape(
      (1, ROWS, COLS), (NUM_CHNLS, ROWS, COLS))
  net.config[
        nengo_converter.layers[conv0].ensemble
  ].block_shape = nengo_loihi.BlockShape((1, 26, 26), conv0_shape)

  net.config[
        nengo_converter.layers[conv1].ensemble
  ].block_shape = nengo_loihi.BlockShape((8, 11, 11), conv1_shape)

  # You don't necessarily need to partition the Dense blocks, therefore comment.
  #net.config[
  #      nengo_converter.layers[dense0].ensemble
  #].block_shape = nengo_loihi.BlockShape((32,), dense0_shape)

################# CHECK FOR ANY TENSORNODES ########################
for node in net._nodes:
  print("NODE: {}".format(node))
################################################################################
# build NengoLoihi Simulator and run network

#with nengo_loihi.Simulator(net, target="loihi", remove_passthrough=False) as loihi_sim:
with nengo_loihi.Simulator(net, target="loihi") as loihi_sim:
    configure_ensemble_for_2x2_max_join_op(loihi_sim, ens)
    loihi_sim.run(n_test * pres_time)

    # get output (last timestep of each presentation period)
    pres_steps = int(round(pres_time / loihi_sim.dt))
    output = loihi_sim.data[nengo_output][pres_steps - 1 :: pres_steps]

    # compute the Loihi accuracy
    loihi_predictions = np.argmax(output, axis=-1)
    correct = 100 * np.mean(loihi_predictions == test_labels[:n_test, 0, 0])
    print("Loihi accuracy: %.2f%%" % correct)
################################################################################
