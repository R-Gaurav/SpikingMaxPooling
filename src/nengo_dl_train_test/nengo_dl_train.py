#
# This file trains a model with NengoDL Simulator.
#
# Author: Ramashish Gaurav
#

import datetime
import nengo
import nengo_dl
import numpy as np
import random
import tensorflow as tf

import _init_paths

from configs.exp_configs import tf_exp_cfg as tf_cfg, nengo_dl_cfg as ndl_cfg
from utils.base_utils import log
from utils.base_utils.data_prep_utils import get_exp_dataset
from utils.nengo_dl_utils import get_nengo_dl_model, percentile_l2_loss_range
from utils.consts.exp_consts import SEED, MNIST, CIFAR10

# Set the SEED.
tf.random.set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

def nengo_dl_train():
  """
  Trains a NengoDL Model.
  """
  log.INFO("TF EXP CONFIG: %s" % tf_cfg)
  log.INFO("NENGO DL EXP CONFIG: %s" % ndl_cfg)
  ##############################################################################
  log.INFO("Getting the dataset: %s" % tf_cfg["dataset"])
  train_x, train_y, _, _ = get_exp_dataset(tf_cfg["dataset"])
  # Flatten the `train_x` images and the `train_y` labels.
  train_x = train_x.reshape((train_x.shape[0], 1, -1))
  train_y = train_y.reshape((train_y.shape[0], 1, -1))

  if tf_cfg["dataset"] == MNIST:
    inpt_shape = (1, 28, 28)
    num_clss = 10
  elif tf_cfg["dataset"] == CIFAR10:
    inpt_shape = (3, 32, 32)
    num_clss = 10
  ##############################################################################
  log.INFO("Getting the NengoDL model to be trained...:")
  ndl_model, ndl_mdl_probes = get_nengo_dl_model(
      inpt_shape, tf_cfg, ndl_cfg, mode="train", num_clss=num_clss)
  ndl_train_cfg = ndl_cfg["train_mode"]
  # TODO: Is there a need to set the following? May be when training neuron is Spiking Neuron?
  # with tf.keras.backend.learning_phase_scope(1), nengo_dl.Simulator(
  log.INFO("Creating the NengoDL Simulator...")
  with nengo_dl.Simulator(
      ndl_model.net, minibatch_size=ndl_train_cfg["train_batch_size"],
      seed=SEED, progress_bar=False) as ndl_sim:
    losses={
      # Last probe is the Output Layer Probe.
      ndl_mdl_probes[-1]: tf.losses.CategoricalCrossentropy(from_logits=True)
    }
    # Regularize the firing rates activity of Ensemble neurons.
    #for probe in ndl_mdl_probes[1:-1]:
    #  losses[probe] = percentile_l2_loss_range

    ndl_sim.compile(
        optimizer=tf.optimizers.Adam(tf_cfg["lr"]),
        loss=losses,
        metrics=["accuracy"]
    )
    log.INFO("Training the model...")
    ndl_sim.fit(
      {ndl_mdl_probes[0]: train_x},
      {ndl_mdl_probes[-1]: train_y},
      epochs=tf_cfg["epochs"]
    )
    log.INFO("Saving the trained model-parameters...")
    ndl_sim.save_params(
        ndl_train_cfg["ndl_train_mode_res_otpt_dir"]+"/ndl_trained_params")

  log.INFO("NengoDL Training Done!")

if __name__ == "__main__":
  log.configure_log_handler(
      "%s_sfr_%s_epochs_%s_timestamp_%s_.log" % (
      ndl_cfg["train_mode"]["ndl_train_mode_res_otpt_dir"] + "_nengo_dl_train_",
      ndl_cfg["train_mode"]["sfr"], tf_cfg["epochs"], datetime.datetime.now()))
  nengo_dl_train()
