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
from utils.base_utils.data_prep_utils import (
    get_exp_dataset, get_batches_of_exp_dataset)
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
  num_imgs = train_x.shape[0]

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
  train_bs = ndl_train_cfg["train_batch_size"]
  # TODO: Is there a need to set the following? May be when training neuron is Spiking Neuron?
  # with tf.keras.backend.learning_phase_scope(1), nengo_dl.Simulator(
  log.INFO("Creating the NengoDL Simulator...")
  with nengo_dl.Simulator(ndl_model.net, minibatch_size=train_bs, seed=SEED,
                          progress_bar=False) as ndl_sim:
    losses={
      # Last probe is the Output Layer Probe.
      ndl_mdl_probes[-1]: tf.losses.CategoricalCrossentropy(from_logits=True)
    }

    ndl_sim.compile(
        optimizer=tf.optimizers.Adam(learning_rate=tf_cfg["lr"], decay=1e-4),
        loss=losses,
        metrics=["accuracy"]
    )
    log.INFO("Training the model...")
    #ndl_sim.fit(
    #  {ndl_mdl_probes[0]: train_x},
    #  {ndl_mdl_probes[-1]: train_y},
    #  epochs=tf_cfg["epochs"]
    #)
    for epoch in range(tf_cfg["epochs"]):
      log.INFO("Executing Epoch: %s ..." % epoch)
      batches = get_batches_of_exp_dataset(ndl_cfg, is_test=False)
      ndl_sim.fit(batches, epochs=1, steps_per_epoch=num_imgs // train_bs)

    log.INFO("Saving the trained model-parameters...")
    ndl_sim.save_params(
        ndl_train_cfg["ndl_train_mode_res_otpt_dir"]+"/ndl_trained_params")

  log.INFO("NengoDL Training Done!")

def nengo_dl_test():
  """
  Does Nengo-DL testing with TensorNode MaxPooling.
  """
  if tf_cfg["dataset"] == MNIST:
    inpt_shape = (1, 28, 28)
    num_clss = 10
  elif tf_cfg["dataset"] == CIFAR10:
    inpt_shape = (3, 32, 32)
    num_clss = 10

  log.INFO("Getting the Nengo-DL model...")
  ndl_model, ngo_probes_lst = get_nengo_dl_model(
      inpt_shape, tf_cfg, ndl_cfg, mode="test", num_clss=num_clss,
      max_to_avg_pool=False)
  log.INFO("Getting the dataset: %s" % ndl_cfg["dataset"])
  test_batches = get_batches_of_exp_dataset(ndl_cfg, is_test=True)
  log.INFO("Start testing...")
  with nengo_dl.Simulator(
      ndl_model.net, minibatch_size=ndl_cfg["test_mode"]["test_batch_size"],
      progress_bar=False) as sim:
    sim.load_params(ndl_cfg["trained_model_params"]+ "/ndl_trained_params")

    acc, n_test_imgs = 0, 0
    for batch in test_batches:
      sim_data = sim.predict_on_batch({ngo_probes_lst[0]: batch[0]})
      for true_lbl, pred_lbl in zip(batch[1], sim_data[ngo_probes_lst[-1]]):
        if np.argmax(true_lbl) == np.argmax(pred_lbl[-1]):
          acc += 1
        n_test_imgs += 1
        # TODO: Collect the intermediate layers spike/synapsed output.

    log.INFO("Testing done! Writing test accuracy results in log...")
    log.INFO("Nengo-DL Test Accuracy: %s" % (acc/n_test_imgs))
    #TODO: Delete the `ndl_model` to reclaim GPU memory.
    log.INFO("*"*100)

if __name__ == "__main__":
  log.configure_log_handler(
      "%s_sfr_%s_epochs_%s_timestamp_%s_.log" % (
      ndl_cfg["train_mode"]["ndl_train_mode_res_otpt_dir"] + "_nengo_dl_train_",
      ndl_cfg["train_mode"]["sfr"], tf_cfg["epochs"], datetime.datetime.now()))
  nengo_dl_train()
  nengo_dl_test()
