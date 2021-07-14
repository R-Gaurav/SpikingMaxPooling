#
# Author: Ramashish Gaurav
#

import nengo
import nengo_loihi
import pathlib

from utils.consts.dir_consts import EXP_OTPT_DIR
from utils.consts.exp_consts import MNIST, CIFAR10
from utils.consts.model_consts import (
    MODEL_1, MODEL_2, MODEL_3, MODEL_4, MODEL_5, MODEL_6, MODEL_7,
    MODEL_1_AP, MODEL_2_AP, MODEL_ALL_CONV)
from .block_configs import block_shapes

# The TF train/test and Nengo-DL test variations are only with `model` and
# `dataset`, however, in case of Nengo-DL test, one can include variations with
# `sfr` and `n_steps`. The `synapse` and `spk_neuron` is (mostly) kept unchanged.

# In case of Nengo-DL train/test, one include variations apart from `model` and
# `dataset`, with `sfr` and "percentile regularization of firing rates" during
# training. And during test, the same `sfr` with different `n_steps` could be
# used. Again, the `synapse` and `spk_neuron` is (mostly) kept unchanged.

model = MODEL_1
dataset = MNIST # One of MNIST, CIFAR10
is_channels_first = True
sfr = 400 # Only for NengoDL. For NengoLoihi, it is set separately.

tf_exp_cfg = {
  "is_channels_first": is_channels_first,
  "batch_size": 100,
  "dataset": dataset,
  "epochs": 6 if dataset == MNIST else 32, #160,
  "lr": 1e-3,
  "nn_dlyr": 128,
  "tf_model": model,
  "tf_res_otpt_dir": EXP_OTPT_DIR + "/%s/%s/tf_otpts/" % (dataset, model["name"]),
  "tf_wts_otpt_dir": (
      EXP_OTPT_DIR + "/%s/%s/tf_otpts/tf_trained_wts/" % (dataset, model["name"])),
}

nengo_loihi_cfg = {
  "dataset": dataset,
  "trained_model_params": (
      EXP_OTPT_DIR + "/%s/%s/ndl_train_test_results/" % (dataset, model["name"])),
  "test_mode": {
    "n_steps": 100, # in milliseconds.
    "n_test": 100, # Number of images to be tested.
    "scale": 10, # Scaling parameter of the output of root neurons. (MODEL_1)
    # "scale": 1.2, # Scaling parameter of the output of root neurons. (MODEL_2)
    ################# WITH MODEL_1 and MNIST ###########################
    # scale=1.2 => 97.2
    # scale=1.1 => 96.8 on first 250 images. 97.2
    # scale=1.0 => 97.2 on first 250 images. 97.6
    # scale=0.9 => 98.0 on first 250 images. 97.6
    # scale=0.8 => 97.2 on first 250 images.
    # scale=0.85 => 97.2 on first 250 images.
    # scale=0.95 => 97.2 on first 250 images.
    ################## WITH MODEL_2 and MNIST ###########################

    "sfr": 400,
    "synapse": 0.005,
    "spk_neuron": nengo_loihi.neurons.LoihiSpikingRectifiedLinear(),
    "test_mode_res_otpt_dir": (
        EXP_OTPT_DIR + "/%s/%s/nengo_loihi_otpts/" % (dataset, model["name"]))
  },
  "layer_blockshapes": block_shapes[
      "channels_first" if is_channels_first else "channels_last"][dataset][
      model["name"]],
}

nengo_dl_cfg = {
  "dataset": dataset,
  "tf_wts_inpt_dir": (
      EXP_OTPT_DIR + "/%s/%s/tf_otpts/tf_trained_wts/weights"
      % (dataset, model["name"])),
  "trained_model_params": (
      EXP_OTPT_DIR + "/%s/%s/ndl_train_test_results/" % (dataset, model["name"])),
  "test_mode": {
    "spk_neuron": nengo_loihi.neurons.LoihiSpikingRectifiedLinear(),
    "synapse": 0.005,
    "sfr": sfr,
    "n_steps": 50, # 80 required for a deeper MODEL_7
    "test_batch_size": 100,
    "test_mode_res_otpt_dir": (
        EXP_OTPT_DIR + "/%s/%s/ndl_train_test_results/ndl_test_only_results/"
        % (dataset, model["name"])),
    # Timestep after which MAX_POOL_MASK will not be updated, rather the learned
    # mask up till this timestep will determine the maximally firing neuron.
    "skip_isi_tstep": 60,
  },
  "train_mode": {
    "neuron": nengo_loihi.neurons.LoihiSpikingRectifiedLinear(),
    "synapse": None,
    "sfr": sfr,
    "n_steps": 1,
    "train_batch_size": 100,
    "ndl_train_mode_res_otpt_dir": (
        EXP_OTPT_DIR + "/%s/%s/ndl_train_test_results/" % (dataset, model["name"])),
  }
}

asctv_max_cfg = {
    "conv2d.0": {"max_rate": 250, "radius": 3, "sf": 1.2, "synapse": 0.001},
    "conv2d_1.0": {"max_rate": 250, "radius": 2, "sf": 1.2, "synapse": 0.001},
    "conv2d_2.0": {"max_rate": 250, "radius": 1.5, "sf": 1.2, "synapse": 0.001},
    "conv2d_3.0": {"max_rate": 250, "radius": 1, "sf": 1.2, "synapse": 0.001},
}

pathlib.Path(tf_exp_cfg["tf_wts_otpt_dir"]).mkdir(parents=True, exist_ok=True)
pathlib.Path(tf_exp_cfg["tf_res_otpt_dir"]).mkdir(parents=True, exist_ok=True)
pathlib.Path(nengo_dl_cfg["test_mode"]["test_mode_res_otpt_dir"]).mkdir(
             parents=True, exist_ok=True)
pathlib.Path(nengo_dl_cfg["train_mode"]["ndl_train_mode_res_otpt_dir"]).mkdir(
             parents=True, exist_ok=True)
pathlib.Path(nengo_loihi_cfg["test_mode"]["test_mode_res_otpt_dir"]).mkdir(
    parents=True, exist_ok=True)
