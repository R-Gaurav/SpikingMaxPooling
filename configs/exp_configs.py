#
# Author: Ramashish Gaurav
#

import nengo
import nengo_loihi
import pathlib

from utils.consts.dir_consts import EXP_OTPT_DIR
from utils.consts.exp_consts import MNIST, CIFAR10, FMNIST, AVAM, MJOP, AVGP
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

model = MODEL_2
dataset = MNIST # One of MNIST, CIFAR10, FMNIST
is_channels_first = True # False for Model 7 (CIFAR10), and FMNIST ( all models)
sfr = 400 # Only for NengoDL. For NengoLoihi, it is set separately.

# FMNIST: False, MODEL_1 -> 24 | False, MODEL_2 -> 64

tf_exp_cfg = {
  "is_channels_first": is_channels_first,
  "batch_size": 100,
  "dataset": dataset,
  "epochs": 8 if dataset == MNIST else 64 if dataset == FMNIST else 164, # 64, 164
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
    "n_steps": 60, # in milliseconds. (40, 50 for MNIST) (50, 60 for CIFAR10)
    "n_test": 100, # Number of images to be tested in each batch.
    "scale": 2, # Scaling parameter of the output of root neurons. (MODEL_1)
    #"scale": 1.5, # Scaling parameter of the output of root neurons. (MODEL_2)

    "sfr": 400,
    "synapse": 0.005,
    "spk_neuron": nengo_loihi.neurons.LoihiSpikingRectifiedLinear(),
    "test_mode_res_otpt_dir": (
        EXP_OTPT_DIR + "/%s/%s/nengo_loihi_otpts/" % (dataset, model["name"]))
  },
  "layer_blockshapes": block_shapes[
      "channels_first" if is_channels_first else "channels_last"][dataset][
      model["name"]],
  "loihi_model_type": MJOP # One of AVAM, MJOP, AVGP
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
    "radius": 0.20,
    "synapse": 0.005,
    "sfr": sfr,
    "n_steps": 60, # 60, # 120 required for a deeper MODEL_7
    "test_batch_size": 10, # 10 for 120
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

pathlib.Path(tf_exp_cfg["tf_wts_otpt_dir"]).mkdir(parents=True, exist_ok=True)
pathlib.Path(tf_exp_cfg["tf_res_otpt_dir"]).mkdir(parents=True, exist_ok=True)
pathlib.Path(nengo_dl_cfg["test_mode"]["test_mode_res_otpt_dir"]).mkdir(
             parents=True, exist_ok=True)
pathlib.Path(nengo_dl_cfg["train_mode"]["ndl_train_mode_res_otpt_dir"]).mkdir(
             parents=True, exist_ok=True)
pathlib.Path(nengo_loihi_cfg["test_mode"]["test_mode_res_otpt_dir"]).mkdir(
    parents=True, exist_ok=True)
pathlib.Path(nengo_loihi_cfg["test_mode"]["test_mode_res_otpt_dir"]+"/"+AVAM
    ).mkdir(parents=True, exist_ok=True)
pathlib.Path(nengo_loihi_cfg["test_mode"]["test_mode_res_otpt_dir"]+"/"+MJOP
    ).mkdir(parents=True, exist_ok=True)
pathlib.Path(nengo_loihi_cfg["test_mode"]["test_mode_res_otpt_dir"]+"/"+AVGP
    ).mkdir(parents=True, exist_ok=True)
