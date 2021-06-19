#
# Author: Ramashish Gaurav
#

import nengo
import nengo_loihi
import pathlib

from utils.consts.dir_consts import EXP_OTPT_DIR
from utils.consts.exp_consts import MNIST, CIFAR10
from utils.consts.model_consts import MODEL_1, MODEL_2, MODEL_3

# The TF train/test and Nengo-DL test variations are only with `model` and
# `dataset`, however, in case of Nengo-DL test, one can include variations with
# `sfr` and `n_steps`. The `synapse` and `spk_neuron` is (mostly) kept unchanged.

# In case of Nengo-DL train/test, one include variations apart from `model` and
# `dataset`, with `sfr` and "percentile regularization of firing rates" during
# training. And during test, the same `sfr` with different `n_steps` could be
# used. Again, the `synapse` and `spk_neuron` is (mostly) kept unchanged.

model = MODEL_2
dataset = MNIST # One of MNIST, CIFAR10

tf_exp_cfg = {
  "batch_size": 200,
  "dataset": dataset,
  "epochs": 16,
  "lr": 1e-3,
  "nn_dlyr": 64,
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
    "n_steps": 40, # in milliseconds.
    "n_test": 100, # Number of images to be tested.
    "scale": 1.1, # Scaling parameter of the output of root neurons.
    "sfr": 400,
    "synapse": 0.005,
    "spk_neuron": nengo_loihi.neurons.LoihiSpikingRectifiedLinear(),
    "test_mode_res_otpt_dir": (
        EXP_OTPT_DIR + "/%s/%s/nengo_loihi_otpts/" % (dataset, model["name"]))
  },
  "layer_blockshapes": {
    "model_1": {
      "conv2d_0": (1, 26, 26),
      "conv2d_1": (8, 11, 11),
    }
  },
}

nengo_dl_cfg = {
  "dataset": dataset,
  "tf_wts_inpt_dir": (
      EXP_OTPT_DIR + "/%s/%s/tf_otpts/tf_trained_wts/weights"
      % (dataset, model["name"])),
  "test_mode": {
    "spk_neuron": nengo.SpikingRectifiedLinear(),
    "synapse": 0.005,
    "sfr": 25,
    "n_steps": 60,
    "test_batch_size": 100,
    "test_mode_res_otpt_dir": (
        EXP_OTPT_DIR + "/%s/%s/tf_otpts/ndl_test_only_results/"
        % (dataset, model["name"])),
    # Timestep after which MAX_POOL_MASK will not be updated, rather the learned
    # mask up till this timestep will determine the maximally firing neuron.
    "skip_isi_tstep": 60,
  },
  "train_mode": {
    #"neuron": nengo.RectifiedLinear(),
    "neuron": nengo_loihi.neurons.LoihiSpikingRectifiedLinear(),
    "synapse": None,
    "sfr": 100,
    "n_steps": 1,
    "train_batch_size": 200,
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
