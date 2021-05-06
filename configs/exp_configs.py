#
# Author: Ramashish Gaurav
#

import nengo
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

model = MODEL_3
dataset = MNIST # One of MNIST, CIFAR10

tf_exp_cfg = {
  "batch_size": 200,
  "dataset": dataset,
  "epochs": 20,
  "lr": 1e-3,
  "nn_dlyr": 64,
  "tf_model": model,
  "tf_res_otpt_dir": EXP_OTPT_DIR + "/%s/%s/tf_otpts/" % (dataset, model["name"]),
  "tf_wts_otpt_dir": (
      EXP_OTPT_DIR + "/%s/%s/tf_otpts/tf_trained_wts/" % (dataset, model["name"])),
}

nengo_dl_cfg = {
  "dataset": dataset,
  "tf_wts_inpt_dir": (
      EXP_OTPT_DIR + "/%s/%s/tf_otpts/tf_trained_wts/weights"
      % (dataset, model["name"])),
  "test_mode": {
    "spk_neuron": nengo.SpikingRectifiedLinear(),
    "synapse": 0.005,
    "sfr": 100,
    "n_steps": 60,
    "test_batch_size": 100,
    "ndl_test_mode_res_otpt_dir": (
        EXP_OTPT_DIR + "/%s/%s/tf_otpts/ndl_test_only_results/"
        % (dataset, model["name"])),
  },
  "train_mode": {
    "neuron": nengo.RectifiedLinear(),
    "synapse": None,
    "sfr": 1,
    "n_steps": 1,
    "train_batch_size": 200,
    "ndl_train_mode_res_otpt_dir": (
        EXP_OTPT_DIR + "/%s/%s/ndl_train_test_results/" % (dataset, model["name"])),
  }
}

pathlib.Path(tf_exp_cfg["tf_wts_otpt_dir"]).mkdir(parents=True, exist_ok=True)
pathlib.Path(tf_exp_cfg["tf_res_otpt_dir"]).mkdir(parents=True, exist_ok=True)
pathlib.Path(nengo_dl_cfg["test_mode"]["ndl_test_mode_res_otpt_dir"]).mkdir(
             parents=True, exist_ok=True)
pathlib.Path(nengo_dl_cfg["train_mode"]["ndl_train_mode_res_otpt_dir"]).mkdir(
             parents=True, exist_ok=True)
