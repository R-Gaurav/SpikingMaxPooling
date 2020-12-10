#
# Author: Ramashish Gaurav
#

import nengo
import pathlib

from utils.consts.dir_consts import EXP_OTPT_DIR
from utils.consts.model_consts import MODEL_1

model = MODEL_1

tf_exp_cfg = {
  "rf": 5e-5,
  "batch_size": 64,
  "epochs": 16,
  "lr": 1e-4,
  "nn_dlyr": 512,
  "tf_model": model,
  "tf_wts_otpt_dir": EXP_OTPT_DIR + "/cifar10/%s/tf_trained_wts/" % model["name"],
}

nengo_dl_cfg = {
  "tf_wts_inpt_dir": (
      EXP_OTPT_DIR + "/cifar10/%s/tf_trained_wts/weights" % model["name"]),
  "ndl_res_otpt_dir": EXP_OTPT_DIR + "/cifar10/%s/ndl_relu_results/" % model["name"],
  "spk_neuron": nengo.RectifiedLinear(),
  "synapse": None,
  "sfr": 1, # 600
  "n_steps": 1, # 80
  "test_batch_size": 100,
  "train_batch_size": 16,
}

pathlib.Path(nengo_dl_cfg["ndl_res_otpt_dir"]).mkdir(parents=True, exist_ok=True)
pathlib.Path(tf_exp_cfg["tf_wts_otpt_dir"]).mkdir(parents=True, exist_ok=True)
