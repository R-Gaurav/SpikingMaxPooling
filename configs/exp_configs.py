#
# Author: Ramashish Gaurav
#

import nengo

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
  "tf_wts_otpt_dir": EXP_OTPT_DIR + "/cifar10/%s/" % model["name"],
}

nengo_dl_cfg = {
  "tf_wts_inpt_dir": EXP_OTPT_DIR + "/cifar10/%s/weights" % model["name"],
  "spk_neuron": nengo.SpikingRectifiedLinear(),
  "synapse": 0.005,
  "sfr": 600,
  "n_steps": 80,
  "test_batch_size": 24,
  "train_batch_size": 16,
}
