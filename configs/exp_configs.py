#
# Author: Ramashish Gaurav
#

import nengo

from utils.consts.dir_consts import EXP_OTPT_DIR
from utils.consts.model_consts import MODEL_1

tf_exp_cfg = {
  "rf": 5e-5,
  "batch_size": 64,
  "epochs": 16,
  "lr": 1e-4,
  "nn_dlyr": 512,
  "tf_model": MODEL_1,
  "tf_wts_dir": EXP_OTPT_DIR + "/cifar10/%s/weights" % MODEL_1["name"],
}

nengo_dl_cfg = {
  "spk_neuron": nengo.SpikingRectifiedLinear(),
  "synapse": 0.005,
  "sfr": 600,
}
