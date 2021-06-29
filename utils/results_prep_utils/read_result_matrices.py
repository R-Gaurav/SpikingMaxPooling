#
# Author: Ramashish Gaurav
#

import _init_paths

import numpy as np
import os

from utils.consts.dir_consts import EXP_OTPT_DIR
from utils.base_utils.data_prep_utils import get_exp_dataset

def read_nengo_loihi_results(dataset, model_name):
  """
  Args:
    dataset <str>: One of MNIST or CIFAR10.
    model_name <str>: Experiment model's name.
  """
  results_dir = EXP_OTPT_DIR + "/%s/%s/nengo_loihi_otpts/" % (dataset, model_name)
  files = os.listdir(results_dir)
  _, _, _, test_y = get_exp_dataset(dataset)
  test_img_idcs_set = set(np.arange(0, test_y.shape[0]))
  matched, total = 0, 0

  for f in files:
    if f.endswith("npy"):
      res = np.load(results_dir + f, allow_pickle=True)
      start_idx, end_idx = int(f.split("_")[5]), int(f.split("_")[7].split(".")[0])
      matched += np.sum(np.argmax(test_y[start_idx:end_idx], axis=-1) == res[1])
      total += (end_idx - start_idx)

      img_idcs_set = set(np.arange(start_idx, end_idx))
      test_img_idcs_set -= img_idcs_set

  print(matched/total, test_img_idcs_set)
