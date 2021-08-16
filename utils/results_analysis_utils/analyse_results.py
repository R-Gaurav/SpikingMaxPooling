#
# Author: Ramashish Gaurav
#

import _init_paths

import numpy as np
import os

from collections import Counter

from utils.consts.dir_consts import EXP_OTPT_DIR
from utils.base_utils.data_prep_utils import get_exp_dataset

def read_nengo_loihi_results(dataset, model_name, directory):
  """
  Returns the accuracy over all the batches of NengoLoihi predicted classes in
  files: Acc_and_preds_batch_start_<start_idx>_end_<end_idx>.npy. Also returns
  the test img indices for which no predicted classes were obtained.

  Args:
    dataset <str>: One of MNIST or CIFAR10.
    model_name <str>: Experiment model's name.
    directory <str>: One of "scale_x_y" where x and y are integers.
  """
  results_dir = (EXP_OTPT_DIR + "/%s/%s/nengo_loihi_otpts/max_join_op/%s/" % (
                 dataset, model_name, directory))
  files = os.listdir(results_dir)
  _, _, _, test_y = get_exp_dataset(dataset)
  test_img_idcs_set = set(np.arange(0, test_y.shape[0]))
  matched, total = 0, 0

  for f in files:
    if f.startswith("Acc") and f.endswith("npy"):
      res = np.load(results_dir + f, allow_pickle=True)
      # f = "Acc_and_preds_batch_start_0_end_75.npy"
      start_idx, end_idx = int(f.split("_")[5]), int(f.split("_")[7].split(".")[0])
      matched += np.sum(np.argmax(test_y[start_idx:end_idx], axis=-1) == res[1])
      total += (end_idx - start_idx)

      img_idcs_set = set(np.arange(start_idx, end_idx))
      test_img_idcs_set -= img_idcs_set

  return (matched/total, test_img_idcs_set)

def get_majority_voted_class(class_otpt_matrix, last_tsteps):
  """
  Returns the predicted classes based on majority voting over `last_tsteps`.

  Args:
    class_otpt_matrix <numpy.ndarray>: The Simulation's predicted class logit
                                       score matrix: (n_test x n_steps x num_clss).
    last_tsteps <int>: Number of last timesteps considered for majority voting.

  Returns:
    [int]: List of predicted classes (by majority voting).
  """
  pred_clss = []
  class_otpt_matrix = np.argmax(class_otpt_matrix, axis=-1)
  num_imgs = class_otpt_matrix.shape[0]
  for i in range(num_imgs):
    pred_clss.append(
        Counter(class_otpt_matrix[i][-last_tsteps:]).most_common(1)[0][0])
  return pred_clss

def get_accuracy_via_majority_voting(class_otpt_matrix, last_tsteps, dataset,
                                     start_idx=None, end_idx=None):
  """
  Returns the accuracy for the `ntest` images for the `dataset` based on majority
  voting over `last_tsteps`.

  Args:
    class_otpt_matrix <numpy.ndarray>: The Simulation's predicted class logit
                                       score matrix: (n_test x n_steps x num_clss).
    last_tsteps <int>: Number of last timesteps considered for majority voting.
    dataset <str>: One of "mnist" or "cifar10".
    start_idx <int>: The start index (inclusive) of the test dataset.
    end_idx <int>: The end index (exclusive) of the test dataset

  Returns:
    float: Accuracy.
  """
  _, _, _, test_y = get_exp_dataset(dataset, start_idx=start_idx, end_idx=end_idx)
  # Make sure last dimension is number of classes.
  assert class_otpt_matrix.shape[-1] == test_y.shape[1]
  # Make sure that the number of rows in class_otpt_matrix = number of test imgs.
  assert class_otpt_matrix.shape[0] == test_y.shape[0]
  pred_clss = get_majority_voted_class(class_otpt_matrix, last_tsteps)
  return 100 * np.mean(pred_clss == np.argmax(test_y, axis=-1))

def get_max_pool_otpt_indices(mp_otpt_matrix):
  """
  Returns a list of indices of the max_pool output matrix which has non zero
  output in any of the simulation timestep.

  Args:
    mp_otpt_matrix <np.ndarray>: NengoDL MaxPool output matrix of shape:
                                 (num_test_images x n_steps x MP output flattend).
  Returns:
    [int]: List of MP output indices which has an output in any timestep.
  """
  mp_ti_otpt_idcs = {}
  test_imgs, _, otpt_size = mp_otpt_matrix.shape
  for ti_idx in range(test_imgs):
    ti_non_zero_otpt = []
    for otp_idx in range(otpt_size):
      if np.any(mp_otpt_matrix[ti_idx, :, otp_idx]):
        ti_non_zero_otpt.append(otp_idx)
    mp_ti_otpt_idcs[ti_idx] = ti_non_zero_otpt

  return mp_ti_otpt_idcs

def get_loihi_probes_output(probe_otpt, pres_steps):
  """
  Returns a list of len: num_test_images where each element is an array of shape:
  (pres_steps x num_neurons) for the passed probe output `probe_otpt`.

  Args:
    probe_otpt <numpy.ndarray>: Object of neurons output.
    pres_steps <int>: Presentation time steps.
  Returns:
    [np.ndarray]
  """
  dikt = probe_otpt.item()
  layer_keys, ret = dikt.keys(), {}

  for layer in layer_keys:
    start, all_tsteps, lst = 0, np.arange(0, len(dikt[layer])), []
    img_wise_end_tsteps = all_tsteps[pres_steps-1 :: pres_steps]
    for end_tstep in img_wise_end_tsteps:
      lst.append(np.array(dikt[layer][start : end_tstep+1]))
      start = end_tstep + 1
    ret[layer] = lst

  return ret

def get_layer_probes_output_dict(dataset, model_name, start_idx, end_idx, n_steps,
                                 directory):
  """
  Constructs output matrix of shape (num_imgs, n_steps, num_neurons)
  """
  results_dir = (EXP_OTPT_DIR + "/%s/%s/nengo_loihi_otpts/max_join_op/%s/" % (
                 dataset, model_name, directory))
  f = np.load(results_dir + "Layer_probes_otpt_batch_start_%s_end_%s.npy" % (
              start_idx, end_idx), allow_pickle=True)
  prb_otpt = get_loihi_probes_output(f, n_steps)
  for layer in prb_otpt.keys():
    prb_otpt[layer] = np.array(prb_otpt[layer])

  return prb_otpt
