import ray
from ray.experimental import tqdm_ray


import random
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import Self

from numba import njit

# from scipy.ndimage import convolve1d
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut, cross_val_score
from dataclasses import dataclass, replace

from admice.eventlda import FilteringStrategy, NeuronSamplingStrategy
from admice.dataload import Session


remote_tqdm = ray.remote(tqdm_ray.tqdm)


def pick_pos_neg_data_sliding(
  filt_nrt: np.ndarray,
  event_ind: int,
  filt_strat: FilteringStrategy,
  seed: int = 42,
  valid_neg_mask_tt: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  Extracts positive and negative datapoints for each trial, at each positive time

  Args:
      filt_nrt (np.ndarray): filtered input data
      event_ind (int): used to exclude negative timepoints after event
      exclude_window_width (int): averaging window width used for filtering
      seed (int, optional): random seed. Defaults to 42.
      valid_neg_mask_tt (pass back in to save time)

  Returns:
      pos_data_nrt, neg_data_nrt, valid_neg_mask_tt
  """

  if valid_neg_mask_tt is None:
    window_width_pre = filt_strat.window_width_pre
    window_buffer_pre = filt_strat.window_buffer_pre

    # pick random pre-tone, non-overlapping indices for each trial for neg timepoint
    N, R, T = filt_nrt.shape

    # valid mask is T(as pos_ind) x T(as eligible) mask of eligible timepoints for each trial for the neg sample

    # disallow idx where the filtered data is missing
    valid_neg_mask_tt = np.zeros((T, T), dtype=np.int32)
    valid_neg_mask_tt[...] = np.logical_not(np.isnan(filt_nrt[0, 0]))

    # disallow idx which overlap with the positive sample (first index)'s averaging window
    exclude_window_width = window_width_pre + window_buffer_pre
    for t in range(T):
      valid_neg_mask_tt[t, t : min(t + exclude_window_width, T)] = False

    # disallow idx after the tone
    valid_neg_mask_tt[:, event_ind:] = False

  pos_data = filt_nrt

  @njit
  def choose_neg_data(filt_nrt: np.ndarray, seed: int):
    # sample the negative timepoint in each trial, for each pos_ind timepoint
    random.seed(seed)
    _, R, T = filt_nrt.shape
    neg_data = np.zeros_like(filt_nrt)
    for t in range(T):
      valid_indices = np.where(valid_neg_mask_tt[t, :])[0]
      for r in range(R):
        which_ind = random.randrange(len(valid_indices))
        neg_time_idx = valid_indices[which_ind]
        neg_data[:, r, t] = filt_nrt[:, r, neg_time_idx]

    return neg_data

  return pos_data, choose_neg_data(filt_nrt, seed), valid_neg_mask_tt


def lda_accuracy(pos_nr: np.ndarray, neg_nr: np.ndarray) -> float:
  """
  Computes LOO cross-validated accuracy of decoding pos vs neg

  Args:
      pos_nr (np.ndarray): positive datapoints, neurons x trials
      neg_nr (np.ndarray): negative datapoints

  Returns:
      float: accuracy
  """

  R_pos = pos_nr.shape[1]
  R_neg = neg_nr.shape[1]
  X = np.hstack((pos_nr, neg_nr)).T  # Shape (R_pos + R_neg, N)
  y = np.hstack((np.ones(R_pos), np.zeros(R_neg)))  # 1 for positive, 0 for negative

  if np.count_nonzero(np.isnan(X)) > 0:
    return np.nan

  lda = LinearDiscriminantAnalysis()
  loo = LeaveOneOut()
  scores = cross_val_score(lda, X, y, cv=loo, scoring="accuracy")
  accuracy = np.mean(scores)
  return accuracy


# process a batch of timepoints
@ray.remote(num_cpus=1)
def process_batch_lda(pos_data_nrt: np.ndarray, neg_data_nrt: np.ndarray, batch_timepoints):
  return [lda_accuracy(pos_data_nrt[:, :, t], neg_data_nrt[:, :, t]) for t in batch_timepoints]


def parallel_lda_sliding(
  pos_nrt: np.ndarray, neg_nrt: np.ndarray, parallel: bool = False, num_batches: int = 8
) -> np.ndarray:
  """
  split the time range into batches and process them

  Args:
      pos_nrt (np.ndarray): _description_
      neg_nrt (np.ndarray): _description_
      num_batches (int, optional): _description_. Defaults to 8.

  Returns:
      _type_: _description_
  """
  if pos_nrt.shape != neg_nrt.shape:
    raise ValueError("Shapes of pos and neg data do not match")

  T = pos_nrt.shape[2]

  if parallel:
    # Split timepoints into batches
    timepoints = list(range(T))
    batch_size = (T + num_batches - 1) // num_batches  # Ceiling division to get batch size
    batches = [timepoints[i : min(i + batch_size, T)] for i in range(0, T, batch_size)]

    pos_nrt_ref = ray.put(pos_nrt)  # create reference to large array in object store
    neg_nrt_ref = ray.put(neg_nrt)  # create reference to large array in object store
    result_ids = [process_batch_lda.remote(pos_nrt_ref, neg_nrt_ref, batch) for batch in batches]

    results = ray.get(result_ids)
    flattened_results = [item for sublist in results for item in sublist]

  else:  # not parallel
    flattened_results = [lda_accuracy(pos_nrt[:, :, t], neg_nrt[:, :, t]) for t in range(T)]

  return np.array(flattened_results)


def pipeline_lda_sliding(
  sess: Session,
  event_ind: int,
  filt_strat: FilteringStrategy,
  samp_strat: NeuronSamplingStrategy,
  seed: int,
  parallel: bool = True,
) -> np.ndarray:
  # filter data in time
  filt_nrt = filt_strat.filter_sliding(sess.data_nrt)

  # sample neurons
  n_idx = samp_strat.get_sample_inds(sess.n_neurons, seed)
  filt_nrt = filt_nrt[n_idx, :, :]

  # take positive and negative samples for each timepoint (sliding)
  # filt strat needed to avoid filtering window
  pos_data_nrt, neg_data_nrt, _ = pick_pos_neg_data_sliding(filt_nrt, event_ind, filt_strat)

  # compute sliding accuracy
  accuracy_t = parallel_lda_sliding(pos_data_nrt, neg_data_nrt, num_batches=8, parallel=parallel)

  return accuracy_t


def align_and_stack_matrices(
  data: list[np.ndarray], align_idx: list[int]
) -> tuple[np.ndarray, int]:
  """returns aligned tensor (len(data), data[0].shape[0], num aligned time)"""

  # Ensure each matrix in data is at least 2D
  data = [np.atleast_2d(matrix) for matrix in data]

  # Determine the number of rows for consistency check
  num_rows = data[0].shape[0]

  # Check that all matrices have the same number of rows
  for matrix in data:
    if matrix.shape[0] != num_rows:
      raise ValueError("All input matrices must have the same number of rows.")

  # Determine the maximum length of the arrays after alignment
  max_left = max(align_idx)
  max_right = max(matrix.shape[1] - idx for matrix, idx in zip(data, align_idx))

  # The width of the result matrix
  total_length = max_left + max_right

  # Initialize the 3D output array with NaNs
  num_matrices = len(data)
  result = np.full((num_matrices, num_rows, total_length), np.nan)

  # Column index where the align_idx values will end up
  column_index = max_left

  # Fill the 3D array with aligned matrices
  for i, (matrix, idx) in enumerate(zip(data, align_idx)):
    start_idx = max_left - idx
    end_idx = start_idx + matrix.shape[1]
    result[i, :, start_idx:end_idx] = matrix

  return result, column_index


@dataclass(eq=True, frozen=True)
class EventLDAResultsSliding:
  group: str | list[str]
  mouse_ids: NDArray[np.int32]  # sessions
  align_inds: NDArray[np.int32]  # sessions
  accuracy_s_ct: list[NDArray[np.float64]]  # sessions [ cell samples x time ]

  filt_strat: FilteringStrategy
  samp_strat: NeuronSamplingStrategy

  fps: float = 30

  @property
  def dt_sec(self) -> float:
    return 1.0 / float(self.fps)

  def invalidated_before_event_inds(self, event_inds: NDArray[np.int32]) -> Self:
    """set accuracy_s_ct[s] data to nan after time where filtering peeks at data after event_inds[s]"""

    if len(self.accuracy_s_ct) != len(event_inds):
      raise ValueError("event_inds length must match n_sessions")

    accuracy_s_ct = self.accuracy_s_ct.copy()
    for s in range(len(accuracy_s_ct)):
      invalid_after_ind = event_inds[s] - self.filt_strat.window_buffer_post
      accuracy_s_ct[s][:, invalid_after_ind:] = np.nan

    return replace(self, accuracy_s_ct=accuracy_s_ct)

  def get_aligned_accuracy(self) -> tuple[NDArray, NDArray, int]:
    """returns aligned_sct, time_rel_event, event_ind"""
    accuracy_sct, event_ind = align_and_stack_matrices(self.accuracy_s_ct, self.align_inds)

    n_time = accuracy_sct.shape[2]
    T = (n_time - 1) * self.dt_sec
    time_vec = np.linspace(0, T, n_time)
    time_vec = time_vec - time_vec[event_ind]

    return accuracy_sct, time_vec, event_ind

  def averaged_over_samples(self) -> Self:
    accuracy_s_1t = [np.mean(a_ct, axis=0, keepdims=True) for a_ct in self.accuracy_s_ct]
    return replace(self, accuracy_s_ct=accuracy_s_1t)

  def to_df(self) -> pd.DataFrame:
    accuracy_sct, time_vec, event_ind = self.get_aligned_accuracy()

    S, C, T = accuracy_sct.shape
    s_idx, c_idx, t_idx = np.meshgrid(np.arange(S), np.arange(C), np.arange(T), indexing="ij")
    mouse_ids_flat = np.array(self.mouse_ids)[s_idx.flatten()]
    c_flat = c_idx.flatten()
    time_flat = time_vec[t_idx.flatten()]

    # time_rel_event = self.time_sec
    # time_rel_event = time_rel_event - time_rel_event[self.align_inds[s]]

    data = {
      "group": np.repeat(self.group, S * C * T),
      "mouse_id": mouse_ids_flat,
      "time": time_flat,
      "sample": c_flat,
      "accuracy": accuracy_sct.flatten(),
    }
    df = pd.DataFrame(data)
    return df


def lda_sliding_parallel_samples(
  sess: Session,
  filt_strat: FilteringStrategy,
  samp_strat: NeuronSamplingStrategy,
  progress: tqdm_ray.tqdm,
  time_range: list[int] | None,  # pre, post time range
  n_samples: int = 10,
  n_batches: int = 8,
) -> np.ndarray:
  filt_nrt = filt_strat.filter_sliding(sess.data_nrt)

  tone_ind = sess.tone_ind  # this is the event left of which we pick the pre samples
  puff_ind = sess.puff_ind
  trial_has_tone = sess.trial_has_tone
  trial_has_puff = sess.trial_has_puff

  T = filt_nrt.shape[2]
  if time_range is None:
    time_range = [0, T]
  else:
    time_range = [max(0, time_range[0]), min(T, time_range[1])]

  # process a batch of cell samples
  @ray.remote(num_cpus=1)
  def process_batch_lda_sliding(
    filt_strat: FilteringStrategy,
    samp_strat: NeuronSamplingStrategy,
    filt_nrt: np.ndarray,
    batch_samples: NDArray[int],
    progress: tqdm_ray.tqdm,
  ):
    n_neurons = filt_nrt.shape[0]
    T = filt_nrt.shape[2]
    C = len(batch_samples)

    accuracy_ct = np.full((C, T), np.nan)
    # loop over cell samples in batch
    progress_every = 50
    valid_neg_mask_tt = None
    for c, sample in enumerate(batch_samples):
      # sample neurons
      n_idx = samp_strat.get_sample_inds(n_neurons, seed=sample)
      filt_nrt_sampled = filt_nrt[n_idx, :, :]

      # sample positive and negative data windows
      pos_nrt, neg_nrt, valid_neg_mask_tt = pick_pos_neg_data_sliding(
        filt_nrt_sampled, tone_ind, filt_strat, seed=sample, valid_neg_mask_tt=valid_neg_mask_tt
      )

      # loop over time
      for t in range(time_range[0], time_range[1]):
        # select trials with requisite events
        if t >= puff_ind:
          r_mask = np.logical_and(trial_has_puff, trial_has_tone)
        elif t >= tone_ind:
          r_mask = trial_has_tone
        else:
          r_mask = slice(None)

        accuracy_ct[c, t] = lda_accuracy(pos_nrt[:, r_mask, t], neg_nrt[:, r_mask, t])
        if t % progress_every == 1:
          progress.update.remote(progress_every)

    return accuracy_ct

  # Split samples into batches
  C = n_samples
  sample_inds = np.arange(C)
  batch_size = (C + n_batches - 1) // n_batches  # Ceiling division to get batch size
  batches = [sample_inds[i : min(i + batch_size, C)] for i in range(0, C, batch_size)]

  # store as refs
  filt_strat_ref = ray.put(filt_strat)
  samp_strat_ref = ray.put(samp_strat)
  filt_nrt_ref = ray.put(filt_nrt)  # create reference to large array in object store
  result_ids = [
    process_batch_lda_sliding.remote(filt_strat_ref, samp_strat_ref, filt_nrt_ref, batch, progress)
    for batch in batches
  ]

  results = ray.get(result_ids)
  flattened_results = [item for sublist in results for item in sublist]
  accuracy_ct = np.array(flattened_results)

  return accuracy_ct


def multimouse_pipeline_lda_sliding(
  mice: list[Session],
  filt_strat: FilteringStrategy,
  samp_strat: NeuronSamplingStrategy,
  time_range_rel_tone: list[int] | None = None,
  n_samples: int = 10,
  n_batches: int = 8,
  group: str | None = None,
) -> EventLDAResultsSliding:
  if group is None:
    group = mice[0].group

  acc_s = []
  if time_range_rel_tone is None:
    TT = sum([mouse.data_nrt.shape[2] for mouse in mice])
  else:
    TT = len(mice) * len(range(time_range_rel_tone[0], time_range_rel_tone[1]))

  C = n_samples
  progress = remote_tqdm.remote(total=C * TT)
  align_inds = []
  for ind, mouse in enumerate(mice):
    align_inds.append(mouse.tone_ind)

    if time_range_rel_tone is None:
      time_range = None
    else:
      time_range = [
        mouse.tone_ind + time_range_rel_tone[0],
        mouse.tone_ind + time_range_rel_tone[1],
      ]

    mouse_desc = f"{group} {ind} / {len(mice)}: id {mouse.mouse_id}"
    progress.set_description.remote(mouse_desc)
    acc_s.append(
      lda_sliding_parallel_samples(
        mouse,
        filt_strat,
        samp_strat,
        time_range=time_range,
        progress=progress,
        n_samples=n_samples,
        n_batches=n_batches,
      )
    )

  mouse_ids = [mouse.mouse_id for mouse in mice]
  # accuracy_sct, event_ind = align_and_stack_matrices(acc_s, event_inds)
  res = EventLDAResultsSliding(
    group=group,
    mouse_ids=mouse_ids,
    align_inds=align_inds,
    accuracy_s_ct=acc_s,
    samp_strat=samp_strat,
    filt_strat=filt_strat,
  )
  return res
