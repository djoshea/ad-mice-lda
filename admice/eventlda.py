import ray
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import Callable
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut, cross_val_score
from dataclasses import dataclass

# needed for filter sliding
from numba import njit, types
from numba.extending import overload, register_jitable
from numba.np.unsafe.ndarray import to_fixed_tuple

from admice.dataload import Session


## ted's approach to extract features around fixed single timepoint
# take data at


def range_start_n(start, n: int, step=1, limit=None):
  """up to n integers with specified start, outputs will be in [0, limit)"""
  stop = start + step * n
  idx = np.linspace(start, stop, n, endpoint=False)
  if limit is not None:
    mask = (idx >= 0) & (idx < limit)
  else:
    mask = idx >= 0
  return idx[mask]


def range_back_n(stop, n: int, step=1, limit=None):
  """up to n integers with specified stop, outputs will be in [0, limit)"""
  start = stop - step * (n - 1)
  idx = np.linspace(start, stop, n, endpoint=True)
  if limit is not None:
    mask = (idx >= 0) & (idx < limit)
  else:
    mask = idx >= 0
  return idx[mask]


def _convert_reduce_mode_to_fn(reduce_mode: str) -> Callable[[NDArray[np.float64]], float]:
  if reduce_mode == "max":
    reduce_fn = np.max
  elif reduce_mode == "mean":
    reduce_fn = np.mean
  elif reduce_mode == "min":
    reduce_fn = np.min
  else:
    raise NotImplementedError("unknown reduce_mode")

  return reduce_fn


# taken from https://github.com/numba/numba/issues/7369
# needed for sliding_block_reduce_anchored below to support axis argument
@overload(np.moveaxis)
def moveaxis(a: np.ndarray, source, destination) -> np.ndarray:
  """Move axes of an array to new positions.

  Other axes remain in their original order.

  Parameters
  ----------
  a : np.ndarray
      The array whose axes should be reordered.
  source : int or sequence of int
      Original positions of the axes to move. These must be unique.
  dest : int or sequence of int
      Destination positions for each of the original axes. These must also be unique.

  Returns
  -------
  result : np.ndarray
      Array with moved axes. This array is a view of the input array.

  Notes
  -----
  If one of (source, destination) is an integer, then the other must be an integer, too.

  See Also
  --------
  `np.moveaxis <https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html>`_
  """

  @register_jitable
  def impl_array(a: np.ndarray, source, destination):
    source_work = np.atleast_1d(np.asarray(source))
    destination_work = np.atleast_1d(np.asarray(destination))

    if source_work.size != destination_work.size:
      raise ValueError(
        "`source` and `destination` arguments must have " "the same number of elements"
      )

    for idx in range(source_work.size):
      if abs(source_work[idx]) > a.ndim:
        raise ValueError("Invalid axis in `source`.")
      if abs(destination_work[idx]) > a.ndim:
        raise ValueError("Invalid axis in `destination`.")

    source_work = [x % a.ndim for x in source_work]
    destination_work = [x % a.ndim for x in destination_work]

    order = [n for n in range(a.ndim) if n not in source_work]
    for dest, src in sorted(zip(destination_work, source_work)):
      order.insert(dest, src)

    oder_tuple = to_fixed_tuple(np.array(order), a.ndim)
    return np.transpose(a, oder_tuple)

  @register_jitable
  def impl_int(a: np.ndarray, source, destination):
    if abs(source) > a.ndim:
      raise ValueError("Invalid axis in `source`.")
    if abs(destination) > a.ndim:
      raise ValueError("Invalid axis in `destination`.")

    source = source % a.ndim
    destination = destination % a.ndim

    order = [n for n in range(a.ndim) if n != source]
    order.insert(destination, source)

    oder_tuple = to_fixed_tuple(np.array(order), a.ndim)
    return np.transpose(a, oder_tuple)

  if isinstance(source, types.Integer) and isinstance(destination, types.Integer):
    return impl_int
  else:
    return impl_array


@njit
def sliding_block_reduce_anchored(
  data: NDArray[np.float64],
  block_size: int,
  anchor: int,
  require_full_window: bool = True,
  invalid_value: float = np.nan,
  axis: int = -1,
  reduce_mode: str = "max",
  # reduce_fn: Callable[[NDArray[np.float64]], float] = np.max,
) -> NDArray[np.float64]:
  # Move the specified axis to the last position
  if axis != -1 and axis != data.ndim:
    data = np.moveaxis(data, axis, -1)

  def reduce_fn(input):
    if reduce_mode == "max":
      output = np.max(input)
    elif reduce_mode == "mean":
      output = np.mean(input)
    elif reduce_mode == "min":
      output = np.min(input)
    else:
      raise NotImplementedError("unknown reduce_mode")

    return output

  num_features = data.shape[:-1]
  num_timepoints = data.shape[-1]
  output = np.full(data.shape, invalid_value, dtype=np.float64)

  if require_full_window:
    for idx in np.ndindex(num_features):
      for time in range(num_timepoints):
        start = time - anchor
        end = start + block_size
        if start >= 0 and end <= num_timepoints and start < end:
          # apply reduction
          output[idx + (time,)] = reduce_fn(data[idx + (slice(start, end),)])

  else:
    # shrink
    for idx in np.ndindex(num_features):
      for time in range(num_timepoints):
        start = max(0, time - anchor)
        end = min(num_timepoints, time - anchor + block_size)
        # apply reduction
        if start < end:
          output[idx + (time,)] = reduce_fn(data[idx + (slice(start, end),)])

  # Move the axis back to its original position
  if axis != -1 and axis != data.ndim:
    output = np.moveaxis(output, -1, axis)
  return output


@dataclass
class FilteringStrategy:
  mode: str
  window_width_post: int = 23
  window_width_pre: int = 23
  reduce_mode_post: str = "max"
  reduce_mode_pre: str = "mean"
  window_buffer_pre: int = 2
  window_buffer_post: int = 0

  # these are stored as strings for numba jitting in eventlda_sliding
  @property
  def reduce_fn_pre(self):
    return _convert_reduce_mode_to_fn(self.reduce_mode_pre)

  @property
  def reduce_fn_post(self):
    return _convert_reduce_mode_to_fn(self.reduce_mode_post)

  def filter_at_event(
    self,
    data_nrt: np.ndarray,
    event_ind: int,
  ):
    """ted's filtering approach for single timepoint, take max (or mean) of post event window minus same for pre event window

    post event window starts at event_ind + window_buffer
    pre event window ends at event_ind - window_buffer
    """
    T = data_nrt.shape[-1]
    event_window = range_start_n(
      event_ind + self.window_buffer_post, self.window_width_post, limit=T
    ).astype(int)
    event_max_nr = self.reduce_fn_post(data_nrt[..., event_window], axis=-1)

    pre_event_window = range_back_n(
      event_ind - self.window_buffer_pre, self.window_width_pre, limit=T
    ).astype(int)
    pre_event_max_nr = self.reduce_fn_pre(data_nrt[..., pre_event_window], axis=-1)

    event_delta_nr = event_max_nr - pre_event_max_nr
    return event_delta_nr

  def filter_random_time_before_event(
    self,
    data_nrt: np.ndarray,
    event_ind: int,
    start_ind: int = 0,
    seed: int = 42,
  ) -> np.ndarray:
    """ted's filtering approach, to collect null data before a single timepoint"""

    # pick random pre-tone, non-overlapping indices for each trial for neg timepoint
    N, R, T = data_nrt.shape

    # valid null data window, must leave room for the pre window to the left and avoid post event time to the right
    valid_indices = np.arange(
      max(start_ind, self.window_width_pre + self.window_buffer_pre),
      event_ind - self.window_width_post - self.window_buffer_post,
    )
    if len(valid_indices) == 0:
      raise ValueError("No valid indices to pick null data from")

    # sample the negative timepoint in each trial
    rng = np.random.default_rng(seed=seed)
    neg_data = np.zeros((N, R))
    for r in range(R):
      neg_time_ind = rng.choice(valid_indices)
      pre = data_nrt[
        :,
        r,
        neg_time_ind - self.window_width_pre - self.window_buffer_pre : neg_time_ind
        - self.window_buffer_pre,
      ]
      post = data_nrt[
        :,
        r,
        neg_time_ind + self.window_buffer_post : neg_time_ind
        + self.window_width_post
        + self.window_buffer_post,
      ]
      neg_data[:, r] = self.reduce_fn_post(post, axis=-1) - self.reduce_fn_pre(pre, axis=-1)

    return neg_data

  def extract_pos_neg_data_at_event(
    self,
    data_nrt: np.ndarray,
    evaluate_ind: int,
    neg_before_ind: int,
    seed: int,
    neg_start_ind: int = 0,
  ) -> tuple[np.ndarray, np.ndarray]:
    """
    extract the positive sample (filtered at evaluate_ind) and
    a random negative sample (filtered before neg_before_ind)
    """
    if self.mode == "two_window_delta":
      pos_nr = self.filter_at_event(data_nrt, evaluate_ind)
      neg_nr = self.filter_random_time_before_event(
        data_nrt, neg_before_ind, start_ind=neg_start_ind, seed=seed
      )

    else:
      raise NotImplementedError("unknown mode")

    return pos_nr, neg_nr

  def filter_sliding(self, data_nrt: NDArray[np.float64]) -> NDArray[np.float64]:
    # here the filtering proceeds acausally, with buffer > 0 implying the current
    # timepoint is anchored left of the start of the window (negative)
    filt_post = sliding_block_reduce_anchored(
      data_nrt,
      self.window_width_post,
      -self.window_buffer_post,
      reduce_mode=self.reduce_mode_post,
    )

    # here the filtering proceeds causally, with buffer > 0 implying the current
    # timepoint is anchored right of the end of the window
    filt_pre = sliding_block_reduce_anchored(
      data_nrt,
      self.window_width_pre,
      self.window_width_pre + self.window_buffer_pre - 1,
      reduce_mode=self.reduce_mode_pre,
    )

    return filt_post - filt_pre

  def split_epochs_causal(
    self, filt_nrt: NDArray[np.float64], start_ind: int, tone_ind: int, puff_ind: int
  ) -> NDArray[np.float64]:
    skip = self.window_width_post + self.window_buffer_post
    # skip = 0

    pre = filt_nrt[..., start_ind : tone_ind - skip]
    tone = filt_nrt[..., tone_ind - skip + self.window_buffer_post : puff_ind - skip]
    puff = filt_nrt[..., puff_ind - skip :]
    return pre, tone, puff


@dataclass
class NeuronSamplingStrategy:
  mode: str
  n_neurons: int
  scores: np.ndarray | None = None  # used for greedy ordering (higher values taken first)

  def get_sample_inds(self, n_neurons_total: int, seed: int) -> np.ndarray:
    if self.mode == "random":
      rng = np.random.default_rng(seed)
      return rng.choice(n_neurons_total, self.n_neurons)

    else:
      raise NotImplementedError("Mode not implemented")


@dataclass
class EventLDAResults:
  event: str
  group: str | list[str]
  mouse_ids: np.ndarray  # M
  event_ind: int
  accuracy_sc: np.ndarray  # sessions x cell samples

  filt_strat: FilteringStrategy
  samp_strat: NeuronSamplingStrategy

  @property
  def n_sessions(self):
    return self.accuracy_sc.shape[0]

  @property
  def n_samples(self):
    return self.accuracy_sc.shape[1]

  def to_df(self) -> pd.DataFrame:
    S, C = self.accuracy_sc.shape

    data = {
      "group": np.repeat(self.group, S * C),
      "mouse_id": np.repeat(self.mouse_ids, C),
      "cell_sample_ind": np.tile(np.arange(C), S),
      "accuracy": self.accuracy_sc.flatten(),
    }
    df = pd.DataFrame(data)
    return df


# process a batch of cell samples
@ray.remote(num_cpus=1)
def process_batch_lda(
  samp_strat: NeuronSamplingStrategy,
  pos_nr: np.ndarray,
  neg_nr: np.ndarray,
  batch_samples,
):
  accuracy_c = []
  for c in batch_samples:
    n_idx = samp_strat.get_sample_inds(pos_nr.shape[0], seed=c)
    pos_nr_ = pos_nr[n_idx, :]
    neg_nr_ = neg_nr[n_idx, :]
    accuracy_c.append(lda_accuracy(pos_nr_, neg_nr_))

  return accuracy_c


def lda_parallel_samples(
  pos_nr: np.ndarray,
  neg_nr: np.ndarray,
  samp_strat: NeuronSamplingStrategy,
  n_samples: int,
  n_batches: int = 8,
) -> np.ndarray:
  """
  performs neuron sampling many times in parallelized batches

  Args:
      pos_nr (np.ndarray): _description_
      neg_nr (np.ndarray): _description_
      n_neuron_samples (int):
      num_batches (int, optional): _description_. Defaults to 8.
  """
  if pos_nr.shape != neg_nr.shape:
    raise ValueError("Shapes of pos and neg data do not match")

  # Split samples into batches
  C = n_samples
  sample_inds = np.arange(C)
  batch_size = (C + n_batches - 1) // n_batches  # Ceiling division to get batch size
  batches = [sample_inds[i : min(i + batch_size, C)] for i in range(0, C, batch_size)]

  samp_strat_ref = ray.put(samp_strat)
  pos_nr_ref = ray.put(pos_nr)  # create reference to large array in object store
  neg_nr_ref = ray.put(neg_nr)  # create reference to large array in object store
  result_ids = [
    process_batch_lda.remote(samp_strat_ref, pos_nr_ref, neg_nr_ref, batch) for batch in batches
  ]

  results = ray.get(result_ids)
  flattened_results = [item for sublist in results for item in sublist]

  return np.array(flattened_results)


def pipeline_lda_event(
  sess: Session,
  filt_strat: FilteringStrategy,
  samp_strat: NeuronSamplingStrategy,
  evaluate_ind: int | None = None,
  n_samples: int = 1000,
  num_batches: int = 8,
  seed: int = 1,
) -> np.ndarray:
  if evaluate_ind is None:
    evaluate_ind = sess.tone_ind
  pos_nr, neg_nr = filt_strat.extract_pos_neg_data_at_event(
    sess.data_nrt, evaluate_ind=evaluate_ind, neg_before_ind=sess.tone_ind, seed=seed
  )

  accuracy_c = lda_parallel_samples(
    pos_nr, neg_nr, samp_strat=samp_strat, n_samples=n_samples, n_batches=num_batches
  )

  return accuracy_c


def multimouse_pipeline_lda_tone(
  mice: list[Session],
  filt_strat: FilteringStrategy,
  samp_strat: NeuronSamplingStrategy,
  n_samples: int = 1000,
  event_inds: np.ndarray | None = None,
  group: str | None = None,
) -> EventLDAResults:
  if group is None:
    group = mice[0].group
  if event_inds is None:
    event_inds = [mouse.tone_ind for mouse in mice]

  acc = []
  for ind, mouse in enumerate(mice):
    print(f"{group} {ind} / {len(mice)}: id {mouse.mouse_id}")
    acc.append(
      pipeline_lda_event(
        mouse, filt_strat, samp_strat, n_samples=n_samples, event_ind=event_inds[ind]
      )
    )

  mouse_ids = [mouse.mouse_id for mouse in mice]
  accuracy_sc = np.stack(acc)
  res = EventLDAResults(
    event="tone",
    group=group,
    mouse_ids=mouse_ids,
    event_ind=event_inds,
    accuracy_sc=accuracy_sc,
    filt_strat=filt_strat,
    samp_strat=samp_strat,
  )
  return res


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


# LDA axes


def lda_axis(pos_nr: np.ndarray, neg_nr: np.ndarray) -> float:
  """
  Computes LDA axis

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
