def anchored_convolution(
  data: np.ndarray,
  kernel: np.ndarray,
  anchor: int,
  flip: bool = False,
  axis: int = -1,
  cval: float = np.nan,
) -> np.ndarray:
  """
  Performs a 1d convolution but allows manual specification of which index of the kernel should align
  with a given sample of the input in the output. Output will be same shape as input

  Args:
      data (np.ndarray):
      kernel (np.ndarray): convolution kernel 1d
      anchor (int): index in kernel to align with input at each output
      flip (bool, optional): whether to flip the kernel in the convolution. Defaults to False, which is typical of a cross-correlation.
      axis (int, optional): axis along which to convolve. Defaults to -1.
      cval (int): constant value passed to convolve1d

  Raises:
      ValueError: _description_

  Returns:
      np.ndarray: _description_
  """
  # Ensure the anchor is a valid index
  if anchor < 0:
    anchor = anchor + len(kernel)
  if not (0 <= anchor < len(kernel)):
    raise ValueError("Anchor must be a valid index in the kernel.")
  if not flip:
    kernel = np.flip(kernel)

  # Perform the convolution using scipy.ndimage.convolve1d with the origin argument
  origin = (len(kernel) - 1) // 2 - anchor
  convolved = convolve1d(
    data.astype(float),
    kernel.astype(float),
    axis=axis,
    mode="constant",
    cval=np.nan,
    origin=origin,
  )

  return convolved


def filter_transients(
  data: np.ndarray,
  window_width_post: int,
  window_buffer: int = 4,
  axis: int = -1,
  subtract_pre_window: bool = True,
) -> np.ndarray:
  """
  Convolves data with a filter to highlight transients
  Averages over a window_width window whose left edge starts at the current timepoint
  and subtracts a similar width average over a window whose right edge starts window_buffer timepoints before the current

  Args:
      data (ndarray): neural signals, typically z-scored
      window_width (int): Width of averaging window(s) in timepoints.
      window_buffer (int, optional): Gap for pre_window. Defaults to 4.
      axis (int, optional): time axis to convolve along. Defaults to -1.
      subtract_pre_window (bool, optional): Defaults to True.

  Returns:
      np.ndarray : filtered data with same shape as data
  """

  pos_kernel = np.ones(window_width) / window_width
  pos_response = anchored_convolution(data, pos_kernel, anchor=0, axis=axis)

  if not subtract_pre_window:
    return pos_response
  else:
    neg_kernel = np.concatenate((pos_kernel, np.zeros(window_buffer)))
    neg_response = anchored_convolution(data, neg_kernel, anchor=-1, axis=axis, flip=False)
    return pos_response - neg_response


def align_and_stack_vectors(data: list[np.ndarray], align_idx: list[int]) -> tuple[np.ndarray, int]:
  # Determine the maximum length of the arrays after alignment
  max_left = max(align_idx)
  max_right = max(len(arr) - idx for arr, idx in zip(data, align_idx))

  # The width of the result matrix
  total_length = max_left + max_right

  # Column index where the align_idx values will end up
  column_index = max_left

  # Initialize the matrix with NaNs
  result = np.full((len(data), total_length), np.nan)

  # Fill the matrix with aligned arrays
  for i, (arr, idx) in enumerate(zip(data, align_idx)):
    start_idx = max_left - idx
    result[i, start_idx : start_idx + len(arr)] = arr

  return result, column_index


def pick_pos_neg_data_sliding(
  filt_nrt: np.ndarray,
  event_ind: int,
  filt_strat: FilteringStrategy,
  seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
  """
  Extracts positive and negative datapoints for each trial, at each positive time

  Args:
      filt_nrt (np.ndarray): filtered input data
      event_ind (int): used to exclude negative timepoints after event
      exclude_window_width (int): averaging window width used for filtering
      seed (int, optional): random seed. Defaults to 42.

  Returns:
      pos_data_nrt, neg_data_nrt
  """

  pos_data = filt_nrt

  # pick random pre-tone, non-overlapping indices for each trial for neg timepoint
  N, R, T = filt_nrt.shape

  # valid mask is T(as pos_ind) x T(as eligible) mask of eligible timepoints for each trial for the neg sample

  # disallow idx where the filtered data is missing
  valid_mask_tt = np.zeros((T, T), dtype=bool)
  valid_mask_tt[...] = np.logical_not(np.isnan(filt_nrt[0, 0]))

  # disallow idx which overlap with the positive sample (first index)'s averaging window
  exclude_window_width = filt_strat.window_width_pre + filt_strat.window_buffer_pre
  for t in range(T):
    valid_mask_tt[t, t : min(t + exclude_window_width, T)] = False

  # disallow idx after the tone
  valid_mask_tt[:, event_ind:] = False

  # sample the negative timepoint in each trial, for each pos_ind timepoint
  neg_data = np.zeros_like(filt_nrt)
  rng = np.random.default_rng(seed=seed)
  for t in range(T):
    valid_indices = np.where(valid_mask_tt[t, :])[0]
    for r in range(R):
      neg_time_idx = rng.choice(valid_indices)
      neg_data[:, r, t] = filt_nrt[:, r, neg_time_idx]

  return pos_data, neg_data
