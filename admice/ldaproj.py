import numpy as np
from numpy.typing import NDArray
from typing import Self
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from dataclasses import dataclass, replace
# import matplotlib as mpl
# import matplotlib.pyplot as plt

import ray
from ray.experimental import tqdm_ray
from tqdm import trange

from admice.eventlda import FilteringStrategy, NeuronSamplingStrategy
from admice.dataload import Session


remote_tqdm = ray.remote(tqdm_ray.tqdm)


def orthonormalize_columns(matrix: np.ndarray, reverse: bool = False) -> np.ndarray:
  if reverse:
    matrix = np.flip(matrix, axis=1)

  # Normalize the first column
  v1 = matrix[:, 0]
  v1_normalized = v1 / np.linalg.norm(v1)

  # Orthogonalize the second column with respect to the first column
  v2 = matrix[:, 1]
  v2_proj_on_v1 = np.dot(v2, v1_normalized) * v1_normalized
  v2_orthogonal = v2 - v2_proj_on_v1

  # Normalize the orthogonalized second column
  v2_orthonormal = v2_orthogonal / np.linalg.norm(v2_orthogonal)

  # Combine the orthonormalized columns into a new matrix
  orthonormal_matrix = np.column_stack((v1_normalized, v2_orthonormal))

  if reverse:
    orthonormal_matrix = np.flip(orthonormal_matrix, axis=1)

  return orthonormal_matrix


# original version without ray
def project_lda_pufftone(
  sess: Session,
  filt_strat_fit: FilteringStrategy,
  filt_strat_project: FilteringStrategy,
  samp_strat: NeuronSamplingStrategy,
  seed: int = 1,
  n_samples: int = 10,
) -> np.ndarray:
  pos_nr_tone, neg_nr_tone = filt_strat_fit.extract_pos_neg_data_at_event(
    sess.data_nrt, evaluate_ind=sess.tone_ind, neg_before_ind=sess.tone_ind, seed=seed
  )
  pos_nr_puff, neg_nr_puff = filt_strat_fit.extract_pos_neg_data_at_event(
    sess.data_nrt, evaluate_ind=sess.puff_ind, neg_before_ind=sess.tone_ind, seed=seed
  )

  N_total, R, T = sess.data_nrt.shape

  # assemble data for LDA (pre-neuron sampling)
  R_pos_tone = pos_nr_tone.shape[1]
  R_pos_puff = pos_nr_puff.shape[1]
  R_neg = neg_nr_tone.shape[1]
  y = np.hstack(
    (np.zeros(R_neg), np.ones(R_pos_tone), np.full(R_pos_puff, 2))
  )  # 1 for positive, 0 for negative

  # trial groups for splits and cross-validation
  r_groups = np.tile(np.arange(R), 3)

  # assemble full-trial data for passing thru transform()
  # filt_nrt --> n(sampled) x rt
  # Reorder the axes so that it becomes R x T x N
  filt_nrt = filt_strat_project.filter_sliding(sess.data_nrt)
  filt_rtn = np.transpose(filt_nrt, (1, 2, 0))

  # Reshape to RT x N
  filt_rt_n = filt_rtn.reshape(R * T, N_total)
  filt_nan_mask_rt = np.isnan(filt_rt_n).any(axis=1)
  filt_rt_n[filt_nan_mask_rt] = 0

  K = 2  # dimensionality of LDA space
  proj_crtk = np.zeros((n_samples, R, T, K))

  for c in trange(n_samples, desc="Running neuron samples"):
    # sample neurons
    n_idx = samp_strat.get_sample_inds(n_neurons_total=sess.n_neurons, seed=c)
    pos_nr_tone_sampled = pos_nr_tone[n_idx, :]
    pos_nr_puff_sampled = pos_nr_puff[n_idx, :]
    neg_nr_tone_sampled = neg_nr_tone[n_idx, :]
    # neg_nr_puff_sampled = neg_nr_puff[n_idx, :]

    # assemble LDA inputs
    X = np.hstack(
      (neg_nr_tone_sampled, pos_nr_tone_sampled, pos_nr_puff_sampled)
    ).T  # Shape (R_neg + R_pos_tone + R_pos_puff, N)

    lda = LinearDiscriminantAnalysis()
    logo = LeaveOneGroupOut()

    # loop over folds, store cv transforms for each trials
    for train_index, test_index in logo.split(X, y, r_groups):
      # fit LDA
      X_train, _ = X[train_index], X[test_index]
      y_train, _ = y[train_index], y[test_index]
      lda.fit(X_train, y_train)

      # list trial groups in the test set
      r_test_index = np.unique(r_groups[test_index])

      # transform the full corresponding test trials through the transforms
      rt_test_index = (r_test_index[:, None] + np.arange(T)).flatten()
      out_rt_k = lda.transform(filt_rt_n[rt_test_index[:, None], n_idx])
      out_rt_k[filt_nan_mask_rt[rt_test_index]] = np.nan
      R_test = len(r_test_index)
      proj_crtk[c, r_test_index] = np.reshape(out_rt_k, (R_test, T, K))

  return proj_crtk


# ray version
# original version without ray
def project_lda_pufftone_parallel_samples(
  sess: Session,
  filt_strat_fit: FilteringStrategy,
  filt_strat_project: FilteringStrategy,
  samp_strat: NeuronSamplingStrategy,
  lda_mode: str = "separate",
  seed: int = 1,
  n_samples: int = 10,
  n_batches: int = 8,
  parallel: bool = True,
  progress: tqdm_ray.tqdm | None = None,
) -> np.ndarray:
  neg_start_ind = sess.tone_ind - int(1.0 * sess.fps)
  pos_nr_tone, neg_nr_tone = filt_strat_fit.extract_pos_neg_data_at_event(
    sess.data_nrt,
    evaluate_ind=sess.tone_ind,
    neg_before_ind=sess.tone_ind,
    neg_start_ind=neg_start_ind,
    seed=seed,
  )
  pos_nr_puff, neg_nr_puff = filt_strat_fit.extract_pos_neg_data_at_event(
    sess.data_nrt,
    evaluate_ind=sess.puff_ind,
    neg_before_ind=sess.tone_ind,
    neg_start_ind=neg_start_ind,
    seed=seed,
  )

  n_neurons, R, T = sess.data_nrt.shape

  # assemble data for LDA (pre-neuron sampling)
  assert pos_nr_tone.shape[1] == R
  assert pos_nr_puff.shape[1] == R
  assert neg_nr_tone.shape[1] == R
  assert neg_nr_puff.shape[1] == R

  X = np.hstack(
    (neg_nr_tone, neg_nr_puff, pos_nr_tone, pos_nr_puff)
  ).T  # Shape (R_neg + R_pos_tone + R_pos_puff, N)
  y = np.hstack(
    (np.zeros(2 * R), np.ones(R), np.full(R, 2))
  )  # 0 for pre-tone, 1 for post-tone, 2 for post-puff
  r_groups = np.tile(np.arange(R), 4)  # trial groups for splits and cross-validation

  # assemble full-trial data for passing thru transform()
  # filt_nrt --> n(sampled) x rt
  # Reorder the axes so that it becomes R x T x N
  filt_nrt = filt_strat_project.filter_sliding(sess.data_nrt)
  filt_rtn = np.transpose(filt_nrt, (1, 2, 0))

  def project_lda_pufftone_batch_samples(
    X: np.ndarray,
    y: np.ndarray,
    r_groups: np.ndarray,
    filt_rtn_: np.ndarray,
    samp_strat: NeuronSamplingStrategy,
    lda_mode: str,
    batch_samples: np.ndarray,
    progress: tqdm_ray.tqdm | None = None,
  ) -> np.ndarray:
    R, T, n_neurons = filt_rtn_.shape

    # Reshape to RT x N
    # filt_rtn = filt_rtn - np.mean(filt_rtn[:, :tone_ind, :], axis=1, keepdims=True)
    filt_rt_n = np.copy(filt_rtn_).reshape(R * T, n_neurons)
    filt_nan_mask_rt = np.isnan(filt_rt_n).any(axis=1)
    filt_rt_n[filt_nan_mask_rt] = 0

    mapping = {0: 0, 1: 1, 2: -1}
    y_pre_v_tone_ = np.vectorize(mapping.get)(y)
    mapping = {0: 0, 1: -1, 2: 2}
    y_tone_v_puff_ = np.vectorize(mapping.get)(y)

    K = 2  # dimensionality of LDA space
    n_samples = len(batch_samples)
    proj_crtk = np.zeros((n_samples, R, T, K))

    if progress is not None:
      progress.refresh.remote()
    for c, sample in enumerate(batch_samples):
      # sample neurons
      n_idx = samp_strat.get_sample_inds(n_neurons_total=n_neurons, seed=sample)
      X_sampled = X[:, n_idx]  # Shape (R_neg + R_pos_tone + R_pos_puff, N)

      logo = LeaveOneGroupOut()

      if lda_mode == "multiclass":
        lda = LinearDiscriminantAnalysis()

        # loop over folds, store cv transforms for each trials
        for train_index, test_index in logo.split(X, y, r_groups):
          # fit LDA
          X_train, _ = X_sampled[train_index], X_sampled[test_index]
          y_train, _ = y[train_index], y[test_index]
          lda.fit(X_train, y_train)

          # list trial groups in the test set
          r_test_index = np.unique(r_groups[test_index])

          # transform the full corresponding test trials through the transforms
          rt_test_index = (r_test_index[:, None] + np.arange(T)).flatten()
          out_rt_k = lda.transform(filt_rt_n[rt_test_index[:, None], n_idx])
          out_rt_k[filt_nan_mask_rt[rt_test_index]] = np.nan
          R_test = len(r_test_index)
          proj_crtk[c, r_test_index] = np.reshape(out_rt_k, (R_test, T, K))

      elif lda_mode == "separate":
        lda_pre_v_tone = LinearDiscriminantAnalysis()
        lda_tone_v_puff = LinearDiscriminantAnalysis()

        # mask = np.logical_or(y == 0, y == 1)
        mask = y_pre_v_tone_ >= 0
        X_pre_v_tone = X_sampled[mask]
        y_pre_v_tone = y_pre_v_tone_[mask]
        r_groups_pre_v_tone = r_groups[mask]

        # mask = np.logical_or(y == 1, y == 2)
        mask = y_tone_v_puff_ >= 0
        X_tone_v_puff = X_sampled[mask]
        y_tone_v_puff = y_tone_v_puff_[mask]
        r_groups_tone_v_puff = r_groups[mask]

        # loop over folds, store cv transforms for each trials
        for (train_index_pre_v_tone, test_index_pre_v_tone), (
          train_index_tone_v_puff,
          test_index_tone_v_puff,
        ) in zip(
          logo.split(X_pre_v_tone, y_pre_v_tone, r_groups_pre_v_tone),
          logo.split(X_tone_v_puff, y_tone_v_puff, r_groups_tone_v_puff),
        ):
          # fit LDA x 2
          lda_pre_v_tone.fit(
            X_pre_v_tone[train_index_pre_v_tone], y_pre_v_tone[train_index_pre_v_tone]
          )
          lda_tone_v_puff.fit(
            X_tone_v_puff[train_index_tone_v_puff], y_tone_v_puff[train_index_tone_v_puff]
          )

          # N x 2
          # scalings = orthonormalize_columns(
          #   np.column_stack((lda_pre_v_tone.coef_.T, lda_tone_v_puff.coef_.T))
          # )
          scalings = orthonormalize_columns(
            np.column_stack((lda_pre_v_tone.scalings_, lda_tone_v_puff.scalings_)), reverse=False
          )

          # check whether we need to flip the axes to get the projected class centroids arranged in the right order
          # projected R x 2
          proj_pre_v_tone = X_pre_v_tone[train_index_pre_v_tone] @ scalings
          if np.mean(proj_pre_v_tone[y_pre_v_tone[train_index_pre_v_tone] == 0, 0]) > np.mean(
            proj_pre_v_tone[y_pre_v_tone[train_index_pre_v_tone] == 1, 0]
          ):
            scalings[:, 0] = -scalings[:, 0]
          proj_tone_v_puff = X_tone_v_puff[train_index_tone_v_puff] @ scalings
          if np.mean(proj_tone_v_puff[y_tone_v_puff[train_index_tone_v_puff] == 1, 1]) > np.mean(
            proj_tone_v_puff[y_tone_v_puff[train_index_tone_v_puff] == 2, 1]
          ):
            scalings[:, 1] = -scalings[:, 1]

          # list trial groups in the test set
          r_test_index = np.unique(r_groups[test_index_pre_v_tone])

          # transform the full corresponding test trials through the transforms
          rt_test_index = (r_test_index[:, None] + np.arange(T)).flatten()

          # RT x N @ N x 2
          out_rt_k = filt_rt_n[rt_test_index[:, None], n_idx] @ scalings

          # restore nans
          out_rt_k[filt_nan_mask_rt[rt_test_index]] = np.nan
          R_test = len(r_test_index)
          proj_crtk[c, r_test_index] = np.reshape(out_rt_k, (R_test, T, K))

        if progress is not None:
          progress.update.remote(1)

    return proj_crtk

  # Split samples into batches
  C = n_samples
  sample_inds = np.arange(C)
  batch_size = (C + n_batches - 1) // n_batches  # Ceiling division to get batch size
  batches = [sample_inds[i : min(i + batch_size, C)] for i in range(0, C, batch_size)]

  if not parallel:
    proj_crtk = project_lda_pufftone_batch_samples(
      X, y, r_groups, filt_rtn, samp_strat, lda_mode, range(C), None
    )
  else:
    ray_project_lda_pufftone_batch_samples = ray.remote(num_cpus=1)(
      project_lda_pufftone_batch_samples
    )

    # store as refs
    X_ref = ray.put(X)
    y_ref = ray.put(y)
    r_groups_ref = ray.put(r_groups)
    samp_strat_ref = ray.put(samp_strat)
    filt_rtn_ref = ray.put(filt_rtn)  # create reference to large array in object store
    if progress is None:
      progress = remote_tqdm.remote(total=C, desc="Running samples")
    result_ids = [
      ray_project_lda_pufftone_batch_samples.remote(
        X_ref, y_ref, r_groups_ref, filt_rtn_ref, samp_strat_ref, lda_mode, batch, progress
      )
      for batch in batches
    ]

    results = ray.get(result_ids)
    flattened_results = [item for sublist in results for item in sublist]
    proj_crtk = np.array(flattened_results)

  return proj_crtk


# new version that uses the whole trial for the LDA!
# not yet cross-validated
def project_wholetrial_lda_pufftone_parallel_samples(
  sess: Session,
  filt_strat_fit: FilteringStrategy,
  filt_strat_project: FilteringStrategy,
  samp_strat: NeuronSamplingStrategy,
  lda_mode: str = "separate",
  seed: int = 1,
  n_samples: int = 10,
  n_batches: int = 8,
  parallel: bool = True,
  progress: tqdm_ray.tqdm | None = None,
) -> np.ndarray:
  filt_fit_nrt = filt_strat_fit.filter_sliding(sess.data_nrt)

  # build data for LDA (pre neuron sampling)
  pre_start_ind = sess.tone_ind - int(2.0 * sess.fps)
  pre_nrt, tone_nrt, puff_nrt = filt_strat_fit.split_epochs_causal(
    filt_fit_nrt, pre_start_ind, sess.tone_ind, sess.puff_ind
  )

  N, R, T = sess.data_nrt.shape
  T_pre = pre_nrt.shape[2]
  T_tone = tone_nrt.shape[2]
  T_puff = puff_nrt.shape[2]
  pre_rt_n = pre_nrt.reshape(N, R * T_pre).T
  tone_rt_n = tone_nrt.reshape(N, R * T_tone).T
  puff_rt_n = puff_nrt.reshape(N, R * T_puff).T
  RT_pre = pre_rt_n.shape[0]
  RT_tone = tone_rt_n.shape[0]
  RT_puff = puff_rt_n.shape[0]

  X = np.vstack((pre_rt_n, tone_rt_n, puff_rt_n))  # Shape (RT_pre + RT_tone + RT_puff, N)
  y = np.hstack(
    (np.zeros(RT_pre), np.ones(RT_tone), np.full(RT_puff, 2))
  )  # 0 for pre-tone, 1 for post-tone, 2 for post-puff
  r_groups = np.hstack(
    (
      np.mgrid[:R, :T_pre][0].flatten(),
      np.mgrid[:R, :T_tone][0].flatten(),
      np.mgrid[:R, :T_puff][0].flatten(),
    )
  )

  # assemble full-trial data for passing thru transform()
  # filt_nrt --> n(sampled) x rt
  # Reorder the axes so that it becomes R x T x N
  filt_proj_nrt = filt_strat_project.filter_sliding(sess.data_nrt)
  filt_proj_rtn = np.transpose(filt_proj_nrt, (1, 2, 0))

  def project_lda_pufftone_batch_samples(
    X: np.ndarray,
    y: np.ndarray,
    r_groups: np.ndarray,
    filt_proj_rtn_: np.ndarray,
    samp_strat: NeuronSamplingStrategy,
    lda_mode: str,
    batch_samples: np.ndarray,
    progress: tqdm_ray.tqdm | None = None,
  ) -> np.ndarray:
    R, T, N = filt_proj_rtn_.shape

    # Reshape to RT x N
    # filt_rtn = filt_rtn - np.mean(filt_rtn[:, :tone_ind, :], axis=1, keepdims=True)
    filt_rt_n = np.copy(filt_proj_rtn_).reshape(R * T, N)
    filt_nan_mask_rt = np.isnan(filt_rt_n).any(axis=1)
    filt_rt_n[filt_nan_mask_rt] = 0

    if lda_mode == "separate" or lda_mode == "separate_cv":
      mapping = {0: 0, 1: 1, 2: -1}
      y_pre_v_tone_ = np.vectorize(mapping.get)(y)
      mapping = {0: 0, 1: 0, 2: 2}
      y_tone_v_puff_ = np.vectorize(mapping.get)(y)

    if lda_mode == "separate_cv":
      pass

    K = 2  # dimensionality of LDA space
    n_samples = len(batch_samples)
    proj_crtk = np.zeros((n_samples, R, T, K))

    if progress is not None:
      progress.refresh.remote()
    for c, sample in enumerate(batch_samples):
      # sample neurons
      n_idx = samp_strat.get_sample_inds(n_neurons_total=N, seed=sample)
      X_sampled = X[:, n_idx]  # Shape (R_neg + R_pos_tone + R_pos_puff, N)

      if lda_mode == "multiclass":
        lda = LinearDiscriminantAnalysis()

        lda.fit(X_sampled, y)

        # transform the full corresponding test trials through the transforms
        out_rt_k = lda.transform(filt_rt_n[:, n_idx])
        out_rt_k[filt_nan_mask_rt] = np.nan
        proj_crtk[c] = np.reshape(out_rt_k, (R, T, K))

      elif lda_mode == "separate":
        lda_pre_v_tone = LinearDiscriminantAnalysis()
        lda_tone_v_puff = LinearDiscriminantAnalysis()

        # mask = np.logical_or(y == 0, y == 1)
        mask = np.logical_and(~np.isnan(X_sampled).any(axis=1), y_pre_v_tone_ >= 0)
        X_pre_v_tone = X_sampled[mask]
        y_pre_v_tone = y_pre_v_tone_[mask]

        # mask = np.logical_or(y == 1, y == 2)
        mask = np.logical_and(~np.isnan(X_sampled).any(axis=1), y_tone_v_puff_ >= 0)
        X_tone_v_puff = X_sampled[mask]
        y_tone_v_puff = y_tone_v_puff_[mask]

        # fit LDA x 2
        lda_pre_v_tone.fit(X_pre_v_tone, y_pre_v_tone)
        lda_tone_v_puff.fit(X_tone_v_puff, y_tone_v_puff)

        scalings = orthonormalize_columns(
          np.column_stack((lda_pre_v_tone.scalings_, lda_tone_v_puff.scalings_)), reverse=False
        )

        # check whether we need to flip the axes to get the projected class centroids arranged in the right order
        # projected R x 2
        proj_pre_v_tone = X_pre_v_tone @ scalings
        if np.mean(proj_pre_v_tone[y_pre_v_tone == 0, 0]) > np.mean(
          proj_pre_v_tone[y_pre_v_tone == 1, 0]
        ):
          scalings[:, 0] = -scalings[:, 0]
        proj_tone_v_puff = X_tone_v_puff @ scalings
        if np.mean(proj_tone_v_puff[y_tone_v_puff == 0, 1]) > np.mean(
          proj_tone_v_puff[y_tone_v_puff == 2, 1]
        ):
          scalings[:, 1] = -scalings[:, 1]

        # RT x N @ N x 2
        out_rt_k = filt_rt_n[:, n_idx] @ scalings

        # restore nans
        out_rt_k[filt_nan_mask_rt] = np.nan
        proj_crtk[c] = np.reshape(out_rt_k, (R, T, K))

      elif lda_mode == "separate_cv":
        lda_pre_v_tone = LinearDiscriminantAnalysis()
        lda_tone_v_puff = LinearDiscriminantAnalysis()
        gkfold = GroupKFold(n_splits=10)

        # mask = np.logical_or(y == 0, y == 1)
        mask = np.logical_and(~np.isnan(X_sampled).any(axis=1), y_pre_v_tone_ >= 0)
        X_pre_v_tone = X_sampled[mask]
        y_pre_v_tone = y_pre_v_tone_[mask]
        r_groups_pre_v_tone = r_groups[mask]

        # mask = np.logical_or(y == 1, y == 2)
        mask = np.logical_and(~np.isnan(X_sampled).any(axis=1), y_tone_v_puff_ >= 0)
        X_tone_v_puff = X_sampled[mask]
        y_tone_v_puff = y_tone_v_puff_[mask]
        r_groups_tone_v_puff = r_groups[mask]

        # loop over folds, store cv transforms for each trials
        for (train_index_pre_v_tone, test_index_pre_v_tone), (
          train_index_tone_v_puff,
          test_index_tone_v_puff,
        ) in zip(
          gkfold.split(X_pre_v_tone, y_pre_v_tone, r_groups_pre_v_tone),
          gkfold.split(X_tone_v_puff, y_tone_v_puff, r_groups_tone_v_puff),
        ):
          # fit LDA x 2
          lda_pre_v_tone.fit(
            X_pre_v_tone[train_index_pre_v_tone], y_pre_v_tone[train_index_pre_v_tone]
          )
          lda_tone_v_puff.fit(
            X_tone_v_puff[train_index_tone_v_puff], y_tone_v_puff[train_index_tone_v_puff]
          )

          scalings = orthonormalize_columns(
            np.column_stack((lda_pre_v_tone.scalings_, lda_tone_v_puff.scalings_)), reverse=False
          )

          # check whether we need to flip the axes to get the projected class centroids arranged in the right order
          # projected R x 2
          proj_pre_v_tone = X_pre_v_tone[train_index_pre_v_tone] @ scalings
          if np.mean(proj_pre_v_tone[y_pre_v_tone[train_index_pre_v_tone] == 0, 0]) > np.mean(
            proj_pre_v_tone[y_pre_v_tone[train_index_pre_v_tone] == 1, 0]
          ):
            scalings[:, 0] = -scalings[:, 0]
          proj_tone_v_puff = X_tone_v_puff[train_index_tone_v_puff] @ scalings
          if np.mean(proj_tone_v_puff[y_tone_v_puff[train_index_tone_v_puff] == 0, 1]) > np.mean(
            proj_tone_v_puff[y_tone_v_puff[train_index_tone_v_puff] == 2, 1]
          ):
            scalings[:, 1] = -scalings[:, 1]

          # list trial groups in the test set
          r_test_index = np.unique(r_groups[test_index_pre_v_tone])

          # transform the full corresponding test trials through the transforms
          rt_test_index = (r_test_index[:, None] + np.arange(T)).flatten()

          # RT x N @ N x 2
          out_rt_k = filt_rt_n[rt_test_index[:, None], n_idx] @ scalings

          # RT x N @ N x 2
          out_rt_k = filt_rt_n[:, n_idx] @ scalings

        # restore nans
        out_rt_k[filt_nan_mask_rt] = np.nan
        proj_crtk[c] = np.reshape(out_rt_k, (R, T, K))

        if progress is not None:
          progress.update.remote(1)

    return proj_crtk

  # Split samples into batches
  C = n_samples
  sample_inds = np.arange(C)
  batch_size = (C + n_batches - 1) // n_batches  # Ceiling division to get batch size
  batches = [sample_inds[i : min(i + batch_size, C)] for i in range(0, C, batch_size)]

  if not parallel:
    proj_crtk = project_lda_pufftone_batch_samples(
      X, y, r_groups, filt_proj_rtn, samp_strat, lda_mode, range(C), None
    )
  else:
    ray_project_lda_pufftone_batch_samples = ray.remote(num_cpus=1)(
      project_lda_pufftone_batch_samples
    )

    # store as refs
    X_ref = ray.put(X)
    y_ref = ray.put(y)
    r_groups_ref = ray.put(r_groups)
    samp_strat_ref = ray.put(samp_strat)
    filt_proj_rtn_ref = ray.put(filt_proj_rtn)  # create reference to large array in object store
    if progress is None:
      progress = remote_tqdm.remote(total=C, desc="Running samples")
    result_ids = [
      ray_project_lda_pufftone_batch_samples.remote(
        X_ref, y_ref, r_groups_ref, filt_proj_rtn_ref, samp_strat_ref, lda_mode, batch, progress
      )
      for batch in batches
    ]

    results = ray.get(result_ids)
    flattened_results = [item for sublist in results for item in sublist]
    proj_crtk = np.array(flattened_results)

  return proj_crtk


def align_and_stack_axis2(data: list[np.ndarray], align_idx: list[int]) -> tuple[np.ndarray, int]:
  """
  Returns aligned tensor with shape
  (len(data), data[0].shape[0], data[0].shape[1], num_aligned_time, data[0].shape[3])
  """
  # Determine the maximum length needed to align all arrays
  max_pre = max(align_idx)
  max_post = max(arr.shape[2] - idx for arr, idx in zip(data, align_idx))
  num_aligned_time = max_pre + max_post

  # Initialize the output tensor with zeros (or another suitable default value)
  output_shape = (len(data), data[0].shape[0], data[0].shape[1], num_aligned_time, data[0].shape[3])
  aligned_tensor = np.full(output_shape, np.nan)

  for i, (arr, idx) in enumerate(zip(data, align_idx)):
    start = max_pre - idx
    end = start + arr.shape[2]
    aligned_tensor[i, :, :, start:end, :] = arr

  alignment_index = max_pre
  return aligned_tensor, alignment_index


@dataclass(eq=True, frozen=True)
class LDAProjResults:
  group: str | list[str]
  mouse_ids: NDArray[np.int32]  # sessions
  align_inds: NDArray[np.int32]  # sessions
  proj_s_crtk: list[NDArray[np.float64]]  # sessions [ cell samples x trials x time x 2 dims ]

  filt_strat_fit: FilteringStrategy
  filt_strat_project: FilteringStrategy
  samp_strat: NeuronSamplingStrategy
  lda_mode: str

  fps: float = 30

  @property
  def dt_sec(self) -> float:
    return 1.0 / float(self.fps)

  # def invalidated_before_event_inds(self, event_inds: NDArray[np.int32]) -> Self:
  #   """set accuracy_s_ct[s] data to nan after time where filtering peeks at data after event_inds[s]"""

  #   if len(self.accuracy_s_ct) != len(event_inds):
  #     raise ValueError("event_inds length must match n_sessions")

  #   accuracy_s_ct = self.accuracy_s_ct.copy()
  #   for s in range(len(accuracy_s_ct)):
  #     invalid_after_ind = event_inds[s] - self.filt_strat.window_buffer_post
  #     accuracy_s_ct[s][:, invalid_after_ind:] = np.nan

  #   return replace(self, accuracy_s_ct=accuracy_s_ct)

  def get_time_aligned_proj(self) -> tuple[NDArray, NDArray, int]:
    """returns proj_scrtk, time_rel_event, event_ind"""
    proj_scrtk, event_ind = align_and_stack_axis2(self.proj_s_crtk, self.align_inds)

    n_time = proj_scrtk.shape[3]
    T = (n_time - 1) * self.dt_sec
    time_vec = np.linspace(0, T, n_time)
    time_vec = time_vec - time_vec[event_ind]

    return proj_scrtk, time_vec, event_ind

  def averaged(
    self, over_samples: bool = True, over_trials: bool = False, omitnan: bool = False
  ) -> Self:
    axes_avg = []
    if over_samples:
      axes_avg.append(0)
    if over_trials:
      axes_avg.append(1)
    if len(axes_avg) == 0:
      return ValueError("must average at least one of over_samples or over_trials")

    mean = np.nanmean if omitnan else np.mean
    proj_s_crtk = [mean(a_crtk, axis=tuple(axes_avg), keepdims=True) for a_crtk in self.proj_s_crtk]
    return replace(self, proj_s_crtk=proj_s_crtk)

  def get_grand_avg_tk(self, omitnan: bool = False):
    proj_scrtk, time_vec, aligned_tone_ind = self.averaged(
      over_samples=True, over_trials=True, omitnan=omitnan
    ).get_time_aligned_proj()
    proj_tk = np.mean(np.squeeze(proj_scrtk, axis=(1, 2)), axis=0)
    return proj_tk, time_vec

  def get_avg_stk(self, omitnan: bool = False):
    proj_scrtk, time_vec, aligned_tone_ind = self.averaged(
      over_samples=True, over_trials=True, omitnan=omitnan
    ).get_time_aligned_proj()
    proj_stk = np.squeeze(proj_scrtk, axis=(1, 2))
    return proj_stk, time_vec

  # def to_df(self) -> pd.DataFrame:
  #   accuracy_sct, time_vec, event_ind = self.get_aligned_accuracy()

  #   S, C, T = accuracy_sct.shape
  #   s_idx, c_idx, t_idx = np.meshgrid(np.arange(S), np.arange(C), np.arange(T), indexing="ij")
  #   mouse_ids_flat = np.array(self.mouse_ids)[s_idx.flatten()]
  #   c_flat = c_idx.flatten()
  #   time_flat = time_vec[t_idx.flatten()]

  #   # time_rel_event = self.time_sec
  #   # time_rel_event = time_rel_event - time_rel_event[self.align_inds[s]]

  #   data = {
  #     "group": np.repeat(self.group, S * C * T),
  #     "mouse_id": mouse_ids_flat,
  #     "time": time_flat,
  #     "sample": c_flat,
  #     "accuracy": accuracy_sct.flatten(),
  #   }
  #   df = pd.DataFrame(data)
  #   return df


def multimouse_pipeline_project_lda_pufftone(
  mice: list[Session],
  filt_strat_fit: FilteringStrategy,
  filt_strat_project: FilteringStrategy,
  samp_strat: NeuronSamplingStrategy,
  seed: int = 1,
  n_samples: int = 10,
  n_batches: int = 8,
  lda_mode: str = "separate",
  group: str | None = None,
) -> LDAProjResults:
  if group is None:
    group = mice[0].group

  proj_s = []

  S = len(mice)
  C = n_samples
  progress = remote_tqdm.remote(total=S * C)
  align_inds = []
  for ind, mouse in enumerate(mice):
    align_inds.append(mouse.tone_ind)

    mouse_desc = f"{group} {ind} / {len(mice)}: id {mouse.mouse_id}"
    progress.set_description.remote(mouse_desc)
    proj_s.append(
      project_lda_pufftone_parallel_samples(
        mouse,
        filt_strat_fit=filt_strat_fit,
        filt_strat_project=filt_strat_project,
        samp_strat=samp_strat,
        progress=progress,
        lda_mode=lda_mode,
        n_samples=n_samples,
        n_batches=n_batches,
        seed=ind,
      )
    )

  mouse_ids = [mouse.mouse_id for mouse in mice]
  # accuracy_sct, event_ind = align_and_stack_matrices(acc_s, event_inds)
  res = LDAProjResults(
    group=group,
    mouse_ids=mouse_ids,
    align_inds=align_inds,
    proj_s_crtk=proj_s,
    filt_strat_fit=filt_strat_fit,
    filt_strat_project=filt_strat_project,
    samp_strat=samp_strat,
    lda_mode=lda_mode,
  )
  return res


def multimouse_pipeline_project_wholetrial_lda_pufftone(
  mice: list[Session],
  filt_strat_fit: FilteringStrategy,
  filt_strat_project: FilteringStrategy,
  samp_strat: NeuronSamplingStrategy,
  seed: int = 1,
  n_samples: int = 10,
  n_batches: int = 8,
  lda_mode: str = "separate",
  parallel: bool = True,
  group: str | None = None,
) -> LDAProjResults:
  if group is None:
    group = mice[0].group

  proj_s = []

  S = len(mice)
  C = n_samples
  progress = remote_tqdm.remote(total=S * C)
  align_inds = []
  for ind, mouse in enumerate(mice):
    align_inds.append(mouse.tone_ind)

    mouse_desc = f"{group} {ind} / {len(mice)}: id {mouse.mouse_id}"
    progress.set_description.remote(mouse_desc)
    proj_s.append(
      project_wholetrial_lda_pufftone_parallel_samples(
        mouse,
        filt_strat_fit=filt_strat_fit,
        filt_strat_project=filt_strat_project,
        samp_strat=samp_strat,
        progress=progress,
        lda_mode=lda_mode,
        parallel=parallel,
        n_samples=n_samples,
        n_batches=n_batches,
        seed=ind,
      )
    )

  mouse_ids = [mouse.mouse_id for mouse in mice]
  # accuracy_sct, event_ind = align_and_stack_matrices(acc_s, event_inds)
  res = LDAProjResults(
    group=group,
    mouse_ids=mouse_ids,
    align_inds=align_inds,
    proj_s_crtk=proj_s,
    filt_strat_fit=filt_strat_fit,
    filt_strat_project=filt_strat_project,
    samp_strat=samp_strat,
    lda_mode=lda_mode,
  )
  return res
