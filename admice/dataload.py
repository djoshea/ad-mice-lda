import glob
import numpy as np
import numpy.typing as npt
import scipy.io as sio

# import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class Session:
  group: str
  mouse_id: int
  data_nrt: npt.NDArray[np.float64]  # neurons x trials x frames

  tone_time: float
  puff_time: float
  trial_has_tone: npt.NDArray[np.bool_]
  trial_has_puff: npt.NDArray[np.bool_]

  fps: float = 30

  @property
  def trialavg_nt(self):
    return np.mean(self.data_nrt, axis=1)

  @property
  def n_neurons(self) -> int:
    return self.data_nrt.shape[0]

  @property
  def n_trials(self) -> int:
    return self.data_nrt.shape[1]

  @property
  def n_time(self) -> int:
    return self.data_nrt.shape[2]

  @property
  def dt_sec(self) -> float:
    return 1.0 / float(self.fps)

  @property
  def tone_ind(self) -> int:
    return int(self.tone_time // self.dt_sec)

  @property
  def puff_ind(self) -> int:
    return int(self.puff_time // self.dt_sec)

  @property
  def time_sec(self) -> np.ndarray:
    T = (self.n_time - 1) * self.dt_sec
    return np.linspace(0, T, self.n_time)

  def heatmap_trialavg(self, ax):
    # extent : floats (left, right, bottom, top)
    dt = self.dt_sec
    half_dt = 0.5 * dt
    extent = [-half_dt, self.n_frames * dt - half_dt, self.n_neurons - 0.5, -0.5]
    ax.imshow(self.trialavg_nt, extent=extent)
    ax.set_xlabel("time (sec)")
    ax.set_ylabel("neurons")
    ax.set_aspect(0.01)
    ax.grid(False)


def session_from_mouse_from_matfile(mouse: list, group: str) -> Session:
  """Builds a Session given one entry of the mouse list in Ted's mat files"""
  mouse_id = mouse[0]
  data_nrt = mouse[1]
  tone_time = mouse[2] / 30.0
  puff_time = mouse[3] / 30.0

  # which trials have a tone
  # from ted: the data has 65 trials, with the first 5 being tone only, then 15 25 35 45 55 65 tone only, and the rest tone-puff paired trials.
  n_trials = data_nrt.shape[1]
  trial_has_tone = np.ones(n_trials, dtype=bool)
  trial_has_puff = np.ones(n_trials, dtype=bool)
  trial_has_puff[:5] = False
  trial_has_puff[14::5] = False

  return Session(
    group=group,
    mouse_id=mouse_id,
    data_nrt=data_nrt,
    tone_time=tone_time,
    puff_time=puff_time,
    trial_has_tone=trial_has_tone,
    trial_has_puff=trial_has_puff,
  )


def load_mat(fname: str) -> dict:
  """Returns the raw data in a mat file"""
  return sio.loadmat(fname, struct_as_record=True, squeeze_me=True, chars_as_strings=True)


def load_mice_from_mat(fname: str, key: str, group: str) -> list[Session]:
  ld = load_mat(fname)
  mice = ld[key][1:]  # first entry is string description
  sessions = [session_from_mouse_from_matfile(mouse, group) for mouse in mice]
  return sessions
