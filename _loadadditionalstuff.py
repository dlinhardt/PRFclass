from glob import glob
from os import makedirs, path

import nibabel as nib
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import zoom
from scipy.signal import fftconvolve
from tqdm import tqdm
import warnings

# imported as a plain function: PRF exposes it as a class attribute, so
# self.spm_hrf_compat would bind self as the time vector
from ._datastuff import spm_hrf_compat

#: gem's default slice timing reference. It only lives in the gem XML config, not
#: in the estimates h5, so it cannot be read back from the results.
GEM_SLICE_TIME_REF = 0.5

#: gem's default number of DCT drift regressors (metadata/analysis/nDCT).
GEM_NDCT = 3


# ----------------------------------------------------------------------------#
def loadStim(self, buildTC=True):
    """
    Load stimulus images from results.mat and build the model time course.

    Args:
        buildTC (bool, optional): Whether to calculate the model time course. Defaults to True.

    Returns:
        np.ndarray or None: Model time course if built, otherwise None.
    """

    if self._dataFrom == "mrVista":
        self.window = (
            self.params["stim"]["stimwindow"].flatten().reshape(101, 101).astype("bool")
        )
        self.stimImages = self.params["analysis"]["allstimimages"].T
        try:
            self.stimImagesUnConv = self.params["analysis"][
                "allstimimages_unconvolved"
            ].T
        except:
            print("could not load allstimimages_unconvolved!")

        if buildTC:
            self.X0 = self.params["analysis"]["X"]
            self.Y0 = self.params["analysis"]["Y"]

            pRF = np.exp(
                (
                    (self.x[:, None] - self.X0[None, :]) ** 2
                    + (self.y[:, None] - self.Y0[None, :]) ** 2
                )
                / (-2 * self.s[:, None] ** 2)
            )

            self.modelTC = pRF.dot(self.stimImages)

    else:
        size = np.sqrt(len(self.params[0]["stim"]["stimwindow"][0][0])).astype(int)
        self.window = (
            self.params[0]["stim"]["stimwindow"][0][0]
            .flatten()
            .reshape(size, size)
            .astype("bool")
        )
        self.stimImages = self.params[0]["analysis"]["allstimimages"][0][0].T
        self.stimImagesUnConv = self.params[0]["analysis"]["allstimimages_unconvolved"][
            0
        ][0].T

        self.X0 = self.params[0]["analysis"]["X"][0][0].flatten()
        self.Y0 = self.params[0]["analysis"]["Y"][0][0].flatten()

        if buildTC:
            pRF = np.exp(
                (
                    (-self.y0[:, None] - self.Y0[None, :]) ** 2
                    + (self.x0[:, None] - self.X0[None, :]) ** 2
                )
                / (-2 * self.s0[:, None] ** 2)
            )

            self._modelTC0 = (
                self.beta0[:, None] * pRF.dot(self.stimImages)
                + self.voxelTC0.mean(1)[:, None]
            )

            return self._modelTC0


# ----------------------------------------------------------------------------#
def loadJitter(self):
    """
    Load the EyeTracker jitter file (.mat) and return x/y jitter.

    Returns:
        tuple: (jitterX, jitterY) arrays for both dimensions.

    Raises:
        FileNotFoundError: If the jitter file is not found.
    """

    if self._dataFrom == "mrVista":
        self.jitterP = path.join(
            self._baseP,
            self._study,
            "subjects",
            self.subject,
            self.session,
            "mrVista",
            self._analysis,
            "Stimuli",
            f'{self.subject}_{self.session}_{self._analysis.split("ET")[0][:-1]}_jitter.mat',
        )

        if path.exists(self.jitterP):
            jitter = loadmat(self.jitterP, simplify_cells=True)
            self.jitterX = jitter["x"] - jitter["x"].mean()
            self.jitterY = jitter["y"] - jitter["y"].mean()
            return self.jitterX, self.jitterY
        else:
            raise FileNotFoundError("No jitter file found at: " + self.jitterP)

    elif self._dataFrom == "docker":
        self.jitterP = path.join(
            self._baseP,
            self._study,
            "BIDS",
            "etdata",
            self.subject,
            self.session,
            f'{self.subject}_{self.session}_task-{self._task.split("-")[0]}_run-0{self._run}_gaze.mat',
        )

        if path.exists(self.jitterP):
            jitter = loadmat(self.jitterP, simplify_cells=True)

            self.jitterX = jitter["x"] - jitter["x"].mean()
            self.jitterY = jitter["y"] - jitter["y"].mean()

            return self.jitterX, self.jitterY
        else:
            raise Warning("No jitter file found!")


# ----------------------------------------------------------------------------#
def loadRealign(self):
    """
    Load realignment parameters and calculate framewise displacement.

    Raises:
        RuntimeError: If called for non-mrVista data.
    """

    if self._dataFrom == "mrVista":
        # framewise displacement
        displ = np.loadtxt(path.join(self.niiFolder, "rp_avols.txt"))
        displ = np.abs(np.diff(displ * [1, 1, 1, 50, 50, 50], axis=0))
        self.dispS = displ.sum()
        self.dispM = displ.mean()
    else:
        raise RuntimeError("[loadRealign] is only possible with data from mrVista!")


# ----------------------------------------------------------------------------#
# gem model time courses
#
# gem does not store a usable modelpred: both write paths in
# run_gem_prf_analysis.py pass refined_signals_present=False, so
# parameters/modelpred is a (nVertices, 1) placeholder. The functions below
# rebuild the model time course exactly the way gem computes it, so that a
# recomputed R2 reproduces gem's own R2. See:
#   gem/model/prf_stimulus.py            resample + HRF convolution
#   gem/kernels/gaussian_kernel.cu       pRF curves, row <-> y, col <-> x
#   gem/signals/signal_synthesizer.py    curves @ stimulus, then downsampling
#   gem/signals/orthogonalization_matrix.py  the DCT drift basis
# ----------------------------------------------------------------------------#


def _gem_ortho_q(n_frames, nDCT=GEM_NDCT):
    """Orthonormal basis of gem's low-frequency drift regressors.

    gem builds ``O = I - Q @ Q.T`` from this and residualises both data and
    model against it. Returning Q instead of O keeps the memory at
    ``n_frames x (2 * nDCT + 1)`` instead of ``n_frames**2``.
    """
    tc = np.linspace(0, 2.0 * np.pi, n_frames)[:, None]
    trends = np.cos(tc.dot(np.arange(0, nDCT + 0.5, 0.5)[None, :]))
    q, _ = np.linalg.qr(trends)
    return q * np.sign(q[0, 0])


def _gem_detrend(a, q):
    """Apply gem's ``O = I - Q Q.T`` to every row of ``a``."""
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    return a - (a @ q) @ q.T


def r2_against(self, signal, tc, nDCT=GEM_NDCT):
    """Variance explained of a model signal against a measured time course.

    This is gem's R2 (``gem/analysis/prf_r2_variance_explain.py``), which
    reduces to the squared cosine between the drift-residualised data and the
    drift-residualised model. Scale-free, so ``signal`` does not have to carry
    the right amplitude.

    Use it to pair any fit's model signal with any run's data, which is what a
    cross-validated R2 is: run-1 parameters scored against run-2 data.

    Args:
        signal (np.ndarray): Model time course(s), (nFrames,) or (n, nFrames).
        tc (np.ndarray): Measured time course(s), same shape as signal.
        nDCT (int, optional): DCT drift regressors. Defaults to gem's 3.

    Returns:
        np.ndarray: R2 per row.
    """
    signal = np.atleast_2d(np.asarray(signal, dtype=np.float64))
    tc = np.atleast_2d(np.asarray(tc, dtype=np.float64))
    if signal.shape != tc.shape:
        raise ValueError(
            f"signal {signal.shape} and tc {tc.shape} must have the same shape"
        )

    q = _gem_ortho_q(signal.shape[-1], nDCT)
    s = _gem_detrend(signal, q)
    y = _gem_detrend(tc, q)

    sy = np.einsum("ij,ij->i", s, y)
    ss = np.einsum("ij,ij->i", s, s)
    yy = np.einsum("ij,ij->i", y, y)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ss * yy > 0, sy**2 / (ss * yy), np.nan)


def _gem_scale_to_data(self, signal, tc, nDCT=GEM_NDCT):
    """Least-squares fit of ``[signal, DCT basis]`` to ``tc``.

    Returns the fitted prediction, so it can be drawn on the same axes as the
    data. Q spans the DCT basis, so the projection below is the exact OLS
    solution without forming the design matrix per vertex.
    """
    signal = np.atleast_2d(np.asarray(signal, dtype=np.float64))
    tc = np.atleast_2d(np.asarray(tc, dtype=np.float64))

    q = _gem_ortho_q(signal.shape[-1], nDCT)
    s = _gem_detrend(signal, q)
    y = _gem_detrend(tc, q)

    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.einsum("ij,ij->i", s, y) / np.einsum("ij,ij->i", s, s)
    beta = np.nan_to_num(beta)[:, None]

    resid = tc - beta * signal
    return beta * signal + (resid @ q) @ q.T


# ----------------------------------------------------------------------------#
def _gem_num_frames(self):
    """Length of one run's measured time course, in TRs.

    Taken from an input file header rather than from voxelTC0, because a
    concatenated gem fit has no BOLD file of its own and voxelTC0 would fail.
    """
    for f in getattr(self, "_gem_input_files", []) or []:
        f = f.decode() if isinstance(f, bytes) else str(f)
        try:
            return int(nib.load(f).shape[-1])
        except Exception:
            continue
    return int(self.voxelTC0.shape[-1])


def _gem_stimulus_file(self):
    """Absolute path of the aperture NIfTI.

    gem records a file for an individual run but only the stimulus *directory*
    for a concatenated one, so both have to be handled.
    """
    p = self._gem_stimulus_filepath
    if path.isfile(p):
        return p

    if path.isdir(p):
        task = self.task if self.task.startswith("task-") else f"task-{self.task}"
        hits = sorted(glob(path.join(p, f"{task}_*apertures.nii*"))) or sorted(
            glob(path.join(p, "*.nii*"))
        )
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise FileNotFoundError(
                f"{p} holds several candidate apertures, cannot pick one: "
                f"{[path.basename(x) for x in hits]}"
            )

    raise FileNotFoundError(f"gem stimulus not found at {p}")


def _gem_stim_cache_path(self, n_frames):
    """Where the HRF-convolved stimulus is cached.

    The cache depends only on the aperture, the grid, the HRF and the
    downsampled length, so every subject and run sharing those reuses one file.
    """
    h = self._gem_hrf
    stim = path.basename(self._gem_stimulus_file()).split(".")[0]
    info = self.stimulusInfo
    name = (
        f"{stim}_grid-{int(info['height'])}x{int(info['width'])}"
        f"_vf-{float(info['visual_field']):g}_frames-{n_frames}"
        f"_hrf-{float(h['peak_delay']):g}-{float(h['under_shoot_delay']):g}"
        f"-{float(h['peak_disp']):g}-{float(h['under_disp']):g}"
        f"-{float(h['peak_to_undershoot']):g}"
        f"_str-{GEM_SLICE_TIME_REF:g}.npy"
    )
    base = path.join(self._derivatives_path, "prfresult", "gemStimCache")
    makedirs(base, exist_ok=True)
    return path.join(base, name)


def _gem_convolved_stimulus(self, force=False):
    """HRF-convolved stimulus, resampled to gem's grid and downsampled to TRs.

    Mirrors ``Stimulus.compute_resample_stimulus_data`` plus
    ``compute_hrf_convolved_stimulus_data``, then applies the downsampling from
    ``SignalSynthesizer.compute_signals_batches``. Returns ``(nPixels, nTRs)``
    in C order, matching the pRF curve layout of gem's CUDA kernel.
    """
    n_frames = self._gem_num_frames()
    cache = self._gem_stim_cache_path(n_frames)
    if path.isfile(cache) and not force:
        return np.load(cache)

    stim_p = self._gem_stimulus_file()

    info = self.stimulusInfo
    H, W = int(info["height"]), int(info["width"])
    img = nib.load(stim_p)
    dt = float(img.header["pixdim"][4])  # stimulus frame time, not the fMRI TR
    T = int(img.shape[-1])

    h = self._gem_hrf
    hrf = spm_hrf_compat(
        np.arange(float(h["t_start"]), float(h["t_stop"]), dt),
        peak_delay=float(h["peak_delay"]),
        under_delay=float(h["under_shoot_delay"]),
        peak_disp=float(h["peak_disp"]),
        under_disp=float(h["under_disp"]),
        p_u_ratio=float(h["peak_to_undershoot"]),
        normalize=bool(h["normalize"]),
    )

    # zoom in time chunks: the time factor is exactly 1.0, so chunking is
    # lossless and the full-resolution stack never has to fit in memory
    step = 200
    res = np.empty((H, W, T), dtype=np.float32)
    zoom_f = (H / img.shape[0], W / img.shape[1], 1.0)
    for t0 in tqdm(
        range(0, T, step), desc="resampling stimulus", unit="chunk", leave=False
    ):
        blk = np.asarray(img.dataobj[..., t0 : t0 + step]).squeeze().astype(np.float32)
        if bool(info.get("binarization_enabled", False)) and not np.all(
            np.isin(blk, (0, 1))
        ):
            blk = (blk > float(info["binarization_threshold"])).astype(np.float32)
        res[..., t0 : t0 + blk.shape[-1]] = zoom(blk, zoom_f, order=1)

    conv = fftconvolve(res.reshape(H * W, T), hrf[None, :], mode="full", axes=1)[:, :T]
    del res

    idx = np.linspace(0, T, n_frames, endpoint=False, dtype=int)
    idx += int(np.round(np.diff(idx).mean() * GEM_SLICE_TIME_REF))
    conv = np.ascontiguousarray(conv[:, idx], dtype=np.float32)

    np.save(cache, conv)
    return conv


def _gem_modelsignal(self, force=False, chunk=2000):
    """Per-vertex model time course for gem results, unscaled.

    The amplitude is left out on purpose: gem's R2 is scale free, and a
    concatenated fit has no data of its own to fit an amplitude against.
    """
    conv = self._gem_convolved_stimulus(force=force)

    info = self.stimulusInfo
    H, W = int(info["height"]), int(info["width"])
    vf = float(info["visual_field"])
    # gem's kernel walks rows in y and columns in x and flattens C order, so
    # meshgrid's default indexing already matches
    X, Y = np.meshgrid(np.linspace(-vf, vf, W), np.linspace(-vf, vf, H))
    xf, yf = X.ravel(), Y.ravel()

    mux = self.x0.astype(np.float64)
    # gem stores Centery0; PRFclass flips it for visual field orientation
    muy = -self.y0.astype(np.float64) if self._orientation == "VF" else self.y0.astype(np.float64)
    sig = self.s0.astype(np.float64)

    out = np.zeros((len(mux), conv.shape[1]), dtype=np.float32)
    ok = np.isfinite(mux) & np.isfinite(muy) & np.isfinite(sig) & (sig > 0)
    idx_ok = np.flatnonzero(ok)

    for i0 in tqdm(
        range(0, len(idx_ok), chunk), desc="model signals", unit="chunk", leave=False
    ):
        sel = idx_ok[i0 : i0 + chunk]
        g = np.exp(
            -(
                (xf[None, :] - mux[sel, None]) ** 2
                + (yf[None, :] - muy[sel, None]) ** 2
            )
            / (2 * sig[sel, None] ** 2)
        ).astype(np.float32)
        out[sel] = g @ conv

    return out
