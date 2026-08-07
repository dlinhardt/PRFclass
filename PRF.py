import json
from glob import glob
from os import path

import nibabel as nib
import numpy as np
from scipy.io import loadmat

class PRF:
    """
    Population Receptive Field (pRF) analysis object.

    This class provides an interface for loading, managing, masking, and analyzing
    pRF mapping results from various sources (mrVista, docker, samsrf, hdf5).
    Use the alternative constructors (from_mrVista, from_docker, etc.) to initialize.
    """

    # load in all the other files with functions
    from ._calculatestuff import (
        calc_kde_diff,
        calc_prf_profiles,
        central_scot_border,
        plot_kdeDiff2d,
    )
    from ._datastuff import (
        from_docker,
        from_file,
        from_hdf5,
        from_mrVista,
        from_samsrf,
        init_variables,
        spm_hrf_compat,
    )
    from ._loadadditionalstuff import (
        _gem_convolved_stimulus,
        _gem_modelsignal,
        _gem_num_frames,
        _gem_scale_to_data,
        _gem_stim_cache_path,
        _gem_stimulus_file,
        loadJitter,
        loadRealign,
        loadStim,
        r2_against,
    )
    from ._maskstuff import (
        _calcMask,
        _get_available_atlases_and_rois,
        list_atlases_and_rois,
        maskBetaThresh,
        maskDartBoard,
        maskEcc,
        maskROI,
        maskSigma,
        maskVarExp,
    )
    from ._flatmapstuff import plot_toFlatmap, prepare_flatmap_surface
    from ._plotstuff import (
        _build_surface_overlay,
        _calcCovMap,
        _createmask,
        _get_covMap_savePath,
        _get_surfaceSavePath,
        _make_gif,
        manual_masking,
        plot_covMap,
        plot_toSurface,
        save_results,
    )

    from_docker = classmethod(from_docker)
    from_mrVista = classmethod(from_mrVista)
    from_samsrf = classmethod(from_samsrf)
    from_file = classmethod(from_file)
    from_hdf5 = classmethod(from_hdf5)

    def __init__(
        self,
        dataFrom,
        study,
        subject,
        session,
        baseP,
        derivatives_path=None,
        mat=None,
        est=None,
        analysis=None,
        task=None,
        run=None,
        area=None,
        coords=None,
        niftiFolder=None,
        hemis=None,
        prfanaMe=None,
        prfanaAn=None,
        orientation="VF",
        method=None,
    ):
        """
        Initialize a PRF object. Use alternative constructors for typical usage.

        Args:
            dataFrom (str): Data source ('mrVista', 'docker', 'samsrf', 'hdf5').
            study (str): Study name.
            subject (str): Subject identifier.
            session (str): Session identifier.
            baseP (str): Base path for data.
            derivatives_path (str, optional): Path to derivatives folder.
            mat (dict, optional): Loaded .mat file data.
            est (dict, optional): Estimates data.
            analysis (str, optional): Analysis name.
            task (str, optional): Task name.
            run (str, optional): Run identifier.
            area (str or list, optional): ROI area(s).
            coords (np.ndarray, optional): Coordinates array.
            niftiFolder (str, optional): Path to NIfTI folder.
            hemis (str or list, optional): Hemisphere(s).
            prfanaMe (str, optional): prfanalyze method.
            prfanaAn (str, optional): prfanalyze analysis.
            orientation (str, optional): 'VF' or 'MP'. Defaults to 'VF'.
            method (str, optional): Analysis method.
        """
        self._dataFrom = dataFrom
        self._study = study
        self._subject = subject
        self._session = session
        self._baseP = baseP
        self._area = "full"

        if mat:
            self._mat = mat
        if est:
            self._estimates = est
        if analysis:
            self._analysis = analysis
        if task:
            self._task = task
        if run:
            self._run = run
        if area:
            self._area = area
        if prfanaMe:
            self._prfanalyze_method = prfanaMe
        if prfanaAn:
            self._prfanaAn = prfanaAn
        if niftiFolder:
            self._niftiFolder = niftiFolder
        if coords is not None:
            self._coords = coords
        if hemis is not None:
            self._hemis = hemis
        if derivatives_path is not None:
            self._derivatives_path = derivatives_path

        self._orientation = orientation.upper()

        if self._dataFrom == "mrVista":
            self._model = self._mat["model"]
            self._params = self._mat["params"]
        elif self._dataFrom == "docker":
            if hasattr(self, "_mat"):
                self._model = [m["model"][0][0][0][0] for m in self._mat]
                self._params = [m["params"][0][0][0][0] for m in self._mat]
        elif self._dataFrom == "samsrf":
            self._model = [m["Srf"][0][0] for m in self._mat]
            self._params = [m["Model"][0][0] for m in self._mat]

        # initialize
        self.init_variables()

    # All properties below should have a short docstring describing what they return.
    @property
    def subject(self):
        """str: Subject identifier."""
        return self._subject

    @property
    def session(self):
        """str: Session identifier."""
        return self._session

    @property
    def task(self):
        """str: Task name."""
        return self._task

    @property
    def run(self):
        """str: Run identifier."""
        return self._run

    @property
    def y0(self):
        """np.ndarray: Unmasked y0 pRF parameter."""
        return self._y0.astype(np.float32)

    @property
    def x0(self):
        """np.ndarray: Unmasked x0 pRF parameter."""
        return self._x0.astype(np.float32)

    @property
    def s0(self):
        """np.ndarray: Unmasked sigma (size) pRF parameter."""
        return self._s0.astype(np.float32)

    @property
    def sigma0(self):
        """np.ndarray: Alias for s0 (unmasked sigma)."""
        return self._s0.astype(np.float32)

    @property
    def r0(self):
        """np.ndarray: Unmasked eccentricity (radius) pRF parameter."""
        return np.sqrt(np.square(self.x0) + np.square(self.y0)).astype(np.float32)

    @property
    def ecc0(self):
        """np.ndarray: Alias for r0 (unmasked eccentricity)."""
        return np.sqrt(np.square(self.x0) + np.square(self.y0)).astype(np.float32)

    @property
    def phi0(self):
        """np.ndarray: Unmasked polar angle (0 to 2pi)."""
        p = np.arctan2(self.y0, self.x0)
        p[p < 0] += 2 * np.pi
        return p.astype(np.float32)

    @property
    def phi0_orig(self):
        """np.ndarray: Unmasked polar angle (original, -pi to pi)."""
        p = np.arctan2(self.y0, self.x0)
        return p.astype(np.float32)

    @property
    def pol0(self):
        """np.ndarray: Alias for phi0 (unmasked polar angle)."""
        return self.phi0.astype(np.float32)

    @property
    def pol0_orig(self):
        """np.ndarray: Alias for phi0_orig (unmasked polar angle, original)."""
        return self.phi0_orig.astype(np.float32)

    @property
    def beta0(self):
        """np.ndarray: Unmasked beta parameter."""
        return self._beta0.astype(np.float32)

    @property
    def varexp0(self):
        """np.ndarray: Unmasked variance explained."""
        return self._varexp0.astype(np.float32)

    @property
    def varexp_easy0(self):
        """
        np.ndarray: Unmasked 'easy' variance explained.

        Calculated if not present, using a simple model fit.
        """
        if not hasattr(self, "_varexp_easy"):
            print(
                f"calculating varexp_easy for {self.subject}_{self.session}_{self.task}_{self.run}..."
            )
            l = self.voxelTC0.shape[-1]
            trends = np.vstack(
                (
                    np.ones(l),
                    np.linspace(-1, 1, l),
                    np.linspace(-1, 1, l) ** 2,
                    np.linspace(-1, 1, l) ** 3,
                )
            )

            beta_easy = np.empty((len(self.modelpred0), len(trends) + 1))
            varexp_easy = np.empty((len(self.modelpred0)))

            def fitter(mod, tc):
                l = len(mod)
                X = np.vstack((mod, trends))

                pinvX = np.linalg.pinv(X)
                b = pinvX.T @ tc
                return b

            for i, (mod, tc) in enumerate(zip(self.modelpred0, self.voxelTC0)):
                mod -= mod.mean()
                b = fitter(mod, tc)

                rawrss = np.sum(np.square(tc - b[1:] @ trends))
                rss = np.sum(np.square(tc - b @ np.vstack((mod, trends))))
                ve = 1 - rss / rawrss

                beta_easy[i, :] = b
                varexp_easy[i] = ve

            self._beta_easy = beta_easy
            self._varexp_easy = varexp_easy

        return self._varexp_easy.astype(np.float32)

    @property
    def voxelTC0(self):
        """
        np.ndarray: Unmasked voxel time courses.

        Loaded from source if not already present.
        """
        if not hasattr(self, "_voxelTC0"):
            if self._dataFrom == "mrVista":
                self._voxelTC0 = loadmat(
                    glob(
                        path.join(
                            self._baseP,
                            self._study,
                            "subjects",
                            self.subject,
                            self.session,
                            "mrVista",
                            self._analysis,
                            "Gray/*/TSeries/Scan1/tSeries1.mat",
                        )
                    )[0],
                    simplify_cells=True,
                )["tSeries"].T

            elif self._dataFrom == "docker":
                try:
                    self._voxelTC0 = np.array(
                        [e["testdata"] for ee in self._estimates for e in ee]
                    )
                except:
                    hemis = ["L", "R"] if not self._hemis else self._hemis

                    tc = []

                    for hemi in hemis:
                        p = glob(
                            path.join(
                                self._baseP,
                                self._study,
                                self._derivatives_path,
                                "prfprepare",
                                f"analysis-{self.prfprepare_analysis}",
                                self.subject,
                                self.session,
                                "func",
                                f"{self.subject}_{self.session}_{self.task}_{self.run}_hemi-{hemi}_bold.nii*",
                            )
                        )[0]
                        tc.append(nib.load(p).get_fdata().squeeze())

                    self._voxelTC0 = np.concatenate(tc, axis=0)

            elif self._dataFrom == "samsrf":
                self._voxelTC0 = self._mat["Srf"]["Y"][0][0].T

            np.seterr(invalid="ignore")
            self._voxelTCpsc0 = self._voxelTC0 / self._voxelTC0.mean(1)[:, None] * 100

        return self._voxelTC0.astype(np.float32)

    def _stored_modelpred(self):
        """Model predictions as stored by the analysis, or None if unusable.

        Handles both estimate layouts: the older list-of-dicts (one dict per
        voxel) and gem's dict-of-arrays (one dict per hemisphere).

        gem currently never stores a real one -- both write paths in
        run_gem_prf_analysis.py pass refined_signals_present=False -- so what
        lands in the h5 is a (nVertices, 1) placeholder. Returning None here
        sends the caller to the recomputation path.
        """
        try:
            if isinstance(self._estimates[0], dict):
                mp = np.vstack([np.asarray(e["modelpred"]) for e in self._estimates])
            else:
                mp = np.array([e["modelpred"] for ee in self._estimates for e in ee])
        except (KeyError, TypeError, IndexError, ValueError):
            return None, "no modelpred in the estimates"

        mp = np.atleast_2d(mp)
        if mp.shape[0] != len(self.x0):
            return None, f"modelpred has {mp.shape[0]} rows for {len(self.x0)} vertices"
        if mp.shape[-1] < 2:
            return None, f"modelpred is a placeholder of shape {mp.shape}"
        if not np.isfinite(mp).any() or not np.any(mp != 0):
            return None, "modelpred is all NaN or all zero"

        # some older vistasoft runs wrote the measured data into modelpred
        try:
            tc = self.voxelTC0
        except Exception:
            tc = None
        if tc is not None and mp.shape == tc.shape:
            probe = slice(None, None, max(1, len(mp) // 50))
            if np.allclose(
                mp[probe] - mp[probe].mean(1)[:, None],
                tc[probe] - tc[probe].mean(1)[:, None],
            ):
                return None, "stored modelpred is a copy of the measured data"

        return mp, None

    @property
    def modelsignal0(self):
        """np.ndarray: Unmasked model time courses, without amplitude scaling.

        A property of the fit alone, so it also works for a concatenated gem
        run, which has no BOLD file of its own. Pair it with any run's data
        through `r2_against`; `modelpred0` is this scaled to `voxelTC0`.
        """
        if not hasattr(self, "_modelsignal0"):
            if self._dataFrom != "docker":
                raise RuntimeError(
                    f"modelsignal0 is only available for docker data, not "
                    f"{self._dataFrom!r}"
                )

            stored, why = self._stored_modelpred()
            if stored is not None:
                self._modelsignal0 = stored
            else:
                print(
                    f"Warning: {self.subject} {self.session} {self.task} "
                    f"{self.run}: {why}; recomputing the model time courses "
                    "from the stimulus. This is cached, the next call is fast."
                )
                self._modelsignal0 = self._gem_modelsignal()

        return self._modelsignal0.astype(np.float32)

    @property
    def modelsignal(self):
        return self.modelsignal0[self.mask, :]

    @property
    def modelpred0(self):
        """
        np.ndarray: Unmasked model predictions, scaled to the measured data.

        Loaded or calculated from source if not already present.
        """
        if not hasattr(self, "_modelpred0"):
            if self._dataFrom == "mrVista":
                print("No modelpred with data_from mrVista")
                return None
            elif self._dataFrom == "docker":
                try:
                    tc = self.voxelTC0
                except IndexError as exc:
                    raise RuntimeError(
                        f"No BOLD file for {self.subject} {self.session} "
                        f"{self.task} {self.run}, so a model prediction cannot "
                        "be scaled to it. A concatenated gem fit has no run of "
                        "its own; use modelsignal0 and pair it with the data "
                        "you want via r2_against()."
                    ) from exc
                self._modelpred0 = self._gem_scale_to_data(self.modelsignal0, tc)
            elif self._dataFrom == "samsrf":
                self._modelpred0 = np.apply_along_axis(
                    lambda m: np.convolve(m, self._mat["Model"]["Hrf"][0][0].flatten()),
                    0,
                    aa := self._mat["Srf"]["X"][0][0] * self.beta0,
                )[
                    : len(aa), :
                ]  #

        return self._modelpred0.astype(np.float32)

    @property
    def maxEcc(self):
        """np.float32: Maximum eccentricity of the *stimulus*.

        This is the stimulus aperture radius, not the model fitting grid. For
        gem results the grid is deliberately larger (e.g. a search radius of 15
        for a stimulus of 10) so pRFs can sit partly outside the aperture; see
        `searchSpace` for that.
        """
        return np.float32(self._maxEcc)

    @property
    def searchSpace(self):
        """dict or None: gem's pRF fitting grid (radius, grid steps, sigma range).

        `searchSpace["visual_field_radius"]` is where fits that ran to the grid
        boundary pile up, so it is useful for masking those artefacts out.
        """
        return getattr(self, "_gem_search_space", None)

    @property
    def stimulusInfo(self):
        """dict or None: gem's stimulus metadata (visual field, size, binarization)."""
        return getattr(self, "_gem_stimulus", None)

    @property
    def model(self):
        """Model object(s) loaded from source."""
        return self._model

    @property
    def params(self):
        """Parameter object(s) loaded from source."""
        return self._params

    @property
    def available_atlases_and_rois(self):
        """dict: All atlases and ROIs available for this subject/session.

        Returns {atlas_name: [roi1, roi2, ...]} and prints a summary table.
        Works regardless of whether RoiPack or legacy maskinfo.json files are present.
        """
        return self.list_atlases_and_rois()

    # -------------------------- DO MASKING? --------------------------#
    @property
    def doROIMsk(self):
        return self._doROIMsk

    @doROIMsk.setter
    def doROIMsk(self, value: bool):
        self._doROIMsk = value

    @property
    def doVarExpMsk(self):
        return self._doVarExpMsk

    @doVarExpMsk.setter
    def doVarExpMsk(self, value: bool):
        self._doVarExpMsk = value

    @property
    def doBetaMsk(self):
        return self._doBetaMsk

    @doBetaMsk.setter
    def doBetaMsk(self, value: bool):
        self._doBetaMsk = value

    @property
    def doEccMsk(self):
        return self._doEccMsk

    @doEccMsk.setter
    def doEccMsk(self, value: bool):
        self._doEccMsk = value

    @property
    def doSigMsk(self):
        return self._doSigMsk

    @doSigMsk.setter
    def doSigMsk(self, value: bool):
        self._doSigMsk = value

    @property
    def doManualMsk(self):
        return self._doManualMsk

    @doManualMsk.setter
    def doManualMsk(self, value: bool):
        self._doManualMsk = value

    # --------------------------- GET MASKS ---------------------------#
    @property
    def mask(self):
        return self._calcMask().astype(bool)

    @property
    def roiMsk(self):
        return self._roiMsk.astype(bool)

    @property
    def varExpMsk(self):
        return self._varExpMsk.astype(bool)

    @property
    def eccMsk(self):
        return self._eccMsk.astype(bool)

    @property
    def sigMsk(self):
        return self._sigMsk.astype(bool)

    @property
    def betaMsk(self):
        return self._betaMsk.astype(bool)

    @property
    def manualMsk(self):
        return self._manual_mask.astype(bool)

    @manualMsk.setter
    def manualMsk(self, value: bool):
        if len(value) != len(self.x0):
            raise Warning(
                f"The passed mask shape does not fit! {len(value)} vs {len(self.x0)}"
            )
        self._isManualMasked = True
        self._manual_mask = value

    # -------------------------- MASKED STUFF --------------------------#
    @property
    def varExpMsk(self):
        return self._varExpMsk.astype(bool)

    @property
    def y(self):
        return self.y0[self.mask]

    @property
    def x(self):
        return self.x0[self.mask]

    @property
    def s(self):
        return self.s0[self.mask]

    @property
    def sigma(self):
        return self.s0[self.mask]

    @property
    def r(self):
        return self.r0[self.mask]

    @property
    def ecc(self):
        return self.r0[self.mask]

    @property
    def phi(self):
        return self.phi0[self.mask]

    @property
    def pol(self):
        return self.phi0[self.mask]

    @property
    def pol_orig(self):
        return self.phi0_orig[self.mask]

    @property
    def beta(self):
        return self.beta0[self.mask]

    @property
    def varexp(self):
        return self.varexp0[self.mask]

    @property
    def r2(self):
        return self.varexp

    @property
    def R2(self):
        return self.varexp

    @property
    def varexp_easy(self):
        return self.varexp_easy0[self.mask]

    @property
    def voxelTC(self):
        return self.voxelTC0[self.mask, :]

    @property
    def modelpred(self):
        return self.modelpred0[self.mask, :]

    @property
    def meanVarExp(self):
        return np.nanmean(self.varexp)

    @property
    def prfanalyzeOpts(self):
        if not hasattr(self, "_prfanalyzeOpts"):
            prfanalyzeOptsF = path.join(
                self._derivatives_path,
                self._prfanalyze_method,
                self._prfanaAn,
                "options.json",
            )
            with open(prfanalyzeOptsF, "r") as fl:
                self._prfanalyzeOpts = json.load(fl)

        return self._prfanalyzeOpts

    @property
    def prfprepare_analysis(self):
        return self.prfanalyzeOpts["prfprepareAnalysis"]

    @property
    def fmriprep_analysis(self):
        return self.prfprepareOpts["fmriprep_analysis"]

    @property
    def prfprepareOpts(self):
        if not hasattr(self, "_prfprepareOpts"):
            prfprepareOptsF = path.join(
                self._derivatives_path,
                "prfprepare",
                f"analysis-{self.prfprepare_analysis}",
                "options.json",
            )
            with open(prfprepareOptsF, "r") as fl:
                self._prfprepareOpts = json.load(fl)

        return self._prfprepareOpts

    @property
    def analysisSpace(self):
        self._analysisSpace = self.prfprepareOpts["analysisSpace"]
        return self._analysisSpace

    @analysisSpace.setter
    def analysisSpace(self, value: str):
        self._analysisSpace = value
