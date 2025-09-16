import json
from glob import glob
from os import path

import numpy as np
from scipy.io import loadmat

# class for loading mrVista results


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
    from ._loadadditionalstuff import loadJitter, loadRealign, loadStim
    from ._maskstuff import (
        _calcMask,
        maskBetaThresh,
        maskDartBoard,
        maskEcc,
        maskROI,
        maskSigma,
        maskVarExp,
    )
    from ._plotstuff import (
        _calcCovMap,
        _createmask,
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
                self._voxelTC0 = np.array(
                    [e["testdata"] for ee in self._estimates for e in ee]
                )

            elif self._dataFrom == "samsrf":
                self._voxelTC0 = self._mat["Srf"]["Y"][0][0].T

            np.seterr(invalid="ignore")
            self._voxelTCpsc0 = self._voxelTC0 / self._voxelTC0.mean(1)[:, None] * 100

        return self._voxelTC0.astype(np.float32)

    @property
    def modelpred0(self):
        """
        np.ndarray: Unmasked model predictions.

        Loaded or calculated from source if not already present.
        """
        if not hasattr(self, "_modelpred0"):
            if self._dataFrom == "mrVista":
                print("No modelpred with data_from mrVista")
                return None
            elif self._dataFrom == "docker":
                self._modelpred0 = np.array(
                    [e["modelpred"] for ee in self._estimates for e in ee]
                )

                if np.allclose(
                    self._modelpred0[::1000, :]
                    - self._modelpred0[::1000, :].mean(1)[:, None],
                    self.voxelTC0[::1000, :]
                    - self.voxelTC0[::1000, :].mean(1)[:, None],
                ):
                    self._modelpred0 = self.loadStim(buildTC=True)
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
        """np.ndarray: Maximum eccentricity for the stimulus grid."""
        return np.float32(self._maxEcc)

    @property
    def model(self):
        """Model object(s) loaded from source."""
        return self._model

    @property
    def params(self):
        """Parameter object(s) loaded from source."""
        return self._params

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
    def beta(self):
        return self.beta0[self.mask]

    @property
    def varexp(self):
        return self.varexp0[self.mask]

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
