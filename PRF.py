import numpy as np
from scipy.io import loadmat
from os import path
import json

# class for loading mrVista results


class PRF:
    """
    This class loads data from PRF mapping analysis done with a matlab
    based mrVista analysis or the prfanalyze-vistasoft docker.
    Initialize with PRF.from_mrVista or PRF.from_docker method!
    """

    # load in all the other files with functions
    from ._datastuff import (
        init_variables,
        from_docker,
        from_mrVista,
        from_samsrf,
        spm_hrf_compat,
    )
    from ._maskstuff import (
        maskROI,
        maskVarExp,
        maskEcc,
        maskSigma,
        maskBetaThresh,
        _calcMask,
        maskDartBoard,
    )
    from ._loadadditionalstuff import loadStim, loadJitter, loadRealign
    from ._calculatestuff import (
        calc_kde_diff,
        calc_prf_profiles,
        central_scot_border,
        plot_kdeDiff2d,
    )
    from ._plotstuff import (
        plot_covMap,
        _calcCovMap,
        plot_toSurface,
        _createmask,
        _get_surfaceSavePath,
        _make_gif,
    )

    from_docker = classmethod(from_docker)
    from_mrVista = classmethod(from_mrVista)
    from_samsrf = classmethod(from_samsrf)

    def __init__(
        self,
        dataFrom,
        study,
        subject,
        session,
        baseP,
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
        if self._dataFrom == "docker":
            self._hemis = hemis

        self._orientation = orientation.upper()

        if self._dataFrom == "mrVista":
            self._model = self._mat["model"]
            self._params = self._mat["params"]
        elif self._dataFrom == "docker":
            if hasattr(self, "_mat"):
                self._model = [m["model"][0][0][0][0] for m in self._mat]
                self._params = [m["params"][0][0][0][0] for m in self._mat]
        elif self._dataFrom == "samsrf":
            self._model = self._mat["Srf"][0][0]
            self._params = self._mat["Model"][0][0]

        # initialize
        self.init_variables()

    @property
    def subject(self):
        return self._subject

    @property
    def session(self):
        return self._session

    @property
    def task(self):
        return self._task

    @property
    def run(self):
        return self._run

    @property
    def y0(self):
        return self._y0

    @property
    def x0(self):
        return self._x0

    @property
    def s0(self):
        return self._s0

    @property
    def r0(self):
        return np.sqrt(np.square(self.x0) + np.square(self.y0))

    @property
    def ecc0(self):
        return np.sqrt(np.square(self.x0) + np.square(self.y0))

    @property
    def phi0(self):
        p = np.arctan2(self.y0, self.x0)
        p[p < 0] += 2 * np.pi
        return p

    @property
    def phi0_orig(self):
        p = np.arctan2(self.y0, self.x0)
        return p

    @property
    def pol0(self):
        return self.phi0

    @property
    def pol0_orig(self):
        return self.phi0_orig

    @property
    def beta0(self):
        return self._beta0

    @property
    def varexp0(self):
        return self._varexp0

    @property
    def varexp_easy0(self):
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

        return self._varexp_easy

    @property
    def voxelTC0(self):
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

            np.seterr(invalid="ignore")
            self._voxelTCpsc0 = self._voxelTC0 / self._voxelTC0.mean(1)[:, None] * 100

        return self._voxelTC0

    @property
    def modelpred0(self):
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

        return self._modelpred0

    @property
    def maxEcc(self):
        return self._maxEcc

    @property
    def model(self):
        return self._model

    @property
    def params(self):
        return self._params

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

    # -------------------------- MASKED STUFF --------------------------#
    @property
    def mask(self):
        return self._calcMask()

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
        # if not self.isROIMasked: self.maskROI()
        return np.nanmean(self.varexp)

    @property
    def prfanalyzeOpts(self):
        if not hasattr(self, "_prfanalyzeOpts"):
            prfanalyzeOptsF = path.join(
                self._baseP,
                self._study,
                "derivatives",
                self._prfanalyze_method,
                self._prfanaAn,
                "options.json",
            )
            with open(prfanalyzeOptsF, "r") as fl:
                self._prfanalyzeOpts = json.load(fl)

        return self._prfanalyzeOpts

    @property
    def prfprepareOpts(self):
        if not hasattr(self, "_prfprepareOpts"):
            prfprepareOptsF = path.join(
                self._baseP,
                self._study,
                "derivatives",
                "prfprepare",
                f'analysis-{self.prfanalyzeOpts["prfprepareAnalysis"]}',
                "options.json",
            )
            with open(prfprepareOptsF, "r") as fl:
                self._prfprepareOpts = json.load(fl)

        return self._prfprepareOpts
