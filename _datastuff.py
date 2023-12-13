from glob import glob
from os import path
import json
import numpy as np
from scipy.io import loadmat
import scipy.stats as sps


def init_variables(self):
    """
    Initializes the variables loaded by the PRF.from_ amethods
    """

    ############################# VISTA #############################
    if self._dataFrom == "mrVista":
        self._x0 = self._model["x0"]

        if self._orientation == "VF":
            self._y0 = -self._model[
                "y0"
            ]  # negative for same flip as in the coverage plots
        elif self._orientation == "MP":
            self._y0 = self._model["y0"]  # same orientation as MP
        else:
            Warning(
                "Choose orientation form [VF, MP], you specified {self._orientation}"
            )

        a = self._model["sigma"]["minor"]
        try:
            b = np.sqrt(self._model["exponent"])
        except:
            b = np.ones(a.shape)
        self._s0 = np.divide(a, b, out=a, where=b != 0)

        self._beta0 = self._model["beta"][
            :, 0
        ]  # the model beta should be the first index, rest are trends

        with np.errstate(divide="ignore", invalid="ignore"):
            self._varexp0 = (1.0 - self._model["rss"] / self._model["rawrss"]).squeeze()

        self._maxEcc = self._params["analysis"]["fieldSize"]

    ############################# DOCKER #############################
    elif self._dataFrom == "docker":
        if "Centerx0" in self._estimates[0][0].keys():
            self._x0 = np.array([e["Centerx0"] for ee in self._estimates for e in ee])

            if self._orientation == "VF":
                self._y0 = -np.array(
                    [e["Centery0"] for ee in self._estimates for e in ee]
                )  # negative for same flip as in the coverage plots
            elif self._orientation == "MP":
                self._y0 = np.array(
                    [e["Centery0"] for ee in self._estimates for e in ee]
                )  # same orientation as MP
            else:
                Warning(
                    "Choose orientation form [VF, MP], you specified {self._orientation}"
                )

            self._s0 = np.array([e["sigmaMajor"] for ee in self._estimates for e in ee])

            try:
                self._varexp0 = np.array(
                    [e["R2"] for ee in self._estimates for e in ee]
                )
            except:
                print("no varexp information")

        elif "centerx0" in self._estimates[0][0].keys():
            self._x0 = np.array([e["centerx0"] for ee in self._estimates for e in ee])

            if self._orientation == "VF":
                self._y0 = -np.array(
                    [e["centery0"] for ee in self._estimates for e in ee]
                )  # negative for same flip as in the coverage plots
            elif self._orientation == "MP":
                self._y0 = np.array(
                    [e["centery0"] for ee in self._estimates for e in ee]
                )  # same orientation as MP
            else:
                Warning(
                    "Choose orientation form [VF, MP], you specified {self._orientation}"
                )

            self._s0 = np.array([e["sigmamajor"] for ee in self._estimates for e in ee])

            try:
                self._varexp0 = np.array(
                    [e["r2"] for ee in self._estimates for e in ee]
                )
            except:
                print("no varexp information")

        if hasattr(self, "_mat"):
            # a = np.hstack([m['sigma'][0][0]['major'][0][0][0] for m in self._model])
            # b = np.hstack([m['exponent'][0][0][0] for m in self._model])

            # self._s0      = np.divide(a, b, out=a, where=b != 0)

            self._beta0 = np.hstack(
                [np.maximum(m["beta"][0][0][0][:, 0, 0], 0) for m in self._model]
            )  # the model beta should be the first index, rest are trends

            with np.errstate(divide="ignore", invalid="ignore"):
                self._rss0 = np.hstack([m["rss"][0][0][0] for m in self._model])
                self._rawrss0 = np.hstack([m["rawrss"][0][0][0] for m in self._model])
                # self._varexp0 = 1.0 - self._rss0 / self._rawrss0

            self._maxEcc = np.hstack(
                [p["analysis"]["fieldSize"][0][0][0][0] for p in self._params]
            )
            if self._hemis == "":
                if self._maxEcc[0] != self._maxEcc[1]:
                    raise Warning("maxEcc for both hemispheres is different!")
            self._maxEcc = self._maxEcc[0]

        #!!! this should be fixed in oprf!
        if "fprf" in self._prfanalyze_method:
            self._y0 = -self._y0

        if np.any([a in self._prfanalyze_method for a in ["oprf", "fprf"]]):
            if self._orientation == "VF":
                self._x0, self._y0 = -self._y0, -self._x0
            elif self._orientation == "MP":
                self._x0, self._y0 = self._y0, self._x0

    ############################# SAMSRF #############################
    elif self._dataFrom == "samsrf":
        self._fit_keys = [a[0][0] for a in self._model["Values"]]
        fit = self._model["Data"]

        self._maxEcc = self._params["Scaling_Factor"][0][0]

        self._x0 = fit[self._fit_keys.index("x0"), :]
        self._y0 = -fit[self._fit_keys.index("y0"), :]
        self._s0 = fit[self._fit_keys.index("Sigma"), :]

        self._varexp0 = fit[self._fit_keys.index("R^2"), :]
        self._beta0 = fit[self._fit_keys.index("Beta"), :]
        self._baseline0 = fit[self._fit_keys.index("Baseline"), :]

        self._hemis = "both"

    self._isROIMasked = None
    self._isVarExpMasked = None
    self._isBetaMasked = None
    self._isEccMasked = None
    self._isSigMasked = None
    self._doROIMsk = True
    self._doVarExpMsk = True
    self._doBetaMsk = True
    self._doEccMsk = True
    self._doSigMsk = True


# --------------------------ALTERNATIVE  CONSTRUCTORS--------------------------#
def from_mrVista(
    cls,
    study,
    subject,
    session,
    analysis,
    server="ceph",
    forcePath=None,
    orientation="VF",
    fit="fFit-fFit-fFit",
):
    """
    With this constructor you can load results that were analyzed
    with matlab based vistasoft pRF analysis in Vienna.

    Args:
        study (str): Study name
        subject (str): Subject name
        session (str): Session name
        analysis (str): Name of the full analysis as found in the subjects mrVista folder
        server (str, optional): Defines the server name. Defaults to 'ceph'.
        forcePath (str, optional): Set a path for the coords.mat file when not in the correct position. Defaults to None.
        orientation (str, optional): Defines if y-axis should be flipped. Use 'VF' for yes (visual field space) or 'MP' for now (retina, microperimetry space). Defaults to 'VF'.
        fit (str, optional): Changes the loaded mrVista name. Defaults to 'fFit-fFit-fFit'.

    Returns:
        cls: Instance of the PRF class with all the results
    """

    if server == "ceph":
        baseP = "/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data"
    elif server == "sacher":
        baseP = "/sacher/melange/fmri/projects"

    if forcePath:
        mat = loadmat(forcePath[0])
        # read in the corresponding coordinates
        coords = loadmat(forcePath[1])["coords"].astype("uint16")
    elif study == "hcpret":  # spetial treatment for study hcpret
        loadROIpossible = False
        mat = loadmat(
            glob(
                path.join(
                    baseP,
                    study,
                    "hcpret_mrVista",
                    "subjects",
                    subject,
                    analysis,
                    "Gray",
                    "*",
                    f"*{fit}.mat",
                )
            )[0],
            simplify_cells=True,
        )
        # read in the corresponding coordinates
        coords = loadmat(
            path.join(
                baseP,
                study,
                "subjects",
                subject,
                session,
                "mrVista",
                analysis,
                "Gray",
                "coords.mat",
            ),
            simplify_cells=True,
        )["coords"].astype("uint16")
    else:
        loadROIpossible = True
        mat = loadmat(
            glob(
                path.join(
                    baseP,
                    study,
                    "subjects",
                    subject,
                    session,
                    "mrVista",
                    analysis,
                    "Gray",
                    "*",
                    f"*{fit}.mat",
                )
            )[0],
            simplify_cells=True,
        )

        # read in the corresponding coordinates
        coords = loadmat(
            path.join(
                baseP,
                study,
                "subjects",
                subject,
                session,
                "mrVista",
                analysis,
                "Gray",
                "coords.mat",
            ),
            simplify_cells=True,
        )["coords"].astype("uint16")

    return cls(
        "mrVista",
        study,
        subject,
        session,
        mat=mat,
        baseP=baseP,
        analysis=analysis,
        coords=coords,
        orientation=orientation,
    )


# --------------------------ALTERNATIVE  CONSTRUCTORS--------------------------#
def from_docker(
    cls,
    study,
    subject,
    session,
    task,
    run,
    method="vista",
    analysis="01",
    hemi="",
    baseP=None,
    orientation="VF",
):
    """
    With this constructor you can load results that were analyzed
    with dockerized solution published in Lerma-Usabiaga 2020 and available
    at github.com/vistalab/PRFmodel

    Args:
        study (str): Study name (e.g. 'stimsim')
        subject (str): Subject name (e.g. 'p001')
        session (str): Session name (e.g. '001')
        task (str): Task name (e.g. 'prf')
        run (str): Run number (e.g. '01')
        method (str, optional): Different methods from the prfanalyze docker possibler, I think. Defaults to 'vista'.
        analysis (str, optional): Number of the analysis within derivatives/prfanalyze. Defaults to '01'.
        hemi (str, optional): When you want to load only one hemisphere ('L' or 'R'), empty when both. Defaults to ''.
        forcePath (str, optional): If the analysis can be found in another location than standard (/z/fmri/data/)this changes the base path. Defaults to None.
        orientation (str, optional): Defines if y-axis should be flipped. Use 'VF' for yes (visual field space) or 'MP' for now (retina, microperimetry space). Defaults to 'VF'.

    Returns:
        cls: Instance of the PRF class with all the results
    """

    prfanaMe = method if method.startswith("prfanalyze-") else f"prfanalyze-{method}"
    prfanaAn = analysis if analysis.startswith("analysis-") else f"analysis-{analysis}"
    subject = subject if subject.startswith("sub-") else f"sub-{subject}"
    session = session if session.startswith("ses-") else f"ses-{session}"
    task = task if task.startswith("task-") else f"task-{task}"
    run = f"{run}" if str(run).startswith("run-") else f"run-{run}"

    if not baseP:
        baseP = "/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data"

    mat = [] if method == "vista" else None
    est = []
    hs = ["L", "R"] if hemi == "" else [hemi]
    for h in hs:
        estimates = path.join(
            baseP,
            study,
            "derivatives",
            prfanaMe,
            prfanaAn,
            subject,
            session,
            f"{subject}_{session}_{task}_{run}_hemi-{h}_estimates.json",
        )

        with open(estimates, "r") as fl:
            this_est = json.load(fl)

        est.append(this_est)

        if method == "vista":
            resP = path.join(
                baseP,
                study,
                "derivatives",
                prfanaMe,
                prfanaAn,
                subject,
                session,
                f"{subject}_{session}_{task}_{run}_hemi-{h}_results.mat",
            )

            if not path.isfile(resP):
                raise Warning(f"file is not existent: {resP}")

            thisMat = loadmat(resP)["results"]

            mat.append(thisMat)

    return cls(
        "docker",
        study,
        subject,
        session,
        baseP,
        task=task,
        run=run,
        hemis=hemi,
        prfanaMe=prfanaMe,
        prfanaAn=prfanaAn,
        orientation=orientation,
        mat=mat,
        est=est,
        method=method,
    )


# --------------------------ALTERNATIVE  CONSTRUCTORS--------------------------#
def from_samsrf(
    cls, study, subject, session, task, run, analysis="01", baseP=None, orientation="VF"
):
    """
    With this constructor you can load results that were analyzed
    with dockerized solution published in Lerma-Usabiaga 2020 and available
    at github.com/vistalab/PRFmodel

    Args:
        study (str): Study name (e.g. 'stimsim')
        subject (str): Subject name (e.g. 'p001')
        session (str): Session name (e.g. '001')
        task (str): Task name (e.g. 'prf')
        run (str): Run number (e.g. '01')
        method (str, optional): Different methods from the prfanalyze docker possibler, I think. Defaults to 'vista'.
        analysis (str, optional): Number of the analysis within derivatives/prfanalyze. Defaults to '01'.
        hemi (str, optional): When you want to load only one hemisphere ('L' or 'R'), empty when both. Defaults to ''.
        forcePath (str, optional): If the analysis can be found in another location than standard (/z/fmri/data/)this changes the base path. Defaults to None.
        orientation (str, optional): Defines if y-axis should be flipped. Use 'VF' for yes (visual field space) or 'MP' for now (retina, microperimetry space). Defaults to 'VF'.

    Returns:
        cls: Instance of the PRF class with all the results
    """

    prfanaAn = analysis if analysis.startswith("analysis-") else f"analysis-{analysis}"
    subject = subject if subject.startswith("sub-") else f"sub-{subject}"
    session = session if session.startswith("ses-") else f"ses-{session}"
    task = task if task.startswith("task-") else f"task-{task}"
    run = f"{run}" if str(run).startswith("run-") else f"run-{run}"

    if not baseP:
        baseP = "/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data"

    func_p = path.join(
        baseP,
        study,
        "derivatives",
        "samsrf",
        prfanaAn,
        subject,
        session,
        f"bi_{subject}_{session}_{task}_{run}_pRF-results.mat",
    )

    mat = loadmat(func_p)

    return cls(
        "samsrf",
        study,
        subject,
        session,
        task=task,
        run=run,
        baseP=baseP,
        mat=mat,
        prfanaAn=prfanaAn,
        orientation=orientation,
        prfanaMe="samsrf",
    )


# --------------------------SPM HRF--------------------------#
def spm_hrf_compat(
    t,
    peak_delay=6,
    under_delay=16,
    peak_disp=1,
    under_disp=1,
    p_u_ratio=6,
    normalize=True,
):
    """SPM HRF function from sum of two gamma PDFs

    This function is designed to be partially compatible with SPMs `spm_hrf.m`
    function.

    The SPN HRF is a *peak* gamma PDF (with location `peak_delay` and dispersion
    `peak_disp`), minus an *undershoot* gamma PDF (with location `under_delay`
    and dispersion `under_disp`, and divided by the `p_u_ratio`).

    Parameters
    ----------
    t : array-like
        vector of times at which to sample HRF.
    peak_delay : float, optional
        delay of peak.
    under_delay : float, optional
        delay of undershoot.
    peak_disp : float, optional
        width (dispersion) of peak.
    under_disp : float, optional
        width (dispersion) of undershoot.
    p_u_ratio : float, optional
        peak to undershoot ratio.  Undershoot divided by this value before
        subtracting from peak.
    normalize : {True, False}, optional
        If True, divide HRF values by their sum before returning.  SPM does this
        by default.

    Returns
    -------
    hrf : array
        vector length ``len(t)`` of samples from HRF at times `t`.

    Notes
    -----
    See ``spm_hrf.m`` in the SPM distribution.
    """
    if len([v for v in [peak_delay, peak_disp, under_delay, under_disp] if v <= 0]):
        raise ValueError("delays and dispersions must be > 0")
    # gamma.pdf only defined for t > 0
    hrf = np.zeros(t.shape, dtype=np.float64)
    pos_t = t[t > 0]
    peak = sps.gamma.pdf(pos_t, peak_delay / peak_disp, loc=0, scale=peak_disp)
    undershoot = sps.gamma.pdf(pos_t, under_delay / under_disp, loc=0, scale=under_disp)
    hrf[t > 0] = peak - undershoot / p_u_ratio
    if not normalize:
        return hrf
    return hrf / np.sum(hrf)
