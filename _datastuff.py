import json
from glob import glob
from os import path

import numpy as np
import scipy.stats as sps
from scipy.io import loadmat

try:
    import h5py
except ImportError:
    print("hdf5 not available, do not use from_hdf5!")


def _h5_text(value):
    """Decode an h5py string scalar.

    str() on the bytes h5py hands back would keep the b'...' wrapper, which
    then travels into file paths and comparisons.
    """
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.dtype.kind in "SO":
        return _h5_text(value.item())
    return str(value)


def init_variables(self):
    """
    Initialize variables loaded by the PRF.from_ methods, depending on data source.

    Raises:
        ValueError: If the data source or orientation is invalid.
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
            raise ValueError(
                f"Choose orientation from [VF, MP], you specified {self._orientation}"
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
        if self._prfanalyze_method == "prfanalyze-deeprf":
            if self._orientation == "MP":
                if "Centerx0" in self._estimates[0][0].keys():
                    self._x0 = np.array(
                        [e["Centery0"] for ee in self._estimates for e in ee]
                    )
                    self._y0 = np.array(
                        [e["Centerx0"] for ee in self._estimates for e in ee]
                    )
                elif "centerx0" in self._estimates[0][0].keys():
                    self._x0 = np.array(
                        [e["centery0"] for ee in self._estimates for e in ee]
                    )
                    self._y0 = np.array(
                        [e["centerx0"] for ee in self._estimates for e in ee]
                    )

                self._s0 = np.abs(
                    np.array([e["sigmaMajor"] for ee in self._estimates for e in ee])
                )
                try:
                    self._varexp0 = np.array(
                        [e["R2"] for ee in self._estimates for e in ee]
                    )
                except:
                    print("no varexp information")
            else:
                Warning(
                    "DeepRF results should be loaded with orientation MP, not {self._orientation}"
                )
        # gem HDF5: estimates are dicts of numpy arrays (not list-of-dicts)
        elif (
            isinstance(self._estimates[0], dict)
            and "Centerx0" in self._estimates[0]
            and isinstance(self._estimates[0]["Centerx0"], np.ndarray)
        ):
            self._x0 = np.hstack([e["Centerx0"] for e in self._estimates])
            self._s0 = np.abs(np.hstack([e["sigmaMajor"] for e in self._estimates]))
            self._varexp0 = np.hstack([e["R2"] for e in self._estimates])
            if self._orientation == "VF":
                self._y0 = -np.hstack([e["Centery0"] for e in self._estimates])
            else:
                self._y0 = np.hstack([e["Centery0"] for e in self._estimates])
            if "_maxEcc" in self._estimates[0]:
                self._maxEcc = np.float32(self._estimates[0]["_maxEcc"])
            if "_model" in self._estimates[0]:
                self._gem_model = self._estimates[0]["_model"]
            if "_hrf" in self._estimates[0]:
                self._gem_hrf = self._estimates[0]["_hrf"]
            if "_run_type" in self._estimates[0]:
                self._gem_run_type = self._estimates[0]["_run_type"]
            if "_duration_sec" in self._estimates[0]:
                self._gem_duration_sec = self._estimates[0]["_duration_sec"]
            if "_input_files" in self._estimates[0]:
                self._gem_input_files = self._estimates[0]["_input_files"]
            if "_stimulus_filepath" in self._estimates[0]:
                self._gem_stimulus_filepath = self._estimates[0]["_stimulus_filepath"]
            if "_searchSpace" in self._estimates[0]:
                self._gem_search_space = self._estimates[0]["_searchSpace"]
            if "_stimulus" in self._estimates[0]:
                self._gem_stimulus = self._estimates[0]["_stimulus"]

        else:
            if "Centerx0" in self._estimates[0][0].keys():
                self._x0 = np.array(
                    [e["Centerx0"] for ee in self._estimates for e in ee]
                )

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

                self._s0 = np.abs(
                    np.array([e["sigmaMajor"] for ee in self._estimates for e in ee])
                )

                try:
                    self._varexp0 = np.array(
                        [e["R2"] for ee in self._estimates for e in ee]
                    )
                except:
                    print("no varexp information")

            elif "centerx0" in self._estimates[0][0].keys():
                self._x0 = np.array(
                    [e["centerx0"] for ee in self._estimates for e in ee]
                )

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

                self._s0 = np.abs(
                    np.array([e["sigmamajor"] for ee in self._estimates for e in ee])
                )

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
                    self._rawrss0 = np.hstack(
                        [m["rawrss"][0][0][0] for m in self._model]
                    )
                    # self._varexp0 = 1.0 - self._rss0 / self._rawrss0

                self._maxEcc = np.hstack(
                    [p["analysis"]["fieldSize"][0][0][0][0] for p in self._params]
                )
                if self._hemis == "":
                    if self._maxEcc[0] != self._maxEcc[1]:
                        raise Warning("maxEcc for both hemispheres is different!")
                self._maxEcc = self._maxEcc[0].astype(np.float32)

            #!!! this should be fixed in gem!
            if "fprf" in self._prfanalyze_method:
                self._y0 = -self._y0

    ############################# SAMSRF #############################
    elif self._dataFrom == "samsrf":
        self._fit_keys = [a[0][0] for a in self._model[0]["Values"]]
        fit = np.hstack([m["Data"] for m in self._model])

        self._maxEcc = self._params[0]["Scaling_Factor"][0][0]

        self._x0 = fit[self._fit_keys.index("x0"), :]
        self._y0 = -fit[self._fit_keys.index("y0"), :]
        self._s0 = fit[self._fit_keys.index("Sigma"), :]

        self._varexp0 = fit[self._fit_keys.index("R^2"), :]
        self._beta0 = fit[self._fit_keys.index("Beta"), :]
        self._baseline0 = fit[self._fit_keys.index("Baseline"), :]

    ############################# HDF5 #############################
    #!!! this should be fixed once we decide on a standard for the hdf5 files
    elif self._dataFrom == "hdf5":
        # Check for 'thetas' dataset, otherwise use x0, y0, s0, r2
        if all("thetas" in m for m in self._estimates):
            self._x0 = np.hstack([m["thetas"][:, 0] for m in self._estimates])
            self._y0 = np.hstack([m["thetas"][:, 1] for m in self._estimates])
            self._s0 = np.hstack([m["thetas"][:, 2] for m in self._estimates])
        else:
            self._x0 = np.hstack([m["x0"][:] for m in self._estimates])
            self._y0 = np.hstack([m["y0"][:] for m in self._estimates])
            self._s0 = np.hstack([m["s0"][:] for m in self._estimates])
            if all("r2" in m for m in self._estimates):
                self._varexp0 = np.hstack([m["r2"][:] for m in self._estimates])
            else:
                self._varexp0 = None

    else:
        raise ValueError(f"Unknown data source '{self._dataFrom}' in init_variables.")

    self._isROIMasked = None
    self._isVarExpMasked = None
    self._isBetaMasked = None
    self._isEccMasked = None
    self._isSigMasked = None
    self._isManualMasked = None
    self._doROIMsk = True
    self._doVarExpMsk = True
    self._doBetaMsk = True
    self._doEccMsk = True
    self._doSigMsk = True
    self._doManualMsk = True


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
    Load results from a MATLAB-based mrVista pRF analysis.

    Args:
        study (str): Study name.
        subject (str): Subject name.
        session (str): Session name.
        analysis (str): Name of the analysis in the subject's mrVista folder.
        server (str, optional): Server name. Defaults to 'ceph'.
        forcePath (str, optional): Path for coords.mat file if not in standard location. Defaults to None.
        orientation (str, optional): 'VF' for visual field space, 'MP' for microperimetry space. Defaults to 'VF'.
        fit (str, optional): mrVista fit name. Defaults to 'fFit-fFit-fFit'.

    Returns:
        cls: Instance of the PRF class with loaded results.
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
        try:
            mat_file_path = path.join(
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
            mat = loadmat(
                glob(mat_file_path)[0],
                simplify_cells=True,
            )
        except IndexError:
            raise FileNotFoundError(
                f"Could not find .mat file: {mat_file_path}.\nPlease check the parameters and try again."
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
        try:
            mat_file_path = path.join(
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
            mat = loadmat(
                glob(mat_file_path)[0],
                simplify_cells=True,
            )
        except IndexError:
            raise FileNotFoundError(
                f"Could not find .mat file: {mat_file_path}.\nPlease check the parameters and try again."
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
    load_mat_file=True,
):
    """
    Load results from dockerized prfanalyze-vistasoft analysis.

    Args:
        study (str): Study name.
        subject (str): Subject name.
        session (str): Session name.
        task (str): Task name.
        run (str): Run number.
        method (str, optional): prfanalyze method. Defaults to 'vista'.
        analysis (str, optional): Analysis number. Defaults to '01'.
        hemi (str, optional): Hemisphere ('L', 'R', or ''). Defaults to ''.
        baseP (str, optional): Base path for data. Defaults to None.
        orientation (str, optional): 'VF' or 'MP'. Defaults to 'VF'.
        load_mat_file (bool, optional): Whether to load .mat file. Defaults to True.

    Returns:
        cls: Instance of the PRF class with loaded results.

    Raises:
        FileNotFoundError: If required files are missing.
    """

    prfanaMe = method if method.startswith("prfanalyze-") else f"prfanalyze-{method}"
    prfanaAn = analysis if analysis.startswith("analysis-") else f"analysis-{analysis}"
    subject = subject if subject.startswith("sub-") else f"sub-{subject}"
    session = session if session.startswith("ses-") else f"ses-{session}"
    task = task if task.startswith("task-") else f"task-{task}"
    run = f"{run}" if str(run).startswith("run-") else f"run-{run}"

    if not baseP:
        baseP = "/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data"

    derivatives_path = path.join(
        baseP,
        study,
        "derivatives",
    )
    if not path.isdir(derivatives_path):
        derivatives_path = path.join(
            baseP,
            study,
            "BIDS",
            "derivatives",
        )

        if not path.isdir(derivatives_path):
            raise FileNotFoundError(
                f"could not find derivatives folder at {derivatives_path} or {path.join(baseP, study, 'derivatives')}"
            )

    mat = [] if method == "vista" and load_mat_file == True else None
    est = []
    hs = ["L", "R"] if hemi == "" else [hemi]
    for h in hs:
        stem = path.join(
            derivatives_path,
            prfanaMe,
            prfanaAn,
            subject,
            session,
            f"{subject}_{session}_{task}_{run}_hemi-{h}_estimates",
        )
        h5_path = stem + ".h5"
        json_path = stem + ".json"

        if path.isfile(h5_path):
            with h5py.File(h5_path, "r") as f:
                this_est = {k: f["parameters"][k][()] for k in f["parameters"].keys()}
                # maxEcc is the stimulus radius, not the fitting grid: gem
                # searches beyond the aperture edge so pRFs can sit partly
                # outside the stimulus, and using that larger radius would
                # stretch every eccentricity scale and coverage map.
                if "metadata/stimulus/visual_field" in f:
                    this_est["_maxEcc"] = float(f["metadata/stimulus/visual_field"][()])
                elif "metadata/search_space/visual_field_radius" in f:
                    this_est["_maxEcc"] = float(
                        f["metadata/search_space/visual_field_radius"][()]
                    )
                    print(
                        f"Warning: {path.basename(h5_path)} has no "
                        "metadata/stimulus/visual_field; falling back to the "
                        "search-space radius as maxEcc, which is usually larger "
                        "than the stimulus."
                    )
                # the search radius is where boundary artefacts accumulate,
                # so keep the grid around for masking them out
                if "metadata/search_space" in f:
                    this_est["_searchSpace"] = {
                        k: f[f"metadata/search_space/{k}"][()]
                        for k in f["metadata/search_space"].keys()
                    }
                if "metadata/stimulus" in f:
                    this_est["_stimulus"] = {
                        k: f[f"metadata/stimulus/{k}"][()]
                        for k in f["metadata/stimulus"].keys()
                    }
                if "metadata/analysis/model" in f:
                    this_est["_model"] = _h5_text(f["metadata/analysis/model"][()])
                if "metadata/hrf" in f:
                    this_est["_hrf"] = {
                        k: f[f"metadata/hrf/{k}"][()] for k in f["metadata/hrf"].keys()
                    }
                if "metadata/run_info/run_type" in f:
                    this_est["_run_type"] = _h5_text(
                        f["metadata/run_info/run_type"][()]
                    )
                if "metadata/run_info/analysis_duration_sec" in f:
                    this_est["_duration_sec"] = float(
                        f["metadata/run_info/analysis_duration_sec"][()]
                    )
                if "metadata/input_files/measured_data" in f:
                    this_est["_input_files"] = [
                        _h5_text(v) for v in f["metadata/input_files/measured_data"][()]
                    ]
                if "metadata/input_files/stimulus_filepath" in f:
                    this_est["_stimulus_filepath"] = _h5_text(
                        f["metadata/input_files/stimulus_filepath"][()]
                    )
        elif path.isfile(json_path):
            with open(json_path, "r") as fl:
                this_est = json.load(fl)
        else:
            raise FileNotFoundError(
                f"Could not find estimates file (.h5 or .json): "
                f"{path.basename(h5_path)}! Please check the params and try again."
            )

        est.append(this_est)

        if method == "vista" and load_mat_file == True:
            try:
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
                    raise print(f"could not find _results.mat file: {resP}")

                thisMat = loadmat(resP)["results"]

                mat.append(thisMat)
            except:
                mat = None

    return cls(
        "docker",
        study,
        subject,
        session,
        baseP,
        derivatives_path=derivatives_path,
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
def from_file(
    cls,
    study,
    force_path,
    orientation="VF",
):
    """
    Load results from a given JSON file path.

    Args:
        study (str): Study name.
        force_path (str or list): Path(s) to JSON file(s).
        orientation (str, optional): 'VF' or 'MP'. Defaults to 'VF'.

    Returns:
        cls: Instance of the PRF class with loaded results.

    Raises:
        ValueError: If file naming is inconsistent.
    """

    print(
        f"[WARNING] As you are providing a path only, some functions may not work properly!"
    )

    if not isinstance(force_path, list):
        force_path = [force_path]

    if len(force_path) > 2:
        print(
            "[WARNING] You are providing more than 2 paths, only the first 2 will be used!"
        )
        force_path = force_path[:2]

    est = []
    for p in force_path:
        with open(p, "r") as fl:
            this_est = json.load(fl)

        est.append(this_est)

    def check_all_elements_equal(lst):
        if len(set(lst)) > 1:
            raise ValueError(f"All elements in the list {lst} are not the same.")
        return lst[0]

    method = check_all_elements_equal(
        [i.split("prfanalyze-")[-1].split("analysis-")[0][:-1] for i in force_path]
    )
    analysis = check_all_elements_equal(
        [i.split("analysis-")[-1].split("sub-")[0][:-1] for i in force_path]
    )
    subject = check_all_elements_equal(
        [path.basename(i).split("sub-")[-1].split("_")[0] for i in force_path]
    )
    session = check_all_elements_equal(
        [path.basename(i).split("ses-")[-1].split("_")[0] for i in force_path]
    )
    task = check_all_elements_equal(
        [path.basename(i).split("task-")[-1].split("_")[0] for i in force_path]
    )
    run = check_all_elements_equal(
        [path.basename(i).split("run-")[-1].split("_")[0] for i in force_path]
    )
    if "hemi" in path.basename(force_path[0]):
        hemi = [path.basename(i).split("hemi-")[-1].split("_")[0] for i in force_path]
    else:
        hemi = "BOTH"

    prfanaMe = method if method.startswith("prfanalyze-") else f"prfanalyze-{method}"
    prfanaAn = analysis if analysis.startswith("analysis-") else f"analysis-{analysis}"
    subject = subject if subject.startswith("sub-") else f"sub-{subject}"
    session = session if session.startswith("ses-") else f"ses-{session}"
    task = task if task.startswith("task-") else f"task-{task}"
    run = f"{run}" if str(run).startswith("run-") else f"run-{run}"

    # sort that left hemi is first
    hemi, est = zip(*sorted(zip(hemi, est)))
    hemi = list(hemi)
    est = list(est)

    # get the base path
    baseP = check_all_elements_equal([i.split(study)[0][:-1] for i in force_path])

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
        est=est,
        method=method,
    )


# --------------------------ALTERNATIVE  CONSTRUCTORS--------------------------#
def from_hdf5(
    cls,
    study,
    subject,
    session,
    task,
    run,
    method="",
    analysis="01",
    baseP=None,
    orientation="VF",
    hemi="",
):
    """
    Load results from an HDF5 file.

    Args:
        study (str): Study name.
        subject (str): Subject name.
        session (str): Session name.
        task (str): Task name.
        run (str): Run number.
        method (str, optional): Analysis method. Defaults to ''.
        analysis (str, optional): Analysis number. Defaults to '01'.
        baseP (str, optional): Base path for data. Defaults to None.
        orientation (str, optional): 'VF' or 'MP'. Defaults to 'VF'.
        hemi (str, optional): Hemisphere ('L', 'R', or ''). Defaults to ''.
        derivatives_folder (str, optional): Name of derivatives folder. Defaults to 'michi'.

    Returns:
        cls: Instance of the PRF class with loaded results.

    Raises:
        FileNotFoundError: If required files are missing.
    """

    prfanaMe = (
        ""
        if method == ""
        else method if method.startswith("prfanalyze-") else f"prfanalyze-{method}"
    )
    prfanaAn = analysis if analysis.startswith("analysis-") else f"analysis-{analysis}"
    subject = subject if subject.startswith("sub-") else f"sub-{subject}"
    session = session if session.startswith("ses-") else f"ses-{session}"
    task = task if task.startswith("task-") else f"task-{task}"
    run = f"{run}" if str(run).startswith("run-") else f"run-{run}"

    if not baseP:
        baseP = "/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data"

    derivatives_path = path.join(
        baseP,
        study,
        "derivatives",
    )
    if not path.isdir(derivatives_path):
        derivatives_path = path.join(
            baseP,
            study,
            "BIDS",
            "derivatives",
        )

        if not path.isdir(derivatives_path):
            raise FileNotFoundError(
                f"could not find derivatives folder at {derivatives_path} or {path.join(baseP, study, 'derivatives')}"
            )

    est = []
    hs = ["L", "R"] if hemi == "" else [hemi]
    for h in hs:
        print("WARNING: i hardcoded a strange name here!!")
        estimates = path.join(
            derivatives_path,
            prfanaMe,
            prfanaAn,
            subject,
            session,
            f"{subject}_{session}_{task}_{run}_hemi-{h}_desc-prfestimates.h5",
        )

        this_est = h5py.File(estimates, "r")

        est.append(this_est)

    return cls(
        "hdf5",
        study,
        subject,
        session,
        baseP,
        derivatives_path=derivatives_path,
        task=task,
        run=run,
        hemis=hemi,
        prfanaMe=prfanaMe,
        prfanaAn=prfanaAn,
        orientation=orientation,
        est=est,
    )


# --------------------------ALTERNATIVE  CONSTRUCTORS--------------------------#
def from_samsrf(
    cls,
    study,
    subject,
    session,
    task,
    run,
    analysis="01",
    baseP=None,
    orientation="VF",
    hemis=None,
):
    """
    Load results from SamSrf analysis.

    Args:
        study (str): Study name.
        subject (str): Subject name.
        session (str): Session name.
        task (str): Task name.
        run (str): Run number.
        analysis (str, optional): Analysis number. Defaults to '01'.
        baseP (str, optional): Base path for data. Defaults to None.
        orientation (str, optional): 'VF' or 'MP'. Defaults to 'VF'.
        hemis (str or None, optional): Hemisphere(s) to load. Defaults to None.

    Returns:
        cls: Instance of the PRF class with loaded results.

    Raises:
        FileNotFoundError: If required files are missing.
    """

    prfanaAn = analysis if analysis.startswith("analysis-") else f"analysis-{analysis}"
    subject = subject if subject.startswith("sub-") else f"sub-{subject}"
    session = session if session.startswith("ses-") else f"ses-{session}"
    task = task if task.startswith("task-") else f"task-{task}"
    run = f"{run}" if str(run).startswith("run-") else f"run-{run}"

    if not baseP:
        baseP = "/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data"

    derivatives_path = path.join(
        baseP,
        study,
        "derivatives",
    )
    if not path.isdir(derivatives_path):
        derivatives_path = path.join(
            baseP,
            study,
            "BIDS",
            "derivatives",
        )

        if not path.isdir(derivatives_path):
            raise FileNotFoundError(
                f"could not find derivatives folder at {derivatives_path} or {path.join(baseP, study, 'derivatives')}"
            )

    if hemis is None:
        func_p = path.join(
            derivatives_path,
            "samsrf",
            prfanaAn,
            subject,
            session,
            f"bi_{subject}_{session}_{task}_{run}_pRF-results.mat",
        )

        mat = [loadmat(func_p)]
        hemis = "both"

    else:
        mat = []

        if hemis == "" or hemis == "both":
            hs = ["L", "R"]
            hemis = ""
        else:
            hs = [hemis]
            hemis = [hemis]

        for h in hs:
            estimates = path.join(
                baseP,
                study,
                "derivatives",
                "samsrf",
                prfanaAn,
                subject,
                session,
                f"bi_{subject}_{session}_{task}_{run}_hemi-{h}_pRF-results.mat",
            )

            thisMat = loadmat(estimates)

            mat.append(thisMat)

    return cls(
        "samsrf",
        study,
        subject,
        session,
        task=task,
        run=run,
        hemis=hemis,
        baseP=baseP,
        derivatives_path=derivatives_path,
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
