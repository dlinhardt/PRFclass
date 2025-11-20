import re
import tempfile
from fnmatch import translate as fnmatch_translate
from glob import glob
from os import makedirs, path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from reportlab.lib.pagesizes import A4, landscape, letter
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from tqdm import tqdm

from . import PRF

# Try to import placeImage function
try:
    from placeImages import placeImage

    HAS_PLACE_IMAGE = True
except ImportError:
    HAS_PLACE_IMAGE = False


class PRFgroup:
    """
    Group container for multiple PRF objects.

    This class manages a collection of PRF objects, typically for a study or cohort.
    Provides methods for batch masking, querying, and property aggregation.
    """

    def __init__(
        self,
        data_from,
        study,
        DF,
        baseP,
        orientation,
        derivatives_path=None,
        prfanalyze=None,
        hemi=None,
        fit=None,
        method=None,
    ):
        """
        Initialize a PRFgroup object.

        Args:
            data_from (str): Data source type.
            study (str): Study name.
            DF (pd.DataFrame): DataFrame with subject/session/task/run and PRF objects.
            baseP (str): Base path for data.
            orientation (str): Orientation ('VF' or 'MP').
            derivatives_path (str, optional): Path to derivatives folder.
            prfanalyze (str, optional): prfanalyze analysis.
            hemi (str, optional): Hemisphere.
            fit (str, optional): Fit name.
            method (str, optional): Analysis method.
        """
        # set the variables
        self.data_from = data_from
        self.study = study
        self.data = DF
        self.baseP = baseP
        self.orientation = orientation

        if method:
            self._method = method
        if prfanalyze:
            self._prfanalyze = prfanalyze
        if hemi:
            self._hemi = hemi
        if fit:
            self._fit = fit
        if derivatives_path:
            self._derivatives_path = derivatives_path

        self._doROIMsk = True
        self._doVarExpMsk = True
        self._doBetaMsk = True
        self._doEccMsk = True
        self._doSigMsk = True

    def __getitem__(self, item):
        """
        Get a row from the group's DataFrame by index.

        Args:
            item (int): Row index.

        Returns:
            pd.Series: Row from the DataFrame.
        """
        return self.data.loc[item]

    def __len__(self):
        """
        Get the number of entries in the group.

        Returns:
            int: Number of rows in the DataFrame.
        """
        return len(self.data)

    # --------------------------------- UTILITIES --------------------------------#
    @staticmethod
    def _pattern(pattern: str, use_glob: bool) -> str:
        """
        Return a regex pattern, optionally translating from shell-style glob syntax.

        Args:
            pattern (str): Input pattern provided by the user.
            use_glob (bool): Whether the pattern should be interpreted as a glob.

        Returns:
            str: Regex pattern ready for pandas.Series.str.match.
        """

        if use_glob:
            pattern = pattern.replace("{", "[").replace("}", "]")
            return fnmatch_translate(pattern)
        return pattern

    # --------------------------ALTERNATIVE  CONSTRUCTORS--------------------------#
    @classmethod
    def from_docker(
        cls,
        study,
        subjects=".*",
        sessions=".*",
        tasks=".*",
        runs=".*",
        use_glob: bool = False,
        method="vista",
        prfanalyze="01",
        hemi="",
        baseP=None,
        orientation="VF",
        load_mat_file=True,
    ):
        """
        Alternative constructor to create PRFgroup from Docker data.

        Args:
            cls: The class being instantiated.
            study (str): Study name.
            subjects (str, optional): Regex for subject selection.
            sessions (str, optional): Regex for session selection.
            tasks (str, optional): Regex for task selection.
            runs (str, optional): Regex for run selection.
            use_glob (bool, optional): Interpret the selection strings as shell-style
                wildcards and translate them to regex automatically.
            method (str, optional): Analysis method.
            prfanalyze (str, optional): prfanalyze analysis.
            hemi (str, optional): Hemisphere.
            baseP (str, optional): Base path for data.
            orientation (str, optional): Orientation ('VF' or 'MP').
            load_mat_file (bool, optional): Flag to load .mat files.

        Returns:
            PRFgroup: An instance of PRFgroup populated with Docker data.
        """
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
                "BIDS" "derivatives",
            )

            if not path.isdir(derivatives_path):
                raise FileNotFoundError(
                    f"could not find derivatives folder at {derivatives_path} or {path.join(baseP, study, 'derivatives')}"
                )

        anas = glob(
            path.join(
                derivatives_path,
                f"prfanalyze-{method}",
                f"analysis-{prfanalyze}",
                "*",
                "*",
                "*_estimates.json",
            )
        )

        su = [path.basename(a).split("_")[0].split("-")[-1] for a in anas]
        se = [path.basename(a).split("_")[1].split("-")[-1] for a in anas]
        ta = [path.basename(a).split("_")[2].split("-")[-1] for a in anas]
        ru = [path.basename(a).split("_")[3].split("-")[-1] for a in anas]

        anasDF = pd.DataFrame(
            np.array([su, se, ta, ru]).T, columns=["subject", "session", "task", "run"]
        )

        if len(anasDF) == 0:
            print("no analyses were found!")
            return

        subj_pat = cls._pattern(subjects, use_glob)
        sess_pat = cls._pattern(sessions, use_glob)
        task_pat = cls._pattern(tasks, use_glob)
        run_pat = cls._pattern(runs, use_glob)

        anasDF = anasDF[
            (anasDF["subject"].str.match(subj_pat))
            & (anasDF["session"].str.match(sess_pat))
            & (anasDF["task"].str.match(task_pat))
            & (anasDF["run"].str.match(run_pat))
        ]

        anasDF = anasDF.drop_duplicates().reset_index(drop=True)
        anasDF = anasDF.sort_values(
            ["subject", "session", "task", "run"], ignore_index=True
        )

        if len(anasDF) >= 10:
            pbar = tqdm(total=len(anasDF))
        for I, a in anasDF.iterrows():
            try:
                anasDF.loc[I, "prf"] = PRF.from_docker(
                    study=study,
                    subject=a["subject"],
                    session=a["session"],
                    task=a["task"],
                    run=a["run"],
                    method=method,
                    analysis=prfanalyze,
                    hemi=hemi,
                    baseP=baseP,
                    orientation=orientation,
                    load_mat_file=load_mat_file,
                )
            except Exception as e:
                print(
                    f'could not load sub-{a["subject"]}_ses-{a["session"]}_task-{a["task"]}_run-{a["run"]}!'
                )
                print(e)
                anasDF.drop(I).reset_index(drop=True)
            if len(anasDF) >= 10:
                pbar.update(1)
        if len(anasDF) >= 10:
            pbar.close()

        # check if something was actually found
        if len(anasDF) == 0:
            raise Warning("No analyses were found!")

        return cls(
            data_from="docker",
            study=study,
            DF=anasDF,
            baseP=baseP,
            derivatives_path=derivatives_path,
            orientation=orientation,
            hemi=hemi,
            prfanalyze=prfanalyze,
            method=method,
        )

    # --------------------------ALTERNATIVE  CONSTRUCTORS--------------------------#
    @classmethod
    def from_hdf5(
        cls,
        study,
        subjects=".*",
        sessions=".*",
        tasks=".*",
        runs=".*",
        use_glob: bool = False,
        method="vista",
        prfanalyze="01",
        hemi="",
        baseP=None,
        orientation="VF",
    ):
        """
        Alternative constructor to create PRFgroup from Docker data.

        Args:
            cls: The class being instantiated.
            study (str): Study name.
            subjects (str, optional): Regex for subject selection.
            sessions (str, optional): Regex for session selection.
            tasks (str, optional): Regex for task selection.
            runs (str, optional): Regex for run selection.
            use_glob (bool, optional): Interpret the selection strings as shell-style
                wildcards and translate them to regex automatically.
            method (str, optional): Analysis method.
            prfanalyze (str, optional): prfanalyze analysis.
            hemi (str, optional): Hemisphere.
            baseP (str, optional): Base path for data.
            orientation (str, optional): Orientation ('VF' or 'MP').
            load_mat_file (bool, optional): Flag to load .mat files.

        Returns:
            PRFgroup: An instance of PRFgroup populated with Docker data.
        """
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
                "BIDS" "derivatives",
            )

            if not path.isdir(derivatives_path):
                raise FileNotFoundError(
                    f"could not find derivatives folder at {derivatives_path} or {path.join(baseP, study, 'derivatives')}"
                )

        print("WARNING: i hardcoded a path here!!")
        anas = glob(
            path.join(
                derivatives_path,
                f"prfanalyze-{method}",
                f"analysis-{prfanalyze}",
                "*",
                "*",
                "*_desc-prfestimates.h5",
            )
        )

        su = [path.basename(a).split("_")[0].split("-")[-1] for a in anas]
        se = [path.basename(a).split("_")[1].split("-")[-1] for a in anas]
        ta = [path.basename(a).split("_")[2].split("-")[-1] for a in anas]
        ru = [path.basename(a).split("_")[3].split("-")[-1] for a in anas]

        anasDF = pd.DataFrame(
            np.array([su, se, ta, ru]).T, columns=["subject", "session", "task", "run"]
        )

        if len(anasDF) == 0:
            print("no analyses were found!")
            return

        subj_pat = cls._pattern(subjects, use_glob)
        sess_pat = cls._pattern(sessions, use_glob)
        task_pat = cls._pattern(tasks, use_glob)
        run_pat = cls._pattern(runs, use_glob)

        anasDF = anasDF[
            (anasDF["subject"].str.match(subj_pat))
            & (anasDF["session"].str.match(sess_pat))
            & (anasDF["task"].str.match(task_pat))
            & (anasDF["run"].str.match(run_pat))
        ]

        anasDF = anasDF.drop_duplicates().reset_index(drop=True)
        anasDF = anasDF.sort_values(
            ["subject", "session", "task", "run"], ignore_index=True
        )

        if len(anasDF) >= 10:
            pbar = tqdm(total=len(anasDF))
        for I, a in anasDF.iterrows():
            try:
                anasDF.loc[I, "prf"] = PRF.from_hdf5(
                    study=study,
                    subject=a["subject"],
                    session=a["session"],
                    task=a["task"],
                    run=a["run"],
                    method=method,
                    analysis=prfanalyze,
                    hemi=hemi,
                    baseP=baseP,
                    orientation=orientation,
                )
            except Exception as e:
                print(
                    f'could not load sub-{a["subject"]}_ses-{a["session"]}_task-{a["task"]}_run-{a["run"]}!'
                )
                print(e)
                anasDF.drop(I).reset_index(drop=True)
            if len(anasDF) >= 10:
                pbar.update(1)
        if len(anasDF) >= 10:
            pbar.close()

        # check if something was actually found
        if len(anasDF) == 0:
            raise Warning("No analyses were found!")

        return cls(
            data_from="hdf5",
            study=study,
            DF=anasDF,
            baseP=baseP,
            derivatives_path=derivatives_path,
            orientation=orientation,
            hemi=hemi,
            prfanalyze=prfanalyze,
            method=method,
        )

    # --------------------------ALTERNATIVE  CONSTRUCTORS--------------------------#
    @classmethod
    def from_samsrf(
        cls,
        study,
        subjects=".*",
        sessions=".*",
        tasks=".*",
        runs=".*",
        use_glob: bool = False,
        prfanalyze="01",
        baseP=None,
        orientation="VF",
    ):
        """
        Alternative constructor to create PRFgroup from SAMS RF data.

        Args:
            cls: The class being instantiated.
            study (str): Study name.
            subjects (str, optional): Regex for subject selection.
            sessions (str, optional): Regex for session selection.
            tasks (str, optional): Regex for task selection.
            runs (str, optional): Regex for run selection.
            use_glob (bool, optional): Interpret the selection strings as shell-style
                wildcards and translate them to regex automatically.
            prfanalyze (str, optional): prfanalyze analysis.
            baseP (str, optional): Base path for data.
            orientation (str, optional): Orientation ('VF' or 'MP').

        Returns:
            PRFgroup: An instance of PRFgroup populated with SAMS RF data.
        """
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
                    "BIDS" "derivatives",
                )

                if not path.isdir(derivatives_path):
                    raise FileNotFoundError(
                        f"could not find derivatives folder at {derivatives_path} or {path.join(baseP, study, 'derivatives')}"
                    )

        anas = glob(
            path.join(
                derivatives_path,
                "samsrf",
                f"analysis-{prfanalyze}",
                "sub-*",
                "ses-*",
                "bi*_pRF-results.mat",
            )
        )

        su = [path.basename(a).split("_")[1].split("-")[-1] for a in anas]
        se = [path.basename(a).split("_")[2].split("-")[-1] for a in anas]
        ta = [path.basename(a).split("_")[3].split("-")[-1] for a in anas]
        ru = [path.basename(a).split("_")[4].split("-")[-1] for a in anas]

        anasDF = pd.DataFrame(
            np.array([su, se, ta, ru, anas]).T,
            columns=["subject", "session", "task", "run", "path"],
        )

        subj_pat = cls._pattern(subjects, use_glob)
        sess_pat = cls._pattern(sessions, use_glob)
        task_pat = cls._pattern(tasks, use_glob)
        run_pat = cls._pattern(runs, use_glob)

        anasDF = anasDF[
            (anasDF["subject"].str.match(subj_pat))
            & (anasDF["session"].str.match(sess_pat))
            & (anasDF["task"].str.match(task_pat))
            & (anasDF["run"].str.match(run_pat))
        ]

        if len(anasDF) == 0:
            print("no analyses were found!")
            return

        anasDF = anasDF.drop_duplicates().reset_index(drop=True)
        anasDF = anasDF.sort_values(
            ["subject", "session", "task", "run"], ignore_index=True
        )
        anasDF = anasDF[["hemi-R" not in a for a in anasDF["path"]]]

        if len(anasDF) >= 10:
            pbar = tqdm(total=len(anasDF))
        for I, a in anasDF.iterrows():
            try:
                if "_hemi-L" in a["path"]:
                    he = ""
                else:
                    he = None

                anasDF.loc[I, "prf"] = PRF.from_samsrf(
                    study=study,
                    subject=a["subject"],
                    session=a["session"],
                    task=a["task"],
                    run=a["run"],
                    analysis=prfanalyze,
                    baseP=baseP,
                    orientation=orientation,
                    hemis=he,
                )
            except Exception as e:
                print(
                    f'could not load sub-{a["subject"]}_ses-{a["session"]}_task-{a["task"]}_run-{a["run"]}!'
                )
                print(e)
                anasDF.drop(I).reset_index(drop=True)
            if len(anasDF) >= 10:
                pbar.update(1)
        if len(anasDF) >= 10:
            pbar.close()

        # check if something was actually found
        if len(anasDF) == 0:
            raise Warning("No analyses were found!")

        return cls(
            data_from="samsrf",
            study=study,
            DF=anasDF,
            baseP=baseP,
            derivatives_path=derivatives_path,
            orientation=orientation,
            prfanalyze=prfanalyze,
        )

    # ------------------------- ALTERNATIVE CONSTRUCTORS -------------------------#
    @classmethod
    def from_mrVista(
        cls,
        study,
        subjects=".*",
        sessions=".*",
        analysis=".*",
        use_glob: bool = False,
        baseP=None,
        orientation="VF",
        fit="fFit-fFit-fFit",
    ):
        """
        Alternative constructor to create PRFgroup from mrVista data.

        Args:
            cls: The class being instantiated.
            study (str): Study name.
            subjects (str, optional): Regex for subject selection.
            sessions (str, optional): Regex for session selection.
            analysis (str, optional): Regex for analysis selection.
            use_glob (bool, optional): Interpret the selection strings as shell-style
                wildcards and translate them to regex automatically.
            baseP (str, optional): Base path for data.
            orientation (str, optional): Orientation ('VF' or 'MP').
            fit (str, optional): Fit name.

        Returns:
            PRFgroup: An instance of PRFgroup populated with mrVista data.
        """
        if not baseP:
            baseP = "/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data"

        anas = glob(path.join(baseP, study, "subjects", "*", "*", "mrVista", "*"))

        anas = [a for a in anas if len(glob(path.join(a, "Gray", "*", f"*{fit}.mat")))]

        su = [a.split("/")[-4] for a in anas]
        se = [a.split("/")[-3] for a in anas]
        an = [a.split("/")[-1] for a in anas]

        anasDF = pd.DataFrame(
            np.array([su, se, an]).T, columns=["subject", "session", "analysis"]
        )

        subj_pat = cls._pattern(subjects, use_glob)
        sess_pat = cls._pattern(sessions, use_glob)
        ana_pat = cls._pattern(analysis, use_glob)

        anasDF = anasDF[
            (anasDF["subject"].str.match(subj_pat))
            & (anasDF["session"].str.match(sess_pat))
            & (anasDF["analysis"].str.match(ana_pat))
        ]

        anasDF = anasDF.drop_duplicates().reset_index()
        anasDF = anasDF.sort_values(
            ["subject", "session", "analysis"], ignore_index=True
        )

        if len(anasDF) == 0:
            print("no analyses were found!")
            return

        for I, a in anasDF.iterrows():
            try:
                anasDF.loc[I, "prf"] = PRF.from_mrVista(
                    study=study,
                    subject=a["subject"],
                    session=a["session"],
                    analysis=a["analysis"],
                    server="ceph",
                    forcePath=None,
                    orientation=orientation,
                    fit=fit,
                )
            except:
                print(
                    f'could not load subject: {a["subject"]}; session: {a["session"]}; analysis: {a["analysis"]}!'
                )
                anasDF.drop(I).reset_index(drop=True)

        # check if something was actually found
        if len(anasDF) == 0:
            raise Warning("No analyses were found!")

        return cls(
            data_from="mrVista",
            study=study,
            DF=anasDF,
            baseP=baseP,
            orientation=orientation,
            fit=fit,
        )

    # ---------------------------------- MASKING ---------------------------------#
    def maskROI(self, area="V1", atlas="benson", doV123=False, forcePath=False):
        """
        Apply ROI masking to all PRF objects in the group.

        Args:
            area (str): ROI area name.
            atlas (str): Atlas name.
            doV123 (bool, optional): Flag to include V1, V2, V3.
            forcePath (bool, optional): Flag to force path.
        """
        self.iterate("maskROI", area, atlas, doV123, forcePath)

    def maskVarExp(
        self, varExpThresh, varexp_easy=False, highThresh=None, spmPath=None
    ):
        """
        Apply variance explained masking to all PRF objects in the group.

        Args:
            varExpThresh (float): Variance explained threshold.
            varexp_easy (bool, optional): Flag for easy variance explained.
            highThresh (float, optional): High threshold value.
            spmPath (str, optional): Path to SPM.
        """
        self.iterate("maskVarExp", varExpThresh, varexp_easy, highThresh, spmPath)

    def maskEcc(self, rad):
        """
        Apply eccentricity masking to all PRF objects in the group.

        Args:
            rad (float): Eccentricity radius.
        """
        self.iterate("maskEcc", rad)

    def maskSigma(self, s_min, s_max=None):
        """
        Apply sigma masking to all PRF objects in the group.

        Args:
            s_min (float): Minimum sigma value.
            s_max (float, optional): Maximum sigma value.
        """
        self.iterate("maskSigma", s_min, s_max)

    def maskBetaThresh(self, betaMax=50, doBorderThresh=False, doHighBetaThresh=True):
        """
        Apply beta threshold masking to all PRF objects in the group.

        Args:
            betaMax (float): Maximum beta value.
            doBorderThresh (bool, optional): Flag to apply border threshold.
            doHighBetaThresh (bool, optional): Flag to apply high beta threshold.
        """
        self.iterate("maskBetaThresh", betaMax, doBorderThresh, doHighBetaThresh)

    # ----------------------------------- QUERY ----------------------------------#
    def query(self, s, silent=False):
        """
        Query the PRFgroup with a string expression.

        Args:
            s (str): Query string.
            silent (bool, optional): Suppress warning message.

        Returns:
            PRFgroup: A new PRFgroup instance with the query result.
        """
        if not silent:
            print("Warning, no full class will be returned!")

        return PRFgroup(
            data_from=self.data_from,
            study=self.study,
            DF=self.data.query(s).reset_index(drop=True),
            baseP=self.baseP,
            orientation=self.orientation,
            prfanalyze=None,
            hemi=None,
            fit=None,
            method=None,
        )

    # ----------------------------------- APPEND ----------------------------------#
    def append(self, to_append):
        """
        Append another PRFgroup to this group.

        Args:
            to_append (PRFgroup): The PRFgroup to append.
        """
        print("Warning: just the data will be appended!")

        self.data = pd.concat([self.data, to_append.data]).reset_index(drop=True)

    # ---------------------------------- ITERATE ---------------------------------#
    def iterate(self, attr_name, *args, **kwargs):
        """
        Call a method or access a property on all PRF objects in the group.

        This allows you to apply any PRF method or property across all objects
        in the group with optional arguments.

        Args:
            attr_name (str): Name of the method or property to call on each PRF object.
            *args: Positional arguments to pass to the method.
            **kwargs: Keyword arguments to pass to the method.

        Returns:
            list: List of return values from each PRF object, or None if no return values.

        Examples:
            >>> # Call a method without arguments
            >>> grp.iterate('plot_covMap')

            >>> # Call a method with arguments
            >>> grp.iterate('maskROI', area='V1', atlas='benson')

            >>> # Get a property from all objects
            >>> results = grp.iterate('x0')
        """
        results = []
        for I, a in self.data.iterrows():
            prf_obj = a["prf"]
            try:
                attr = getattr(prf_obj, attr_name)
            except:
                attr = None

            # Check if it's a callable (method) or a property
            if callable(attr):
                result = attr(*args, **kwargs)
            else:
                result = attr

            results.append(result)

        # Return results if they contain meaningful data, otherwise None
        if all(r is None for r in results):
            return None
        return results

    # --------------------------------- PROPERTIS --------------------------------#
    @property
    def subject(self):
        """
        Get unique subjects in the group.

        Returns:
            np.ndarray: Array of unique subjects.
        """
        return self.data["subject"].unique()

    @property
    def session(self):
        """
        Get unique sessions in the group.

        Returns:
            np.ndarray: Array of unique sessions.
        """
        return self.data["session"].unique()

    @property
    def task(self):
        """
        Get the task information for the group.

        Returns:
            str or np.ndarray: Task name(s) for the group.
        """
        if self.data_from == "docker" or self.data_from == "samsrf":
            t = self.data["task"].unique()
            return t if len(t) > 1 else t[0]
        else:
            print("No task field in mrVista data")

    @property
    def run(self):
        """
        Get the run information for the group.

        Returns:
            np.ndarray: Array of unique runs.
        """
        if self.data_from == "docker" or self.data_from == "samsrf":
            return self.data["run"].unique()
        else:
            print("No run field in mrVista data")

    @property
    def analysis(self):
        """
        Get the analysis information for the group.

        Returns:
            np.ndarray: Array of unique analyses.
        """
        if self.data_from == "mrVista":
            return self.data["analysis"].unique()
        else:
            print("No analysis field in docker data")

    def _get_prf_property(self, prop, combine_fn=np.hstack):
        """
        Get a property from all PRF objects in the group.

        Args:
            prop (str): Property name.
            combine_fn (callable, optional): Function to combine property values.

        Returns:
            np.ndarray: Combined property values.
        """
        return combine_fn([getattr(a["prf"], prop) for _, a in self.data.iterrows()])

    @property
    def x0(self):
        """
        Get the x0 property for all PRF objects in the group.

        Returns:
            np.ndarray: Combined x0 values.
        """
        return self._get_prf_property("x0")

    @property
    def y0(self):
        """
        Get the y0 property for all PRF objects in the group.

        Returns:
            np.ndarray: Combined y0 values.
        """
        return self._get_prf_property("y0")

    @property
    def s0(self):
        """
        Get the s0 property for all PRF objects in the group.

        Returns:
            np.ndarray: Combined s0 values.
        """
        return self._get_prf_property("s0")

    @property
    def sigma0(self):
        """
        Get the sigma0 property for all PRF objects in the group.

        Returns:
            np.ndarray: Combined sigma0 values.
        """
        return self.s0

    @property
    def r0(self):
        """
        Get the r0 property for all PRF objects in the group.

        Returns:
            np.ndarray: Combined r0 values.
        """
        return self._get_prf_property("r0")

    @property
    def ecc0(self):
        """
        Get the ecc0 property for all PRF objects in the group.

        Returns:
            np.ndarray: Combined ecc0 values.
        """
        return self.r0

    @property
    def phi0(self):
        """
        Get the phi0 property for all PRF objects in the group.

        Returns:
            np.ndarray: Combined phi0 values.
        """
        return self._get_prf_property("phi0")

    @property
    def pol0(self):
        """
        Get the pol0 property for all PRF objects in the group.

        Returns:
            np.ndarray: Combined pol0 values.
        """
        return self.phi0

    @property
    def pol0_orig(self):
        """
        Get the pol0_orig property for all PRF objects in the group.

        Returns:
            np.ndarray: Combined pol0 values.
        """
        return self._get_prf_property("pol0_orig")

    @property
    def varexp0(self):
        """
        Get the varexp0 property for all PRF objects in the group.

        Returns:
            np.ndarray: Combined varexp0 values.
        """
        return self._get_prf_property("varexp0")

    @property
    def varexp_easy0(self):
        """
        Get the varexp_easy0 property for all PRF objects in the group.

        Returns:
            np.ndarray: Combined varexp_easy0 values.
        """
        return self._get_prf_property("varexp_easy0")

    @property
    def beta0(self):
        """
        Get the beta0 property for all PRF objects in the group.

        Returns:
            np.ndarray: Combined beta0 values.
        """
        return self._get_prf_property("beta0")

    @property
    def x(self):
        """
        Get the x property for all PRF objects in the group.

        Returns:
            np.ndarray: Combined x values.
        """
        return self._get_prf_property("x")

    @property
    def y(self):
        """
        Get the y property for all PRF objects in the group.

        Returns:
            np.ndarray: Combined y values.
        """
        return self._get_prf_property("y")

    @property
    def s(self):
        """
        Get the s property for all PRF objects in the group.

        Returns:
            np.ndarray: Combined s values.
        """
        return self._get_prf_property("s")

    @property
    def sigma(self):
        """
        Get the sigma property for all PRF objects in the group.

        Returns:
            np.ndarray: Combined sigma values.
        """
        return self.s

    @property
    def r(self):
        """
        Get the r property for all PRF objects in the group.

        Returns:
            np.ndarray: Combined r values.
        """
        return self._get_prf_property("r")

    @property
    def ecc(self):
        """
        Get the eccentricity property for all PRF objects in the group.

        Returns:
            np.ndarray: Combined eccentricity values.
        """
        return self.__reduce__

    @property
    def phi(self):
        """
        Get the phi property for all PRF objects in the group.

        Returns:
            np.ndarray: Combined phi values.
        """
        return self._get_prf_property("phi")

    @property
    def pol(self):
        """
        Get the polar angle property for all PRF objects in the group.

        Returns:
            np.ndarray: Combined polar angle values.
        """
        return self.phi

    @property
    def pol_orig(self):
        """
        Get the polar angle property for all PRF objects in the group.

        Returns:
            np.ndarray: Combined polar angle values.
        """
        return self._get_prf_property("pol_orig")

    @property
    def varexp(self):
        """
        Get the variance explained property for all PRF objects in the group.

        Returns:
            np.ndarray: Combined variance explained values.
        """
        return self._get_prf_property("varexp")

    @property
    def varexp_easy(self):
        """
        Get the easy variance explained property for all PRF objects in the group.

        Returns:
            np.ndarray: Combined easy variance explained values.
        """
        return self._get_prf_property("varexp_easy")

    @property
    def meanVarExp(self):
        """
        Calculate the mean variance explained across all PRF objects in the group.

        Returns:
            float: Mean variance explained value.
        """
        values = [getattr(a["prf"], "meanVarExp") for _, a in self.data.iterrows()]
        return np.mean(values)

    @property
    def beta(self):
        """
        Get the beta property for all PRF objects in the group.

        Returns:
            np.ndarray: Combined beta values.
        """
        return self._get_prf_property("beta")

    # ------------------------- SUBJECT VARS as MATRIX -----------------------#

    def _get_sub_property(self, prop):
        """
        Get a subject-specific property from all PRF objects in the group.

        Args:
            prop (str): Property name.

        Returns:
            dict: Dictionary with subject IDs as keys and property values as arrays.
        """
        cache_attr = "_sub_" + prop
        if not hasattr(self, cache_attr):
            setattr(
                self,
                cache_attr,
                {
                    s: np.array(
                        [
                            getattr(a["prf"], prop)
                            for _, a in self.data.iterrows()
                            if a["subject"] == s
                        ]
                    )
                    for s in self.subject
                },
            )
        return getattr(self, cache_attr)

    @property
    def sub_x0(self):
        """
        Get the x0 property for each subject in the group.

        Returns:
            dict: Subject-specific x0 values.
        """
        return self._get_sub_property("x0")

    @property
    def sub_y0(self):
        """
        Get the y0 property for each subject in the group.

        Returns:
            dict: Subject-specific y0 values.
        """
        return self._get_sub_property("y0")

    @property
    def sub_s0(self):
        """
        Get the s0 property for each subject in the group.

        Returns:
            dict: Subject-specific s0 values.
        """
        return self._get_sub_property("s0")

    @property
    def sub_r0(self):
        """
        Get the r0 property for each subject in the group.

        Returns:
            dict: Subject-specific r0 values.
        """
        return self._get_sub_property("r0")

    @property
    def sub_pol0(self):
        """
        Get the pol0 property for each subject in the group.

        Returns:
            dict: Subject-specific pol0 values.
        """
        return self._get_sub_property("pol0")

    @property
    def sub_varexp0(self):
        """
        Get the varexp0 property for each subject in the group.

        Returns:
            dict: Subject-specific varexp0 values.
        """
        return self._get_sub_property("varexp0")

    @property
    def sub_x(self):
        """
        Get the x property for each subject in the group.

        Returns:
            dict: Subject-specific x values.
        """
        return self._get_sub_property("x")

    @property
    def sub_y(self):
        """
        Get the y property for each subject in the group.

        Returns:
            dict: Subject-specific y values.
        """
        return self._get_sub_property("y")

    @property
    def sub_s(self):
        """
        Get the s property for each subject in the group.

        Returns:
            dict: Subject-specific s values.
        """
        return self._get_sub_property("s")

    @property
    def sub_r(self):
        """
        Get the r property for each subject in the group.

        Returns:
            dict: Subject-specific r values.
        """
        return self._get_sub_property("r")

    @property
    def sub_pol(self):
        """
        Get the polar angle property for each subject in the group.

        Returns:
            dict: Subject-specific polar angle values.
        """
        return self._get_sub_property("pol")

    @property
    def sub_varexp(self):
        """
        Get the variance explained property for each subject in the group.

        Returns:
            dict: Subject-specific variance explained values.
        """
        return self._get_sub_property("varexp")

    # --------------------------------- MASKS --------------------------------#
    @property
    def mask(self):
        """
        Get the combined mask from all PRF objects in the group.

        Returns:
            np.ndarray: Combined mask array.
        """
        return np.hstack([a["prf"].mask for I, a in self.data.iterrows()])

    @property
    def varExpMsk(self):
        """
        Get the combined variance explained mask from all PRF objects in the group.

        Returns:
            np.ndarray: Combined variance explained mask array.
        """
        return np.hstack([a["prf"].varExpMsk for I, a in self.data.iterrows()])

    @property
    def roiMsk(self):
        """
        Get the combined ROI mask from all PRF objects in the group.

        Returns:
            np.ndarray: Combined ROI mask array.
        """
        return np.hstack([a["prf"].roiMsk for I, a in self.data.iterrows()])

    @property
    def sub_mask(self):
        """
        Get the subject-specific masks for all PRF objects in the group.

        Returns:
            dict: Dictionary with subject IDs as keys and combined mask arrays as values.
        """
        self._sub_mask = {}
        for s in self.subject:
            self._sub_mask[s] = np.all(
                [a["prf"].mask for I, a in self.data.iterrows() if a["subject"] == s],
                0,
            )
        return self._sub_mask

    @property
    def doROIMsk(self):
        """
        Get or set the ROI masking flag for the group.

        Returns:
            bool: Current ROI masking flag value.
        """
        return self._doROIMsk

    @doROIMsk.setter
    def doROIMsk(self, value: bool):
        """
        Set the ROI masking flag for the group.

        Args:
            value (bool): New ROI masking flag value.
        """
        for I, a in self.data.iterrows():
            a["prf"].doROIMsk = value
        self._doROIMsk = value

    @property
    def doVarExpMsk(self):
        """
        Get or set the variance explained masking flag for the group.

        Returns:
            bool: Current variance explained masking flag value.
        """
        return self._doVarExpMsk

    @doVarExpMsk.setter
    def doVarExpMsk(self, value: bool):
        """
        Set the variance explained masking flag for the group.

        Args:
            value (bool): New variance explained masking flag value.
        """
        for I, a in self.data.iterrows():
            a["prf"].doVarExpMsk = value
        self._doVarExpMsk = value

    @property
    def doBetaMsk(self):
        """
        Get or set the beta threshold masking flag for the group.

        Returns:
            bool: Current beta threshold masking flag value.
        """
        return self._doBetaMsk

    @doBetaMsk.setter
    def doBetaMsk(self, value: bool):
        """
        Set the beta threshold masking flag for the group.

        Args:
            value (bool): New beta threshold masking flag value.
        """
        for I, a in self.data.iterrows():
            a["prf"].doBetaMsk = value
        self._doBetaMsk = value

    @property
    def doEccMsk(self):
        """
        Get or set the eccentricity masking flag for the group.

        Returns:
            bool: Current eccentricity masking flag value.
        """
        return self._doEccMsk

    @doEccMsk.setter
    def doEccMsk(self, value: bool):
        """
        Set the eccentricity masking flag for the group.

        Args:
            value (bool): New eccentricity masking flag value.
        """
        for I, a in self.data.iterrows():
            a["prf"].doEccMsk = value
        self._doEccMsk = value

    @property
    def doSigMsk(self):
        """
        Get or set the significance masking flag for the group.

        Returns:
            bool: Current significance masking flag value.
        """
        return self._doSigMsk

    @doEccMsk.setter
    def doSigMsk(self, value: bool):
        """
        Set the significance masking flag for the group.

        Args:
            value (bool): New significance masking flag value.
        """
        for I, a in self.data.iterrows():
            a["prf"].doSigMsk = value
        self._doSigMsk = value

    # -------------------------------- PLOTTING ---------------------------------#
    def plot_coverage_summary_pdf(
        self,
        maxEcc=None,
        method="max",
        cmapMin=0,
        output_format="svg",
        force_regenerate=False,
        custom_suffix="",
    ):
        """
        Generate a PDF summary of coverage maps for all subjects and sessions.

        Creates a multi-page PDF where each page shows one subject with coverage maps
        organized by session-task combinations (rows) and runs (columns). Each row shows
        a specific session-task pair (e.g., ses-1_task-1, ses-1_task-2). Individual runs
        are shown first, followed by averaged runs if available.

        Args:
            maxEcc (float, optional): Maximum eccentricity for coverage maps. If None,
                will attempt to use value from PRF objects or require user input.
            method (str, optional): Method for combining runs ('max', 'mean', etc.).
                Default is 'max'.
            cmapMin (float, optional): Minimum value for colormap. Default is 0.
            output_format (str, optional): Format for coverage map images ('svg', 'png', 'pdf').
                Default is 'svg'.
            force_regenerate (bool, optional): If True, regenerate all coverage maps even
                if they already exist. Default is False.
            custom_suffix (str, optional): Custom string to append to filename before
                '_covMap', starting with "-". Default is "".

        Returns:
            str: Path to the generated PDF file.

        Notes:
            - PDF is saved to baseP/reportlab/summary/
            - Each page shows one subject with session-task labels rotated on the left
            - Run numbers are displayed above each coverage map
            - Page dimensions adapt based on number of session-task combinations and runs
            - Each row represents a unique session-task pair (e.g., task-1_run-1, task-1_run-2)
        """
        if not HAS_PLACE_IMAGE:
            print(
                "Warning: placeImage function not available. Using basic image placement."
            )

        # Organize data by subject and session
        subjects = self.subject

        # Collect data for all subjects
        all_subjects_data = {}

        # Calculate total number of coverage maps to generate
        total_maps = 0
        for subj in subjects:
            subj_data = self.data[self.data["subject"] == subj]
            for sess in subj_data["session"].unique():
                sess_data = subj_data[subj_data["session"] == sess]
                for task in sess_data["task"].unique():
                    task_data = sess_data[sess_data["task"] == task]
                    total_maps += len(task_data["run"].unique())

        print(f"Processing {len(subjects)} subjects with {total_maps} coverage maps...")
        pbar = tqdm(total=total_maps, desc="Generating coverage maps")

        for subj in subjects:
            # Filter data for this subject
            subj_data = self.data[self.data["subject"] == subj]

            # Get unique sessions for this subject
            sessions = sorted(subj_data["session"].unique())

            # Organize by session, then task, then runs
            session_task_runs = {}
            for sess in sessions:
                sess_data = subj_data[subj_data["session"] == sess]
                session_task_runs[sess] = {}

                # Get unique tasks for this session
                tasks = sorted(sess_data["task"].unique())

                for task in tasks:
                    task_data = sess_data[sess_data["task"] == task]
                    runs = sorted(task_data["run"].unique())

                    # Move runs containing "avg" to the end
                    runs_with_avg = [r for r in runs if "avg" in str(r).lower()]
                    runs_without_avg = [r for r in runs if "avg" not in str(r).lower()]
                    session_task_runs[sess][task] = sorted(runs_without_avg) + sorted(
                        runs_with_avg
                    )

            # Generate or find coverage map paths
            coverage_map_paths = {}
            for sess in sessions:
                coverage_map_paths[sess] = {}

                for task in session_task_runs[sess]:
                    coverage_map_paths[sess][task] = {}

                    for run in session_task_runs[sess][task]:
                        # Get the PRF object for this subject/session/task/run
                        prf_obj = subj_data[
                            (subj_data["session"] == sess)
                            & (subj_data["task"] == task)
                            & (subj_data["run"] == run)
                        ]["prf"].values[0]

                        # Try to get existing coverage map path
                        try:
                            cov_path = prf_obj._get_covMap_savePath()
                            if path.isfile(cov_path) and not force_regenerate:
                                coverage_map_paths[sess][task][run] = cov_path
                                pbar.set_postfix_str(
                                    f"Found: sub-{subj}_ses-{sess}_task-{task}_run-{run}"
                                )
                            else:
                                # Generate coverage map
                                pbar.set_postfix_str(
                                    f"Generating: sub-{subj}_ses-{sess}_task-{task}_run-{run}"
                                )
                                coverage_map_paths[sess][task][run] = (
                                    self._generate_coverage_map(
                                        prf_obj, maxEcc, method, cmapMin, output_format
                                    )
                                )
                        except Exception as e:
                            pbar.write(
                                f"  Warning: Could not get/generate coverage map for sub-{subj}_ses-{sess}_task-{task}_run-{run}: {e}"
                            )
                            coverage_map_paths[sess][task][run] = None
                        finally:
                            pbar.update(1)

            # Store data for this subject
            all_subjects_data[subj] = {
                "sessions": sessions,
                "session_task_runs": session_task_runs,
                "coverage_map_paths": coverage_map_paths,
            }

        # Close progress bar
        pbar.close()

        # Create single PDF with all subjects (one page per subject)
        pdf_path = self._create_multi_subject_pdf(all_subjects_data, custom_suffix)
        print(f"Created PDF: {pdf_path}")

        return pdf_path

    def _generate_coverage_map(self, prf_obj, maxEcc, method, cmapMin, output_format):
        """
        Generate a coverage map for a PRF object.

        Args:
            prf_obj: PRF object to generate coverage map for
            maxEcc: Maximum eccentricity
            method: Method for combining data
            cmapMin: Minimum colormap value
            output_format: Output format ('svg', 'png', 'pdf')

        Returns:
            str: Path to generated coverage map
        """
        # Determine maxEcc if not provided
        if maxEcc is None:
            if hasattr(prf_obj, "maxEcc"):
                maxEcc = prf_obj.maxEcc
            else:
                raise ValueError(
                    "maxEcc must be provided or available in PRF object as .maxEcc"
                )

        # Generate the coverage map
        try:
            prf_obj.plot_covMap(
                maxEcc=maxEcc, method=method, cmapMin=cmapMin, save=True, show=False
            )
            cov_path = prf_obj._get_covMap_savePath()

            # Convert to desired format if needed
            if output_format != "svg" and cov_path.endswith(".svg"):
                # Convert format using matplotlib
                base_path = cov_path.rsplit(".", 1)[0]
                new_path = f"{base_path}.{output_format}"
                # This assumes plot_covMap can handle output_format parameter
                # or we need to convert the file
                return new_path

            return cov_path
        except Exception as e:
            print(f"    Error generating coverage map: {e}")
            return None

    def _create_multi_subject_pdf(self, all_subjects_data, custom_suffix=""):
        """
        Create a single PDF with all subjects, one page per subject.

        Args:
            all_subjects_data: Dict mapping subject IDs to their data
                (sessions, session_task_runs, coverage_map_paths)
            custom_suffix: Custom string to append to filename before '_coverage_summary'

        Returns:
            str: Path to created PDF
        """
        # Create output directory
        output_dir = path.join(self.baseP, self.study, "reportlab", "summary")
        makedirs(output_dir, exist_ok=True)

        # Determine PDF filename based on first PRF object's naming convention
        first_prf = self.data["prf"].values[0]
        try:
            base_name = path.basename(first_prf._get_covMap_savePath())
            # Insert custom suffix before _covMap if provided
            if custom_suffix:
                # Ensure suffix starts with '-'
                suffix = (
                    custom_suffix
                    if custom_suffix.startswith("-")
                    else f"-{custom_suffix}"
                )
                base_name = base_name.replace("_covmap", f"{suffix}_covmap")
            # Modify name for summary
            base_name = base_name.replace("_covmap", "_coverage_summary")
            # Remove subject/session/run-specific info
            base_name = re.sub(r"sub-[^_]+_", "", base_name)
            base_name = re.sub(r"ses-[^_]+_", "", base_name)
            base_name = re.sub(r"_run-\d+", "", base_name)
            pdf_filename = base_name.replace(".svg", ".pdf").replace(".png", ".pdf")
            if not pdf_filename.startswith("group_"):
                pdf_filename = f"group_{pdf_filename}"
        except:
            suffix = (
                custom_suffix
                if not custom_suffix or custom_suffix.startswith("_desc-")
                else f"_desc-{custom_suffix}"
            )
            pdf_filename = (
                f"group{suffix}_coverage_summary.pdf"
                if custom_suffix
                else "group_coverage_summary.pdf"
            )

        pdf_path = path.join(output_dir, pdf_filename)

        # Calculate maximum dimensions across all subjects
        max_sessions_global = 0
        max_columns_global = 0
        for subj_data in all_subjects_data.values():
            sessions = subj_data["sessions"]
            session_task_runs = subj_data["session_task_runs"]

            max_sessions_global = max(max_sessions_global, len(sessions))

            for sess in sessions:
                tasks = session_task_runs.get(sess, {})
                columns = sum(len(runs) for runs in tasks.values()) if tasks else 0
                max_columns_global = max(max_columns_global, columns)

        # Set dimensions (in cm)
        plot_width = 8.0  # Width per coverage map
        plot_height = 8.0  # Height per coverage map
        margin = 0.5
        label_space = 1.0  # Space for labels
        title_space = 1.5  # Space for title

        page_width = (max_columns_global * plot_width + label_space + 2 * margin) * cm
        page_height = (
            max_sessions_global * plot_height + title_space + 2 * margin
        ) * cm

        # Create canvas
        c = canvas.Canvas(pdf_path, pagesize=(page_width, page_height))

        # Create progress bar for PDF page generation
        print("\nCreating PDF with all subjects...")
        pbar_pdf = tqdm(total=len(all_subjects_data), desc="Writing PDF pages")

        # Draw each subject on a separate page
        for subj_idx, (subject, subj_data) in enumerate(all_subjects_data.items()):
            sessions = subj_data["sessions"]
            session_task_runs = subj_data["session_task_runs"]
            coverage_map_paths = subj_data["coverage_map_paths"]

            # Draw title (subject name)
            c.setFont("Helvetica-Bold", 16)
            title_x = page_width / 2
            title_y = page_height - margin * cm
            c.drawCentredString(title_x, title_y, f"Subject {subject}")

            # Starting positions
            start_y = page_height - (margin + title_space) * cm
            start_x = (margin + label_space) * cm

            # Draw coverage maps organized by session rows and task/run columns
            for sess_idx, sess in enumerate(sessions):
                y_pos = start_y - (sess_idx * plot_height * 1.05 * cm)

                # Draw session label (rotated)
                c.saveState()
                label_x = margin * cm + 0.5 * cm
                label_y = y_pos - (plot_height * cm / 2)
                c.translate(label_x, label_y)
                c.rotate(90)
                c.setFont("Helvetica-Bold", 12)
                c.drawCentredString(0, 0, f"ses-{sess}")
                c.restoreState()

                # Sequentially draw task/run columns for this session
                col_idx = 0
                tasks = session_task_runs.get(sess, {})
                for task in sorted(tasks.keys()):
                    runs = tasks[task]
                    for run in runs:
                        x_pos = start_x + (col_idx * plot_width * 0.95 * cm)

                        # Draw combined task and run label
                        c.setFont("Helvetica", 10)
                        c.drawCentredString(
                            x_pos + (plot_width * cm / 2),
                            y_pos + 0.05 * cm,
                            f"task-{task}_run-{run}",
                        )

                        # Place coverage map image
                        cov_path = coverage_map_paths[sess][task].get(run)
                        if cov_path and path.isfile(cov_path):
                            self._place_image_on_canvas(
                                c,
                                cov_path,
                                x_pos,
                                y_pos - plot_height * cm,
                                plot_width * cm,
                                plot_height * cm,
                            )
                        else:
                            # Draw placeholder
                            c.rect(
                                x_pos,
                                y_pos - plot_height * cm,
                                plot_width * cm,
                                plot_height * cm,
                            )
                            c.drawCentredString(
                                x_pos + (plot_width * cm / 2),
                                y_pos - (plot_height * cm / 2),
                                "No coverage map",
                            )

                        col_idx += 1

            # Update progress bar with subject info
            pbar_pdf.set_postfix_str(f"Completed: {subject}")
            pbar_pdf.update(1)

            # Start new page for next subject (if not the last one)
            if subj_idx < len(all_subjects_data) - 1:
                c.showPage()

        # Close progress bar
        pbar_pdf.close()

        # Save PDF
        c.save()
        return pdf_path

    def _place_image_on_canvas(self, c, image_path, x, y, width, height):
        """
        Place an image on the canvas at specified position and size.

        Args:
            c: ReportLab canvas object
            image_path: Path to image file
            x, y: Position coordinates (bottom-left corner)
            width, height: Desired dimensions
        """
        if HAS_PLACE_IMAGE:
            # Use the placeImage function from placeImages.py
            try:
                placeImage(
                    c,
                    image_path,
                    x,
                    y + height,  # placeImage expects top-left corner
                    height=height,
                    ttype="path",
                    forceAR=True,
                )
            except Exception as e:
                print(f"    Error using placeImage: {e}, falling back to basic method")
                self._place_image_basic(c, image_path, x, y, width, height)
        else:
            self._place_image_basic(c, image_path, x, y, width, height)

    def _place_image_basic(self, c, image_path, x, y, width, height):
        """
        Basic image placement fallback using reportlab's drawImage.

        Args:
            c: ReportLab canvas object
            image_path: Path to image file
            x, y: Position coordinates (bottom-left corner)
            width, height: Desired dimensions
        """
        try:
            if image_path.endswith(".svg"):
                # Convert SVG to PNG temporarily
                from reportlab.graphics import renderPDF
                from svglib.svglib import svg2rlg

                drawing = svg2rlg(image_path)
                if drawing:
                    # Scale to fit
                    scale_x = width / drawing.width
                    scale_y = height / drawing.height
                    scale = min(scale_x, scale_y)
                    drawing.width *= scale
                    drawing.height *= scale
                    drawing.scale(scale, scale)
                    renderPDF.draw(drawing, c, x, y)
            else:
                # Use standard image placement
                c.drawImage(
                    image_path,
                    x,
                    y,
                    width=width,
                    height=height,
                    preserveAspectRatio=True,
                    mask="auto",
                )
        except Exception as e:
            print(f"    Error placing image {image_path}: {e}")
            # Draw error placeholder
            c.rect(x, y, width, height)
            c.drawCentredString(x + width / 2, y + height / 2, "Image error")
