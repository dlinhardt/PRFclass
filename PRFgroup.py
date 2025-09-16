import re
from glob import glob
from os import path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import PRF


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

    # --------------------------ALTERNATIVE  CONSTRUCTORS--------------------------#
    @classmethod
    def from_docker(
        cls,
        study,
        subjects=".*",
        sessions=".*",
        tasks=".*",
        runs=".*",
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

        anasDF = anasDF[
            (anasDF["subject"].str.match(subjects))
            & (anasDF["session"].str.match(sessions))
            & (anasDF["task"].str.match(tasks))
            & (anasDF["run"].str.match(runs))
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
    def from_samsrf(
        cls,
        study,
        subjects=".*",
        sessions=".*",
        tasks=".*",
        runs=".*",
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

        anasDF = anasDF[
            (anasDF["subject"].str.match(subjects))
            & (anasDF["session"].str.match(sessions))
            & (anasDF["task"].str.match(tasks))
            & (anasDF["run"].str.match(runs))
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

        anasDF = anasDF[
            (anasDF["subject"].str.match(subjects))
            & (anasDF["session"].str.match(sessions))
            & (anasDF["analysis"].str.match(analysis))
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
        for I, a in self.data.iterrows():
            a["prf"].maskROI(area, atlas, doV123, forcePath)

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
        for I, a in self.data.iterrows():
            a["prf"].maskVarExp(varExpThresh, varexp_easy, highThresh, spmPath)

    def maskEcc(self, rad):
        """
        Apply eccentricity masking to all PRF objects in the group.

        Args:
            rad (float): Eccentricity radius.
        """
        for I, a in self.data.iterrows():
            a["prf"].maskEcc(rad)

    def maskSigma(self, s_min, s_max=None):
        """
        Apply sigma masking to all PRF objects in the group.

        Args:
            s_min (float): Minimum sigma value.
            s_max (float, optional): Maximum sigma value.
        """
        for I, a in self.data.iterrows():
            a["prf"].maskSigma(s_min, s_max)

    def maskBetaThresh(self, betaMax=50, doBorderThresh=False, doHighBetaThresh=True):
        """
        Apply beta threshold masking to all PRF objects in the group.

        Args:
            betaMax (float): Maximum beta value.
            doBorderThresh (bool, optional): Flag to apply border threshold.
            doHighBetaThresh (bool, optional): Flag to apply high beta threshold.
        """
        for I, a in self.data.iterrows():
            a["prf"].maskBetaThresh(betaMax, doBorderThresh, doHighBetaThresh)

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
