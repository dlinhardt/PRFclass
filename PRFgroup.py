import numpy as np
from os import path
from glob import glob
import re
import pandas as pd
from tqdm import tqdm
from . import PRF


class PRFgroup:
    """
    This is a list class for PRFclass.
    You can give study and masks for subjects/sessions/...
    as regexp or it will load everything.
    """

    def __init__(
        self,
        data_from,
        study,
        DF,
        baseP,
        orientation,
        prfanalyze=None,
        hemi=None,
        fit=None,
        method=None,
    ):
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

        self._doROIMsk = True
        self._doVarExpMsk = True
        self._doBetaMsk = True
        self._doEccMsk = True
        self._doSigMsk = True

    def __getitem__(self, item):
        return self.data.loc[item]

    def __len__(self):
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
    ):
        if not baseP:
            baseP = "/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data"

        anas = glob(
            path.join(
                baseP,
                study,
                "derivatives",
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
        if not baseP:
            baseP = "/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data"

        anas = glob(
            path.join(
                baseP,
                study,
                "derivatives",
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
        for I, a in self.data.iterrows():
            a["prf"].maskROI(area, atlas, doV123, forcePath)

    def maskVarExp(
        self, varExpThresh, varexp_easy=False, highThresh=None, spmPath=None
    ):
        for I, a in self.data.iterrows():
            a["prf"].maskVarExp(varExpThresh, varexp_easy, highThresh, spmPath)

    def maskEcc(self, rad):
        for I, a in self.data.iterrows():
            a["prf"].maskEcc(rad)

    def maskSigma(self, s_min, s_max=None):
        for I, a in self.data.iterrows():
            a["prf"].maskSigma(s_min, s_max)

    def maskBetaThresh(self, betaMax=50, doBorderThresh=False, doHighBetaThresh=True):
        for I, a in self.data.iterrows():
            a["prf"].maskBetaThresh(betaMax, doBorderThresh, doHighBetaThresh)

    # ----------------------------------- QUERY ----------------------------------#
    def query(self, s):
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
        print("Warning: just the data will be appended!")

        self.data = pd.concat([self.data, to_append.data]).reset_index(drop=True)

    # --------------------------------- PROPERTIS --------------------------------#
    @property
    def subject(self):
        return self.data["subject"].unique()

    @property
    def session(self):
        return self.data["session"].unique()

    @property
    def task(self):
        if self.data_from == "docker" or self.data_from == "samsrf":
            t = self.data["task"].unique()
            return t if len(t) > 1 else t[0]
        else:
            print("No task field in mrVista data")

    @property
    def run(self):
        if self.data_from == "docker" or self.data_from == "samsrf":
            return self.data["run"].unique()
        else:
            print("No run field in mrVista data")

    @property
    def analysis(self):
        if self.data_from == "mrVista":
            return self.data["analysis"].unique()
        else:
            print("No analysis field in docker data")

    @property
    def x0(self):
        return np.hstack([a["prf"].x0 for I, a in self.data.iterrows()])

    @property
    def y0(self):
        return np.hstack([a["prf"].y0 for I, a in self.data.iterrows()])

    @property
    def s0(self):
        return np.hstack([a["prf"].s0 for I, a in self.data.iterrows()])

    @property
    def sigma0(self):
        return self.s0

    @property
    def r0(self):
        return np.hstack([a["prf"].r0 for I, a in self.data.iterrows()])

    @property
    def ecc0(self):
        return self.r0

    @property
    def phi0(self):
        return np.hstack([a["prf"].phi0 for I, a in self.data.iterrows()])

    @property
    def pol0(self):
        return self.phi0

    @property
    def varexp0(self):
        return np.hstack([a["prf"].varexp0 for I, a in self.data.iterrows()])

    @property
    def varexp_easy0(self):
        return np.hstack([a["prf"].varexp_easy0 for I, a in self.data.iterrows()])

    @property
    def beta0(self):
        return np.hstack([a["prf"].beta0 for I, a in self.data.iterrows()])

    @property
    def x(self):
        return np.hstack([a["prf"].x for I, a in self.data.iterrows()])

    @property
    def y(self):
        return np.hstack([a["prf"].y for I, a in self.data.iterrows()])

    @property
    def s(self):
        return np.hstack([a["prf"].s for I, a in self.data.iterrows()])

    @property
    def sigma(self):
        return self.s

    @property
    def r(self):
        return np.hstack([a["prf"].r for I, a in self.data.iterrows()])

    @property
    def ecc(self):
        return self.s

    @property
    def phi(self):
        return np.hstack([a["prf"].phi for I, a in self.data.iterrows()])

    @property
    def pol(self):
        return self.phi

    @property
    def varexp(self):
        return np.hstack([a["prf"].varexp for I, a in self.data.iterrows()])

    @property
    def varexp_easy(self):
        return np.hstack([a["prf"].varexp_easy for I, a in self.data.iterrows()])

    @property
    def meanVarExp(self):
        return np.mean([a["prf"].meanVarExp for I, a in self.data.iterrows()])

    @property
    def beta(self):
        return np.hstack([a["prf"].beta for I, a in self.data.iterrows()])

    # ------------------------- SUBJECT VARS as MATRIX -----------------------#

    @property
    def sub_x0(self):
        if not hasattr(self, "_sub_x0"):
            self._sub_x0 = {}
            for s in self.subject:
                self._sub_x0[s] = np.array([a["prf"].x0 for I, a in self.data.iterrows() if a["subject"] == s])
        return self._sub_x0

    @property
    def sub_y0(self):
        if not hasattr(self, "_sub_y0"):
            self._sub_y0 = {}
            for s in self.subject:
                self._sub_y0[s] = np.array([a["prf"].x0 for I, a in self.data.iterrows() if a["subject"] == s])
        return self._sub_y0

    @property
    def sub_s0(self):
        if not hasattr(self, "_sub_s0"):
            self._sub_s0 = {}
            for s in self.subject:
                self._sub_s0[s] = np.array([a["prf"].x0 for I, a in self.data.iterrows() if a["subject"] == s])
        return self._sub_s0

    @property
    def sub_r0(self):
        if not hasattr(self, "_sub_r0"):
            self._sub_r0 = {}
            for s in self.subject:
                self._sub_r0[s] = np.array([a["prf"].x0 for I, a in self.data.iterrows() if a["subject"] == s])
        return self._sub_r0

    @property
    def sub_x(self):
        self._sub_x = {}
        for s in self.sub_x0:
            self._sub_x[s] = self.sub_x0[s][:, self.sub_mask[s]]
        return self._sub_x

    @property
    def sub_y(self):
        self._sub_y = {}
        for s in self.sub_x0:
            self._sub_y[s] = self.sub_x0[s][:, self.sub_mask[s]]
        return self._sub_y

    @property
    def sub_s(self):
        self._sub_s = {}
        for s in self.sub_x0:
            self._sub_s[s] = self.sub_x0[s][:, self.sub_mask[s]]
        return self._sub_s

    @property
    def sub_r(self):
        self._sub_r = {}
        for s in self.sub_x0:
            self._sub_r[s] = self.sub_x0[s][:, self.sub_mask[s]]
        return self._sub_r


    # --------------------------------- MASKS --------------------------------#
    @property
    def mask(self):
        return np.hstack([a["prf"].mask for I, a in self.data.iterrows()])

    @property
    def varExpMsk(self):
        return np.hstack([a["prf"].varExpMsk for I, a in self.data.iterrows()])

    @property
    def roiMsk(self):
        return np.hstack([a["prf"].roiMsk for I, a in self.data.iterrows()])

    @property
    def sub_mask(self):
        if not hasattr(self, "_sub_mask"):
            self._sub_mask = {}
            for s in self.subject:
                self._sub_mask[s] = np.all(
                    [a["prf"].mask for I, a in self.data.iterrows() if a["subject"] == s], 0
                )
        return self._sub_mask

    @property
    def doROIMsk(self):
        return self._doROIMsk

    @doROIMsk.setter
    def doROIMsk(self, value: bool):
        for I, a in self.data.iterrows():
            a["prf"].doROIMsk = value
        self._doROIMsk = value

    @property
    def doVarExpMsk(self):
        return self._doVarExpMsk

    @doVarExpMsk.setter
    def doVarExpMsk(self, value: bool):
        for I, a in self.data.iterrows():
            a["prf"].doVarExpMsk = value
        self._doVarExpMsk = value

    @property
    def doBetaMsk(self):
        return self._doBetaMsk

    @doBetaMsk.setter
    def doBetaMsk(self, value: bool):
        for I, a in self.data.iterrows():
            a["prf"].doBetaMsk = value
        self._doBetaMsk = value

    @property
    def doEccMsk(self):
        return self._doEccMsk

    @doEccMsk.setter
    def doEccMsk(self, value: bool):
        for I, a in self.data.iterrows():
            a["prf"].doEccMsk = value
        self._doEccMsk = value

    @property
    def doSigMsk(self):
        return self._doSigMsk

    @doEccMsk.setter
    def doSigMsk(self, value: bool):
        for I, a in self.data.iterrows():
            a["prf"].doSigMsk = value
        self._doSigMsk = value
