from glob import glob
from os import path

import nibabel as nib
import numpy as np
from scipy.io import loadmat
import warnings


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
