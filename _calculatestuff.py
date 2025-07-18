import matplotlib.colors as col
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


# ----------------------------------------------------------------------------#
def calc_kde_diff(self):
    """
    Calculate the 1D KDE (kernel density estimate) difference for pRF eccentricity, as in Hummer et al.

    Returns:
        tuple:
            - np.ndarray: Linspace for x-axis (eccentricity).
            - np.ndarray: KDE difference (y-axis).
    """

    kde = st.gaussian_kde(self.r)
    sstim = (
        "eightbars"
        if "bar" in self._analysis
        else "wedgesrings" if "wedge" in self._analysis else ""
    )
    kdeRefR = np.load(
        f"/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data/retcomp17/scripts/KDE_Turtle/meanKDE_{sstim}.npy"
    )

    self.kdeX = np.linspace(0, 7, 700)
    self.kdeR = kde.evaluate(self.kdeX)  # * len(self.r)
    self.kdeDiff = (kdeRefR - self.kdeR) / (kdeRefR + self.kdeR)

    return self.kdeX, self.kdeDiff


# ----------------------------------------------------------------------------#
def central_scot_border(self, scotVal=0.1):
    """
    Calculate the central scotoma border using the KDE difference, as in Hummer et al.

    Args:
        scotVal (float, optional): Threshold value to define scotoma border. Defaults to 0.1.

    Returns:
        float: Interpolated eccentricity value at the scotoma border.
    """

    if not hasattr(self, "kdeDiff"):
        self.calcKdeDiff()

    first_below = np.where(self.kdeDiff < scotVal)[0][0]
    last_above = first_below - 1

    # linear interpolate the actual border
    self.border = np.interp(
        0.1,
        [self.kdeDiff[first_below], self.kdeDiff[last_above]],
        [self.kdeX[first_below], self.kdeX[last_above]],
    )

    return self.border


# ----------------------------------------------------------------------------#
def calc_prf_profiles(self):
    """
    Calculate pRF profiles as in Urale et al. 2022.

    Returns:
        np.ndarray: The calculated pRF profiles.
    """

    from scipy.signal import detrend

    if not hasattr(self, "stimImages"):
        self.loadStim(buildTC=False)

    # demean and detrend the measured data
    TC = self.voxelTC - self.voxelTC.mean(1)[:, None]
    TC = detrend(TC, axis=1).T

    # models have sigma=0 -> just the stimulus pixel time courses, demean them
    model = self.stimImages - self.stimImages.mean(1)[:, None]
    model /= np.linalg.norm(model, axis=1)[:, None]

    # do "fitting", everything demeaned -> no pinv necessary
    b = model.dot(TC)

    # calc the VarExp (R2) for the max beta voxel
    bMax = b.max(0)
    bMaxLoc = np.argmax(b, 0)
    self.PRFprofilesR2 = 1 - np.sqrt(
        np.sum(np.square(np.subtract(TC, model[bMaxLoc, :].T * bMax)), 0)
    ) / np.sqrt(np.sum(np.square(TC), 0))

    # normalize the PRF profiles
    bMax[bMax == 0] = 1
    self.PRFprofiles = b / bMax

    return self.PRFprofiles


# ----------------------------------------------------------------------------#
def _calcKdeDiff2d(self, scotVal=0.1):
    """
    Calculate the 2D KDE difference for pRF centers.

    Args:
        scotVal (float, optional): Threshold value to define scotoma border. Defaults to 0.1.

    Returns:
        np.ndarray: 2D KDE difference array.
    """

    sstim = (
        "eightbars"
        if "bar" in self._analysis
        else "wedgesrings" if "wedge" in self._analysis else ""
    )
    kdeRef2d = np.load(
        f"/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data/retcomp17/scripts/KDE_Turtle/meanKDE2d_{sstim}.npy"
    )

    lspace = np.linspace(-self.maxEcc, self.maxEcc, 140)
    xx, yy = np.meshgrid(lspace, lspace)

    kde2d = (
        st.gaussian_kde(np.array([self.x, self.y]))
        .evaluate(np.vstack((xx.flatten(), yy.flatten())))
        .reshape(len(lspace), len(lspace))
    )
    self.kdeDiff2d = (kdeRef2d - kde2d) / (kdeRef2d + kde2d) - scotVal
    self.kdeDiff2d[~self._createmask(xx.shape)] = 0

    return self.kdeDiff2d


def plot_kdeDiff2d(self, title=None, scotVal=0.1):
    """
    Plot the 2D KDE difference map.

    Args:
        title (str, optional): Title for the plot. Defaults to None.
        scotVal (float, optional): Threshold value to define scotoma border. Defaults to 0.1.

    Returns:
        matplotlib.figure.Figure: Figure handle for the plot.
    """

    if not hasattr(self, "kdeDiff2d"):
        self._calcKdeDiff2d(scotVal)

    # define colormap
    cmap = plt.get_cmap("tab10")
    myCol = np.array([cmap(2), (1, 1, 1, 1), cmap(3)])
    myCmap = col.LinearSegmentedColormap.from_list("", myCol)

    fig = plt.figure(constrained_layout=True)
    ax = plt.gca()

    mm = np.abs(self.kdeDiff2d).max()

    im = ax.imshow(
        self.kdeDiff2d,
        cmap=myCmap,
        extent=(-self.maxEcc, self.maxEcc, -self.maxEcc, self.maxEcc),
        origin="lower",
        vmin=-mm,
        vmax=mm,
    )
    ax.set_xlim((-self.maxEcc, self.maxEcc))
    ax.set_ylim((-self.maxEcc, self.maxEcc))
    ax.set_aspect("equal", "box")
    fig.colorbar(im, location="right", ax=ax)

    maxEcc13 = np.round(self.maxEcc / 3, 1)
    maxEcc23 = np.round(self.maxEcc / 3 * 2, 1)
    si = np.sin(np.pi / 4) * self.maxEcc
    co = np.cos(np.pi / 4) * self.maxEcc

    # draw grid
    for e in [maxEcc13, maxEcc23, self.maxEcc]:
        ax.add_patch(plt.Circle((0, 0), e, color="grey", fill=False, linewidth=0.8))

    ax.plot((-self.maxEcc, self.maxEcc), (0, 0), color="grey", linewidth=0.8)
    ax.plot((0, 0), (-self.maxEcc, self.maxEcc), color="grey", linewidth=0.8)
    ax.plot((-co, co), (-si, si), color="grey", linewidth=0.8)
    ax.plot((-co, co), (si, -si), color="grey", linewidth=0.8)

    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.yaxis.set_ticks([0, maxEcc13, maxEcc23, self.maxEcc])

    if title is not None:
        ax.set_title(title)

    # fig.show()

    return fig
