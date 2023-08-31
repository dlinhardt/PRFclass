import matplotlib.colors as col
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


#----------------------------------------------------------------------------#
def calcKdeDiff(self):
    """
    calculate the KDE differentials as in Hummer et al.

    Returns:
        self.kdeX: lispace (x-axis)
        self.kdeDiff: the difference (y-axis)
    """

    kde = st.gaussian_kde(self.r)
    sstim = 'eightbars' if 'bar' in self._analysis else 'wedgesrings' if 'wedge' in self._analysis else ''
    kdeRefR = np.load(f'/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data/retcomp17/scripts/KDE_Turtle/meanKDE_{sstim}.npy')

    self.kdeX    = np.linspace(0, 7, 700)
    self.kdeR    = kde.evaluate(self.kdeX)  # * len(self.r)
    self.kdeDiff = (kdeRefR - self.kdeR) / (kdeRefR + self.kdeR)

    return self.kdeX, self.kdeDiff


#----------------------------------------------------------------------------#
def centralScotBorder(self, scotVal=.1):
    """
    calculate the central scotoma border as in Hummer et al.

    Args:
        scotVal (float, optional): Threshold value to define Scotoma border. Defaults to .1.

    Returns:
        self.border: Scotoma border-line
    """

    if not hasattr(self, 'kdeDiff'):
        self.calcKdeDiff()

    firstBelow = np.where(self.kdeDiff < scotVal)[0][0]
    lastAbove  = firstBelow - 1

    # linear interpolate the actual border
    self.border = np.interp(.1, [self.kdeDiff[firstBelow], self.kdeDiff[lastAbove]], [self.kdeX[firstBelow], self.kdeX[lastAbove]])

    return self.border


#----------------------------------------------------------------------------#
def calcPRFprofiles(self):
    """
    calculate PRF profiles as in Urale et al. 2022

    Returns:
        self.PRFprofiles: the calculated profile
    """

    from scipy.signal import detrend
    if not hasattr(self, 'stimImages'):
        self.loadStim(buildTC=False)

    # demean and detrend the measured data
    TC = self.voxelTC - self.voxelTC.mean(1)[:, None]
    TC = detrend(TC, axis=1).T

    # models have sigma=0 -> just the stimulus pixel time courses, demean them
    model  = (self.stimImages - self.stimImages.mean(1)[:, None])
    model /= np.linalg.norm(model, axis=1)[:, None]

    # do "fitting", everything demeaned -> no pinv necessary
    b = model.dot(TC)

    # calc the VarExp (R2) for the max beta voxel
    bMax    = b.max(0)
    bMaxLoc = np.argmax(b, 0)
    self.PRFprofilesR2 = 1 - np.sqrt(np.sum(np.square(np.subtract(TC, model[bMaxLoc, :].T * bMax)), 0)) / np.sqrt(np.sum(np.square(TC), 0))

    # normalize the PRF profiles
    bMax[bMax == 0] = 1
    self.PRFprofiles = b / bMax

    return self.PRFprofiles

#----------------------------------------------------------------------------#
def _calcKdeDiff2d(self, scotVal=.1):
    """
    calculate the 2D KDE differentials

    Args:
        scotVal (float, optional): Threshold value to define Scotoma border. Defaults to .1.

    Returns:
        self.kdeDiff2d: returns the difference of 2D KDE
    """

    sstim = 'eightbars' if 'bar' in self._analysis else 'wedgesrings' if 'wedge' in self._analysis else ''
    kdeRef2d = np.load(f'/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data/retcomp17/scripts/KDE_Turtle/meanKDE2d_{sstim}.npy')

    lspace = np.linspace(-self.maxEcc, self.maxEcc, 140)
    xx, yy = np.meshgrid(lspace, lspace)

    kde2d = st.gaussian_kde(np.array([self.x, self.y])).evaluate(np.vstack((xx.flatten(), yy.flatten()))).reshape(len(lspace), len(lspace))
    self.kdeDiff2d = (kdeRef2d - kde2d) / (kdeRef2d + kde2d) - scotVal
    self.kdeDiff2d[~self._createmask(xx.shape)] = 0

    return self.kdeDiff2d


def plot_kdeDiff2d(self, title=None, scotVal=.1):
    """
    plots the 2D KDE difference

    Args:
        title (str, optional): Title of the plot. Defaults to None.
        scotVal (float, optional): Threshold value to define Scotoma border. Defaults to .1.

    Returns:
        fig: figure handle
    """

    if not hasattr(self, 'kdeDiff2d'):
        self._calcKdeDiff2d(scotVal)

    # define colormap
    cmap = plt.get_cmap('tab10')
    myCol = np.array([cmap(2), (1, 1, 1, 1), cmap(3)])
    myCmap = col.LinearSegmentedColormap.from_list("", myCol)

    fig = plt.figure(constrained_layout=True)
    ax  = plt.gca()

    mm = np.abs(self.kdeDiff2d).max()

    im = ax.imshow(self.kdeDiff2d, cmap=myCmap, extent=(-self.maxEcc, self.maxEcc, -self.maxEcc, self.maxEcc),
                   origin='lower', vmin=-mm, vmax=mm)
    ax.set_xlim((-self.maxEcc, self.maxEcc))
    ax.set_ylim((-self.maxEcc, self.maxEcc))
    ax.set_aspect('equal', 'box')
    fig.colorbar(im, location='right', ax=ax)

    maxEcc13 = np.round(self.maxEcc / 3, 1)
    maxEcc23 = np.round(self.maxEcc / 3 * 2, 1)
    si = np.sin(np.pi / 4) * self.maxEcc
    co = np.cos(np.pi / 4) * self.maxEcc

    # draw grid
    for e in [maxEcc13, maxEcc23, self.maxEcc]:
        ax.add_patch(plt.Circle((0, 0), e, color='grey', fill=False, linewidth=.8))

    ax.plot((-self.maxEcc, self.maxEcc), (0, 0), color='grey', linewidth=.8)
    ax.plot((0, 0), (-self.maxEcc, self.maxEcc), color='grey', linewidth=.8)
    ax.plot((-co, co), (-si, si), color='grey', linewidth=.8)
    ax.plot((-co, co), (si, -si), color='grey', linewidth=.8)

    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.yaxis.set_ticks([0, maxEcc13, maxEcc23, self.maxEcc])

    if title is not None:
        ax.set_title(title)

    # fig.show()

    return fig
