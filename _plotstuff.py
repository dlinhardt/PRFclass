try:
    from mayavi import mlab
except (ModuleNotFoundError, ImportError):
    print("Package mayavi not installed!")
try:
    from surfer import Brain
except (ModuleNotFoundError, ImportError):
    print("Package pysurfer not installed!")
    print("Install it or don't run plot_toSurface()")

import json
from copy import deepcopy
from glob import glob
from os import makedirs, path, remove
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import nibabel as nib
import numpy as np
from matplotlib import colors, patches
from matplotlib.widgets import Button, Slider
from nilearn.surface import vol_to_surf
from PIL import Image

try:
    import neuropythy as ny
except:
    print("neuropythy not installed, samsrf data not available")


def _resolve_fs_subject_dir(
    fs_dir: Path | str, sub: str, sess: str | None = None
) -> Path:
    """
    Resolve FreeSurfer subject directory, checking for new layout if old doesn't exist.

    Parameters
    ----------
    fs_dir : Path | str
        Base FreeSurfer directory
    sub : str
        Subject ID
    sess : str | None, optional
        Session ID for new layout fallback

    Returns
    -------
    Path
        Resolved subject directory path
    """
    fs_dir = Path(fs_dir)
    old_path = fs_dir / f"{sub}"

    if old_path.exists():
        return f"{sub}", str(old_path)

    # Try new layout with session if available
    if sess:
        new_path = fs_dir / f"{sub}_{sess}"
    if new_path.exists():
        return f"{sub}_{sess}", str(new_path)

    # Try to find any session directory for this subject
    pattern = fs_dir / f"{sub}_ses-*"
    matches = sorted(pattern.parent.glob(pattern.name))
    if matches:
        return f"{sub}_{sess}", str(matches[0])

    # Return old path as default (will fail later if doesn't exist)
    return f"{sub}", str(old_path)


class label:
    """
    Mini class that defines a label, needed to plot borders
    """

    def __init__(self, ar, at, he, allAreaFiles, left_hemi):
        self.name = ar
        self.hemi = he[0].lower()[0] + "h"

        if not "sourcedata/freesurfer" in allAreaFiles[0]:
            vJsonName = [
                j
                for j in allAreaFiles
                if f"hemi-{he[0].upper()}_" in path.basename(j)
                and f"desc-{ar}-" in path.basename(j)
                and at in path.basename(j)
            ][0]
            with open(vJsonName, "r") as fl:
                maskinfo = json.load(fl)

            if "roiIndOrig" in maskinfo.keys():
                self.vertices = np.array(maskinfo["roiIndOrig"])
            else:
                key = [a for a in maskinfo.keys() if a.startswith("roiIndFs")][0]
                self.vertices = np.array(maskinfo[key])
        else:
            f = [j for j in allAreaFiles if f"/{he[0].lower()}h."][0]
            maskinfo = nib.load(f).get_fdata().squeeze()
            mdl = ny.vision.retinotopy_model("benson17", "lh")
            areaLabels = dict(mdl.area_id_to_name)
            areaLabels = {areaLabels[k]: k for k in areaLabels}
            labels = areaLabels[ar]
            self.vertices = np.where(maskinfo == labels)
            if he.upper() == "R":
                self.vertices -= left_hemi


# ----------------------------------------------------------------------------#
def _createmask(self, shape, otherRratio=None):
    """
    Create a circular boolean mask of a given size.

    Args:
        shape (tuple): Shape of the mask (height, width).
        otherRratio (float, optional): Ratio for the mask radius. If None, uses full size.

    Returns:
        np.ndarray: Boolean mask array of the specified shape.
    """

    x0, y0 = shape[0] // 2, shape[1] // 2
    n = shape[0]
    if otherRratio is not None:
        r = shape[0] / 2 * otherRratio + 1
    else:
        r = shape[0] // 2 + 1

    y, x = np.ogrid[-x0 : n - x0, -y0 : n - y0]
    return x * x + y * y <= r * r


def _draw_grid(ax, maxEcc):
    """
    Draw a polar grid on the given matplotlib axis for coverage map visualization.

    Args:
        ax (matplotlib.axes.Axes): The axis to draw the grid on.
        maxEcc (float): Maximum eccentricity for the grid.
    """
    # draw grid
    maxEcc13 = maxEcc / 3
    maxEcc23 = maxEcc / 3 * 2
    si = np.sin(np.pi / 4) * maxEcc
    co = np.cos(np.pi / 4) * maxEcc

    for e in [maxEcc13, maxEcc23, maxEcc]:
        ax.add_patch(plt.Circle((0, 0), e, color="grey", fill=False, linewidth=0.8))

    ax.plot((-1 * maxEcc, maxEcc), (0, 0), color="grey", linewidth=0.8)
    ax.plot((0, 0), (-1 * maxEcc, maxEcc), color="grey", linewidth=0.8)
    ax.plot((-co, co), (-si, si), color="grey", linewidth=0.8)
    ax.plot((-co, co), (si, -si), color="grey", linewidth=0.8)

    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.yaxis.set_ticks(
        [np.round(maxEcc13, 1), np.round(maxEcc23, 1), np.round(maxEcc, 1)]
    )

    ax.tick_params(axis="y", direction="in", pad=-ax.get_window_extent().height * 0.48)

    plt.setp(ax.yaxis.get_majorticklabels(), va="bottom")

    ax.tick_params(axis="both", which="both", length=0)

    ax.set_frame_on(False)

    ax.set_box_aspect(1)


def _calcCovMap(self, maxEcc, method="max", force=False, background="black"):
    """
    Calculate the pRF coverage map and save it as a .npy file.

    Args:
        maxEcc (float): Maximum eccentricity for the coverage map.
        method (str, optional): Method to calculate the coverage map ('max', 'mean', 'sumClip'). Defaults to 'max'.
        force (bool, optional): If True, overwrite existing file. Defaults to False.
        background (str, optional): Background color for the map ('black' or 'white'). Defaults to 'black'.

    Returns:
        np.ndarray: The calculated coverage map.
    """

    # create the filename
    if self._dataFrom == "mrVista":
        VEstr = (
            f"_VarExp-{int(self._isVarExpMasked*100)}"
            if self._isVarExpMasked and self.doVarExpMsk
            else ""
        )
        Bstr = (
            f"_betaThresh-{self._isBetaMasked}"
            if self._isBetaMasked and self.doBetaMsk
            else ""
        )
        Sstr = "_MPspace" if self._orientation == "MP" else ""
        Estr = (
            f"_maxEcc-{self._isEccMasked}"
            if self._isEccMasked and self.doEccMsk
            else ""
        )
        Mstr = (
            f"-manualMask{self._isManualMasked}"
            if self._isManualMasked and self.doManualMsk
            else ""
        )
        methodStr = f"_{method}"

        savePathB = path.join(
            self._baseP,
            self._study,
            "plots",
            "cover",
            "data",
            self.subject,
            self.session,
        )
        savePathF = f"{self.subject}_{self.session}_{self._analysis}{VEstr}{Estr}{Bstr}{Mstr}{Sstr}{methodStr}.npy"

    elif self._dataFrom in ["docker", "samsrf", "hdf5"]:
        VEstr = (
            f"-VarExp{int(self._isVarExpMasked*100)}"
            if self._isVarExpMasked and self.doVarExpMsk
            else ""
        )
        Bstr = (
            f"-betaThresh{self._isBetaMasked}"
            if self._isBetaMasked and self.doBetaMsk
            else ""
        )
        Sistr = (
            f"-sigmaThresh{self._isSigMasked}"
            if self._isSigMasked and self.doSigMsk
            else ""
        )
        Mstr = (
            f"-manualMask{self._isManualMasked}"
            if self._isManualMasked and self.doManualMsk
            else ""
        )
        Sstr = "-MPspace" if self._orientation == "MP" else ""
        Estr = (
            f"_maxEcc{self._isEccMasked}" if self._isEccMasked and self.doEccMsk else ""
        )
        hemiStr = f"_hemi-{self._hemis.upper()}" if self._hemis != "" else ""
        methodStr = f"-{method}"
        areaStr = "multipleAreas" if len(self._area) > 10 else "".join(self._area)

        savePathB = path.join(
            self._derivatives_path,
            "prfresult",
            self._prfanalyze_method,
            self._prfanaAn,
            "covMapData",
            self.subject,
            self.session,
        )

        savePathF = f"{self.subject}_{self.session}_{self._task}_{self._run}{hemiStr}_desc-{areaStr}{VEstr}{Estr}{Bstr}{Mstr}{Sistr}{Sstr}{methodStr}_covmapData.npy"

    savePath = path.join(savePathB, savePathF)

    def gaussian(x, mu, sig):
        return np.exp(-np.power((x - mu) / sig, 2.0) / 2)

    if path.isfile(savePath) and not force:
        self.covMap = np.load(savePath, allow_pickle=True)
        return self.covMap
    else:
        if not path.isdir(savePathB):
            makedirs(savePathB)

        xx = np.linspace(-1 * maxEcc, maxEcc, max(int(maxEcc * 30), 300) + 1)

        covMap = np.zeros((len(xx), len(xx)))

        jj = 0
        for i in range(len(self.x)):
            kern1dx = gaussian(xx, self.x[i], self.s[i])
            kern1dy = gaussian(xx, self.y[i], self.s[i])
            kern2d = np.outer(kern1dx, kern1dy)

            if np.max(kern2d) > 0:
                jj += 1

                if method == "max":
                    covMap = np.max((covMap, kern2d), 0)
                elif method == "mean" or method == "sumClip":
                    covMap = np.sum((covMap, kern2d), 0)

        if method == "mean":
            covMap /= jj

        msk = self._createmask(covMap.shape)
        if background == "black":
            covMap[~msk] = 0
        else:
            covMap[~msk] = np.nan

        self.covMap = covMap.T

        np.save(savePath, self.covMap, allow_pickle=True)

        return self.covMap


def _get_covMap_savePath(self, method="max", cmapMin=0, output_format="svg"):
    if self._dataFrom == "mrVista":
        VEstr = (
            f"_VarExp-{int(self._isVarExpMasked*100)}"
            if self._isVarExpMasked and self.doVarExpMsk
            else ""
        )
        Bstr = (
            f"_betaThresh-{self._isBetaMasked}"
            if self._isBetaMasked and self.doBetaMsk
            else ""
        )
        Sstr = "_MPspace" if self._orientation == "MP" else ""
        Estr = (
            f"_maxEcc-{self._isEccMasked}"
            if self._isEccMasked and self.doEccMsk
            else ""
        )
        Mstr = (
            f"-manualMask{self._isManualMasked}"
            if self._isManualMasked and self.doManualMsk
            else ""
        )
        CBstr = f"_colBar-{cmapMin}".replace(".", "") if cmapMin != 0 else ""
        methodStr = f"_{method}"

        savePathB = path.join(
            self._baseP,
            self._study,
            "plots",
            "cover",
            self.subject,
            self.session,
        )
        savePathF = f"{self.subject}_{self.session}_{self._analysis}{CBstr}{VEstr}{Estr}{Bstr}{Mstr}{Sstr}{methodStr}.{output_format}"

    elif self._dataFrom in ["docker", "samsrf", "hdf5"]:
        VEstr = (
            f"-VarExp{int(self._isVarExpMasked*100)}"
            if self._isVarExpMasked and self.doVarExpMsk
            else ""
        )
        Bstr = (
            f"-betaThresh{self._isBetaMasked}"
            if self._isBetaMasked and self.doBetaMsk
            else ""
        )
        Sistr = (
            f"-sigmaThresh{self._isSigMasked}"
            if self._isSigMasked and self.doSigMsk
            else ""
        )
        Mstr = (
            f"-manualMask{self._isManualMasked}"
            if self._isManualMasked and self.doManualMsk
            else ""
        )
        Sstr = "-MPspace" if self._orientation == "MP" else ""
        Estr = (
            f"_maxEcc{self._isEccMasked}" if self._isEccMasked and self.doEccMsk else ""
        )
        CBstr = f"-colBar{cmapMin}".replace(".", "") if cmapMin != 0 else ""
        hemiStr = f"_hemi-{self._hemis.upper()}" if self._hemis != "" else ""
        methodStr = f"-{method}"
        areaStr = "multipleAreas" if len(self._area) > 10 else "".join(self._area)

        savePathB = path.join(
            self._derivatives_path,
            "prfresult",
            self._prfanalyze_method,
            self._prfanaAn,
            "covMap",
            self.subject,
            self.session,
        )
        savePathF = f"{self.subject}_{self.session}_{self._task}_{self._run}{hemiStr}_desc-{areaStr}{VEstr}{Estr}{Bstr}{Mstr}{Sistr}{Sstr}{methodStr}{CBstr}_covmap.{output_format}"

    # create folder if not exist
    if not path.isdir(savePathB):
        makedirs(savePathB)

    return path.join(savePathB, savePathF)


# ----------------------------------------------------------------------------#
def plot_covMap(
    self,
    method="max",
    cmapMin=0,
    cmapMax=None,
    title=None,
    show=True,
    save=False,
    force=False,
    maxEcc=None,
    do_scatter=True,
    output_format="svg",
    plot_colorbar=True,
    background="white",
):
    """
    Plot the pRF coverage map and optionally save it.

    Args:
        method (str, optional): Method to calculate the coverage map ('max', 'mean', 'sumClip'). Defaults to 'max'.
        cmapMin (float, optional): Minimum value for the colormap (0-1). Defaults to 0.
        cmapMax (float, optional): Maximum value for the colormap. If None, set automatically. Defaults to None.
        title (str, optional): Title for the plot. Defaults to None.
        show (bool, optional): Whether to display the plot interactively. Defaults to True.
        save (bool, optional): Whether to save the plot to disk. Defaults to False.
        force (bool, optional): If True, overwrite existing files. Defaults to False.
        maxEcc (float, optional): Maximum eccentricity for the plot axes. If None, uses self.maxEcc. Defaults to None.
        do_scatter (bool, optional): Whether to overlay scatter plot of pRF centers. Defaults to True.
        output_format (str, optional): File format for saving (e.g., 'svg', 'pdf'). Defaults to 'svg'.
        plot_colorbar (bool, optional): Whether to display a colorbar. Defaults to True.
        background (str, optional): Background color for the plot. Defaults to 'white'.

    Returns:
        matplotlib.figure.Figure or str: Figure handle if not saving, otherwise the path to the saved file.

    Raises:
        ValueError: If method is not recognized or colormap limits are invalid.
    """

    if not show:
        plt.ioff()

    # do maxEcc
    maxEcc = maxEcc if maxEcc else self.maxEcc

    # create the filename
    savePath = self._get_covMap_savePath(method, cmapMin, output_format)

    # warning if the chosen method is not possible
    if not path.isfile(savePath) or show or force:
        methods = ["max", "mean", "sumClip"]
        if method not in methods:
            raise ValueError(
                f'Chosen method "{method}" is not available. Choose from {methods}.'
            )

        # calculate the coverage map
        self._calcCovMap(maxEcc, method, force=force, background=background)

        # set method-specific stuff
        if method == "max":
            vmax = 1
        elif method == "mean":
            vmax = self.covMap.max()
        elif method == "sumClip":
            vmax = 1

        # adapt the colorbar for the passed cmapMax and cmapMin
        if cmapMin > 1 or cmapMin < 0:
            raise ValueError("Choose a cmap min between 0 and 1.")
        else:
            vmin = cmapMin

        if cmapMax is not None:
            if cmapMax < vmin:
                raise ValueError("Choose a cmapMax bigger than cmapMin.")
            vmax = cmapMax

        fig = plt.figure(constrained_layout=True, facecolor="white")
        ax = plt.gca()

        im = ax.imshow(
            self.covMap,
            cmap="hot",
            extent=(-1 * maxEcc, maxEcc, -1 * maxEcc, maxEcc),
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        if do_scatter:
            ax.scatter(
                self.x[self.r < maxEcc], self.y[self.r < maxEcc], s=0.3, c="grey"
            )
        ax.set_xlim((-1 * maxEcc, maxEcc))
        ax.set_ylim((-1 * maxEcc, maxEcc))
        ax.set_aspect("equal", "box")
        if plot_colorbar:
            fig.colorbar(im, location="right", ax=ax)

        # draw grid
        _draw_grid(ax, maxEcc)

        if title is not None:
            ax.set_title(title, pad=16, fontsize=16)

    if save and not path.isfile(savePath):
        fig.savefig(savePath, bbox_inches="tight")
        print(f"new Coverage Map saved to {savePath}")
        # plt.close('all')

    if not show:
        # plt.ion()
        pass
    if save:
        return savePath
    else:
        return fig


# ----------------------------------------------------------------------------#
def _get_surfaceSavePath(self, param, hemi, surface="cortex", plain=False):
    """
    Define the path and filename to save the cortex plot.

    Args:
        param (str): The parameter to plot.
        hemi (str): The hemisphere ('L', 'R', or 'both').
        surface (str, optional): Surface type (e.g., 'cortex', 'inflated'). Defaults to 'cortex'.
        plain (bool, optional): If True, use a simplified filename. Defaults to False.

    Returns:
        tuple: (savePathB, savePathF) where savePathB is the directory and savePathF is the filename (without extension).
    """

    VEstr = (
        f"-VarExp{int(self._isVarExpMasked*100)}"
        if self._isVarExpMasked and self.doVarExpMsk
        else ""
    )
    Bstr = (
        f"-betaThresh{self._isBetaMasked}"
        if self._isBetaMasked and self.doBetaMsk
        else ""
    )
    Mstr = (
        f"-manualMask{self._isManualMasked}"
        if self._isManualMasked and self.doManualMsk
        else ""
    )
    if isinstance(param, str):
        Pstr = f"-{param}"
    else:
        Pstr = "-manualParam"

    if self._dataFrom == "mrVista":
        savePathB = path.join(
            self._baseP,
            self._study,
            "plots",
            "cortex",
            self.subject,
            self.session,
        )
    else:
        savePathB = path.join(
            self._derivatives_path,
            "prfresult",
            self._prfanalyze_method,
            self._prfanaAn,
            "cortex",
            self.subject,
            self.session,
        )
    ending = surface
    areaStr = "multipleAreas" if len(self._area) > 10 else "".join(self._area)

    if not plain:
        savePathF = f"{self.subject}_{self.session}_{self._task}_{self._run}_hemi-{hemi[0].upper()}_desc-{areaStr}{VEstr}{Bstr}{Mstr}{Pstr}_{ending}"
    else:
        savePathF = f"{self.subject}_{self.session}_{self._task}_{self._run}_hemi-{hemi[0].upper()}_desc{Pstr}_{ending}"

    if not path.isdir(savePathB):
        makedirs(savePathB)

    return savePathB, savePathF


def _make_gif(self, frameFolder, outFilename):
    """
    Create a GIF from a sequence of PNG frames in a folder.

    Args:
        frameFolder (str): Folder containing the frames and output GIF.
        outFilename (str): Output GIF filename (with extension).
    """

    # Read the images
    frames = [Image.open(image) for image in sorted(glob(f"{frameFolder}/frame*.png"))]
    # Create the gif
    frame_one = frames[0]
    frame_one.save(
        path.join(frameFolder, outFilename),
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=500,
        loop=0,
    )
    print(f"new Cortex Map saved to {outFilename}")
    # Delete the png-s
    [remove(image) for image in glob(f"{frameFolder}/frame*.png")]


def _prepare_mrVista_for_surface_plot(self):
    pass


# ----------------------------------------------------------------------------#
def plot_toSurface(
    self,
    param="ecc",
    hemi="left",
    save=False,
    forceNewPosition=False,
    force=False,
    surface="inflated",
    showBordersAtlas=None,
    showBordersArea=None,
    interactive=True,
    create_gif=False,
    headless=False,
    maxEcc=None,
    plot_colorbar=True,
    background="black",
    output_format="pdf",
    pmax=None,
    pmin=None,
):
    """
    Plot a parameter (eccentricity, polar angle, sigma, or variance explained) to the cortical surface.

    Args:
        param (str or np.ndarray, optional): Parameter to plot ('ecc', 'pol', 'sig', 'var') or array. Defaults to 'ecc'.
        hemi (str, optional): Hemisphere to plot ('L', 'R', or 'both'). Defaults to 'left'.
        fmriprepAna (str, optional): fMRIPrep analysis number. Defaults to '01'.
        save (bool, optional): Whether to save the screenshot. Defaults to False.
        forceNewPosition (bool, optional): Force manual positioning. Defaults to False.
        force (bool, optional): Overwrite existing files. Defaults to False.
        surface (str, optional): Surface type (e.g., 'inflated', 'sphere'). Defaults to 'inflated'.
        showBordersAtlas (list or str, optional): Atlases to show area borders from. Defaults to None.
        showBordersArea (list or str, optional): Areas to show borders for. Defaults to None.
        interactive (bool, optional): Enable interactive mode. Defaults to True.
        create_gif (bool, optional): Create a GIF instead of a static image. Defaults to False.
        headless (bool, optional): Suppress pop-ups. Defaults to False.
        maxEcc (float, optional): Maximum eccentricity for plotting. Defaults to None.
        plot_colorbar (bool, optional): Show colorbar. Defaults to True.
        background (str, optional): Background color. Defaults to 'black'.
        output_format (str, optional): Output file format. Defaults to 'pdf'.
        pmax (float, optional): Maximum value for color scaling. Defaults to None.
        pmin (float, optional): Minimum value for color scaling. Defaults to None.

    Raises:
        RuntimeError: If plotting is not supported for the data type or space.
        ValueError: If the parameter is not recognized.
    """

    maxEcc = maxEcc if maxEcc else self.maxEcc

    if headless:
        mlab.options.offscreen = True
        # mlab.init_notebook('x3d', 800, 800)
    else:
        mlab.options.offscreen = False

    if surface == "sphere":
        create_gif = False

    # turn of other functionality when creating gif
    if create_gif:
        manualPosition = False
        save = False
        interactive = True
    else:
        manualPosition = True if not surface == "sphere" else False

    if self._dataFrom in ["docker", "samsrf", "hdf5"]:
        if self.analysisSpace == "volume":
            raise RuntimeError(
                "plot_toSurface is not yet supported for volumetric data!"
            )
            need_to_convert = True

        fsP = path.join(
            self._derivatives_path,
            "fmriprep",
            f"analysis-{self.fmriprep_analysis}",
            "sourcedata",
            "freesurfer",
        )

        # define the subject as fsaverage
        if self.analysisSpace == "fsaverage":
            plot_subject = "fsaverage"
        else:
            plot_subject = self.subject
        need_to_convert = False
        positioning_subject = plot_subject

    elif self._dataFrom == "mrVista":
        fsP = path.join(
            self._baseP,
            self._study,
            "subjects",
            self.subject,
            self.session,
            "segmentation",
        )
        plot_subject = "freesurfer"
        need_to_convert = True
        positioning_subject = self.subject

    # if self._dataFrom == "mrVista":
    #     raise RuntimeError("plot_toSurface is not supported for non-docker data!")
    # elif self._dataFrom in ["docker", "samsrf", "hdf5"]:

    if hemi == "both":
        hemis = ["L", "R"]
    else:
        hemis = [hemi]

    for hemi in hemis:
        hemi_full = "left" if hemi[0].upper() == "L" else "right"

        if save:
            if self._dataFrom == "mrVista":
                plot_out_p = path.join(
                    self._baseP,
                    self._study,
                    "plots",
                    "volumeResults",
                    self.subject,
                    self.session,
                    self._analysis,
                )
                n = f"{self.subject}_{self.session}_{self._analysis}_hemi-{hemi_full[0].upper()}_desc-{param}_{surface}"
            else:
                plot_out_p, n = self._get_surfaceSavePath(param, hemi, surface)
            if (
                path.isfile(path.join(plot_out_p, n + f".{output_format}"))
                and not force
            ):
                return

            makedirs(plot_out_p, exist_ok=True)
        if create_gif:
            plot_out_p, n = self._get_surfaceSavePath(param, hemi)
            if path.isfile(path.join(plot_out_p, n + ".gif")) and not force:
                return

            makedirs(plot_out_p, exist_ok=True)
        (
            a,
            p,
        ) = _resolve_fs_subject_dir(fsP, plot_subject, self.session)

        pialP = path.join(
            p,
            "surf",
            f"{hemi[0].lower()}h.pial",
        )
        if not path.isfile(pialP):
            pialP = path.join(
                p,
                "surf",
                f"{hemi[0].lower()}h.pial.T1",
            )
        if not path.isfile(pialP):
            raise RuntimeError(f"Pial surface not found, {pialP}!")

        pial = nib.freesurfer.read_geometry(pialP)

        nVertices = len(pial[0])

        # create mask dependent on used hemisphere
        if need_to_convert:
            # now we need to convert the volume data to the surface
            # first get the 3D versions of the data
            data_in_3d = self.save_results(
                params=[param + "0", "mask"], save=False, force=True
            )
            fs_pial = path.join(
                p,
                "surf",
                f"{hemi[0].lower()}h.pial",
            )
            fs_white = path.join(
                p,
                "surf",
                f"{hemi[0].lower()}h.white",
            )
        else:
            if not hasattr(self, "_roiWhichHemi"):
                raise RuntimeError(
                    "Please mask for visual area with instance.maskROI()!"
                )

            hemiM = self._roiWhichHemi == hemi[0].upper()

            roiIndOrigHemi = self._roiIndOrig[hemiM]
            roiIndBoldHemi = self._roiIndBold[hemiM]

            # write data array to plot
            plotData = np.ones(nVertices) * np.nan

        # depending on used parameter set the plot data, colormap und ranges
        if isinstance(param, str):
            if param == "ecc":
                if need_to_convert:
                    plotData = vol_to_surf(
                        data_in_3d["r0"],
                        surf_mesh=fs_pial,
                        inner_mesh=fs_white,
                        mask_img=data_in_3d["mask"] if "mask" in data_in_3d else None,
                    )
                else:
                    plotData[roiIndOrigHemi] = self.r0[roiIndBoldHemi]
                cmap = "rainbow_r"
                datMin, datMax = 0, maxEcc

            elif param == "pol":
                if need_to_convert:
                    plotData = vol_to_surf(
                        data_in_3d["phi0"],
                        surf_mesh=fs_pial,
                        inner_mesh=fs_white,
                        mask_img=data_in_3d["mask"] if "mask" in data_in_3d else None,
                    )
                else:
                    plotData[roiIndOrigHemi] = self.phi0[roiIndBoldHemi]
                cmap = colors.LinearSegmentedColormap.from_list(
                    "", ["yellow", (0, 0, 1), (0, 1, 0), (1, 0, 0), "yellow"]
                )
                datMin, datMax = 0, 2 * np.pi

            elif param == "sig":
                if need_to_convert:
                    plotData = vol_to_surf(
                        data_in_3d["s0"],
                        surf_mesh=fs_pial,
                        inner_mesh=fs_white,
                        mask_img=data_in_3d["mask"] if "mask" in data_in_3d else None,
                    )
                else:
                    plotData[roiIndOrigHemi] = self.s0[roiIndBoldHemi]
                cmap = "rainbow_r"
                datMin, datMax = 0, 4

            elif param == "var":
                if need_to_convert:
                    plotData = vol_to_surf(
                        data_in_3d["varexp0"],
                        surf_mesh=fs_pial,
                        inner_mesh=fs_white,
                        mask_img=data_in_3d["mask"] if "mask" in data_in_3d else None,
                    )
                else:
                    plotData[roiIndOrigHemi] = self.varexp0[roiIndBoldHemi]
                cmap = "hot"
                datMin, datMax = 0, 1

        elif isinstance(param, (np.ndarray, np.generic)):
            plotData[roiIndOrigHemi] = param[roiIndBoldHemi]
            cmap = "hot"
            datMin, datMax = (
                param[roiIndBoldHemi].min(),
                param[roiIndBoldHemi].max(),
            )

        else:
            raise ValueError(
                'Parameter string must be in ["ecc", "pol", "sig", "var"] or a np.array of the same size with values!'
            )

        # manually set plot max and min
        if pmax is not None:
            datMax = pmax
        if pmin is not None:
            datMin = pmin

        # set everything outside mask (ROI, VarExp, ...) to nan
        if not self._dataFrom == "mrVista":
            plotData = deepcopy(plotData)
            if isinstance(param, str):
                if not param == "var":
                    plotData[roiIndOrigHemi[~self.mask[roiIndBoldHemi]]] = np.nan
            else:
                plotData[roiIndOrigHemi[~self.mask[roiIndBoldHemi]]] = np.nan

        # plot the brain
        brain = Brain(
            a,
            f"{hemi[0].lower()}h",
            surface,
            subjects_dir=fsP,
            background=background,
        )
        # plot the data
        brain.add_data(
            np.float16(plotData),
            colormap=cmap,
            min=datMin,
            max=datMax,
            smoothing_steps="nearest",
            remove_existing=True,
            colorbar=plot_colorbar,
        )

        # set nan to transparent
        brain.data["surfaces"][0].module_manager.scalar_lut_manager.lut.nan_color = (
            0,
            0,
            0,
            0,
        )
        brain.data["surfaces"][0].update_pipeline()

        # print borders (freesurfer)
        if showBordersArea is not None and showBordersAtlas is not None:
            if showBordersAtlas == "all":
                ats = self._atlas
            elif isinstance(showBordersAtlas, list):
                ats = showBordersAtlas
            elif isinstance(showBordersAtlas, str):
                ats = [showBordersAtlas]
            elif showBordersAtlas is True:
                ats = ["benson"]

            if isinstance(showBordersArea, list):
                ars = showBordersArea
            else:
                ars = self._area

            for at in ats:
                for ar in ars:
                    try:
                        if self._dataFrom == "samsrf":
                            aaf = self._area_files_p
                            left_hemi = self._vertices_left_hemi
                        else:
                            aaf = self._allAreaFiles
                            left_hemi = None
                        brain.add_label(
                            label(ar, at, hemi, aaf, left_hemi),
                            borders=True,
                            color="black",
                            alpha=0.7,
                        )
                    except:
                        pass

        # save the positioning for left and right once per subject
        if manualPosition:
            if self._dataFrom == "mrVista":
                posSavePath = path.join(
                    self._baseP,
                    self._study,
                    "plots",
                    "volumeResults",
                    "positioning",
                    positioning_subject,
                )

            else:
                posSavePath = path.join(
                    self._derivatives_path,
                    "prfresult",
                    "positioning",
                    positioning_subject,
                )
            if not path.isdir(posSavePath):
                makedirs(posSavePath)

            areaStr = "multipleAreas" if len(self._area) > 10 else "".join(self._area)

            posSaveFile = f"{positioning_subject}_hemi-{hemi[0].upper()}_desc-{areaStr}_cortex.npy"
            posPath = path.join(posSavePath, posSaveFile)

            if not path.isfile(posPath) or forceNewPosition:
                if hemi[0].upper() == "L":
                    brain.show_view(
                        {
                            "azimuth": -57.5,
                            "elevation": 106,
                            "distance": 300,
                            "focalpoint": np.array([-43, -23, -8]),
                        },
                        roll=-130,
                    )
                elif hemi[0].upper() == "R":
                    brain.show_view(
                        {
                            "azimuth": -127,
                            "elevation": 105,
                            "distance": 300,
                            "focalpoint": np.array([-11, -93, -49]),
                        },
                        roll=142,
                    )

                print(posPath)

                mlab.show(stop=True)
                pos = np.array(brain.show_view(), dtype="object")
                np.save(posPath, pos.astype("object"), allow_pickle=True)
                # print(pos)
            else:
                pos = np.load(posPath, allow_pickle=True)
                # print(pos)
                brain.show_view(
                    {
                        "azimuth": pos[0][0],
                        "elevation": pos[0][1],
                        "distance": pos[0][2],
                        "focalpoint": pos[0][3],
                    },
                    roll=pos[1],
                )
        else:
            if create_gif:
                plot_out_p, n = self._get_surfaceSavePath(param, hemi)

                if hemi[0].upper() == "L":
                    for iI, i in enumerate(np.linspace(-1, 89, 10)):
                        brain.show_view(
                            {
                                "azimuth": -i,
                                "elevation": 90,
                                "distance": 350,
                                "focalpoint": np.array([30, -130, -60]),
                            },
                            roll=-90,
                        )
                        brain.save_image(path.join(p, f"frame-{iI}.png"))

                elif hemi[0].upper() == "R":
                    for iI, i in enumerate(np.linspace(-1, 89, 10)):
                        brain.show_view(
                            {
                                "azimuth": i,
                                "elevation": -90,
                                "distance": 350,
                                "focalpoint": np.array([-30, -130, -60]),
                            },
                            roll=90,
                        )
                        brain.save_image(path.join(plot_out_p, f"frame-{iI}.png"))

                self._make_gif(plot_out_p, n + ".gif")

            else:
                if surface == "sphere":
                    if hemi[0].upper() == "L":
                        brain.show_view(
                            {
                                "azimuth": -80,
                                "elevation": 125,
                                "distance": 500,
                                "focalpoint": np.array([0, 0, 0]),
                            },
                            roll=-170,
                        )
                    elif hemi[0].upper() == "R":
                        brain.show_view(
                            {
                                "azimuth": 80,
                                "elevation": -125,
                                "distance": 500,
                                "focalpoint": np.array([0, 0, 0]),
                            },
                            roll=170,
                        )

                else:
                    if hemi[0].upper() == "L":
                        brain.show_view(
                            {
                                "azimuth": -57.5,
                                "elevation": 106,
                                "distance": 300,
                                "focalpoint": np.array([-43, -23, -8]),
                            },
                            roll=-130,
                        )
                    elif hemi[0].upper() == "R":
                        brain.show_view(
                            {
                                "azimuth": -127,
                                "elevation": 105,
                                "distance": 300,
                                "focalpoint": np.array([-11, -93, -49]),
                            },
                            roll=142,
                        )

        if save:
            brain.save_image(path.join(plot_out_p, n + f".{output_format}"))
            print(
                f'new Cortex Map saved to {path.join(plot_out_p, n + "." + output_format)}'
            )

        if interactive:
            mlab.show(stop=True)
        else:
            mlab.clf()


# ----------------------------------------------------------------------------#
def manual_masking(self):
    """
    Interactively draw a manual mask on the coverage map using sliders and buttons.

    Returns:
        np.ndarray: Boolean mask array indicating selected points.
    """

    maxEcc = self.maxEcc + 1e-12

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25, left=0.25)
    scatter = ax.scatter(self.x, self.y, s=0.2)
    ax.set_aspect("equal", "box")
    _draw_grid(ax, maxEcc)
    ax.set_xlim(-maxEcc, maxEcc)
    ax.set_ylim(-maxEcc, maxEcc)

    # Create axes for the sliders
    axcenterx = plt.axes([0.25, 0.1, 0.45, 0.03])
    axcentery = plt.axes([0.15, 0.25, 0.03, 0.55])
    axradius1 = plt.axes([0.1, 0.25, 0.03, 0.55])
    axradius2 = plt.axes([0.05, 0.25, 0.03, 0.55])
    axrotation = plt.axes([0.25, 0.05, 0.45, 0.03])
    axzoom = plt.axes([0.95, 0.25, 0.03, 0.55])

    # Create the sliders
    slider_centerx = Slider(axcenterx, "Center X", min(self.x), max(self.x), valinit=0)
    slider_centery = Slider(
        axcentery,
        "Center Y",
        min(self.y),
        max(self.y),
        valinit=0,
        orientation="vertical",
    )
    slider_radius1 = Slider(
        axradius1,
        "Radius 1",
        0,
        maxEcc,
        valinit=1,
        orientation="vertical",
    )
    slider_radius2 = Slider(
        axradius2,
        "Radius 2",
        0,
        maxEcc,
        valinit=1,
        orientation="vertical",
    )
    slider_rotation = Slider(axrotation, "Rotation", -np.pi, np.pi, valinit=0)
    slider_zoom = Slider(
        axzoom,
        "Zoom",
        maxEcc,
        max(self.x),
        valinit=maxEcc,
        orientation="vertical",
    )

    # Create the ellipse with initial width and height
    initX, initY = 0, 0
    initR1, initR2, initRot = 1, 1, 0
    ellipse = patches.Ellipse(
        (initX, initY), 2 * initR1, 2 * initR2, fill=False, color="green", linewidth=2
    )
    ax.add_patch(ellipse)
    mask = (
        (self.x - initX) * np.cos(initRot) + (self.y - initY) * np.sin(initRot)
    ) ** 2 / initR1**2 + (
        (self.x - initX) * np.sin(initRot) - (self.y - initY) * np.cos(initRot)
    ) ** 2 / initR2**2 < 1
    scatter.set_facecolor(["blue" if m else "grey" for m in mask])

    def update(val):
        centerx = slider_centerx.val
        centery = slider_centery.val
        radius1 = slider_radius1.val
        radius2 = slider_radius2.val
        rotation = slider_rotation.val
        zoom = slider_zoom.val
        mask = (
            (self.x - centerx) * np.cos(rotation)
            + (self.y - centery) * np.sin(rotation)
        ) ** 2 / radius1**2 + (
            (self.x - centerx) * np.sin(rotation)
            - (self.y - centery) * np.cos(rotation)
        ) ** 2 / radius2**2 < 1
        scatter.set_facecolor(["blue" if m else "grey" for m in mask])

        # Update the ellipse properties
        ellipse.center = (centerx, centery)
        ellipse.width = 2 * radius1
        ellipse.height = 2 * radius2
        ellipse.angle = np.degrees(rotation)
        # zoom th plot
        ax.set_xlim(-zoom, zoom)
        ax.set_ylim(-zoom, zoom)
        fig.canvas.draw_idle()

    slider_centerx.on_changed(update)
    slider_centery.on_changed(update)
    slider_radius1.on_changed(update)
    slider_radius2.on_changed(update)
    slider_rotation.on_changed(update)
    slider_zoom.on_changed(update)

    # Create a button
    axbutton = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(axbutton, "Get Mask", hovercolor="0.975")
    axbutton2 = plt.axes([0.8, 0.075, 0.1, 0.04])
    button2 = Button(axbutton2, "Add to Mask", hovercolor="0.975")

    # create the manual_mask variable
    plot_select_mask = np.zeros(self.x0.shape, dtype=bool)
    self._manual_mask = np.zeros(self.x0.shape, dtype=bool)

    def on_click(event):
        # Check if the click was on the plot
        if event.inaxes == ax:
            # Update the sliders
            slider_centerx.set_val(event.xdata)
            slider_centery.set_val(event.ydata)
            # Update the ellipse
            ellipse.center = (event.xdata, event.ydata)
            fig.canvas.draw_idle()
        elif event.inaxes == axbutton:
            # If plot_select_mask is None, calculate the mask based on the current ellipse
            if plot_select_mask.sum() == 0:
                centerx = slider_centerx.val
                centery = slider_centery.val
                radius1 = slider_radius1.val
                radius2 = slider_radius2.val
                rotation = slider_rotation.val
                mask = (
                    (self.x - centerx) * np.cos(rotation)
                    + (self.y - centery) * np.sin(rotation)
                ) ** 2 / radius1**2 + (
                    (self.x - centerx) * np.sin(rotation)
                    - (self.y - centery) * np.cos(rotation)
                ) ** 2 / radius2**2 < 1
                plot_select_mask[self.mask] = mask
            # Close the plot
            plt.close(fig)
            # Save the mask
            self._isManualMasked = True
            self._manual_mask = plot_select_mask
        elif event.inaxes == axbutton2:
            # Add the points in the selected area to the mask
            centerx = slider_centerx.val
            centery = slider_centery.val
            radius1 = slider_radius1.val
            radius2 = slider_radius2.val
            rotation = slider_rotation.val
            mask = (
                (self.x - centerx) * np.cos(rotation)
                + (self.y - centery) * np.sin(rotation)
            ) ** 2 / radius1**2 + (
                (self.x - centerx) * np.sin(rotation)
                - (self.y - centery) * np.cos(rotation)
            ) ** 2 / radius2**2 < 1
            if plot_select_mask.sum() == 0:
                plot_select_mask[self.mask] = mask
            else:
                plot_select_mask[self.mask] = np.logical_or(
                    plot_select_mask[self.mask], mask
                )
            # Color the selected points
            ax.scatter(self.x[mask], self.y[mask], color="magenta", s=0.3)
            fig.canvas.draw_idle()
        else:
            # Ignore clicks outside the plot
            return

    # Connect the click event to the on_click function
    cid = fig.canvas.mpl_connect("button_press_event", on_click)

    # Show the plot and wait for it to be closed
    plt.show()

    # Return the mask after the plot is closed
    return plot_select_mask


def _save_results_volume_docker(self, params, force, save):
    outFpath = path.join(
        self._derivatives_path,
        "prfresult",
        self._prfanalyze_method,
        self._prfanaAn,
        "volumeResults",
        self.subject,
        self.session,
    )
    makedirs(outFpath, exist_ok=True)

    try:
        dummy_file = nib.load(
            glob(
                path.join(
                    self._derivatives_path,
                    "fmriprep",
                    self.fmriprep_analysis,
                    self.subject,
                    self.session,
                    "func",
                    f"*{self.task}*space-T1w_desc-preproc_bold.nii*",
                )
            )[0]
        )

    except (KeyError, IndexError) as e:
        with open(
            glob(
                path.join(
                    self._derivatives_path,
                    "prfprepare",
                    f"analysis-{self.prfprepare_analysis}",
                    self.subject,
                    self.session,
                    "func",
                    f"{self.subject}_{self.session}_hemi-*_desc-*-*_maskinfo.json",
                )
            )[0],
            "r",
        ) as f:
            maskinfo = json.load(f)
        img_shape = maskinfo["origImageSize"]
        img_header = None

        if self._study == "hcpret":
            img_affine = np.array(
                [
                    [-1.60000002, 0.0, 0.0, 90.0],
                    [0.0, 1.60000002, 0.0, -126.0],
                    [0.0, 0.0, 1.60000002, -72.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        else:
            img_affine = np.eye(4)

    else:
        img = nib.funcs.four_to_three(dummy_file)[0]
        img_shape = img.shape
        img_header = img.header
        img_affine = img.affine

    # loop through prarams and save them
    all_params = {}
    for param in params:
        if not param.endswith("0") and param != "mask":
            param_orig = param + "0"
        else:
            param_orig = param

        if param == "mask":
            outFname = self._get_surfaceSavePath(
                param,
                "BOTH",
                "results",
                plain=False,
            )
        else:
            outFname = self._get_surfaceSavePath(param, "BOTH", "results", plain=True)
        print(outFname[1])
        outF = path.join(outFpath, outFname[1] + ".nii.gz")
        if not path.isfile(outF) or force:
            if param == "voxelTC0":
                dat = np.zeros((*img_shape, self.voxelTC0.shape[-1])) * np.nan
            else:
                dat = np.zeros(img_shape) * np.nan

            for pos, boldI in zip(self._roiIndOrig, self._roiIndBold):
                dat[tuple(pos)] = getattr(self, param_orig)[boldI]

            if param != param_orig:
                # set everything outside mask (ROI, VarExp, ...) to nan
                dat = deepcopy(dat)
                for pos in self._roiIndOrig[~self.mask[self._roiIndBold]]:
                    dat[tuple(pos)] = np.nan

            newNii = nib.Nifti1Image(dat, header=img_header, affine=img_affine)
            if save:
                nib.save(newNii, outF)
            all_params[param] = newNii

    return all_params


def _save_results_surface_docker(self, params, force, save):
    outFpath = path.join(
        self._derivatives_path,
        "prfresult",
        self._prfanalyze_method,
        self._prfanaAn,
        "surfaceResults",
        self.subject,
        self.session,
    )
    makedirs(outFpath, exist_ok=True)

    if self._hemis.lower() == "both":
        if self._study == "hcpret":
            hemis = ["b"]
        else:
            hemis = ["L", "R"]
    else:
        hemis = [self._hemis]

    all_hemis_params = {}
    for hemi in hemis:
        if self._study == "hcpret" and self._hemis.lower() == "both":
            img_shape = 91282
        else:
            try:
                dummyFile = nib.load(
                    glob(
                        path.join(
                            self._derivatives_path,
                            "fmriprep",
                            self.fmriprep_analysis,
                            self.subject,
                            self.session,
                            "func",
                            f"*task-{self.task}*hemi-{hemi}_space-{self.analysisSpace}_bold.func.gii",
                        )
                    )[0]
                )
            except (KeyError, IndexError) as e:
                with open(
                    glob(
                        path.join(
                            self._derivatives_path,
                            "prfprepare",
                            f"analysis-{self.prfprepare_analysis}",
                            self._subject,
                            self._session,
                            "func",
                            f"{self._subject}_{self._session}_hemi-{hemi}_desc-*-*_maskinfo.json",
                        )
                    )[0],
                    "r",
                ) as f:
                    maskinfo = json.load(f)
                img_shape = maskinfo["thisHemiSize"]
            else:
                img_shape = dummyFile.agg_data().shape

        all_params = {}
        for param in params:
            if not param.endswith("0") and param != "mask":
                param_orig = param + "0"
            else:
                param_orig = param

            if param == "mask":
                outFname = self._get_surfaceSavePath(
                    param,
                    hemi,
                    "results",
                    plain=False,
                )
            else:
                outFname = self._get_surfaceSavePath(param, hemi, "results", plain=True)

            outF = path.join(outFpath, outFname[1] + ".func.gii")
            if not path.isfile(outF) or force:
                if param == "voxelTC0":
                    dat = np.zeros((img_shape, self.voxelTC0.shape[-1])) * np.nan
                else:
                    dat = np.zeros(img_shape) * np.nan

                # create mask dependent on used hemisphere
                if hemi[0].upper() == "L":
                    hemiM = self._roiWhichHemi == "L"
                elif hemi[0].upper() == "R":
                    hemiM = self._roiWhichHemi == "R"
                elif hemi[0].upper() == "B":
                    hemiM = self._roiWhichHemi == "b"

                roiIndOrigHemi = self._roiIndOrig[hemiM]
                roiIndBoldHemi = self._roiIndBold[hemiM]

                for pos, boldI in zip(roiIndOrigHemi, roiIndBoldHemi):
                    dat[pos] = getattr(self, param_orig)[boldI]

                if param != param_orig:
                    # set everything outside mask (ROI, VarExp, ...) to nan
                    dat = deepcopy(dat)
                    dat[roiIndOrigHemi[~self.mask[roiIndBoldHemi]]] = np.nan

                newGii = nib.gifti.gifti.GiftiImage()
                newGii.add_gifti_data_array(
                    nib.gifti.gifti.GiftiDataArray(data=dat.astype(np.float32))
                )
                if save:
                    nib.save(newGii, outF)
                all_params[param] = newGii

        # if not saving, return all params for this hemisphere
        all_hemis_params[hemi] = all_params
    # if not saving, return all params for this hemisphere
    return all_params


def _save_results_mrVista(self, params, force, save):
    outFpath = path.join(
        self._baseP,
        self._study,
        "plots",
        "volumeResults",
        self.subject,
        self.session,
        self._analysis,
    )
    makedirs(outFpath, exist_ok=True)

    print("Warning: this is not properly implemented tested!")

    # lets load a dummy file to get the shape and header

    img = nib.load(
        glob(
            path.join(
                self._baseP,
                self._study,
                "subjects",
                self.subject,
                self.session,
                f"anatomy",
                "nifti",
                f"lpibrain.nii",
            )
        )[0]
    )

    # img = nib.funcs.four_to_three(dummy_file)[0]
    img_shape = img.shape
    img_header = img.header
    img_affine = img.affine

    # loop through prarams and save them
    all_params = {}
    for param in params:
        if not param.endswith("0") and param != "mask":
            param_orig = param + "0"
            varExpStr = f"_VarExp-{int(self._isVarExpMasked*100)}"
        else:
            param_orig = param
            varExpStr = ""

        if param == "mask":
            outFname = f"{self.subject}_{self.session}_hemi-BOTH_VarExp-{int(self._isVarExpMasked*100)}_desc-mask.nii.gz"
        else:
            outFname = f"{self.subject}_{self.session}_hemi-BOTH{varExpStr}_desc-{param}.nii.gz"

        print(outFname)
        outF = path.join(outFpath, outFname)
        if not path.isfile(outF) or force:
            if param == "voxelTC0":
                dat = np.zeros((*img_shape, self.voxelTC0.shape[-1])) * np.nan
            else:
                dat = np.zeros(img_shape) * np.nan

            coords_swap = self._coords[[2, 1, 0], ...].T.astype(int)
            coords_swap[:, 1] = coords_swap[:, 1] * -1 + img_shape[1] - 1
            coords_swap[:, 2] = coords_swap[:, 2] * -1 + img_shape[2] - 1

            if param == param_orig or param == "mask":
                for pos, boldI in zip(coords_swap, range(len(coords_swap))):
                    dat[tuple(pos)] = getattr(self, param_orig)[boldI]
            else:
                for pos, boldI in zip(coords_swap[self.mask], range(self.mask.sum())):
                    dat[tuple(pos)] = getattr(self, param)[boldI]

            self.dat = dat
            newNii = nib.Nifti1Image(dat, header=img_header, affine=img_affine)

            if save:
                nib.save(newNii, outF)
            all_params[param] = newNii

    return all_params


def save_results(
    self,
    params=None,
    force=False,
    save=True,
):
    """
    Save pRF results (parameters and mask) to disk in NIfTI or GIFTI format.

    Args:
        params (list, optional): List of parameter names to save. Defaults to ['x0', 'y0', 's0', 'r0', 'phi0', 'varexp0', 'mask'].
        force (bool, optional): Overwrite existing files if True. Defaults to False.
        save (bool, optional): Save the results to disk if True. Defaults to True.

    Raises:
        ValueError: If data is not from docker.
    """

    # set params to all if not defined
    if params is None:
        params = ["x0", "y0", "s0", "r0", "phi0", "varexp0", "mask"]
    else:
        # replace other names for params
        params_old = params
        params = []
        for p in params_old:
            if "ecc" in p:
                params.append(p.replace("ecc", "r"))
            elif "pol" in p:
                params.append(p.replace("pol", "phi"))
            else:
                params.append(p)

    if not isinstance(params, list):
        params = [params]

    # data from docker analysis
    if self._dataFrom == "docker":
        if self.analysisSpace == "volume":
            return _save_results_volume_docker(self, params, force, save)

        elif self.analysisSpace == "fsnative" or self.analysisSpace == "fsaverage":
            return _save_results_surface_docker(self, params, force, save)

    # data from mrVista
    elif self._dataFrom == "mrVista":
        return _save_results_mrVista(self, params, force, save)

    else:
        raise NotImplementedError(
            f"Unknown data source: {self._dataFrom}. Cannot save results."
        )
