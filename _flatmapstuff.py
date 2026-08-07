"""Cortical flatmaps via pycortex.

Companion to :func:`_plotstuff.plot_toSurface`.  The 3D views are rendered with
PyVista; flatmaps are rendered here, because pycortex is far better at them --
it brings curvature, ROI outlines, ROI labels and a proper colorbar for free,
at any dpi.

pycortex is imported lazily, so PRFclass keeps working without it.  Install
with::

    pip install setuptools wheel numpy cython   # build requirements first
    pip install pycortex

pycortex keeps its own subject filestore, separate from ``$SUBJECTS_DIR``.
Subjects are imported into it automatically on first use.  Flatmaps
additionally need a flattened surface (``?h.<patch>.patch.flat``); fsaverage
ships one, individual subjects need :func:`prepare_flatmap_surface`.
"""

import os
import subprocess
import warnings
from os import makedirs, path

import numpy as np

from ._plotstuff import _build_surface_overlay, _resolve_fs_subject_dir
from ._surfaceplot import colorbar_style_for, make_colorbar

__all__ = ["plot_toFlatmap", "prepare_flatmap_surface"]

_POLAR_CMAP_NAME = "prf_polar"


def _import_cortex():
    try:
        import cortex
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "pycortex is required for flatmaps. Install it with:\n"
            "    pip install setuptools wheel numpy cython\n"
            "    pip install pycortex"
        ) from exc
    return cortex


def _register_cmap(cmap):
    """pycortex takes colormaps by name, so a custom one has to be registered."""
    import matplotlib

    if not isinstance(cmap, str):
        name = getattr(cmap, "name", "") or _POLAR_CMAP_NAME
        if name in ("", "prf_polar"):
            name = _POLAR_CMAP_NAME
        try:
            matplotlib.colormaps.register(cmap, name=name)
        except ValueError:
            pass  # already registered
        return name
    return cmap


def _fs_paths(self):
    """FreeSurfer subjects dir and the resolved subject name for this instance."""
    if self._dataFrom == "mrVista":
        fsP = path.join(
            self._baseP, self._study, "subjects", self.subject, self.session,
            "segmentation",
        )
        fs_subject = "freesurfer"
    else:
        fsP = path.join(
            self._derivatives_path, "fmriprep",
            f"analysis-{self.fmriprep_analysis}", "sourcedata", "freesurfer",
        )
        fs_subject = (
            "fsaverage" if self.analysisSpace == "fsaverage" else self.subject
        )
    resolved, _ = _resolve_fs_subject_dir(fsP, fs_subject, self.session)
    return fsP, resolved


def _surfaces_present(cortex, cortex_subject):
    """True if the filestore entry is complete enough to render a flatmap."""
    if cortex_subject not in cortex.db.subjects:
        return False
    try:
        cortex.db.get_surf(cortex_subject, "wm", hemisphere="lh")
        cortex.db.get_surf(cortex_subject, "pia", hemisphere="lh")
        return True
    except Exception:
        return False


def _write_gifti(coords, faces, out_file):
    import nibabel as nib
    from nibabel import gifti

    img = gifti.GiftiImage()
    img.add_gifti_data_array(
        gifti.GiftiDataArray(
            np.ascontiguousarray(coords, dtype=np.float32),
            intent="NIFTI_INTENT_POINTSET",
        )
    )
    img.add_gifti_data_array(
        gifti.GiftiDataArray(
            np.ascontiguousarray(faces, dtype=np.int32),
            intent="NIFTI_INTENT_TRIANGLE",
        )
    )
    nib.save(img, out_file)


def _import_subj_surfaces_only(cortex, fsP, fs_subject, cortex_subject):
    """Populate a pycortex subject from surfaces alone.

    pycortex's ``import_subj`` converts ``mri/wm.mgz``, which fsaverage does
    not ship, and writes fiducials back into ``$SUBJECTS_DIR``, which is often
    read-only. Vertex flatmaps need neither, so the surface and surface-info
    files are written directly.

    Volume-based pycortex features (``cortex.Volume``, transforms) are not
    available for a subject imported this way; run
    ``cortex.freesurfer.import_subj`` if you need those.
    """
    import nibabel as nib

    filestore = path.join(cortex.database.default_filestore, cortex_subject)
    for sub in ("transforms", "anatomicals", "cache", "surfaces", "surface-info",
                "views"):
        makedirs(path.join(filestore, sub), exist_ok=True)

    surf_dir = path.join(fsP, fs_subject, "surf")

    # pycortex name <- freesurfer name
    for fs_name, cx_name in (("smoothwm", "wm"), ("pial", "pia"),
                             ("inflated", "inflated")):
        for hemi in ("lh", "rh"):
            src = path.join(surf_dir, f"{hemi}.{fs_name}")
            if not path.isfile(src):
                raise RuntimeError(f"Missing surface required by pycortex: {src}")
            coords, faces, meta = nib.freesurfer.read_geometry(
                src, read_metadata=True
            )
            # replicate `mris_convert --to-scanner`: shift out of TKR space
            cras = meta.get("cras")
            if cras is not None:
                coords = coords + np.asarray(cras, dtype=coords.dtype)
            _write_gifti(
                coords,
                faces,
                path.join(filestore, "surfaces", f"{cx_name}_{hemi}.gii"),
            )

    # curvature/sulc/thickness -- pycortex stores these NEGATED
    for fs_name, cx_name in (("sulc", "sulcaldepth"), ("thickness", "thickness"),
                             ("curv", "curvature")):
        try:
            lh = nib.freesurfer.read_morph_data(path.join(surf_dir, f"lh.{fs_name}"))
            rh = nib.freesurfer.read_morph_data(path.join(surf_dir, f"rh.{fs_name}"))
        except (OSError, FileNotFoundError):
            warnings.warn(f"No {fs_name} for {fs_subject}; skipping.")
            continue
        np.savez(
            path.join(filestore, "surface-info", f"{cx_name}.npz"), left=-lh, right=-rh
        )

    cortex.database.db = cortex.database.Database()


def _ensure_subject(cortex, fsP, fs_subject, cortex_subject=None, force=False):
    """Import a FreeSurfer subject into pycortex's filestore if it is missing."""
    cortex_subject = cortex_subject or fs_subject
    if _surfaces_present(cortex, cortex_subject) and not force:
        return cortex_subject

    print(f"Importing {fs_subject} into the pycortex filestore as {cortex_subject} ...")
    _import_subj_surfaces_only(cortex, fsP, fs_subject, cortex_subject)
    return cortex_subject


def _flat_patch_path(fsP, fs_subject, hemi, patch):
    """Locate a flattened patch, accepting either naming convention.

    pycortex expects ``{hemi}.{patch}.flat.patch.3d`` (what its own
    cut_surface/flatten workflow produces), while FreeSurfer's ``mris_flatten``
    writes ``{hemi}.{patch}.patch.flat``. Returns ``(path, is_pycortex_style)``
    or ``(None, False)``.
    """
    surf_dir = path.join(fsP, fs_subject, "surf")
    cx = path.join(surf_dir, f"{hemi}.{patch}.flat.patch.3d")
    if path.isfile(cx):
        return cx, True
    fs = path.join(surf_dir, f"{hemi}.{patch}.patch.flat")
    if path.isfile(fs):
        return fs, False
    return None, False


def _shadow_subject_dir(fsP, fs_subject, patch, hemis=("lh", "rh")):
    """Build a symlink-only view of the subject that pycortex can read.

    Needed when the flat patches follow FreeSurfer's naming: rather than
    renaming files inside $SUBJECTS_DIR (often read-only, e.g. a shared
    fsaverage), link the few files pycortex touches into a scratch directory
    under its own filestore, using the names it expects.
    """
    import cortex

    root = path.join(
        path.dirname(cortex.database.default_filestore), "prfclass_flatshadow"
    )
    surf_dir = path.join(root, fs_subject, "surf")
    makedirs(surf_dir, exist_ok=True)

    src_surf = path.join(fsP, fs_subject, "surf")
    for hemi in hemis:
        # get_surf(..., "patch") reads smoothwm plus the patch itself
        for name in (f"{hemi}.smoothwm", f"{hemi}.inflated", f"{hemi}.curv"):
            src, dst = path.join(src_surf, name), path.join(surf_dir, name)
            if path.isfile(src) and not path.exists(dst):
                os.symlink(src, dst)

        flat, _ = _flat_patch_path(fsP, fs_subject, hemi, patch)
        dst = path.join(surf_dir, f"{hemi}.{patch}.flat.patch.3d")
        if flat and not path.exists(dst):
            os.symlink(flat, dst)

    return root


def _ensure_flat(cortex, fsP, fs_subject, cortex_subject, patch="cortex"):
    """Make sure the flat surface is available in the pycortex filestore."""
    try:
        cortex.db.get_surf(cortex_subject, "flat", merge=False)
        return
    except Exception:
        pass

    found = {h: _flat_patch_path(fsP, fs_subject, h, patch) for h in ("lh", "rh")}
    missing = [h for h, (p, _) in found.items() if p is None]
    if missing:
        raise RuntimeError(
            f"No flattened surface for {fs_subject} "
            f"({', '.join(missing)} missing {patch}.patch.flat).\n"
            "Run instance.prepare_flatmap_surface() to create it -- note that "
            "flattening takes hours per hemisphere and needs a manual cut if no "
            f"{patch}.patch.3d exists."
        )

    # if any hemisphere uses FreeSurfer's naming, read through a shadow dir
    if not all(is_cx for _, is_cx in found.values()):
        import_dir = _shadow_subject_dir(fsP, fs_subject, patch)
    else:
        import_dir = fsP

    print(f"Importing flat patches for {cortex_subject} ...")
    # auto_overwrite avoids an interactive prompt that would deadlock any
    # non-tty run
    cortex.freesurfer.import_flat(
        fs_subject, patch, hemis=["lh", "rh"], cx_subject=cortex_subject,
        freesurfer_subject_dir=import_dir, auto_overwrite=True,
    )
    cortex.db.reload_subjects()


def _draw_flat_borders(cortex, fig, cortex_subject, vert_lists, color, width):
    """Outline ROIs on a flatmap by contouring them in flat-surface coordinates.

    pycortex's own ``with_rois`` draws whatever is in the subject's
    ``overlays.svg``; our ROIs come from the analysis atlases instead, so they
    are contoured directly on the flat triangulation and drawn onto the same
    axes as the flatmap image.
    """
    import matplotlib.tri as mtri
    from cortex.quickflat.utils import _get_extents

    pts, polys = cortex.db.get_surf(cortex_subject, "flat", merge=True, nudge=True)
    x, y = pts[:, 0], pts[:, 1]

    ax = fig.axes[0]
    left, right, bottom, top = _get_extents(fig)

    # drop degenerate triangles so tricontour does not choke on the medial wall
    good = np.isfinite(x) & np.isfinite(y)
    mask = ~np.all(good[polys], axis=1)
    tri = mtri.Triangulation(np.nan_to_num(x), np.nan_to_num(y), polys)
    tri.set_mask(mask)

    for verts in vert_lists:
        idx = np.asarray(verts).ravel().astype(np.int64)
        idx = idx[(idx >= 0) & (idx < len(pts))]
        if idx.size == 0:
            continue
        roi = np.zeros(len(pts), dtype=float)
        roi[idx] = 1.0
        ax.tricontour(
            tri, roi, levels=[0.5], colors=[color], linewidths=width, zorder=5,
        )

    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)


# ----------------------------------------------------------------------------#
def plot_toFlatmap(
    self,
    param="ecc",
    hemi="both",
    save=False,
    force=False,
    surface="flatmap",
    showBordersAtlas=None,
    showBordersArea=None,
    maxEcc=None,
    plot_colorbar=True,
    output_format="pdf",
    pmax=None,
    pmin=None,
    dpi=300,
    height=1024,
    with_curvature=True,
    with_rois=False,
    with_labels=False,
    with_sulci=False,
    border_color="black",
    border_width=1.5,
    colorbar_style="auto",
    colorbar_format="svg",
    cbar_figsize=(2.5, 2.5),
    cbar_fontsize=12,
    cortex_subject=None,
    patch="cortex",
):
    """
    Plot a pRF parameter on a cortical flatmap using pycortex.

    Shares its data preparation, masking, colormaps and colorbars with
    :func:`plot_toSurface`, so a flatmap and a 3D view of the same parameter
    are directly comparable.

    Args:
        param (str or np.ndarray, optional): Parameter to plot ('ecc', 'pol',
            'sig', 'var') or an array. Defaults to 'ecc'.
        hemi (str, optional): 'both' (the usual case for a flatmap), 'L' or 'R'.
            A single hemisphere still renders the whole flatmap, with the other
            side left blank. Defaults to 'both'.
        save (bool, optional): Write the figure to disk. Defaults to False.
        force (bool, optional): Overwrite an existing file. Defaults to False.
        showBordersAtlas (list or str, optional): Atlases to outline. Defaults to None.
        showBordersArea (list or str, optional): Areas to outline. Defaults to None.
        maxEcc (float, optional): Maximum eccentricity. Defaults to None.
        plot_colorbar (bool, optional): Write a colorbar. Defaults to True.
        output_format (str, optional): Figure format. Defaults to 'pdf'.
        pmax, pmin (float, optional): Manual colour limits. Defaults to None.
        dpi (int, optional): Output resolution. Defaults to 300.
        height (int, optional): Flatmap raster height in pixels. Defaults to 1024.
        with_curvature (bool, optional): Draw the curvature underlay. Defaults to True.
        with_rois (bool, optional): Draw pycortex's own overlays.svg ROIs.
            Defaults to False -- use showBordersAtlas/Area for analysis ROIs.
        with_labels (bool, optional): Label pycortex ROIs. Defaults to False.
        with_sulci (bool, optional): Draw pycortex sulci. Defaults to False.
        border_color (str, optional): ROI outline colour. Defaults to 'black'.
        border_width (float, optional): ROI outline width. Defaults to 1.5.
        colorbar_style (str, optional): 'auto', 'wheel_ecc', 'wheel_pol',
            'bar_h', 'bar_v', 'inline' (pycortex's own bar) or 'none'.
            Defaults to 'auto'.
        colorbar_format (str, optional): Format of the separate colorbar file.
            Defaults to 'svg'.
        cbar_figsize, cbar_fontsize: Colorbar styling. Defaults to (2.5, 2.5), 12.
        cortex_subject (str, optional): Name in the pycortex filestore. Defaults
            to the FreeSurfer subject name.
        patch (str, optional): Flat patch name. Defaults to 'cortex'.

    Returns:
        matplotlib.figure.Figure: The flatmap figure.

    Raises:
        RuntimeError: If the data is volumetric, or no flat surface exists.
        ImportError: If pycortex is not installed.
    """
    cortex = _import_cortex()

    if self._dataFrom in ["docker", "samsrf", "hdf5"] and self.analysisSpace == "volume":
        raise RuntimeError("plot_toFlatmap is not supported for volumetric data!")

    maxEcc = maxEcc if maxEcc else self.maxEcc

    fsP, fs_subject = _fs_paths(self)
    cortex_subject = _ensure_subject(cortex, fsP, fs_subject, cortex_subject)
    _ensure_flat(cortex, fsP, fs_subject, cortex_subject, patch=patch)

    # pycortex concatenates the hemispheres, left first
    n_left = cortex.db.get_surf(cortex_subject, "flat", hemisphere="lh")[0].shape[0]
    n_right = cortex.db.get_surf(cortex_subject, "flat", hemisphere="rh")[0].shape[0]

    hemis = ["L", "R"] if str(hemi).lower() in ("both", "b") else [hemi[0].upper()]

    if save:
        plot_out_p, n = self._get_surfaceSavePath(param, "".join(hemis), surface)
        out_file = path.join(plot_out_p, n + f".{output_format}")
        if path.isfile(out_file) and not force:
            return None
        makedirs(plot_out_p, exist_ok=True)

    # --- data, identical to the 3D path ------------------------------------
    full = np.full(n_left + n_right, np.nan, dtype=np.float32)
    cmap = datMin = datMax = None
    for h in hemis:
        n_vert = n_left if h == "L" else n_right
        data, cmap, datMin, datMax = _build_surface_overlay(
            self, param, h, n_vert, maxEcc, pmin, pmax
        )
        if h == "L":
            full[:n_left] = data
        else:
            full[n_left:] = data

    cmap_name = _register_cmap(cmap)

    inline_cbar = plot_colorbar and colorbar_style == "inline"
    vol = cortex.Vertex(
        full, cortex_subject, vmin=datMin, vmax=datMax, cmap=cmap_name
    )
    fig = cortex.quickflat.make_figure(
        vol,
        height=height,
        dpi=dpi,
        with_curvature=with_curvature,
        with_rois=with_rois,
        with_labels=with_labels,
        with_sulci=with_sulci,
        with_colorbar=inline_cbar,
        nanmean=False,
    )

    # --- ROI outlines from the analysis atlases ----------------------------
    if showBordersArea is not None and showBordersAtlas is not None:
        from ._maskstuff import _get_available_atlases_and_rois
        from ._plotstuff import label

        roi_info = _get_available_atlases_and_rois(self)
        if self._dataFrom == "samsrf":
            roi_info["area_files"] = self._area_files_p
            left_hemi = self._vertices_left_hemi
        else:
            left_hemi = None

        if showBordersAtlas == "all" or showBordersAtlas is True:
            ats = list(roi_info["atlases"].keys())
        elif isinstance(showBordersAtlas, list):
            ats = showBordersAtlas
        else:
            ats = [showBordersAtlas]
        ars = (
            showBordersArea
            if isinstance(showBordersArea, list)
            else [showBordersArea]
        )

        vert_lists = []
        for at in ats:
            for ar in ars:
                for h in hemis:
                    try:
                        lab = label(ar, at, h, roi_info, left_hemi)
                    except (IndexError, KeyError, ValueError) as exc:
                        warnings.warn(f"Skipping border {at}/{ar}/{h}: {exc}")
                        continue
                    v = np.asarray(lab.vertices).ravel().astype(np.int64)
                    vert_lists.append(v if h == "L" else v + n_left)

        if vert_lists:
            _draw_flat_borders(
                cortex, fig, cortex_subject, vert_lists, border_color, border_width
            )

    # --- save --------------------------------------------------------------
    if save:
        fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
        print(f"new Flatmap saved to {out_file}")

        if plot_colorbar and colorbar_style not in ("inline", "none", None):
            import matplotlib.pyplot as plt

            style = (
                colorbar_style_for(param)
                if colorbar_style == "auto"
                else colorbar_style
            )
            cbar_file = path.join(plot_out_p, n + f"_colorbar.{colorbar_format}")
            cfig = make_colorbar(
                style, cmap, vmin=datMin, vmax=datMax,
                figsize=cbar_figsize, fontsize=cbar_fontsize,
            )
            cfig.savefig(cbar_file, dpi=dpi, bbox_inches="tight", transparent=True)
            plt.close(cfig)
            print(f"colorbar saved to {cbar_file}")

    return fig


# ----------------------------------------------------------------------------#
def prepare_flatmap_surface(
    self, hemi="both", patch="cortex", force=False, run=True, log_dir=None
):
    """
    Create the flattened FreeSurfer surface a flatmap needs.

    fsaverage ships ``?h.cortex.patch.flat`` already, so this is only needed for
    individual subjects.  Flattening takes **hours per hemisphere**, so by
    default it is launched detached with its output going to a log file.

    Args:
        hemi (str, optional): 'both', 'L' or 'R'. Defaults to 'both'.
        patch (str, optional): Patch name. Defaults to 'cortex'.
        force (bool, optional): Re-flatten even if a .flat already exists.
            Defaults to False.
        run (bool, optional): Actually launch mris_flatten. With False the
            commands are only printed, e.g. to submit them to a cluster.
            Defaults to True.
        log_dir (str, optional): Where to write the logs. Defaults to the
            subject's surf directory.

    If no ``?h.<patch>.patch.3d`` cut exists, one is generated from
    ``?h.<patch>.label`` with ``label2patch``. That label already excludes the
    medial wall, so it gives a usable whole-hemisphere flatmap; adding a manual
    calcarine relief cut in freeview reduces distortion but is optional.

    Returns:
        dict: ``{hemi: status}``, where status is 'exists', 'started',
        'printed' or 'failed'.

    Raises:
        RuntimeError: If neither a cut nor a label exists, or if a FreeSurfer
            command is unavailable or fails.
    """
    fsP, fs_subject = _fs_paths(self)
    surf_dir = path.join(fsP, fs_subject, "surf")
    hemis = ["lh", "rh"] if str(hemi).lower() in ("both", "b") else [
        f"{hemi[0].lower()}h"
    ]
    log_dir = log_dir or surf_dir
    makedirs(log_dir, exist_ok=True)

    status = {}
    for h in hemis:
        flat = path.join(surf_dir, f"{h}.{patch}.patch.flat")
        cut = path.join(surf_dir, f"{h}.{patch}.patch.3d")

        if path.isfile(flat) and not force:
            print(f"{h}: {flat} already exists")
            status[h] = "exists"
            continue

        if not path.isfile(cut):
            label = path.join(fsP, fs_subject, "label", f"{h}.{patch}.label")
            if not path.isfile(label):
                raise RuntimeError(
                    f"{h}: neither {cut} nor {label} exists.\n"
                    "A flat patch needs a cut surface first. Either:\n"
                    f"  1. freeview -f {path.join(surf_dir, h + '.inflated')} , "
                    f"cut the medial wall and calcarine, save as {h}.{patch}.patch.3d\n"
                    "  2. cortex.segment.cut_surface(subject, hemi) for the "
                    "pycortex/Blender workflow.\n"
                    "Then run this again."
                )

            # ?h.cortex.label (medial wall already excluded) converts straight
            # into a patch, which is the standard whole-hemisphere flatmap cut.
            # A manual calcarine relief cut lowers distortion but is optional.
            cmd = ["label2patch", "-sdir", fsP, fs_subject, h, label, cut]
            if not run:
                print(" ".join(cmd))
            else:
                print(f"{h}: creating {path.basename(cut)} from {path.basename(label)}")
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                except FileNotFoundError:
                    raise RuntimeError(
                        f"{h}: label2patch not found. Source FreeSurfer first."
                    ) from None
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(
                        f"{h}: label2patch failed:\n"
                        f"{exc.stderr.decode(errors='replace')[-2000:]}"
                    ) from None

        cmd = ["mris_flatten", "-w", "0", "-distances", "20", "7", cut, flat]
        logf = path.join(log_dir, f"{h}.{patch}.flatten.log")

        if not run:
            print(" ".join(cmd) + f"  > {logf} 2>&1")
            status[h] = "printed"
            continue

        print(
            f"{h}: flattening -- this takes hours. Progress: tail -f {logf}"
        )
        try:
            with open(logf, "w") as fl:
                subprocess.Popen(
                    cmd, stdout=fl, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            status[h] = "started"
        except (OSError, FileNotFoundError) as exc:
            print(f"{h}: could not launch mris_flatten ({exc}). Is FreeSurfer set up?")
            status[h] = "failed"

    if any(v == "started" for v in status.values()):
        print(
            "\nWhen flattening finishes, run plot_toFlatmap() again -- the flat "
            "patches are imported into pycortex automatically."
        )
    return status
