"""Rendering of cortical surfaces with PyVista/VTK.

Provides a ``Brain`` object that follows the pysurfer API for the subset of
calls PRFclass needs. VTK is the only native dependency; it ships as a binary
wheel for every platform.

Design note
-----------
``Brain`` is a scene description, not a live window. A ``pyvista.Plotter`` is
built lazily for each render and destroyed afterwards (~0.1 s), which keeps
interactive positioning, off-screen saving and GIF frames independent of each
other.

Headless rendering
------------------
On macOS ``off_screen=True`` works out of the box. On a headless Linux cluster
pick one of:

* ``xvfb-run -a python your_script.py``  (``pv.start_xvfb()`` was removed in
  pyvista 0.48, do not call it)
* ``pip install --extra-index-url https://wheels.vtk.org vtk-osmesa``  (CPU)
* ``pip install --extra-index-url https://wheels.vtk.org vtk-egl``  (GPU)

The last two require uninstalling the regular ``vtk`` package first.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from os import path

import nibabel as nib
import numpy as np
import pyvista as pv
import vtk
from matplotlib import colors as mcolors

__all__ = [
    "Brain",
    "CameraState",
    "mayavi_view_to_camera",
    "camera_to_mayavi_view",
    "resolve_surface",
    "load_curvature",
    "roi_contour",
    "roi_border_vertices",
    "make_colorbar",
    "POLAR_CMAP",
]


# ---------------------------------------------------------------------------#
# constants
# ---------------------------------------------------------------------------#

#: The cyclic polar-angle colormap used throughout PRFclass.
POLAR_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "prf_polar", ["yellow", (0, 0, 1), (0, 1, 0), (1, 0, 0), "yellow"]
)

#: Curvature underlay presets. "classic" maps the binarised curvature to grey
#: 0.778 (gyri) and 0.408 (sulci).
_CORTEX_PRESETS = {
    "classic": dict(cmap="Greys", clim=(-1.0, 2.0)),
    "high_contrast": dict(cmap="Greys", clim=(-0.1, 1.3)),
    "low_contrast": dict(cmap="Greys", clim=(-5.0, 5.0)),
    "bone": dict(cmap="bone", clim=(-0.2, 2.0)),
}

#: How far ROI border lines are pushed along the vertex normals, per surface.
#: Without an offset the surface occludes most of the line; more inflated
#: surfaces need a bigger push because their triangles are larger.
_BORDER_OFFSET = {
    "inflated": 0.6,
    "very_inflated": 0.8,
    "sphere": 0.4,
    "pial": 0.25,
    "white": 0.25,
    "midthickness": 0.25,
    "smoothwm": 0.25,
    "orig": 0.25,
    "pial_semi_inflated": 0.45,
}

#: ``surface=`` aliases -> FreeSurfer file suffix.
_SURF_ALIASES = {
    "grey": "pial",
    "gray": "pial",
    "cortex": "inflated",
    "semi_inflated": "pial_semi_inflated",
    "semiinflated": "pial_semi_inflated",
    "mid": "midthickness",
}


# ---------------------------------------------------------------------------#
# camera
# ---------------------------------------------------------------------------#


@dataclass(frozen=True)
class CameraState:
    """Complete VTK camera state, enough for a lossless round-trip."""

    position: tuple = (0.0, 0.0, 1.0)
    focal_point: tuple = (0.0, 0.0, 0.0)
    view_up: tuple = (0.0, 1.0, 0.0)
    view_angle: float = 30.0
    parallel_projection: bool = False
    parallel_scale: float = 1.0
    clipping_range: tuple = (0.1, 1000.0)
    window_size: tuple = (800, 800)

    # -- conversions ------------------------------------------------------
    @classmethod
    def from_camera(cls, cam, window_size=(800, 800)):
        return cls(
            position=tuple(float(v) for v in cam.GetPosition()),
            focal_point=tuple(float(v) for v in cam.GetFocalPoint()),
            view_up=tuple(float(v) for v in cam.GetViewUp()),
            view_angle=float(cam.GetViewAngle()),
            parallel_projection=bool(cam.GetParallelProjection()),
            parallel_scale=float(cam.GetParallelScale()),
            clipping_range=tuple(float(v) for v in cam.GetClippingRange()),
            window_size=tuple(int(v) for v in window_size),
        )

    def to_camera(self, cam=None):
        cam = vtk.vtkCamera() if cam is None else cam
        cam.SetPosition(*self.position)
        cam.SetFocalPoint(*self.focal_point)
        cam.SetViewUp(*self.view_up)
        cam.SetViewAngle(self.view_angle)
        cam.SetParallelProjection(bool(self.parallel_projection))
        cam.SetParallelScale(self.parallel_scale)
        cam.SetClippingRange(*self.clipping_range)
        return cam

    # -- hdf5 -------------------------------------------------------------
    def to_h5(self, group):
        group.create_dataset("position", data=np.asarray(self.position, float))
        group.create_dataset("focal_point", data=np.asarray(self.focal_point, float))
        group.create_dataset("view_up", data=np.asarray(self.view_up, float))
        group.create_dataset(
            "clipping_range", data=np.asarray(self.clipping_range, float)
        )
        group.create_dataset("window_size", data=np.asarray(self.window_size, int))
        group.attrs["view_angle"] = float(self.view_angle)
        group.attrs["parallel_projection"] = bool(self.parallel_projection)
        group.attrs["parallel_scale"] = float(self.parallel_scale)

    @classmethod
    def from_h5(cls, group):
        return cls(
            position=tuple(np.asarray(group["position"][()], float)),
            focal_point=tuple(np.asarray(group["focal_point"][()], float)),
            view_up=tuple(np.asarray(group["view_up"][()], float)),
            view_angle=float(group.attrs.get("view_angle", 30.0)),
            parallel_projection=bool(group.attrs.get("parallel_projection", False)),
            parallel_scale=float(group.attrs.get("parallel_scale", 1.0)),
            clipping_range=tuple(
                np.asarray(group["clipping_range"][()], float)
                if "clipping_range" in group
                else (0.1, 1000.0)
            ),
            window_size=tuple(
                np.asarray(group["window_size"][()], int)
                if "window_size" in group
                else (800, 800)
            ),
        )


def mayavi_view_to_camera(cam, azimuth, elevation, distance, focalpoint, roll=None):
    """Set a VTK camera from a spherical view specification.

    * ``azimuth``   -- angle in the x-y plane measured from +x, degrees
    * ``elevation`` -- zenith angle measured from +z, degrees
    * ``distance``  -- ``|position - focalpoint|``
    * ``roll``      -- ``vtkCamera::SetRoll``, applied last

    Order matters: focal point, position, view-up, then ``SetRoll``.
    """
    phi = np.deg2rad(float(azimuth))
    theta = np.deg2rad(float(elevation))
    fp = np.asarray(focalpoint, dtype=float).ravel()
    r = float(distance)

    # do NOT canonicalise elevation into [0, 180]: the GIF path uses
    # elevation=-90 and relies on sin(theta) < 0
    pos = fp + r * np.array(
        [
            np.cos(phi) * np.sin(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(theta),
        ]
    )

    cam.SetFocalPoint(*fp)
    cam.SetPosition(*pos)

    # near the poles the +z view-up degenerates, so use the azimuth instead
    if abs(float(elevation)) < 5.0 or abs(float(elevation)) > 175.0:
        cam.SetViewUp(np.sin(phi), np.cos(phi), 0.0)
    else:
        cam.SetViewUp(0.0, 0.0, 1.0)
    cam.ComputeViewPlaneNormal()

    if roll is not None:
        cam.SetRoll(float(roll))
    cam.OrthogonalizeViewUp()
    return cam


def camera_to_mayavi_view(cam):
    """Inverse of :func:`mayavi_view_to_camera`.

    Returns ``((azimuth, elevation, distance, focalpoint), roll)``.

    Elevation comes back in ``[0, 180]``, so a camera built from
    ``(az=75, el=-125)`` reads back as ``(-105, 125)``. Both describe the same
    camera; the position and view-up are identical.
    """
    fp = np.asarray(cam.GetFocalPoint(), dtype=float)
    pos = np.asarray(cam.GetPosition(), dtype=float)
    x, y, z = pos - fp
    r = float(np.sqrt(x * x + y * y + z * z))
    azimuth = float(np.rad2deg(np.arctan2(y, x)))
    elevation = float(np.rad2deg(np.arccos(np.clip(z / r, -1.0, 1.0)))) if r else 0.0
    return (azimuth, elevation, r, fp), float(cam.GetRoll())


# ---------------------------------------------------------------------------#
# surface + curvature loading
# ---------------------------------------------------------------------------#


def _read_geometry(subjects_dir, subject, hemi, name):
    """Read ``{hemi}.{name}``, with the pial -> pial.T1 fallback."""
    base = path.join(subjects_dir, subject, "surf", f"{hemi}.{name}")
    for cand in (base, base + ".T1"):
        if path.isfile(cand):
            return nib.freesurfer.read_geometry(cand)
    raise FileNotFoundError(f"Surface not found: {base}")


def resolve_surface(subjects_dir, subject, hemi, name, very_inflated_t=0.35):
    """Return ``(coords, faces)`` for a named surface.

    Three tiers are supported:

    1. native FreeSurfer files -- ``pial``, ``white``, ``smoothwm``, ``orig``,
       ``inflated``, ``sphere``, ``pial_semi_inflated``, ...
    2. aliases -- ``grey``/``gray`` -> ``pial``, ``cortex`` -> ``inflated``,
       ``semi_inflated`` -> ``pial_semi_inflated``, ``mid`` -> ``midthickness``
    3. derived on the fly, nothing written to ``$SUBJECTS_DIR``:

       * ``midthickness`` -- ``0.5 * (white + pial)``, HCP's definition
       * ``very_inflated`` -- the real file if present, otherwise a blend of
         ``inflated`` towards a radius-matched ``sphere``.  FreeSurfer has no
         very_inflated surface (it is an HCP/fsLR product), so this is an
         approximation; tune it with ``very_inflated_t``.

    Faces are identical across all surfaces of a subject, so overlay data,
    curvature and ROI vertex indices are reusable unchanged.
    """
    name = _SURF_ALIASES.get(str(name).lower(), name)

    if name == "midthickness":
        w_coords, faces = _read_geometry(subjects_dir, subject, hemi, "white")
        p_coords, _ = _read_geometry(subjects_dir, subject, hemi, "pial")
        return 0.5 * (w_coords + p_coords), faces

    if name == "very_inflated":
        try:
            return _read_geometry(subjects_dir, subject, hemi, "very_inflated")
        except FileNotFoundError:
            pass
        infl, faces = _read_geometry(subjects_dir, subject, hemi, "inflated")
        sph, _ = _read_geometry(subjects_dir, subject, hemi, "sphere")
        t = float(very_inflated_t)
        # match the sphere's scale to the inflated surface before blending
        infl_c = infl.mean(axis=0)
        infl_r = np.sqrt(((infl - infl_c) ** 2).sum(axis=1)).mean()
        sph_c = sph.mean(axis=0)
        sph_r = np.sqrt(((sph - sph_c) ** 2).sum(axis=1)).mean()
        sph_scaled = (sph - sph_c) * (infl_r / sph_r) + infl_c
        return (1.0 - t) * infl + t * sph_scaled, faces

    return _read_geometry(subjects_dir, subject, hemi, name)


def load_curvature(subjects_dir, subject, hemi, n_points):
    """Read ``{hemi}.curv``; returns ``None`` (with a warning) if unusable."""
    fname = path.join(subjects_dir, subject, "surf", f"{hemi}.curv")
    if not path.isfile(fname):
        warnings.warn(f"No curvature file at {fname}; using flat grey.")
        return None
    curv = nib.freesurfer.read_morph_data(fname)
    if curv.size != n_points:
        warnings.warn(
            f"{fname}: {curv.size} values for {n_points} vertices; using flat grey."
        )
        return None
    return curv


def _make_polydata(coords, faces):
    faces = np.asarray(faces, dtype=np.int64)
    padded = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(np.ascontiguousarray(coords, dtype=np.float32), padded)


# ---------------------------------------------------------------------------#
# ROI borders
# ---------------------------------------------------------------------------#


def _parse_polylines(poly):
    """Split a ``PolyData``'s line connectivity into per-polyline index arrays."""
    lines = np.asarray(poly.lines)
    segs, i = [], 0
    while i < len(lines):
        n = int(lines[i])
        segs.append(np.asarray(lines[i + 1 : i + 1 + n], dtype=np.int64))
        i += 1 + n
    return segs


def _smooth_polylines(poly, iterations=2, factor=0.5):
    """Laplacian-smooth each polyline in place along its own connectivity.

    The marching-squares contour follows mesh edges, which makes it visibly
    stair-stepped.  A couple of passes of this removes that without rounding
    the corners of the ROI away.  Closed loops are handled cyclically; the
    endpoints of open strips are pinned.
    """
    if iterations <= 0 or poly.n_points == 0:
        return poly

    pts = np.asarray(poly.points, dtype=float).copy()
    segs = [s for s in _parse_polylines(poly) if len(s) >= 3]
    if not segs:
        return poly

    for _ in range(int(iterations)):
        new = pts.copy()
        for idx in segs:
            closed = idx[0] == idx[-1]
            ring = idx[:-1] if closed else idx
            if len(ring) < 3:
                continue
            p = pts[ring]
            if closed:
                lap = 0.5 * (np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0)) - p
            else:
                lap = np.zeros_like(p)
                lap[1:-1] = 0.5 * (p[:-2] + p[2:]) - p[1:-1]
            new[ring] = p + float(factor) * lap
        pts = new

    out = poly.copy()
    out.points = pts
    return out


def roi_contour(mesh, vertex_idx, normals, offset=0.6, smooth=2):
    """Border of an ROI as smooth polylines lifted off the surface.

    Drawn with a screen-space line width, so the stroke stays crisp and
    constant at any zoom or render scale, unlike a border painted onto mesh
    vertices (see :func:`roi_border_vertices`).

    ``offset`` is required, not cosmetic: without it the surface occludes most
    of the line.
    """
    idx = np.asarray(vertex_idx).ravel().astype(np.int64)
    idx = idx[(idx >= 0) & (idx < mesh.n_points)]
    if idx.size == 0:
        return None

    roi = np.zeros(mesh.n_points, dtype=np.float32)
    roi[idx] = 1.0

    work = mesh.copy(deep=False)
    work.point_data.clear()
    work.point_data["roi"] = roi
    cont = work.contour([0.5], scalars="roi")
    if cont.n_points == 0:
        return None

    try:
        cont = cont.strip(join=True)
    except Exception:  # older/newer pyvista, or already stripped
        pass

    if smooth:
        cont = _smooth_polylines(cont, iterations=smooth)

    if offset:
        from scipy.spatial import cKDTree

        # contour vertices sit on mesh edges, so the nearest mesh vertex's
        # normal is a good enough direction to push along
        _, nn = cKDTree(np.asarray(mesh.points)).query(np.asarray(cont.points))
        cont.points = np.asarray(cont.points) + normals[nn] * float(offset)

    return cont


def roi_border_vertices(faces, roi, borders=True):
    """ROI vertices adjacent to the ROI edge, for ``style="painted"``.

    Reproduces pysurfer's border look, where the line is a band of mesh
    vertices and so scales with the geometry rather than the screen.

    ``borders`` may be an int giving the number of face-ring dilations;
    ``True`` means one, after which the result is restricted back to the ROI.
    """
    from scipy import sparse

    faces = np.asarray(faces, dtype=np.int64)
    roi = np.asarray(roi).astype(bool)
    n = roi.size
    a, b, c = faces.T
    ones = np.ones(len(faces))
    E = (
        sparse.coo_matrix((ones, (a, b)), shape=(n, n))
        + sparse.coo_matrix((ones, (b, c)), shape=(n, n))
        + sparse.coo_matrix((ones, (c, a)), shape=(n, n))
    )
    E = (E + E.T).tocoo()
    keep = np.unique(E.row[roi[E.row] != roi[E.col]])
    if isinstance(borders, int) and not isinstance(borders, bool):
        n_dilate = int(borders)
    else:
        n_dilate = 1 if borders else 0
    for _ in range(n_dilate):
        m = np.isin(faces.ravel(), keep).reshape(faces.shape)
        keep = np.unique(faces[np.any(m, axis=1)])
    if n_dilate:
        keep = keep[np.isin(keep, np.flatnonzero(roi))]
    return keep


# ---------------------------------------------------------------------------#
# colorbars
# ---------------------------------------------------------------------------#


def _fmt_tick(v):
    """One decimal at most, without a trailing '.0' (10 -> '10', 3.333 -> '3.3')."""
    return f"{round(float(v), 1):g}"


def make_colorbar(
    style,
    cmap,
    vmin=0.0,
    vmax=1.0,
    figsize=(2.5, 2.5),
    fontsize=12,
    ticks=None,
    label=None,
):
    """Build one of the lab's standard publication colorbars.

    Ported from ``~/ownCloud/txt/Vorlagen/colorbars/create_colorwheels_pyhton.py``
    with the geometry and styling kept, but the figure size, font size and tick
    values parameterised instead of hard-coded.

    Styles
    ------
    ``wheel_ecc``
        Radial polar wheel, colour = radius.  Three radius annotations at
        1/3, 2/3 and 1 of ``vmax``.  Used for eccentricity.
    ``wheel_pol``
        Full colour wheel via ``ColorbarBase`` on a polar axis.  Used for polar
        angle -- a linear bar for a cyclic quantity hides the fact that 0 and
        2*pi are the same colour.
    ``bar_h`` / ``bar_v``
        Plain horizontal / vertical colorbar.  Used for sigma, variance
        explained and custom arrays.

    Returns the matplotlib ``Figure``; the caller saves it.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    style = str(style).lower()

    if style == "wheel_ecc":
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=figsize)
        N = 100j
        rr, tt = np.mgrid[0:1:N, 0 : 2 * np.pi : N]
        # rasterized: a smooth ramp does not need ~10k vector patches
        ax.pcolormesh(
            tt,
            rr,
            rr / rr.max(),
            cmap=cmap,
            shading="nearest",
            edgecolors="face",
            rasterized=True,
        )
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_yticks([1 / 3, 2 / 3, 1])
        vals = ticks if ticks is not None else [vmax / 3, 2 * vmax / 3, vmax]
        shift = 0.07
        for r, v in zip((1 / 3, 2 / 3, 1), vals):
            ax.text(
                np.pi / 2,
                r + shift,
                _fmt_tick(v),
                ha="center",
                va="center",
                fontsize=fontsize,
            )
        if label:
            ax.set_xlabel(label, fontsize=fontsize)
        return fig

    if style == "wheel_pol":
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection="polar")
        norm = mpl.colors.Normalize(vmin, vmax)
        cb = mpl.colorbar.ColorbarBase(
            ax, cmap=cmap, norm=norm, orientation="horizontal"
        )
        cb.outline.set_visible(False)
        ax.set_axis_off()
        ax.grid(False)
        # no tight_layout here: a polar axes is not compatible with it and
        # matplotlib warns. bbox_inches="tight" at save time does the job.
        return fig

    if style in ("bar_h", "bar_v"):
        vertical = style == "bar_v"
        fig = plt.figure(figsize=figsize)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = (
            fig.add_axes([0.1, 0.1, 0.08, 0.8])
            if vertical
            else fig.add_axes([0.1, 0.45, 0.8, 0.09])
        )
        cbar = fig.colorbar(
            sm,
            cax=cax,
            orientation="vertical" if vertical else "horizontal",
        )
        if ticks is not None:
            cbar.set_ticks(list(ticks))
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.ax.yaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda v, _pos: _fmt_tick(v))
        )
        cbar.ax.xaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda v, _pos: _fmt_tick(v))
        )
        if label:
            cbar.set_label(label, fontsize=fontsize)
        return fig

    raise ValueError(
        f"Unknown colorbar style {style!r}; expected one of "
        "'wheel_ecc', 'wheel_pol', 'bar_h', 'bar_v'."
    )


# ---------------------------------------------------------------------------#
# the Brain shim
# ---------------------------------------------------------------------------#

_RASTER_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
_VECTOR_EXT = {".pdf", ".svg", ".eps", ".ps"}


@dataclass
class _DataLayer:
    array: np.ndarray
    cmap: object
    vmin: float
    vmax: float
    alpha: float = 1.0
    colorbar: bool = True
    n_colors: int = 256
    nan_color: tuple = (0, 0, 0)
    title: str = ""


@dataclass
class _LabelLayer:
    vertices: np.ndarray
    color: object = "black"
    alpha: float = 1.0
    borders: object = True
    style: str = "contour"
    line_width: float = 4.0
    offset: float = None
    smooth: int = 2
    name: str = ""


class Brain:
    """Drop-in replacement for ``surfer.Brain``, rendered with PyVista.

    Parameters mirror pysurfer's where they overlap.  Extra keyword-only
    parameters control the things pysurfer could not do.
    """

    def __init__(
        self,
        subject_id,
        hemi,
        surf,
        subjects_dir=None,
        background="black",
        foreground=None,
        size=(800, 800),
        cortex="classic",
        alpha=1.0,
        *,
        offscreen=True,
        smooth_shading=True,
        lighting="three lights",
        anti_aliasing="ssaa",
        specular=0.0,
        ambient=0.25,
        diffuse=0.75,
        very_inflated_t=0.35,
        notebook=False,
        title=None,
    ):
        if subjects_dir is None:
            raise ValueError("subjects_dir is required")

        self.subject_id = subject_id
        self.hemi = hemi if hemi.endswith("h") else f"{hemi[0].lower()}h"
        self.surf = _SURF_ALIASES.get(str(surf).lower(), surf)
        self.subjects_dir = str(subjects_dir)
        self.alpha = float(alpha)

        self._background = background
        self._foreground = (
            foreground if foreground is not None else _contrast(background)
        )
        self._size = _as_size(size)
        self._cortex = cortex
        self._offscreen = bool(offscreen)
        self._smooth_shading = bool(smooth_shading)
        self._lighting = lighting
        self._anti_aliasing = anti_aliasing
        self._notebook = notebook
        self._title = title or f"{subject_id} {self.hemi} {self.surf}"
        # matte by default: a specular highlight blows out the light grey gyri
        # and shifts overlay hues. Raise `specular` for a glossier look.
        self._shade = dict(
            specular=float(specular), ambient=float(ambient), diffuse=float(diffuse)
        )

        coords, faces = resolve_surface(
            self.subjects_dir,
            subject_id,
            self.hemi,
            self.surf,
            very_inflated_t=very_inflated_t,
        )
        self.mesh = _make_polydata(coords, faces)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.n_vertices = self.mesh.n_points

        curv = load_curvature(self.subjects_dir, subject_id, self.hemi, self.n_vertices)
        self.geo = {"coords": coords, "faces": self.faces, "curv": curv}
        if curv is not None:
            self.mesh.point_data["bin_curv"] = (curv > 0).astype(np.float32)

        # point normals, used for smooth shading and for lifting border lines
        self._normals = np.asarray(
            self.mesh.compute_normals(
                point_normals=True,
                cell_normals=False,
                split_vertices=False,
                consistent_normals=False,
                auto_orient_normals=False,
                inplace=False,
            )["Normals"]
        )

        self._data_layers: list[_DataLayer] = []
        self._label_layers: list[_LabelLayer] = []
        self._colorbar_in_scene = False

        self._camera = vtk.vtkCamera()
        self._camera_initialised = False

    # -- layers -----------------------------------------------------------
    def add_data(
        self,
        array,
        min=None,
        max=None,
        colormap="auto",
        alpha=1.0,
        smoothing_steps=None,
        colorbar=True,
        remove_existing=False,
        *,
        nan_color=(0, 0, 0),
        n_colors=256,
        title="",
        **kwargs,
    ):
        """Add a scalar overlay.  NaN vertices become fully transparent.

        ``smoothing_steps`` is accepted for signature compatibility and
        ignored: pysurfer only built a smoothing matrix when the array was
        shorter than the vertex count, which never happened here.
        """
        if remove_existing:
            self._data_layers.clear()

        arr = np.asarray(array, dtype=np.float32)
        if arr.size != self.n_vertices:
            raise ValueError(
                f"data has {arr.size} values but the surface has "
                f"{self.n_vertices} vertices"
            )
        if colormap == "auto" or colormap is None:
            colormap = "rainbow_r"

        vmin = float(np.nanmin(arr)) if min is None else float(min)
        vmax = float(np.nanmax(arr)) if max is None else float(max)

        self._data_layers.append(
            _DataLayer(
                array=arr,
                cmap=colormap,
                vmin=vmin,
                vmax=vmax,
                alpha=float(alpha),
                colorbar=bool(colorbar),
                n_colors=int(n_colors),
                nan_color=tuple(nan_color[:3]),
                title=title,
            )
        )

    def add_label(
        self,
        label,
        color=None,
        alpha=1.0,
        borders=False,
        *,
        style="contour",
        line_width=4.0,
        offset=None,
        smooth=2,
        **kwargs,
    ):
        """Draw an ROI, or just its border when ``borders`` is truthy.

        ``label`` may be an object with a ``.vertices`` attribute (PRFclass'
        ``label`` class), a plain array of vertex indices, or a path to a
        FreeSurfer ``.label`` file.
        """
        verts = _label_vertices(label)
        self._label_layers.append(
            _LabelLayer(
                vertices=verts,
                color="black" if color is None else color,
                alpha=float(alpha),
                borders=borders,
                style=style,
                line_width=float(line_width),
                offset=offset,
                smooth=int(smooth),
                name=getattr(label, "name", ""),
            )
        )

    def remove_data(self, hemi=None):
        self._data_layers.clear()

    def remove_labels(self, labels=None, hemi=None):
        self._label_layers.clear()

    # -- camera -----------------------------------------------------------
    def show_view(self, view=None, roll=None, distance=None, **kwargs):
        """Set or read the camera, using mayavi's spherical convention.

        Called with no arguments it returns ``((az, el, dist, fp), roll)`` --
        pysurfer's contract.
        """
        if view is None and roll is None:
            return camera_to_mayavi_view(self._camera)

        if isinstance(view, dict):
            az = view.get("azimuth", 0.0)
            el = view.get("elevation", 90.0)
            dist = view.get("distance", distance if distance else 400.0)
            fp = view.get("focalpoint", (0.0, 0.0, 0.0))
        elif view is not None:
            az, el, dist, fp = view
        else:
            (az, el, dist, fp), _ = camera_to_mayavi_view(self._camera)

        mayavi_view_to_camera(self._camera, az, el, dist, fp, roll=roll)
        self._camera_initialised = True
        return camera_to_mayavi_view(self._camera)

    @property
    def camera_state(self) -> CameraState:
        return CameraState.from_camera(self._camera, window_size=self._size)

    @camera_state.setter
    def camera_state(self, state):
        if isinstance(state, dict):
            state = CameraState(**state)
        state.to_camera(self._camera)
        self._camera_initialised = True

    @property
    def camera_position(self):
        return [
            tuple(self._camera.GetPosition()),
            tuple(self._camera.GetFocalPoint()),
            tuple(self._camera.GetViewUp()),
        ]

    # -- rendering --------------------------------------------------------
    def _build_plotter(self, off_screen, window_size=None):
        ws = _as_size(window_size or self._size)
        p = pv.Plotter(
            off_screen=off_screen,
            window_size=list(ws),
            lighting=self._lighting,
            title=self._title,
            # pyvista's default auto-detects the kernel and renders inline in
            # Jupyter, where show() returns immediately instead of blocking and
            # interactive positioning would silently do nothing
            notebook=self._notebook,
        )
        p.set_background(self._background)
        if self._anti_aliasing:
            try:
                p.enable_anti_aliasing(self._anti_aliasing)
            except Exception as exc:  # pragma: no cover - driver dependent
                warnings.warn(f"anti-aliasing {self._anti_aliasing!r} failed: {exc}")

        # --- layer 0: binarised curvature ------------------------------
        if "bin_curv" in self.mesh.point_data:
            prm = (
                _CORTEX_PRESETS[self._cortex]
                if isinstance(self._cortex, str)
                else self._cortex
            )
            p.add_mesh(
                self.mesh,
                scalars="bin_curv",
                cmap=prm["cmap"],
                clim=prm["clim"],
                n_colors=256,
                interpolate_before_map=False,
                show_scalar_bar=False,
                smooth_shading=self._smooth_shading,
                opacity=self.alpha,
                reset_camera=False,
                name="curvature",
                **self._shade,
            )
        else:
            p.add_mesh(
                self.mesh,
                color=(0.5, 0.5, 0.5),
                show_scalar_bar=False,
                smooth_shading=self._smooth_shading,
                opacity=self.alpha,
                reset_camera=False,
                name="curvature",
                **self._shade,
            )

        # --- layer 1..n: data overlays ---------------------------------
        for i, lay in enumerate(self._data_layers):
            dm = self.mesh.copy(deep=False)
            dm.point_data.clear()
            dm.point_data[f"data{i}"] = lay.array
            p.add_mesh(
                dm,
                scalars=f"data{i}",
                cmap=lay.cmap,
                clim=(lay.vmin, lay.vmax),
                n_colors=lay.n_colors,
                nan_color=lay.nan_color,
                nan_opacity=0.0,
                interpolate_before_map=False,
                opacity=lay.alpha,
                smooth_shading=self._smooth_shading,
                show_scalar_bar=False,
                reset_camera=False,
                name=f"data{i}",
                **self._shade,
            )
            if lay.colorbar and self._colorbar_in_scene:
                p.add_scalar_bar(
                    title=lay.title,
                    color=self._foreground,
                    n_labels=5,
                    vertical=False,
                    width=0.6,
                    height=0.07,
                    position_x=0.2,
                    position_y=0.03,
                    # printf format, so a trailing .0 cannot be stripped here
                    fmt="%.1f",
                )

        # --- layer n+1..: ROI borders ----------------------------------
        default_offset = _BORDER_OFFSET.get(self.surf, 0.5)
        for j, lab in enumerate(self._label_layers):
            off = default_offset if lab.offset is None else lab.offset
            if lab.borders and lab.style == "contour":
                cont = roi_contour(
                    self.mesh,
                    lab.vertices,
                    self._normals,
                    offset=off,
                    smooth=lab.smooth,
                )
                if cont is None:
                    continue
                p.add_mesh(
                    cont,
                    color=lab.color,
                    opacity=lab.alpha,
                    line_width=lab.line_width,
                    render_lines_as_tubes=True,
                    lighting=False,
                    show_scalar_bar=False,
                    reset_camera=False,
                    name=f"label{j}",
                )
            else:
                roi = np.zeros(self.n_vertices, dtype=bool)
                idx = np.asarray(lab.vertices).ravel().astype(np.int64)
                roi[idx[(idx >= 0) & (idx < self.n_vertices)]] = True
                if lab.borders:
                    keep = roi_border_vertices(self.faces, roi, lab.borders)
                else:
                    keep = np.flatnonzero(roi)
                paint = np.full(self.n_vertices, np.nan, dtype=np.float32)
                paint[keep] = 1.0
                lm = self.mesh.copy(deep=False)
                lm.point_data.clear()
                lm.point_data[f"label{j}"] = paint
                rgb = mcolors.to_rgb(lab.color)
                p.add_mesh(
                    lm,
                    scalars=f"label{j}",
                    cmap=mcolors.ListedColormap([rgb]),
                    clim=(0.0, 1.0),
                    nan_opacity=0.0,
                    interpolate_before_map=False,
                    opacity=lab.alpha,
                    smooth_shading=self._smooth_shading,
                    show_scalar_bar=False,
                    reset_camera=False,
                    name=f"label{j}",
                    **self._shade,
                )

        # --- camera, last, so add_mesh cannot reset it -----------------
        if self._camera_initialised:
            p.camera.DeepCopy(self._camera)
            p.renderer.ResetCameraClippingRange()
            p.renderer.camera_set = True
        else:
            p.reset_camera()
            self._camera.DeepCopy(p.camera)
            self._camera_initialised = True

        return p

    def show(self, title=None, block=True):
        """Open an interactive window and block until the user closes it.

        The camera the user ends on is absorbed into this ``Brain``, so the
        next ``save_image`` uses it.  Returns ``show_view()``'s tuple.
        """
        if self._offscreen:
            warnings.warn("Brain.show() called with offscreen=True; no window opened.")
            return camera_to_mayavi_view(self._camera)

        p = self._build_plotter(off_screen=False)
        p.add_text(
            "position the brain, then close the window",
            position="upper_left",
            font_size=10,
            color=self._foreground,
        )
        try:
            p.show(title=title or self._title, auto_close=False, interactive=True)
            self._camera.DeepCopy(p.camera)
            self._camera_initialised = True
        finally:
            p.close()
        return camera_to_mayavi_view(self._camera)

    def screenshot(self, scale=None, transparent_background=None, window_size=None):
        p = self._build_plotter(off_screen=True, window_size=window_size)
        try:
            return p.screenshot(
                None,
                scale=scale,
                return_img=True,
                transparent_background=bool(transparent_background),
            )
        finally:
            p.close()

    def save_image(
        self,
        filename,
        mode="rgb",
        antialiased=True,
        *,
        scale=4,
        dpi=300,
        transparent_background=None,
        window_size=None,
    ):
        """Save the current scene.

        Raster extensions are written directly at ``scale`` x the window size.
        Vector extensions (pdf/svg/eps) wrap that high-resolution raster at
        ``dpi``, which is what you want for smooth-shaded surfaces --
        ``Plotter.save_graphic`` (GL2PS) would emit a >100 MB file with Gouraud
        banding for a 300k-triangle mesh.
        """
        if transparent_background is None:
            transparent_background = mode == "rgba"

        ext = path.splitext(str(filename))[1].lower()
        p = self._build_plotter(off_screen=True, window_size=window_size)
        try:
            if ext in _RASTER_EXT:
                p.screenshot(
                    str(filename),
                    scale=scale,
                    transparent_background=bool(transparent_background),
                )
            elif ext in _VECTOR_EXT:
                img = p.screenshot(
                    None,
                    scale=scale,
                    return_img=True,
                    transparent_background=bool(transparent_background),
                )
                _embed_raster(img, str(filename), dpi=dpi)
            else:
                raise ValueError(
                    f"Unsupported output format {ext!r}; expected one of "
                    f"{sorted(_RASTER_EXT | _VECTOR_EXT)}"
                )
        finally:
            p.close()

    def save_colorbar(self, filename, style="auto", *, param=None, dpi=300, **kwargs):
        """Write a standalone colorbar for the topmost data layer."""
        if not self._data_layers:
            raise RuntimeError("no data layer to build a colorbar from")
        lay = self._data_layers[-1]
        if style == "auto":
            style = colorbar_style_for(param, lay.cmap)
        fig = make_colorbar(style, lay.cmap, vmin=lay.vmin, vmax=lay.vmax, **kwargs)
        fig.savefig(str(filename), dpi=dpi, bbox_inches="tight", transparent=True)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def close(self):
        self._data_layers.clear()
        self._label_layers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ---------------------------------------------------------------------------#
# helpers
# ---------------------------------------------------------------------------#


def colorbar_style_for(param, cmap=None):
    """Pick the colorbar shape that suits a parameter."""
    if isinstance(param, str):
        p = param.lower()
        if p in ("r", "ecc", "eccentricity", "radius"):
            return "wheel_ecc"
        if p in ("pol", "polar", "polar angle"):
            return "wheel_pol"
    return "bar_v"


def _label_vertices(label):
    if hasattr(label, "vertices"):
        return np.asarray(label.vertices).ravel().astype(np.int64)
    if isinstance(label, (str, bytes)) or hasattr(label, "__fspath__"):
        return (
            np.asarray(nib.freesurfer.read_label(str(label))).ravel().astype(np.int64)
        )
    return np.asarray(label).ravel().astype(np.int64)


def _as_size(size):
    if np.isscalar(size):
        return (int(size), int(size))
    return (int(size[0]), int(size[1]))


def _contrast(background):
    try:
        r, g, b = mcolors.to_rgb(background)
    except ValueError:
        return "white"
    return "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.5 else "white"


def _embed_raster(img, filename, dpi=300):
    """Write a vector container whose only content is a high-resolution raster.

    Physical page size is pixels/dpi, so the effective resolution really is
    ``dpi``.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    img = np.asarray(img)
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / float(dpi), h / float(dpi)), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(img, interpolation="none")
    fig.savefig(
        filename,
        dpi=dpi,
        transparent=(img.ndim == 3 and img.shape[2] == 4),
        pad_inches=0,
    )
    plt.close(fig)
