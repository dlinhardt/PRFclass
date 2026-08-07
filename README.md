# PRFclass

## Surface visualisation

Cortical surfaces are rendered with **PyVista/VTK**; flatmaps use **pycortex**.
mayavi and pysurfer are no longer used — do not reinstall them, as having PyQt5
alongside PyQt6 in one environment causes the `QT_ROOT_LEVEL_POOL is implemented
in both libQt5Core and libQt6Core` warning, and mayavi's off-screen renderer
needs OSMesa and so does not work on macOS at all.

### Install

Dependencies are in `environment.yml` (env `david`): `pyvista` and `vtk` from
conda-forge, `pycortex` from pip.

Headless Linux (cluster) needs an off-screen VTK build:

```bash
pip uninstall vtk
pip install --extra-index-url https://wheels.vtk.org vtk-osmesa   # CPU
pip install --extra-index-url https://wheels.vtk.org vtk-egl      # GPU
```

or just run under `xvfb-run -a python your_script.py`. Note that
`pv.start_xvfb()` was removed in pyvista 0.48 — do not call it. On macOS nothing
extra is needed.

### 3D surfaces — `plot_toSurface()`

```python
prf.plot_toSurface(param="pol", hemi="both", surface="inflated",
                   save=True, headless=True, output_format="pdf")
```

**Surfaces.** Any native FreeSurfer surface (`pial`, `white`, `smoothwm`,
`inflated`, `sphere`, …) plus:

| `surface=` | what you get |
|---|---|
| `grey` / `gray` | the pial surface |
| `cortex` | inflated (the historical default) |
| `midthickness` / `mid` | `0.5 * (white + pial)`, computed on the fly |
| `very_inflated` | the file if it exists, otherwise inflated blended towards a radius-matched sphere (`very_inflated_t`, default 0.35). FreeSurfer has no very_inflated surface, so this is an approximation. |

**Resolution.** `scale=4` (default) renders at 4× `window_size`, i.e. 3200×3200.
Raster formats are written directly; `pdf`/`svg`/`eps` wrap that raster at `dpi`
(default 300). Vector export via GL2PS is deliberately not used — for a
300k-triangle mesh it produces a >100 MB file with Gouraud banding.

**ROI borders** are true contour lines drawn at a screen-space width, so they
stay a crisp 2 px stroke at any zoom or scale. Tune with `border_width`,
`border_color`, `border_smooth`, `border_offset`. The old pysurfer look (painted
mesh vertices — thick and stair-stepped) is still available via
`border_style="painted"` for reproducing earlier figures.

**Colorbars** go to a separate `*_colorbar.svg` by default
(`colorbar_style="auto"`) so the brain image stays clean: a radial wheel for
eccentricity, a colour wheel for polar angle (a linear bar would hide that 0 and
2π are the same colour), a vertical bar otherwise. Use `colorbar_style="scene"`
for the old in-image bar, or `"none"`.

**Camera positions** are stored per subject, hemisphere, area *and surface* in
`derivatives/prfresult/positioning/`. Files written by the old mayavi code are
still read. Run once with `headless=False` to pick a view interactively; in
headless mode a missing position falls back to the default view and is not
persisted, so you can still position it later.

### Flatmaps — `plot_toFlatmap()`

```python
prf.plot_toFlatmap(param="pol", showBordersAtlas="benson", showBordersArea="V1",
                   save=True)
```

Uses the same parameters, masking, colormaps and colorbars as
`plot_toSurface`. Subjects are imported into pycortex's filestore automatically
on first use, from surfaces only: `cortex.freesurfer.import_subj` cannot be used
because it requires `mri/wm.mgz`, which fsaverage does not ship, and it writes
fiducials back into `$SUBJECTS_DIR`, which is usually read-only. Volume-based
pycortex features (`cortex.Volume`, transforms) are therefore unavailable for
auto-imported subjects — run `import_subj` yourself if you need them.

Flatmaps need a flattened surface. fsaverage ships one; for individual subjects:

```python
prf.prepare_flatmap_surface()           # runs mris_flatten, hours per hemisphere
prf.prepare_flatmap_surface(run=False)  # only print the commands, e.g. for a cluster
```

That needs an existing `?h.cortex.patch.3d` cut; where to cut is a manual
decision and is not automated. Both naming conventions are accepted —
FreeSurfer's `?h.<patch>.patch.flat` and pycortex's `?h.<patch>.flat.patch.3d`.

### Tests

```bash
pytest PRFclass/tests/test_surfaceplot.py -q
```
