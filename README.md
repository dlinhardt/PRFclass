# PRFclass

A Python interface for working with **population receptive field (pRF)** results:
loading them from disk, masking them down to the voxels/vertices you care about,
computing derived measures, and plotting them — in the visual field, on the
cortical surface, or on a flatmap.

Two classes do the work:

| class | scope |
|---|---|
| `PRF` | one subject / session / task / run |
| `PRFgroup` | many of them at once, held in a pandas DataFrame |

`PRFgroup` mirrors the `PRF` interface, so the same masking calls and the same
parameter names work whether you are looking at one dataset or a whole study.

---

## Where the data comes from

PRFclass is built to read the output of
**[prfprepare](https://github.com/fmriat/prfprepare) → prfanalyze**. It works
with **[GEM-pRF](https://github.com/fmriat/gemprf)** (`method="gem"`) and with
vistasoft (`method="vista"`), and it also reads mrVista sessions, SAMSrf
results, plain HDF5 estimates and loose files through the other constructors.

The expected layout is the standard BIDS derivatives tree:

```
<baseP>/<study>/
└── derivatives
    ├── prfprepare
    │   └── analysis-01
    └── prfanalyze-gem          # or prfanalyze-vista, …
        └── analysis-01
            └── sub-<subject>/ses-<session>/
                └── sub-<subject>_ses-<session>_task-<task>_run-<run>_hemi-<L|R>_estimates.h5
```

A `<study>/BIDS/derivatives` layout is picked up automatically as a fallback.
The ROI masks and atlases written by prfprepare are read through the
[roipack](https://github.com/dlinhardt/roipack) submodule, which is what makes
`maskROI("V1", "benson")` and the ROI borders in the surface plots work.

### Scope

This package was developed at the Medical University of Vienna, around our data
organisation and our file server — that is where the default `baseP` and a few
path conventions come from. It is not tied to it, though: **if your data went
through prfprepare, PRFclass should work for you** by passing your own `baseP`
and `study`. The Vienna-specific parts are defaults, not requirements.

---

## Install

Clone the repository *with its submodule* into a directory that is on your
Python path:

```bash
git clone --recursive https://github.com/dlinhardt/PRFclass
```

If you already have a clone without `roipack`:

```bash
git submodule update --init
```

Then import it as a package from the parent directory:

```python
import sys
sys.path.append("/path/to/where/you/cloned")   # the folder *containing* PRFclass

from PRFclass import PRF, PRFgroup
```

### Required libraries

Install these with conda or pip, whichever your environment uses:

**Core** — needed to import the package at all:
`numpy`, `scipy`, `matplotlib`, `pandas`, `nibabel`, `nilearn`, `h5py`,
`pillow`, `tqdm`, `reportlab`, `pyvista`, `vtk`

**Flatmaps only** — `plot_toFlatmap()` and `prepare_flatmap_surface()`:
`pycortex`, `svglib`

**Optional** — `neuropythy` (some atlas paths and SAMSrf data)

Surfaces are rendered with PyVista/VTK, flatmaps with pycortex. **mayavi and
pysurfer are no longer used — do not install them.** PyQt5 next to PyQt6 in one
environment produces the `QT_ROOT_LEVEL_POOL is implemented in both libQt5Core
and libQt6Core` warning, and mayavi's off-screen renderer needs OSMesa, so it
does not work on macOS at all.

**Headless Linux (cluster)** needs an off-screen VTK build:

```bash
pip uninstall vtk
pip install --extra-index-url https://wheels.vtk.org vtk-osmesa   # CPU
pip install --extra-index-url https://wheels.vtk.org vtk-egl      # GPU
```

or run your script under `xvfb-run -a python your_script.py`. Note that
`pv.start_xvfb()` was removed in pyvista 0.48 — do not call it. On macOS nothing
extra is needed.

---

## Quick start

A fuller, commented walkthrough lives in
[PRFclass_example.py](PRFclass_example.py).

### One dataset

```python
from PRFclass import PRF

ana = PRF.from_docker(
    baseP="/path/to/data",
    study="my_study",
    subject="001", session="001", task="bars", run="01",
    method="gem",        # or "vista"
    analysis="01",
    hemi="",             # "" = both hemispheres, or "L" / "R"
)

ana.list_atlases_and_rois()   # see what atlases and ROIs are available

ana.maskVarExp(0.1)           # keep voxels with R² > 0.1
ana.maskROI("V1", "benson")   # …inside V1  (a list of ROIs also works)

ana.x, ana.y, ana.ecc, ana.pol, ana.sigma, ana.varexp   # masked parameters
```

### Hand-drawn ROIs

When the atlas delineation is not good enough for a subject, draw the ROIs by hand and drop
the label GIFTIs into a per-subject folder — subject level, since a hand-drawn ROI follows the
anatomy and not the session or task:

```
<study>/derivatives/manual_rois/sub-01/
├── lh.V1.label.gii
├── rh.V1.label.gii
└── lh.V2.label.gii
```

They then behave like any other atlas, under the name `manual`:

```python
ana.maskROI("V1", "manual")     # ingests them on first use, then just works
ana.plot_toSurface("ecc", hemi="L", showBordersAtlas="manual", showBordersArea="V1")
```

`maskROI` writes them into the prfprepare ROI pack (`all_roi_masks.h5`) the first time it needs
them, so no separate step is required. The explicit calls are there when you want them:

```python
ana.manual_roi_files()      # what is on disk, without touching the pack
ana.sync_manual_rois()      # ingest / re-ingest; idempotent, cheap to call in a script
ana.check_manual_rois()     # per-ROI report: how much of each ROI is inside the analysed space
anas.sync_manual_rois()     # same for a whole PRFgroup, once per subject
```

The label files stay the source of truth. Rerunning prfprepare regenerates the pack and drops
the `manual` atlas; the next `sync_manual_rois()` (or `maskROI(..., "manual")`) notices and puts
it back.

**Use a `full_brain` prfprepare analysis for hand-drawn work.** The pack stores ROIs in the
analysed space, so if prfprepare cropped the analysis to an atlas, the part of your delineation
outside that crop cannot carry pRF parameters and is dropped from the mask. `sync_manual_rois`
reports exactly how many vertices that costs, and `check_manual_rois` keeps the numbers around.
With `atlases: full_brain` every vertex is analysed and nothing is lost. The full delineation is
stored either way, so the ROI borders in the surface and flatmap plots always show what you
actually drew.

Two details worth knowing: `maskROI(area="all", atlas="all")` includes the `manual` atlas, and
where a vertex belongs to both `benson/V1` and `manual/V1` it is attributed to the
alphabetically first atlas, so `_roiWhichAtlas` says `benson`. And hand-drawn ROIs need a
surface analysis — a surface label has no meaning in a volume one.

### A whole study

```python
from PRFclass import PRFgroup

anas = PRFgroup.from_docker(
    baseP="/path/to/data",
    study="my_study",
    subjects=".*", sessions=".*", tasks=".*", runs=".*",   # regex selection
    method="gem",
    prfanalyze="01",
)

anas.maskVarExp(0.1)
anas.maskROI("V1", "benson")

anas.data      # DataFrame: subject, session, task, run, prf
anas.ecc       # all subjects concatenated
anas.sub_ecc   # dict, one entry per subject
```

---

## Three things worth knowing

**Masking is a stack, not a filter.** Nothing is ever thrown away. Each
`maskX()` call defines one mask; `.mask` is the combination of all active ones.
Every mask can be switched off again without reloading:

```python
ana.doVarExpMsk = False   # also doROIMsk, doEccMsk, doSigMsk, doBetaMsk, doManualMsk
```

**A trailing `0` means unmasked.** `ana.x` gives you the masked voxels,
`ana.x0` gives you all of them. This holds for every parameter — `ecc`/`ecc0`,
`pol`/`pol0`, `varexp`/`varexp0`, `voxelTC`/`voxelTC0`, and so on.

**A group is a DataFrame of `PRF` objects.** `anas.data["prf"]` holds the real
`PRF` instances, so anything you can do to one you can do to all of them.
`anas.query("subject == '001'")` returns a filtered `PRFgroup`, and
`anas.iterate("plot_covMap", save=True)` runs a method across the group.

---

## What you can do with it

| | |
|---|---|
| **Load** | prfanalyze results (gem, vistasoft) via Docker/BIDS derivatives, mrVista sessions, SAMSrf, HDF5, or individual files |
| **Mask** | by ROI and atlas, variance explained, eccentricity, pRF size, beta amplitude, a dartboard segmentation, or an interactive hand-drawn mask |
| **Manual ROIs** | ingest hand-drawn `lh.V1.label.gii` delineations into the ROI pack and use them as the `manual` atlas |
| **Parameters** | `x`, `y`, `ecc`, `pol`, `sigma`, `beta`, `varexp` — masked or unmasked, per dataset or pooled across a group |
| **Time courses** | measured BOLD (`voxelTC`), the fitted model signal (`modelsignal`, `modelpred`), and `r2_against()` to score a model against any run |
| **Stimulus & motion** | load the stimulus apertures, eye-tracker jitter, and realignment/framewise-displacement parameters |
| **Analyse** | visual field coverage, KDE-difference maps and central scotoma borders, pRF profiles |
| **Plot** | coverage maps, multi-subject coverage PDFs, 3D cortical surfaces, pycortex flatmaps |
| **Export** | write parameters and masks back out as NIfTI or GIFTI (`save_results`) |

The `gem` results also expose the fitting grid (`searchSpace`) and stimulus
metadata (`stimulusInfo`) — useful, for example, to mask out fits that ran into
the edge of the search space.

---

## Visualisation

### Coverage maps

```python
ana.plot_covMap(method="max", save=True)
anas.plot_coverage_summary_pdf()   # one PDF for the whole group
```

### Cortical surfaces

```python
ana.plot_toSurface(param="pol", hemi="both", surface="inflated",
                   save=True, headless=True, output_format="pdf")
```

Any native FreeSurfer surface works (`pial`, `white`, `inflated`, `sphere`, …),
plus a few convenience names (`midthickness`, `very_inflated`, `grey`,
`cortex`). ROI borders, colormaps and colorbars are configurable; colorbars are
written to a separate file by default so the brain image stays clean.

**Camera positions** are stored per subject, hemisphere, area and surface under
`derivatives/prfresult/positioning/`. Run once with `headless=False` to pick a
view interactively; in headless mode a missing position falls back to a default
view and is not saved, so you can still set it later.

### Flatmaps

```python
ana.plot_toFlatmap(param="pol", showBordersAtlas="benson", showBordersArea="V1",
                   save=True)
```

Same parameters, masking, colormaps and colorbars as `plot_toSurface`. Subjects
are imported into pycortex's filestore automatically on first use, from surfaces
only — so volume-based pycortex features (`cortex.Volume`, transforms) are not
available for auto-imported subjects; run `cortex.freesurfer.import_subj`
yourself if you need them.

Flatmaps need a flattened surface. fsaverage ships one; for individual subjects:

```python
ana.prepare_flatmap_surface()           # runs mris_flatten, hours per hemisphere
ana.prepare_flatmap_surface(run=False)  # only print the commands, e.g. for a cluster
```

This requires an existing `?h.cortex.patch.3d` cut — where to cut is a manual
decision and is not automated.


## Citing PRFclass

If PRFclass contributed to work you publish, please cite it. GitHub's *Cite this
repository* button (top right, from [CITATION.cff](CITATION.cff)) gives you
ready-made APA and BibTeX entries:

```bibtex
@software{prfclass,
  author  = {Linhardt, David},
  title   = {PRFclass: a Python interface for population receptive field results},
  version = {2.1},
  url     = {https://github.com/dlinhardt/PRFclass},
}
```

Please also cite [prfprepare](https://github.com/fmriat/prfprepare) and, if you
used it, [GEM-pRF](https://github.com/fmriat/gemprf).

## License

Apache License 2.0 — see [LICENSE](LICENSE). Do whatever you like with it; the
license asks only that you keep the copyright notice, pass on the
[NOTICE](NOTICE) file, and state any significant changes you made.

Citing is a request, not a license condition: if you use PRFclass in published
work, please cite it as described in [CITATION.cff](CITATION.cff).
