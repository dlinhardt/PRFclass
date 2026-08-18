"""Support for hand-drawn (manual) ROI delineations.

Where the atlas delineation is not good enough, ROIs can be drawn by hand and
exported as FreeSurfer-style label GIFTIs (``lh.V1.label.gii``). Dropping those
into

    <derivatives>/manual_rois/<sub-id>/lh.V1.label.gii

and calling :func:`sync_manual_rois` writes them into the prfprepare ROI pack
(``all_roi_masks.h5``) as an atlas named ``manual``, after which everything that
already works for an atlas works for them too::

    ana.maskROI("V1", "manual")
    ana.plot_toSurface("ecc", showBordersAtlas="manual", showBordersArea="V1")

``maskROI`` also ingests them on first use, so the explicit call is optional.

The label files stay the source of truth: prfprepare regenerating the pack drops
the added atlas, and the next :func:`sync_manual_rois` puts it back.
"""

import hashlib
import json
from datetime import datetime, timezone
from glob import glob
from os import path, stat

import nibabel as nib
import numpy as np

from ._version import __version__

try:
    from .roipack.roipack import RoiPack  # preferred (package style)
except Exception:
    try:
        from roipack.roipack import RoiPack  # fallback if installed in path
    except Exception:
        RoiPack = None

#: Atlas name the hand-drawn ROIs live under inside the pack.
MANUAL_ATLAS = "manual"

#: NIFTI intent codes: label data, and the two geometry ones we never want.
_INTENT_LABEL = 1002
_INTENT_GEOMETRY = (1008, 1009)

_HEMI_TOKENS = {
    "lh": "l",
    "l": "l",
    "left": "l",
    "rh": "r",
    "r": "r",
    "right": "r",
}

#: Extensions stripped from a label file name, longest first.
_GII_EXTS = (".label.gii", ".dlabel.gii", ".shape.gii", ".func.gii", ".gii")

#: Label table names that carry no information about the ROI.
_EMPTY_LABEL_NAMES = ("", "???", "unknown", "none", "background", "other")

_SURFACE_SPACES = ("fsnative", "fsaverage", "surface")


# ----------------------------------HELPERS-----------------------------------#
def _sha1_12(fpath):
    """Short content hash of a file, used to notice edited label files.

    Args:
        fpath (str): File to hash.

    Returns:
        str: First 12 hex characters of the SHA1.
    """
    h = hashlib.sha1()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _parse_label_filename(fname):
    """Guess hemisphere and ROI name from a label GIFTI file name.

    Handles the FreeSurfer style (``lh.V1.label.gii``, ``r.hV4.label.gii``) and
    BIDS-ish entity names (``sub-01_hemi-L_desc-V1_dseg.label.gii``).

    Args:
        fname (str): File name or path.

    Returns:
        tuple: ``(hemi, roi)`` with hemi in {'l', 'r'}; either may be None when
            it could not be determined.
    """
    base = path.basename(str(fname))

    stem = None
    for ext in _GII_EXTS:  # longest first, so .label.gii wins over .gii
        if base.endswith(ext):
            stem = base[: -len(ext)]
            break
    if stem is None:
        return None, None

    # --- BIDS entities ------------------------------------------------------
    if "hemi-" in stem:
        ents = {}
        for tok in stem.split("_"):
            k, sep, v = tok.partition("-")
            if sep:
                ents[k] = v
        hemi = _HEMI_TOKENS.get(ents.get("hemi", "").lower())
        if hemi:
            roi = ents.get("desc") or ents.get("label") or ents.get("roi")
            return hemi, (roi or None)

    # --- FreeSurfer <hemi>.<roi>[.<more>] ----------------------------------
    # partition, never split(".")[1]: that breaks lh.hV4 and lh.V1.thresh
    head, sep, rest = stem.partition(".")
    hemi = _HEMI_TOKENS.get(head.lower())
    if hemi and sep and rest:
        roi = rest[: -len(".label")] if rest.endswith(".label") else rest
        return hemi, (roi or None)

    return None, None


def _read_label_gifti(fpath, n_vertices, roi_hint=None, split_multilabel=True):
    """Read a label GIFTI into one or more vertex index arrays.

    Two on-disk layouts are supported and told apart by length, which is the
    only reliable signal:

    * dense -- one value per vertex of the hemisphere (what ``mris_convert``
      and most delineation tools write). Membership is any nonzero value, not
      specifically 1, and distinct nonzero values are distinct ROIs.
    * sparse -- the array *is* the list of vertex indices.

    Args:
        fpath (str): Path to the label GIFTI.
        n_vertices (int): Vertex count of the full hemisphere mesh.
        roi_hint (str, optional): ROI name from the file name, used when the
            label table carries no usable name.
        split_multilabel (bool, optional): Return one entry per label value.
            When False, all nonzero values are merged into one ROI.

    Returns:
        tuple: ``(rois, info)`` where rois is ``{roi_name: sorted unique vertex
            indices}`` and info holds the detected ``layout``, the label
            ``values`` used and the ``table`` that was found.

    Raises:
        ValueError: for an unreadable, empty, all-zero or wrong-sized file.
    """
    img = nib.load(str(fpath))
    name = path.basename(str(fpath))

    darrays = [d for d in img.darrays if d.data is not None and np.size(d.data)]
    # Never pick up geometry, in case a .surf.gii was handed to us by mistake
    non_geom = [
        d for d in darrays if int(getattr(d, "intent", 0)) not in _INTENT_GEOMETRY
    ]
    darrays = non_geom or darrays
    if not darrays:
        raise ValueError(f"{name}: no usable data array in this GIFTI")

    # Prefer the actual label array when several are present
    labelled = [d for d in darrays if int(getattr(d, "intent", 0)) == _INTENT_LABEL]
    dat = np.squeeze(np.asarray((labelled or darrays)[0].data))
    if dat.ndim > 1:
        dat = dat[:, 0]
    if dat.ndim != 1:
        raise ValueError(f"{name}: unexpected data shape {dat.shape}")

    n = int(dat.size)
    integral = bool(np.isfinite(dat).all()) and bool(np.allclose(dat, np.rint(dat)))
    vals = np.rint(dat).astype(np.int64) if integral else None

    # --- layout detection --------------------------------------------------
    if n == n_vertices:
        layout = "dense"  # a length match is unambiguous
    elif n > n_vertices:
        raise ValueError(
            f"{name}: {n} values for a {n_vertices}-vertex hemisphere. Wrong "
            "hemisphere, or an fsaverage label given to an fsnative analysis?"
        )
    elif not integral:
        raise ValueError(
            f"{name}: {n} non-integer values, which is neither a per-vertex map "
            f"of {n_vertices} vertices nor a list of vertex indices"
        )
    elif vals.min() >= 0 and vals.max() < n_vertices and np.unique(vals).size == n:
        layout = "sparse"  # in range and no repeats -> index list
    else:
        # Not a per-vertex map (wrong length) and not an index list (repeated or
        # out-of-range values). Guessing here would hand back indices into the
        # wrong mesh, which is worse than stopping, so refuse.
        raise ValueError(
            f"{name}: {n} values for a {n_vertices}-vertex hemisphere. That is "
            "neither a per-vertex label map (which would need exactly "
            f"{n_vertices} values) nor a list of vertex indices (whose values "
            f"would have to be unique and below {n_vertices}). Wrong mesh "
            "(fsnative vs fsaverage), wrong hemisphere, or wrong subject?"
        )

    if layout == "sparse":
        return (
            {roi_hint or "roi": np.unique(vals)},
            {"layout": layout, "values": None, "table": {}},
        )

    # --- dense: pull the ROIs out of the label map -------------------------
    table = {}
    try:
        table = {
            int(k): str(v) for k, v in (img.labeltable.get_labels_as_dict() or {}).items()
        }
    except Exception:
        table = {}

    present = np.unique(vals[vals != 0])
    if present.size == 0:
        raise ValueError(f"{name}: the label map is all zeros, nothing to ingest")

    if not split_multilabel:
        return (
            {roi_hint or "roi": np.where(vals != 0)[0]},
            {"layout": layout, "values": present.tolist(), "table": table},
        )

    out = {}
    for value in present:
        tname = table.get(int(value), "")
        if tname.strip().lower() in _EMPTY_LABEL_NAMES:
            # no usable name in the file, fall back to the file name
            tname = (
                roi_hint
                if (present.size == 1 and roi_hint)
                else f"{roi_hint or 'label'}{int(value)}"
            )
        out[tname] = np.where(vals == value)[0]

    return out, {"layout": layout, "values": present.tolist(), "table": table}


def _label_to_pack_indices(label_vertices, flat_index):
    """Map original-space vertex indices onto the analysed (masked) space.

    This is the whole bridge between the surface the ROI was drawn on and the
    space the pRF parameters live in. ``flat_index`` maps masked position ->
    original vertex and is sorted, so searchsorted gives the positions exactly;
    np.isin would only give membership.

    Label vertices outside the analysed space are dropped, because the pack
    requires the two index spaces to stay positionally aligned. That happens
    whenever prfprepare cropped the analysis to an atlas; with a full_brain
    analysis ``flat_index`` is the identity and nothing is lost.

    Args:
        label_vertices (array-like): Vertex indices of the drawn ROI.
        flat_index (array-like): The pack's ``flat_index_<hemi>``, sorted.

    Returns:
        tuple: ``(orig_kept, masked_kept, n_dropped)`` with
            ``flat_index[masked_kept] == orig_kept``, both int32 and strictly
            increasing, which is the alignment the pack relies on.
    """
    v = np.unique(np.asarray(label_vertices, dtype=np.int64))
    flat = np.asarray(flat_index, dtype=np.int64)

    if flat.size == 0 or v.size == 0:
        return (
            np.empty(0, np.int32),
            np.empty(0, np.int32),
            int(v.size),
        )

    pos = np.clip(np.searchsorted(flat, v), 0, flat.size - 1)
    hit = flat[pos] == v
    return (
        v[hit].astype(np.int32),
        pos[hit].astype(np.int32),
        int((~hit).sum()),
    )


def _manual_roi_dir(self, manual_roi_dir=None):
    """Directory holding this subject's hand-drawn label GIFTIs.

    Subject level on purpose: a hand-drawn ROI follows the subject's anatomy and
    does not depend on session, task or run.

    Args:
        manual_roi_dir (str, optional): Explicit override.

    Returns:
        str: The directory (which need not exist).
    """
    if manual_roi_dir is not None:
        return str(manual_roi_dir)
    return path.join(self._derivatives_path, "manual_rois", self._subject)


def _require_surface_space(self, action="hand-drawn ROIs"):
    """Raise unless this analysis lives on the surface.

    Args:
        action (str, optional): What was attempted, for the message.

    Raises:
        ValueError: For a volume analysis, where a surface label has no meaning.
    """
    space = str(self.analysisSpace).lower()
    if space not in _SURFACE_SPACES:
        raise ValueError(
            f"{action} need a surface analysis, but {self._subject} "
            f"prfprepare analysis-{self.prfprepare_analysis} is '{self.analysisSpace}'. "
            "Sample the label into the volume grid yourself, or rerun prfprepare "
            "on the surface."
        )


def _roipack_dir(self, prfprepare_analysis=None):
    """Directory of the ROI pack for a given prfprepare analysis."""
    ana = self.prfprepare_analysis if prfprepare_analysis is None else prfprepare_analysis
    return path.join(
        self._derivatives_path, "prfprepare", f"analysis-{ana}", self._subject
    )


def _open_roipack(self, prfprepare_analysis=None):
    """Open this subject's ROI pack.

    Args:
        prfprepare_analysis (str, optional): Analysis id, defaults to this
            object's own.

    Returns:
        tuple: ``(RoiPack, roipack_dir)``.

    Raises:
        RuntimeError: When the roipack submodule is missing or too old to write.
        FileNotFoundError: When the pack does not exist.
    """
    if RoiPack is None:
        raise RuntimeError(
            "the roipack submodule is not available; run\n"
            "  git submodule update --init roipack"
        )

    rp_dir = _roipack_dir(self, prfprepare_analysis)
    h5_file = path.join(rp_dir, "all_roi_masks.h5")
    meta_file = path.join(rp_dir, "all_roi_masks_meta.json")
    if not path.isfile(h5_file) or not path.isfile(meta_file):
        raise FileNotFoundError(
            f"no ROI pack for {self._subject} at {h5_file}; "
            "run prfprepare for this analysis first"
        )

    with open(meta_file, "r") as f:
        meta = json.load(f)
    rp = RoiPack(key=meta["key"], h5_path=h5_file, meta=meta)
    rp.select_grid_for_input(self.analysisSpace)

    if not hasattr(rp, "write_surface_rois"):
        raise RuntimeError(
            "the roipack submodule is too old to write ROIs; run\n"
            "  git submodule update --init --remote roipack"
        )
    return rp, rp_dir


def _manifest_path(self, prfprepare_analysis=None):
    """Path of the manual ROI manifest next to the pack."""
    return path.join(_roipack_dir(self, prfprepare_analysis), "manual_rois.json")


def _manifest_read(self, prfprepare_analysis=None):
    """Read the manual ROI manifest, or an empty dict when there is none."""
    p = _manifest_path(self, prfprepare_analysis)
    if not path.isfile(p):
        return {}
    try:
        with open(p, "r") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _manifest_write(self, manifest, prfprepare_analysis=None):
    """Write the manual ROI manifest next to the pack."""
    p = _manifest_path(self, prfprepare_analysis)
    with open(p, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    return p


def _hemi_vertex_count(self, rp, hemi):
    """Vertex count of a full hemisphere mesh.

    Needed to tell a dense per-vertex label map from a list of vertex indices.
    Prefers the pack (authoritative and free), then the fmriprep GIFTI or the
    FreeSurfer white surface.

    Args:
        rp (RoiPack): The opened pack.
        hemi (str): 'l' or 'r'.

    Returns:
        int: Number of vertices in that hemisphere.
    """
    n = rp.hemi_vertex_count(hemi)
    if n:
        return int(n)

    from ._plotstuff import _resolve_surface_hemi_size

    return int(_resolve_surface_hemi_size(self, hemi.upper()))


def _provenance(fpath, layout, label_values, n_label, n_kept, n_dropped):
    """Attributes recording where an ingested ROI came from.

    Args:
        fpath (str): The source label file.
        layout (str): 'dense' or 'sparse'.
        label_values (list or None): Label values found in the file.
        n_label (int): Vertices in the delineation.
        n_kept (int): Vertices inside the analysed space.
        n_dropped (int): Vertices dropped outside the analysed space.

    Returns:
        dict: HDF5-writable attributes.
    """
    st = stat(fpath)
    return {
        "source": "manual",
        "source_file": path.abspath(fpath),
        "source_mtime": float(st.st_mtime),
        "source_size": int(st.st_size),
        "source_sha1": _sha1_12(fpath),
        "source_layout": str(layout),
        "source_label_values": json.dumps(label_values),
        "n_label_vertices": int(n_label),
        "n_kept": int(n_kept),
        "n_dropped_outside_mask": int(n_dropped),
        "ingested_by": "PRFclass",
        "ingested_version": str(__version__),
        "ingested_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _pack_stat(rp):
    """Size and mtime of the pack file, to notice that it was rewritten."""
    st = stat(str(rp.h5_path))
    return {"mtime_ns": int(st.st_mtime_ns), "size": int(st.st_size)}


# ----------------------------------DISCOVERY---------------------------------#
def manual_roi_files(self, manual_roi_dir=None, verbose=True):
    """Find this subject's hand-drawn label GIFTIs, without touching the pack.

    Args:
        manual_roi_dir (str, optional): Look here instead of the default
            ``<derivatives>/manual_rois/<sub-id>/``.
        verbose (bool, optional): Print what was found.

    Returns:
        dict: ``{(roi, hemi): filepath}``, hemi lowercase.
    """
    d = _manual_roi_dir(self, manual_roi_dir)
    hits = sorted(glob(path.join(d, "*.gii")))

    found, skipped = {}, []
    for f in hits:
        hemi, roi = _parse_label_filename(f)
        if hemi is None or roi is None:
            skipped.append(f)
            continue
        if "/" in roi:
            skipped.append(f)
            continue
        key = (roi, hemi)
        if key in found:
            print(
                f"[WARNING] two files for {roi}/{hemi}, keeping "
                f"{path.basename(found[key])} and ignoring {path.basename(f)}"
            )
            continue
        found[key] = f

    if verbose:
        print(f"\nmanual ROIs for {self._subject} in {d}")
        if not path.isdir(d):
            print("  directory does not exist")
        elif not found:
            print("  no label GIFTIs found")
        else:
            for roi, hemi in sorted(found):
                print(f"  {roi:<8s} {hemi}h  {path.basename(found[(roi, hemi)])}")
        for f in skipped:
            print(f"  [skipped, cannot parse name] {path.basename(f)}")

    return found


# ----------------------------------INGEST------------------------------------#
def add_manual_rois(
    self,
    files=None,
    atlas=MANUAL_ATLAS,
    roi_names=None,
    manual_roi_dir=None,
    prfprepare_analysis=None,
    split_multilabel=True,
    min_vertices=1,
    overwrite=True,
    dry_run=False,
    verbose=True,
):
    """Read hand-drawn label GIFTIs and write them into the ROI pack.

    They become a normal atlas inside ``all_roi_masks.h5``, so afterwards
    ``maskROI("V1", "manual")``, the ROI borders and the flatmaps all work
    unchanged.

    Vertices of the delineation that fall outside the analysed space are
    dropped, because the pack requires the two index spaces to stay aligned.
    The counts are printed and recorded, and the full delineation is kept
    separately so the plotted borders still show it. With a full_brain
    prfprepare analysis nothing is dropped at all.

    Args:
        files (list, optional): Explicit label files. None searches for them.
        atlas (str, optional): Atlas name the ROIs appear under.
        roi_names (dict, optional): ``{file basename: roi name}`` override for
            names the file name parser gets wrong.
        manual_roi_dir (str, optional): Where to look for the label files.
        prfprepare_analysis (str or list, optional): Which pack(s) to write to.
            Defaults to this object's own analysis; 'all' writes to every
            prfprepare analysis that has a pack for this subject.
        split_multilabel (bool, optional): Treat distinct label values in one
            file as distinct ROIs.
        min_vertices (int, optional): Skip an ROI with fewer surviving vertices.
        overwrite (bool, optional): Replace an existing manual atlas. When
            False and the atlas exists, raises instead.
        dry_run (bool, optional): Report what would be written, write nothing.
        verbose (bool, optional): Print a per-ROI report.

    Returns:
        dict: ``{analysis: {"rois": {(roi, hemi): counts}, "manifest": ...}}``.

    Raises:
        ValueError: For a volume analysis, or an ROI entirely outside the
            analysed space, or an existing atlas with overwrite=False.
    """
    _require_surface_space(self, "hand-drawn ROIs")

    # --- which packs to write to ------------------------------------------
    if prfprepare_analysis is None:
        analyses = [self.prfprepare_analysis]
    elif isinstance(prfprepare_analysis, str) and prfprepare_analysis.lower() == "all":
        analyses = sorted(
            path.basename(path.dirname(p)).replace("analysis-", "")
            for p in glob(
                path.join(
                    self._derivatives_path,
                    "prfprepare",
                    "analysis-*",
                    self._subject,
                    "all_roi_masks.h5",
                )
            )
        )
        if not analyses:
            raise FileNotFoundError(
                f"no ROI pack found for {self._subject} in any prfprepare analysis"
            )
    elif isinstance(prfprepare_analysis, (list, tuple)):
        analyses = list(prfprepare_analysis)
    else:
        analyses = [prfprepare_analysis]

    # --- collect the label files ------------------------------------------
    if files is None:
        found = manual_roi_files(self, manual_roi_dir, verbose=False)
    else:
        found = {}
        for f in files if isinstance(files, (list, tuple)) else [files]:
            hemi, roi = _parse_label_filename(f)
            if hemi is None:
                raise ValueError(
                    f"cannot tell the hemisphere from {path.basename(str(f))}; "
                    "name it like lh.V1.label.gii"
                )
            found[(roi, hemi)] = str(f)

    if not found:
        d = _manual_roi_dir(self, manual_roi_dir)
        raise FileNotFoundError(
            f"no manual ROIs for {self._subject}: put lh.V1.label.gii / "
            f"rh.V1.label.gii into {d}"
        )

    out = {}
    for ana in analyses:
        out[str(ana)] = _add_manual_rois_to_pack(
            self,
            found=found,
            atlas=atlas,
            roi_names=roi_names or {},
            prfprepare_analysis=ana,
            split_multilabel=split_multilabel,
            min_vertices=min_vertices,
            overwrite=overwrite,
            dry_run=dry_run,
            verbose=verbose,
        )
    return out


def _add_manual_rois_to_pack(
    self,
    found,
    atlas,
    roi_names,
    prfprepare_analysis,
    split_multilabel,
    min_vertices,
    overwrite,
    dry_run,
    verbose,
):
    """Ingest the given label files into one prfprepare analysis' pack.

    Everything is read and mapped before anything is written, so a bad file
    cannot leave a half-written atlas behind.
    """
    rp, rp_dir = _open_roipack(self, prfprepare_analysis)

    if atlas in rp.list_atlases() and not overwrite:
        raise ValueError(
            f"atlas '{atlas}' already exists in {rp.h5_path}; "
            "pass overwrite=True to replace it"
        )

    # --- per hemisphere context -------------------------------------------
    flats, n_verts = {}, {}
    for _, hemi in found:
        if hemi in flats:
            continue
        flats[hemi] = rp.get_masked_space_flat_index(hemi).astype(np.int64)
        n_verts[hemi] = _hemi_vertex_count(self, rp, hemi)

    cropped = {h: len(flats[h]) < n_verts[h] for h in flats}
    if verbose:
        print(f"\ningesting manual ROIs into analysis-{prfprepare_analysis}")
        for h in sorted(flats):
            if cropped[h]:
                print(
                    f"  pack is cropped: {len(flats[h])} of {n_verts[h]} {h}h vertices "
                    "are in the analysed space"
                )
            else:
                print(
                    f"  pack covers the full mesh: {n_verts[h]} {h}h vertices, "
                    "nothing will be dropped"
                )

    # --- read and map everything first ------------------------------------
    specs, report, problems = [], {}, []
    for (roi_hint, hemi), fpath in sorted(found.items()):
        override = roi_names.get(path.basename(fpath)) or roi_names.get(fpath)
        try:
            rois, info = _read_label_gifti(
                fpath,
                n_verts[hemi],
                roi_hint=override or roi_hint,
                split_multilabel=split_multilabel,
            )
        except ValueError as e:
            problems.append(str(e))
            continue

        for roi, verts in rois.items():
            if override:
                roi = override
            if roi != roi_hint and not override and len(rois) == 1:
                print(
                    f"[WARNING] {path.basename(fpath)} is named for '{roi_hint}' but "
                    f"its label table says '{roi}'; using '{roi}'"
                )

            orig, masked, dropped = _label_to_pack_indices(verts, flats[hemi])

            if orig.size == 0:
                problems.append(
                    f"{roi}/{hemi}: {len(verts)} delineated vertices, none of them "
                    f"inside the analysed space of analysis-{prfprepare_analysis} "
                    f"({len(flats[hemi])} of {n_verts[hemi]} vertices). The "
                    "delineation is on a different surface, the wrong hemisphere, "
                    "or entirely outside the analysed mask."
                )
                continue
            if orig.size < min_vertices:
                print(
                    f"[WARNING] {roi}/{hemi}: only {orig.size} vertices survive, "
                    f"below min_vertices={min_vertices}, skipping"
                )
                continue

            prov = _provenance(
                fpath, info["layout"], info["values"], len(verts), orig.size, dropped
            )
            specs.append(
                {
                    "atlas": atlas,
                    "roi": roi,
                    "hemi": hemi,
                    "original_indices": orig,
                    "masked_indices": masked,
                    "n_vertices": n_verts[hemi],
                    "provenance": prov,
                    # keep the unclipped delineation for the plotted borders
                    "full_extent": np.unique(np.asarray(verts, dtype=np.int64)),
                }
            )
            report[(roi, hemi)] = {
                "file": fpath,
                "layout": info["layout"],
                "n_label": int(len(verts)),
                "n_kept": int(orig.size),
                "n_dropped": int(dropped),
                "source_mtime": prov["source_mtime"],
                "source_size": prov["source_size"],
                "source_sha1": prov["source_sha1"],
            }

            if verbose:
                pct = 100.0 * dropped / max(len(verts), 1)
                tail = (
                    f" ({dropped} dropped, {pct:.1f} %)"
                    if dropped
                    else "  (nothing dropped)"
                )
                print(
                    f"  {roi:<8s} {hemi}h  {info['layout']:<6s} {len(verts):6d} "
                    f"delineated -> {orig.size:6d} in the analysed space{tail}"
                )

    if problems:
        raise ValueError(
            "nothing was written, because:\n  " + "\n  ".join(problems)
        )
    if not specs:
        raise ValueError("no manual ROIs survived reading; nothing to write")

    if any(r["n_dropped"] for r in report.values()) and verbose:
        full = _full_brain_analyses(self)
        hint = f" (here analysis-{'/'.join(full)})" if full else ""
        print(
            "  -> part of the delineation lies outside the analysed space. For the "
            f"full hand delineation use a full_brain prfprepare analysis{hint}."
        )

    if dry_run:
        if verbose:
            print(f"  dry run, nothing written to {rp.h5_path}")
        return {"rois": report, "manifest": None, "written": 0, "dry_run": True}

    # --- write, as one atomic batch ---------------------------------------
    if atlas in rp.list_atlases():
        rp.drop_atlas(atlas, rebuild_index=False)
    n = rp.write_surface_rois(specs)

    manifest = {
        "schema": "prfclass_manual_rois_v1",
        "atlas": atlas,
        "prfclass_version": str(__version__),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pack": {**_pack_stat(rp), "key": rp.key},
        "rois": {},
    }
    for (roi, hemi), r in report.items():
        manifest["rois"].setdefault(roi, {})[hemi] = r
    mpath = _manifest_write(self, manifest, prfprepare_analysis)

    if verbose:
        n_rois = len({roi for roi, _ in report})
        print(
            f"  wrote atlas '{atlas}' ({n_rois} ROIs, {n} hemispheres) into "
            f"{rp.h5_path}"
        )
        print(f"  manifest: {mpath}")

    return {"rois": report, "manifest": manifest, "written": n, "dry_run": False}


def _full_brain_analyses(self):
    """prfprepare analyses of this subject whose pack covers the full mesh.

    Returns:
        list: Analysis ids, sorted.
    """
    out = []
    for meta_file in sorted(
        glob(
            path.join(
                self._derivatives_path,
                "prfprepare",
                "analysis-*",
                self._subject,
                "all_roi_masks_meta.json",
            )
        )
    ):
        try:
            with open(meta_file, "r") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            continue
        if "full_brain" in (meta.get("atlases") or []):
            ana = path.basename(
                path.dirname(path.dirname(meta_file))
            ).replace("analysis-", "")
            out.append(ana)
    return out


# ----------------------------------SYNC--------------------------------------#
def _manual_rois_state(self, atlas=MANUAL_ATLAS, manual_roi_dir=None,
                       prfprepare_analysis=None):
    """Work out whether the pack still matches the label files on disk.

    Args:
        atlas (str, optional): Atlas name to check.
        manual_roi_dir (str, optional): Where the label files live.
        prfprepare_analysis (str, optional): Which pack to check.

    Returns:
        dict: ``{"ok": bool, "reasons": [...], "files": {...}, "orphans": [...]}``.
    """
    files = manual_roi_files(self, manual_roi_dir, verbose=False)
    manifest = _manifest_read(self, prfprepare_analysis)
    reasons, orphans = [], []

    if not files:
        return {"ok": True, "reasons": ["no label files"], "files": {}, "orphans": []}

    if not manifest:
        return {
            "ok": False,
            "reasons": ["no manifest, never ingested"],
            "files": files,
            "orphans": [],
        }

    try:
        rp, _ = _open_roipack(self, prfprepare_analysis)
    except (FileNotFoundError, RuntimeError) as e:
        return {"ok": False, "reasons": [str(e)], "files": files, "orphans": []}

    # 2) the atlas vanished -> prfprepare regenerated the pack
    if atlas not in rp.list_atlases():
        reasons.append(f"atlas '{atlas}' is missing from the pack")
    else:
        for roi, hemis in (manifest.get("rois") or {}).items():
            for hemi in hemis:
                if not rp.has_roi(atlas, roi, hemi):
                    reasons.append(f"{roi}/{hemi} is missing from the pack")

    # 3) the pack was rewritten under us
    pack_now = _pack_stat(rp)
    pack_was = manifest.get("pack") or {}
    if pack_was and (
        pack_now["mtime_ns"] != pack_was.get("mtime_ns")
        or pack_now["size"] != pack_was.get("size")
    ):
        reasons.append("the pack file changed since the last ingest")

    # Match on the source file, not on the ROI name: a label table may name the
    # ROI differently than the file does, and then a name-based comparison would
    # call every ingest stale forever.
    by_src = {}
    for roi, hemis in (manifest.get("rois") or {}).items():
        for hemi, rec in hemis.items():
            by_src[path.abspath(str(rec.get("file", "")))] = (roi, hemi, rec)

    # 4/5) source files changed, or are new
    for (roi, hemi), fpath in files.items():
        entry = by_src.get(path.abspath(fpath))
        if entry is None:
            reasons.append(f"{path.basename(fpath)} is not in the manifest")
            continue
        rec = entry[2]
        st = stat(fpath)
        if int(st.st_size) != int(rec.get("source_size", -1)) or not np.isclose(
            float(st.st_mtime), float(rec.get("source_mtime", -1)), atol=1e-6
        ):
            if _sha1_12(fpath) != rec.get("source_sha1"):
                reasons.append(f"{path.basename(fpath)} changed")

    # 6) manifest entries whose source is gone
    on_disk = {path.abspath(f) for f in files.values()}
    for src, (roi, hemi, rec) in by_src.items():
        if src not in on_disk:
            orphans.append((roi, hemi, src))

    return {
        "ok": not reasons,
        "reasons": reasons,
        "files": files,
        "orphans": orphans,
    }


def sync_manual_rois(
    self,
    atlas=MANUAL_ATLAS,
    manual_roi_dir=None,
    prfprepare_analysis=None,
    force=False,
    prune=False,
    verbose=True,
):
    """Make the ROI pack match the hand-drawn labels on disk. Idempotent.

    Cheap and safe to call at the top of a script: it only re-ingests when the
    pack was regenerated by prfprepare or a label file changed, and reports
    'up to date' otherwise. ``maskROI("V1", "manual")`` calls this for you the
    first time, so the explicit call is optional.

    Args:
        atlas (str, optional): Atlas name the ROIs live under.
        manual_roi_dir (str, optional): Where the label files live.
        prfprepare_analysis (str or list, optional): Which pack(s) to sync.
        force (bool, optional): Re-ingest even when nothing changed.
        prune (bool, optional): Drop ROIs whose label file has disappeared.
        verbose (bool, optional): Print what happened.

    Returns:
        dict: ``{"action": 'none'|'ingested'|'skipped'|'nothing to do',
            "reasons": [...]}``.
    """
    try:
        _require_surface_space(self, "hand-drawn ROIs")
    except ValueError as e:
        # never break a group-wide call over a mixed study
        if verbose:
            print(f"skipping manual ROIs for {self._subject}: {e}")
        return {"action": "skipped", "reasons": [str(e)]}

    state = _manual_rois_state(self, atlas, manual_roi_dir, prfprepare_analysis)

    if not state["files"]:
        if verbose:
            print(
                f"no manual label files for {self._subject} in "
                f"{_manual_roi_dir(self, manual_roi_dir)}"
            )
        return {"action": "nothing to do", "reasons": state["reasons"]}

    if state["orphans"]:
        for roi, hemi, src in state["orphans"]:
            print(
                f"[WARNING] {roi}/{hemi} is ingested but its label file is gone ({src})"
                + ("; dropping it" if prune else "; keeping it, pass prune=True to drop")
            )

    if state["ok"] and not force and not (prune and state["orphans"]):
        if verbose:
            print(f"manual ROIs up to date for {self._subject}")
        return {"action": "none", "reasons": []}

    if verbose and state["reasons"]:
        print("re-ingesting manual ROIs because:")
        for r in state["reasons"]:
            print(f"  {r}")

    res = add_manual_rois(
        self,
        atlas=atlas,
        manual_roi_dir=manual_roi_dir,
        prfprepare_analysis=prfprepare_analysis,
        verbose=verbose,
    )
    return {"action": "ingested", "reasons": state["reasons"], "result": res}


# ----------------------------------REPORT------------------------------------#
def check_manual_rois(
    self,
    atlas=MANUAL_ATLAS,
    manual_roi_dir=None,
    prfprepare_analysis=None,
    verbose=True,
):
    """Report the hand-drawn ROIs in the pack and how they line up.

    Prints per ROI and hemisphere how much of the delineation is inside the
    analysed space, where it came from and whether the source has changed
    since, and re-checks the pack's ``flat_index[masked] == original``
    invariant.

    Args:
        atlas (str, optional): Atlas name to report on.
        manual_roi_dir (str, optional): Where the label files live.
        prfprepare_analysis (str, optional): Which pack to report on.
        verbose (bool, optional): Print the report.

    Returns:
        dict: ``{(roi, hemi): {...}}`` with the counts and provenance.
    """
    try:
        rp, _ = _open_roipack(self, prfprepare_analysis)
    except (FileNotFoundError, RuntimeError) as e:
        if verbose:
            print(f"[WARNING] {e}")
        return {}

    files = manual_roi_files(self, manual_roi_dir, verbose=False)
    ana = (
        self.prfprepare_analysis if prfprepare_analysis is None else prfprepare_analysis
    )

    if atlas not in rp.list_atlases():
        if verbose:
            print(
                f"\nno '{atlas}' atlas in the analysis-{ana} pack for {self._subject}"
            )
            if files:
                n = len(files)
                print(
                    f"  {n} label file(s) are waiting in "
                    f"{_manual_roi_dir(self, manual_roi_dir)}; "
                    "run ana.sync_manual_rois()"
                )
        return {}

    out = {}
    if verbose:
        print(f"\n{'='*78}")
        print(
            f"MANUAL ROIs for {self._subject}  [atlas '{atlas}', "
            f"prfprepare analysis-{ana}]"
        )
        print(f"{'='*78}")
        print(
            f"  {'roi':<8s} {'h':<2s} {'drawn':>7s} {'in space':>9s} {'kept':>6s} "
            f"{'source':<24s} status"
        )

    for roi in rp.list_rois(atlas):
        for hemi in rp.list_hemispheres(atlas, roi):
            attrs = rp.list_roi_attrs(atlas, roi, hemi)
            orig = rp.get_original_space_indices(atlas, roi, hemi)
            masked = rp.get_masked_space_indices(atlas, roi, hemi)
            flat = rp.get_masked_space_flat_index(hemi)

            aligned = len(orig) == len(masked) and np.array_equal(
                np.asarray(flat)[np.asarray(masked)], np.asarray(orig)
            )
            n_label = int(attrs.get("n_label_vertices", len(orig)))
            status = [] if aligned else ["MISALIGNED"]

            # Follow the recorded source file rather than the ROI name, since a
            # label table can name an ROI differently than its file does.
            src = attrs.get("source_file")
            src = src.decode() if isinstance(src, bytes) else src
            if not src or not path.isfile(src):
                status.append("source missing")
            else:
                sha = attrs.get("source_sha1")
                sha = sha.decode() if isinstance(sha, bytes) else sha
                if _sha1_12(src) != sha:
                    status.append("source changed")

            rec = {
                "n_label_vertices": n_label,
                "n_kept": int(len(orig)),
                "n_dropped_outside_mask": int(attrs.get("n_dropped_outside_mask", 0)),
                "aligned": bool(aligned),
                "source_file": src,
                "ingested_utc": attrs.get("ingested_utc"),
                "has_full_extent": rp.has_full_extent(atlas, roi, hemi),
                "status": status or ["ok"],
            }
            out[(roi, hemi)] = rec

            if verbose:
                ing = rec["ingested_utc"]
                ing = ing.decode() if isinstance(ing, bytes) else (ing or "?")
                print(
                    f"  {roi:<8s} {hemi}h {n_label:7d} {len(orig):9d} "
                    f"{100.0*len(orig)/max(n_label,1):5.1f}% {str(ing)[:24]:<24s} "
                    f"{', '.join(rec['status'])}"
                )

    if verbose:
        n_drop = sum(r["n_dropped_outside_mask"] for r in out.values())
        if n_drop:
            print(
                f"\n  {n_drop} vertices in total lie outside the analysed space and "
                "are not in the mask."
            )
            print(
                "  The plotted borders still show the full delineation "
                "(stored separately)."
            )
        print(f"{'='*78}\n")

    return out
