import json
import warnings
from glob import glob
from os import path

import numpy as np
from scipy.io import loadmat

try:
    import neuropythy as ny
    import nibabel as nib
except:
    print("neuropythy not installed, samsrf data not available")

# New ROI pack helper
try:
    from .roipack.roipack import RoiPack  # preferred (package style)
except Exception:
    try:
        from roipack.roipack import RoiPack  # fallback if installed in path
    except Exception:
        RoiPack = None


# ----------------------------------MASKING-----------------------------------#
def maskROI(
    self, area="V1", atlas="benson", doV123=False, forcePath=False, masking_style=None
):
    """
    Masks the data with defined ROIs

    Args:
        area (str or list, optional): Define the areas to load. Only used when docker data. Use 'all' for all available areas. Defaults to 'V1'.
        atlas (str or list, optional): Define atlases to load (benson, wang, custom annots). Only used when docker data. Use 'all' for all available areas.. Defaults to 'benson'.
        doV123 (bool, optional): Load V1, V2 and V3, otherwise only V1. Only used when matlab mrVista data. Defaults to False.
        forcePath (bool, optional): Force path for coords.mat, again?. Only used when matlab mrVista data. Defaults to False.
    """

    if masking_style is None:
        data_from = self._dataFrom
    else:
        data_from = masking_style

    if data_from == "mrVista":
        if isinstance(area, list):
            warnings.warn("You cannot give area lists for mrVista data!", UserWarning)
        if doV123:
            self._roiV1 = loadmat(
                path.join(
                    self._baseP, self._study, "ROIs", self.subject + "_V1_Combined.mat"
                ),
                simplify_cells=True,
            )["ROI"]["coords"].astype("uint16")
            self._roiV2 = loadmat(
                path.join(
                    self._baseP, self._study, "ROIs", self.subject + "_V2_Combined.mat"
                ),
                simplify_cells=True,
            )["ROI"]["coords"].astype("uint16")
            self._roiV3 = loadmat(
                path.join(
                    self._baseP, self._study, "ROIs", self.subject + "_V3_Combined.mat"
                ),
                simplify_cells=True,
            )["ROI"]["coords"].astype("uint16")

            dtypeV1 = {
                "names": ["f{}".format(i) for i in range(3)],
                "formats": 3 * [self._roiV1.dtype],
            }
            dtypeV2 = {
                "names": ["f{}".format(i) for i in range(3)],
                "formats": 3 * [self._roiV2.dtype],
            }
            dtypeV3 = {
                "names": ["f{}".format(i) for i in range(3)],
                "formats": 3 * [self._roiV3.dtype],
            }

            self._roiIndV1 = np.intersect1d(
                self._coords.T.view(dtypeV1).T,
                self._roiV1.T.view(dtypeV1).T,
                return_indices=True,
            )[1]
            self._roiIndV2 = np.intersect1d(
                self._coords.T.view(dtypeV2).T,
                self._roiV2.T.view(dtypeV2).T,
                return_indices=True,
            )[1]
            self._roiIndV3 = np.intersect1d(
                self._coords.T.view(dtypeV3).T,
                self._roiV3.T.view(dtypeV3).T,
                return_indices=True,
            )[1]

            m1 = np.zeros(self.x0.shape)
            m2 = np.zeros(self.x0.shape)
            m3 = np.zeros(self.x0.shape)
            m1[self._roiIndV1] = 1
            m2[self._roiIndV2] = 1
            m3[self._roiIndV3] = 1

            self._roiMsk = np.any((m1, m2, m3), 0)

            self._isROIMasked = 3

        else:
            if forcePath:
                self._roi = (
                    loadmat(forcePath, simplify_cells=True)["ROI"]["coords"]
                ).astype("uint16")
            else:
                if path.isdir(
                    path.join(self._baseP, self._study, "ROIs", self.subject)
                ):
                    self._roi = (
                        loadmat(
                            path.join(
                                self._baseP,
                                self._study,
                                "ROIs",
                                self.subject,
                                self.subject + "_V1.mat",
                            ),
                            simplify_cells=True,
                        )["ROI"]["coords"]
                    ).astype("uint16")
                else:
                    self._roi = (
                        loadmat(
                            path.join(
                                self._baseP,
                                self._study,
                                "ROIs",
                                self.subject + "_V1_Combined.mat",
                            ),
                            simplify_cells=True,
                        )["ROI"]["coords"]
                    ).astype("uint16")

            dtype = {
                "names": ["f{}".format(i) for i in range(3)],
                "formats": 3 * [self._roi.dtype],
            }
            self._roiInd = np.intersect1d(
                self._coords.T.view(dtype).T,
                self._roi.T.view(dtype).T,
                return_indices=True,
            )[1]

            self._roiMsk = np.zeros(self.x0.shape)
            self._roiMsk[self._roiInd] = 1

            # if hasattr(self, 'tValues'): self._tValues = self._tValues[self._msk]

    elif data_from in ["docker", "hdf5"]:
        # Prefer consolidated ROI pack (HDF5 + meta JSON) if present in the subject folder
        roipack_dir = path.join(
            self._derivatives_path,
            "prfprepare",
            f"analysis-{self.prfprepare_analysis}",
            self._subject,
        )
        h5_file = path.join(roipack_dir, "all_roi_masks.h5")
        meta_file = path.join(roipack_dir, "all_roi_masks_meta.json")

        if RoiPack is not None and path.isfile(h5_file) and path.isfile(meta_file):
            # Try new RoiPack-based path; pass space from analysisSpace by default
            _used = _maskROI_from_roipack(
                self,
                area=area,
                atlas=atlas,
                roipack_path=roipack_dir,
                img=None,
            )
            if _used:
                self._isROIMasked = 1
                print("Using RoiPack path for ROI masking")
                return
        print("Falling back to legacy path using the maskinfo.json files")

        # --- Backward compatibility path (legacy maskinfo.json) ---
        self._atlas = atlas if isinstance(atlas, list) else [atlas]
        self._area = area if isinstance(area, list) else [area]

        self._allAreaFiles = sorted(
            glob(
                path.join(
                    self._derivatives_path,
                    "prfprepare",
                    f"analysis-{self.prfprepare_analysis}",
                    self._subject,
                    self._session,
                    "func",
                    f"{self._subject}_{self._session}_hemi-*_desc-*-*_maskinfo.json",
                )
            )
        )

        _allAreas = np.array(
            [
                path.basename(a).split("desc-")[1].split("-")[0]
                for a in self._allAreaFiles
            ]
        )
        _allAtlas = np.unique(
            [
                path.basename(a).split("desc-")[1].split("-")[1].split("_")[0]
                for a in self._allAreaFiles
            ]
        )

        if self._area[0] == "all":
            self._area = _allAreas

        if self._atlas[0] == "all":
            self._atlas = _allAtlas

        self._roiMsk = np.zeros(self.x0.shape)

        if self.analysisSpace == "fsnative":
            roiIndOrigName = "roiIndFsnative"
            roiIndOrigShape = 0
        elif self.analysisSpace == "fsaverage":
            roiIndOrigName = "roiIndFsaverage"
            roiIndOrigShape = 0
        elif self.analysisSpace == "volume":
            roiIndOrigName = "roiPos3D"
            roiIndOrigShape = (0, 3)

        self._roiIndBold = np.empty(0, dtype=int)
        self._roiIndOrig = np.empty(roiIndOrigShape, dtype=int)
        self._roiWhichHemi = np.empty(0)
        self._roiWhichArea = np.empty(0)
        self._areaJsons = []

        hs = ["L", "R"] if self._hemis == "" else [*self._hemis]
        for at in self._atlas:
            for ar in self._area:
                # load the left hemi size if only looking at right
                if not "L" in hs:
                    lHemiSize = 0

                for h in hs:
                    try:
                        areaJson = [
                            j
                            for j in self._allAreaFiles
                            if f"hemi-{h}" in path.basename(j)
                            and f"desc-{ar}-" in path.basename(j)
                            and at in path.basename(j)
                        ][0]
                    except IndexError:
                        continue

                    with open(areaJson, "r") as fl:
                        maskinfo = json.load(fl)

                    self._areaJsons.append(maskinfo)

                    if len(self._roiWhichHemi) == 0:
                        doubleMask = np.ones(len(maskinfo["roiIndBold"]))
                    else:
                        if self.analysisSpace == "volume":
                            doubleMask = np.invert(
                                np.any(
                                    [
                                        np.all(
                                            i
                                            == self._roiIndOrig[
                                                self._roiWhichHemi == h
                                            ],
                                            axis=1,
                                        )
                                        for i in maskinfo[roiIndOrigName]
                                    ],
                                    axis=1,
                                )
                            )
                        else:
                            doubleMask = np.array(
                                [
                                    i not in self._roiIndOrig[self._roiWhichHemi == h]
                                    for i in maskinfo[roiIndOrigName]
                                ]
                            )
                    doubleMask = doubleMask.astype(bool)

                    if len(maskinfo["roiIndBold"]) != len(maskinfo[roiIndOrigName]):
                        print(
                            f"WARNING: ROI {ar} in {at} has different indices for bold and orig!"
                        )
                        continue

                    if h.upper() == "L":
                        self._roiIndBold = np.hstack(
                            (
                                self._roiIndBold,
                                np.array(maskinfo["roiIndBold"])[doubleMask],
                            )
                        )
                        lHemiSize = maskinfo["thisHemiSize"]
                    elif h.upper() == "R":
                        self._roiIndBold = np.hstack(
                            (
                                self._roiIndBold,
                                np.array(maskinfo["roiIndBold"])[doubleMask]
                                + lHemiSize,
                            )
                        )
                    elif h.upper() == "B":
                        self._roiIndBold = np.hstack(
                            (
                                self._roiIndBold,
                                np.array(maskinfo["roiIndBold"])[doubleMask],
                            )
                        )

                    if self.analysisSpace == "volume":
                        self._roiIndOrig = np.vstack(
                            (
                                self._roiIndOrig,
                                np.array(maskinfo[roiIndOrigName])[doubleMask],
                            )
                        )
                    else:
                        self._roiIndOrig = np.hstack(
                            (
                                self._roiIndOrig,
                                np.array(maskinfo[roiIndOrigName])[doubleMask],
                            )
                        )

                    self._roiWhichHemi = np.hstack(
                        (self._roiWhichHemi, np.tile(h, sum(doubleMask)))
                    )

                    self._roiWhichArea = np.hstack(
                        (self._roiWhichArea, np.tile(ar, sum(doubleMask)))
                    )

                    if h.upper() == "L" or h.upper() == "B":
                        self._roiMsk[maskinfo["roiIndBold"]] = 1
                    elif h.upper() == "R":
                        self._roiMsk[np.array(maskinfo["roiIndBold"]) + lHemiSize] = 1

        if self._roiMsk.sum() == 0:
            print(f"WARNING: No data in ROI {self._area} in {self._atlas}!")
            print("Check if the area and atlas are available in the maskinfo files!")

    elif data_from == "samsrf":
        # get the occ mask used in samsrf
        occ_file = path.join(
            self._derivatives_path,
            "samsrf",
            self._prfanaAn,
            self._subject,
            self._session,
            "occ.label",
        )
        occ_label = nib.freesurfer.read_label(occ_file)  # as this is coming from matlab
        # get the mask
        self._atlas = atlas  # if isinstance(atlas, list) else [atlas] # this we need if we want to use list of atlases
        self._area = area  # if isinstance(area, list) else [area]

        if isinstance(atlas, list) or isinstance(area, list):
            raise ValueError("Only one area and one atlas implemented for samsrf data")
        if not "benson" in atlas:
            raise ValueError("Only benson atlas implemented for samsrf data")

        self._area_files_p = sorted(
            glob(
                path.join(
                    self._derivatives_path,
                    "fmriprep",
                    f"analysis-01",
                    "sourcedata",
                    "freesurfer",
                    self._subject,
                    "surf",
                    "*h.benson14_varea.mgz",
                )
            )
        )

        # get the areas and hemis
        area_masks, which_hemi = np.hstack(
            [
                (
                    ll := nib.load(f).get_fdata().squeeze(),
                    np.tile(path.basename(f)[0].upper(), len(ll)),
                )
                for f in self._area_files_p
            ]
        )
        area_masks_occ = area_masks.astype(np.float32).astype(np.int16)[occ_label]

        # load the label area dependency
        mdl = ny.vision.retinotopy_model("benson17", "lh")
        areaLabels = dict(mdl.area_id_to_name)
        areaLabels = {areaLabels[k]: k for k in areaLabels}
        # labelNames = list(areaLabels.keys())

        self._roiMsk = np.zeros(self.x0.shape)
        self._roiMsk[area_masks_occ == areaLabels[area]] = 1

        # define the roiIndOrig and roiIndBold
        self._roiIndOrig = occ_label[self.roiMsk]
        self._roiIndBold = np.where(self.roiMsk)[0]

        # check which hemi
        which_hemi_occ = which_hemi[occ_label]
        self._roiWhichHemi = which_hemi_occ[self.roiMsk]

        # check right hemi and remove len of left hemi
        self._vertices_left_hemi = (which_hemi == "L").sum()
        self._roiIndOrig[self._roiWhichHemi == "R"] -= self._vertices_left_hemi

        self.analysisSpace = "fsnative"

    self._isROIMasked = 1


def _maskROI_from_roipack(self, area, atlas, roipack_path, img=None) -> bool:
    """Internal: fill ROI fields using RoiPack when available.

    Returns True if successfully applied, else False to allow fallback.
    Populates:
      - _roiMsk (bool [n_masked_space])
      - _roiIndBold (int index into masked space ~ self.x0)
      - _roiIndOrig (int per-vertex index for surface; int[:,3] ijk for volume)
      - _roiWhichHemi ("L"/"R" for surface; "B" for volume)
      - _atlas, _area (normalized lists)
    """
    if RoiPack is None:
        return False

    try:
        # Read the all_roi_masks_meta.json
        with open(path.join(roipack_path, "all_roi_masks_meta.json"), "r") as f:
            meta = json.load(f)
        self.rp = RoiPack(
            key=meta["key"],
            h5_path=path.join(roipack_path, "all_roi_masks.h5"),
            meta=meta,
        )
        # Choose grid selection using either provided image or explicit space label
        try:
            if self.analysisSpace == "volume":
                if img is not None:
                    self.rp.select_grid_for_input(img)
                else:
                    self.rp.select_grid_for_input(self.analysisSpace)
            else:
                self.rp.select_grid_for_input(self.analysisSpace)
        except Exception as e:
            print(
                f"RoiPack grid selection failed (analysisSpace={self.analysisSpace}): {e}"
            )
            return False
    except Exception as e:
        print(f"Failed to open ROI pack: {e}")
        return False

    # ---------------- Normalize atlas/area filters ----------------
    self._atlas = atlas if isinstance(atlas, list) else [atlas]
    self._area = area if isinstance(area, list) else [area]

    # Build atlas and area lists (expand "all" if needed)
    available_atlases = self.rp.list_atlases()
    if len(self._atlas) == 1 and str(self._atlas[0]).lower() == "all":
        self._atlas = available_atlases

    all_areas = set()
    for a in self._atlas:
        if a in available_atlases:
            all_areas.update(self.rp.list_rois(a))

    if len(self._area) == 1 and str(self._area[0]).lower() == "all":
        self._area = sorted(list(all_areas))

    # Hemisphere query list (lowercase for RoiPack, uppercase for output)
    hemi_query = ["l", "r"] if self._hemis == "" else [h.lower() for h in self._hemis]

    # Get masked space flat indices for each hemisphere to build total space
    masked_space_indices = {}
    total_masked_len = 0
    hemi_order = []

    for h in hemi_query:
        try:
            flat_idx = self.rp.get_masked_space_flat_index(h)
            masked_space_indices[h] = flat_idx
            total_masked_len += len(flat_idx)
            hemi_order.append(h)
        except Exception as e:
            print(f"Failed to load masked space flat index for hemi {h}: {e}")
            return False

    if total_masked_len != len(self.x0):
        print(
            f"Combined masked space length {total_masked_len} != parameter length {len(self.x0)}; falling back."
        )
        return False

    # Precompute offsets for combined masked space ordering (l then r)
    offsets = {}
    running = 0
    for h in hemi_order:
        offsets[h] = running
        running += len(masked_space_indices[h])

    # Accumulators
    masked_pos_list: list[int] = []
    hemi_list: list[str] = []  # 'L' / 'R'
    area_list: list[str] = []
    atlas_list: list[str] = []
    orig_indices_list: list = []

    # Iterate through requested ROIs
    for a in self._atlas:
        if a not in available_atlases:
            continue
        for r in self._area:
            available_rois = self.rp.list_rois(a)
            if r not in available_rois:
                continue
            for h in hemi_query:
                if not self.rp.has_roi(a, r, h):
                    continue

                try:
                    # Get indices in masked space for this ROI
                    roi_masked_indices = self.rp.get_masked_space_indices(a, r, h)
                    if roi_masked_indices.size == 0:
                        continue

                    # Convert to combined masked space positions (offset by hemisphere)
                    combined_positions = roi_masked_indices + offsets[h]

                    # Get original space indices
                    roi_orig_indices = self.rp.get_original_space_indices(a, r, h)

                    # Store data
                    masked_pos_list.extend(combined_positions.tolist())
                    hemi_list.extend(
                        ["L" if h == "l" else "R"] * len(roi_masked_indices)
                    )
                    area_list.extend([r] * len(roi_masked_indices))
                    atlas_list.extend([a] * len(roi_masked_indices))

                    if self.analysisSpace in ["fsnative", "fsaverage"]:
                        orig_indices_list.extend(roi_orig_indices.tolist())
                    else:  # volume - convert flat indices to (i,j,k) coordinates
                        # We need the volume shape - try to get it from meta or infer
                        try:
                            # Try to get shape from first dense mask
                            dummy_shape = (100, 100, 100)  # fallback
                            if hasattr(self, "_vol_shape"):
                                vol_shape = self._vol_shape
                            else:
                                # Try to infer from meta or use a reasonable default
                                vol_shape = dummy_shape

                            ijk = np.vstack(
                                np.unravel_index(roi_orig_indices, vol_shape)
                            ).T
                            orig_indices_list.extend(
                                [tuple(map(int, row)) for row in ijk]
                            )
                        except Exception:
                            # Fallback: just store flat indices
                            orig_indices_list.extend(roi_orig_indices.tolist())

                except Exception as e:
                    print(f"Failed to process ROI {a}/{r}/{h}: {e}")
                    continue

    if not masked_pos_list:
        print(f"WARNING: No ROI data found for atlas={self._atlas}, area={self._area}")
        return False

    # Remove duplicates while preserving order
    seen = set()
    keep_idx = []
    for i, p in enumerate(masked_pos_list):
        if p not in seen:
            seen.add(p)
            keep_idx.append(i)

    # Build final arrays
    self._roiIndBold = np.asarray([masked_pos_list[i] for i in keep_idx], dtype=int)
    self._roiWhichHemi = np.asarray([hemi_list[i] for i in keep_idx])
    self._roiWhichArea = np.asarray([area_list[i] for i in keep_idx])
    self._roiWhichAtlas = np.asarray([atlas_list[i] for i in keep_idx])

    if self.analysisSpace in ["fsnative", "fsaverage"]:
        self._roiIndOrig = np.asarray(
            [orig_indices_list[i] for i in keep_idx], dtype=int
        )
    else:
        self._roiIndOrig = np.asarray([orig_indices_list[i] for i in keep_idx])

    # Build boolean mask over combined masked space
    roi_mask = np.zeros(total_masked_len, dtype=bool)
    roi_mask[self._roiIndBold] = True
    self._roiMsk = roi_mask

    self._isROIMasked = 1
    return True


# ---------------------------------------------------------------------------#
def maskVarExp(self, varExpThresh, varexp_easy=False, highThresh=None, spmPath=None):
    """
    defines the mask for data points above the given variance explained Threshold

    Args:
        varExpThresh (float): variance explained threshold within [0,1]
        varexp_easy (bool): defined if we should use varexp as from analysis (False) or from calc_varexp_easy method (True)
        highThresh (float, optional): Threshold for masking high VarExp voxels. Defaults to None.
        spmPath (str, optional): Defines an SPM output (fullfield) as mask. Only use when matlab mrVista data. Defaults to None.
    """

    if spmPath:
        if not hasattr(self, "tValues"):
            self.loadSPMmat(spmPath)

        self._spmMask = abs(self._tValues) > varExpThresh

    else:
        ve = self.varexp0 if not varexp_easy else self.varexp_easy0
        if not highThresh:
            with np.errstate(invalid="ignore"):
                self._varExpMsk = np.isfinite(ve) & (ve >= varExpThresh)
        else:
            with np.errstate(invalid="ignore"):
                self._varExpMsk = (
                    np.isfinite(ve) & (ve >= varExpThresh) & (ve <= highThresh)
                )

        # if hasattr(self, 'tValues'): self._tValues = self._tValues[self._msk]

    self._isVarExpMasked = varExpThresh


# ---------------------------------------------------------------------------#
def maskEcc(self, rad):
    """
    mask the data with given eccentricity

    Args:
        rad (float): Threshold
    """

    self._eccMsk = self.r0 < rad

    self._isEccMasked = rad


# ---------------------------------------------------------------------------#
def maskSigma(self, s_min, s_max=None):
    """
    mask the data with given sigma

    Args:
        rad (float): Threshold
    """

    if s_max is not None:
        self._sigMsk = (self.s0 > s_min) & (self.s0 < s_max)
        self._isSigMasked = f"{s_min}_{s_max}"
    else:
        self._sigMsk = self.s0 > s_min
        self._isSigMasked = f"{s_min}"


# ---------------------------------------------------------------------------#
def maskBetaThresh(
    self,
    betaMax=50,
    betaMin=0,
    doBorderThresh=False,
    doLowBetaThresh=True,
    doHighBetaThresh=True,
):
    """
    mask with beta threshold and high sigmas as calculated from macfunc18-c01 & c02 zeroth as below

    Args:
        betaMax (int, optional): Threshold for beta values (above will be excluded). Defaults to 50.
        doBorderThresh (bool, optional): Don't do this. Defaults to False.
        doHighBetaThresh (bool, optional): Threshold for beta values (below will be excluded). Defaults to True.
    """

    self._betaMsk = np.ones(self.x0.shape)

    if doBorderThresh:
        self._betaMsk = np.all(
            (self._betaMsk, self.s0 < self.r0 * 0.26868116 + 1.61949558), 0
        )
    if doHighBetaThresh:
        self._betaMsk = np.all((self._betaMsk, self.beta0 < betaMax), 0)
    if doLowBetaThresh:
        self._betaMsk = np.all((self._betaMsk, self.beta0 > betaMin), 0)

    self._isBetaMasked = (
        2 if (doHighBetaThresh and doBorderThresh) else 1 if doHighBetaThresh else 3
    )


# ---------------------------------------------------------------------------#
def _calcMask(self):
    """
    This convolvs all the defined masks, you can set masks to true or false
    by changing the parameter for:
        self.doROIMsk
        self.doVarExpMsk
        self.doBetaMsk
        self.doEccMsk

    Returns:
        self._mask: the total mask
    """

    self._mask = np.ones(self.x0.shape)

    if self._isROIMasked and self.doROIMsk:
        self._mask = np.all((self._mask, self._roiMsk), 0)

    if self._isVarExpMasked and self.doVarExpMsk:
        self._mask = np.all((self._mask, self._varExpMsk), 0)

    if self._isBetaMasked and self.doBetaMsk:
        self._mask = np.all((self._mask, self._betaMsk), 0)

    if self._isEccMasked and self.doEccMsk:
        self._mask = np.all((self._mask, self._eccMsk), 0)

    if self._isSigMasked and self.doSigMsk:
        self._mask = np.all((self._mask, self._sigMsk), 0)

    if self._isManualMasked and self.doManualMsk:
        self._mask = np.all((self._mask, self._manual_mask), 0)

    self._mask = self._mask.astype(bool)

    return self._mask


# ---------------------------------------------------------------------------#
def maskDartBoard(self, split="8"):
    """
    calculate number of voxels in the dartboard elemets as used for
    Linhardt et al. 2021

    Args:
        split (str, optional): possible: 8, 8tilt, 4, 4tilt . Defaults to '8'.

    Returns:
        nVoxDart: the count in the dartboard
    """

    ppi = 2 * np.pi
    pi2 = np.pi / 2
    pi4 = np.pi / 4
    pi8 = np.pi / 8

    if split == "8":
        nVoxDart = np.empty((18))
        k = 1

        # central
        nVoxDart[0] = np.sum(self.r <= 2)

        for i in range(2):
            for j in range(8):
                if i == 0:
                    nVoxDart[k] = np.sum(
                        np.all(
                            [
                                self.r > 2,
                                self.r <= 4.5,
                                self.phi > j * pi4,
                                self.phi <= (j + 1) * pi4,
                            ],
                            0,
                        )
                    )
                elif i == 1:
                    nVoxDart[k] = np.sum(
                        np.all(
                            [
                                self.r > 4.5,
                                self.r <= 7,
                                self.phi > j * pi4,
                                self.phi <= (j + 1) * pi4,
                            ],
                            0,
                        )
                    )
                k += 1

        # outer
        nVoxDart[17] = np.sum(self.r > 7)

    if split == "16":
        nVoxDart = np.empty((34))
        k = 1

        # central
        nVoxDart[0] = np.sum(self.r <= 2)

        for i in range(2):
            for j in range(16):
                if i == 0:
                    nVoxDart[k] = np.sum(
                        np.all(
                            [
                                self.r > 2,
                                self.r <= 4.5,
                                self.phi >= j * pi8,
                                self.phi < (j + 1) * pi8,
                            ],
                            0,
                        )
                    )
                elif i == 1:
                    nVoxDart[k] = np.sum(
                        np.all(
                            [
                                self.r > 4.5,
                                self.r <= 7,
                                self.phi >= j * pi8,
                                self.phi < (j + 1) * pi8,
                            ],
                            0,
                        )
                    )
                k += 1

        # outer
        nVoxDart[33] = np.sum(self.r > 7)

    elif split == "8tilt":
        # instead of tilting the grid -pi/8, tilt the data +pi/8
        self.phi += pi8
        self.phi[self.phi > ppi] -= ppi

        nVoxDart = np.empty((18))
        k = 1

        # central
        nVoxDart[0] = np.sum(self.r <= 2)

        for i in range(2):
            for j in range(8):
                if i == 0:
                    nVoxDart[k] = np.sum(
                        np.all(
                            [
                                self.r > 2,
                                self.r <= 4.5,
                                self.phi > j * pi4,
                                self.phi <= (j + 1) * pi4,
                            ],
                            0,
                        )
                    )
                elif i == 1:
                    nVoxDart[k] = np.sum(
                        np.all(
                            [
                                self.r > 4.5,
                                self.r <= 7,
                                self.phi > j * pi4,
                                self.phi <= (j + 1) * pi4,
                            ],
                            0,
                        )
                    )
                k += 1

        # outer
        nVoxDart[17] = np.sum(self.r > 7)

        # tilt back (reinitialize polar coords)

    elif split == "4":
        nVoxDart = np.empty((10))
        k = 1

        # central
        nVoxDart[0] = np.sum(self.r <= 2)

        for i in range(2):
            for j in range(4):
                if i == 0:
                    nVoxDart[k] = np.sum(
                        np.all(
                            [
                                self.r > 2,
                                self.r <= 4.5,
                                self.phi > j * pi2,
                                self.phi <= (j + 1) * pi2,
                            ],
                            0,
                        )
                    )
                elif i == 1:
                    nVoxDart[k] = np.sum(
                        np.all(
                            [
                                self.r > 4.5,
                                self.r <= 7,
                                self.phi > j * pi2,
                                self.phi <= (j + 1) * pi2,
                            ],
                            0,
                        )
                    )
                k += 1

        # outer
        nVoxDart[9] = np.sum(self.r > 7)

    elif split == "4tilt":
        # instead of tilting the grid -pi/4, tilt the data +pi/2
        self.phi += pi4
        self.phi[self.phi > ppi] -= ppi

        nVoxDart = np.empty((10))
        k = 1

        # central
        nVoxDart[0] = np.sum(self.r <= 2)

        for i in range(2):
            for j in range(4):
                if i == 0:
                    nVoxDart[k] = np.sum(
                        np.all(
                            [
                                self.r > 2,
                                self.r <= 4.5,
                                self.phi > j * pi2,
                                self.phi <= (j + 1) * pi2,
                            ],
                            0,
                        )
                    )
                elif i == 1:
                    nVoxDart[k] = np.sum(
                        np.all(
                            [
                                self.r > 4.5,
                                self.r <= 7,
                                self.phi > j * pi2,
                                self.phi <= (j + 1) * pi2,
                            ],
                            0,
                        )
                    )
                k += 1

        # outer
        nVoxDart[9] = np.sum(self.r > 7)

        # tilt back (reinitialize polar coords)

    return nVoxDart


def loadSPMmat(self, path):
    """
    load mat file from SPM analysis for tValues, convert nii to mat first in MakeAllImages.m
    I guess this is not working.

    Args:
        path (str): path to SPM tValues mat file
    """

    self.tValues = loadmat(path, squeeze_me=True)["bar"]

    if self.isROIMasked:
        self.tValues = self.tValues[self.roiMsk]
    if self._isVarExpMasked:
        self.tValues = self.tValues[self.msk]
