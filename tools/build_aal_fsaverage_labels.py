#!/usr/bin/env python3
"""
Build resources/aal90_fsaverage5.npz: the AAL region under each fsaverage5 vertex.

hub_surface_map.py uses this table to paint hub values on the fsaverage
cortex without needing the AAL volume at plotting time. Rerun only if the
atlas or the method changes.

Method
  - Atlas: original AAL for SPM (ROI_MNI_V4, 116 regions; the cerebral 90
    match the names in hubs.py), MNI space, 2 mm.
  - Each fsaverage5 vertex is sampled along the line between the white and
    pial surfaces (nilearn vol_to_surf, kind='depth') and takes the most
    frequent AAL label there ('nearest_most_frequent').
  - Labels from the wrong hemisphere, the cerebellum and vermis, and the deep
    nuclei with no cortical surface (thalamus, caudate, putamen, pallidum)
    are discarded.
  - A vertex left unlabeled takes the label of the nearest labeled vertex
    within 4 mm on the pial surface, to close small gaps between the volume
    atlas and the surface. The medial wall is not handled here; the plotting
    script masks it with the fsaverage aparc annotation.
  - fsaverage5 sphere coordinates are stored so the labels can be carried to
    any fsaverage resolution by nearest vertex on the sphere (the first 10242
    vertices of the full fsaverage are the fsaverage5 vertices).

Usage
  python3 tools/build_aal_fsaverage_labels.py ROI_MNI_V4.nii ROI_MNI_V4.txt
The two AAL files are in FieldTrip (template/atlas/aal/) and in the AAL for
SPM distribution.
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, surface
from scipy.spatial import cKDTree

FILL_MM = 4.0
DEEP_NUCLEI = {"Thalamus", "Caudate", "Putamen", "Pallidum"}
OUT = Path(__file__).resolve().parent.parent / "resources" / "aal90_fsaverage5.npz"


def main(nii_path, txt_path):
    table = pd.read_csv(txt_path, sep="\t", header=None, names=["abbr", "name", "code"])
    code_to_name = dict(zip(table["code"], table["name"]))
    img = nib.load(nii_path)
    fs5 = datasets.fetch_surf_fsaverage("fsaverage5")   # bundled with nilearn

    region_names = [""]                     # index 0 = no region
    result = {}
    for hemi, side, key in (("lh", "L", "left"), ("rh", "R", "right")):
        pial = nib.load(fs5[f"pial_{key}"]).darrays[0].data.astype(float)
        sphere = nib.load(fs5[f"sphere_{key}"]).darrays[0].data.astype(np.float32)
        codes = surface.vol_to_surf(
            img, fs5[f"pial_{key}"], inner_mesh=fs5[f"white_{key}"],
            interpolation="nearest_most_frequent").round().astype(int)
        names = np.array([code_to_name.get(c, "") for c in codes], dtype=object)
        keep = np.array([
            bool(n) and n.endswith("_" + side)
            and not n.startswith(("Cerebel", "Vermis"))
            and n[:-2] not in DEEP_NUCLEI
            for n in names])
        names[~keep] = ""

        # Close small gaps: nearest labeled vertex within FILL_MM on the pial surface.
        dist, nearest = cKDTree(pial[keep]).query(pial[~keep])
        gap = np.flatnonzero(~keep)
        close = dist <= FILL_MM
        names[gap[close]] = names[keep][nearest[close]]

        for n in sorted(set(names) - {""}):
            if n not in region_names:
                region_names.append(n)
        index = np.array([region_names.index(n) for n in names], dtype=np.int16)
        result[f"{hemi}_labels"] = index
        result[f"{hemi}_sphere"] = sphere
        print(f"{hemi}: {np.count_nonzero(index)}/{len(index)} vertices labeled, "
              f"{len(set(names) - {''})} regions")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT, names=np.array(region_names), **result,
        source=np.array("AAL for SPM (ROI_MNI_V4) projected to fsaverage5; "
                        "white-pial depth sampling, most frequent label, "
                        f"{FILL_MM:g} mm gap fill"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
