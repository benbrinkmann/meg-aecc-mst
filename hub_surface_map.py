#!/usr/bin/env python3
"""
hub_surface_map.py

Map hub z-scores from stage 08 (zscores.csv) onto the fsaverage cortex, in
the same figure style as surface_map.py: pial surface, black background,
LH/RH lateral and medial panels, diverging colors kept saturated and darkened
in the sulci.

Each hub is a group of AAL regions (hubs.py). Every cortical vertex is
assigned the AAL region beneath it from the stored table
resources/aal90_fsaverage5.npz, then the value of that region's hub. Hubs with no
cortical surface (Thalamus, Basal ganglia) cannot be shown and are reported.
The medial wall is left grey, using the fsaverage aparc annotation.

The value shown is the mean z-score of the target group's subjects relative
to one reference group, for one measure and one band. By default one figure
is made for every band and every reference group in the file.

Usage
    python3 hub_surface_map.py derivatives/group/hubs/ses01/zscores.csv \\
        --subjects-dir ~/mne_data/MNE-fsaverage-data

    # one band, one reference, a wider color range:
    python3 hub_surface_map.py zscores.csv --band alpha --reference TLE \\
        --zrange 3 --subjects-dir ~/mne_data/MNE-fsaverage-data

    # other stage 08 measures: --measure within | roi_bc | roi_degree | hub_bc

Output: <output dir>/<measure>_<band>_vs_<reference>.png
(default output dir: surface_maps/ next to zscores.csv)

Requires the full fsaverage subject (surf/?h.pial, ?h.sulc, ?h.sphere and
label/?h.aparc.annot) in --subjects-dir. No atlas download is needed.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LABELS_FILE = HERE / "resources" / "aal90_fsaverage5.npz"
MEDIAL_WALL_NAMES = ("unknown", "corpuscallosum")


# ════════════════════════════════════════════════════════════════════════════
# Vertex -> AAL region -> hub
# ════════════════════════════════════════════════════════════════════════════

def vertex_regions(subjects_dir):
    """
    AAL region name for every vertex of the full fsaverage, per hemisphere.

    The stored fsaverage5 labels are carried to the full mesh by nearest vertex
    on the sphere; the first 10242 full-mesh vertices are the fsaverage5
    vertices, so those keep their label exactly. Medial wall vertices (aparc
    'unknown' / 'corpuscallosum') get no region.

    Returns {'lh': array of str, 'rh': array of str}; '' means no region.
    """
    from nibabel.freesurfer import read_annot, read_geometry
    from scipy.spatial import cKDTree

    if not LABELS_FILE.exists():
        sys.exit(f"ERROR: label table not found: {LABELS_FILE}\n"
                 "It is part of the repository; run git pull.")
    table = np.load(LABELS_FILE)
    names = table["names"].astype(str)
    fs = Path(subjects_dir) / "fsaverage"

    regions = {}
    for hemi in ("lh", "rh"):
        sphere_file = fs / "surf" / f"{hemi}.sphere"
        if not sphere_file.exists():
            sys.exit(f"ERROR: {sphere_file} not found; copy the whole fsaverage "
                     f"folder into {subjects_dir}.")
        full_sphere, _ = read_geometry(str(sphere_file))
        small_sphere = table[f"{hemi}_sphere"]
        labels5 = table[f"{hemi}_labels"]

        dist, nearest = cKDTree(small_sphere).query(full_sphere)
        n5 = len(small_sphere)
        if dist[:n5].max() > 0.01:
            print(f"  WARNING ({hemi}): this fsaverage's vertices do not line up "
                  f"with fsaverage5 (max {dist[:n5].max():.2f} mm); regions are "
                  f"assigned by nearest vertex instead.")
        region = names[labels5[nearest]].astype(object)

        annot = fs / "label" / f"{hemi}.aparc.annot"
        if annot.exists():
            lab, _, annot_names = read_annot(str(annot))
            annot_names = [n.decode() for n in annot_names]
            wall_ids = [annot_names.index(n) for n in MEDIAL_WALL_NAMES
                        if n in annot_names]
            wall = (lab == -1) | np.isin(lab, wall_ids)
            region[wall] = ""
        else:
            print(f"  WARNING: {annot} not found; the medial wall is not masked.")
        regions[hemi] = region
    return regions


def hub_lookup():
    """AAL base name -> hub name, from hubs.py."""
    from hubs import HUB_REGIONS
    return {base: hub for hub, bases in HUB_REGIONS.items() for base in bases}


def hub_values_to_vertices(hub_values, regions):
    """
    Per-vertex arrays from {'<hub> <L|R>': value}. Vertices whose hub has no
    value, and vertices with no region, are NaN (shown as grey folding).
    """
    base_to_hub = hub_lookup()
    data = {}
    for hemi, side in (("lh", "L"), ("rh", "R")):
        region = regions[hemi]
        out = np.full(len(region), np.nan)
        for name in set(region) - {""}:
            base = name[:-2]                     # 'Precentral_L' -> 'Precentral'
            hub = f"{base_to_hub[base]} {side}"
            if hub in hub_values and np.isfinite(hub_values[hub]):
                out[region == name] = hub_values[hub]
        data[hemi] = out
    return data


def surface_hubs(regions):
    """Hub names ('<hub> <L|R>') that have at least one cortical vertex."""
    base_to_hub = hub_lookup()
    present = set()
    for hemi, side in (("lh", "L"), ("rh", "R")):
        for name in set(regions[hemi]) - {""}:
            present.add(f"{base_to_hub[name[:-2]]} {side}")
    return present


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def _split(arg):
    return [s.strip() for s in arg.split(",") if s.strip()] if arg else None


def default_subjects_dir():
    """~/mne_data/MNE-fsaverage-data if it holds fsaverage, else None."""
    p = Path.home() / "mne_data" / "MNE-fsaverage-data"
    return str(p) if (p / "fsaverage").is_dir() else None


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Map stage 08 hub z-scores onto the fsaverage cortex.")
    ap.add_argument("zscores", help="zscores.csv written by s08_hubs.py")
    ap.add_argument("--measure", default="strength",
                    help="Hub measure to map (default strength).")
    ap.add_argument("--band", default=None,
                    help="Band(s) to map, comma separated (default all).")
    ap.add_argument("--reference", default=None,
                    help="Reference group(s), comma separated (default all).")
    ap.add_argument("--subjects", default=None,
                    help="Target subjects to average, comma separated "
                         "(default all in the file).")
    ap.add_argument("-o", "--outdir", default=None,
                    help="Output folder (default surface_maps/ next to the CSV).")
    ap.add_argument("--subjects-dir", default=None,
                    help="Folder containing fsaverage (default "
                         "~/mne_data/MNE-fsaverage-data).")
    ap.add_argument("--zrange", type=float, default=2.0, metavar="N",
                    help="Color scale -N..+N (default 2).")
    ap.add_argument("--surf", default="pial", help="pial (default), white, inflated.")
    ap.add_argument("--shading", type=float, default=0.75,
                    help="How strongly sulci darken the colors, 0-1 (default 0.75).")
    ap.add_argument("--bg-color", default="black")
    args = ap.parse_args(argv)

    import pandas as pd
    import surface_map

    zs = pd.read_csv(args.zscores)
    needed = {"subject", "reference", "band", "measure", "hub", "z"}
    if not needed <= set(zs.columns):
        sys.exit(f"ERROR: {args.zscores} lacks columns {sorted(needed - set(zs.columns))}; "
                 "expected the zscores.csv from s08_hubs.py.")
    if args.zrange <= 0:
        sys.exit("ERROR: --zrange must be positive.")

    subjects_dir = args.subjects_dir or default_subjects_dir()
    if not subjects_dir:
        sys.exit("ERROR: fsaverage not found; pass --subjects-dir (the folder "
                 "that contains fsaverage).")

    zs = zs[zs["measure"] == args.measure]
    if zs.empty:
        sys.exit(f"ERROR: no rows for measure '{args.measure}'.")
    subjects = _split(args.subjects)
    if subjects:
        missing = sorted(set(subjects) - set(zs["subject"].astype(str)))
        if missing:
            sys.exit(f"ERROR: subject(s) not in the file: {missing}")
        zs = zs[zs["subject"].astype(str).isin(subjects)]
    bands = _split(args.band) or list(dict.fromkeys(zs["band"]))
    refs = _split(args.reference) or list(dict.fromkeys(zs["reference"]))
    for kind, wanted, have in (("band", bands, set(zs["band"])),
                               ("reference", refs, set(zs["reference"]))):
        bad = sorted(set(wanted) - have)
        if bad:
            sys.exit(f"ERROR: {kind}(s) {bad} not in the file; available: "
                     f"{sorted(have)}")

    outdir = Path(args.outdir) if args.outdir else Path(args.zscores).parent / "surface_maps"
    outdir.mkdir(parents=True, exist_ok=True)

    print("Assigning AAL regions to fsaverage vertices ...")
    regions = vertex_regions(subjects_dir)
    on_surface = surface_hubs(regions)
    off_surface = sorted(set(zs["hub"]) - on_surface)
    if off_surface:
        print(f"  Not shown (no cortical surface): {', '.join(off_surface)}")

    n_subj = zs["subject"].nunique()
    for ref in refs:
        for band in bands:
            sel = zs[(zs["reference"] == ref) & (zs["band"] == band)]
            hub_mean = sel.groupby("hub")["z"].mean().to_dict()
            data = hub_values_to_vertices(hub_mean, regions)
            if not np.isfinite(np.concatenate([data["lh"], data["rh"]])).any():
                print(f"  {band} vs {ref}: no finite z-scores; skipped.")
                continue
            out = outdir / f"{args.measure}_{band}_vs_{ref}.png"
            title = (f"{args.measure}, {band} band: mean z of "
                     f"{sel['subject'].nunique()} target subject(s) vs {ref}")
            surface_map.render_png(
                data, str(out), subjects_dir, title=title,
                vmin=-args.zrange, vmax=args.zrange,
                colorbar_label=f"z-score ({args.measure})",
                surf=args.surf, shading=args.shading, bg_color=args.bg_color)
            print(f"  wrote {out}")
    print(f"Done: {n_subj} target subject(s), measure {args.measure}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
