#!/usr/bin/env python3
"""
hub_surface_map.py

Map stage 08 hub z-scores onto the fsaverage cortex, in the same figure style
as surface_map.py: pial surface, black background, diverging colors kept
saturated and darkened in the sulci.

Layout: four rows (LH lateral, RH lateral, LH medial, RH medial) and one
column per frequency band, with one shared color scale.

Each hub is a group of AAL regions (hubs.py). Every cortical vertex is
assigned the AAL region beneath it from the stored table
resources/aal90_fsaverage5.npz, then the value of that region's hub. Hubs
with no cortical surface (Thalamus, Basal ganglia) cannot be shown and are
reported. The medial wall is left grey, using the fsaverage aparc annotation.

Two inputs are accepted:
  zscores.csv         target subjects scored against one reference group at
                      a time. One figure per reference group: the mean z of
                      the target subjects.
  zscores_cohort.csv  every subject scored against the whole cohort (all
                      groups pooled; cohort_zscores.py). One figure per
                      group: the mean z of that group's subjects.

Usage
    python3 hub_surface_map.py derivatives/group/hubs/ses01/zscores.csv \\
        --subjects-dir ~/mne_data/MNE-fsaverage-data

    python3 hub_surface_map.py derivatives/group/hubs/ses01/zscores_cohort.csv \\
        --subjects-dir ~/mne_data/MNE-fsaverage-data

    # selected bands, one reference, a wider color range:
    python3 hub_surface_map.py zscores.csv --band theta,alpha1,alpha2 \\
        --reference TLE --zrange 3 --subjects-dir ~/mne_data/MNE-fsaverage-data

    # other stage 08 measures: --measure within | roi_bc | roi_degree | hub_bc

Output (default folder: surface_maps/ next to the CSV):
    <measure>_<target>_vs_<reference>.png    from zscores.csv
    <measure>_<group>_vs_cohort.png          from zscores_cohort.csv

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


def _check_choice(kind, wanted, available):
    bad = sorted(set(wanted) - set(available))
    if bad:
        sys.exit(f"ERROR: {kind}(s) {bad} not in the file; available: "
                 f"{sorted(set(available))}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Map stage 08 hub z-scores onto the fsaverage cortex.")
    ap.add_argument("zscores",
                    help="zscores.csv or zscores_cohort.csv from stage 08")
    ap.add_argument("--measure", default="strength",
                    help="Hub measure to map (default strength).")
    ap.add_argument("--band", default=None,
                    help="Band columns, comma separated, in display order "
                         "(default all bands in the file).")
    ap.add_argument("--reference", default=None,
                    help="zscores.csv only: reference group(s), comma "
                         "separated (default all).")
    ap.add_argument("--group", default=None,
                    help="zscores_cohort.csv only: group(s) to map, comma "
                         "separated (default all).")
    ap.add_argument("--subjects", default=None,
                    help="Subjects to average, comma separated (default all "
                         "subjects of the group).")
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
    ap.add_argument("--full-res", action="store_true",
                    help="Draw panels on the full fsaverage mesh (slow, several "
                         "GB of memory per figure). Default: fsaverage6.")
    args = ap.parse_args(argv)

    import pandas as pd
    import surface_map
    from hubs import display_name

    zs = pd.read_csv(args.zscores)
    base = {"subject", "band", "measure", "hub", "z"}
    if "reference" in zs.columns:
        mode, split_col = "reference", "reference"
    elif "group" in zs.columns:
        mode, split_col = "cohort", "group"
    else:
        mode = None
    if mode is None or not base <= set(zs.columns):
        sys.exit(f"ERROR: {args.zscores} is not a stage 08 zscores.csv or "
                 "zscores_cohort.csv (needs columns subject, band, measure, hub, "
                 "z and either reference or group).")
    if mode == "reference" and args.group:
        sys.exit("ERROR: --group applies to zscores_cohort.csv; use --reference.")
    if mode == "cohort" and args.reference:
        sys.exit("ERROR: --reference applies to zscores.csv; use --group.")
    if args.zrange <= 0:
        sys.exit("ERROR: --zrange must be positive.")

    subjects_dir = args.subjects_dir or default_subjects_dir()
    if not subjects_dir:
        sys.exit("ERROR: fsaverage not found; pass --subjects-dir (the folder "
                 "that contains fsaverage).")

    zs = zs[zs["measure"] == args.measure]
    if zs.empty:
        sys.exit(f"ERROR: no rows for measure '{args.measure}'.")
    zs = zs.assign(subject=zs["subject"].astype(str))
    subjects = _split(args.subjects)
    if subjects:
        _check_choice("subject", subjects, zs["subject"])
        zs = zs[zs["subject"].isin(subjects)]

    bands = _split(args.band) or list(dict.fromkeys(zs["band"]))
    _check_choice("band", bands, zs["band"])
    splits = _split(args.reference if mode == "reference" else args.group) \
        or list(dict.fromkeys(zs[split_col]))
    _check_choice(split_col, splits, zs[split_col])

    outdir = Path(args.outdir) if args.outdir else Path(args.zscores).parent / "surface_maps"
    outdir.mkdir(parents=True, exist_ok=True)

    print("Assigning AAL regions to fsaverage vertices ...")
    regions = vertex_regions(subjects_dir)
    off_surface = sorted(set(zs["hub"]) - surface_hubs(regions))
    if off_surface:
        print(f"  Not shown (no cortical surface): {', '.join(off_surface)}")

    if mode == "reference":
        try:
            import config
            target = config.TARGET_GROUP
        except Exception:
            target = "target"
        target_name = display_name(target, zs["subject"].unique())

    for key in splits:
        sel = zs[zs[split_col] == key]
        n_subj = sel["subject"].nunique()
        columns, labels = [], []
        for band in bands:
            hub_mean = sel[sel["band"] == band].groupby("hub")["z"].mean().to_dict()
            columns.append(hub_values_to_vertices(hub_mean, regions))
            labels.append(band)
        if not any(np.isfinite(np.concatenate([c["lh"], c["rh"]])).any()
                   for c in columns):
            print(f"  {key}: no finite z-scores; skipped.")
            continue
        if mode == "reference":
            title = (f"{args.measure}: mean z of {n_subj} {target_name} "
                     f"subject(s) vs {display_name(key)}")
            out = outdir / f"{args.measure}_{target}_vs_{key}.png"
        else:
            name = display_name(key, sel["subject"].unique())
            title = (f"{args.measure}: mean z of {n_subj} {name} subject(s) "
                     f"vs whole cohort")
            out = outdir / f"{args.measure}_{key}_vs_cohort.png"
        surface_map.render_grid(
            columns, labels, str(out), subjects_dir, title=title,
            vmin=-args.zrange, vmax=args.zrange,
            colorbar_label=f"z-score ({args.measure})",
            surf=args.surf, shading=args.shading, bg_color=args.bg_color,
            coarsen=0 if args.full_res else 1)
        print(f"  wrote {out}")
    print(f"Done: measure {args.measure}, bands {', '.join(bands)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
