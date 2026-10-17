#!/usr/bin/env python3
"""
surface_map.py

Render a scalar value per FreeSurfer Desikan-Killiany (aparc) cortical region
as a color map on the fsaverage brain surface.

Two input routes:
  1. An uploaded table of values keyed by FreeSurfer column names, e.g. a
     z-score matrix with columns like 'lh_superiorfrontal_thickness' or
     'rh_insula_grayvol' (and aseg subcortical columns, which are skipped
     because they are volumetric, not cortical-surface, structures).
  2. A nodal value array coming straight from the connectivity pipeline
     (one value per aparc region), via map_region_values().

Rendering uses MNE's read_labels_from_annot to get the aparc parcellation on
fsaverage, paints each label's vertices with that region's value, and plots
with nilearn.plotting.plot_surf_stat_map (pure matplotlib, works headless on
Linux with no OpenGL / Xvfb). Output is a PNG panel of lateral + medial views
for both hemispheres, and optionally an interactive HTML view.

Requires: mne, nibabel, nilearn, numpy, matplotlib, pandas
A FreeSurfer-style fsaverage must be available to MNE (mne.datasets.
fetch_fsaverage() on first use); on an offline machine point --subjects-dir
at an existing fsaverage.

Author: Benjamin H. Brinkmann, Ph.D.
"""

import os
import re
import sys
import argparse
import warnings

import numpy as np


# ════════════════════════════════════════════════════════════════════════════
# FreeSurfer column-name parsing
# ════════════════════════════════════════════════════════════════════════════

# aseg subcortical / brainstem structures that may appear in the table but are
# NOT cortical surface regions — these are skipped (reported, not mapped).
_SUBCORTICAL = {
    "thalamus", "caudate", "putamen", "pallidum", "hippocampus", "amygdala",
    "accumbens-area", "accumbens", "ventraldc", "brain-stem", "brainstem",
    "cerebellum-cortex", "cerebellum-white-matter", "lateral-ventricle",
    "choroid-plexus", "vessel", "csf",
}

# Measurement suffixes that may be appended to a cortical region name in a
# FreeSurfer aparc stats table.
_MEASURE_SUFFIXES = ("_grayvol", "_thickness", "_area", "_volume",
                     "_meancurv", "_thickstd", "_foldind", "_curvind",
                     "_gauscurv")


def parse_column(col):
    """
    Parse a single FreeSurfer-style column name into
    (hemi, region, measure, is_cortical).

    Cortical examples:
        'lh_superiorfrontal_thickness' -> ('lh', 'superiorfrontal',
                                           'thickness', True)
        'rh_insula_grayvol'            -> ('rh', 'insula', 'grayvol', True)

    Subcortical examples (is_cortical False, region is the aseg name lowered):
        'Left-Thalamus'  -> ('lh', 'left-thalamus', None, False)
        'Brain-Stem'     -> (None, 'brain-stem', None, False)

    Returns None for a column that cannot be interpreted as a brain region
    (e.g. an index column, a subject id, 'eTIV').
    """
    c = col.strip()
    low = c.lower()

    # --- Cortical aparc columns: lh_/rh_<region>[_<measure>] ---
    m = re.match(r"^(lh|rh)[_\.](.+)$", low)
    if m:
        hemi, rest = m.group(1), m.group(2)
        measure = None
        for suf in _MEASURE_SUFFIXES:
            if rest.endswith(suf):
                measure = suf[1:]            # strip leading underscore
                rest = rest[: -len(suf)]
                break
        region = rest
        # A subcortical name can also carry an lh_/rh_ style prefix in some
        # exports; treat known subcortical names as non-cortical.
        if region in _SUBCORTICAL:
            return (hemi, region, measure, False)
        return (hemi, region, measure, True)

    # --- aseg subcortical columns: 'Left-Thalamus', 'Right-Amygdala', ... ---
    m = re.match(r"^(left|right)-(.+)$", low)
    if m:
        side, struct = m.group(1), m.group(2)
        hemi = "lh" if side == "left" else "rh"
        return (hemi, f"{side}-{struct}", None, False)

    # --- Unsided subcortical, e.g. 'Brain-Stem' ---
    if low in _SUBCORTICAL:
        return (None, low, None, False)

    return None


def select_cortical(columns, measure=None):
    """
    From a list of column names, return the cortical aparc entries, optionally
    restricted to a single measure (e.g. 'thickness').

    Returns
    -------
    mapped : list of (column, hemi, region) for cortical columns kept
    skipped_subcortical : list of column names that are subcortical/brainstem
    skipped_other       : list of column names not recognized as regions
    measures_found      : sorted set of measure names seen among cortical cols
    """
    mapped, skipped_sub, skipped_other = [], [], []
    measures_found = set()

    for col in columns:
        p = parse_column(col)
        if p is None:
            skipped_other.append(col)
            continue
        hemi, region, meas, is_ctx = p
        if not is_ctx:
            skipped_sub.append(col)
            continue
        if meas:
            measures_found.add(meas)
        if measure is not None and meas is not None and meas != measure:
            continue   # a different measure than requested
        mapped.append((col, hemi, region))

    return mapped, skipped_sub, skipped_other, sorted(measures_found)


# ════════════════════════════════════════════════════════════════════════════
# fsaverage aparc labels
# ════════════════════════════════════════════════════════════════════════════

def load_aparc_labels(subjects_dir=None, parc="aparc"):
    """
    Load the aparc (Desikan-Killiany) labels for fsaverage via MNE.

    Returns a dict keyed by (hemi, region_lower) -> mne.Label, so a parsed
    column can be matched directly. 'region_lower' is the FreeSurfer region
    name lowercased (e.g. 'superiorfrontal', 'insula').

    Parameters
    ----------
    subjects_dir : str | None
        FreeSurfer SUBJECTS_DIR containing 'fsaverage'. If None, MNE's default
        is used, fetching fsaverage if needed.
    parc : str
        Parcellation name: 'aparc' (DK, 68 ctx) or 'aparc.a2009s' (Destrieux).
    """
    import mne

    if subjects_dir is None:
        # Ensures fsaverage exists; downloads on first use.
        fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
        subjects_dir = os.path.dirname(fs_dir)

    labels = mne.read_labels_from_annot(
        "fsaverage", parc=parc, subjects_dir=subjects_dir, verbose=False)

    table = {}
    for lab in labels:
        if lab.name.lower().startswith("unknown"):
            continue
        # lab.name is like 'superiorfrontal-lh'
        m = re.match(r"^(.*)-(lh|rh)$", lab.name)
        if not m:
            continue
        region, hemi = m.group(1).lower(), m.group(2)
        table[(hemi, region)] = lab

    return table, subjects_dir


# ════════════════════════════════════════════════════════════════════════════
# Build per-vertex data for both hemispheres
# ════════════════════════════════════════════════════════════════════════════

def build_surface_data(region_values, subjects_dir=None, parc="aparc",
                       fill=np.nan):
    """
    Convert {(hemi, region): value} into per-vertex arrays on fsaverage.

    region_values : dict
        Keys are (hemi, region_lower); values are floats.
    fill : float
        Value for vertices in no mapped region (default NaN -> transparent in
        plotting). Use 0.0 if your colormap should show those as mid-scale.

    Returns
    -------
    data : dict with 'lh' and 'rh' -> np.ndarray (n_vertices_fsaverage,)
    labels_table : the (hemi,region)->Label dict used (for reuse)
    n_mapped : number of regions actually painted
    missing : list of (hemi, region) requested but absent from the atlas
    """
    import mne

    labels_table, subjects_dir = load_aparc_labels(subjects_dir, parc)

    # fsaverage has 163842 vertices per hemisphere (ico-7). Derive from a label.
    # Safer: read the surface vertex count from the fsaverage source space size
    # implied by the labels. We get it from the max vertex index across labels.
    nverts = {"lh": 0, "rh": 0}
    for (hemi, _), lab in labels_table.items():
        if lab.vertices.size:
            nverts[hemi] = max(nverts[hemi], int(lab.vertices.max()) + 1)
    # fsaverage standard is 163842; use that as a floor.
    for h in ("lh", "rh"):
        nverts[h] = max(nverts[h], 163842)

    data = {"lh": np.full(nverts["lh"], fill, dtype=float),
            "rh": np.full(nverts["rh"], fill, dtype=float)}

    n_mapped = 0
    missing = []
    for (hemi, region), val in region_values.items():
        lab = labels_table.get((hemi, region))
        if lab is None:
            missing.append((hemi, region))
            continue
        data[hemi][lab.vertices] = val
        n_mapped += 1

    return data, labels_table, n_mapped, missing


# ════════════════════════════════════════════════════════════════════════════
# Rendering
# ════════════════════════════════════════════════════════════════════════════

def render_png(data, out_path, subjects_dir, title=None,
               cmap="RdBu_r", vmin=None, vmax=None, symmetric=True,
               threshold=None, colorbar_label="value",
               surf="inflated", bg="sulc"):
    """
    Render a 2x2 panel (lateral + medial, both hemispheres) to a PNG using
    nilearn (matplotlib backend, headless-safe).

    data : dict 'lh'/'rh' -> per-vertex array (NaN where unmapped)
    symmetric : if True and vmin/vmax not given, use a symmetric range about 0
        (appropriate for z-scores / signed metrics).
    threshold : values with abs < threshold are left transparent.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn import plotting, surface

    # fsaverage geometry files from the FreeSurfer subject (surf/<hemi>.<surf>)
    fs = os.path.join(subjects_dir, "fsaverage", "surf")
    meshes = {
        "lh": os.path.join(fs, f"lh.{surf}"),
        "rh": os.path.join(fs, f"rh.{surf}"),
    }
    bg_maps = {
        "lh": os.path.join(fs, f"lh.{bg}"),
        "rh": os.path.join(fs, f"rh.{bg}"),
    }

    # Determine color limits
    allvals = np.concatenate([data["lh"], data["rh"]])
    finite = allvals[np.isfinite(allvals)]
    if finite.size == 0:
        raise ValueError("No finite values to plot (all vertices unmapped).")
    if vmin is None or vmax is None:
        if symmetric:
            m = float(np.nanmax(np.abs(finite)))
            vmin, vmax = -m, m
        else:
            vmin, vmax = float(finite.min()), float(finite.max())

    # nilearn expects 'left'/'right' for the hemi argument
    _hemi_full = {"lh": "left", "rh": "right"}

    views = [("lh", "lateral"), ("rh", "lateral"),
             ("lh", "medial"),  ("rh", "medial")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9),
                             subplot_kw={"projection": "3d"})
    axes = axes.ravel()

    for ax, (hemi, view) in zip(axes, views):
        plotting.plot_surf_stat_map(
            meshes[hemi], data[hemi], bg_map=bg_maps[hemi],
            hemi=_hemi_full[hemi],
            view=view, cmap=cmap, vmin=vmin, vmax=vmax,
            threshold=threshold, colorbar=False, axes=ax)
        ax.set_title(f"{hemi.upper()} {view}", fontsize=10)

    # One shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.04,
                        shrink=0.6)
    cbar.set_label(colorbar_label)

    if title:
        fig.suptitle(title, fontsize=13)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_html(data, out_path, subjects_dir, title=None,
                cmap="RdBu_r", vmin=None, vmax=None, symmetric=True,
                threshold=None, surf="inflated", bg="sulc"):
    """
    Render an interactive rotatable view (one HTML per hemisphere, lateral
    start) using nilearn.plotting.view_surf. Writes <out_path>_lh.html and
    <out_path>_rh.html and returns the two paths.
    """
    from nilearn import plotting

    fs = os.path.join(subjects_dir, "fsaverage", "surf")
    allvals = np.concatenate([data["lh"], data["rh"]])
    finite = allvals[np.isfinite(allvals)]
    if vmin is None or vmax is None:
        if symmetric:
            m = float(np.nanmax(np.abs(finite)))
            vmin, vmax = -m, m
        else:
            vmin, vmax = float(finite.min()), float(finite.max())

    base = out_path[:-5] if out_path.endswith(".html") else out_path
    outs = []
    for hemi in ("lh", "rh"):
        mesh = os.path.join(fs, f"{hemi}.{surf}")
        bg_map = os.path.join(fs, f"{hemi}.{bg}")
        view = plotting.view_surf(
            mesh, data[hemi], bg_map=bg_map, cmap=cmap,
            vmin=vmin, vmax=vmax, threshold=threshold,
            symmetric_cmap=symmetric, title=title)
        p = f"{base}_{hemi}.html"
        view.save_as_html(p)
        outs.append(p)
    return outs


# ════════════════════════════════════════════════════════════════════════════
# High-level entry points
# ════════════════════════════════════════════════════════════════════════════

def map_region_values(region_values, out_path, subjects_dir=None,
                      parc="aparc", html=False, **kw):
    """
    Map a dict {(hemi, region_lower): value} to the surface. This is the entry
    point the connectivity pipeline calls directly with its nodal values.

    Returns a dict describing what was produced and what was skipped.
    """
    data, _, n_mapped, missing = build_surface_data(
        region_values, subjects_dir=subjects_dir, parc=parc)
    subjects_dir = subjects_dir or _default_subjects_dir()

    png = render_png(data, out_path, subjects_dir, **kw)
    result = {"png": png, "n_mapped": n_mapped, "missing": missing}
    if html:
        result["html"] = render_html(data, out_path, subjects_dir, **kw)
    return result


def map_table(values_by_column, out_path, measure=None, subjects_dir=None,
              parc="aparc", html=False, **kw):
    """
    Map an uploaded table given as {column_name: value} (one row of z-scores).

    If multiple measures are present (grayvol, thickness) and `measure` is not
    specified, raises with the list of measures so the caller can choose.

    Returns a dict with png path, skipped-column lists, and the measure used.
    """
    cols = list(values_by_column.keys())
    mapped, skipped_sub, skipped_other, measures = select_cortical(cols, measure)

    if measure is None and len(measures) > 1:
        raise ValueError(
            f"Table has multiple measures {measures}; pass measure=... to "
            f"choose which to map.")

    region_values = {}
    for col, hemi, region in mapped:
        v = values_by_column[col]
        if v is None:
            continue
        try:
            region_values[(hemi, region)] = float(v)
        except (TypeError, ValueError):
            continue

    if not region_values:
        raise ValueError("No cortical aparc values found to map.")

    res = map_region_values(region_values, out_path, subjects_dir=subjects_dir,
                            parc=parc, html=html, **kw)
    res.update({
        "measure": measure or (measures[0] if measures else None),
        "skipped_subcortical": skipped_sub,
        "skipped_other": skipped_other,
        "n_regions_input": len(mapped),
    })
    return res


def _default_subjects_dir():
    import mne
    fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    return os.path.dirname(fs_dir)


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def _parse_rows(spec, n):
    """
    Parse a --rows spec into a list of 0-based row indices.

    Accepts:
      '0-5'         -> [0,1,2,3,4,5]   (inclusive range)
      '6-11'        -> [6,7,8,9,10,11]
      '0,2,4'       -> [0,2,4]
      '3'           -> [3]
    Indices must be within [0, n).
    """
    idx = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            idx.extend(range(a, b + 1))
        elif part:
            idx.append(int(part))
    bad = [i for i in idx if i < 0 or i >= n]
    if bad:
        sys.exit(f"ERROR: row index/indices out of range (table has {n} "
                 f"rows, 0-{n-1}): {bad}")
    return idx


def _read_value_table(path, row=None, rows=None, id_col=None):
    """
    Read a CSV/TSV table of z-scores. Returns {column: value} for one row, or
    the per-column mean across a set of rows when `rows` is given.

    Selection (first that applies):
      rows   : average the given rows (e.g. '0-5' for a group mean)
      row    : a single 0-based row index
      id_col : rows indexed by an identifier column (used with --id)
      else   : the first data row
    Only numeric columns are averaged; non-numeric columns (e.g. a label
    column) are dropped from a group mean.
    """
    import pandas as pd
    sep = "\t" if path.lower().endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(path, sep=sep)

    if id_col is not None:
        if id_col not in df.columns:
            sys.exit(f"ERROR: id column '{id_col}' not in table.")
        df = df.set_index(id_col)

    if rows is not None:
        idx = _parse_rows(rows, len(df))
        subset = df.iloc[idx]
        numeric = subset.apply(pd.to_numeric, errors="coerce")
        series = numeric.mean(axis=0, skipna=True)
        print(f"  averaging {len(idx)} row(s): {idx}")
        return series.to_dict()

    if row is not None:
        series = df.iloc[int(row)]
    elif id_col is not None:
        sys.exit("ERROR: --id-col given but no --id value to select a row.")
    else:
        if len(df) > 1:
            print(f"  (table has {len(df)} rows; using row 0 — use --row or "
                  f"--rows to choose)")
        series = df.iloc[0]
    return series.to_dict()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Map FreeSurfer aparc region values onto the fsaverage "
                    "surface.")
    ap.add_argument("table", help="CSV/TSV of values keyed by FreeSurfer "
                    "column names (e.g. lh_insula_thickness).")
    ap.add_argument("-o", "--output", default="surface_map.png",
                    help="Output PNG path.")
    ap.add_argument("--measure", default=None,
                    help="Which measure to map when the table has several "
                         "(e.g. thickness, grayvol).")
    ap.add_argument("--row", default=None,
                    help="0-based row index to map (default: first row).")
    ap.add_argument("--rows", default=None,
                    help="Average a set of rows before mapping, e.g. '0-5' for "
                         "a group mean, or '0,2,4'. Numeric columns only.")
    ap.add_argument("--id-col", default=None,
                    help="Name of an identifier column to index rows by.")
    ap.add_argument("--id", dest="id_val", default=None,
                    help="Row identifier value (used with --id-col).")
    ap.add_argument("--subjects-dir", default=None,
                    help="FreeSurfer SUBJECTS_DIR containing fsaverage.")
    ap.add_argument("--parc", default="aparc",
                    help="Parcellation: aparc (DK) or aparc.a2009s.")
    ap.add_argument("--cmap", default="RdBu_r", help="Matplotlib colormap.")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Hide |value| below this.")
    ap.add_argument("--no-symmetric", action="store_true",
                    help="Do not force a symmetric color range about 0.")
    ap.add_argument("--label", default="z-score",
                    help="Colorbar label.")
    ap.add_argument("--title", default=None)
    ap.add_argument("--html", action="store_true",
                    help="Also write interactive HTML views.")
    args = ap.parse_args(argv)

    values = _read_value_table(args.table, row=args.row, rows=args.rows,
                               id_col=args.id_col)

    if args.id_val is not None and args.id_col is not None:
        # Re-read indexed and select by id
        import pandas as pd
        sep = "\t" if args.table.lower().endswith((".tsv", ".txt")) else ","
        df = pd.read_csv(args.table, sep=sep).set_index(args.id_col)
        values = df.loc[args.id_val].to_dict()

    try:
        res = map_table(
            values, args.output, measure=args.measure,
            subjects_dir=args.subjects_dir, parc=args.parc, html=args.html,
            cmap=args.cmap, vmin=args.vmin, vmax=args.vmax,
            symmetric=not args.no_symmetric, threshold=args.threshold,
            colorbar_label=args.label, title=args.title)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")

    print(f"Mapped {res['n_mapped']} cortical region(s); measure="
          f"{res.get('measure')}")
    if res.get("missing"):
        print(f"  {len(res['missing'])} region(s) had no atlas match: "
              f"{res['missing'][:5]}{' ...' if len(res['missing'])>5 else ''}")
    if res.get("skipped_subcortical"):
        print(f"  Skipped {len(res['skipped_subcortical'])} subcortical/"
              f"brainstem column(s) (not on cortical surface).")
    if res.get("skipped_other"):
        print(f"  Skipped {len(res['skipped_other'])} unrecognized column(s).")
    print(f"  PNG: {res['png']}")
    if res.get("html"):
        for h in res["html"]:
            print(f"  HTML: {h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
