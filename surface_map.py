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

# Rows of every figure: hemisphere and view, in display order.
VIEWS = (("lh", "lateral"), ("rh", "lateral"), ("lh", "medial"), ("rh", "medial"))
_HEMI_FULL = {"lh": "left", "rh": "right"}


def _surface_files(subjects_dir, surf, bg):
    """Paths of the fsaverage mesh and sulcal map for each hemisphere."""
    fs = os.path.join(subjects_dir, "fsaverage", "surf")
    meshes = {h: os.path.join(fs, f"{h}.{surf}") for h in ("lh", "rh")}
    bg_maps = {h: os.path.join(fs, f"{h}.{bg}") for h in ("lh", "rh")}
    return meshes, bg_maps


def coarsen_ico(coords, faces):
    """
    One level coarser icosahedral mesh (e.g. fsaverage -> fsaverage6).

    FreeSurfer's fsaverage meshes are nested: subdividing each triangle of
    level k into four gives level k+1, the first N_k vertices are the level k
    vertices, and every new vertex sits on the midpoint of one old edge. So
    each old triangle (a, b, c) survives as the central child triangle made
    of the three new midpoints (m_ab, m_bc, m_ca), wound the same way. Each
    midpoint's two old neighbours are the ends of its edge, which recovers
    a, b and c exactly.

    Returns (coords[:N_old], faces_old). Per-vertex data for the coarse mesh
    is the first N_old entries of the fine data.
    """
    n = len(coords)
    n_old = (n - 2) // 4 + 2
    if 4 * (n_old - 2) + 2 != n or len(faces) != 2 * (n - 2):
        raise ValueError(f"not an icosahedral mesh: {n} vertices, {len(faces)} faces")

    # The two old-vertex neighbours ("parents") of every new vertex.
    parents = [set() for _ in range(n)]
    for i, j in ((0, 1), (1, 2), (2, 0)):
        a, b = faces[:, i], faces[:, j]
        for x, y in ((a, b), (b, a)):
            sel = (x >= n_old) & (y < n_old)
            for xi, yi in zip(x[sel], y[sel]):
                parents[xi].add(yi)
    if any(len(parents[v]) != 2 for v in range(n_old, n)):
        raise ValueError("mesh is not a regular icosahedral subdivision")

    central = faces[(faces >= n_old).all(axis=1)]
    out = np.empty_like(central)
    for k, (m1, m2, m3) in enumerate(central):
        p1, p2, p3 = parents[m1], parents[m2], parents[m3]
        # m1 = mid(a,b), m2 = mid(b,c), m3 = mid(c,a)
        (a,), (b,), (c,) = p3 & p1, p1 & p2, p2 & p3
        out[k] = (a, b, c)
    return coords[:n_old], out


def _load_meshes(subjects_dir, surf, bg, levels):
    """
    In-memory meshes and sulcal maps, coarsened `levels` times.
    Returns (meshes, bg_maps, n_vertices) with meshes[h] = [coords, faces].
    """
    from nibabel.freesurfer import read_geometry, read_morph_data
    fs = os.path.join(subjects_dir, "fsaverage", "surf")
    meshes, bg_maps = {}, {}
    for h in ("lh", "rh"):
        coords, faces = read_geometry(os.path.join(fs, f"{h}.{surf}"))
        for _ in range(levels):
            coords, faces = coarsen_ico(coords, faces)
        meshes[h] = [coords, faces]
        bg_maps[h] = read_morph_data(os.path.join(fs, f"{h}.{bg}"))[:len(coords)]
    return meshes, bg_maps, len(meshes["lh"][0])


def _color_limits(datasets, vmin, vmax, symmetric):
    """Color range: the given vmin/vmax, else derived from the data."""
    allvals = np.concatenate([d[h] for d in datasets for h in ("lh", "rh")])
    finite = allvals[np.isfinite(allvals)]
    if finite.size == 0:
        raise ValueError("No finite values to plot (all vertices unmapped).")
    if vmin is None or vmax is None:
        if symmetric:
            m = float(np.nanmax(np.abs(finite)))
            vmin, vmax = -m, m
        else:
            vmin, vmax = float(finite.min()), float(finite.max())
    return vmin, vmax


class _SulcalShading:
    """
    Context manager: combine value colors with the sulcal (gyral) shading.

    Transparency blending (nilearn's default with bg_on_data) mixes each color
    with grey, which washes the colors out. Instead we shade multiplicatively,
    like lighting: each colored face keeps its full colormap color and is only
    darkened according to the sulcal depth.
        shade = (1 - shading) + shading * gyral_brightness   (in [1-shading, 1])
        face  = colormap_color * shade
    Gyral crowns stay at full color; sulcal fundi are darkened by `shading`.
    Faces with no value (alpha 0) show the plain grey sulcal background.

    nilearn does not expose this step, so its color-mixing function is
    wrapped while the context is active and always restored afterwards.
    """

    def __init__(self, shading):
        self.depth = float(shading)
        if not 0.0 <= self.depth < 1.0:
            raise ValueError(f"shading must be in [0, 1); got {self.depth}")

    def __enter__(self):
        import nilearn.plotting.surface._matplotlib_backend as mb
        self._mb, self._orig = mb, mb.mix_colormaps
        depth, orig = self.depth, self._orig

        def mix_shaded(fg, bg_rgba):
            out = orig(fg, bg_rgba)          # handles uncolored faces
            colored = fg[:, 3] > 0           # faces carrying a value color
            if not np.any(colored):
                return out
            # Background is nilearn's Greys map of the sulcal values; its
            # luminance is high on gyri and low in sulci. Most faces sit in a
            # narrow middle band, so stretch between the 20th and 80th
            # percentiles (clipping beyond) to separate gyri from sulci.
            lum = bg_rgba[:, :3].mean(axis=1)
            lo, hi = np.percentile(lum, [20, 80])
            if hi > lo:
                bright = np.clip((lum - lo) / (hi - lo), 0.0, 1.0)
            else:
                bright = np.ones_like(lum)
            shade = (1.0 - depth) + depth * bright
            out[colored, :3] = fg[colored, :3] * shade[colored, None]
            out[colored, 3] = 1.0
            return out

        mb.mix_colormaps = mix_shaded
        return self

    def __exit__(self, *exc):
        self._mb.mix_colormaps = self._orig
        return False


def _draw_view(ax, meshes, bg_maps, data, hemi, view, cmap, vmin, vmax,
               threshold, bg_color):
    """One hemisphere/view of one data set into a 3D axes."""
    from nilearn import plotting
    ax.set_facecolor(bg_color)
    plotting.plot_surf_stat_map(
        meshes[hemi], data[hemi], bg_map=bg_maps[hemi],
        hemi=_HEMI_FULL[hemi], view=view, cmap=cmap,
        vmin=vmin, vmax=vmax, threshold=threshold,
        bg_on_data=True, colorbar=False, axes=ax)


def _add_colorbar(fig, axes, cmap, vmin, vmax, label, txt, **kw):
    """Shared colorbar to the right of the given axes, text in color txt."""
    import matplotlib.pyplot as plt
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, **kw)
    cbar.set_label(label, color=txt)
    cbar.ax.yaxis.set_tick_params(color=txt)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=txt)
    return cbar


def _save_figure(fig, out_path, bg_color, dpi=150, jpg=True, jpg_dpi=300):
    """
    Save the figure as PNG at `dpi` and, if `jpg`, also as a high quality
    JPEG next to it (same name, .jpg) at `jpg_dpi`: quality 98, no chroma
    subsampling (4:4:4, keeps colored region edges crisp), optimized coding.
    Returns the JPEG path, or None.
    """
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=bg_color)
    if not jpg:
        return None
    jpg_path = os.path.splitext(out_path)[0] + ".jpg"
    fig.savefig(jpg_path, format="jpg", dpi=jpg_dpi, bbox_inches="tight",
                facecolor=bg_color,
                pil_kwargs={"quality": 98, "subsampling": 0, "optimize": True})
    return jpg_path


def render_png(data, out_path, subjects_dir, title=None,
               cmap="RdBu_r", vmin=-2.0, vmax=2.0, symmetric=True,
               threshold=None, colorbar_label="z-score",
               surf="pial", bg="sulc", bg_color="black",
               shading=0.75,
               dpi=150, jpg=True, jpg_dpi=300):
    """
    Render a panel of cortical surface views to a PNG using nilearn
    (matplotlib backend, headless-safe).

    Panels: LH lateral, RH lateral, LH medial, RH medial (2x2). Colors are
    combined with the sulcal shading so cortical folding stays visible.

    data : dict 'lh'/'rh' -> per-vertex array (NaN where unmapped)
    surf : cortical surface to render on ('pial', 'white', 'inflated', ...).
    bg_color : figure/axes background color ('black' by default).
    vmin/vmax : color range; default fixed -2..+2 for z-scores. Pass None with
        symmetric=True to auto-scale symmetrically about 0.
    threshold : values with abs < threshold are left transparent.
    shading : how strongly sulci darken the colors, in [0, 1). 0 = flat
        color; 0.75 (default) darkens sulcal fundi to 25% brightness while
        gyral crowns keep full color. Higher shows more gyral pattern.

    dpi / jpg / jpg_dpi : PNG resolution; also write a high quality JPEG
        (same name, .jpg) at jpg_dpi (default 300).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meshes, bg_maps = _surface_files(subjects_dir, surf, bg)
    vmin, vmax = _color_limits([data], vmin, vmax, symmetric)
    txt = "white" if bg_color == "black" else "black"

    # Layout: 2x2 grid (LH/RH lateral on top, LH/RH medial below) with a
    # shared colorbar on the right.
    fig = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor(bg_color)

    axes = []
    with _SulcalShading(shading):
        for pos, (hemi, view) in enumerate(VIEWS, start=1):
            ax = fig.add_subplot(2, 2, pos, projection="3d")
            _draw_view(ax, meshes, bg_maps, data, hemi, view, cmap, vmin, vmax,
                       threshold, bg_color)
            ax.set_title(f"{hemi.upper()} {view}", fontsize=10, color=txt)
            axes.append(ax)

    _add_colorbar(fig, axes, cmap, vmin, vmax, colorbar_label, txt,
                  fraction=0.025, pad=0.04, shrink=0.6)

    if title:
        fig.suptitle(title, fontsize=13, color=txt)

    _save_figure(fig, out_path, bg_color, dpi, jpg, jpg_dpi)
    plt.close(fig)
    return out_path


def render_grid(columns, column_labels, out_path, subjects_dir, title=None,
                cmap="RdBu_r", vmin=-2.0, vmax=2.0, symmetric=True,
                threshold=None, colorbar_label="z-score",
                surf="pial", bg="sulc", bg_color="black", shading=0.75,
                coarsen=1,
                dpi=150, jpg=True, jpg_dpi=300):
    """
    Render several data sets side by side: one column per data set (e.g. one
    per frequency band) and four rows, LH lateral, RH lateral, LH medial,
    RH medial. Same coloring and shading as render_png, one shared color
    scale and colorbar for the whole figure.

    columns       : list of dicts 'lh'/'rh' -> per-vertex arrays (full fsaverage)
    column_labels : list of column headings, same length as columns
    coarsen       : draw each panel on a mesh this many icosahedral levels
                    coarser than the full fsaverage (default 1: fsaverage6,
                    81920 triangles per hemisphere). Each grid panel is only a
                    few hundred pixels wide, so the full mesh adds no visible
                    detail but costs several GB of memory per figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(columns) != len(column_labels):
        raise ValueError("columns and column_labels differ in length")
    if not columns:
        raise ValueError("nothing to plot")
    meshes, bg_maps, n_vert = _load_meshes(subjects_dir, surf, bg, coarsen)
    # Values of the coarse mesh are the first n_vert entries (nested meshes).
    columns = [{h: np.asarray(d[h])[:n_vert] for h in ("lh", "rh")} for d in columns]
    vmin, vmax = _color_limits(columns, vmin, vmax, symmetric)
    txt = "white" if bg_color == "black" else "black"

    n_rows, n_cols = len(VIEWS), len(columns)
    fig = plt.figure(figsize=(2.6 * n_cols + 1.6, 2.1 * n_rows + 0.9))
    fig.patch.set_facecolor(bg_color)

    axes = []
    with _SulcalShading(shading):
        for r, (hemi, view) in enumerate(VIEWS):
            for c, (data, label) in enumerate(zip(columns, column_labels)):
                ax = fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1,
                                     projection="3d")
                _draw_view(ax, meshes, bg_maps, data, hemi, view, cmap, vmin,
                           vmax, threshold, bg_color)
                if r == 0:
                    ax.set_title(label, fontsize=12, color=txt)
                if c == 0:
                    ax.text2D(-0.02, 0.5, f"{hemi.upper()} {view}",
                              transform=ax.transAxes, rotation=90,
                              ha="right", va="center", fontsize=10, color=txt)
                axes.append(ax)
    fig.subplots_adjust(left=0.05, right=0.9, top=0.92, bottom=0.02,
                        wspace=0.0, hspace=0.0)

    _add_colorbar(fig, axes, cmap, vmin, vmax, colorbar_label, txt,
                  fraction=0.015, pad=0.02, shrink=0.5)

    if title:
        fig.suptitle(title, fontsize=13, color=txt)

    _save_figure(fig, out_path, bg_color, dpi, jpg, jpg_dpi)
    plt.close(fig)
    return out_path


def render_html(data, out_path, subjects_dir, title=None,
                cmap="RdBu_r", vmin=-2.0, vmax=2.0, symmetric=True,
                threshold=None, surf="pial", bg="sulc", **_):
    """
    Render an interactive rotatable view (one HTML per hemisphere, lateral
    start) using nilearn.plotting.view_surf. Writes <out_path>_lh.html and
    <out_path>_rh.html and returns the two paths.

    Extra keyword arguments (e.g. bg_color, colorbar_label) are accepted and
    ignored so the same kw set can be shared with render_png.
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
    ap.add_argument("--zrange", type=float, default=None, metavar="N",
                    help="Symmetric color range: sets the scale to -N..+N "
                         "(e.g. --zrange 3). Overrides --vmin/--vmax.")
    ap.add_argument("--vmin", type=float, default=-2.0,
                    help="Color range minimum (default -2).")
    ap.add_argument("--vmax", type=float, default=2.0,
                    help="Color range maximum (default +2).")
    ap.add_argument("--surf", default="pial",
                    help="Cortical surface: pial (default), white, inflated.")
    ap.add_argument("--shading", type=float, default=0.75,
                    help="How strongly sulci darken the colors, 0-1 "
                         "(default 0.75). Higher shows more gyral pattern; "
                         "colors stay fully saturated on the gyri.")
    ap.add_argument("--bg-color", default="black",
                    help="Figure background color (default black).")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Hide |value| below this.")
    ap.add_argument("--no-symmetric", action="store_true",
                    help="Do not force a symmetric color range about 0.")
    ap.add_argument("--label", default="z-score",
                    help="Colorbar label.")
    ap.add_argument("--title", default=None)
    ap.add_argument("--no-jpg", action="store_true",
                    help="Do not write the high quality JPEG next to the PNG.")
    ap.add_argument("--jpg-dpi", type=int, default=300,
                    help="JPEG resolution in dpi (default 300).")
    ap.add_argument("--html", action="store_true",
                    help="Also write interactive HTML views.")
    args = ap.parse_args(argv)

    # --zrange N is shorthand for a symmetric scale -N..+N
    if args.zrange is not None:
        if args.zrange <= 0:
            sys.exit("ERROR: --zrange must be a positive number.")
        args.vmin, args.vmax = -abs(args.zrange), abs(args.zrange)

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
            colorbar_label=args.label, title=args.title,
            surf=args.surf, bg_color=args.bg_color,
            shading=args.shading,
            jpg=not args.no_jpg, jpg_dpi=args.jpg_dpi)
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
    if not args.no_jpg:
        print(f"  JPG: {os.path.splitext(res['png'])[0]}.jpg")
    if res.get("html"):
        for h in res["html"]:
            print(f"  HTML: {h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
