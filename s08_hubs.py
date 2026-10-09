"""
Stage 08: Hub analysis and group contrasts.

The 90 AAL regions are grouped into hemisphere-specific hubs (see hubs.py:
mesial, polar, lateral and basal temporal; orbitofrontal, dorsolateral,
inferior and medial frontal; pericentral; insula; superior, inferior and
medial parietal; plus occipital, thalamus and basal ganglia).

Per session, for each band, five hub measures are computed:
  strength    mean AECc between the hub's regions and all regions outside it
              (how strongly the hub is connected to the rest of the brain)
  within      mean AECc among the hub's own regions (NaN for one-region hubs)
  roi_bc      mean MST betweenness centrality of the hub's regions (90-node trees)
  roi_degree  mean MST degree of the hub's regions (90-node trees)
  hub_bc      betweenness centrality of the hub in an MST built on the
              hub-to-hub AECc matrix (one node per hub)
Tree measures are computed per epoch and averaged over epochs.

Group level, for each session label: the target group is compared with each
reference group, and the reference groups with each other (Mann-Whitney U,
Cliff's delta, Benjamini-Hochberg FDR across hubs within each band and
measure). Each target subject is also expressed as a z-score relative to each
reference group, which is usable even when the target group is small.

Outputs:
  derivatives/<subject>/<session>/hub_metrics.csv, hub_matrix.npz
  derivatives/group/hubs/<session>/hub_metrics_all.csv, contrasts.csv,
      zscores.csv, group_hub_matrices.npz, contrast_<measure>.png
"""
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, mannwhitneyu

import config
from common import (earlier_failure, load_subjects, mst_edges, parse_args, require,
                    run_each, session_dir)
from hubs import assign_hubs

MEASURES = ["strength", "within", "roi_bc", "roi_degree", "hub_bc"]
FIGURE_MEASURES = ["hub_bc", "roi_bc", "strength"]
MIN_GROUP_N = 3          # smallest group size for a Mann-Whitney test
MIN_REF_N_FOR_Z = 3      # smallest reference group for z-scores


# ----------------------------------------------------------------------------
# Per session
# ----------------------------------------------------------------------------
def hub_matrix(conn, hub_names, members):
    """Hub x hub matrix: mean connectivity between (off diagonal) and within (diagonal) hubs."""
    n_hub = len(hub_names)
    h = np.full((n_hub, n_hub), np.nan)
    for a, ha in enumerate(hub_names):
        ia = members[ha]
        for b, hb in enumerate(hub_names):
            ib = members[hb]
            block = conn[np.ix_(ia, ib)]
            if a == b:
                if len(ia) > 1:   # mean of off-diagonal entries within the hub
                    h[a, b] = block[~np.eye(len(ia), dtype=bool)].mean()
            else:
                h[a, b] = block.mean()
    return h


def tree_bc(conn):
    """Normalized betweenness centrality of each node in the maximum spanning tree of conn."""
    n = conn.shape[0]
    tree = nx.Graph()
    tree.add_nodes_from(range(n))
    tree.add_edges_from(mst_edges(conn).tolist())
    bc = nx.betweenness_centrality(tree, normalized=True)
    return np.array([bc[i] for i in range(n)])


def session_hub_metrics(subj, ses, group):
    """Compute the hub measures for one session; returns a long DataFrame and hub matrices."""
    out = session_dir(subj, ses)
    a = np.load(require(out / "aecc.npz", "s05_connectivity.py"))
    m = np.load(require(out / "mst.npz", "s06_mst.py"))
    aecc_epochs, bands, names = a["aecc_epochs"], list(a["bands"]), list(a["names"])
    edges, roi_bc_all = m["edges"], m["bc"]     # (bands, epochs, N-1, 2), (bands, epochs, N)
    n_bands, n_epochs, n_roi, _ = aecc_epochs.shape
    hub_names, members = assign_hubs(names)
    n_hub = len(hub_names)

    rows, mats = [], np.zeros((n_bands, n_hub, n_hub))
    for b, band in enumerate(bands):
        mean_conn = aecc_epochs[b].mean(axis=0)
        mats[b] = hub_matrix(mean_conn, hub_names, members)

        # ROI level tree measures, averaged over epochs.
        degree = np.zeros(n_roi)
        for e in range(n_epochs):
            degree += np.bincount(edges[b, e].ravel(), minlength=n_roi)
        degree /= n_epochs
        roi_bc = roi_bc_all[b].mean(axis=0)

        # Hub level tree: MST on the hub x hub matrix of each epoch.
        hub_bc = np.zeros(n_hub)
        for e in range(n_epochs):
            h = hub_matrix(aecc_epochs[b, e], hub_names, members)
            np.fill_diagonal(h, 0.0)   # within-hub values are not edges of the hub graph
            hub_bc += tree_bc(h)
        hub_bc /= n_epochs

        for k, hub in enumerate(hub_names):
            idx = members[hub]
            outside = np.setdiff1d(np.arange(n_roi), idx)
            values = {
                "strength": mean_conn[np.ix_(idx, outside)].mean(),
                "within": mats[b, k, k],
                "roi_bc": roi_bc[idx].mean(),
                "roi_degree": degree[idx].mean(),
                "hub_bc": hub_bc[k],
            }
            for measure, value in values.items():
                rows.append(dict(subject=subj, session=ses, group=group, band=band,
                                 hub=hub, n_regions=len(idx), measure=measure,
                                 value=float(value)))
    return pd.DataFrame(rows), mats, hub_names, bands


# ----------------------------------------------------------------------------
# Group level
# ----------------------------------------------------------------------------
def cliffs_delta(u, n1, n2):
    """Cliff's delta from the Mann-Whitney U of the first sample (-1 to 1)."""
    return 2.0 * u / (n1 * n2) - 1.0


def contrasts(df, pairs):
    """Mann-Whitney U per (pair, band, measure, hub) with BH FDR across hubs."""
    out = []
    for g1, g2 in pairs:
        for (band, measure), sub in df.groupby(["band", "measure"], sort=False):
            block = []
            for hub, h in sub.groupby("hub", sort=False):
                x = h.loc[h["group"] == g1, "value"].dropna().to_numpy()
                y = h.loc[h["group"] == g2, "value"].dropna().to_numpy()
                if len(x) < MIN_GROUP_N or len(y) < MIN_GROUP_N:
                    continue
                res = mannwhitneyu(x, y, alternative="two-sided")
                block.append(dict(contrast=f"{g1} vs {g2}", band=band, measure=measure,
                                  hub=hub, n1=len(x), n2=len(y),
                                  median1=np.median(x), median2=np.median(y),
                                  U=res.statistic, p=res.pvalue,
                                  cliffs_delta=cliffs_delta(res.statistic, len(x), len(y))))
            if block:
                block = pd.DataFrame(block)
                # p is NaN when all values are tied (e.g. every value zero); leave
                # those out of the correction and report p_fdr as NaN.
                ok = np.isfinite(block["p"].to_numpy())
                block["p_fdr"] = np.nan
                if ok.any():
                    block.loc[ok, "p_fdr"] = false_discovery_control(
                        block.loc[ok, "p"].to_numpy(), method="bh")
                out.append(block)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def zscores(df, target, refs):
    """z-score of each target subject's value relative to each reference group."""
    out = []
    for ref in refs:
        ref_stats = (df[df["group"] == ref].groupby(["band", "measure", "hub"])["value"]
                     .agg(["mean", "std", "count"]).reset_index())
        ref_stats = ref_stats[ref_stats["count"] >= MIN_REF_N_FOR_Z]
        t = df[df["group"] == target].merge(ref_stats, on=["band", "measure", "hub"])
        with np.errstate(divide="ignore", invalid="ignore"):
            t["z"] = (t["value"] - t["mean"]) / t["std"].where(t["std"] > 0)
        t["reference"] = ref
        out.append(t[["subject", "session", "reference", "band", "measure", "hub",
                      "value", "mean", "std", "count", "z"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def plot_contrasts(con, zs, measure, hub_names, bands, target, refs, fname):
    """
    Heatmap, hubs x bands, one panel per reference group: Cliff's delta of
    target vs reference when it could be tested, otherwise the mean z-score
    of the target subjects (scaled to +-1 at z = +-3). '*' marks p_fdr < 0.05.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # Diverging blue - neutral gray - red, as in the dataviz reference palette.
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 9})

    fig, axes = plt.subplots(1, len(refs), figsize=(3.2 * len(refs) + 1.6, 0.24 * len(hub_names) + 1.6),
                             sharey=True, squeeze=False)
    image = None
    for ax, ref in zip(axes[0], refs):
        grid = np.full((len(hub_names), len(bands)), np.nan)
        stars = np.zeros_like(grid, dtype=bool)
        label = "Cliff's delta"
        c = (con[(con["contrast"] == f"{target} vs {ref}") & (con["measure"] == measure)]
             if len(con) else con)
        if len(c):
            for _, r in c.iterrows():
                i, j = hub_names.index(r["hub"]), bands.index(r["band"])
                grid[i, j], stars[i, j] = r["cliffs_delta"], r["p_fdr"] < 0.05
        elif len(zs):
            label = "mean z / 3"
            z = zs[(zs["reference"] == ref) & (zs["measure"] == measure)]
            for (hub, band), v in z.groupby(["hub", "band"])["z"].mean().items():
                grid[hub_names.index(hub), bands.index(band)] = np.clip(v / 3.0, -1, 1)
        image = ax.imshow(grid, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        for i, j in zip(*np.nonzero(stars)):
            ax.text(j, i, "*", ha="center", va="center", fontsize=11, color="#222222")
        ax.set_xticks(range(len(bands)), bands, rotation=45, ha="right")
        ax.set_yticks(range(len(hub_names)), hub_names)
        ax.set_title(f"{target} vs {ref}\n({label})", fontsize=9)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(image, ax=axes[0].tolist(), shrink=0.6, pad=0.02)
    cbar.set_label(f"{measure}: higher in {target}  →", fontsize=8)
    cbar.outline.set_visible(False)
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args(__doc__.splitlines()[1])
    rows = load_subjects(args.subject, args.session)

    def process(row):
        subj, ses = row["subject"], row["session"]
        print(f"\n=== {subj} / {ses} ===")
        df, mats, hub_names, bands = session_hub_metrics(subj, ses, row["group"])
        out = session_dir(subj, ses)
        df.to_csv(out / "hub_metrics.csv", index=False)
        np.savez(out / "hub_matrix.npz", hub_matrix=mats, hubs=hub_names, bands=bands)
        print(f"{len(hub_names)} hubs x {len(bands)} bands")

    run_each(rows, "s08", process)

    # ---- Group level, run on every row of subjects.csv.
    target, refs = config.TARGET_GROUP, list(config.REFERENCE_GROUPS)
    pairs = [(target, r) for r in refs] + [(refs[i], refs[j]) for i in range(len(refs))
                                           for j in range(i + 1, len(refs))]
    all_rows = load_subjects()
    for ses, ses_rows in all_rows.groupby("session"):
        frames, mats, groups = [], [], []
        for _, r in ses_rows.iterrows():
            d = session_dir(r["subject"], ses)
            if (d / "hub_metrics.csv").exists() and earlier_failure(r["subject"], ses, "s08") is None \
                    and not (d / "FAILED_s08.txt").exists():
                frames.append(pd.read_csv(d / "hub_metrics.csv"))
                mats.append(np.load(d / "hub_matrix.npz")["hub_matrix"])
                groups.append(r["group"])
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        hub_names = list(dict.fromkeys(df["hub"]))
        bands = list(dict.fromkeys(df["band"]))
        gdir = config.DERIV_DIR / "group" / "hubs" / ses
        gdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(gdir / "hub_metrics_all.csv", index=False)

        counts = df.drop_duplicates("subject")["group"].value_counts().to_dict()
        print(f"\n=== Group hub analysis, session {ses}: {counts} ===")
        missing = [g for g in [target] + refs if g not in counts]
        if missing:
            print(f"No subjects in group(s) {missing}; check REFERENCE_GROUPS and "
                  f"TARGET_GROUP in config.py against the group column of subjects.csv.")

        # Group mean hub matrices (bands x hubs x hubs) for each group.
        mats, groups = np.array(mats), np.array(groups)
        np.savez(gdir / "group_hub_matrices.npz", hubs=hub_names, bands=bands,
                 **{str(g): mats[groups == g].mean(axis=0) for g in np.unique(groups)})

        con = contrasts(df, pairs)
        zs = zscores(df, target, refs)
        con.to_csv(gdir / "contrasts.csv", index=False)
        zs.to_csv(gdir / "zscores.csv", index=False)
        if len(con):
            sig = con[con["p_fdr"] < 0.05]
            print(f"{len(con)} tests, {len(sig)} with FDR-corrected p < 0.05")
            if len(sig):
                print(sig.sort_values("p_fdr")[["contrast", "band", "measure", "hub",
                                                 "cliffs_delta", "p_fdr"]]
                      .head(20).round(4).to_string(index=False))
        else:
            print(f"No group tests (need at least {MIN_GROUP_N} subjects per group); "
                  f"z-scores only.")
        for measure in FIGURE_MEASURES:
            plot_contrasts(con, zs, measure, hub_names, bands, target, refs,
                           gdir / f"contrast_{measure}.png")
        print(f"Results in {gdir}")


if __name__ == "__main__":
    main()
