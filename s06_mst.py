"""
Stage 06: Individual MSTs and MST metrics.

For each band and epoch, build the MST from the AECc matrix and compute leaf
fraction, diameter, tree hierarchy and betweenness centrality; then average
the metrics over epochs, as in the paper.

Outputs per session:
  mst.npz              edges (n_bands, n_epochs, 89, 2) and per-node BC
  mst_metrics.csv      global metrics, mean over epochs, one row per band
  mst_bc.csv           betweenness centrality per ROI and band, mean over epochs
"""
import numpy as np
import pandas as pd

from common import load_subjects, mst_edges, mst_metrics, parse_args, require, session_dir

GLOBAL_METRICS = ["leaf_fraction", "diameter", "diameter_norm", "bc_max", "tree_hierarchy"]


def main():
    args = parse_args(__doc__.splitlines()[1])
    rows = load_subjects(args.subject, args.session)

    for _, row in rows.iterrows():
        subj, ses = row["subject"], row["session"]
        print(f"\n=== {subj} / {ses} ===")
        out = session_dir(subj, ses)
        f = np.load(require(out / "aecc.npz", "s05_connectivity.py"))
        aecc, bands, names = f["aecc_epochs"], list(f["bands"]), list(f["names"])
        n_bands, n_epochs, n_nodes, _ = aecc.shape

        edges = np.zeros((n_bands, n_epochs, n_nodes - 1, 2), dtype=int)
        bc = np.zeros((n_bands, n_epochs, n_nodes))
        metric_rows = []
        for b in range(n_bands):
            per_epoch = []
            for e in range(n_epochs):
                edges[b, e] = mst_edges(aecc[b, e])
                m = mst_metrics(edges[b, e], n_nodes)
                bc[b, e] = m.pop("bc")
                per_epoch.append(m)
            means = pd.DataFrame(per_epoch)[GLOBAL_METRICS].mean()
            metric_rows.append({"subject": subj, "session": ses, "band": bands[b], **means})

        metrics = pd.DataFrame(metric_rows)
        metrics.to_csv(out / "mst_metrics.csv", index=False)
        pd.DataFrame(bc.mean(axis=1).T, index=names, columns=bands).to_csv(out / "mst_bc.csv")
        np.savez(out / "mst.npz", edges=edges, bc=bc, bands=bands, names=names)
        print(metrics[["band"] + GLOBAL_METRICS].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
