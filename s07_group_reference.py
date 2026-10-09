"""
Stage 07: Reference MSTs and network similarity (group level).

For each session label (e.g. baseline, followup):
  - Reference MST per reference group: MST of the mean AECc matrix over all
    subjects in that group (each subject's matrix already averaged over epochs).
  - For each target group subject: network similarity = fraction of MST edges
    shared with the reference MST, computed per epoch, then averaged.
  - If enough subjects: Wilcoxon signed-rank test per band comparing similarity
    to reference A vs reference B, with Benjamini-Hochberg FDR across bands.

If a target subject has two sessions, a paired baseline vs follow-up test is
also run for each reference. Run this only after stages 01-06 for all subjects.

Outputs (in DERIV_DIR/group):
  network_similarity.csv, stats_between_references.csv, stats_longitudinal.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, wilcoxon

import config
from common import (earlier_failure, edge_overlap, load_subjects, mst_edges, require,
                    session_dir)


def load_aecc(subj, ses):
    f = np.load(require(session_dir(subj, ses) / "aecc.npz", "s05_connectivity.py"))
    return f["aecc_epochs"], f["aecc_mean"], list(f["bands"])


def wilcoxon_fdr(a, b, bands, label):
    """Paired Wilcoxon per band with BH FDR. a, b: (n_subjects, n_bands)."""
    rows = []
    for i, band in enumerate(bands):
        res = wilcoxon(a[:, i], b[:, i])
        rows.append({"comparison": label, "band": band, "n": a.shape[0],
                     "median_a": np.median(a[:, i]), "median_b": np.median(b[:, i]),
                     "W": res.statistic, "p": res.pvalue})
    df = pd.DataFrame(rows)
    df["p_fdr"] = false_discovery_control(df["p"].to_numpy(), method="bh")
    return df


def main():
    subjects = load_subjects()
    out_dir = config.DERIV_DIR / "group"
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_rows = []
    for ses, ses_rows in subjects.groupby("session"):
        print(f"\n=== Session {ses} ===")
        # Leave out sessions that failed or have no connectivity results.
        usable = [(session_dir(r["subject"], ses) / "aecc.npz").exists()
                  and earlier_failure(r["subject"], ses, "s06") is None
                  and not (session_dir(r["subject"], ses) / "FAILED_s06.txt").exists()
                  for _, r in ses_rows.iterrows()]
        excluded = ses_rows.loc[[not u for u in usable], "subject"].tolist()
        if excluded:
            print(f"Excluded (no aecc.npz, see failures.log): {excluded}")
        ses_rows = ses_rows[usable]

        # --- Reference MSTs.
        ref_trees, bands = {}, None
        for grp in config.REFERENCE_GROUPS:
            members = ses_rows[ses_rows["group"] == grp]
            if members.empty:
                print(f"No subjects in reference group '{grp}'; skipping session.")
                break
            means = []
            for subj in members["subject"]:
                _, mean, bands = load_aecc(subj, ses)
                means.append(mean)
            group_mean = np.mean(means, axis=0)               # (n_bands, N, N)
            ref_trees[grp] = [mst_edges(group_mean[b]) for b in range(len(bands))]
            print(f"Reference '{grp}': {len(members)} subjects")
        if len(ref_trees) != len(config.REFERENCE_GROUPS):
            continue

        # --- Similarity of each target subject to each reference.
        for subj in ses_rows.loc[ses_rows["group"] == config.TARGET_GROUP, "subject"]:
            epochs, _, _ = load_aecc(subj, ses)
            for b, band in enumerate(bands):
                for grp, trees in ref_trees.items():
                    sims = [edge_overlap(mst_edges(epochs[b, e]), trees[b])
                            for e in range(epochs.shape[1])]
                    sim_rows.append({"subject": subj, "session": ses, "band": band,
                                     "reference": grp, "similarity": float(np.mean(sims))})

    if not sim_rows:
        print("No similarity values computed; check groups in subjects.csv and config.py.")
        return
    sim = pd.DataFrame(sim_rows)
    sim.to_csv(out_dir / "network_similarity.csv", index=False)
    print(f"\nWrote {len(sim)} similarity values.")

    # --- Reference A vs reference B, within each session.
    ref_a, ref_b = config.REFERENCE_GROUPS[:2]
    wide = sim.pivot_table(index=["session", "subject"], columns=["reference", "band"],
                           values="similarity")
    band_order = list(config.BANDS)
    between = []
    for ses in wide.index.get_level_values("session").unique():
        w = wide.loc[ses].dropna()
        if len(w) < config.MIN_SUBJECTS_FOR_STATS:
            print(f"Session {ses}: n = {len(w)}, below MIN_SUBJECTS_FOR_STATS; no test.")
            continue
        df = wilcoxon_fdr(w[ref_a][band_order].to_numpy(), w[ref_b][band_order].to_numpy(),
                          band_order, f"{ses}: {ref_a} vs {ref_b}")
        between.append(df)
    if between:
        pd.concat(between).to_csv(out_dir / "stats_between_references.csv", index=False)

    # --- Longitudinal: first vs second session, paired by subject.
    sessions = sorted(sim["session"].unique())
    if len(sessions) >= 2:
        s1, s2 = sessions[:2]
        long_rows = []
        for ref in config.REFERENCE_GROUPS:
            t = sim[sim["reference"] == ref].pivot_table(index="subject",
                                                         columns=["session", "band"],
                                                         values="similarity").dropna()
            if len(t) < config.MIN_SUBJECTS_FOR_STATS:
                print(f"Longitudinal ({ref}): n = {len(t)} paired; no test.")
                continue
            long_rows.append(wilcoxon_fdr(t[s1][band_order].to_numpy(),
                                          t[s2][band_order].to_numpy(),
                                          band_order, f"{ref}: {s1} vs {s2}"))
        if long_rows:
            pd.concat(long_rows).to_csv(out_dir / "stats_longitudinal.csv", index=False)


if __name__ == "__main__":
    main()
