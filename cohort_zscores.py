#!/usr/bin/env python3
"""
cohort_zscores.py

z-score every subject's hub values against the whole cohort: all subjects of
the session pooled, whatever their group (target and reference groups
together). For each band, measure and hub:

    z = (value - cohort mean) / cohort standard deviation   (ddof = 1)

This complements zscores.csv from stage 08, which scores target subjects
against one reference group at a time. s08_hubs.py writes the result as
zscores_cohort.csv; this script makes it from an existing hub_metrics_all.csv
without rerunning stage 08:

    python3 cohort_zscores.py derivatives/group/hubs/ses01/hub_metrics_all.csv

With --baseline, the mean and standard deviation come from one group only
(for example the normal controls), and every subject of every group is
scored against that group:

    python3 cohort_zscores.py derivatives/group/hubs/ses01/hub_metrics_all.csv \
        --baseline controls

Output, next to the input: zscores_cohort.csv, or zscores_vs_<baseline>.csv
with --baseline. Columns:
    subject, session, group, band, measure, hub, value, mean, std, count, z,
    baseline   ("cohort" or the baseline group label)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIN_COHORT_N = 3     # smallest number of subjects for a z-score


def zscores_cohort(df, baseline=None):
    """
    Cohort z-scores from a long hub table (hub_metrics_all.csv layout:
    subject, session, group, band, hub, measure, value). One session is
    expected; with several, each session is scored separately.

    baseline: None pools all groups for the mean and standard deviation;
    a group label uses only that group's subjects. All subjects are scored
    either way.
    """
    keys = ["session", "band", "measure", "hub"]
    ref = df if baseline is None else df[df["group"] == baseline]
    stats = (ref.groupby(keys)["value"].agg(["mean", "std", "count"]).reset_index())
    stats = stats[stats["count"] >= MIN_COHORT_N]
    out = df.merge(stats, on=keys)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["z"] = (out["value"] - out["mean"]) / out["std"].where(out["std"] > 0)
    out["baseline"] = "cohort" if baseline is None else baseline
    return out[["subject", "session", "group", "band", "measure", "hub",
                "value", "mean", "std", "count", "z", "baseline"]]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hub_metrics", help="hub_metrics_all.csv from s08_hubs.py")
    ap.add_argument("--baseline", default=None,
                    help="Group label to use as the comparison group (for "
                         "example controls). Default: all groups pooled.")
    args = ap.parse_args(argv)
    src = Path(args.hub_metrics)
    df = pd.read_csv(src)
    needed = {"subject", "session", "group", "band", "measure", "hub", "value"}
    if not needed <= set(df.columns):
        sys.exit(f"ERROR: {src} lacks columns {sorted(needed - set(df.columns))}; "
                 "expected hub_metrics_all.csv from s08_hubs.py.")
    counts = df.drop_duplicates("subject")["group"].value_counts().to_dict()
    if args.baseline is not None and args.baseline not in counts:
        sys.exit(f"ERROR: no subjects in group '{args.baseline}'; groups are "
                 f"{sorted(counts)}.")
    out = zscores_cohort(df, args.baseline)
    name = ("zscores_cohort.csv" if args.baseline is None
            else f"zscores_vs_{args.baseline}.csv")
    dst = src.parent / name
    out.to_csv(dst, index=False)
    basis = ("all groups pooled" if args.baseline is None
             else f"baseline group '{args.baseline}' ({counts[args.baseline]} subjects)")
    print(f"{sum(counts.values())} subjects {counts}; scored against {basis}; "
          f"wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
