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

Output: zscores_cohort.csv next to the input, with columns
    subject, session, group, band, measure, hub, value, mean, std, count, z
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIN_COHORT_N = 3     # smallest number of subjects for a z-score


def zscores_cohort(df):
    """
    Cohort z-scores from a long hub table (hub_metrics_all.csv layout:
    subject, session, group, band, hub, measure, value). One session is
    expected; with several, each session is scored separately.
    """
    keys = ["session", "band", "measure", "hub"]
    stats = (df.groupby(keys)["value"].agg(["mean", "std", "count"]).reset_index())
    stats = stats[stats["count"] >= MIN_COHORT_N]
    out = df.merge(stats, on=keys)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["z"] = (out["value"] - out["mean"]) / out["std"].where(out["std"] > 0)
    return out[["subject", "session", "group", "band", "measure", "hub",
                "value", "mean", "std", "count", "z"]]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        sys.exit(__doc__)
    src = Path(argv[0])
    df = pd.read_csv(src)
    needed = {"subject", "session", "group", "band", "measure", "hub", "value"}
    if not needed <= set(df.columns):
        sys.exit(f"ERROR: {src} lacks columns {sorted(needed - set(df.columns))}; "
                 "expected hub_metrics_all.csv from s08_hubs.py.")
    out = zscores_cohort(df)
    dst = src.parent / "zscores_cohort.csv"
    out.to_csv(dst, index=False)
    counts = df.drop_duplicates("subject")["group"].value_counts().to_dict()
    print(f"Cohort of {sum(counts.values())} subjects {counts}; wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
