"""
Helpers shared by the pipeline stages: subject table, output paths, and the
minimum spanning tree (MST) functions.
"""
import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

import config

REQUIRED_COLUMNS = ["subject", "session", "raw_fif", "er_fif", "mri_fif",
                    "fs_subject", "group"]


# ----------------------------------------------------------------------------
# Subject table and paths
# ----------------------------------------------------------------------------
def parse_args(description):
    """Standard command line: optionally restrict to one subject and/or session."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--subject", default=None, help="process only this subject")
    parser.add_argument("--session", default=None, help="process only this session")
    return parser.parse_args()


def load_subjects(subject=None, session=None):
    """Read subjects.csv and return the rows to process as a DataFrame."""
    if not config.SUBJECTS_CSV.exists():
        raise FileNotFoundError(f"Subject table not found: {config.SUBJECTS_CSV}")
    df = pd.read_csv(config.SUBJECTS_CSV, dtype=str).fillna("")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"subjects.csv is missing columns: {missing}")
    if subject is not None:
        df = df[df["subject"] == subject]
    if session is not None:
        df = df[df["session"] == session]
    if df.empty:
        raise ValueError("No rows in subjects.csv match the requested subject/session.")
    return df


def session_dir(subject, session):
    """Output folder for one recording session (created if needed)."""
    d = config.DERIV_DIR / subject / session
    d.mkdir(parents=True, exist_ok=True)
    return d


def anat_dir(subject):
    """Output folder for session-independent anatomy (created if needed)."""
    d = config.DERIV_DIR / subject / "anat"
    d.mkdir(parents=True, exist_ok=True)
    return d


def require(path, made_by):
    """Stop with a clear message if an input from an earlier stage is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run {made_by} first.")
    return path


# ----------------------------------------------------------------------------
# Minimum spanning tree
# ----------------------------------------------------------------------------
def mst_edges(conn):
    """
    Return the maximum spanning tree of a connectivity matrix as a sorted
    (N-1, 2) array of node pairs (i < j).

    The paper runs Kruskal's algorithm on 1/AECc, which keeps the strongest
    connections; that is the same tree as the maximum spanning tree on AECc
    itself, and this form also works if any AECc values are negative.
    """
    conn = np.asarray(conn, dtype=float)
    n = conn.shape[0]
    if conn.ndim != 2 or conn.shape[1] != n or n < 2:
        raise ValueError(f"Connectivity matrix must be square, got {conn.shape}")
    if not np.all(np.isfinite(conn[np.triu_indices(n, 1)])):
        raise ValueError("Connectivity matrix contains NaN or infinite values.")

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    iu, ju = np.triu_indices(n, 1)
    graph.add_weighted_edges_from(zip(iu.tolist(), ju.tolist(), conn[iu, ju].tolist()))
    tree = nx.maximum_spanning_tree(graph, algorithm="kruskal")

    edges = np.array(sorted(tuple(sorted(e)) for e in tree.edges()), dtype=int)
    if edges.shape != (n - 1, 2):
        raise RuntimeError(f"Spanning tree has {len(edges)} edges, expected {n - 1}.")
    return edges


def mst_metrics(edges, n_nodes):
    """
    Global and regional MST metrics (Stam et al., 2014; Tewarie et al., 2015).

    Returns a dict with:
      leaf_fraction   leaves / M, where M = N - 1 edges
      diameter        longest shortest path, in edges
      diameter_norm   diameter / M
      bc_max          largest normalized betweenness centrality
      tree_hierarchy  leaves / (2 * M * bc_max)
      bc              (N,) normalized betweenness centrality per node
    """
    tree = nx.Graph()
    tree.add_nodes_from(range(n_nodes))
    tree.add_edges_from(edges.tolist())
    m = n_nodes - 1

    degree = np.array([tree.degree(i) for i in range(n_nodes)])
    n_leaves = int(np.sum(degree == 1))
    diameter = nx.diameter(tree)
    bc_dict = nx.betweenness_centrality(tree, normalized=True)
    bc = np.array([bc_dict[i] for i in range(n_nodes)])
    bc_max = float(bc.max())

    # Tree hierarchy is undefined if bc_max is zero (only possible for N <= 2).
    tree_hierarchy = n_leaves / (2.0 * m * bc_max) if bc_max > 0 else np.nan

    return dict(leaf_fraction=n_leaves / m, diameter=diameter,
                diameter_norm=diameter / m, bc_max=bc_max,
                tree_hierarchy=tree_hierarchy, bc=bc)


def edge_overlap(edges_a, edges_b):
    """Fraction of edges shared by two spanning trees on the same nodes (0 to 1)."""
    if edges_a.shape != edges_b.shape:
        raise ValueError("Trees must have the same number of edges to be compared.")
    set_a = {tuple(e) for e in edges_a.tolist()}
    set_b = {tuple(e) for e in edges_b.tolist()}
    return len(set_a & set_b) / len(set_a)


# ----------------------------------------------------------------------------
# Per-subject error handling: a failure is recorded and the run continues
# ----------------------------------------------------------------------------
STAGE_ORDER = ["s01", "s02", "s03", "s04", "s05", "s06"]


def _marker_dirs(subject, session):
    """Folders that may hold failure markers for a subject (and session)."""
    dirs = [config.DERIV_DIR / subject / "anat"]
    if session:
        dirs.append(config.DERIV_DIR / subject / session)
    return dirs


def earlier_failure(subject, session, stage):
    """Return the marker file of a failure in an earlier stage, or None."""
    idx = STAGE_ORDER.index(stage)
    for d in _marker_dirs(subject, session):
        for marker in sorted(d.glob("FAILED_*.txt")) if d.is_dir() else []:
            failed_stage = marker.stem[len("FAILED_"):]
            if failed_stage in STAGE_ORDER and STAGE_ORDER.index(failed_stage) < idx:
                return marker
    return None


def run_each(rows, stage, process, per_session=True):
    """
    Call process(row) for every row of the subject table. If it raises an
    exception, write derivatives/<subject>/<session or anat>/FAILED_<stage>.txt
    with the error, add a line to derivatives/failures.log, and go on to the next
    row. Rows whose subject or session failed in an earlier stage are skipped.
    A successful run removes this stage's old failure marker.
    """
    import datetime
    import traceback

    failed, skipped = [], []
    for _, row in rows.iterrows():
        subj = row["subject"]
        ses = row["session"] if per_session else None
        label = f"{subj} / {ses}" if ses else subj
        marker_dir = config.DERIV_DIR / subj / (ses if ses else "anat")
        marker = marker_dir / f"FAILED_{stage}.txt"

        blocker = earlier_failure(subj, ses, stage)
        if blocker is not None:
            print(f"\n=== {label}: skipped (failed earlier, see {blocker}) ===")
            skipped.append(label)
            continue
        try:
            process(row)
        except Exception as err:
            text = traceback.format_exc()
            marker_dir.mkdir(parents=True, exist_ok=True)
            marker.write_text(text)
            stamp = datetime.datetime.now().isoformat(timespec="seconds")
            config.DERIV_DIR.mkdir(parents=True, exist_ok=True)
            with open(config.DERIV_DIR / "failures.log", "a") as log:
                log.write(f"{stamp}\t{stage}\t{label}\t{type(err).__name__}: {err}\n")
            print(f"\nERROR in {stage} for {label}: {type(err).__name__}: {err}")
            print(f"Details in {marker}; continuing with the next subject.")
            failed.append(label)
        else:
            if marker.exists():
                marker.unlink()   # an earlier failure of this stage is now fixed

    if failed or skipped:
        print(f"\n{stage}: {len(failed)} failed {failed}, "
              f"{len(skipped)} skipped because of earlier failures")
    return failed


def print_failure_summary():
    """List every subject and session that currently has a failure marker."""
    markers = sorted(config.DERIV_DIR.glob("*/*/FAILED_*.txt"))
    if not markers:
        print("No failures.")
        return
    print(f"{len(markers)} failure(s); details in each file:")
    for m in markers:
        lines = m.read_text().strip().splitlines()
        print(f"  {m.relative_to(config.DERIV_DIR)}: {lines[-1] if lines else ''}")
