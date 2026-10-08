"""
Stage 02: MEG to MRI coregistration by surface matching.

Fiducials are first estimated from the MNI template, then refined by ICP
fitting of the digitized head shape points to the FreeSurfer scalp surface.
This follows the MNE automated coregistration tutorial. Outlier head shape
points (> 5 mm from the scalp after the first fit) are dropped before the
final fit. The result is written as <session>/coreg-trans.fif.

Visual inspection is still recommended, as in the paper:
    mne coreg -s <fs_subject> -d <SUBJECTS_DIR> -f <raw_fif> \
        --trans <derivatives>/<subject>/<session>/coreg-trans.fif
"""
import mne
import numpy as np
from mne.coreg import Coregistration

import config
from common import load_subjects, parse_args, session_dir


def main():
    args = parse_args(__doc__.splitlines()[1])
    rows = load_subjects(args.subject, args.session)

    for _, row in rows.iterrows():
        subj, ses, fs_subj = row["subject"], row["session"], row["fs_subject"]
        print(f"\n=== {subj} / {ses} ===")
        info = mne.io.read_info(row["raw_fif"])

        coreg = Coregistration(info, subject=fs_subj,
                               subjects_dir=config.SUBJECTS_DIR,
                               fiducials="estimated")
        coreg.fit_fiducials(verbose=False)
        coreg.fit_icp(n_iterations=20, nasion_weight=2.0, verbose=False)
        coreg.omit_head_shape_points(distance=5.0 / 1000)   # metres
        coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=False)

        # Distance from each remaining head shape point to the scalp, in mm.
        dists_mm = coreg.compute_dig_mri_distances() * 1000
        if dists_mm.size == 0:
            raise RuntimeError("No head shape points left after outlier removal.")
        mean_mm = float(np.mean(dists_mm))
        print(f"Head shape to scalp: mean {mean_mm:.1f} mm, "
              f"median {np.median(dists_mm):.1f} mm, max {np.max(dists_mm):.1f} mm, "
              f"n = {dists_mm.size}")
        if mean_mm > config.COREG_WARN_MM:
            print(f"WARNING: mean distance exceeds {config.COREG_WARN_MM} mm; "
                  f"check this coregistration visually.")

        out = session_dir(subj, ses)
        mne.write_trans(out / "coreg-trans.fif", coreg.trans, overwrite=True)
        np.savetxt(out / "coreg_distances_mm.txt", dists_mm, fmt="%.2f")


if __name__ == "__main__":
    main()
