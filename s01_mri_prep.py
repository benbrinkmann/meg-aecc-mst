"""
Stage 01: MRI preparation.

For each subject:
  1. Read the MEGIN MRI .fif wrapper, dump its tag structure to a text file,
     and pull out the DICOM slice paths and any coordinate transform it holds.
  2. Run FreeSurfer recon-all on the DICOM series (T1.mgz, brainmask.mgz).
  3. Build the scalp surface used for coregistration and the sphere fit.

Requires FreeSurfer on the PATH with FREESURFER_HOME set.

Note: reading the wrapper uses MNE's private FIF reader (mne._fiff), which may
change between MNE versions. The scalp surface and coregistration in later
stages do not depend on the wrapper's transform; it is saved for reference only.
"""
import os
import subprocess
from pathlib import Path

import mne
from mne._fiff.constants import FIFF
from mne._fiff.open import fiff_open
from mne._fiff.tag import read_tag

import config
from common import anat_dir, load_subjects, parse_args


def read_mri_wrapper(fname):
    """
    Scan every tag in the wrapper. Return (paths, transforms):
      paths       strings that look like file paths, in file order
      transforms  coordinate transforms found in the file
    """
    paths, transforms = [], []
    fid, tree, directory = fiff_open(fname)
    with fid:
        for entry in directory:
            try:
                tag = read_tag(fid, entry.pos)
            except Exception:
                continue  # skip tag types MNE cannot decode
            if entry.kind == FIFF.FIFF_COORD_TRANS:
                transforms.append(tag.data)
            elif isinstance(tag.data, str) and "/" in tag.data:
                paths.append(tag.data.strip())
    return paths, transforms


def remap(path):
    """Apply DICOM_PATH_REMAP from config, if set."""
    if config.DICOM_PATH_REMAP is None:
        return path
    old, new = config.DICOM_PATH_REMAP
    return new + path[len(old):] if path.startswith(old) else path


def main():
    args = parse_args(__doc__.splitlines()[1])
    rows = load_subjects(args.subject, args.session)
    config.SUBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, SUBJECTS_DIR=str(config.SUBJECTS_DIR))

    # Each FreeSurfer subject only needs processing once, even with several sessions.
    for _, row in rows.drop_duplicates("fs_subject").iterrows():
        subj, fs_subj = row["subject"], row["fs_subject"]
        out = anat_dir(subj)
        print(f"\n=== {subj} (FreeSurfer subject {fs_subj}) ===")

        # 1. Inspect the wrapper and keep a human-readable dump for reference.
        wrapper = Path(row["mri_fif"])
        (out / "mri_wrapper_dump.txt").write_text(mne.io.show_fiff(wrapper))
        paths, transforms = read_mri_wrapper(wrapper)
        print(f"Wrapper: {len(paths)} path strings, {len(transforms)} transform(s)")
        for i, t in enumerate(transforms):
            print(f"  transform {i}: {t}")
            mne.write_trans(out / f"wrapper_trans_{i}-trans.fif", t, overwrite=True)

        # 2. Run recon-all unless T1.mgz already exists.
        fs_mri = config.SUBJECTS_DIR / fs_subj / "mri"
        if (fs_mri / "T1.mgz").exists() and (fs_mri / "brainmask.mgz").exists():
            print("recon-all output found; skipping.")
        else:
            dicoms = [remap(p) for p in paths if Path(remap(p)).is_file()]
            if not dicoms:
                example = remap(paths[0]) if paths else "(none found)"
                raise FileNotFoundError(
                    f"No DICOM slices from the wrapper exist on disk. First path: "
                    f"{example}. Set DICOM_PATH_REMAP in config.py.")
            print(f"Found {len(dicoms)} DICOM slices; running recon-all.")
            # recon-all reads the whole series given one slice from it.
            cmd = ["recon-all", "-s", fs_subj, "-i", dicoms[0]] + config.RECON_ALL_FLAGS
            subprocess.run(cmd, env=env, check=True)

        # 3. Scalp surfaces (uses FreeSurfer's mkheadsurf).
        head_surf = config.SUBJECTS_DIR / fs_subj / "bem" / f"{fs_subj}-head-dense.fif"
        if head_surf.exists():
            print("Scalp surface found; skipping.")
        else:
            mne.bem.make_scalp_surfaces(fs_subj, subjects_dir=config.SUBJECTS_DIR,
                                        force=True, overwrite=True)


if __name__ == "__main__":
    main()
