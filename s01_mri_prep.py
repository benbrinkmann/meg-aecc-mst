"""
Stage 01: MRI preparation.

For each subject:
  1. Read the MEGIN MRI .fif wrapper, dump its tag structure to a text file,
     and pull out the DICOM slice paths and any coordinate transform it holds.
  2. Run FreeSurfer recon-all on the DICOM series (T1.mgz, brainmask.mgz).
  3. Build the scalp surface used for coregistration and the sphere fit.

The FreeSurfer environment is set up from FREESURFER_HOME in config.py (by
sourcing its SetUpFreeSurfer.sh), with SUBJECTS_DIR from config.py. A FreeSurfer
license file must be in FREESURFER_HOME, or FS_LICENSE must be set.

Note: reading the wrapper uses MNE's private FIF reader (mne._fiff), which may
change between MNE versions. The scalp surface and coregistration in later
stages do not depend on the wrapper's transform; it is saved for reference only.
"""
import os
import subprocess
from pathlib import Path

import mne
from mne.bem import make_scalp_surfaces
from mne._fiff.constants import FIFF
from mne._fiff.open import fiff_open
from mne._fiff.tag import read_tag

import config
from common import anat_dir, load_subjects, parse_args


def freesurfer_env():
    """
    Return the environment FreeSurfer commands need, as a dict, by sourcing
    SetUpFreeSurfer.sh from config.FREESURFER_HOME in a bash shell and reading
    back the resulting variables. SUBJECTS_DIR is always set from config.py.
    """
    fs_home = config.FREESURFER_HOME
    setup = fs_home / "SetUpFreeSurfer.sh"
    if not setup.is_file():
        raise FileNotFoundError(f"{setup} not found; check FREESURFER_HOME in config.py.")
    script = (f'export FREESURFER_HOME="{fs_home}"; '
              f'export SUBJECTS_DIR="{config.SUBJECTS_DIR}"; '
              f'export FS_FREESURFERENV_NO_OUTPUT=1; '
              f'source "{setup}" >/dev/null 2>&1; env -0')
    out = subprocess.run(["bash", "-c", script], check=True, capture_output=True).stdout
    # "env -0" separates variables with NUL characters, so values may contain newlines.
    env = dict(item.split("=", 1) for item in out.decode(errors="replace").split("\0")
               if "=" in item)
    env["FREESURFER_HOME"] = str(fs_home)
    env["SUBJECTS_DIR"] = str(config.SUBJECTS_DIR)   # in case the setup script reset it
    if config.FS_ALLOW_DEEP:
        env["FS_ALLOW_DEEP"] = "1"   # needed by FreeSurfer 8 recon-all; see config.py
    return env


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
    """Swap the first matching old prefix in DICOM_PATH_REMAP for its new prefix."""
    pairs = config.DICOM_PATH_REMAP or []
    if isinstance(pairs, tuple) and len(pairs) == 2 and isinstance(pairs[0], str):
        pairs = [pairs]   # also accept a single (old, new) pair
    for old, new in pairs:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def main():
    args = parse_args(__doc__.splitlines()[1])
    rows = load_subjects(args.subject, args.session)
    config.SUBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    if not os.access(config.SUBJECTS_DIR, os.W_OK):
        raise PermissionError(f"No write permission for {config.SUBJECTS_DIR}; "
                              f"recon-all needs to create subject folders there.")

    # Put the FreeSurfer environment into this process too, so the FreeSurfer
    # programs MNE calls (mkheadsurf, for the scalp surface) find it as well.
    env = freesurfer_env()
    os.environ.update(env)

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
            result = subprocess.run(cmd, env=env)
            if result.returncode != 0:
                # FreeSurfer 8 can fail in a later step of -autorecon1 (e.g. CC
                # segmentation) after T1.mgz and brainmask.mgz are written. Those
                # two files are all this pipeline uses, so continue if they exist.
                log = config.SUBJECTS_DIR / fs_subj / "scripts" / "recon-all.log"
                if (fs_mri / "T1.mgz").exists() and (fs_mri / "brainmask.mgz").exists():
                    print(f"WARNING: recon-all exited with errors (see {log}), but "
                          f"T1.mgz and brainmask.mgz were created; continuing. "
                          f"Check brainmask.mgz in freeview.")
                else:
                    raise RuntimeError(f"recon-all failed before creating T1.mgz and "
                                       f"brainmask.mgz; see {log}")

        # 3. Scalp surfaces (uses FreeSurfer's mkheadsurf, then VTK to make the
        #    medium and sparse versions). The sparse file is written last, so
        #    check for it: a run that stopped part way is redone.
        head_surf = config.SUBJECTS_DIR / fs_subj / "bem" / f"{fs_subj}-head-sparse.fif"
        if head_surf.exists():
            print("Scalp surface found; skipping.")
        else:
            make_scalp_surfaces(fs_subj, subjects_dir=config.SUBJECTS_DIR,
                                        force=True, overwrite=True)


if __name__ == "__main__":
    main()
