"""
Stage 01: MRI preparation.

For each subject:
  1. Read the MEGIN MRI .fif wrapper, dump its tag structure to a text file,
     and pull out the DICOM slice paths and any coordinate transform it holds.
  2. Run FreeSurfer recon-all on the DICOM series (T1.mgz, brainmask.mgz).
  3. Build the scalp surface used for coregistration and the sphere fit.

If any of these steps fails (wrapper unreadable, slices missing, recon-all or
scalp surface failure), or mri_fif is blank in subjects.csv, and MRI_FALLBACK
is True in config.py, a surrogate anatomy is made instead: the fsaverage
template, scaled to fit the subject's digitized head shape, is written under
the subject's FreeSurfer name so later stages run unchanged. Which anatomy
was used, and why, is recorded in derivatives/<subject>/anat/anatomy_source.json.

The FreeSurfer environment is set up from FREESURFER_HOME in config.py (by
sourcing its SetUpFreeSurfer.sh), with SUBJECTS_DIR from config.py. A FreeSurfer
license file must be in FREESURFER_HOME, or FS_LICENSE must be set.

Note: reading the wrapper uses MNE's private FIF reader (mne._fiff), which may
change between MNE versions. The scalp surface and coregistration in later
stages do not depend on the wrapper's transform; it is saved for reference only.
"""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import mne
import numpy as np
from mne.coreg import Coregistration
from mne.bem import make_scalp_surfaces
from mne._fiff.constants import FIFF
from mne._fiff.open import fiff_open
from mne._fiff.tag import read_tag

import config
from common import anat_dir, load_subjects, parse_args, run_each


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


IDENTITY_XFM = """MNI Transform File
% identity: fsaverage is already in MNI305 space

Transform_Type = Linear;
Linear_Transform =
1.0 0.0 0.0 0.0
0.0 1.0 0.0 0.0
0.0 0.0 1.0 0.0;
"""
HEAD_LEVELS = ("dense", "medium", "sparse")


def anatomy_complete(fs_subj):
    """True if the files later stages need exist for this FreeSurfer subject."""
    d = config.SUBJECTS_DIR / fs_subj
    return ((d / "mri" / "T1.mgz").exists() and (d / "mri" / "brainmask.mgz").exists()
            and (d / "bem" / f"{fs_subj}-head-sparse.fif").exists())


def subject_mri(row, fs_subj, out, env):
    """Steps 1-3 with the subject's own MRI. Raises an exception on any failure."""
    # 1. Inspect the wrapper and keep a human-readable dump for reference.
    if not row["mri_fif"].strip():
        raise ValueError("no MRI wrapper listed in subjects.csv")
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


def ensure_template():
    """
    Build the template subject (once) in SUBJECTS_DIR from FreeSurfer's fsaverage:
    T1.mgz, brainmask.mgz, talairach.xfm and scalp surfaces. Returns its name.
    """
    name = config.TEMPLATE_NAME
    tdir = config.SUBJECTS_DIR / name
    if (tdir / "bem" / f"{name}-head-sparse.fif").exists():
        return name

    src = Path(config.TEMPLATE_SOURCE) / "mri"
    def first_existing(names):
        found = [src / n for n in names if (src / n).is_file()]
        return found[0] if found else None
    t1 = first_existing(("T1.mgz", "orig.mgz", "brain.mgz"))
    mask = first_existing(("brainmask.mgz", "brain.mgz"))
    if t1 is None or mask is None:
        raise FileNotFoundError(f"Template MRI files not found in {src}; "
                                f"check TEMPLATE_SOURCE in config.py.")
    print(f"Building template subject {name} from {src}")
    for sub in ("mri/transforms", "bem", "surf"):
        (tdir / sub).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(t1, tdir / "mri" / "T1.mgz")
    shutil.copyfile(mask, tdir / "mri" / "brainmask.mgz")
    xfm = src / "transforms" / "talairach.xfm"
    if xfm.is_file():
        shutil.copyfile(xfm, tdir / "mri" / "transforms" / "talairach.xfm")
    else:
        (tdir / "mri" / "transforms" / "talairach.xfm").write_text(IDENTITY_XFM)
    make_scalp_surfaces(name, subjects_dir=config.SUBJECTS_DIR, force=True, overwrite=True)
    return name


def surrogate_mri(row, fs_subj):
    """
    Write a copy of the template, scaled to the subject's digitized head shape,
    as FreeSurfer subject fs_subj. Returns the scale factors.
    """
    name = ensure_template()

    # Fit a uniform scale (plus rotation and translation) of the template scalp
    # to the head shape points, the same way stage 02 fits the subject's own MRI.
    info = mne.io.read_info(row["raw_fif"], verbose=False)
    coreg = Coregistration(info, subject=name, subjects_dir=config.SUBJECTS_DIR,
                           fiducials="estimated", on_defects="warn")
    coreg.set_scale_mode("uniform")
    coreg.fit_fiducials(verbose=False)
    coreg.fit_icp(n_iterations=20, nasion_weight=2.0, verbose=False)
    coreg.omit_head_shape_points(distance=5.0 / 1000)   # metres
    coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=False)
    scale = np.broadcast_to(np.asarray(coreg.scale, dtype=float), (3,)).copy()
    if not np.all(np.isfinite(scale)) or np.any(scale < 0.7) or np.any(scale > 1.3):
        raise RuntimeError(f"Implausible template scale {scale}; check the head shape points.")
    print(f"Template scale factor: {np.round(scale, 3)}")

    # Keep any partial output from the failed attempt for inspection.
    dest = config.SUBJECTS_DIR / fs_subj
    if dest.exists():
        kept = dest.with_name(f"{fs_subj}_failed_{time.strftime('%Y%m%d_%H%M%S')}")
        dest.rename(kept)
        print(f"Moved the failed attempt to {kept}")

    # Scaled copy of the template MRI volumes and transforms...
    mne.scale_mri(name, fs_subj, scale, overwrite=True, subjects_dir=config.SUBJECTS_DIR,
                  skip_fiducials=True, labels=False, annot=False, on_defects="warn",
                  verbose=False)
    # ...and of the scalp surfaces, which scale_mri does not copy.
    for level in HEAD_LEVELS:
        src = config.SUBJECTS_DIR / name / "bem" / f"{name}-head-{level}.fif"
        surfs = mne.read_bem_surfaces(src, on_defects="warn", verbose=False)
        for s in surfs:
            s["rr"] = s["rr"] * scale
        mne.write_bem_surfaces(dest / "bem" / f"{fs_subj}-head-{level}.fif", surfs,
                               overwrite=True, verbose=False)
    return scale


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
    def process(row):
        subj, fs_subj = row["subject"], row["fs_subject"]
        out = anat_dir(subj)
        record_file = out / "anatomy_source.json"
        print(f"\n=== {subj} (FreeSurfer subject {fs_subj}) ===")

        # A surrogate made on an earlier run is reused; delete this subject's
        # FreeSurfer folder and anat/ folder to try the subject's MRI again.
        if record_file.exists() and anatomy_complete(fs_subj):
            record = json.loads(record_file.read_text())
            if record.get("anatomy") == "scaled_template":
                print(f"Using the scaled template made earlier ({record.get('reason')}).")
                return

        try:
            subject_mri(row, fs_subj, out, env)
            record = {"anatomy": "subject_mri"}
        except Exception as err:
            if not config.MRI_FALLBACK:
                raise
            reason = f"{type(err).__name__}: {err}"
            print(f"WARNING: subject MRI could not be used ({reason})")
            print("Falling back to the scaled fsaverage template.")
            scale = surrogate_mri(row, fs_subj)
            record = {"anatomy": "scaled_template", "template": config.TEMPLATE_NAME,
                      "scale": scale.tolist(), "reason": reason}
        record_file.write_text(json.dumps(record, indent=1))


    run_each(rows.drop_duplicates("fs_subject"), "s01", process, per_session=False)

if __name__ == "__main__":
    main()
