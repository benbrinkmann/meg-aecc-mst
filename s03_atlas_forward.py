"""
Stage 03: AAL centroids in native space, sphere head model, and forward model.

Per subject (once):
  - Nonlinearly register the MNI152 template to the subject's brainmask.mgz.
  - Warp the AAL atlas into the subject's MRI with nearest neighbour sampling.
  - Take the centroid of each of the 90 cerebral AAL regions as its source point.

Per session:
  - Fit a single sphere to the MRI scalp surface (head coordinates).
  - Build a 90-point discrete source space and the free-orientation forward model.
"""
import json

import mne
# Import submodule functions directly: some MNE versions do not expose
# mne.transforms as an attribute of the lazily loaded mne package.
from mne.transforms import (apply_trans, apply_volume_registration,
                            compute_volume_registration, invert_transform)
import nibabel as nib
import numpy as np
from nilearn import datasets

import config
from common import anat_dir, load_subjects, parse_args, require, session_dir, run_each


def aal_rois():
    """Return (atlas image, [(label value, name), ...]) for the 90 cerebral ROIs."""
    aal = datasets.fetch_atlas_aal(version=config.AAL_VERSION)
    # Newer nilearn versions may list a "Background" entry (value 0); skip it.
    keep = [(int(idx), name) for idx, name in zip(aal.indices, aal.labels)
            if int(idx) != 0 and name != "Background"
            and not name.startswith(config.AAL_EXCLUDE_PREFIXES)]
    if len(keep) != config.N_ROIS_EXPECTED:
        raise RuntimeError(f"Expected {config.N_ROIS_EXPECTED} AAL regions, got {len(keep)}.")
    return nib.load(aal.maps), keep


def compute_centroids(fs_subj, out):
    """Warp AAL into subject space and return ROI names and centroids (m, MRI coords)."""
    brain = nib.load(require(config.SUBJECTS_DIR / fs_subj / "mri" / "brainmask.mgz",
                             "s01_mri_prep.py"))
    template = datasets.load_mni152_template(resolution=1)   # skull-stripped T1
    atlas_img, rois = aal_rois()

    print("Registering MNI152 template to subject (several minutes)...")
    reg_affine, sdr = compute_volume_registration(
        template, brain, pipeline="all", zooms=config.REG_ZOOMS)
    warped = apply_volume_registration(
        atlas_img, brain, reg_affine, sdr, interpolation="nearest")
    nib.save(warped, out / "aal_in_subject.nii.gz")   # for visual QC

    labels = np.asarray(warped.dataobj).round().astype(int)
    vox2tkr = brain.header.get_vox2ras_tkr()   # voxel -> FreeSurfer surface RAS (mm)

    names, centroids_mm = [], []
    for value, name in rois:
        vox = np.argwhere(labels == value)
        if vox.size == 0:
            raise RuntimeError(f"AAL region {name} is empty after warping; check registration.")
        centroid_vox = vox.mean(axis=0)
        centroids_mm.append(vox2tkr[:3, :3] @ centroid_vox + vox2tkr[:3, 3])
        names.append(name)
    return names, np.array(centroids_mm) / 1000.0   # metres


def fit_sphere(points):
    """Least-squares sphere fit. Returns (centre (3,), radius) in input units."""
    # |p - c|^2 = r^2  rearranges to the linear system  2 p.c + (r^2 - |c|^2) = |p|^2
    a = np.column_stack([2 * points, np.ones(len(points))])
    b = np.sum(points ** 2, axis=1)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    centre = sol[:3]
    radius = float(np.sqrt(sol[3] + centre @ centre))
    return centre, radius


def main():
    args = parse_args(__doc__.splitlines()[1])
    rows = load_subjects(args.subject, args.session)

    def process(row):
        subj, ses, fs_subj = row["subject"], row["session"], row["fs_subject"]
        print(f"\n=== {subj} / {ses} ===")
        anat, out = anat_dir(subj), session_dir(subj, ses)

        # --- Centroids: computed once per subject, reused across sessions.
        cent_file = anat / "aal_centroids.json"
        if cent_file.exists():
            cent = json.loads(cent_file.read_text())
            names, rr = cent["names"], np.array(cent["rr_mri_m"])
        else:
            names, rr = compute_centroids(fs_subj, anat)
            cent_file.write_text(json.dumps({"names": names, "rr_mri_m": rr.tolist()}, indent=1))

        # --- Sphere fitted to the scalp surface, in head coordinates.
        trans = mne.read_trans(require(out / "coreg-trans.fif", "s02_coreg.py"))
        # Use the dense scalp surface from stage 01 (MNE's default search does
        # not find the -head-dense/-medium/-sparse file names).
        head = mne.get_head_surf(fs_subj, source=("head-dense", "head"),
                                 subjects_dir=config.SUBJECTS_DIR, on_defects="warn")
        scalp_head = apply_trans(invert_transform(trans), head["rr"])
        scalp_head = scalp_head[scalp_head[:, 2] > config.SPHERE_MIN_Z_M]
        if len(scalp_head) < 100:
            raise RuntimeError("Too few scalp points above SPHERE_MIN_Z_M for a sphere fit.")
        centre, radius = fit_sphere(scalp_head)
        print(f"Sphere: centre {np.round(centre * 1000, 1)} mm, radius {radius * 1000:.1f} mm")
        # head_radius=None gives a MEG-only sphere (no conductivity layers needed).
        sphere = mne.make_sphere_model(r0=centre, head_radius=None)

        # --- Discrete source space at the 90 centroids. Orientation is left
        #     free here; the beamformer picks it per source.
        nn = np.tile([0.0, 0.0, 1.0], (len(rr), 1))
        src = mne.setup_volume_source_space(pos=dict(rr=rr, nn=nn))

        info = mne.io.read_info(row["raw_fif"])
        fwd = mne.make_forward_solution(info, trans, src, sphere, meg=True, eeg=False,
                                        mindist=0.0)
        mne.write_forward_solution(out / "aal90-fwd.fif", fwd, overwrite=True)
        (out / "sphere.json").write_text(json.dumps(
            {"centre_head_m": centre.tolist(), "radius_m": radius}, indent=1))


    run_each(rows, "s03", process)

if __name__ == "__main__":
    main()
