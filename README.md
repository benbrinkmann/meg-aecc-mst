# AECc + MST pipeline (MNE-Python)

Replication of the methods in Govaarts et al. (2025), *Network Neuroscience*
9(3):824–841, https://doi.org/10.1162/netn_a_00450, for tSSS-processed MEGIN data.

## Setup

1. `pip install -r requirements.txt`
2. FreeSurfer installed, with `FREESURFER_HOME` set and `recon-all` on the PATH.
3. Edit the PATHS section of `config.py`.
4. Copy `subjects_template.csv` to the `SUBJECTS_CSV` path in `config.py` and fill it in,
   one row per subject and session. `subjects.csv` is ignored by git so subject
   identifiers stay out of the repository.

| column | meaning |
|---|---|
| subject | subject ID used for output folders |
| session | session label; labels must sort chronologically (ses01, ses02) |
| raw_fif | tSSS-processed resting state recording |
| er_fif | empty room recording, processed with the same SSS/tSSS settings |
| mri_fif | MEGIN MRI wrapper .fif (lists the DICOM slices) |
| fs_subject | FreeSurfer subject name (shared across sessions) |
| group | group label used by stage 07 (see `REFERENCE_GROUPS`, `TARGET_GROUP`) |

## Stages

Each script accepts `--subject` and `--session` to limit what it processes.
`./run_all.sh` runs stages 01–06, then stage 07 if no arguments were given.

| script | does | main output |
|---|---|---|
| s01_mri_prep.py | reads the MRI wrapper, runs recon-all, makes scalp surfaces | FreeSurfer subject, `anat/mri_wrapper_dump.txt` |
| s02_coreg.py | fiducial estimate + ICP head shape fit | `coreg-trans.fif`, distances |
| s03_atlas_forward.py | AAL 90 centroids in native space, sphere fit, forward model | `anat/aal_centroids.json`, `aal90-fwd.fif` |
| s04_beamformer.py | scalar LCMV, ROI time series for the analysis segment | `roi_timeseries.npz` |
| s05_connectivity.py | 8 epochs, 6 bands, AECc | `aecc.npz` |
| s06_mst.py | per epoch MSTs, leaf fraction, diameter, tree hierarchy, BC | `mst_metrics.csv`, `mst_bc.csv` |
| s07_group_reference.py | reference MSTs, network similarity, Wilcoxon + FDR | `group/*.csv` |

## Departures from the paper and open choices

- **xSSS** is not available; the input data are already tSSS-processed.
- **Epochs**: the first `N_EPOCHS` consecutive epochs starting at `SEGMENT_START_S`
  are used, rather than visually selected artifact-free epochs. Review the
  segment for artifacts.
- **Beamformer normalization**: `unit-noise-gain` is the closest MNE option to the
  paper's normalized weights; whether it matches exactly is uncertain. The
  regularization value is not reported in the paper. `reduce_rank=True` is
  required because a sphere model has no radial sensitivity.
- **Noise covariance**: MNE whitens with the empty room covariance to combine
  magnetometers and gradiometers; the paper does not describe this step.
- **Template registration**: MNI152 (nilearn) to the subject's `brainmask.mgz`
  using MNE/dipy symmetric diffeomorphic registration, rather than the
  normalization used in the paper.
- **Sphere fit**: least squares to MRI scalp points above `SPHERE_MIN_Z_M` in
  head coordinates; the cutoff is a choice not specified in the paper.
- **Band filter**: per-epoch FFT brick-wall filter by default (`BAND_FILTER`),
  believed but not confirmed to match the authors' software.
- **AECc rescaling** to (AECc + 1)/2 is inferred from the figure color scales;
  it does not affect the MSTs.
- **FDR** is applied across the six bands within each comparison; the paper
  corrected across six bands and two groups.
- The MRI wrapper's own transform is saved for reference but not used, because
  its MRI coordinate frame is MEGIN's, not FreeSurfer's.

## License

BSD 3-Clause; see `LICENSE`.
