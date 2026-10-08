"""
Shared configuration for the AECc + MST pipeline.

Replicates the methods of Govaarts et al. (2025), Network Neuroscience 9(3):824-841,
https://doi.org/10.1162/netn_a_00450, using MNE-Python.

Edit the PATHS section for your machine. Everything else defaults to the values
reported in the paper; where the paper does not report a value, the default is
marked "NOT REPORTED IN PAPER" so it can be revisited.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------------
# Root for this analysis: the folder this config.py file is in, wherever the
# repository was cloned. subjects.csv, derivatives/ and freesurfer/ all live
# here and are excluded from git by .gitignore.
PROJECT_DIR = Path(__file__).resolve().parent
SUBJECTS_CSV = PROJECT_DIR / "subjects.csv"        # one row per subject/session
DERIV_DIR = PROJECT_DIR / "derivatives"            # all pipeline outputs
SUBJECTS_DIR = PROJECT_DIR / "freesurfer"          # FreeSurfer SUBJECTS_DIR

# The MRI .fif wrapper stores the DICOM slice paths as they were when it was
# written (possibly on another machine). If those paths moved, set this to
# (old_prefix, new_prefix) and the prefix will be swapped before use.
# Example: ("/data/mrilab/", "/neuro/data/archive/mri/")
DICOM_PATH_REMAP = None

# ----------------------------------------------------------------------------
# MRI / ANATOMY (stage 01 and 03)
# ----------------------------------------------------------------------------
# Only T1.mgz and brainmask.mgz are needed, which "-autorecon1" produces in
# well under an hour. Use "-all" if you also want cortical surfaces.
RECON_ALL_FLAGS = ["-autorecon1"]

# AAL atlas: the 90 cerebral regions (78 cortical + 12 subcortical); the 26
# cerebellar and vermis regions are dropped, matching the paper.
AAL_VERSION = "SPM12"
AAL_EXCLUDE_PREFIXES = ("Cerebelum", "Vermis")
N_ROIS_EXPECTED = 90

# Template-to-subject nonlinear registration (MNI152 template -> subject brain).
# Voxel size (mm) used during registration; larger is faster, less accurate.
REG_ZOOMS = dict(translation=5.0, rigid=5.0, affine=3.0, sdr=3.0)

# Sphere fit to the MRI scalp surface: only scalp points above this height in
# head coordinates (metres) are used, to exclude face and neck.
# NOT REPORTED IN PAPER.
SPHERE_MIN_Z_M = 0.0

# Coregistration QC: warn if mean headshape-to-scalp distance exceeds this (mm).
COREG_WARN_MM = 5.0

# ----------------------------------------------------------------------------
# BEAMFORMER (stage 04)
# ----------------------------------------------------------------------------
BROADBAND_HZ = (0.5, 48.0)            # paper: 0.5-48 Hz broadband
LCMV_PICK_ORI = "max-power"           # scalar beamformer, optimal orientation
LCMV_WEIGHT_NORM = "unit-noise-gain"  # closest MNE option to "normalized weights"
LCMV_REG = 0.05                       # NOT REPORTED IN PAPER (MNE default)
# Required with a sphere model: the radial dipole component produces no field,
# so the 3-orientation lead field has rank 2 and LCMV is otherwise singular.
LCMV_REDUCE_RANK = True

# ----------------------------------------------------------------------------
# EPOCHS (stage 04 and 05)
# ----------------------------------------------------------------------------
SEGMENT_START_S = 60.0                # take data starting 1 minute in
N_EPOCHS = 8                          # paper: 8 epochs per subject
EPOCH_DUR_S = 16384 / 1250.0          # paper: 16,384 samples at 1,250 Hz = 13.1072 s
DOWNSAMPLE_FACTOR = 4                 # paper: downsampled by a factor of 4

# ----------------------------------------------------------------------------
# CONNECTIVITY (stage 05)
# ----------------------------------------------------------------------------
BANDS = {
    "delta":  (0.5, 4.0),
    "theta":  (4.0, 8.0),
    "alpha1": (8.0, 10.0),
    "alpha2": (10.0, 13.0),
    "beta":   (13.0, 30.0),
    "gamma":  (30.0, 48.0),
}

# "fft": brick-wall FFT filter applied to each epoch separately (believed to
#        match the BrainWave software used by the Amsterdam group; not confirmed).
# "fir": MNE zero-phase FIR filter on the continuous segment, then epoched.
BAND_FILTER = "fft"

# Rescale AECc from [-1, 1] to [0, 1] as (AECc + 1) / 2. The paper's matrices
# (range about 0.48-0.58) suggest this was done; this is an inference, not stated.
# It does not change the MST, only the reported values.
AECC_RESCALE = True

# ----------------------------------------------------------------------------
# GROUP ANALYSIS (stage 07)
# ----------------------------------------------------------------------------
# Group labels must match the "group" column of subjects.csv.
REFERENCE_GROUPS = ["ref_A", "ref_B"]     # groups used to build reference MSTs
TARGET_GROUP = "target"                   # group compared against the references
MIN_SUBJECTS_FOR_STATS = 6                # skip Wilcoxon tests below this n
