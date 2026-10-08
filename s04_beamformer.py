"""
Stage 04: Scalar LCMV beamformer and ROI time series.

  - Band-pass the tSSS-processed recording to 0.5-48 Hz (broadband).
  - Data covariance from the whole broadband recording, as in the paper.
  - Noise covariance from the empty room recording, filtered identically.
    MNE uses it to whiten when combining magnetometers and gradiometers.
  - Beamformer weights at the 90 AAL centroids; apply to the analysis segment
    (N_EPOCHS x EPOCH_DUR_S seconds starting at SEGMENT_START_S).

The empty room recording should have the same SSS/tSSS processing as the
subject recording; a warning is printed if it appears not to.
"""
import json

import mne
import numpy as np
from mne.beamformer import apply_lcmv_raw, make_lcmv

import config
from common import load_subjects, parse_args, require, session_dir


def load_broadband(fname, bads=None):
    """
    Read a .fif recording, drop bad channels, keep MEG, and filter to the
    broadband range. Returns (raw, list of bad channels that were dropped).
    If bads is given it replaces the file's own bad channel list.
    """
    raw = mne.io.read_raw_fif(fname, preload=True, verbose=False)
    if bads is not None:
        raw.info["bads"] = [ch for ch in bads if ch in raw.ch_names]
    dropped = list(raw.info["bads"])   # record before pick() clears the list
    raw.pick("meg", exclude="bads")
    raw.filter(*config.BROADBAND_HZ, verbose=False)
    return raw, dropped


def main():
    args = parse_args(__doc__.splitlines()[1])
    rows = load_subjects(args.subject, args.session)

    for _, row in rows.iterrows():
        subj, ses = row["subject"], row["session"]
        print(f"\n=== {subj} / {ses} ===")
        out = session_dir(subj, ses)
        fwd = mne.read_forward_solution(require(out / "aal90-fwd.fif", "s03_atlas_forward.py"))
        names = json.loads((config.DERIV_DIR / subj / "anat" / "aal_centroids.json")
                           .read_text())["names"]

        raw, bads = load_broadband(row["raw_fif"])
        # Use the subject recording's bad channels so both covariances match.
        er, _ = load_broadband(row["er_fif"], bads=bads)
        if not er.info.get("proc_history"):
            print("WARNING: empty room file shows no MaxFilter history; it should be "
                  "processed with the same SSS/tSSS settings as the subject data.")

        # Both recordings must contain exactly the same channels.
        common_chs = [ch for ch in raw.ch_names if ch in er.ch_names]
        if len(common_chs) != len(raw.ch_names):
            print(f"Note: {len(raw.ch_names) - len(common_chs)} channel(s) missing from "
                  f"empty room; dropping them from both.")
        raw.pick(common_chs)
        er.pick(common_chs)

        # Covariances. rank="info" accounts for the rank reduction from SSS.
        data_cov = mne.compute_raw_covariance(raw, rank="info", verbose=False)
        noise_cov = mne.compute_raw_covariance(er, rank="info", verbose=False)
        print(f"Data covariance from {raw.times[-1]:.0f} s of data")

        filters = make_lcmv(raw.info, fwd, data_cov, reg=config.LCMV_REG,
                            noise_cov=noise_cov, pick_ori=config.LCMV_PICK_ORI,
                            weight_norm=config.LCMV_WEIGHT_NORM, rank="info",
                            reduce_rank=config.LCMV_REDUCE_RANK, verbose=False)

        # Analysis segment in samples, relative to the start of the loaded data.
        sfreq = raw.info["sfreq"]
        start = int(round(config.SEGMENT_START_S * sfreq))
        n_samp = int(round(config.N_EPOCHS * config.EPOCH_DUR_S * sfreq))
        if start + n_samp > raw.n_times:
            raise ValueError(
                f"Segment {config.SEGMENT_START_S:.0f}-"
                f"{config.SEGMENT_START_S + n_samp / sfreq:.0f} s exceeds the "
                f"{raw.n_times / sfreq:.0f} s recording.")
        stc = apply_lcmv_raw(raw, filters, start=start, stop=start + n_samp, verbose=False)

        if stc.data.shape[0] != len(names):
            raise RuntimeError(f"Got {stc.data.shape[0]} source time series, "
                               f"expected {len(names)}.")
        np.savez(out / "roi_timeseries.npz", data=stc.data, sfreq=sfreq, names=names,
                 segment_start_s=config.SEGMENT_START_S)
        print(f"Saved {stc.data.shape[0]} ROI time series, {n_samp / sfreq:.1f} s at "
              f"{sfreq:.0f} Hz")


if __name__ == "__main__":
    main()
