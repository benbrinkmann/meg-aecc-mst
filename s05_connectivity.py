"""
Stage 05: Band-limited AECc connectivity per epoch.

  - Downsample the ROI time series by DOWNSAMPLE_FACTOR (data are already
    low-passed at 48 Hz, so plain decimation is safe; this is checked).
  - Cut N_EPOCHS consecutive, non-overlapping epochs of EPOCH_DUR_S.
  - Filter each epoch into the six bands and compute the corrected amplitude
    envelope correlation with pairwise orthogonalization (Hipp et al., 2012).

Output: aecc.npz with
  aecc_epochs  (n_bands, n_epochs, 90, 90), diagonal set to 0
  aecc_mean    (n_bands, 90, 90), mean over epochs
"""
import mne
import numpy as np
from mne_connectivity import envelope_correlation

import config
from common import load_subjects, parse_args, require, session_dir


def fft_bandpass(x, sfreq, lo, hi):
    """Brick-wall band-pass by zeroing FFT bins outside [lo, hi] Hz. x: (..., n_times)."""
    n = x.shape[-1]
    spec = np.fft.rfft(x, axis=-1)
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    spec[..., (freqs < lo) | (freqs > hi)] = 0.0
    return np.fft.irfft(spec, n=n, axis=-1)


def full_matrices(conn_dense):
    """
    Convert MNE-Connectivity dense output to full symmetric (n_epochs, N, N)
    matrices with a zero diagonal. MNE fills one triangle only, so mirror it.
    """
    m = np.squeeze(conn_dense)
    if m.ndim == 2:            # a single epoch loses its epoch axis in squeeze
        m = m[np.newaxis]
    lower = np.tril(m, k=-1)
    if np.allclose(lower, 0):  # fall back to the upper triangle if that is the one filled
        lower = np.transpose(np.triu(m, k=1), (0, 2, 1))
    return lower + np.transpose(lower, (0, 2, 1))


def main():
    args = parse_args(__doc__.splitlines()[1])
    rows = load_subjects(args.subject, args.session)

    for _, row in rows.iterrows():
        subj, ses = row["subject"], row["session"]
        print(f"\n=== {subj} / {ses} ===")
        out = session_dir(subj, ses)
        ts = np.load(require(out / "roi_timeseries.npz", "s04_beamformer.py"))
        data, sfreq = ts["data"], float(ts["sfreq"])

        # --- Downsample. Check the new Nyquist frequency is above the top band edge.
        sfreq_ds = sfreq / config.DOWNSAMPLE_FACTOR
        top = max(hi for _, hi in config.BANDS.values())
        if sfreq_ds / 2 <= top:
            raise ValueError(f"Downsampled Nyquist {sfreq_ds / 2:.1f} Hz is not above "
                             f"the highest band edge {top} Hz.")
        data = data[:, ::config.DOWNSAMPLE_FACTOR]

        # --- Cut epochs.
        n_ep = int(round(config.EPOCH_DUR_S * sfreq_ds))
        if n_ep * config.N_EPOCHS > data.shape[1]:
            raise ValueError(f"Need {n_ep * config.N_EPOCHS} samples for "
                             f"{config.N_EPOCHS} epochs, have {data.shape[1]}.")

        def to_epochs(x):
            """(n_roi, n_times) -> (n_epochs, n_roi, n_ep)."""
            x = x[:, :n_ep * config.N_EPOCHS]
            return x.reshape(x.shape[0], config.N_EPOCHS, n_ep).transpose(1, 0, 2)

        print(f"{config.N_EPOCHS} epochs x {n_ep} samples at {sfreq_ds:.2f} Hz "
              f"({n_ep / sfreq_ds:.2f} s each)")

        aecc = np.zeros((len(config.BANDS), config.N_EPOCHS, data.shape[0], data.shape[0]))
        for b, (band, (lo, hi)) in enumerate(config.BANDS.items()):
            if config.BAND_FILTER == "fft":
                band_epochs = fft_bandpass(to_epochs(data), sfreq_ds, lo, hi)
            elif config.BAND_FILTER == "fir":
                filt = mne.filter.filter_data(data, sfreq_ds, lo, hi, verbose=False)
                band_epochs = to_epochs(filt)
            else:
                raise ValueError(f"Unknown BAND_FILTER: {config.BAND_FILTER}")

            # absolute=False keeps the sign of each correlation, as in the paper's AECc.
            conn = envelope_correlation(band_epochs, orthogonalize="pairwise",
                                        absolute=False, verbose=False)
            mats = full_matrices(conn.get_data(output="dense"))
            if mats.shape != aecc.shape[1:]:
                raise RuntimeError(f"Unexpected connectivity shape {mats.shape}.")

            if config.AECC_RESCALE:
                mats = (mats + 1.0) / 2.0
                for e in range(mats.shape[0]):
                    np.fill_diagonal(mats[e], 0.0)
            aecc[b] = mats
            off = mats[:, ~np.eye(mats.shape[1], dtype=bool)]
            print(f"  {band:6s}: AECc mean {off.mean():.3f}, range {off.min():.3f} to {off.max():.3f}")

        np.savez(out / "aecc.npz", aecc_epochs=aecc, aecc_mean=aecc.mean(axis=1),
                 bands=list(config.BANDS), names=ts["names"], sfreq_ds=sfreq_ds,
                 rescaled=config.AECC_RESCALE)


if __name__ == "__main__":
    main()
