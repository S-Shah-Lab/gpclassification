"""
Generate synthetic EEG-like epochs with controllable 1/f noise and band-limited alpha/beta injections, then export covariance features

Description:
  - Creates pink/brown 1/f^y noise (pink: y~1, brown: y~2) per channel with slight amplitude jitter to avoid perfectly flat spectra
  - Injects narrowband sinusoids around alpha (~9-11 Hz) and beta (~19-21 Hz) with configurable amplitudes
  - Encodes motor-imagery-style lateralization by removing injections on contralateral channel groups for left/right labels
  - Computes a covariance matrix per epoch and saves feature/label arrays

Inputs:
  fs        : sampling rate (Hz), e.g., 1000
  duration  : epoch duration (s), e.g., 1.5
  s         : number of channels, e.g., 64
  N         : number of epochs to synthesize
  fmin/fmax : band edges for noise shaping, e.g., 8-25 Hz

Outputs:
  ./data/classification_X.npy  -> (N, C, C) covariance matrices (float64)
  ./data/classification_Y.npy  -> (N,) labels in {0.0, 1.0} for left/right

Notes:
  - Time series are zero-mean and unit-variance per channel before injection
  - Covariance uses np.cov with default normalization; adjust if you need unbiased scaling or regularization
  - PSD plotting is optional and off by default to keep runs fast
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_smooth_noise_iir(
    n_series,
    fs,
    duration,
    noise_type="pink",
    fmin=None,
    fmax=None,
    variation=0.1,
):
    """
    Generate noise with a smooth power spectrum for pink or brown noise,
    but with a bit of white-noise-style variation so it's not too flat.
    """
    # Number of samples and frequency bins
    n_samples = int(fs * duration)
    freqs = np.fft.rfftfreq(n_samples, 1.0 / fs)

    # Exponent: 1 for pink, 2 for brown
    exponent = 1.0 if noise_type == "pink" else 2.0

    # Base scaling ~ 1/f^(exponent/2); zero DC
    scaling = np.where(freqs == 0, 0.0, freqs ** (-exponent / 2.0))

    # Apply band limits
    if fmin is not None:
        scaling[freqs < fmin] = 0.0
    if fmax is not None:
        scaling[freqs > fmax] = 0.0

    noise = np.zeros((n_series, n_samples))

    for i in range(n_series):
        # Randomize phase in [0, 2π)
        phases = np.exp(2j * np.pi * np.random.rand(len(freqs)))

        # Inject white-noise-style amplitude jitter
        jitter = 1.0 + variation * np.random.randn(len(freqs))
        amplitude = scaling * jitter

        # Build the spectrum and enforce real DC/Nyquist
        spectrum = amplitude * phases
        spectrum[0] = 0.0
        if n_samples % 2 == 0:
            spectrum[-1] = spectrum[-1].real

        # Inverse FFT to time domain
        ts = np.fft.irfft(spectrum, n_samples)

        # Zero-mean & unit-variance
        ts -= np.mean(ts)
        ts /= np.std(ts, ddof=1)

        noise[i] = ts

    return noise


def inject_sinusoid(
    data: np.ndarray,
    fs: float,
    freq: float,
    amplitude: float,
    phase: float = 0.0,
    series_idx: np.ndarray = None,
) -> np.ndarray:
    """
    Inject a sinusoidal signal of given frequency and amplitude into time series.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("`data` must be 2D: (n_series, n_samples)")
    n_series, n_samples = data.shape

    # Time vector
    t = np.arange(n_samples) / fs

    # Construct sinusoid
    sinusoid = amplitude * np.sin(2 * np.pi * freq * t + phase)

    # Prepare output
    out = data.copy()

    # Determine which series to modify
    if series_idx is None:
        targets = range(n_series)
    else:
        targets = series_idx

    # Inject
    for idx in targets:
        out[idx] += sinusoid

    return out


def band_power(data: np.ndarray, fs: float, fmin: float, fmax: float) -> np.ndarray:
    """
    Compute total power in a frequency band [fmin, fmax] for each channel.
    """
    # Number of samples
    n_channels, n_samples = data.shape
    # Frequencies for one-sided FFT
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    # Compute real FFT along time axis
    fft_vals = np.fft.rfft(data, axis=1)
    # Power spectral density (unnormalized): |X|^2 / N
    psd = (np.abs(fft_vals) ** 2) / n_samples
    # Frequency resolution
    df = fs / n_samples
    # Boolean mask for band
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    # Integrate PSD within band for each channel
    power = np.sum(psd[:, band_mask], axis=1) * df
    return power


def plot_time_series(time_series, fs, fmin, fmax):
    n_samples = len(time_series)

    # Time vector for plotting
    # t = np.arange(n_samples) / fs
    # Plot time-domain waveform
    # plt.figure()
    # plt.plot(t, time_series)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude")
    # plt.title("IIR-Filtered Noise Time Series")
    # plt.grid(True)

    # Compute and plot power spectral density (PSD)
    freq = np.fft.rfftfreq(n_samples, d=1 / fs)
    spectrum = np.fft.rfft(time_series)
    psd = (np.abs(spectrum) ** 2) / n_samples
    # Plot PSD
    plt.figure()
    plt.plot(freq, psd)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (V²/Hz)")
    plt.title("IIR-Filtered Noise PSD")
    plt.grid(True, which="both")
    plt.xlim(fmin, fmax)
    plt.ylim(1e-3, 1e3)
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage and plotting
    fs = 1000  # Sampling frequency (Hz)
    duration = 1.5  # Duration (seconds)
    s = 64

    fmin = 8
    fmax = 25

    N = 1000  # Number of epochs to generate

    """
    # This is how you grab the indices of the channels of interest given an mne.montage
    for ch in ['fc2', 'fc4', 'fc6', 'c2', 'c4', 'c6', 'cp2', 'cp6']:
        print(montage.ch_names.index(ch))
        
    for ch in ['fc1', 'fc3', 'fc5', 'c1', 'c3', 'c5', 'cp1', 'cp5']:
        print(montage.ch_names.index(ch))
    """
    # Indices to use for the descync (motor imagery)
    ch_idx = {
        "left": {
            "fc2": 53,
            "fc4": 52,
            "fc6": 56,
            "c2": 50,
            "c4": 49,
            "c6": 48,
            "cp2": 40,
            "cp6": 45,
        },
        "right": {
            "fc1": 6,
            "fc3": 14,
            "fc5": 13,
            "c1": 15,
            "c3": 19,
            "c5": 21,
            "cp1": 20,
            "cp5": 25,
        },
    }

    # Define empty container for all the scaled epochs
    epochs = []
    # Define container for the labels, we will encode 0 as 'left' and 1 as 'right'
    Y = np.random.randint(0, 2, N)
    # Define scaling for all trials
    alpha_amps = np.random.uniform(0.25, 4, size=N)

    for i, alpha_amp in enumerate(alpha_amps):
        # Generate pink or brown noise, it contains white noise if variation > 0
        eeg_series = generate_smooth_noise_iir(
            s,
            fs,
            duration,
            noise_type="brown",
            fmin=fmin,
            fmax=fmax,
            variation=0.1,
        )

        # Inject alpha waves
        injected0_series = inject_sinusoid(
            eeg_series, fs, freq=9, amplitude=alpha_amp / 6
        )
        injected0_series = inject_sinusoid(
            injected0_series, fs, freq=9.5, amplitude=alpha_amp / 3
        )
        injected0_series = inject_sinusoid(
            injected0_series, fs, freq=10, amplitude=alpha_amp
        )
        injected0_series = inject_sinusoid(
            injected0_series, fs, freq=10.5, amplitude=alpha_amp / 3
        )
        injected0_series = inject_sinusoid(
            injected0_series, fs, freq=11, amplitude=alpha_amp / 6
        )
        # Inject beta waves
        beta_amp = alpha_amp / 3
        injected1_series = inject_sinusoid(
            injected0_series, fs, freq=19, amplitude=beta_amp / 6
        )
        injected1_series = inject_sinusoid(
            injected1_series, fs, freq=19.5, amplitude=beta_amp / 3
        )
        injected1_series = inject_sinusoid(
            injected1_series, fs, freq=20, amplitude=beta_amp
        )
        injected1_series = inject_sinusoid(
            injected1_series, fs, freq=20.5, amplitude=beta_amp / 3
        )
        injected1_series = inject_sinusoid(
            injected1_series, fs, freq=21, amplitude=beta_amp / 6
        )

        if Y[i] == 0:
            # This is a "move left" epoch
            # We replace injected alpha and beta waves in the set of electrodes associtated to c4 (contralateral)
            for ch in ch_idx["left"].keys():
                injected0_series[ch_idx["left"][ch], :] = eeg_series[
                    ch_idx["left"][ch], :
                ]
                injected1_series[ch_idx["left"][ch], :] = eeg_series[
                    ch_idx["left"][ch], :
                ]

        elif Y[i] == 1:
            # This is a "move right" epoch
            # We replace injected alpha and beta waves in the set of electrodes associtated to c3 (contralateral)
            for ch in ch_idx["right"].keys():
                injected0_series[ch_idx["right"][ch], :] = eeg_series[
                    ch_idx["right"][ch], :
                ]
                injected1_series[ch_idx["right"][ch], :] = eeg_series[
                    ch_idx["right"][ch], :
                ]

        # fig, ax = plt.subplots(1, 3)
        # ax[0].matshow(np.cov(eeg_series))
        # ax[1].matshow(np.cov(injected0_series))
        # ax[2].matshow(np.cov(injected1_series))

        epochs.append(injected1_series)

        # if i > 10:
        #    break

    # fig, ax = plt.subplots(1,3)
    # ax[0].matshow(np.cov(eeg_series))
    # ax[1].matshow(np.cov(injected0_series))
    # ax[2].matshow(np.cov(injected1_series))

    # plot_time_series(eeg_series[0], fs, fmin, fmax)
    # plot_time_series(injected0_series[0], fs, fmin, fmax)
    # plot_time_series(injected1_series[0], fs, fmin, fmax)

    # Compute covariance matrix of the time series for each epoch, independent variable
    X = []

    for epoch in epochs:
        # pwr0 = np.mean(band_power(eeg_series, fs, fmin=8, fmax=12))
        # pwr1 = np.mean(band_power(injected0_series, fs, fmin=8, fmax=12))
        X.append(np.cov(epoch))

    X = np.array(X)
    Y = Y.astype(float)

    np.save("./data/classification_X.npy", X)
    np.save("./data/classification_Y.npy", Y)
