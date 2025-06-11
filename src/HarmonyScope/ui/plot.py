import matplotlib.pyplot as plt, numpy as np
from HarmonyScope.core.constants import PITCH_CLASS_NAMES


def plot_wave(wave: np.ndarray, start: float, win: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 2.7))
    t = np.linspace(start, start + win, len(wave))
    ax.plot(t, wave, linewidth=0.6, color="#1e90ff")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    return fig


def plot_spec(
    spec: np.ndarray, start: float, win: float, sr: int = 22050
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 3.6))

    n_bins = spec.shape[0]
    max_freq_khz = sr / 2 / 1000
    freqs = np.linspace(0, max_freq_khz, n_bins)

    energy_per_bin = spec.mean(axis=1)

    threshold = 0.1 * energy_per_bin.max()
    active_bins = np.where(energy_per_bin > threshold)[0]

    if active_bins.size > 0:
        min_freq = freqs[active_bins[0]]
        max_freq = freqs[active_bins[-1]]
    else:
        min_freq, max_freq = 0, max_freq_khz  # fallback

    im = ax.imshow(
        spec,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[start, start + win, 0, max_freq_khz],
        vmin=0,
        vmax=1,
    )
    ax.set_ylim(min_freq, max_freq)
    ax.set_ylabel("Frequency (kHz)")
    ax.set_xlabel("Time (s)")
    fig.colorbar(im, ax=ax, fraction=0.046).set_label("Norm dB")
    fig.tight_layout()
    return fig


def plot_chroma(chroma: np.ndarray, start: float, win: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    im = ax.imshow(
        chroma,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[start, start + win, 0, 12],
        vmin=0,
        vmax=1,
    )
    ax.set_yticks(np.arange(0.5, 12.5, 1))
    ax.set_yticklabels(PITCH_CLASS_NAMES)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch class")
    fig.colorbar(im, ax=ax, fraction=0.046).set_label("Energy")
    fig.tight_layout()
    return fig
