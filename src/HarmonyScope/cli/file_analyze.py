from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any

import gradio as gr
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from HarmonyScope import set_verbosity
from HarmonyScope.analyzer.chord_analyzer import ChordAnalyzer
from HarmonyScope.cli.common_args import add_common_args
from HarmonyScope.io.file_reader import FileReader

Frame = Dict[str, Any]

# ---------- stream_file_live → frames ----------


def prepare_frames(
    ana: ChordAnalyzer, wav_path: Path
) -> tuple[List[Frame], float, float]:
    y, sr = librosa.load(wav_path, sr=None)
    results = list(ana.stream_file_live(str(wav_path)))
    hop_sec, win_sec = ana.hop_sec, ana.win_sec
    win_len = int(win_sec * sr)
    hop_len = int(hop_sec * sr)

    frames: List[Frame] = []
    for idx, res in enumerate(results):
        t0 = round(idx * hop_sec, 3)
        seg = y[idx * hop_len : idx * hop_len + win_len]

        # Down-sampled wave (for faster plot)
        wave = librosa.resample(seg, orig_sr=sr, target_sr=2000)

        # Pre-compute dB-scaled spec as 0-1 float32 image (saves conversion later)
        spec = librosa.amplitude_to_db(
            np.abs(librosa.stft(seg, n_fft=512, hop_length=128)), ref=np.max
        )[:128]
        spec = ((np.nan_to_num(spec, neginf=-80, posinf=0) + 80) / 80).clip(0, 1)

        chroma = librosa.feature.chroma_stft(y=seg, sr=sr, hop_length=128)
        chroma = np.nan_to_num(chroma, nan=0.0).clip(0, 1)

        frames.append(
            dict(
                t=t0,
                wave=wave.astype(np.float32),
                spec=spec.astype(np.float32),
                chroma=chroma.astype(np.float32),
                chord=res[0] or "None",
                pc_summary=res[2],
            )
        )
    return frames, hop_sec, win_sec


# ---------- draw ----------


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
    ax.set_yticklabels(
        ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch class")
    fig.colorbar(im, ax=ax, fraction=0.046).set_label("Energy")
    fig.tight_layout()
    return fig


# ---------- Gradio ----------


def build_gradio_app(frames: List[Frame], hop: float, win: float) -> gr.Blocks:
    """Minimal-whitespace responsive layout using only props compatible with older Gradio versions."""

    def render(idx: int):
        f = frames[idx]
        w_fig = plot_wave(f["wave"], f["t"], win)
        s_fig = plot_spec(f["spec"], f["t"], win)
        c_fig = plot_chroma(f["chroma"], f["t"], win)
        t_range = f"{f['t']:.2f} – {(f['t']+win):.2f} s"
        df = pd.DataFrame(f["pc_summary"])[["name", "detection_count"]]
        df.columns = ["PC", "Frames"]
        return w_fig, s_fig, c_fig, f["chord"], t_range, df

    # Styles: shrink container + limit table height via CSS so we can drop max_rows param
    css = """
        .container { max-width: 100% !important; }
        .pc-table .wrap.svelte-1ipelgc { height: 240px !important; overflow-y: auto; }
    """

    with gr.Blocks(title="HarmonyScope Viewer", css=css) as demo:
        gr.Markdown("## 🎧 HarmonyScope Interactive Viewer")

        with gr.Row(equal_height=True):
            # ---------- Left side ----------
            with gr.Column(scale=1, min_width=260):
                slider = gr.Slider(
                    0,
                    len(frames) - 1,
                    step=1,
                    value=0,
                    label=f"Frame (hop = {hop:.2f}s)",
                )
                time_box = gr.Textbox(label="Time range", interactive=False)
                chord_box = gr.Textbox(label="Detected chord", interactive=False)
                pc_table = gr.Dataframe(
                    headers=["PC", "Frames"],
                    datatype=["str", "int"],
                    label="Pitch-class summary",
                    wrap=True,
                    elem_classes=["pc-table"],
                )

            # ---------- Right side ----------
            with gr.Column(scale=3):
                wave_plot = gr.Plot(label="Waveform")
                spec_plot = gr.Plot(label="Spectrogram")
                chroma_plot = gr.Plot(label="Chroma")

        # Connect interaction
        slider.change(
            render,
            inputs=slider,
            outputs=[wave_plot, spec_plot, chroma_plot, chord_box, time_box, pc_table],
            show_progress="minimal",
        )
        render(0)  # initial render

    return demo


# ---------- CLI ----------


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="harmonyscope-file",
        description="Analyze single audio file and launch Gradio viewer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(ap)
    ap.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to audio file (wav/flac/mp3), e.g. --path ./samples/example.wav",
    )
    args = ap.parse_args()
    set_verbosity(args.verbose)

    wav_path = Path(args.path).expanduser().resolve()
    if not wav_path.exists():
        raise FileNotFoundError(wav_path)

    ana = ChordAnalyzer(
        reader=FileReader(),
        win_sec=args.window,
        min_frame_ratio=args.min_frame_ratio,
        min_prominence_db=args.min_prominence_db,
        max_level_diff_db=args.max_level_diff_db,
        frame_energy_thresh_db=args.frame_energy_thresh_db,
        hop_sec=args.interval,
    )

    frames, hop_sec, win_sec = prepare_frames(ana, wav_path)
    gradio_app = build_gradio_app(frames, hop_sec, win_sec)
    print(f"🔗 Launching Gradio for: {wav_path.name}")
    gradio_app.launch()


if __name__ == "__main__":
    main()
