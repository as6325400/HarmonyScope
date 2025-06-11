from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any

from HarmonyScope.ui import gradio_viewer as viewer
from HarmonyScope.prep import file_frames
import numpy as np

from HarmonyScope import set_verbosity
from HarmonyScope.analyzer.chord_analyzer import ChordAnalyzer
from HarmonyScope.cli.common_args import add_common_args
from HarmonyScope.io.file_reader import FileReader


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
    frames, hop_sec, win_sec = file_frames.prepare_frames(ana=ana, path=wav_path)
    viewer.build_gradio_app(frames, hop_sec, win_sec).launch()
    print(f"🔗 Launching Gradio for: {wav_path.name}")


if __name__ == "__main__":
    main()
