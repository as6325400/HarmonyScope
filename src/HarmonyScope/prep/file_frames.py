from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import librosa, numpy as np
from tqdm import tqdm
from HarmonyScope.analyzer.chord_analyzer import ChordAnalyzer

Frame = Dict[str, Any]


def prepare_frames(path: Path, ana: ChordAnalyzer) -> tuple[List[Frame], float, float]:
    y, sr = librosa.load(path, sr=None, mono=True)
    hop_sec, win_sec = ana.hop_sec, ana.win_sec
    hop_len, win_len = int(hop_sec * sr), int(win_sec * sr)

    results = ana.stream_file_live(str(path))
    frames: List[Frame] = []
    for idx, res in enumerate(tqdm(results, desc="Analyzing")):
        seg = y[idx * hop_len : idx * hop_len + win_len]
        frames.append(
            dict(
                t=idx * hop_sec,
                wave=librosa.resample(seg, orig_sr=sr, target_sr=2_000).astype("f4"),
                spec=_spec(seg, sr),
                chroma=_chroma(seg, sr),
                chord=res[0] or "None",
                pc_summary=res[2],
            )
        )
    return frames, hop_sec, win_sec


# -- private helpers ---------------------------------
import numpy as np


def _spec(seg: np.ndarray, sr: int):
    S = librosa.amplitude_to_db(
        np.abs(librosa.stft(seg, n_fft=512, hop_length=128)), ref=np.max
    )[:128]
    return ((np.nan_to_num(S, neginf=-80) + 80) / 80).clip(0, 1).astype("f4")


def _chroma(seg: np.ndarray, sr: int):
    C = librosa.feature.chroma_stft(y=seg, sr=sr, hop_length=128)
    return np.nan_to_num(C).clip(0, 1).astype("f4")
