from typing import Generator
import numpy as np

from ..io.base import AudioReader
from ..core.pitch import active_pitches_array
from ..core.chord import identify_chord


class ChordAnalyzer:
    """High‑level API: file, timeline, stream."""

    def __init__(self,
                 reader: AudioReader,
                 win_sec: float = 1.0,
                 hop_sec: float = 0.5,
                 frame_thresh_db: float = -40,
                 delta_db: float = 2):
        self.reader = reader
        self.win_sec = win_sec
        self.hop_sec = hop_sec
        self.frame_thresh_db = frame_thresh_db
        self.delta_db = delta_db

    # -------- single file --------
    def analyze_file(self, path: str) -> str | None:
        y, sr = self.reader(path)
        pitches = active_pitches_array(
            y, sr,
            frame_energy_thresh_db=self.frame_thresh_db,
            delta_db=self.delta_db,
        )
        return identify_chord(pitches)

    # -------- sliding‑window timeline --------
    def timeline(self, path: str) -> Generator[tuple[float,float,str|None], None, None]:
        y, sr = self.reader(path)
        hop = int(self.hop_sec * sr)
        win = int(self.win_sec * sr)
        for start in range(0, len(y) - win + 1, hop):
            seg = y[start:start + win]
            chord = identify_chord(
                active_pitches_array(
                    seg, sr,
                    frame_energy_thresh_db=self.frame_thresh_db,
                    delta_db=self.delta_db
                )
            )
            yield start / sr, (start + win) / sr, chord
