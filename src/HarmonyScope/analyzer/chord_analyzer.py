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
        pitches, _ = active_pitches_array(
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
            
            pitches = active_pitches_array(
                seg, sr,
                frame_energy_thresh_db=self.frame_thresh_db,
                delta_db=self.delta_db
            )
            
            chord = identify_chord(pitches)
            
            yield start / sr, (start + win) / sr, chord
            
    def stream_mic_live(self, interval_sec: float = 0.5):
        """
        Keep fetching buffer from the reader and analyze it every interval.
        """
        import time
        
        reader = self.reader
        win_sec = interval_sec
        try:
            time.sleep(0.1)
            while True:
                y = reader.get_buffer()

                if len(y) < win_sec * reader.sr:
                    # print(len(y), win_sec * reader.sr)
                    print("(Waiting for more data...)")
                    time.sleep(interval_sec)
                    continue

                # 分析最近一段
                seg = y[-int(win_sec * reader.sr):]
                pitches, table = active_pitches_array(
                    seg, reader.sr,
                    frame_energy_thresh_db=self.frame_thresh_db,
                    delta_db=self.delta_db
                )
                chord = identify_chord(pitches)
                yield chord, table
                time.sleep(interval_sec)
                
        except KeyboardInterrupt:
            print("Stopped by user.")
