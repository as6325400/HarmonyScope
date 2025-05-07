from typing import Generator, Set, Tuple, List, Dict  # Import types
import numpy as np
import logging
import librosa

logger = logging.getLogger(__name__)

from ..io.base import AudioReader

# active_pitches_array now returns pitch classes and PC data
from ..core.pitch import active_pitches_array
from ..core.chord import identify_chord


class ChordAnalyzer:
    # ... (init and parameters unchanged) ...
    """High‑level API: file, timeline, stream."""

    def __init__(
        self,
        reader: AudioReader,
        win_sec: float = 1.0,
        hop_sec: float = 0.5,
        frame_energy_thresh_db: float = -40,
        min_frame_ratio: float = 0.3,
        min_prominence_db: float = 8,
        max_level_diff_db: float = 15,
    ):
        self.reader = reader
        self.win_sec = win_sec
        self.hop_sec = hop_sec
        self.frame_energy_thresh_db = frame_energy_thresh_db
        self.min_frame_ratio = min_frame_ratio
        self.min_prominence_db = min_prominence_db
        self.max_level_diff_db = max_level_diff_db

    # Removed _get_active_pitch_classes helper as pitch.py now returns pitch classes

    # -------- single file --------
    def analyze_file(self, path: str) -> str | None:
        y, sr = self.reader(path)
        # active_pitches_array now returns pitch classes, PC data, total voiced frames
        active_pitch_classes, _, _ = active_pitches_array(
            y,
            sr,
            frame_energy_thresh_db=self.frame_energy_thresh_db,
            min_frame_ratio=self.min_frame_ratio,
            min_prominence_db=self.min_prominence_db,
            max_level_diff_db=self.max_level_diff_db,
        )
        # No need to convert to pitch classes here, pitch.py already did it

        return identify_chord(active_pitch_classes)

    # -------- sliding‑window timeline --------
    def timeline(
        self, path: str
    ) -> Generator[tuple[float, float, str | None], None, None]:
        y, sr = self.reader(path)
        hop = int(self.hop_sec * sr)
        win = int(self.win_sec * sr)
        for start in range(0, len(y) - win + 1, hop):
            seg = y[start : start + win]

            # active_pitches_array now returns pitch classes, PC data, total voiced frames
            active_pitch_classes, _, _ = active_pitches_array(
                seg,
                sr,
                frame_energy_thresh_db=self.frame_energy_thresh_db,
                min_frame_ratio=self.min_frame_ratio,
                min_prominence_db=self.min_prominence_db,
                max_level_diff_db=self.max_level_diff_db,
            )
            # No need to convert

            chord = identify_chord(active_pitch_classes)

            yield start / sr, (start + win) / sr, chord

    # Update type hint to reflect new return values
    def stream_mic_live(
        self, interval_sec: float = 0.05
    ) -> Generator[Tuple[str | None, Set[int], List[Dict], float, int], None, None]:
        """
        Keep fetching buffer from the reader and analyze it periodically.
        Analyzes the latest `win_sec` data every `interval_sec` (or faster if possible).
        Yields detected chord, active pitch classes, detailed PC data, segment RMS dB,
        and total voiced frame count.
        """
        import time

        reader = self.reader
        analysis_window_frames = int(self.win_sec * reader.sr)
        process_interval_sec = interval_sec

        logger.info(f"Waiting for initial buffer ({self.win_sec:.1f} seconds)...")
        while len(reader.get_buffer()) < analysis_window_frames:
            time.sleep(0.05)
        logger.info("Buffer filled. Starting analysis.")

        try:
            last_process_time = time.time()
            while True:
                current_time = time.time()

                if current_time - last_process_time >= process_interval_sec:

                    y = reader.get_buffer()

                    if len(y) < analysis_window_frames:
                        logger.debug(
                            f"Buffer size ({len(y)}) smaller than window size ({analysis_window_frames}). Waiting..."
                        )
                        time.sleep(0.05)
                        continue

                    seg = y[-analysis_window_frames:]

                    # Calculate overall RMS dB for the segment
                    segment_rms = np.sqrt(np.mean(seg**2))
                    if segment_rms > 1e-9:
                        segment_rms_db = librosa.amplitude_to_db(segment_rms, ref=1e-9)
                    else:
                        segment_rms_db = -120.0

                    # active_pitches_array now returns active PCs, PC data, total voiced frames
                    active_pitch_classes, pitch_data_by_pc, total_voiced_frames = (
                        active_pitches_array(
                            seg,
                            reader.sr,
                            frame_energy_thresh_db=self.frame_energy_thresh_db,
                            min_frame_ratio=self.min_frame_ratio,
                            min_prominence_db=self.min_prominence_db,
                            max_level_diff_db=self.max_level_diff_db,
                        )
                    )

                    # Identify the chord based on the active pitch classes
                    chord = identify_chord(active_pitch_classes)

                    # Yield the chord, active pitch classes, detailed PC data, RMS dB, and total voiced frames
                    yield chord, active_pitch_classes, pitch_data_by_pc, segment_rms_db, total_voiced_frames

                    last_process_time = current_time

                time.sleep(0.005)

        except KeyboardInterrupt:
            print("Stopped by user.")
        except Exception as e:
            logger.error(
                f"An error occurred during stream analysis: {e}", exc_info=True
            )
            raise
        finally:
            if hasattr(reader, "stop"):
                logger.info("Stopping audio stream.")
                reader.stop()
