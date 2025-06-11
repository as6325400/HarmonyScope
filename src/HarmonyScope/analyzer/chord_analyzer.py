from typing import Generator, Set, Tuple, List, Dict, Any
import numpy as np
import logging
import librosa
import time  # Ensure time is imported
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

from ..io.base import AudioReader

# active_pitches_array returns active PCs, PC data, detailed peak data, total voiced frames
from ..core.pitch import active_pitches_array

# identify_chord now accepts active_pitch_classes AND detailed_peak_detections
from ..core.chord import identify_chord


class ChordAnalyzer:
    """High‑level API: file, timeline, stream."""

    def __init__(
        self,
        reader: AudioReader,
        win_sec: float = 1.0,
        hop_sec: float = 0.5,  # hop_sec is not used in stream_mic_live currently, but kept for timeline
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

    def _analyze_segment(self, seg: np.ndarray, sr: int) -> Tuple[
        str | None,
        Set[int],
        List[Dict],
        List[Dict[str, Any]],
        float,
        int,
    ]:
        """Core analysis shared by mic & file."""
        segment_rms = np.sqrt(np.mean(seg**2))
        segment_rms_db = (
            librosa.amplitude_to_db(segment_rms, ref=1e-10)
            if segment_rms > 1e-10
            else -120.0
        )

        active_pcs, pc_data, detailed_peaks, voiced = active_pitches_array(
            seg,
            sr,
            frame_energy_thresh_db=self.frame_energy_thresh_db,
            min_frame_ratio=self.min_frame_ratio,
            min_prominence_db=self.min_prominence_db,
            max_level_diff_db=self.max_level_diff_db,
        )

        chord = identify_chord(active_pcs, detailed_peaks)

        return (
            chord,
            active_pcs,
            pc_data,
            detailed_peaks,
            segment_rms_db,
            voiced,
        )

    # -------- single file --------
    # This method is currently only used by the file_analyze CLI, which doesn't display detailed notes.
    # It will continue to use the simpler identify_chord logic that only takes pitch classes.
    # If we wanted file_analyze to also display detailed notes, we'd need to modify it.
    # For now, we'll keep analyze_file focused on just the final chord string result.
    def analyze_file(self, path: str) -> str | None:
        y, sr = self.reader(path)
        # active_pitches_array returns active PCs, PC data, detailed peak data, total voiced frames
        # We only need the active_pitch_classes for identify_chord in this specific method
        active_pitch_classes, _, detailed_peak_detections, _ = (
            active_pitches_array(  # Get detailed peaks here too
                y,
                sr,
                frame_energy_thresh_db=self.frame_energy_thresh_db,
                min_frame_ratio=self.min_frame_ratio,
                min_prominence_db=self.min_prominence_db,
                max_level_diff_db=self.max_level_diff_db,
            )
        )

        # Now passing detailed_peak_detections to identify_chord
        return identify_chord(
            active_pitch_classes, detailed_peak_detections
        )  # Pass detailed peaks

    # -------- sliding‑window timeline --------
    # Similar to analyze_file, this method is for generating a sequence of chords.
    # It also only needs the final chord string per segment.
    def timeline(
        self, path: str
    ) -> Generator[tuple[float, float, str | None], None, None]:
        y, sr = self.reader(path)
        hop = int(self.hop_sec * sr)
        win = int(self.win_sec * sr)
        for start in range(0, len(y) - win + 1, hop):
            seg = y[start : start + win]

            # active_pitches_array returns active PCs, PC data, detailed peak data, total voiced frames
            # We only need the active_pitch_classes for identify_chord in this specific method
            active_pitch_classes, _, detailed_peak_detections, _ = (
                active_pitches_array(  # Get detailed peaks here too
                    seg,
                    sr,
                    frame_energy_thresh_db=self.frame_energy_thresh_db,
                    min_frame_ratio=self.min_frame_ratio,
                    min_prominence_db=self.min_prominence_db,
                    max_level_diff_db=self.max_level_diff_db,
                )
            )

            # Now passing detailed_peak_detections to identify_chord
            chord = identify_chord(
                active_pitch_classes, detailed_peak_detections
            )  # Pass detailed peaks

            yield start / sr, (start + win) / sr, chord

    def stream_file_live(
        self, path: str
    ) -> list[
        Tuple[str | None, set[int], list[dict], list[dict[str, Any]], float, int]
    ]:
        """
        Parallel version: analyzes all segments in parallel and returns a list instead of a generator.
        """
        y, sr = self.reader(path)
        win = int(self.win_sec * sr)
        hop = int(self.hop_sec * sr)

        segments = [y[start : start + win] for start in range(0, len(y) - win + 1, hop)]

        results = Parallel(n_jobs=-1, prefer="processes")(
            delayed(self._analyze_segment)(seg, sr) for seg in segments
        )

        return results

    # This is the main method for the mic_analyze CLI
    def stream_mic_live(self, interval_sec: float = 0.05) -> Generator[
        Tuple[str | None, Set[int], List[Dict], List[Dict[str, Any]], float, int],
        None,
        None,
    ]:
        """
        Keep fetching buffer from the reader and analyze it periodically.
        Analyzes the latest `win_sec` data every `interval_sec` (or faster if possible).
        Yields detected chord, active pitch classes, aggregated PC data,
        detailed list of detected notes (peaks), segment RMS dB, and total voiced frame count.
        """

        reader = self.reader
        analysis_window_frames = int(self.win_sec * reader.sr)
        process_interval_sec = interval_sec

        logger.info(f"Waiting for initial buffer ({self.win_sec:.1f} seconds)...")
        buffer_fill_start_time = time.time()
        while len(reader.get_buffer()) < analysis_window_frames:
            time.sleep(0.01)
            if time.time() - buffer_fill_start_time > 5:
                logger.warning(
                    "Buffer not filling. Check microphone or device settings."
                )
                break

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
                        time.sleep(0.01)
                        continue

                    seg = y[-analysis_window_frames:]

                    result_tuple = self._analyze_segment(seg, reader.sr)

                    yield result_tuple

                    last_process_time = current_time

                time.sleep(0.05)

        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.error(
                "An error occurred during stream analysis: %s", e, exc_info=True
            )
            raise
        finally:
            if hasattr(reader, "stop"):
                logger.info("Stopping audio stream.")
                reader.stop()
