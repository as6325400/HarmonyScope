from typing import Generator, Set, Tuple, List, Dict, Any
import numpy as np
import logging
import librosa
import time  # Ensure time is imported

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
        # Give the buffer a moment to fill initially
        # Use a small timeout to prevent indefinite waiting if reader isn't working
        buffer_fill_start_time = time.time()
        while len(reader.get_buffer()) < analysis_window_frames:
            time.sleep(0.01)
            if (
                time.time() - buffer_fill_start_time > 5
            ):  # Wait max 5 seconds for buffer
                logger.warning(
                    "Buffer not filling. Check microphone or device settings."
                )
                # Attempt analysis with partial buffer? Or raise an error?
                # Let's continue but log the warning.
                break  # Exit buffer wait loop

        logger.info("Buffer filled. Starting analysis.")

        try:
            last_process_time = time.time()
            while True:
                current_time = time.time()

                # Process only if enough time has passed since the last analysis
                if current_time - last_process_time >= process_interval_sec:

                    y = reader.get_buffer()

                    # We need at least the analysis window size to process
                    if len(y) < analysis_window_frames:
                        # This should ideally not happen often if buffer_fill_start_time logic worked
                        logger.debug(
                            f"Buffer size ({len(y)}) smaller than window size ({analysis_window_frames}). Waiting..."
                        )
                        time.sleep(0.01)  # Wait briefly
                        continue

                    # Get the latest segment covering the analysis window size
                    seg = y[-analysis_window_frames:]

                    # Calculate overall RMS dB for the segment
                    segment_rms = np.sqrt(np.mean(seg**2))
                    # Use a low reference for dB conversion
                    if segment_rms > 1e-10:
                        segment_rms_db = librosa.amplitude_to_db(segment_rms, ref=1e-10)
                    else:
                        segment_rms_db = -120.0

                    # active_pitches_array returns active PCs, aggregated PC data, detailed peak data, total voiced frames
                    (
                        active_pitch_classes,
                        pitch_data_by_pc,
                        detailed_peak_detections,  # This list contains note info with octaves
                        total_voiced_frames,
                    ) = active_pitches_array(
                        seg,
                        reader.sr,
                        frame_energy_thresh_db=self.frame_energy_thresh_db,
                        min_frame_ratio=self.min_frame_ratio,
                        min_prominence_db=self.min_prominence_db,
                        max_level_diff_db=self.max_level_diff_db,
                    )

                    # Identify the chord. Pass BOTH active pitch classes (for pattern matching)
                    # and detailed peak detections (to inform root selection via bass note).
                    chord = identify_chord(
                        active_pitch_classes, detailed_peak_detections
                    )

                    # Yield all the data needed by the CLI display
                    yield (
                        chord,
                        active_pitch_classes,  # Still yield for debugging/display if needed
                        pitch_data_by_pc,
                        detailed_peak_detections,  # Include detailed data here
                        segment_rms_db,
                        total_voiced_frames,
                    )

                    last_process_time = current_time

                # Small sleep to prevent busy-waiting
                time.sleep(0.05)

        except KeyboardInterrupt:
            # This is handled gracefully by the CLI main function's outer try/except
            pass
        except Exception as e:
            logger.error(
                f"An error occurred during stream analysis: {e}", exc_info=True
            )
            # Re-raise the exception after logging so the CLI's outer block catches it
            raise
        finally:
            # Ensure the MicReader stream is stopped when the generator loop exits
            if hasattr(reader, "stop"):
                logger.info("Stopping audio stream.")
                reader.stop()
