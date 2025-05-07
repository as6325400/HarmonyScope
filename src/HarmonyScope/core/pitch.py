import numpy as np
import librosa
import logging
import scipy.signal
from collections import Counter
from typing import Set, Tuple, List, Dict, Any  # Import Dict, Any
from .constants import PITCH_CLASS_NAMES

logger = logging.getLogger(__name__)


def active_pitches_array(
    y,
    sr,
    *,
    frame_energy_thresh_db=-40,  # Still used for filtering silent frames
    min_frame_ratio=0.3,  # Minimum ratio of voiced frames a pitch must be detected in
    cqt_bins_per_octave=24,  # CQT resolution (higher = more detailed)
    peak_height_percentile=90,  # Peaks must be above this percentile in frame magnitude (linear)
    min_prominence_db=8,  # Peaks must have this minimum prominence in dB (relative to local spectral floor)
    max_level_diff_db=15,  # Peaks must be within this many dB of the loudest peak in the frame
    peak_distance_bins=3,  # Minimum horizontal distance (in CQT bins) between peaks
):
    """
    Identify active pitch classes (0-11) using spectral peak picking, prominence,
    and relative level filtering, aggregating results across octaves.

    Args:
        y (np.ndarray): Audio waveform.
        sr (int): Sampling rate.
        frame_energy_thresh_db (float): Energy threshold (dB relative to a low ref)
                                        to consider a frame "voiced".
        min_frame_ratio (float): Minimum ratio of voiced frames in the window
                                 a *pitch class* must be detected in to be active.
        cqt_bins_per_octave (int): Number of bins per octave for CQT.
        peak_height_percentile (float): Percentile of CQT *linear* magnitude values
                                        within a frame a peak must exceed.
        min_prominence_db (float): Minimum required prominence (in dB) for a peak.
        max_level_diff_db (float): Maximum allowed difference (in dB) between a peak's
                                   level and the maximum peak level within the frame.
        peak_distance_bins (int): Minimum horizontal distance (in CQT bins) between peaks.

    Returns:
        Tuple[Set[int], List[Dict], List[Dict[str, Any]], int]: A tuple containing:
            - A set of active pitch class indices (0-11).
            - A list of 12 dictionaries, one for each pitch class (0-11),
              with aggregated debug information ('pc', 'name', 'detection_count',
              'avg_prominence_db', 'avg_peak_level_db', 'avg_level_diff_db', 'active').
              Includes 'min_required_frames', 'total_voiced_frames'.
            - A list of dictionaries, where each dictionary represents a single
              detected spectral peak (note detection) including its MIDI note,
              pitch class, octave, frequency, and metrics within its frame.
            - The total count of voiced CQT frames processed.
    """
    # Using default hop_length for librosa functions (usually 512) for consistent frame counts
    hop_length = 512

    # 1. frame RMS to find voiced frames
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=1e-9)
    voiced_rms_frames = rms_db > frame_energy_thresh_db

    # Handle case with no audio or silence
    if np.all(~voiced_rms_frames):
        logger.debug("No voiced frames detected within energy threshold.")
        # Return empty set, a list of 12 zero-filled dicts, an empty peak list, and 0 voiced frames
        table_data = []
        for i, name in enumerate(PITCH_CLASS_NAMES):
            table_data.append(
                {
                    "pc": i,
                    "name": name,
                    "detection_count": 0,
                    "avg_prominence_db": -np.inf,
                    "avg_peak_level_db": -np.inf,
                    "avg_level_diff_db": np.inf,
                    "active": False,
                    "min_required_frames": 0,
                    "total_voiced_frames": 0,
                }
            )
        return set(), table_data, [], 0  # Return empty list for peak detections

    # 2. Compute CQT magnitude (linear and dB)
    fmin = librosa.midi_to_hz(36)  # C1
    n_bins = int(7 * cqt_bins_per_octave)  # C1 to C8

    CQT = librosa.cqt(
        y=y,
        sr=sr,
        hop_length=hop_length,
        bins_per_octave=cqt_bins_per_octave,
        fmin=fmin,
        n_bins=n_bins,
    )
    CQT_mag_linear = np.abs(CQT)
    CQT_mag_db = librosa.amplitude_to_db(CQT_mag_linear, ref=1e-9)

    # Ensure CQT and RMS have compatible number of frames
    min_frames = min(CQT_mag_linear.shape[1], len(voiced_rms_frames))
    CQT_mag_linear = CQT_mag_linear[:, :min_frames]
    CQT_mag_db = CQT_mag_db[:, :min_frames]
    voiced_rms_frames = voiced_rms_frames[:min_frames]

    # Get CQT frequencies and map to MIDI notes
    cqt_freqs = librosa.cqt_frequencies(
        n_bins=CQT.shape[0], fmin=fmin, bins_per_octave=cqt_bins_per_octave
    )
    cqt_midi = librosa.hz_to_midi(cqt_freqs)

    # --- Data structures to aggregate per Pitch Class (0-11) ---
    pc_detection_counts = Counter()  # Count frames where PC was detected
    pc_prominence_sums = Counter()
    pc_peak_level_sums = Counter()
    pc_level_diff_sums = Counter()
    pc_contributions_count = (
        Counter()
    )  # Track how many peak detections contributed to sums per PC

    # --- Data structure to collect individual peak detections across all frames ---
    all_peak_detections: List[Dict[str, Any]] = []

    voiced_cqt_frames_count = 0

    for frame_idx in range(CQT_mag_linear.shape[1]):
        if voiced_rms_frames[frame_idx]:
            voiced_cqt_frames_count += 1
            frame_mag_linear = CQT_mag_linear[:, frame_idx]
            frame_mag_db = CQT_mag_db[:, frame_idx]

            if np.max(frame_mag_linear) < librosa.db_to_amplitude(
                frame_energy_thresh_db + 5, ref=1e-9
            ):
                logger.debug(f"Frame {frame_idx}: Low energy, skipping peak detection.")
                continue

            peak_threshold_linear = 0
            if np.max(frame_mag_linear) > 1e-12:
                # Only consider positive magnitudes for percentile calculation
                positive_mags = frame_mag_linear[frame_mag_linear > 1e-12]
                if len(positive_mags) > 0:
                    peak_threshold_linear = np.percentile(
                        positive_mags, peak_height_percentile
                    )

            initial_peaks, _ = scipy.signal.find_peaks(
                frame_mag_linear,
                height=peak_threshold_linear,
                distance=peak_distance_bins,
            )

            if len(initial_peaks) == 0:
                logger.debug(f"Frame {frame_idx}: No initial peaks found.")
                continue

            prominences_linear, _, _ = scipy.signal.peak_prominences(
                frame_mag_linear, initial_peaks
            )
            prominences_db = librosa.amplitude_to_db(prominences_linear, ref=1e-9)

            prominence_filtered_indices = initial_peaks[
                prominences_db >= min_prominence_db
            ]
            prominence_filtered_prominences_db = prominences_db[
                prominences_db >= min_prominence_db
            ]

            prominence_filtered_peak_levels_db = frame_mag_db[
                prominence_filtered_indices
            ]

            if len(prominence_filtered_peak_levels_db) > 0:
                max_peak_level_db_in_frame = np.max(prominence_filtered_peak_levels_db)
                level_diffs_db = (
                    max_peak_level_db_in_frame - prominence_filtered_peak_levels_db
                )
                final_filtered_indices = prominence_filtered_indices[
                    level_diffs_db <= max_level_diff_db
                ]

                # Get corresponding metrics for the final filtered peaks
                final_filtered_prominences_db = prominence_filtered_prominences_db[
                    level_diffs_db <= max_level_diff_db
                ]
                final_filtered_levels_db = prominence_filtered_peak_levels_db[
                    level_diffs_db <= max_level_diff_db
                ]
                final_filtered_level_diffs_db = level_diffs_db[
                    level_diffs_db <= max_level_diff_db
                ]

            else:
                logger.debug(f"Frame {frame_idx}: No peaks passed prominence filter.")
                continue

            # Convert final filtered CQT bin indices to MIDI notes and then Pitch Classes
            frame_detected_pcs = set()
            for i, peak_bin in enumerate(final_filtered_indices):
                if 0 <= peak_bin < len(cqt_midi):
                    midi_note_float = cqt_midi[peak_bin]
                    # Filter out potential NaNs or out-of-range MIDI values from conversion
                    if not np.isnan(midi_note_float) and 0 <= midi_note_float <= 127:
                        midi_note = round(midi_note_float)
                        pc = midi_note % 12
                        # MIDI note to octave: C0 is MIDI 12. Octave = (MIDI - 12) // 12
                        octave = (
                            (midi_note - 12) // 12 if midi_note >= 12 else -1
                        )  # Handle notes below C0 if they somehow appear

                        # Add PC to the set detected in THIS frame (for frame count)
                        frame_detected_pcs.add(pc)

                        # Collect individual peak detection details
                        peak_info = {
                            "frame_idx": frame_idx,
                            "midi_note": midi_note,
                            "pc": pc,
                            "octave": octave,
                            "freq": cqt_freqs[peak_bin],  # Use the actual frequency
                            "prominence_db": final_filtered_prominences_db[i],
                            "peak_level_db": final_filtered_levels_db[i],
                            "level_diff_db": final_filtered_level_diffs_db[i],
                        }
                        all_peak_detections.append(peak_info)

                        # Aggregate metrics per PC (can have multiple contributions per frame across octaves)
                        pc_prominence_sums[pc] += final_filtered_prominences_db[i]
                        pc_peak_level_sums[pc] += final_filtered_levels_db[i]
                        pc_level_diff_sums[pc] += final_filtered_level_diffs_db[i]
                        pc_contributions_count[
                            pc
                        ] += 1  # Increment contribution count for this PC

            # Increment the frame count for each PC detected in this frame
            pc_detection_counts.update(frame_detected_pcs)  # THIS counts frames

    # --- Aggregate detections and determine active Pitch Classes ---

    # Handle case where no voiced CQT frames had significant peaks after filtering
    if (
        voiced_cqt_frames_count == 0 and not all_peak_detections
    ):  # pc_detection_counts would be empty too
        logger.debug("No MIDI notes detected in any voiced frames after filtering.")
        # Return empty set, a list of 12 zero-filled dicts, an empty peak list, and 0 voiced frames
        table_data = []
        for i, name in enumerate(PITCH_CLASS_NAMES):
            table_data.append(
                {
                    "pc": i,
                    "name": name,
                    "detection_count": 0,
                    "avg_prominence_db": -np.inf,
                    "avg_peak_level_db": -np.inf,
                    "avg_level_diff_db": np.inf,
                    "active": False,
                    "min_required_frames": 0,
                    "total_voiced_frames": 0,
                }
            )
        return set(), table_data, [], 0  # Return empty list for peak detections

    # Calculate minimum required *frames* for a PC to be active
    min_required_frames = int(voiced_cqt_frames_count * min_frame_ratio)
    # Ensure at least 1 frame is required if there are voiced frames and ratio > 0
    if voiced_cqt_frames_count > 0 and min_frame_ratio > 0 and min_required_frames == 0:
        min_required_frames = 1
    # If no voiced frames, min_required_frames remains 0, correctly yielding no active PCs

    active_pitch_classes = set()
    table_data = []  # List of 12 dicts, one per PC

    # Populate table_data for all 12 pitch classes
    for pc in range(12):
        name = PITCH_CLASS_NAMES[pc]
        frame_count = pc_detection_counts.get(
            pc, 0
        )  # Number of frames this PC was detected in
        total_contributions = pc_contributions_count.get(
            pc, 0
        )  # Total number of peaks found for this PC across all octaves

        # Calculate averages (handle case where PC was never detected)
        avg_prominence_db = (
            pc_prominence_sums.get(pc, 0) / total_contributions
            if total_contributions > 0
            else -np.inf
        )
        avg_peak_level_db = (
            pc_peak_level_sums.get(pc, 0) / total_contributions
            if total_contributions > 0
            else -np.inf
        )
        avg_level_diff_db = (
            pc_level_diff_sums.get(pc, 0) / total_contributions
            if total_contributions > 0
            else np.inf
        )  # Use +inf as default diff

        # A pitch class is active if its frame count meets the threshold
        # Only consider a PC active if there was at least one contribution/detection,
        # and the frame count meets the minimum required frames.
        is_active = (total_contributions > 0) and (frame_count >= min_required_frames)

        if is_active:
            active_pitch_classes.add(pc)

        info = {
            "pc": pc,
            "name": name,
            "detection_count": frame_count,  # This is the frame count
            "total_contributions": total_contributions,  # Total peaks detected for this PC across octaves
            "min_required_frames": min_required_frames,
            "total_voiced_frames": voiced_cqt_frames_count,
            "active": is_active,
            "avg_prominence_db": avg_prominence_db,
            "avg_peak_level_db": avg_peak_level_db,
            "avg_level_diff_db": avg_level_diff_db,
        }
        table_data.append(info)

    # Sort table data by pitch class index (C, C#, ...) for consistent order
    table_data.sort(key=lambda x: x["pc"])

    # Sort individual peak detections by MIDI note for consistent display order
    all_peak_detections.sort(key=lambda x: x["midi_note"])

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"Active Pitch Classes debug dump (min frames: {min_required_frames}/{voiced_cqt_frames_count}, min prom: {min_prominence_db:.1f}dB, max diff: {max_level_diff_db:.1f}dB):"
        )
        if voiced_cqt_frames_count > 0:
            for pitch_info in table_data:
                flag = "✔" if pitch_info["active"] else " "
                # Ensure denominator is not zero before calculating ratio
                ratio = (
                    pitch_info["detection_count"] / pitch_info["total_voiced_frames"]
                    if pitch_info["total_voiced_frames"] > 0
                    else 0
                )
                logger.debug(
                    f"{pitch_info['name']:>2}: Frames: {pitch_info['detection_count']:3} ({ratio:.1%}), "
                    f"Contribs: {pitch_info['total_contributions']:3}, "
                    f"Prom Avg: {pitch_info['avg_prominence_db']:5.1f}dB, "
                    f"Level Avg: {pitch_info['avg_peak_level_db']:5.1f}dB, "
                    f"Diff Avg: {pitch_info['avg_level_diff_db']:5.1f}dB {flag}"
                )

            # Log some details about individual peak detections if there are many
            if all_peak_detections:
                logger.debug(
                    f"Sample of individual peak detections ({len(all_peak_detections)} total):"
                )
                # Log up to the first 10 detections
                for peak_info in all_peak_detections[:10]:
                    logger.debug(
                        f"  Frame {peak_info['frame_idx']}: Note {PITCH_CLASS_NAMES[peak_info['pc']]}{peak_info['octave']} (MIDI {peak_info['midi_note']}, Freq {peak_info['freq']:.1f}Hz), "
                        f"Prom: {peak_info['prominence_db']:.1f}dB, Level: {peak_info['peak_level_db']:.1f}dB, Diff: {peak_info['level_diff_db']:.1f}dB"
                    )
        else:
            logger.debug("No voiced frames to report detections.")

    # Return the set of active Pitch Classes, the 12-entry table data,
    # the list of individual peak detections, and total voiced frame count
    return (
        active_pitch_classes,
        table_data,
        all_peak_detections,
        voiced_cqt_frames_count,
    )
