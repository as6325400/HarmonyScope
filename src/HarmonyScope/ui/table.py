from rich.table import Table
import numpy as np
from HarmonyScope.core.constants import PITCH_CLASS_NAMES
from typing import List, Dict, Any

__all__ = ["make_pitch_class_table", "make_detected_notes_table"]


def make_pitch_class_table(pitch_data_by_pc: List[Dict]) -> Table:
    """
    Creates a fixed-row rich Table (1 row per pitch class) displaying aggregated info.

    pitch_data_by_pc: a list of 12 dicts, one for each pitch class (0-11), with aggregated info.
      Each dict includes: {'pc', 'name', 'detection_count', 'total_contributions',
                           'min_required_frames', 'total_voiced_frames', 'active',
                           'avg_prominence_db', 'avg_peak_level_db', 'avg_level_diff_db'}
    """
    # Define columns for the fixed pitch class table
    columns = [
        {"header": "PC", "justify": "center"},  # Pitch Class (C, C#, etc.)
        {
            "header": "Frames W/PC",
            "justify": "center",
        },  # Count of frames this PC was detected in across ANY octave
        {
            "header": "Min Req Frames",
            "justify": "center",
        },  # Minimum frames required for PC to be active
        {
            "header": "Total Voiced Frames",
            "justify": "center",
        },  # Total voiced frames in window
        {
            "header": "Total Peaks Found",
            "justify": "center",
        },  # Total individual peaks detected for this PC across ALL octaves
        {
            "header": "Avg Prom (dB)",
            "justify": "center",
        },  # Average prominence across detected peaks FOR THIS PC
        {
            "header": "Avg Level (dB)",
            "justify": "center",
        },  # Average peak level across detected peaks FOR THIS PC
        {
            "header": "Avg Diff (dB)",
            "justify": "center",
        },  # Average level diff across detected peaks FOR THIS PC
        {
            "header": "Active",
            "justify": "center",
        },  # Whether this PC is considered "active" based on frame count
    ]

    table = Table(
        title="Pitch Class Activity (Aggregated across Octaves)", expand=False
    )

    # Add columns
    for col in columns:
        table.add_column(**col)

    # The pitch_data_by_pc list should always have 12 entries, one for each PC, sorted 0-11
    for pc_info in pitch_data_by_pc:
        name = pc_info.get("name", "N/A")
        frame_count = pc_info.get("detection_count", 0)  # Frame count
        total_peaks_found = pc_info.get(
            "total_contributions", 0
        )  # Total peak detections for this PC
        required_frames = pc_info.get("min_required_frames", 0)
        total_voiced = pc_info.get("total_voiced_frames", 0)
        active = "✔" if pc_info.get("active", False) else ""

        # Metrics are averages across individual peak detections for this PC
        prominence_avg = pc_info.get("avg_prominence_db", -np.inf)
        peak_level_avg = pc_info.get("avg_peak_level_db", -np.inf)
        level_diff_avg = pc_info.get("avg_level_diff_db", np.inf)

        # Handle -inf and +inf for display when no detections occurred for this PC
        prominence_display = (
            f"{prominence_avg:.1f}" if np.isfinite(prominence_avg) else "--"
        )
        peak_level_display = (
            f"{peak_level_avg:.1f}" if np.isfinite(peak_level_avg) else "--"
        )
        level_diff_display = (
            (f"{level_diff_avg:.1f}" if np.isfinite(level_diff_avg) else "--")
            if total_peaks_found > 0
            else "--"
        )

        table.add_row(
            name,
            str(frame_count),
            str(required_frames),
            str(total_voiced),
            str(total_peaks_found),  # Display total peaks
            prominence_display,
            peak_level_display,
            level_diff_display,
            active,
        )

    return table


def make_detected_notes_table(detected_notes_data: List[Dict[str, Any]]) -> Table:
    """
    Creates a rich Table displaying detailed information about individual detected notes (peaks).

    detected_notes_data: a list of dicts, each representing a single spectral peak detection.
      Each dict includes: {'frame_idx', 'midi_note', 'pc', 'octave', 'freq',
                           'prominence_db', 'peak_level_db', 'level_diff_db'}
    """
    columns = [
        {"header": "Note", "justify": "center"},  # e.g., C4, G#5
        {"header": "MIDI", "justify": "center"},  # MIDI note number
        {"header": "Freq (Hz)", "justify": "right"},  # Detected frequency
        {"header": "Prom (dB)", "justify": "right"},  # Prominence in dB
        {"header": "Level (dB)", "justify": "right"},  # Peak level in dB
        {
            "header": "Level Diff (dB)",
            "justify": "right",
        },  # dB difference from loudest peak in its frame
        {"header": "Frame", "justify": "center"},  # Frame index (for debugging)
    ]

    table = Table(
        title=f"Detected Individual Notes ({len(detected_notes_data)} total)",
        expand=False,
    )

    # Add columns
    for col in columns:
        table.add_column(**col)

    # Sort notes by MIDI note for consistent display
    sorted_notes = sorted(detected_notes_data, key=lambda x: x.get("midi_note", -1))

    if not sorted_notes:
        table.add_row(
            "[dim]No individual notes detected[/dim]", *["--"] * (len(columns) - 1)
        )
        return table

    for note_info in sorted_notes:
        # Format note name (e.g., C4, G#5)
        pc_name = PITCH_CLASS_NAMES[note_info.get("pc", 0)]
        octave = note_info.get("octave", -1)
        note_display = (
            f"{pc_name}{octave}" if octave >= 0 else pc_name
        )  # Handle octave 0 or potentially negative from conversion

        midi_note = note_info.get("midi_note", -1)
        freq = note_info.get("freq", np.nan)
        prominence_db = note_info.get("prominence_db", -np.inf)
        peak_level_db = note_info.get("peak_level_db", -np.inf)
        level_diff_db = note_info.get("level_diff_db", np.inf)
        frame_idx = note_info.get("frame_idx", -1)

        # Format numeric values
        freq_display = f"{freq:.1f}" if np.isfinite(freq) else "--"
        prominence_display = (
            f"{prominence_db:.1f}" if np.isfinite(prominence_db) else "--"
        )
        peak_level_display = (
            f"{peak_level_db:.1f}" if np.isfinite(peak_level_db) else "--"
        )
        level_diff_display = (
            f"{level_diff_db:.1f}" if np.isfinite(level_diff_db) else "--"
        )

        table.add_row(
            note_display,
            str(midi_note) if midi_note >= 0 else "--",
            freq_display,
            prominence_display,
            peak_level_display,
            level_diff_display,
            str(frame_idx) if frame_idx >= 0 else "--",
        )

    return table
