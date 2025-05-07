from HarmonyScope.io.mic_reader import list_input_devices, MicReader
from HarmonyScope.analyzer.chord_analyzer import ChordAnalyzer
import argparse, sys
import questionary
import numpy as np
from questionary import Choice
from HarmonyScope import set_verbosity
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.console import Group
from rich.panel import Panel
import logging
from HarmonyScope.core.constants import PITCH_CLASS_NAMES
from typing import List, Dict, Any  # Import for type hints

logger = logging.getLogger(__name__)


def choose_device_interactive() -> int:
    """Arrow‑key selector – returns the chosen PortAudio device id."""
    devices = list_input_devices()
    if not devices:
        raise RuntimeError(
            "No input devices found. Ensure PortAudio is installed and a microphone is connected."
        )

    choices = [Choice(title=f"[{idx}] {name}", value=idx) for idx, name in devices]

    print("Listing available input devices...")
    try:
        device_id = questionary.select(
            "Select input device (arrow keys, <Enter> to confirm):",
            choices=choices,
            qmark="❯",
            pointer="▶",
            instruction="",
        ).ask()

        if device_id is None:  # <Esc> or Ctrl-C during selection
            raise KeyboardInterrupt
        return device_id
    except KeyboardInterrupt:
        print("\nDevice selection cancelled. Exiting.")
        sys.exit(1)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--device",
        type=int,
        default=None,
        help="device id (use --device -1 to list & choose interactively)",
    )
    ap.add_argument(
        "--window",
        type=float,
        default=0.75,
        help="Analysis window size in seconds (default: 0.75). Affects latency and accuracy.",
    )
    ap.add_argument(
        "--interval",
        type=float,
        default=0.05,
        help="Minimum analysis interval in seconds (default: 0.05). Controls update rate.",
    )
    ap.add_argument(
        "--min-frame-ratio",
        type=float,
        default=0.3,
        help="Min ratio of voiced frames a pitch class must be detected in to be active (default: 0.3).",
    )
    ap.add_argument(
        "--min-prominence-db",
        type=float,
        default=8.0,
        help="Minimum peak prominence in dB for pitch detection (default: 8.0). Higher values filter more noise.",
    )
    ap.add_argument(
        "--max-level-diff-db",
        type=float,
        default=15.0,
        help="Maximum dB difference from loudest peak in frame for pitch detection (default: 15.0). Lower values filter more weak peaks/harmonics.",
    )
    ap.add_argument(
        "--frame-energy-thresh-db",
        type=float,
        default=-40.0,
        help="Energy threshold (dB relative to a low ref) to consider a frame voiced (default: -40.0). Lower values are more sensitive to quiet sounds.",
    )
    ap.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="-v Display DEBUG logs for HarmonyScope (e.g., frame-level pitch detections)",
    )

    args = ap.parse_args()

    set_verbosity(args.verbose)

    dev_id = args.device
    if dev_id is None or dev_id == -1:
        try:
            dev_id = choose_device_interactive()
        except KeyboardInterrupt:
            logger.info("Device selection cancelled. Exiting.")
            sys.exit(1)

    try:
        # Default sample rate is 44100
        sample_rate = 44100
        logger.info(f"Using device ID: {dev_id}, Sample Rate: {sample_rate}")
        reader = MicReader(
            device=dev_id, sr=sample_rate, maxlen_sec=args.window + 1
        )  # Ensure buffer is slightly larger than window
        ana = ChordAnalyzer(
            reader=reader,
            win_sec=args.window,
            min_frame_ratio=args.min_frame_ratio,
            min_prominence_db=args.min_prominence_db,
            max_level_diff_db=args.max_level_diff_db,
            frame_energy_thresh_db=args.frame_energy_thresh_db,
        )

        logger.info(
            f"Starting live analysis (Window: {args.window}s, Interval: {args.interval}s, Min Ratio: {args.min_frame_ratio:.1%}, Min Prominence: {args.min_prominence_db}dB, Max Level Diff: {args.max_level_diff_db}dB, Energy Thresh: {args.frame_energy_thresh_db}dB)"
        )

        with Live(auto_refresh=False, screen=True) as live:
            # stream_mic_live yields chord, active_pitch_classes, pitch_data_by_pc (aggregated),
            # detailed_peak_detections (new!), segment_rms_db, total_voiced_frames
            for (
                chord,
                active_pitch_classes,
                pitch_data_by_pc,
                detailed_peak_detections,  # Unpack the new list here
                segment_rms_db,
                total_voiced_frames,
            ) in ana.stream_mic_live(interval_sec=args.interval):

                renderables = []  # List to hold rich renderables

                # 1. Display Overall Level and Threshold
                level_text = Text(
                    f"Overall Segment Level: {segment_rms_db:.1f} dB | Voiced Threshold: {args.frame_energy_thresh_db:.1f} dB | Voiced Frames in Window: {total_voiced_frames}"
                )
                renderables.append(level_text)

                # 2. Display Aggregated Pitch Class Table
                # This table summarizes activity per PC across all octaves
                pc_table_render = make_pitch_class_table(pitch_data_by_pc)
                renderables.append(pc_table_render)

                # 3. Display Detailed Detected Notes Table
                # This table lists each individual peak detection with octave and metrics
                # notes_table_render = make_detected_notes_table(detailed_peak_detections)
                # renderables.append(notes_table_render)

                # 4. Display Active Pitch Classes (0-11) Summary (Redundant with table but quick glance)
                active_pc_names = [
                    PITCH_CLASS_NAMES[pc] for pc in sorted(list(active_pitch_classes))
                ]
                if active_pc_names:
                    pitches_text = Text(
                        f"Active Pitch Classes (for chord ID): {', '.join(active_pc_names)}"
                    )
                else:
                    pitches_text = Text("[dim]No active pitch classes[/dim]")

                renderables.append(
                    Panel(pitches_text, title="Active PC Summary", expand=False)
                )

                # 5. Display Identified Chord
                if chord:
                    chord_display_text = Text(
                        f"Identified Chord: [bold green]{chord}[/bold green]"
                    )
                else:
                    chord_display_text = Text("Identified Chord: [dim]None[/dim]")

                renderables.append(
                    Panel(chord_display_text, title="Chord Result", expand=False)
                )

                # 6. Combine renderables using Group and update Live
                combined_renderable = Group(*renderables)

                live.update(combined_renderable, refresh=True)

    except KeyboardInterrupt:
        logger.info("\nStopped by user.")
    except RuntimeError as e:
        logger.error(f"Error during startup or streaming: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)
    finally:
        pass  # MicReader stream is stopped in ChordAnalyzer's finally block


if __name__ == "__main__":
    main()
