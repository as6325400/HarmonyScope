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

logger = logging.getLogger(__name__)


def choose_device_interactive() -> int:
    # ... (unchanged) ...
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


def make_pitch_class_table(pitch_data_by_pc):
    """
    Creates a fixed-row rich Table (1 row per pitch class) displaying aggregated info.

    pitch_data_by_pc: a list of 12 dicts, one for each pitch class (0-11), with aggregated info.
      Each dict includes: {'pc', 'name', 'detection_count', 'total_contributions',
                           'min_required_frames', 'total_voiced_frames', 'active',
                           'avg_prominence_db', 'avg_peak_level_db', 'avg_level_diff_db'}
    """
    # Define columns for the fixed pitch class table
    columns = [
        {"header": "Pitch Class", "justify": "center"},  # C, C#, etc.
        {
            "header": "Frames Detected",
            "justify": "center",
        },  # Count of frames this PC was in
        {"header": "Required", "justify": "center"},  # Minimum frames required
        {
            "header": "Total Voiced",
            "justify": "center",
        },  # Total voiced frames in window
        {
            "header": "Avg Prom (dB)",
            "justify": "center",
        },  # Average prominence across detected peaks
        {
            "header": "Avg Level (dB)",
            "justify": "center",
        },  # Average peak level across detected peaks
        {
            "header": "Avg Diff (dB)",
            "justify": "center",
        },  # Average level diff across detected peaks
        {
            "header": "Active",
            "justify": "center",
        },  # Whether this PC is considered "active"
    ]

    table = Table(title="Pitch Class Activity (Across Octaves)")

    # Add columns
    for col in columns:
        table.add_column(**col)

    # The pitch_data_by_pc list should always have 12 entries, one for each PC
    # It's sorted by PC index (0-11) in pitch.py
    for pc_info in pitch_data_by_pc:
        name = pc_info.get("name", "N/A")
        frame_count = pc_info.get("detection_count", 0)
        required = pc_info.get("min_required_frames", 0)
        total = pc_info.get("total_voiced_frames", 0)
        active = "✔" if pc_info.get("active", False) else ""
        prominence_avg = pc_info.get("avg_prominence_db", -np.inf)
        peak_level_avg = pc_info.get("avg_peak_level_db", -np.inf)
        level_diff_avg = pc_info.get(
            "avg_level_diff_db", np.inf
        )  # Use +inf as default diff

        # Handle -inf and +inf for display when no detections occurred
        prominence_display = (
            f"{prominence_avg:.1f}" if np.isfinite(prominence_avg) else "--"
        )
        peak_level_display = (
            f"{peak_level_avg:.1f}" if np.isfinite(peak_level_avg) else "--"
        )
        level_diff_display = (
            f"{level_diff_avg:.1f}" if np.isfinite(level_diff_avg) else "--"
        )

        table.add_row(
            name,
            str(frame_count),
            str(required),
            str(total),
            prominence_display,
            peak_level_display,
            level_diff_display,
            active,
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
        logger.info(f"Using device ID: {dev_id}, Sample Rate: 44100")
        reader = MicReader(device=dev_id, sr=44100)
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
            # stream_mic_live yields chord, active_pitch_classes, pitch_data_by_pc, segment_rms_db, total_voiced_frames
            for (
                chord,
                active_pitch_classes,
                pitch_data_by_pc,
                segment_rms_db,
                total_voiced_frames,
            ) in ana.stream_mic_live(interval_sec=args.interval):

                renderables = []  # List to hold rich renderables

                # 1. Display Overall Level and Threshold
                level_text = Text(
                    f"Overall Segment Level: {segment_rms_db:.1f} dB | Voiced Threshold: {args.frame_energy_thresh_db:.1f} dB | Voiced Frames in Window: {total_voiced_frames}"
                )  # Include total voiced frames here
                renderables.append(level_text)

                # 2. Display Fixed Pitch Class Table
                pc_table_render = make_pitch_class_table(pitch_data_by_pc)
                renderables.append(pc_table_render)

                # 3. Display Active Pitch Classes (0-11) Summary
                # This is redundant with the table now, but might be a cleaner list view
                active_pc_names = [
                    PITCH_CLASS_NAMES[pc] for pc in sorted(list(active_pitch_classes))
                ]
                if active_pc_names:
                    pitches_text = Text(
                        f"Active Pitch Classes (from table): {', '.join(active_pc_names)}"
                    )
                else:
                    pitches_text = Text("[dim]No active pitch classes[/dim]")

                renderables.append(
                    Panel(pitches_text, title="Active Summary", expand=False)
                )  # Use Panel for structure

                # 4. Display Identified Chord
                # Fix formatting: apply [bold green] only if chord is not None
                if chord:
                    chord_display_text = Text(
                        f"Identified Chord: [bold green]{chord}[/bold green]"
                    )
                else:
                    chord_display_text = Text("Identified Chord: [dim]None[/dim]")

                renderables.append(
                    Panel(chord_display_text, title="Chord Result", expand=False)
                )  # Use Panel

                # 5. Combine renderables using Group and update Live
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
