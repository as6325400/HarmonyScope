from HarmonyScope.io.mic_reader import list_input_devices, MicReader
from HarmonyScope.analyzer.chord_analyzer import ChordAnalyzer
import argparse, sys
import questionary
from questionary import Choice
from HarmonyScope import set_verbosity
from HarmonyScope.ui.rich_live import LiveMicUI, live_mic_loop
import logging
from HarmonyScope.core.constants import PITCH_CLASS_NAMES

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

        ui = LiveMicUI()
        live_mic_loop(analyzer=ana, ui=ui, interval_sec=args.interval)

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
