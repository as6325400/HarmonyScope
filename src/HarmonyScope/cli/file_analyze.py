from HarmonyScope.io.file_reader import FileReader
from HarmonyScope.analyzer.chord_analyzer import ChordAnalyzer
from pathlib import Path
import time
import argparse
from HarmonyScope import set_verbosity
from HarmonyScope.cli.common_args import add_common_args


def main():

    ap = argparse.ArgumentParser(
        prog="harmonyscope-file",
        description="📂  Offline audio-file chord analyzer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(ap)
    args = ap.parse_args()

    set_verbosity(args.verbose)

    start = time.time()
    data_dir = Path(__file__).resolve().parent.parent / "data"

    ana = ChordAnalyzer(
        reader=FileReader(),
        win_sec=args.window,
        min_frame_ratio=args.min_frame_ratio,
        min_prominence_db=args.min_prominence_db,
        max_level_diff_db=args.max_level_diff_db,
        frame_energy_thresh_db=args.frame_energy_thresh_db,
    )

    for wav_path in sorted(data_dir.glob("*.wav")):
        result = ana.analyze_file(str(wav_path))
        print(f"{wav_path.name:<20} → {result}")

    print(f"\nTotal time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
