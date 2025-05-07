from HarmonyScope.io.file_reader import FileReader
from HarmonyScope.analyzer.chord_analyzer import ChordAnalyzer
from pathlib import Path
import time
import argparse
from HarmonyScope import set_verbosity

def main():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="count", default=0,
                    help="-v Display DEBUG")
    args = ap.parse_args()

    set_verbosity(args.verbose)
    
    start = time.time()
    data_dir = Path(__file__).resolve().parent.parent / "data"

    ana = ChordAnalyzer(reader=FileReader())

    for wav_path in sorted(data_dir.glob("*.wav")):
        result = ana.analyze_file(str(wav_path))
        print(f"{wav_path.name:<20} → {result}")

    print(f"\nTotal time: {time.time() - start:.2f}s")
    


if __name__ == "__main__":
    main()
