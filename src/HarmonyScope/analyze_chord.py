import librosa
import numpy as np
import os
import time

CHORD_RELATIONS = [
    ("",      {0, 4, 7}),          # major
    ("m",     {0, 3, 7}),          # minor
    ("sus4",  {0, 5, 7}),          # sus4
    ("sus2",  {0, 2, 7}),          # sus2
    ("dim",   {0, 3, 6}),          # diminished
    ("maj7",  {0, 4, 7, 11}),      # major 7
    ("m7",    {0, 3, 7, 10}),      # minor 7
    ("7",     {0, 4, 7, 10}),      # dominant 7
    ("6",     {0, 4, 7, 9}),        # major 6
    ("m6",    {0, 3, 7, 9}),        # minor 6
    ("add9",  {0, 2, 4, 7}),        # add 9
    ("madd9", {0, 2, 3, 7}),        # minor add9
    ("dim7",  {0, 3, 6, 9}),        # diminished 7
    ("m7b5",  {0, 3, 6, 10}),       # half‑dim (ø) / minor 7♭5
    ("9",     {0, 4, 7, 10, 14}),   # dominant 9
    ("maj9",  {0, 4, 7, 11, 14}),   # major 9
    ("11",    {0, 4, 7, 10, 14, 17}),# dominant 11
    ("13",    {0, 4, 7, 10, 14, 21}) # dominant 13
]

PITCH_CLASS_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
                     'F#', 'G', 'G#', 'A', 'A#', 'B']


def active_pitches(
        wav_path: str,
        frame_energy_thresh_db: float = -40,   # Determine whether a frame is “voiced”
        delta_db: float = 2,                  # Treat a pitch as active if its energy lies within Δ dB of the peak
        debug: bool = False):

    y, sr = librosa.load(wav_path, sr=None)

    rms = librosa.feature.rms(y=y)[0]                # shape=(N_frames,)
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    voiced_frames = rms_db > frame_energy_thresh_db
    if voiced_frames.sum() == 0:
        return set()

    chroma = librosa.feature.chroma_cqt(
                y=y, sr=sr, n_chroma=12)[:, voiced_frames]

    # For each pitch‑class row, take its maximum energy; any row whose value is within Δ dB of the peak is marked as active.
    pc_energy = librosa.amplitude_to_db(chroma, ref=np.max).max(axis=1)
    peak_db   = pc_energy.max()
    active_idx = np.where(pc_energy >= peak_db - delta_db)[0]
    active_set = set(active_idx)

    if debug:
        print(f"\n>>> {os.path.basename(wav_path)}  (frames={len(voiced_frames)}, voiced={voiced_frames.sum()})")
        for i, n in enumerate(PITCH_CLASS_NAMES):
            flag = "✔" if i in active_set else " "
            diff = peak_db - pc_energy[i]
            print(f"{n:2}: {pc_energy[i]:6.1f} dB  (Δ={diff:4.1f}) {flag}")
    return active_set



def identify_chord(pitches: set[int]) -> str | None:
    if not pitches:
        return None

    best: tuple[int, str] | None = None   # (complexity, "Gm", …)

    for root_pc in pitches:
        intervals = {(pc - root_pc) % 12 for pc in pitches}
        intervals.add(0)    

        for suffix, relation in CHORD_RELATIONS:
            if intervals.issubset(relation):
                complexity = len(relation)
                name = f"{PITCH_CLASS_NAMES[root_pc]}{suffix}"
                if best is None or complexity < best[0]:
                    best = (complexity, name)
                break

    return best[1] if best else None

def analyze_chord(wav_path):
    pcs = active_pitches(wav_path)
    return identify_chord(pcs)

def main():
    start = time.time()
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    for fn in sorted(os.listdir(data_dir)):
        if fn.endswith(".wav"):
            path = os.path.join(data_dir, fn)
            pred = analyze_chord(path)
            print(f"{fn:<20} → {pred}")

    print(f"\nTotal time: {time.time() - start:.2f} s")


if __name__ == "__main__":
    main()
