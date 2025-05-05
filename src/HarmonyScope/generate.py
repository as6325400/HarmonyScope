import numpy as np
from scipy.io.wavfile import write
import os

SAMPLE_RATE = 44100  # Hz
DURATION = 2         # seconds

NOTE_FREQUENCIES = {
    'C': 261.63,
    'C#': 277.18,
    'D': 293.66,
    'D#': 311.13,
    'E': 329.63,
    'F': 349.23,
    'F#': 369.99,
    'G': 392.00,
    'G#': 415.30,
    'A': 440.00,
    'A#': 466.16,
    'B': 493.88
}

CHORD_PATTERNS = {
    '': [0, 4, 7],         # major
    'm': [0, 3, 7],        # minor
    'dim': [0, 3, 6]       # diminished
}

def generate_chord_wave(chord_name):
    if len(chord_name) >= 2 and chord_name[1] == '#':
        root_note = chord_name[:2]
        quality = chord_name[2:]
    else:
        root_note = chord_name[0]
        quality = chord_name[1:]
    
    freqs = [NOTE_FREQUENCIES[root_note] * (2 ** (semitone / 12)) for semitone in CHORD_PATTERNS[quality]]
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
    wave = sum(np.sin(2 * np.pi * freq * t) for freq in freqs)
    wave = wave / np.max(np.abs(wave))  # normalize to [-1, 1]
    audio = np.int16(wave * 32767)
    return audio

def create_wav_files(chords, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for chord in chords:
        audio = generate_chord_wave(chord)
        file_path = os.path.join(output_dir, f"{chord}.wav")
        write(file_path, SAMPLE_RATE, audio)

if __name__ == "__main__":
    chords = ["C", "Am", "F", "G"]
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    from pathlib import Path
    Path(output_dir).mkdir(exist_ok=True)

    create_wav_files(chords, output_dir)
    print(f"Generated .wav files in {output_dir}")
