import librosa
import numpy as np

CHORD_PATTERNS = {
    '': [0, 4, 7],         # major
    'm': [0, 3, 7],        # minor
    'dim': [0, 3, 6]       # diminished
}

def build_chord_templates():
    templates = {}
    for root_idx, root in enumerate(['C', 'C#', 'D', 'D#', 'E', 'F',
                                     'F#', 'G', 'G#', 'A', 'A#', 'B']):
        for quality, intervals in CHORD_PATTERNS.items():
            vec = np.zeros(12)
            for i in intervals:
                vec[(root_idx + i) % 12] = 1
            templates[f"{root}{quality}"] = vec
    return templates

def analyze_chord(wav_path):
    y, sr = librosa.load(wav_path)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    templates = build_chord_templates()
    best_match = None
    best_score = -1

    for chord_name, template in templates.items():
        score = np.dot(chroma_mean, template) / (np.linalg.norm(chroma_mean) * np.linalg.norm(template))
        if score > best_score:
            best_score = score
            best_match = chord_name

    return best_match

if __name__ == "__main__":
    import os
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    for filename in os.listdir(data_dir):
        if filename.endswith(".wav"):
            path = os.path.join(data_dir, filename)
            predicted = analyze_chord(path)
            print(f"{filename}: predicted → {predicted}")
