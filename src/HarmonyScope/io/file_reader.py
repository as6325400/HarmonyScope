from pathlib import Path
import librosa
import numpy as np
from typing import Tuple

class FileReader:
    def __init__(self, sr=None):
        self.sr = sr
    def __call__(self, path: str | Path) -> Tuple[np.ndarray, int]:
        y, sr = librosa.load(path, sr=self.sr)
        return y, sr
