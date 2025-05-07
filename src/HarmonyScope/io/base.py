from typing import Protocol, runtime_checkable, Tuple
import numpy as np

@runtime_checkable
class AudioReader(Protocol):
    """
    Any object that is 'callable' and returns ``(y, sr)`` is considered an AudioReader.
    - ``y``: 1-D numpy array, audio waveform (float32, -1~1)
    - ``sr``: sampling rate (Hz)
    """
    # *args / **kwargs allow different sources to customize parameters: filename, seconds, device ID...
    def __call__(self, *args, **kwargs) -> Tuple[np.ndarray, int]: ...