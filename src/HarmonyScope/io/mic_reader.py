import sounddevice as sd
import numpy as np
from typing import Tuple
from collections import deque
import threading

# --------  List devices -----------------------------------------------------
def list_input_devices() -> list[tuple[int, str]]:
    """
    Returns [(id, name), ...] only for devices that support **recording**.
    """
    devices = []
    for idx, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] > 0:
            devices.append((idx, info["name"]))
    return devices

class MicReader:
    """An object reader that conforms to AudioReader and starts streaming automatically."""
    
    def __init__(self,
                 device: int | None = None,   # Device ID; None = default
                 sr: int = 44100,            # Sampling rate
                 maxlen_sec: float = 10.0):  # Max buffer size (in seconds)
        self.device = device
        self.sr = sr
        self.maxlen_frames = int(maxlen_sec * sr)  # 計算 buffer 的 frame 數
        self.buffer = deque(maxlen=self.maxlen_frames)  # buffer 用 deque 固定長度
        self.lock = threading.Lock()  # 線程鎖
        self.stream = None            # 放 InputStream

        # 🚀 立刻啟動 stream！
        self._start_stream()

    def _start_stream(self):
        """
        Internal method: start a background stream recording.
        """
        if self.stream is not None:
            raise RuntimeError("Stream already running")

        self.stream = sd.InputStream(
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._callback
        )
        self.stream.start()  # 立刻開啟 stream

    def _callback(self, indata, frames, time_info, status):
        """
        Stream callback: 每次有新資料到時會被呼叫。
        """
        data = indata[:, 0]  # 取得單聲道資料
        with self.lock:
            self.buffer.extend(data)  # 新資料推進 buffer

    def stop(self):
        """
        Stop the stream manually if needed.
        """
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get_buffer(self) -> np.ndarray:
        """
        Return the current buffer as a numpy array.
        """
        with self.lock:
            return np.array(self.buffer)

    def __call__(self, *args, **kwargs) -> Tuple[np.ndarray, int]:
        """
        Instead of recording, return the latest buffer window.
        The window size can be passed via kwargs['win_sec'].
        """
        win_sec = kwargs.get("win_sec", 1)  # 預設抓 1 秒資料
        num_frames = int(win_sec * self.sr)

        with self.lock:
            if len(self.buffer) < num_frames:
                # 資料太少，回傳目前 buffer
                y = np.array(self.buffer)
            else:
                # 回傳最後 win_sec 秒
                y = np.array(self.buffer)[-num_frames:]
        
        return y, self.sr