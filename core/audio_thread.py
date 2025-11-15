# core/audio_thread.py
from PySide6.QtCore import QThread, Signal
import sounddevice as sd
import numpy as np
import time
import queue
import platform
IS_MAC = platform.system() == "Darwin"

class AudioRecorder(QThread):
    audio_ready = Signal(object)  # numpy array (int16)
    error_occurred = Signal(str)

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024, device: int = None):
        super().__init__()
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device = device
        self._running = False
        self._q = queue.Queue()
        if IS_MAC:
            import sounddevice as sd
            sd.default.latency = 'low'
            sd.default.blocksize = self.chunk_size


    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[AudioRecorder] status: {status}")
        # indata is float32 in range [-1,1] by default
        pcm = (indata[:, 0] * 32767).astype('int16')
        self._q.put(pcm)

    def run(self):
        self._running = True
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, blocksize=self.chunk_size,
                                callback=self._callback, device=self.device):
                while self._running:
                    try:
                        pcm = self._q.get(timeout=0.2)
                        # emit collected chunk(s)
                        self.audio_ready.emit(pcm)
                    except Exception:
                        pass
                    # small sleep to stay cooperative
                    time.sleep(0.01)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def stop(self):
        self._running = False
        self.wait()
