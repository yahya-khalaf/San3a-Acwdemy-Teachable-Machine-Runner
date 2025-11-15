# core/audio_worker.py
from PySide6.QtCore import QThread, Signal
import time
import numpy as np

class AudioInferenceWorker(QThread):
    prediction_ready = Signal(list, str, float)

    def __init__(self, model_wrapper, frames_to_accumulate: int = 4):
        """
        frames_to_accumulate: number of incoming audio chunks to accumulate before running inference.
        This reduces spurious predictions and matches expected model input length.
        """
        super().__init__()
        self.model_wrapper = model_wrapper
        self._running = False
        self._buffer = []
        self._accumulate = frames_to_accumulate

    def add_audio(self, pcm_chunk):
        self._buffer.append(pcm_chunk)

    def run(self):
        self._running = True
        while self._running:
            if len(self._buffer) >= self._accumulate:
                # concatenate earliest chunks
                chunks = self._buffer[:self._accumulate]
                self._buffer = self._buffer[self._accumulate:]
                try:
                    audio = np.concatenate(chunks).astype('int16')
                    arr = self.model_wrapper.preprocess_audio(audio)
                    if arr is not None:
                        preds, best = self.model_wrapper.predict(arr)
                        if preds:
                            confidence = max(preds)
                            self.prediction_ready.emit(preds, best, float(confidence))
                except Exception as e:
                    print(f"[audio_worker] error: {e}")
            else:
                time.sleep(0.02)

    def stop(self):
        self._running = False
        self.wait()
