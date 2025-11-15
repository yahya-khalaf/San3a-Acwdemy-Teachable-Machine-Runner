# core/image_worker.py
from PySide6.QtCore import QThread, Signal
import time
import numpy as np

class ImageInferenceWorker(QThread):
    prediction_ready = Signal(list, str, float)

    def __init__(self, model_wrapper, max_fps: int = 8):
        super().__init__()
        self.model_wrapper = model_wrapper
        self.max_fps = max_fps
        self._running = False
        self._latest_frame = None
        self._lock = False

    def add_frame(self, frame):
        self._latest_frame = frame

    def run(self):
        self._running = True
        interval = 1.0 / max(1, self.max_fps)
        while self._running:
            start = time.time()
            if self._latest_frame is not None and not self._lock:
                self._lock = True
                frame = self._latest_frame
                self._latest_frame = None
                try:
                    arr = self.model_wrapper.preprocess_image(frame)
                    if arr is not None:
                        preds, best = self.model_wrapper.predict(arr)
                        if preds:
                            confidence = max(preds)
                            self.prediction_ready.emit(preds, best, float(confidence))
                except Exception as e:
                    print(f"[image_worker] error: {e}")
                finally:
                    self._lock = False
            elapsed = time.time() - start
            to_sleep = interval - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

    def stop(self):
        self._running = False
        self.wait()
