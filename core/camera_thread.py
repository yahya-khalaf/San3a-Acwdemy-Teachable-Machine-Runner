# core/camera_thread.py
from PySide6.QtCore import QThread, Signal
import cv2
import time
import platform
IS_MAC = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"

PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 480

class CameraThread(QThread):
    frame_ready = Signal(object)  # numpy.ndarray
    error_occurred = Signal(str)

    def __init__(self, camera_index: int = 0, fps: int = 30):
        super().__init__()
        self.camera_index = camera_index
        self.fps = fps
        self._running = False
        self.cap = None

    def run(self):
        self._running = True
        try:
            # Try default backends; OpenCV picks appropriate backend itself
            if IS_MAC:
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)
            elif IS_WINDOWS:
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.camera_index)  # Linux fallback
            

            if not self.cap.isOpened():
                self.error_occurred.emit(f"Cannot open camera index {self.camera_index}")
                return

            # set properties as best effort
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            interval = 1.0 / max(1, self.fps)
            while self._running:
                ret, frame = self.cap.read()
                if not ret:
                    self.error_occurred.emit("Camera read failed")
                    break
                self.frame_ready.emit(frame)
                time.sleep(interval)
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            if self.cap and self.cap.isOpened():
                self.cap.release()

    def stop(self):
        self._running = False
        self.wait()