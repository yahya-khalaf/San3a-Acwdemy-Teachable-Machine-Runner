# core/model_wrapper.py
import os
import traceback
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
from PySide6.QtCore import QMutex, QMutexLocker

# tflite runtime import logic
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
    print("[model_wrapper] using tflite_runtime")
except Exception:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        print("[model_wrapper] using tensorflow.lite")
    except Exception:
        Interpreter = None
        print("[model_wrapper] TensorFlow Lite not available; install tflite-runtime or tensorflow.")

from .utils import load_metadata, ensure_tuple

@dataclass
class ModelInfo:
    model_path: str
    labels: List[str]
    model_type: str  # "image" or "audio"
    input_shape: Tuple[int, ...]
    normalization: dict
    metadata: dict

class ModelWrapper:
    def __init__(self):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.model_info: Optional[ModelInfo] = None
        self.mutex = QMutex()

    def load_model(self, model_path: str, labels_path: Optional[str] = None) -> bool:
        """Load tflite model and parse metadata.json (Teachable Machine export)."""
        try:
            with QMutexLocker(self.mutex):
                if Interpreter is None:
                    raise ImportError("No TFLite Interpreter available")

                # instantiate interpreter
                self.interpreter = Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()

                # labels
                labels = self._load_labels(labels_path, model_path)

                # metadata.json
                metadata = load_metadata(model_path)

                # detect model type and input shape
                input_shape = tuple(self.input_details[0]["shape"])
                model_type = self._detect_model_type(input_shape, metadata)

                # normalization rules (from metadata if present)
                normalization = self._parse_normalization_from_metadata(metadata)

                self.model_info = ModelInfo(
                    model_path=str(Path(model_path).resolve()),
                    labels=labels,
                    model_type=model_type,
                    input_shape=input_shape,
                    normalization=normalization,
                    metadata=metadata
                )
                print(f"[model_wrapper] Loaded model {model_path} type={model_type} input_shape={input_shape}")
                return True
        except Exception as e:
            print(f"[model_wrapper] Error loading model: {e}")
            traceback.print_exc()
            return False

    def _load_labels(self, labels_path: Optional[str], model_path: str) -> List[str]:
        if labels_path and os.path.exists(labels_path):
            try:
                with open(labels_path, "r", encoding="utf-8") as f:
                    return [l.strip() for l in f.readlines() if l.strip()]
            except Exception as e:
                print(f"[model_wrapper] label load error: {e}")

        # fallback to labels.txt in same dir
        p = Path(model_path).resolve().parent / "labels.txt"
        if p.exists():
            try:
                return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
            except Exception as e:
                print(f"[model_wrapper] label load error: {e}")

        # final fallback: use output shape to generate labels
        if self.output_details:
            shape = self.output_details[0]["shape"]
            num = int(shape[-1]) if len(shape) > 1 else int(shape[0])
            return [f"Class_{i}" for i in range(num)]

        return ["Unknown"]

    def _detect_model_type(self, input_shape: Tuple[int, ...], metadata: dict) -> str:
        # Prefer explicit metadata if present
        try:
            if metadata:
                # Teachable Machine metadata may have "modelType" or "convertedBy"
                if "modelType" in metadata:
                    mt = metadata["modelType"].lower()
                    if "audio" in mt:
                        return "audio"
                    if "image" in mt:
                        return "image"
                # Some exports include an "audio" key
                if "audio" in metadata:
                    return "audio"
                if "image" in metadata or "vision" in metadata:
                    return "image"
        except Exception:
            pass

        # Heuristic fallback
        if len(input_shape) == 4:
            return "image"
        elif len(input_shape) == 2:
            return "audio"
        elif len(input_shape) == 3:
            # ambiguous: could be image (batch, height, width) or audio (batch, time, features)
            # assume audio if time dimension is large
            if input_shape[1] > 1000:
                return "audio"
            return "image"
        return "unknown"

    def _parse_normalization_from_metadata(self, metadata: dict) -> dict:
        # Return a dict describing how to normalize images: {"scale": ..., "offset": ...}
        # Teachable Machine often uses 0-255 -> 0-1 or -1..1
        if not metadata:
            return {"mode": "auto"}  # leave to default behavior

        try:
            # common structure in TM exports:
            # metadata["modelInfo"]["preprocessing"] or metadata["image"] ...
            if "image" in metadata:
                img = metadata["image"]
                return img.get("preprocessing", {"mode": "0-1"})
            if "preprocessing" in metadata:
                return metadata["preprocessing"]
            if "modelInfo" in metadata and "preprocessing" in metadata["modelInfo"]:
                return metadata["modelInfo"]["preprocessing"]
        except Exception:
            pass
        return {"mode": "auto"}

    def predict(self, input_array: np.ndarray) -> Tuple[List[float], str]:
        """Run inference on preprocessed array.
        input_array must be already preprocessed and shaped according to model_info.input_shape without batch axis.
        """
        try:
            with QMutexLocker(self.mutex):
                if not self.interpreter or not self.model_info:
                    return [], "No model"

                # ensure batch dim
                inp = np.expand_dims(input_array, axis=0).astype(np.float32)

                # If interpreter expects a specific dtype, cast
                target_dtype = self.input_details[0].get("dtype", np.float32)
                if inp.dtype != target_dtype:
                    try:
                        inp = inp.astype(target_dtype)
                    except Exception:
                        inp = inp.astype(np.float32)

                self.interpreter.set_tensor(self.input_details[0]['index'], inp)
                self.interpreter.invoke()
                out = self.interpreter.get_tensor(self.output_details[0]['index'])
                preds = out[0].tolist()
                best_idx = int(np.argmax(preds))
                best_label = self.model_info.labels[best_idx] if best_idx < len(self.model_info.labels) else "Unknown"
                return preds, best_label
        except Exception as e:
            print(f"[model_wrapper] Prediction error: {e}")
            traceback.print_exc()
            return [], str(e)

    # High-level helpers for image/audio preprocessing (used by worker modules)
    def preprocess_image(self, frame_bgr) -> Optional[np.ndarray]:
        """
        Convert BGR frame (OpenCV) to model input shape and normalization.
        Returns an array shaped as model input without batch dimension.
        """
        import cv2
        if not self.model_info:
            return None
        # convert to RGB
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        _, h, w = None, None, None
        shape = self.model_info.input_shape  # e.g., (1, height, width, channels)
        if len(shape) == 4:
            _, h, w, c = shape
        elif len(shape) == 3:
            h, w, c = shape
        else:
            # fallback
            h, w, c = 224, 224, 3

        img_resized = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)

        arr = img_resized.astype(np.float32)

        # normalization
        norm = self.model_info.normalization
        mode = norm.get("mode", "auto") if isinstance(norm, dict) else norm

        if mode in ("0-1", "0_1") or mode == "auto":
            arr = arr / 255.0
        elif mode in ("-1-1", "-1_1"):
            arr = (arr / 127.5) - 1.0
        else:
            arr = arr  # assume model expects 0-255

        # If model expects single channel, convert
        if int(arr.shape[2]) != int(shape[-1]):
            if int(shape[-1]) == 1:
                arr = np.mean(arr, axis=2, keepdims=True)
            else:
                # Expand or trim channels
                if arr.shape[2] == 1 and shape[-1] == 3:
                    arr = np.repeat(arr, 3, axis=2)

        return arr

    def preprocess_audio(self, pcm_int16: np.ndarray) -> Optional[np.ndarray]:
        """
        Convert raw PCM (int16) 1D array to model input (e.g., MFCC or spectrogram).
        This function tries to detect audio preprocessing rules in metadata; otherwise uses MFCC defaults.
        Returns an array shaped as model input without batch dimension.
        """
        # Lazy import to keep module light
        try:
            from python_speech_features import mfcc
            have_psf = True
        except Exception:
            have_psf = False

        try:
            import scipy.signal as sig
        except Exception:
            sig = None

        if not self.model_info:
            return None

        metadata = self.model_info.metadata or {}
        sample_rate = 16000
        if "audio" in metadata:
            sample_rate = int(metadata["audio"].get("sampleRate", sample_rate))

        # convert to float32 in range [-1,1]
        audio = pcm_int16.astype(np.float32) / 32768.0

        # If metadata provides spectrogram or feature settings, try to use them
        # Fallback: compute MFCC with python_speech_features
        if have_psf:
            # choose parameters but allow overrides
            winlen = 0.025
            winstep = 0.010
            numcep = 13
            nfilt = 26
            nfft = 512

            # custom overrides from metadata (if present)
            if "audio" in metadata:
                audio_meta = metadata["audio"]
                winlen = float(audio_meta.get("windowSizeSeconds", winlen))
                winstep = float(audio_meta.get("hopSizeSeconds", winstep))
                numcep = int(audio_meta.get("numCepstra", numcep))
                nfilt = int(audio_meta.get("numMelBins", nfilt))
                nfft = int(audio_meta.get("fftSize", nfft))

            # Ensure correct length: python_speech_features.mfcc expects 1D array
            try:
                mfcc_feat = mfcc(audio, samplerate=sample_rate, winlen=winlen, winstep=winstep,
                                 numcep=numcep, nfilt=nfilt, nfft=nfft)
                # Many TM models expect shape (time, features). Align to model input shape if possible
                input_shape = self.model_info.input_shape
                # Remove batch dim if present
                target_shape = input_shape[1:] if len(input_shape) > 1 else input_shape
                mfcc_arr = np.array(mfcc_feat, dtype=np.float32)

                # If shape mismatch, pad or truncate in time dimension (axis 0)
                if len(target_shape) == 2:
                    t_target, f_target = int(target_shape[0]), int(target_shape[1])
                    t_cur, f_cur = mfcc_arr.shape
                    # Trim/pad features axis if needed
                    if f_cur != f_target:
                        if f_cur > f_target:
                            mfcc_arr = mfcc_arr[:, :f_target]
                        else:
                            pad_width = ((0, 0), (0, f_target - f_cur))
                            mfcc_arr = np.pad(mfcc_arr, pad_width, mode="constant")
                    # Trim/pad time axis
                    if t_cur > t_target:
                        mfcc_arr = mfcc_arr[:t_target, :]
                    elif t_cur < t_target:
                        pad_width = ((0, t_target - t_cur), (0, 0))
                        mfcc_arr = np.pad(mfcc_arr, pad_width, mode="constant")
                    return mfcc_arr
                else:
                    # If model expects flat vector, flatten
                    return mfcc_arr.flatten()
            except Exception as e:
                print(f"[model_wrapper] MFCC compute error: {e}")

        # fallback: compute simple log-mel patch via numpy+scipy (if available)
        if sig is not None:
            try:
                # compute spectrogram
                nfft = 512
                win = int(0.025 * sample_rate)
                hop = int(0.010 * sample_rate)
                f, t, Sxx = sig.spectrogram(audio, fs=sample_rate, window="hann",
                                            nperseg=win, noverlap=(win - hop), nfft=nfft, scaling="spectrum")
                # convert power to log
                Slog = np.log1p(Sxx)
                # Resize/pad/truncate to model shape
                input_shape = self.model_info.input_shape
                target_shape = input_shape[1:] if len(input_shape) > 1 else input_shape
                arr = Slog.astype(np.float32)
                # If arr has shape (freq, time), transpose if needed
                if arr.shape != tuple(target_shape):
                    # attempt simple resize using numpy (reshape may break semantics)
                    try:
                        arr = np.resize(arr, tuple(target_shape)).astype(np.float32)
                    except Exception:
                        pass
                return arr
            except Exception as e:
                print(f"[model_wrapper] spectrogram fallback error: {e}")

        # final fallback: return raw waveform normalized and truncated/padded to match model input length
        try:
            target_len = 1
            input_shape = self.model_info.input_shape
            if len(input_shape) > 1:
                target_len = int(np.prod(input_shape[1:]))
            pcm = audio.flatten()
            if pcm.size > target_len:
                pcm = pcm[:target_len]
            elif pcm.size < target_len:
                pcm = np.pad(pcm, (0, target_len - pcm.size))
            return pcm.astype(np.float32)
        except Exception as e:
            print(f"[model_wrapper] final audio fallback error: {e}")
            return None
