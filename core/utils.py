# core/utils.py
from pathlib import Path
import json
import numpy as np

def load_metadata(model_path: str) -> dict:
    """
    Load metadata.json if present in same folder as model_path.
    Returns empty dict if not found or invalid.
    """
    try:
        p = Path(model_path).resolve().parent / "metadata.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[utils] metadata load error: {e}")
    return {}

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def ensure_tuple(shape):
    try:
        return tuple(int(x) for x in shape)
    except Exception:
        return tuple(shape)
