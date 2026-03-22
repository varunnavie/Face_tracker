"""
utils/helpers.py
Shared utility functions: config loading, UUID generation, frame annotation.
"""

import json
import os
import uuid
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Any


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load and return the JSON config. Raises FileNotFoundError if missing."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def generate_face_id() -> str:
    """Generate a short unique face identifier, e.g. 'face_3a2f1b'."""
    return "face_" + uuid.uuid4().hex[:8]


def annotate_frame(
    frame: np.ndarray,
    tracked_faces: list,          # list of (x1,y1,x2,y2, face_id, is_new)
    unique_count: int,
    fps: float = 0.0
) -> np.ndarray:
    """
    Draw bounding boxes, face IDs, and the unique visitor counter on a frame.
    Returns the annotated frame (modifies in place).
    """
    annotated = frame.copy()

    for item in tracked_faces:
        x1, y1, x2, y2, face_id, is_new = item
        color = (0, 255, 0) if not is_new else (0, 128, 255)   # green=known, orange=new
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = face_id[-12:]   # show last 12 chars to keep it readable
        cv2.putText(annotated, label, (x1, max(y1 - 8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # HUD: unique visitor count + fps
    cv2.rectangle(annotated, (0, 0), (280, 48), (30, 30, 30), -1)
    cv2.putText(annotated, f"Unique Visitors: {unique_count}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if fps > 0:
        cv2.putText(annotated, f"FPS: {fps:.1f}", (8, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return annotated


def ensure_dirs(base: str, *subdirs: str) -> None:
    """Create directories (including subdirs) if they don't exist."""
    for sub in subdirs:
        os.makedirs(os.path.join(base, sub), exist_ok=True)


def timestamp_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
