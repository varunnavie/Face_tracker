"""
detector/face_detector.py
YOLOv8-based face detector with ByteTrack tracking.

Primary:  downloads yolov8n-face.pt (dedicated face detector).
Fallback: uses standard yolov8n.pt filtered to the 'person' class if all
          face-model mirrors are unavailable. Person bounding boxes are
          slightly larger than tight face boxes; InsightFace alignment
          compensates for this downstream.
"""

import os
import urllib.request
import numpy as np
from typing import List, Tuple, Optional
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)

# Community mirrors for yolov8n-face.pt (tried in order)
_FACE_MODEL_URLS = [
    "https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt",
    "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt",
    "https://github.com/derronqi/yolov8-face/releases/download/v8/yolov8n-face.pt",
]


class FaceDetector:
    def __init__(self, model_path: str, confidence: float = 0.45, iou: float = 0.45):
        self.confidence = confidence
        self.iou = iou
        self.person_mode = False   # True when using standard yolov8n (person class)

        resolved_path = self._resolve_model(model_path)
        self.model = YOLO(resolved_path)
        # Warm-up pass
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False)

    # ------------------------------------------------------------------
    # Internal: model resolution
    # ------------------------------------------------------------------

    def _resolve_model(self, model_path: str) -> str:
        """
        Return a path to a loadable YOLO model.
        1. If model_path already exists, use it.
        2. Try downloading face-specific weights to model_path.
        3. If all downloads fail, use ultralytics' auto-downloaded yolov8n.pt.
        """
        if os.path.exists(model_path):
            return model_path

        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

        for url in _FACE_MODEL_URLS:
            try:
                logger.info("[FaceDetector] Downloading from %s", url)
                print(f"[FaceDetector] Downloading yolov8n-face.pt from {url} ...")
                urllib.request.urlretrieve(url, model_path)
                print("[FaceDetector] Download complete.")
                return model_path
            except Exception as e:
                logger.warning("[FaceDetector] Mirror failed: %s", e)

        # All face-model mirrors failed → fall back to standard yolov8n.pt
        print("[FaceDetector] Face model unavailable. Falling back to yolov8n.pt (person mode).")
        logger.warning("[FaceDetector] Falling back to yolov8n.pt — person detection mode.")
        self.person_mode = True
        return "yolov8n.pt"   # ultralytics auto-downloads on first YOLO() call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Run single-frame detection (no tracking).
        Returns list of (x1, y1, x2, y2, confidence).
        """
        kwargs = dict(conf=self.confidence, iou=self.iou, verbose=False)
        if self.person_mode:
            kwargs["classes"] = [0]   # person class only

        results = self.model.predict(frame, **kwargs)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                detections.append((x1, y1, x2, y2, conf))
        return detections

    def track(
        self, frame: np.ndarray, persist: bool = True
    ) -> List[Tuple[int, int, int, int, float, int]]:
        """
        Run detection + ByteTrack on a single frame.
        Returns list of (x1, y1, x2, y2, confidence, track_id).
        track_id is -1 when ByteTrack has not yet assigned an ID.
        """
        kwargs = dict(
            conf=self.confidence,
            iou=self.iou,
            persist=persist,
            tracker="bytetrack.yaml",
            verbose=False
        )
        if self.person_mode:
            kwargs["classes"] = [0]

        results = self.model.track(frame, **kwargs)
        tracked = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                tid = int(box.id[0]) if box.id is not None else -1
                tracked.append((x1, y1, x2, y2, conf, tid))
        return tracked

    @staticmethod
    def crop_face(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                  padding: int = 10) -> Optional[np.ndarray]:
        """Crop a face region with optional padding. Returns None if region is invalid."""
        h, w = frame.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()
