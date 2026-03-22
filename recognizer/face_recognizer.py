"""
recognizer/face_recognizer.py
Generates 512-d ArcFace embeddings via InsightFace (buffalo_l model).
Handles face matching using cosine similarity.
"""

import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors."""
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


class FaceRecognizer:
    def __init__(self, model_name: str = "buffalo_l", ctx_id: int = 0):
        """
        Args:
            model_name : InsightFace model pack name ('buffalo_l', 'buffalo_s', etc.)
            ctx_id     : 0 = GPU, -1 = CPU
        """
        # Try GPU first, fall back to CPU
        self.app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"]
        )
        try:
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            logger.info("[FaceRecognizer] InsightFace running on GPU (ctx_id=%d)", ctx_id)
        except Exception:
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
            logger.info("[FaceRecognizer] InsightFace running on CPU")

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def get_embedding_from_crop(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a 512-d embedding from a pre-cropped face image (BGR).
        InsightFace internally runs its own detector on the crop; we take the
        largest detected face embedding.

        Returns None if no face is found in the crop.
        """
        if face_crop is None or face_crop.size == 0:
            return None

        # Resize crop to a reasonable size for InsightFace
        h, w = face_crop.shape[:2]
        scale = max(112 / h, 112 / w)
        if scale > 1.0:
            face_crop = cv2.resize(face_crop, (int(w * scale), int(h * scale)))

        faces = self.app.get(face_crop)
        if not faces:
            return None

        # Pick the face with the largest bounding-box area
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = largest.embedding
        if emb is None:
            return None
        return emb.astype(np.float32)

    def get_embedding_from_frame(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Extract embedding directly from the full frame + bounding box.
        Preferred over the crop variant because alignment is more accurate.
        """
        faces = self.app.get(frame)
        if not faces:
            return None

        x1, y1, x2, y2 = bbox
        best_face = None
        best_iou = 0.0
        for face in faces:
            fx1, fy1, fx2, fy2 = map(int, face.bbox)
            iou = self._bbox_iou((x1, y1, x2, y2), (fx1, fy1, fx2, fy2))
            if iou > best_iou:
                best_iou = iou
                best_face = face

        if best_face is None or best_iou < 0.2:
            return None
        emb = best_face.embedding
        return emb.astype(np.float32) if emb is not None else None

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def find_best_match(
        self,
        query_embedding: np.ndarray,
        registered_embeddings: List[Tuple[str, np.ndarray]],
        threshold: float = 0.45
    ) -> Optional[str]:
        """
        Compare query_embedding against all registered embeddings.

        Returns the face_id with the highest cosine similarity above
        `threshold`, or None if no match is found.
        """
        if not registered_embeddings:
            return None

        best_id = None
        best_score = -1.0
        for face_id, emb in registered_embeddings:
            score = _cosine_similarity(query_embedding, emb)
            if score > best_score:
                best_score = score
                best_id = face_id

        if best_score >= threshold:
            logger.debug("[FaceRecognizer] Match: %s (score=%.4f)", best_id, best_score)
            return best_id
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / union if union > 0 else 0.0
