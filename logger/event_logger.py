"""
logger/event_logger.py
Handles:
  1. Structured file logging to events.log
  2. Saving cropped face images to logs/entries/ and logs/exits/
  3. Wiring entry/exit events to the DatabaseManager
"""

import os
import logging
import cv2
import numpy as np
from datetime import datetime
from database.db_manager import DatabaseManager


def setup_file_logger(log_dir: str) -> logging.Logger:
    """Configure the root events logger to write to events.log."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "events.log")

    fmt = logging.Formatter(
        "%(asctime)s  [%(levelname)-8s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    console_handler.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Avoid adding duplicate handlers if setup_file_logger is called again
    if not root.handlers:
        root.addHandler(file_handler)
        root.addHandler(console_handler)
    else:
        root.handlers.clear()
        root.addHandler(file_handler)
        root.addHandler(console_handler)

    return logging.getLogger("events")


class EventLogger:
    def __init__(self, logs_dir: str, db: DatabaseManager):
        self.logs_dir = logs_dir
        self.db = db
        self.logger = logging.getLogger("events")

    # ------------------------------------------------------------------
    # Image saving
    # ------------------------------------------------------------------

    def _save_face_image(self, face_img: np.ndarray, event_type: str, face_id: str) -> str:
        """
        Save cropped face to  logs/<event_type>/YYYY-MM-DD/<face_id>_<timestamp>.jpg
        Returns the saved path.
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        folder_name = "entries" if event_type == "entry" else "exits"
        folder = os.path.join(self.logs_dir, folder_name, date_str)
        os.makedirs(folder, exist_ok=True)

        ts = datetime.now().strftime("%H%M%S_%f")[:13]
        filename = f"{face_id}_{ts}.jpg"
        path = os.path.join(folder, filename)

        if face_img is not None and face_img.size > 0:
            cv2.imwrite(path, face_img)
        return path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_entry(self, face_id: str, face_crop: np.ndarray, is_new: bool = True) -> None:
        """Log a face entry event."""
        img_path = self._save_face_image(face_crop, "entry", face_id)
        self.db.log_event(face_id, "entry", img_path)
        status = "NEW REGISTRATION" if is_new else "RE-ENTRY"
        self.logger.info(
            "ENTRY  | %-8s | face_id=%-20s | img=%s",
            status, face_id, img_path
        )

    def log_exit(self, face_id: str, face_crop: np.ndarray) -> None:
        """Log a face exit event."""
        img_path = self._save_face_image(face_crop, "exit", face_id)
        self.db.log_event(face_id, "exit", img_path)
        self.logger.info(
            "EXIT   |          | face_id=%-20s | img=%s",
            face_id, img_path
        )

    def log_registration(self, face_id: str) -> None:
        self.logger.info("REGISTER          | face_id=%s embedding generated and stored", face_id)

    def log_recognition(self, face_id: str, score: float) -> None:
        self.logger.debug("RECOGNIZE         | face_id=%s  similarity=%.4f", face_id, score)

    def log_tracking(self, face_id: str, track_id: int, bbox: tuple) -> None:
        self.logger.debug(
            "TRACKING          | face_id=%s  track_id=%d  bbox=%s",
            face_id, track_id, bbox
        )

    def log_system(self, msg: str) -> None:
        self.logger.info("SYSTEM            | %s", msg)
