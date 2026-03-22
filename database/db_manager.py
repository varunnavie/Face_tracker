"""
database/db_manager.py
Handles all SQLite database operations for face registration, event logging,
and unique visitor counting.
"""

import sqlite3
import numpy as np
import json
import os
from datetime import datetime
from typing import Optional, List, Tuple


class DatabaseManager:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create tables if they do not exist."""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS faces (
                    face_id     TEXT PRIMARY KEY,
                    embedding   TEXT NOT NULL,
                    first_seen  TEXT NOT NULL,
                    last_seen   TEXT NOT NULL,
                    visit_count INTEGER DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_id     TEXT NOT NULL,
                    event_type  TEXT NOT NULL CHECK(event_type IN ('entry','exit')),
                    timestamp   TEXT NOT NULL,
                    image_path  TEXT,
                    FOREIGN KEY(face_id) REFERENCES faces(face_id)
                );

                CREATE TABLE IF NOT EXISTS visitor_count (
                    id            INTEGER PRIMARY KEY CHECK(id = 1),
                    unique_count  INTEGER DEFAULT 0
                );

                INSERT OR IGNORE INTO visitor_count(id, unique_count) VALUES (1, 0);
            """)

    # ------------------------------------------------------------------
    # Face registration
    # ------------------------------------------------------------------

    def register_face(self, face_id: str, embedding: np.ndarray) -> None:
        """Insert a new face into the faces table and increment unique count."""
        now = datetime.now().isoformat()
        emb_json = json.dumps(embedding.tolist())
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO faces(face_id, embedding, first_seen, last_seen) VALUES (?,?,?,?)",
                (face_id, emb_json, now, now)
            )
            conn.execute("UPDATE visitor_count SET unique_count = unique_count + 1 WHERE id = 1")

    def update_face_last_seen(self, face_id: str) -> None:
        """Update last_seen timestamp for a known face."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE faces SET last_seen = ?, visit_count = visit_count + 1 WHERE face_id = ?",
                (datetime.now().isoformat(), face_id)
            )

    def get_all_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        """Return list of (face_id, embedding_array) for every registered face."""
        with self._get_conn() as conn:
            rows = conn.execute("SELECT face_id, embedding FROM faces").fetchall()
        return [(row["face_id"], np.array(json.loads(row["embedding"]), dtype=np.float32))
                for row in rows]

    # ------------------------------------------------------------------
    # Event logging
    # ------------------------------------------------------------------

    def log_event(self, face_id: str, event_type: str, image_path: str) -> None:
        """Insert an entry or exit event."""
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO events(face_id, event_type, timestamp, image_path) VALUES (?,?,?,?)",
                (face_id, event_type, now, image_path)
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_unique_visitor_count(self) -> int:
        with self._get_conn() as conn:
            row = conn.execute("SELECT unique_count FROM visitor_count WHERE id = 1").fetchone()
        return row["unique_count"] if row else 0

    def get_all_events(self) -> List[dict]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT face_id, event_type, timestamp, image_path FROM events ORDER BY timestamp"
            ).fetchall()
        return [dict(r) for r in rows]

    def face_exists(self, face_id: str) -> bool:
        with self._get_conn() as conn:
            row = conn.execute("SELECT 1 FROM faces WHERE face_id = ?", (face_id,)).fetchone()
        return row is not None
