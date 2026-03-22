"""
tracker/face_tracker.py
Manages the lifecycle of tracked faces:
  - Maps YOLO ByteTrack IDs  →  application Face IDs
  - Detects entry (first time a track_id is associated with a face_id)
  - Detects exit  (track_id absent for > exit_patience_frames consecutive frames)
  - Keeps a count of how many frames each track has been missing
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple, List
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrackState:
    face_id: str
    last_bbox: Tuple[int, int, int, int]   # (x1,y1,x2,y2) at last seen frame
    last_crop: Optional[np.ndarray]         # face image at last seen frame
    missing_frames: int = 0
    has_entered: bool = False               # entry event already fired?


class FaceTracker:
    def __init__(self, exit_patience_frames: int = 40):
        """
        Args:
            exit_patience_frames : number of consecutive frames a track must be
                                   absent before an exit event is fired.
        """
        self.exit_patience = exit_patience_frames

        # track_id (int) -> TrackState
        self._active: Dict[int, TrackState] = {}

        # face_id -> set of track_ids that have ever been this face
        # (helps handle the same person being re-tracked under a new track_id)
        self._face_to_tracks: Dict[str, Set[int]] = {}

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(
        self,
        seen_tracks: Dict[int, Tuple[Tuple[int,int,int,int], Optional[np.ndarray], Optional[str]]],
    ) -> Tuple[List[Tuple[str, np.ndarray]], List[Tuple[str, np.ndarray]]]:
        """
        Called once per frame with the currently visible tracks.

        Args:
            seen_tracks : { track_id: (bbox, face_crop, face_id_or_None) }
                          face_id is None when recognition hasn't been performed yet.

        Returns:
            (entries, exits)
            entries : list of (face_id, crop) for newly entered faces
            exits   : list of (face_id, crop) for newly exited faces
        """
        entries: List[Tuple[str, np.ndarray]] = []
        exits:   List[Tuple[str, np.ndarray]] = []

        current_ids = set(seen_tracks.keys())

        # --- Process visible tracks ---
        for tid, (bbox, crop, face_id) in seen_tracks.items():
            if face_id is None:
                # Recognition not done yet; update bbox/crop only
                if tid in self._active:
                    if crop is not None:
                        self._active[tid].last_crop = crop
                    self._active[tid].last_bbox = bbox
                    self._active[tid].missing_frames = 0
                continue

            if tid not in self._active:
                # Brand new track
                state = TrackState(
                    face_id=face_id,
                    last_bbox=bbox,
                    last_crop=crop,
                    has_entered=False
                )
                self._active[tid] = state
                self._face_to_tracks.setdefault(face_id, set()).add(tid)

            state = self._active[tid]
            state.missing_frames = 0
            if crop is not None:
                state.last_crop = crop
            state.last_bbox = bbox

            # Fire entry event once per track
            if not state.has_entered:
                state.has_entered = True
                entries.append((face_id, crop if crop is not None else np.zeros((112,112,3), dtype=np.uint8)))
                logger.debug("[Tracker] ENTRY  face_id=%s  track_id=%d", face_id, tid)

        # --- Increment missing counter for absent tracks ---
        for tid in list(self._active.keys()):
            if tid not in current_ids:
                self._active[tid].missing_frames += 1

                if self._active[tid].missing_frames >= self.exit_patience:
                    state = self._active.pop(tid)
                    exits.append((
                        state.face_id,
                        state.last_crop if state.last_crop is not None
                        else np.zeros((112, 112, 3), dtype=np.uint8)
                    ))
                    logger.debug("[Tracker] EXIT   face_id=%s  track_id=%d", state.face_id, tid)

        return entries, exits

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_face_id(self, track_id: int) -> Optional[str]:
        state = self._active.get(track_id)
        return state.face_id if state else None

    def assign_face_id(self, track_id: int, face_id: str,
                       bbox: Tuple[int,int,int,int], crop: Optional[np.ndarray]) -> None:
        """Assign a face_id to an existing or new track."""
        if track_id not in self._active:
            self._active[track_id] = TrackState(
                face_id=face_id, last_bbox=bbox, last_crop=crop
            )
        else:
            self._active[track_id].face_id = face_id
            self._active[track_id].last_bbox = bbox
            if crop is not None:
                self._active[track_id].last_crop = crop
        self._face_to_tracks.setdefault(face_id, set()).add(track_id)

    @property
    def active_face_ids(self) -> Set[str]:
        return {s.face_id for s in self._active.values()}

    def flush_all(self) -> List[Tuple[str, np.ndarray]]:
        """
        Force-exit all remaining active tracks (call at end of stream).
        Returns list of (face_id, crop) for each flushed track.
        """
        exits = []
        for tid, state in list(self._active.items()):
            if state.has_entered:
                exits.append((
                    state.face_id,
                    state.last_crop if state.last_crop is not None
                    else np.zeros((112, 112, 3), dtype=np.uint8)
                ))
        self._active.clear()
        return exits
