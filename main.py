"""
main.py
Entry point for the Intelligent Face Tracker.

Usage:
    python main.py                          # uses config.json defaults
    python main.py --source videos/foo.mp4  # override video source
    python main.py --source rtsp://...      # live RTSP stream
    python main.py --config my_config.json  # custom config file
"""

import argparse
import os
import sys
import time
import logging
import cv2
import numpy as np

from utils.helpers import load_config, generate_face_id, annotate_frame, ensure_dirs
from detector.face_detector import FaceDetector
from recognizer.face_recognizer import FaceRecognizer
from tracker.face_tracker import FaceTracker
from database.db_manager import DatabaseManager
from logger.event_logger import EventLogger, setup_file_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intelligent Face Tracker")
    p.add_argument("--source", type=str, default=None,
                   help="Video file path or RTSP URL (overrides config.json)")
    p.add_argument("--config", type=str, default="config.json",
                   help="Path to config file (default: config.json)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(cfg: dict, source: str) -> None:
    logs_dir  = cfg["logs_dir"]
    db_path   = cfg["database_path"]

    # -- Ensure required directories exist ---------------------------------
    ensure_dirs(logs_dir, "entries", "exits")
    ensure_dirs(os.path.dirname(db_path) or ".", "")
    ensure_dirs(os.path.dirname(cfg.get("output_video_path", "output/out.mp4")) or "output", "")

    # -- Setup logging -------------------------------------------------------
    setup_file_logger(logs_dir)
    log = logging.getLogger("events")
    log.info("=" * 70)
    log.info("SYSTEM  | Face Tracker starting  source=%s", source)

    # -- Instantiate modules -------------------------------------------------
    log.info("SYSTEM  | Loading YOLOv8 face detector...")
    detector = FaceDetector(
        model_path=cfg["yolo_model_path"],
        confidence=cfg["yolo_confidence"],
        iou=cfg["yolo_iou"]
    )

    log.info("SYSTEM  | Loading InsightFace recognizer (%s)...", cfg["insightface_model"])
    recognizer = FaceRecognizer(model_name=cfg["insightface_model"])

    db        = DatabaseManager(db_path)
    event_log = EventLogger(logs_dir, db)
    tracker   = FaceTracker(exit_patience_frames=cfg["exit_patience_frames"])

    skip       = cfg["detection_skip_frames"]   # frames to skip between full detections
    sim_thresh = cfg["similarity_threshold"]
    min_face   = cfg["min_face_size"]

    # -- Open video source ---------------------------------------------------
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error("Cannot open source: %s", source)
        sys.exit(1)

    fps_src   = cap.get(cv2.CAP_PROP_FPS) or 25
    w         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("SYSTEM  | Stream  %dx%d  @%.1f fps", w, h, fps_src)

    # Optional output video writer
    writer = None
    if cfg.get("save_output_video"):
        out_path = cfg["output_video_path"]
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_src, (w, h))

    # -- Processing loop -----------------------------------------------------
    frame_idx       = 0
    prev_time       = time.time()
    display_fps     = 0.0

    # Cache of embeddings loaded from DB (refreshed after each registration)
    registered_embeddings = db.get_all_embeddings()

    # Track → face_id mapping that persists BETWEEN detection cycles
    # (so we don't re-run recognition on every frame for known tracks)
    track_face_cache: dict = {}          # track_id -> face_id
    track_crop_cache: dict = {}          # track_id -> last crop
    track_bbox_cache: dict = {}          # track_id -> last bbox

    log.info("SYSTEM  | Processing started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            log.info("SYSTEM  | End of stream / cannot read frame.")
            break

        run_detection = (frame_idx % (skip + 1) == 0)

        # ----------------------------------------------------------------
        # Detection + Tracking (ByteTrack) every N frames
        # ----------------------------------------------------------------
        seen_tracks: dict = {}   # track_id -> (bbox, crop, face_id_or_None)

        if run_detection:
            detections = detector.track(frame, persist=True)

            for (x1, y1, x2, y2, conf, tid) in detections:
                # Filter too-small faces
                if (x2 - x1) < min_face or (y2 - y1) < min_face:
                    continue

                crop = detector.crop_face(frame, x1, y1, x2, y2)
                bbox = (x1, y1, x2, y2)
                track_crop_cache[tid] = crop
                track_bbox_cache[tid] = bbox

                # Already know this track → skip recognition
                if tid in track_face_cache:
                    seen_tracks[tid] = (bbox, crop, track_face_cache[tid])
                    continue

                # --------------------------------------------------------
                # Recognition: get embedding, match or register
                # --------------------------------------------------------
                embedding = recognizer.get_embedding_from_crop(crop)
                if embedding is None:
                    seen_tracks[tid] = (bbox, crop, None)
                    continue

                matched_id = recognizer.find_best_match(
                    embedding, registered_embeddings, threshold=sim_thresh
                )

                if matched_id is not None:
                    # Known face re-entering or continuing
                    db.update_face_last_seen(matched_id)
                    event_log.log_recognition(matched_id, 0.0)
                    face_id = matched_id
                    is_new = False
                else:
                    # Brand-new face → register
                    face_id = generate_face_id()
                    db.register_face(face_id, embedding)
                    event_log.log_registration(face_id)
                    registered_embeddings = db.get_all_embeddings()   # refresh cache
                    is_new = True

                track_face_cache[tid] = face_id
                seen_tracks[tid] = (bbox, crop, face_id)

        else:
            # On skipped frames, propagate last known state for all active tracks
            for tid in list(track_face_cache.keys()):
                if tid in track_bbox_cache:
                    seen_tracks[tid] = (
                        track_bbox_cache[tid],
                        track_crop_cache.get(tid),
                        track_face_cache[tid]
                    )

        # ----------------------------------------------------------------
        # Tracker state machine → generate entry / exit events
        # ----------------------------------------------------------------
        entries, exits = tracker.update(seen_tracks)

        for face_id, crop in entries:
            is_new = not any(e["face_id"] == face_id
                             for e in db.get_all_events()
                             if e["event_type"] == "entry")
            event_log.log_entry(face_id, crop, is_new=is_new)

        for face_id, crop in exits:
            event_log.log_exit(face_id, crop)
            # Clean stale track cache entries for this face
            stale = [tid for tid, fid in track_face_cache.items() if fid == face_id]
            for tid in stale:
                track_face_cache.pop(tid, None)

        # ----------------------------------------------------------------
        # Annotation + Display
        # ----------------------------------------------------------------
        unique_count = db.get_unique_visitor_count()

        # Build annotation list from currently seen tracks
        annotation_items = []
        for tid, (bbox, crop, face_id) in seen_tracks.items():
            if face_id:
                x1, y1, x2, y2 = bbox
                annotation_items.append((x1, y1, x2, y2, face_id, False))

        now = time.time()
        display_fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        annotated = annotate_frame(frame, annotation_items, unique_count, display_fps)

        if writer:
            writer.write(annotated)

        if cfg.get("display_output", True):
            cv2.imshow("Face Tracker", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                log.info("SYSTEM  | User requested quit.")
                break

        frame_idx += 1

    # ----------------------------------------------------------------
    # Flush remaining active tracks as exits
    # ----------------------------------------------------------------
    remaining_exits = tracker.flush_all()
    for face_id, crop in remaining_exits:
        event_log.log_exit(face_id, crop)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    final_count = db.get_unique_visitor_count()
    log.info("=" * 70)
    log.info("SYSTEM  | Processing complete.")
    log.info("SYSTEM  | Total unique visitors: %d", final_count)
    log.info("SYSTEM  | Total frames processed: %d", frame_idx)
    log.info("=" * 70)
    print(f"\n✓ Done. Unique visitors detected: {final_count}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)

    # Resolve video source: CLI arg > config rtsp > config video_source
    if args.source:
        source = args.source
    elif cfg.get("rtsp_stream_url"):
        source = cfg["rtsp_stream_url"]
    else:
        source = cfg["video_source"]

    # Change working directory to the project root so relative paths work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    run(cfg, source)
