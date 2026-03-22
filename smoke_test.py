"""
smoke_test.py — quick pipeline validation (processes only first 200 frames).
Run: python smoke_test.py
"""

import os, sys, json, time, logging
import cv2

# Limit to first 200 frames for speed
MAX_FRAMES = 200

# Patch config: no display, no video save
cfg = json.load(open("config.json"))
cfg["display_output"] = False
cfg["save_output_video"] = False

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import generate_face_id, ensure_dirs
from detector.face_detector import FaceDetector
from recognizer.face_recognizer import FaceRecognizer
from tracker.face_tracker import FaceTracker
from database.db_manager import DatabaseManager
from logger.event_logger import EventLogger, setup_file_logger

logs_dir = "logs_test"
db_path  = "data/smoke_test.db"

ensure_dirs(logs_dir, "entries", "exits")
ensure_dirs(os.path.dirname(db_path), "")

setup_file_logger(logs_dir)
log = logging.getLogger("events")
log.info("=== SMOKE TEST START ===")

print("Loading YOLOv8 face detector...")
detector   = FaceDetector(cfg["yolo_model_path"], cfg["yolo_confidence"], cfg["yolo_iou"])
print("Loading InsightFace recognizer...")
recognizer = FaceRecognizer(cfg["insightface_model"])
db         = DatabaseManager(db_path)
event_log  = EventLogger(logs_dir, db)
tracker    = FaceTracker(cfg["exit_patience_frames"])

cap = cv2.VideoCapture(cfg["video_source"])
assert cap.isOpened(), f"Cannot open {cfg['video_source']}"

skip = cfg["detection_skip_frames"]
sim_thresh = cfg["similarity_threshold"]
min_face   = cfg["min_face_size"]

registered_embeddings = []
track_face_cache = {}
track_crop_cache = {}
track_bbox_cache = {}

frame_idx = 0
t0 = time.time()

while frame_idx < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    run_detection = (frame_idx % (skip + 1) == 0)
    seen_tracks = {}

    if run_detection:
        detections = detector.track(frame, persist=True)
        for (x1, y1, x2, y2, conf, tid) in detections:
            if (x2-x1) < min_face or (y2-y1) < min_face:
                continue
            crop = detector.crop_face(frame, x1, y1, x2, y2)
            bbox = (x1, y1, x2, y2)
            track_crop_cache[tid] = crop
            track_bbox_cache[tid] = bbox

            if tid in track_face_cache:
                seen_tracks[tid] = (bbox, crop, track_face_cache[tid])
                continue

            embedding = recognizer.get_embedding_from_crop(crop)
            if embedding is None:
                seen_tracks[tid] = (bbox, crop, None)
                continue

            matched_id = recognizer.find_best_match(embedding, registered_embeddings, sim_thresh)
            if matched_id:
                db.update_face_last_seen(matched_id)
                face_id = matched_id
            else:
                face_id = generate_face_id()
                db.register_face(face_id, embedding)
                event_log.log_registration(face_id)
                registered_embeddings = db.get_all_embeddings()

            track_face_cache[tid] = face_id
            seen_tracks[tid] = (bbox, crop, face_id)
    else:
        for tid in list(track_face_cache.keys()):
            if tid in track_bbox_cache:
                seen_tracks[tid] = (track_bbox_cache[tid], track_crop_cache.get(tid), track_face_cache[tid])

    entries, exits = tracker.update(seen_tracks)
    for face_id, crop in entries:
        event_log.log_entry(face_id, crop)
    for face_id, crop in exits:
        event_log.log_exit(face_id, crop)

    frame_idx += 1

cap.release()

for face_id, crop in tracker.flush_all():
    event_log.log_exit(face_id, crop)

elapsed = time.time() - t0
unique  = db.get_unique_visitor_count()
events  = db.get_all_events()

print(f"\n{'='*50}")
print(f"Smoke test PASSED")
print(f"Frames processed : {frame_idx}")
print(f"Elapsed time     : {elapsed:.1f}s  ({frame_idx/elapsed:.1f} fps)")
print(f"Unique visitors  : {unique}")
print(f"Total events     : {len(events)}")
print(f"{'='*50}")
log.info("=== SMOKE TEST END  unique=%d  events=%d ===", unique, len(events))
