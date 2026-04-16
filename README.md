# Intelligent Face Tracker with Auto-Registration and Visitor Counting

A production-grade AI system that processes a video stream (or live RTSP camera feed) to detect, track, and recognise faces in real time. Every new face is automatically registered with a unique ID; every entry and exit is logged with a timestamped cropped image and persisted to a SQLite database. The system maintains an accurate count of unique visitors across the entire session.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Feature List](#feature-list)
3. [Tech Stack](#tech-stack)
4. [Setup Instructions](#setup-instructions)
5. [Usage](#usage)
6. [Configuration Reference](#configuration-reference)
7. [Output Structure](#output-structure)
8. [Database Schema](#database-schema)
9. [AI Planning Document](#ai-planning-document)
10. [Compute Load Estimate](#compute-load-estimate)
11. [Assumptions](#assumptions)
12. [Demo Video](#demo-video)

---

## Architecture

```
┌─────────────────────┐
│  Video File         │
│                     │
└────────┬────────────┘
         │  frame
         ▼
┌─────────────────────────────────────────────────────────┐
│                        main.py                          │
│              (Pipeline Orchestrator)                    │
└──┬──────────────┬───────────────┬───────────────────────┘
   │              │               │
   ▼              ▼               ▼
┌──────────┐ ┌──────────┐ ┌────────────────┐
│ Face     │ │ Face     │ │  Face Tracker  │
│ Detector │ │ Recogn-  │ │  (Entry/Exit   │
│ (YOLO +  │ │ izer     │ │   State        │
│ ByteTrack│ │ (InsightF│ │   Machine)     │
│ )        │ │ ace ArcF │ └───────┬────────┘
└──────────┘ │ ace)     │         │
             └────┬─────┘         │ entry / exit events
                  │               │
                  └───────┬───────┘
                          │
              ┌───────────┴──────────┐
              │                      │
              ▼                      ▼
     ┌─────────────────┐   ┌──────────────────┐
     │  Database Mgr   │   │  Event Logger    │
     │  (SQLite)       │   │  events.log      │
     │  faces / events │   │  logs/entries/   │
     │  visitor_count  │   │  logs/exits/     │
     └─────────────────┘   └──────────────────┘
              │
              ▼
     ┌─────────────────┐
     │  Annotated      │
     │  Display Window │
     │  (OpenCV)       │
     └─────────────────┘
```

### Module Responsibilities

| Module | File | Role |
|---|---|---|
| **Orchestrator** | `main.py` | Frame loop, pipeline wiring, CLI entry point |
| **Face Detector** | `detector/face_detector.py` | YOLOv8n-face detection + ByteTrack tracking; returns bboxes with stable track IDs |
| **Face Recognizer** | `recognizer/face_recognizer.py` | InsightFace ArcFace 512-d embeddings; cosine-similarity matching |
| **Face Tracker** | `tracker/face_tracker.py` | State machine: maps ByteTrack IDs → Face IDs; fires entry/exit events |
| **Database Manager** | `database/db_manager.py` | SQLite CRUD: registers faces, logs events, maintains unique count |
| **Event Logger** | `logger/event_logger.py` | Writes `events.log`; saves cropped face images per event |
| **Helpers** | `utils/helpers.py` | Config loading, UUID generation, frame annotation |

---

## Feature List

1. **Stable multi-face tracking** via ByteTrack — assigns consistent track IDs across frames, handles occlusion
2. **ArcFace identity embeddings** (512-d) via InsightFace `buffalo_l` model — SOTA recognition accuracy
3. **Auto-registration** — any face not matching an existing embedding gets a new UUID and is stored in the DB
4. **Re-identification** — when a previously registered person re-enters, they are recognised (not counted twice)
5. **Entry event** fired exactly once per track appearance; includes cropped face image + timestamp
6. **Exit event** fired after configurable patience period of absence; includes last-seen face crop
7. **Unique visitor counter** — lifetime count stored in DB, never inflated by re-visits
8. **SQLite database** — three tables: `faces` (embeddings + metadata), `events` (entry/exit log), `visitor_count`
9. **`events.log`** — structured file log of every system event: registration, entry, exit, tracking, recognition
10. **Cropped face images** saved under `logs/entries/YYYY-MM-DD/` and `logs/exits/YYYY-MM-DD/`
11. **`config.json` driven** — all tunable parameters (skip frames, similarity threshold, patience, etc.) in one file
12. **CLI overrides** — `--source` and `--config` flags for flexible deployment
13. **Live annotated display** — OpenCV window with bounding boxes, face IDs, visitor count HUD, FPS
14. **Graceful shutdown** — remaining active tracks are flushed as exits when stream ends or user presses `q`
15. **Modular, replaceable components** — each module is independently testable and swappable

---

## Tech Stack

| Layer | Technology |
|---|---|
| Face Detection | YOLOv8n-face (Ultralytics) |
| Face Recognition | InsightFace `buffalo_l` — ArcFace R50 |
| Object Tracking | ByteTrack (built into Ultralytics YOLO) |
| Inference Runtime | ONNX Runtime |
| Backend | Python 3.9+ |
| Database | SQLite 3 (via stdlib `sqlite3`) |
| Video I/O | OpenCV |
| Configuration | JSON (`config.json`) |
| Logging | Python `logging` module + filesystem |

---

## Setup Instructions

### Prerequisites

- Python 3.9 or 3.10 (Python 3.11 also works)
- Windows 10/11, Linux, or macOS
- (Recommended) NVIDIA GPU with CUDA 11.8+ for real-time performance

### 1. Clone the repository

```bash
git clone https://github.com/varunnavie/Face_tracker.git
cd Face_tracker
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Windows note:** `insightface` requires compilation of a Cython extension.
> If you encounter a `Microsoft Visual C++ 14.0 required` error, install the free
> [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
> and retry, **or** install insightface from the patched source (see below).

**Alternative insightface install (skips unneeded 3D mesh Cython extension):**

```bash
# Download the source
pip download insightface==0.7.3 --no-deps -d /tmp/if_src
cd /tmp && tar -xzf if_src/insightface-0.7.3.tar.gz

# Edit setup.py: remove ext_modules= and the Cython imports (lines 11-12, 59-60, 75, 95)
# Then install:
pip install /tmp/insightface-0.7.3

# Create stub for the uncompiled C extension (not needed for ArcFace pipeline):
# In <venv>/Lib/site-packages/insightface/thirdparty/face3d/mesh/cython/
# create mesh_core_cython.py with stub functions that raise NotImplementedError
```

### 4. YOLO face model (auto-downloaded on first run)

The `yolov8n-face.pt` model is downloaded automatically on first run from a public
HuggingFace mirror. To pre-download manually:

```bash
mkdir models
python -c "
import urllib.request
urllib.request.urlretrieve(
  'https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt',
  'models/yolov8n-face.pt'
)"
```

### 5. InsightFace model pack (auto-downloaded on first run)

On first run, InsightFace downloads `buffalo_l` (~300 MB) to `~/.insightface/models/`.
Ensure you have an internet connection the first time.

---

## Usage

### Run on the sample video (default)

```bash
python main.py
```

### Run on a specific video file

```bash
python main.py --source videos/record_20250620_183903.mp4
```

### Run on a live RTSP stream (interview / production mode)

```bash
python main.py --source rtsp://username:password@192.168.1.100:554/stream
```

### Use a custom config file

```bash
python main.py --config my_config.json
```

### Controls

| Key | Action |
|---|---|
| `q` | Quit and flush remaining active tracks as exits |

---

## Configuration Reference

### Sample `config.json`

```json
{
    "video_source": "videos/video_sample1.mp4",
    "detection_skip_frames": 2,
    "similarity_threshold": 0.45,
    "exit_patience_frames": 40,
    "yolo_model_path": "models/yolov8n-face.pt",
    "insightface_model": "buffalo_l",
    "database_path": "data/face_tracker.db",
    "logs_dir": "logs",
    "display_output": true,
    "save_output_video": false,
    "output_video_path": "output/output.mp4",
    "min_face_size": 30,
    "yolo_confidence": 0.45,
    "yolo_iou": 0.45,
    "rtsp_stream_url": ""
}
```

### Parameter Reference

| Parameter | Default | Description |
|---|---|---|
| `video_source` | `videos/video_sample1.mp4` | Path to input video file |
| `detection_skip_frames` | `2` | Run YOLO every N+1 frames (2 = detect every 3rd frame). Higher = faster but may miss brief appearances |
| `similarity_threshold` | `0.45` | Cosine similarity above which two ArcFace embeddings are considered the same person. Lower = stricter (fewer false positives, more false negatives) |
| `exit_patience_frames` | `40` | Number of consecutive frames a track must be absent before an exit event fires. Prevents spurious exits during brief occlusions |
| `yolo_model_path` | `models/yolov8n-face.pt` | Path to YOLO face model weights |
| `insightface_model` | `buffalo_l` | InsightFace model pack name (`buffalo_l` for best accuracy, `buffalo_s` for speed) |
| `database_path` | `data/face_tracker.db` | SQLite database file path |
| `logs_dir` | `logs` | Root directory for log file and face images |
| `display_output` | `true` | Show live annotated OpenCV window |
| `save_output_video` | `false` | Write annotated video to file |
| `output_video_path` | `output/output.mp4` | Output video path (used when `save_output_video` is true) |
| `min_face_size` | `30` | Minimum bounding box side length in pixels; smaller faces are ignored |
| `yolo_confidence` | `0.45` | YOLO detection confidence threshold |
| `yolo_iou` | `0.45` | YOLO NMS IoU threshold |
| `rtsp_stream_url` | `""` | If set, overrides `video_source` with this RTSP URL |

---

## Output Structure

```
logs/
├── events.log              ← Structured log of all system events
├── entries/
│   └── YYYY-MM-DD/
│       └── face_<id>_<HHMMSSffffff>.jpg   ← Cropped face at entry moment
└── exits/
    └── YYYY-MM-DD/
        └── face_<id>_<HHMMSSffffff>.jpg   ← Cropped face at exit moment

data/
└── face_tracker.db         ← SQLite database
    ├── faces               ← Registered face identities + embeddings
    ├── events              ← All entry/exit events with image paths
    └── visitor_count       ← Running unique visitor total

models/
└── yolov8n-face.pt         ← YOLOv8 face detection weights

output/
└── output.mp4              ← Annotated output video (if enabled)
```

### Sample `events.log` output

```
2026-03-22 19:46:13  [INFO    ]  REGISTER          | face_id=face_4d55d254 embedding generated and stored
2026-03-22 19:46:13  [INFO    ]  ENTRY  | NEW REGISTRATION | face_id=face_4d55d254 | img=logs/entries/2026-03-22/face_4d55d254_194613.jpg
2026-03-22 19:46:22  [INFO    ]  REGISTER          | face_id=face_e75b0933 embedding generated and stored
2026-03-22 19:46:22  [INFO    ]  ENTRY  | NEW REGISTRATION | face_id=face_e75b0933 | img=logs/entries/2026-03-22/face_e75b0933_194622.jpg
2026-03-22 19:59:21  [INFO    ]  ENTRY  | RE-ENTRY         | face_id=face_4d55d254 | img=logs/entries/2026-03-22/face_4d55d254_195921.jpg
2026-03-22 20:00:27  [INFO    ]  EXIT   |                  | face_id=face_4d55d254 | img=logs/exits/2026-03-22/face_4d55d254_200027.jpg
```

---

## Database Schema

```sql
-- Registered unique face identities
CREATE TABLE faces (
    face_id     TEXT PRIMARY KEY,          -- e.g. "face_4d55d254"
    embedding   TEXT NOT NULL,             -- JSON array of 512 floats (ArcFace)
    first_seen  TEXT NOT NULL,             -- ISO 8601 timestamp
    last_seen   TEXT NOT NULL,
    visit_count INTEGER DEFAULT 1          -- number of times this face has entered
);

-- Entry and exit event log
CREATE TABLE events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id     TEXT NOT NULL,
    event_type  TEXT NOT NULL,             -- 'entry' or 'exit'
    timestamp   TEXT NOT NULL,
    image_path  TEXT,                      -- path to cropped face image
    FOREIGN KEY(face_id) REFERENCES faces(face_id)
);

-- Single-row unique visitor counter
CREATE TABLE visitor_count (
    id           INTEGER PRIMARY KEY CHECK(id = 1),
    unique_count INTEGER DEFAULT 0
);
```

---

## AI Planning Document

### Phase 1 — Problem Analysis

The core challenge has three distinct sub-problems:

1. **Detection** — find every face in every frame, fast enough for real-time
2. **Identity** — determine if a detected face belongs to a known person or is new
3. **Lifecycle** — track when someone enters vs. exits without double-counting

A naive approach (detect + compare every frame) would be too slow and fragile. The architecture separates these concerns into independent, replaceable modules.

### Phase 2 — Design Decisions

**Why YOLOv8 for detection?**
YOLOv8n-face is a single-pass CNN trained specifically on face detection. At 640×640 input it runs at ~30 fps on GPU and ~5 fps on CPU. It returns confidence scores and bounding boxes in one forward pass, and integrates natively with ByteTrack for multi-object tracking.

**Why InsightFace / ArcFace for recognition?**
ArcFace (Additive Angular Margin Loss) is specifically designed for face verification. It maps faces to a hypersphere where intra-class angles are minimised and inter-class angles are maximised. The 512-d embeddings are highly discriminative — cosine similarity above ~0.45 reliably indicates the same person across lighting, angle, and partial occlusion changes. The `face_recognition` library (dlib HOG + ResNet) is far less accurate for production use.

**Why ByteTrack for tracking?**
ByteTrack assigns stable integer track IDs across frames using a Kalman filter + Hungarian algorithm for IoU-based assignment. It handles brief occlusions (a face passes behind an object for 1-2 seconds) without creating a new track ID. This means the system doesn't falsely fire a new entry/exit event during momentary occlusion.

**Why cosine similarity, not Euclidean distance?**
ArcFace embeddings are L2-normalised onto a unit hypersphere. Cosine similarity is equivalent to the dot product of two unit vectors and is the canonical metric for this embedding space. It is scale-invariant and well-calibrated: similarity > 0.45 → same person, < 0.3 → different person, 0.3–0.45 → ambiguous.

**Why SQLite?**
Zero configuration, single file, no server process, ACID-compliant, and fully sufficient for the throughput of a single-camera system. The database is used for persistence across runs and for querying unique visitor counts.

### Phase 3 — Pipeline Flow (per frame)

```
1. Read frame from VideoCapture
2. Every (skip+1)th frame: run YOLO.track() → list of (bbox, track_id)
   - On non-detection frames: reuse last known track positions
3. For each (bbox, track_id):
   a. If track_id already mapped to a face_id → skip recognition
   b. Otherwise:
      i.  Crop face from frame
      ii. Extract 512-d ArcFace embedding via InsightFace
      iii. Compare embedding against all registered faces (cosine similarity)
      iv. If match found (score ≥ threshold): use existing face_id, update last_seen in DB
      v.  If no match: generate new face_id, register embedding in DB, refresh cache
      vi. Map track_id → face_id in memory
4. Call tracker.update(seen_tracks):
   - For newly mapped track IDs: fire ENTRY event
   - For track IDs absent > patience frames: fire EXIT event
5. Log all entry/exit events to DB + log file + save cropped images
6. Annotate frame with bboxes, face IDs, visitor count HUD
7. Display / write to output video
```

### Phase 4 — Performance Optimisation

- **Frame skipping:** `detection_skip_frames=2` means YOLO runs every 3rd frame. On skipped frames, last-known track positions are used — negligible accuracy loss at typical walking speed.
- **Embedding cache:** Once a track_id is mapped to a face_id, recognition is not repeated for that track. This eliminates ~80% of InsightFace inference calls.
- **Embedding comparison is O(n)** where n = number of registered faces. For a single-camera deployment, n rarely exceeds a few hundred — no indexing needed.

---

## Compute Load Estimate

| Component | CPU Load | GPU Load | VRAM |
|---|---|---|---|
| YOLOv8n-face detection | ~15% | ~20% | ~0.5 GB |
| InsightFace buffalo_l recognition | ~10% | ~15% | ~1.0 GB |
| ByteTrack (Kalman + Hungarian) | ~3% | — | — |
| OpenCV + DB + file logging | ~5% | — | — |
| **Total (GPU mode)** | **~33%** | **~35%** | **~1.5 GB** |
| **Total (CPU mode)** | **~75–85%** | — | — |

**Throughput:**
- GPU (RTX 3060+): ~25–30 fps at 1080p
- CPU (modern 8-core): ~4–6 fps at 1080p (sufficient for post-processing; not real-time display)

---

## Assumptions

1. **Minimum face size of 30 pixels** — faces smaller than 30×30 px are ignored. At typical CCTV mounting heights, this filters background noise without missing close-range faces.
2. **Cosine similarity threshold of 0.45** — empirically validated for buffalo_l ArcFace embeddings. Should be tuned per deployment environment if lighting conditions are unusual.
3. **Exit patience of 40 frames** — a face must be absent for 40 consecutive frames (~1.6 seconds at 25 fps) before an exit is logged. This prevents spurious exits from momentary occlusion.
4. **One entry event per track appearance** — if the same person leaves and re-enters, they generate a second entry event but are NOT counted as a new unique visitor.
5. **Unique visitor count is lifetime (per database file)** — running `main.py` on multiple videos using the same database file accumulates counts across sessions. Delete `data/face_tracker.db` to reset.
6. **Video files are H.264 encoded** — standard encoding readable by OpenCV. Other codecs may require additional codec packages.
7. **Single camera, fixed angle** — the system is designed for one feed. Multi-camera re-identification would require a shared embedding database.
8. **No frontend UI required** — the OpenCV annotated window and log files constitute the output interface.

---

## Demo Video

> **https://www.loom.com/share/8de8b3a2ab804dd4af6d62b44f2e7d1b**



---

*This project is a part of a hackathon run by https://katomaran.com*
