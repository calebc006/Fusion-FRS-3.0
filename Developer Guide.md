# Developer's Guide for SimpliFRy

Technical documentation for SimpliFRy's facial recognition pipeline, tuning parameters, and API integration.

## Table of Contents

- [Developer's Guide for SimpliFRy](#developers-guide-for-simplifry)
  - [Table of Contents](#table-of-contents)
  - [FR Algorithm](#fr-algorithm)
  - [Enhancement: Differentiator](#enhancement-differentiator)
  - [Enhancement: Persistor](#enhancement-persistor)
  - [Configuration](#configuration)
  - [API Endpoints](#api-endpoints)
  - [Performance Optimizations](#performance-optimizations)

---

## FR Algorithm

### Core Pipeline

![FR Algorithm Diagram](./simpliFRy/assets/fr_algorithm.jpg)

1. **Database Indexing**: Face images are embedded via InsightFace and indexed in Voyager vector store (or stored as numpy array for brute force)
2. **Face Detection**: Query image → InsightFace detects faces, produces embeddings
3. **KNN Search**: Each query embedding queried against all faces in database using search algorithm.
4. **Similarity Matching**: Closest embedding retrieved; cosine similarity computed (lower = closer match)
5. **Classification**: If similarity score > threshold → unknown; if ≤ threshold → recognized (matches indexed face)

**Search Algorithms**:
- **Voyager (default)**: Fast approximate nearest neighbor search, ideal for large databases (>10k faces)
- **Brute Force**: Exact search computing all distances, slower but guaranteed optimal. May be faster for small databases (<10k faces)

**Note**: Cosine distance metric means lower scores indicate better matches.

---

## Enhancement: Differentiator

Addresses the threshold dilemma: strict thresholds reduce false positives but increase misses; lenient thresholds increase detection but false positives. 

**Mechanism**: 
1. If detection fails using normal `threshold`, retrieves top-2 closest embeddings:
2. If `dist(score₁, score₂) > similarity_gap` → Use **lenient threshold** (`threshold_lenient_diff`) and see if this allows for a match

---

## Enhancement: Persistor

Handles head rotation/minor pose changes within a frame sequence. When a face fails standard detection, check if it matches a recently-detected individual within spatial + temporal constraints.

**Mechanism**:
1. On successful detection, store query embedding + bounding box in persistor queue (up to `q_max_size`)
2. If detection fails, check persisted detections and see if
   - BBox overlap with old detection (IOU) ≥ `threshold_iou` AND
   - Cosine similarity to old detection ≥ `threshold_sim`   
3. If both pass → Use **lenient threshold** (`threshold_lenient_pers`) and see if this allows for a match.

**Execution Order**: Core FR → Differentiator → Persistor

![Persistor Diagram](./simpliFRy/assets/persistor.JPG)

**Note**: Currently uses Intersection-Over-Union for position matching; consider DeepSORT for better temporal tracking. Can be disabled in settings if unreliable for your use case.

---

## Configuration

Tunable parameters via `/submit_settings` endpoint. Settings are saved to `settings.json` All parameters are optional; unspecified values retain current settings.

### Core Parameters

| Parameter | Key | Default | Range | Description |
|-----------|-----|---------|-------|-------------|
| **FR Threshold** | `threshold` | 0.45 | [0.30, 0.90], step 0.01 | Max cosine distance for face match. Higher = more lenient |
| **Holding Time** | `holding_time` | 2 | [1, 120]s, step 1 | Duration to hold frontend display of detection |
| **Max Detections** | `max_detections` | 50 | [5, 100], step 1 | Maximum number of detections processed by backend |
| **Perf Logging** | `perf_logging` | false | bool | If enabled, periodically logs inference FPS, avg inference time, and search timings to `data/logs/` |
| **Frame Skip** | `frame_skip` | 1 | [1, 10], step 1 | Process every Nth frame (1=no skip, 2=every other frame). Higher values reduce CPU/GPU load |
| **Video Width** | `video_width` | 1920 | [1, 4000], step 1 | Width of video streamed to frontend |
| **Video Height** | `video_height` | 1080 | [1, 4000], step 1 | Height of video streamed to frontend |

### Differentiator Parameters

| Parameter | Key | Default | Range | Description |
|-----------|-----|---------|-------|-------------|
| **Enable** | `use_differentiator` | true | bool | Toggle differentiator mechanic |
| **Lenient Threshold** | `threshold_lenient_diff` | 0.55 | [0.30, 0.90], step 0.01 | Threshold when score gap > similarity_gap. Should be > FR Threshold |
| **Similarity Gap** | `similarity_gap` | 0.10 | [0.01, 0.20], step 0.01 | Min score difference to trigger lenient threshold |

### Persistor Parameters

| Parameter | Key | Default | Range | Description |
|-----------|-----|---------|-------|-------------|
| **Enable** | `use_persistor` | true | bool | Toggle persistor mechanic |
| **Max Queue Length** | `q_max_size` | 100 | [10, 500], step 10 | Max queue length for storing old detections for persistor |
| **IOU Threshold** | `threshold_iou` | 0.70 | [0.01, 1.00], step 0.01 | Min bbox overlap (low = lenient position check) |
| **Similarity Threshold** | `threshold_sim` | 0.60 | [0.01, 0.60], step 0.01 | Max distance between query and cached embedding (strict) |
| **Lenient Threshold** | `threshold_lenient_pers` | 0.60 | [0.30, 0.90], step 0.01 | Max distance to database embedding. Should be > Differentiator Lenient |

---

## API Endpoints

### 1. Start FR
**`POST /start`** - Start video stream and FR inference

**Request** (form-data):
- `stream_src` (required): RTSP URL
  ```
  rtsp://[username:password@]ip_address[:rtsp_port]/server_URL[?param1=val1&...]
  ```
- `data_file` (optional): JSON file path (relative to `data/` directory) mapping names to face image paths

**Response**:
```json
{"stream": true, "message": "Success!"}              // Started
{"stream": false, "message": "Stream already started!"}  // Already running
```

---

### 2. End FR
**`POST /end`** - Stop video stream and FR inference

**Response**:
```json
{"stream": true, "message": "Success!"}              // Stopped
{"stream": false, "message": "Stream not started!"}  // Not running
```

---

### 3. Check FR Status
**`GET /checkAlive`** - Check if FR is running

**Response**: `"Yes"` or `"No"` (plain text)

---

### 4. Video Feed
**`GET /vidFeed`** - HTTP stream of annotated video

**Response**: `multipart/x-mixed-replace; boundary=frame`

**Usage**:
```html
<object type="image/jpeg" data="/vidFeed"></object>
```

---

### 5. FR Results
**`GET /frResults`** - HTTP stream of detection results

**Response**: `application/json` stream
```json
{
  "data": [
    {
      "bbox": [0.2, 0.1, 0.4, 0.3],  // [x1, y1, x2, y2] normalized
      "label": "John Doe",
      "score": 0.35                   // cosine distance
    },
    {
      "label": "Jane Smith"           // recently detected (no bbox/score)
    }
  ]
}
```

**Integration**: See `static/js/detections.js` → `processStream()` for SSE parsing example

---

### 6. Update Settings
**`POST /submit_settings`** - Modify FR configuration

**Request** (form-data): See [Configuration](#configuration) table for all parameters

**Response**: HTTP 200 → redirect to `/settings`

---

## Performance Optimizations

### Video Streaming Latency

Several optimizations reduce end-to-end latency:

1. **Hardware Acceleration**: FFmpeg uses GPU decoding (`-hwaccel auto`) when available, on top of other latency optimization settings
2. **JPEG Quality**: Reduced to 75% for faster encoding (configurable in `VideoPlayer.py`)
3. **Frame Dropping**: Clients automatically skip to latest frame if behind, preventing buffering lag
4. **Thread Synchronization (RWLock)**: The video pipeline uses a **Read-Write Lock** (`RWLock` implemented in `VideoPlayer.py`) instead of a standard mutex for optimal concurrent access. Only the RTSP stream **writes** new frames, while the inference and broadcast threads only **read** `frame_bytes`. RWLock allows multiple simultaneous readers, only blocking for writes. This eliminates the ~90% wait time that occurred when video feed was open with a standard mutex.

### Inference Performance

**Model Configuration** (`FRVidPlayer.py`):
- **buffalo_l** model optimized with `det_size=(640,640)` and `det_thresh=0.5`
- Only loads detection + recognition modules (skips age/gender estimation)
- Supports CUDA for GPU acceleration (automatically detected)

**Frame Skip** (`frame_skip` setting):
- Process every Nth frame to reduce computational load
- Default=1 (no skip), recommended 2-3 for high-resolution streams
- Persistor mechanic maintains detections between skipped frames

**Expected Performance** (RTX A5000 + Buffalo_L @ 720p):
- 1 face: ~55-80 FPS (~12-18ms per inference)
- 3 faces: ~33-50 FPS (~20-30ms per inference)

**Performance Logging** (`perf_logging=true`):
```
[PERF] infer:12fps/18ms search:0.2ms skip:2
```
- `infer`: Actual inference rate / average time per inference
- `search`: Vector search time (Voyager or brute force)
- `skip`: Current frame_skip setting

### Frontend Rendering

**DOM Optimization** (`utils.js` → `updateBBoxes`):
- Reuses existing bounding box DOM elements instead of recreating
- Configurable via `showLabels` and `showUnknown` options

**Detection Broadcast Throttling** (`FRVidPlayer.py`):
- Caps detection updates at ~30 FPS
- Only sends updates when results change
- Reduces network overhead and client-side processing
