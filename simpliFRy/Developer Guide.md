# Developer's Guide for SimpliFRy

Technical documentation for SimpliFRy's facial recognition pipeline, tuning parameters, and API integration.

## Table of Contents

- [FR Algorithm](#fr-algorithm)
  - [Core Pipeline](#core-pipeline)
  - [Enhancement: Differentiator](#enhancement-differentiator)
  - [Enhancement: Persistor](#enhancement-persistor)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)

---

## FR Algorithm

### Core Pipeline

![FR Algorithm Diagram](assets/fr_algorithm.jpg)

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

**Key Insight**: False positives typically occur when top-2 similarity scores are close; legitimate matches show large score gaps.

**Mechanism**: Retrieves top-2 closest embeddings:
- If `(score₁ - score₂) > similarity_gap` → Use **lenient threshold** (high confidence in top match)
- Otherwise → Use **strict threshold** (ambiguous match)

---

## Enhancement: Persistor

Handles head rotation/minor pose changes within a frame sequence. When a face fails standard detection, check if it matches a recently-detected individual within spatial + temporal constraints.

**Mechanism**:
1. On successful detection, store query embedding + bounding box for `holding_time` seconds
2. If detection fails, check stored embeddings:
   - BBox overlap (IOU) ≥ threshold AND
   - Similarity to stored embedding ≤ strict threshold AND  
   - Similarity to database embedding ≤ lenient threshold
3. If all pass → recognition via persistor; update stored embedding

**Execution Order**: Core FR → Differentiator → Persistor

![Persistor Diagram](assets/persistor.JPG)

**Note**: Currently uses Intersection-Over-Union for position matching; consider DeepSORT for better temporal tracking. Can be disabled in settings if unreliable for your use case.

---

## Configuration

Tunable parameters via `/submit_settings` endpoint (Settings UI posts here). All parameters are optional; unspecified values retain current settings.

### Core Parameters

| Parameter | Key | Default | Range | Description |
|-----------|-----|---------|-------|-------------|
| **FR Threshold** | `threshold` | 0.45 | [0.30, 0.90], step 0.01 | Max cosine distance for face match. Higher = more lenient |
| **Holding Time** | `holding_time` | 3 | [1, 120]s, step 1 | Duration to cache recognized faces (affects persistor + sidebar display) |
| **Use Brute Force** | `use_brute_force` | false | bool | Toggle between brute force (exact) and Voyager (fast ANN) search |
| **Perf Logging** | `perf_logging` | false | bool | If enabled, periodically logs stream FPS and inference/search timings to `data/logs/` |

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
| **Persistor Threshold** | `threshold_prev` | 0.30 | [0.01, 0.60], step 0.01 | Max distance between query and cached embedding (strict) |
| **IOU Threshold** | `threshold_iou` | 0.20 | [0.01, 1.00], step 0.01 | Min bbox overlap (low = lenient position check) |
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
