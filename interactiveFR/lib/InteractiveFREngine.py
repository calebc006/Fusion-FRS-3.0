import sys
import os
import threading
import logging
import time
import traceback
from datetime import datetime
from typing import Generator, TypedDict
import subprocess
from collections import deque
import json
import numpy as np
import cv2
from insightface.app import FaceAnalysis
import voyager

from lib.VideoPlayer import VideoPlayer, StreamState
from lib.utils import calc_iou, log_info


FR_SETTINGS_PATH = 'settings.json'
FR_DEFAULT_SETTINGS = {
    "threshold": 0.45, 
    "holding_time": 2, 
    "max_detections": 50, 
    "perf_logging": False, 
    "frame_skip": 1, 
    "max_broadcast_fps": 50,
    "video_width": 1920,
    "video_height": 1080,

    "use_differentiator": True, 
    "threshold_lenient_diff": 0.55,
    "similarity_gap": 0.10, 

    "use_persistor": True, 
    "q_max_size": 100,
    "threshold_iou": 0.7, 
    "threshold_sim": 0.6,
    "threshold_lenient_pers": 0.60, 
}

def is_cuda_available() -> bool:
    try:
        subprocess.check_output(['nvidia-smi'])
        return True
    except Exception:
        return False


class FRResult(TypedDict, total=False):
    label: str
    bbox: list[float]
    norm_embed: np.ndarray
    score: float
    last_seen: float
    is_target: bool


class FRSettings(TypedDict):
    threshold: float
    holding_time: int
    perf_logging: bool
    max_detections: int
    use_differentiator: bool
    threshold_lenient_diff: float
    similarity_gap: float
    use_persistor: bool
    q_max_size: int
    threshold_sim: float
    threshold_iou: float
    threshold_lenient_pers: float
    frame_skip: int
    max_broadcast_fps: int
    video_width: int
    video_height: int

class PerfLog(TypedDict):
    fps: float
    avg_ms: float
    infer_ms: float
    match_ms: float
    remaining_ms: float
    skip_frac: float
    

class VoyagerEmbeddingIndex:
    '''An embedding index for ANN query using the Spotify Voyager algorithm'''

    def __init__(self, n_dimensions=512):
        self.n_dimensions = n_dimensions

        self.vector_index: voyager.Index = voyager.Index(voyager.Space.Cosine, num_dimensions=n_dimensions)
        self.idx_to_name = []
        self.embeddings_list = []
    
    def reset_index(self): 
        '''Removes all embeddings from index and clears state'''
        self.vector_index = voyager.Index(voyager.Space.Cosine, num_dimensions=self.n_dimensions)
        self.idx_to_name = []
        self.embeddings_list = []

    def add_item(self, name: str, new_embedding):
        '''Adds a new entry in the index if name is new. Else updates the previous entry with new embedding'''
        if name in self.idx_to_name:
            idx = self.idx_to_name.index(name)
            self.embeddings_list[idx] = new_embedding
        else:
            self.idx_to_name.append(name)
            self.embeddings_list.append(new_embedding)

        self.vector_index = voyager.Index(voyager.Space.Cosine, num_dimensions=self.n_dimensions)
        self.vector_index.add_items(self.embeddings_list)
        
    def add_items(self, names: list[str], embeddings):
        '''Replaces the contents of the index with new names and embeddings'''
        self.reset_index()
        self.vector_index.add_items(embeddings)
        self.idx_to_name = names
        self.embeddings_list = embeddings

    def remove_item(self, name):
        '''Removes an item from the index by name'''
        idx = self.idx_to_name.index(name)
        self.idx_to_name.pop(idx)
        self.embeddings_list.pop(idx)

        self.vector_index = voyager.Index(voyager.Space.Cosine, num_dimensions=self.n_dimensions)
        self.vector_index.add_items(self.embeddings_list)

    def query(self, embedding, k=1):
        return self.vector_index.query(embedding, k)
    
    def size(self):
        return len(self.idx_to_name)
    
    def get_name(self, idx: int):
        return self.idx_to_name[idx]


class FREngine:
    '''
    Facial recognition engine using VideoPlayer and EmbeddingsIndex. 
    Supports interactive features for adding/removing faces.
    '''

    def __init__(self, videoplayer: VideoPlayer, inference_width: int, inference_height: int) -> None:
        # This is the config for the stream
        self.videoplayer = videoplayer
        self.width = videoplayer.width
        self.height = videoplayer.height
        self.fps = videoplayer.fps

        # This is the size that images are rescaled to before inference is run
        self.INFERENCE_WIDTH = inference_width
        self.INFERENCE_HEIGHT = inference_height

        # Load other settings
        self.fr_settings = self._load_settings()
        self.perf_interval: float = 5.0
        self.last_perf: PerfLog | None = None

        # Set logging level
        logging.getLogger('insightface').setLevel(logging.ERROR)
        os.environ['ORT_LOGGING_LEVEL'] = '3'

        # Initialize and prepare model
        provider = "CUDAExecutionProvider" if is_cuda_available() else "CPUExecutionProvider"
        self.model = FaceAnalysis(
            name="buffalo_l",
            providers=[provider],
            allowed_modules=["detection", "recognition"],
        )
        self.model.prepare(
            ctx_id=0, 
            det_size=(self.INFERENCE_WIDTH, self.INFERENCE_HEIGHT), 
            det_thresh=0.5,
        )

        # Index state
        self.embedding_index = VoyagerEmbeddingIndex(n_dimensions=512)

        # Inference
        self.inference_lock = threading.Lock()
        self.inferenceThread = None
        self.fr_results = []

        # Capture 
        self.capture_lock = threading.Lock()
        self.latest_target_frame: np.ndarray | None = None
        self.latest_target_detection: dict | None = None
        
        # Other state
        self.persisted_detections: deque[FRResult] = deque(maxlen=self.fr_settings["q_max_size"])
        self.embeddings_loaded = False
        self.embeddings_loading = False

        log_info(f"FR Model initialised! Input: {self.width}x{self.height}, {self.fps}fps; Inference: {inference_width}x{inference_height}")

    
    # ───────────────────────────── Settings and Embeddings ─────────────────────────────────

    def adjust_values(self, new_settings: FRSettings) -> FRSettings:
        '''Adjust settings to new_settings'''

        previous_settings = self.fr_settings or self._load_settings()
        self.fr_settings = {**previous_settings, **new_settings} # Merge new_settings with previous
        with open(FR_SETTINGS_PATH, 'w') as f:
            json.dump(self.fr_settings, f)
        return self.fr_settings
    
    def load_embeddings(self) -> None:
        '''Resets index, reloads from cache (recomputes cache if not found)'''
        self.embeddings_loading = True
        self.embeddings_loaded = False

        try:
            self.embedding_index.reset_index()
            self.persisted_detections.clear()

            self._load_captures()

            self.embeddings_loading = False
            self.embeddings_loaded = True
            
        except:
            self.embeddings_loading = False
            self.embeddings_loaded = False

            log_info("Failed to load embeddings!")

    def _load_settings(self) -> FRSettings:
        ''''
        Returns settings from FR_SETTINGS_PATH json file
        If not specified, uses default values and writes back
        '''
        
        defaults: FRSettings = FR_DEFAULT_SETTINGS
        if os.path.exists(FR_SETTINGS_PATH):
            with open(FR_SETTINGS_PATH) as f:
                saved = json.load(f)
            for k, v in defaults.items():
                defaults[k] = saved.get(k, v)
        
        # Write back to replace missing entries with default
        with open(FR_SETTINGS_PATH, 'w') as f:
            json.dump(defaults, f)
        return defaults


    # ──────────────────────────── Inference ────────────────────────────────

    def start_inference(self) -> None:
        '''Starts inference loop in another thread'''
        with self.inference_lock:
            self.fr_results = []

        self.inferenceThread = threading.Thread(target=self._loop_inference, daemon=True)
        self.inferenceThread.start()

    def start_detection_broadcast(self) -> Generator[str, None, None]:
        '''Generator for detection broadcast (for /frResults)'''
        last_send_time = 0
        while self.inferenceThread and self.inferenceThread.is_alive():
            # Cap rate of broadcast to MAX_BROADCAST_FPS
            now = time.monotonic()
            if now - last_send_time < 1 / self.fr_settings["max_broadcast_fps"]:
                time.sleep(0.005)
                continue
            
            with self.inference_lock:
                results = self.fr_results

            filtered_results = [
                {
                    "label": r["label"],
                    "bbox": r["bbox"],
                    "score": r["score"],
                    "last_seen": r["last_seen"],
                    "is_target": r["is_target"],
                }
                for r in results
            ]
            last_send_time = now
            yield json.dumps({"data": filtered_results}) + '\n'

    def cleanup(self, force_exit=False):
        '''Sets event to trigger termination of inference and streaming processees'''
        log_info("CLEANING UP...")

        # Reset index, state
        self.embedding_index.reset_index()
        self.persisted_detections.clear()
        self.embeddings_loaded = False
        self.embeddings_loading = False

        # Stop VideoPlayer
        self.videoplayer.end_stream()

        # Join threads briefly to allow clean exit
        try:
            if self.videoplayer.streamThread and self.videoplayer.streamThread.is_alive():
                self.videoplayer.streamThread.join(timeout=1)
            if self.inferenceThread and self.inferenceThread.is_alive():
                self.inferenceThread.join(timeout=1)
        except Exception:
            pass

        if force_exit:
            log_info("Exiting...")
            try:
                sys.exit(0)
            except SystemExit:
                # In case sys.exit is swallowed by threads, force exit
                os._exit(0)

    def _loop_inference(self) -> None:
        '''Thread to continuously run inference on latest frames'''

        last_id, frame_cnt = -1, 0

        # Perf logging
        last_perf_log = time.monotonic()
        infer_calls = 0
        skipped_same_frame = 0
        skipped_for_performance = 0
        total_infer_time = 0.0
        total_match_time = 0.0
        total_remaining_time = 0.0

        while not self.videoplayer.end_event.is_set():
            # If stream is dead, exit inference loop
            if not self.videoplayer.streamThread or not self.videoplayer.streamThread.is_alive():
                break

            try:
                with self.videoplayer.vid_lock.read_lock():
                    curr_id, frame = self.videoplayer.frame_id, self.videoplayer.current_frame

                # Skip inference if repeated frame, wait for a short time
                if curr_id == last_id:
                    skipped_same_frame += 1
                    time.sleep(0.005)
                    continue

                last_id = curr_id
                frame_cnt += 1
                
                # Enforced frame skipping
                skip = self.fr_settings["frame_skip"]
                if skip > 1 and frame_cnt % skip != 0:
                    skipped_for_performance += 1
                    continue
                
                # Call infer(), log inference time
                results, infer_time, match_time, remaining_time = self._infer(frame)
                total_infer_time += infer_time
                total_match_time += match_time
                total_remaining_time += remaining_time
                infer_calls += 1

                # Write inference results
                with self.inference_lock:
                    self.fr_results = results

                if frame_cnt == 1:
                    log_info("Successfully performed inference on first RGB frame")

                # Write perf log if it has been perf_interval_s since the last log
                now = time.monotonic()
                if self.fr_settings["perf_logging"] and (now - last_perf_log) >= self.perf_interval:
                    interval = max(now - last_perf_log, 1e-9)
                    fps_infer = infer_calls / interval
                    avg_infer_ms = (total_infer_time / max(infer_calls, 1)) * 1000.0
                    avg_match_ms = (total_match_time / max(infer_calls, 1)) * 1000.0
                    avg_remaining_ms = (total_remaining_time / max(infer_calls, 1)) * 1000.0
                    avg_ms = avg_infer_ms + avg_match_ms + avg_remaining_ms
                    skip_frac = (skipped_same_frame + skipped_for_performance) / (skipped_same_frame + skipped_for_performance + infer_calls)
                    
                    log_info(f"[PERF] infer:{fps_infer:.1f}fps|{avg_ms:.0f}ms ({avg_infer_ms:.1f}|{avg_match_ms:.1f}|{avg_remaining_ms:.1f}) skip:{skip_frac*100:.0f}%")
                    self.last_perf = {
                        "fps": fps_infer,
                        "avg_ms": avg_ms,
                        "infer_ms": avg_infer_ms,
                        "match_ms": avg_match_ms,
                        "remaining_ms": avg_remaining_ms,
                        "skip_frac": skip_frac
                    }

                    last_perf_log = now
                    infer_calls = 0
                    skipped_same_frame = 0
                    skipped_for_performance = 0
                    total_infer_time = 0.0
                    total_match_time = 0.0
                    total_remaining_time = 0.0

            except Exception as e:
                log_info(f"Inference loop error: {e}")
        
        # When loop breaks, reset index and state
        self.embedding_index.reset_index()
        self.persisted_detections.clear()
    
    def _infer(self, frame: np.ndarray) -> list[FRResult]:
        '''Run inference on a single frame. Returns a list of FRResult, with time taken'''

        t0 = time.monotonic()

        if frame is None:
            return []

        try:
            # Read frame, pass through model
            resized_frame = cv2.resize(frame, (self.INFERENCE_WIDTH, self.INFERENCE_HEIGHT))
            preds = self.model.get(resized_frame)
            preds.sort(key=lambda x: x.det_score)
            preds = preds[:self.fr_settings["max_detections"]] # Limit to top few detections

            t1 = time.monotonic() # t1-t0 is the time to get predictions from model

            # No current detections; return
            if not preds or len(preds) == 0:
                return [], 0.0, 0.0, 0.0
            
            embeddings = [y.normed_embedding for y in preds]
            bboxes = [self._frac_bbox(self.INFERENCE_WIDTH, self.INFERENCE_HEIGHT, y.bbox.tolist()) for y in preds]

            if self.embedding_index.size() == 0:
                labels = ["Unknown"] * len(preds)
                dists = [[1.0]] * len(preds)
            else:
                k = min(2 if self.fr_settings["use_differentiator"] else 1, self.embedding_index.size())
                neighbors, dists = self.embedding_index.query(embeddings, k=k)
                labels = self._match_labels(embeddings, neighbors, dists, bboxes)

            t2 = time.monotonic() # t2-t1 is the time to match labels

            # Update target
            target_idx = self._update_capture_target(frame, embeddings, labels, bboxes)
            
            current_detections = [
                FRResult({
                    "label": labels[i],
                    "bbox": bboxes[i],
                    "norm_embed": embeddings[i], 
                    "score": float(dists[i][0]),
                    "last_seen": time.monotonic(),
                    "is_target": i == target_idx
                })
                for i in range(len(preds))
            ]

            # Persist non-unknown detections
            for r in current_detections:
                if r["label"] != "Unknown":
                    self.persisted_detections.append(r)

            t3 = time.monotonic() # t3-t2 is the time to do remaining tasks: persist, update target

            return current_detections, t1-t0, t2-t1, t3-t2

        except Exception as e:
            log_info(f"Infer error: {e}\n{traceback.format_exc()}")
            return [], 0.0, 0.0, 0.0
        
    def _match_labels(self, embeddings, neighbors, dists, bboxes) -> list[str]:
        '''Matches detections to embeddings, optionally using the differentiator and persistor mechanics'''
        labels = []
        s = self.fr_settings
        for i, dist in enumerate(dists):
            match = None
            if dist[0] < s["threshold"]:
                match = self.embedding_index.get_name(neighbors[i][0])
            elif s["use_differentiator"] and dist[0] < s["threshold_lenient_diff"] and len(dist) > 1 and (dist[1] - dist[0]) > s["similarity_gap"]:
                match = self.embedding_index.get_name(neighbors[i][0])
            elif s["use_persistor"]:
                match = self._catch_recent(embeddings[i], dist[0], bboxes[i])
            labels.append(match or "Unknown")
        return labels

    def _catch_recent(self, emb_norm: np.ndarray, score: float, bbox: list[float]) -> str | None:
        '''Implements the persistor mechanic'''
        
        for r in self.persisted_detections:
            # Enforce lenient score 
            if score > self.fr_settings["threshold_lenient_pers"]:
                continue

            # Enforce IOU similarity
            IOU = calc_iou(r["bbox"], bbox)
            if IOU < self.fr_settings["threshold_iou"]:
                continue
            
            # Enforce embedding similarity
            sim = float(np.dot(emb_norm, r["norm_embed"]))
            if sim < self.fr_settings["threshold_sim"]:
                continue

            return r["label"]
            # log_info(f"Persistor hit! Score: {score}; IOU: {IOU}; Sim: {sim}")

        return None

    @staticmethod
    def _norm(emb: np.ndarray) -> np.ndarray:
        emb = np.asarray(emb, dtype=np.float32)
        n = np.linalg.norm(emb)
        return emb / n if n else emb

    @staticmethod
    def _frac_bbox(w: int, h: int, bbox: list[float]) -> list[float]:
        return [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]


    # ───────────────────────────── Capture ─────────────────────────────────
    
    def capture_unknown(self, name: str, allow_duplicate: bool=False, target_image=None) -> dict:
        '''
        Captures the latest target, saves image and updates the vector index.
        If name is in database and allow_duplicate=False, returns target crop for retry.
        If target_image specified, uses that instead of latest target detection.
        '''
        name = (name or "").strip()
        if not name or name == "":
            return {"ok": False, "message": "Name is required."}
        
        # Process name: remove spaces, non-legal characters, convert to uppercase
        safe_name = "".join(c for c in name if c.isalnum() or c in "_- ").strip().replace(" ", "_").upper()
        data_dir = os.path.join("data", "captures", safe_name)
        current_names = os.listdir(os.path.join("data", "captures"))

        # If target_image not provided, store snapshot of current target using latest_target_detection
        crop = target_image

        if crop is None:
            with self.capture_lock:
                detection, frame = self.latest_target_detection, self.latest_target_frame
            if detection is None or frame is None:
                return {"ok": False, "message": "No target face available."}
            crop = self._crop_image(np.asarray(frame), detection["bbox"]) # convert back from memoryview to np array first!

        # Check if name is already in database. Block if allow_duplicate=False
        if safe_name in current_names:
            if not allow_duplicate:
                return {"ok": False, "message": f"{safe_name} is already in database!", "crop": crop}
        else:
            os.makedirs(data_dir, exist_ok=True)
        
        try:
            # Save image
            img_path = os.path.join(data_dir, f"{safe_name}_{datetime.now():%Y%m%d_%H%M%S}.jpg")
            crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR) # convert to BGR before using cv2.imwrite
            cv2.imwrite(img_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 100])

            # Recompute cached embedding for that person
            updated_emb = self._recompute_cached_embedding(data_dir)
            
            # Check if no face detected!
            if updated_emb is None:
                log_info(f"Capture failed for {safe_name}. No face detected!")
                os.remove(img_path)
                return {"ok": False, "message": "No face detected!"}
            
            # Update vector index
            self.embedding_index.add_item(safe_name, updated_emb)

            log_info(f"Captured face for {safe_name}")
            return {"ok": True, "message": f"Captured {safe_name} successfully."}

        except Exception as e:
            log_info(f"Capture failed for {safe_name}: {e}")
            return {"ok": False, "message": "Failed to save capture."}
    
    def remove_capture_image(self, image_path: str) -> dict:
        '''Delete image then refresh cached + in-memory embedding'''

        relative_path = image_path.lstrip("/\\")
        captures_root = os.path.realpath(os.path.join(os.getcwd(), "data", "captures"))
        target_path = os.path.realpath(os.path.join(captures_root, relative_path))
        person_dir = os.path.dirname(target_path)
        name = os.path.basename(person_dir).strip().upper()

        if not (target_path == captures_root or target_path.startswith(captures_root + os.sep)):
            return {"ok": False, "message": "Invalid image_path"}
        if not os.path.isfile(target_path):
            return {"ok": False, "message": "Image not found"}

        # Delete image and recompute embedding for that person
        os.remove(target_path)
        emb = self._recompute_cached_embedding(person_dir)

        # Check if there is no more embedding for the person 
        if emb is None or not emb.size:
            # Person should be completely removed
            self.embedding_index.remove_item(name)
        else:
            # Image removed but others still exist; update embeddings
            self.embedding_index.add_item(name, emb)

        log_info(f"Removed image: {image_path}")
        return {"ok": True, "message": "Success!"}

    def _crop_image(self, frame: np.ndarray, bbox: list[float], margin: float=0.7):
        '''Crops a frame based on the bbox, leaving a margin'''

        l = max(int(bbox[0] * self.width), 0)
        t = max(int(bbox[1] * self.height), 0)
        r = min(int(bbox[2] * self.width), self.width)
        b = min(int(bbox[3] * self.height), self.height)

        # Expand bbox by margin while staying within frame
        bw, bh = max(r - l, 0), max(b - t, 0)
        pad_x = int(bw * margin)
        pad_y = int(bh * margin)
        l = max(l - pad_x, 0)
        r = min(r + pad_x, self.width)
        t = max(t - pad_y, 0)
        b = min(b + pad_y, self.height)

        crop = frame[t:b, l:r]

        return crop

    def _update_capture_target(self, frame: np.ndarray, embeddings: list, labels: list[str], bboxes: list) -> int | None:
        '''
        Chooses the unknown detection with greatest area as target. 
        Updates latest_target_frame and latest_target_detection for use by capture_unknown
        Returns target_idx (or None if no unknown detections are found)
        '''
        target_idx = None
        max_area = -1
        for i, label in enumerate(labels):
            if label != "Unknown" or i >= len(bboxes):
                continue
            box = bboxes[i]
            area = max(0, (box[2] - box[0]) * (box[3] - box[1]))
            if area > max_area:
                max_area, target_idx = area, i

        if target_idx is not None and target_idx < len(embeddings):
            with self.capture_lock:
                self.latest_target_frame = memoryview(frame)
                self.latest_target_detection = {
                    "bbox": bboxes[target_idx], 
                    "embedding": np.asarray(embeddings[target_idx], dtype=np.float32)
                }

        return target_idx
    
    def _load_captures(self) -> int:
        '''Reload all images from cache, update index (recompute cache file if not found)'''

        root = os.path.join("data", "captures")
        if not os.path.isdir(root):
            return 0

        added = 0
        scanned = 0

        for name in sorted(os.listdir(root)):
            person_dir = os.path.join(root, name)
            if not os.path.isdir(person_dir):
                continue

            images = self._list_capture_images(person_dir)
            scanned += len(images)

            if not images:
                _ = self._recompute_cached_embedding(person_dir) # run this to delete the folder
                continue

            # Prefer cached average; recompute if missing
            emb = self._load_cached_embedding(person_dir)
            if emb is None:
                emb = self._recompute_cached_embedding(person_dir)

            if emb is not None and emb.size:
                self.embedding_index.add_item(name, emb)
                added += 1

        log_info(f"Loaded {added} capture embeddings from {scanned} images.")
        return added

    def _get_avg_embedding(self, folder: str) -> np.ndarray:
        embeddings = []
        images = self._list_capture_images(folder)

        for img_name in images:
            try:
                img_path = os.path.join(folder, img_name)

                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB after using cv2.imread
                faces = self.model.get(img)

                if faces:
                    embeddings.append(faces[0].normed_embedding)
                else:
                    log_info(f"No face detected from {img_name}.")

            except Exception as e:
                log_info(f"Capture image load failed: {img_name} ({e})")
        
        return np.mean(embeddings, axis=0) if embeddings else np.array([])

    def _load_cached_embedding(self, folder: str) -> np.ndarray | None:
        cache_path = self._embedding_cache_path(folder)
        if not os.path.exists(cache_path):
            return None
        try:
            return np.load(cache_path)
        except Exception as e:
            log_info(f"Embedding cache load failed: {cache_path} ({e})")
            return None

    def _recompute_cached_embedding(self, folder: str) -> np.ndarray | None:
        '''
        Recalculates average embedding from images in folder and updates cache. 
        Returns the embedding if no issues arise, else returns None
        '''
        # Remove cache file and folder if no images
        images = self._list_capture_images(folder)
        if not images:
            cache_path = self._embedding_cache_path(folder)
            if os.path.exists(cache_path):
                os.remove(cache_path)
            os.rmdir(folder)
            return None

        emb = self._get_avg_embedding(folder)
        if emb.size:
            try:
                np.save(self._embedding_cache_path(folder), emb)
            except Exception as e:
                log_info(f"Embedding cache save failed: {folder} ({e})")
            return emb

        log_info(f"Embedding cache save failed: no embeddings found in {folder}!")
        return None

    @staticmethod
    def _embedding_cache_path(folder: str) -> str:
        return os.path.join(folder, "embedding_avg.npy")

    @staticmethod
    def _count_capture_images(folder: str) -> int:
        try:
            return len([i for i in os.listdir(folder) if i.lower().endswith((".jpg", ".jpeg", ".png"))])
        except Exception:
            return 0

    @staticmethod
    def _list_capture_images(folder: str) -> list[str]:
        return [
            i for i in os.listdir(folder)
            if i.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
