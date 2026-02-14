import os
import sys
import threading
import time
import traceback
import warnings
import json
from datetime import datetime, timedelta
from typing import Generator, TypedDict
import subprocess
import logging
from collections import deque
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from voyager import Index, Space

from fr import VideoPlayer
from utils import calc_iou, log_info

logging.getLogger('insightface').setLevel(logging.ERROR)
os.environ['ORT_LOGGING_LEVEL'] = '3'

FR_SETTINGS_PATH = 'settings.json'
FR_DEFAULT_SETTINGS = {
    "threshold": 0.45, 
    "holding_time": 2, 
    "max_detections": 50, 
    "perf_logging": False, 
    "frame_skip": 1, 
    "max_broadcast_fps": 50,

    "use_differentiator": True, 
    "threshold_lenient_diff": 0.55,
    "similarity_gap": 0.10, 

    "use_persistor": True, 
    "q_max_size": 100,
    "threshold_prev": 0.4,
    "threshold_iou": 0.7, 
    "threshold_lenient_pers": 0.60, 
}

def is_cuda_available() -> bool:
    try:
        subprocess.check_output(['nvidia-smi'])
        return True
    except Exception:
        return False


class FRResult(TypedDict, total=False):
    bbox: list[float]
    label: str
    score: float
    last_seen: float
    is_target: bool


class RecentDetection(TypedDict):
    name: str
    bbox: list[float]
    norm_embed: np.ndarray
    last_seen: float


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
    threshold_prev: float
    threshold_iou: float
    threshold_lenient_pers: float
    frame_skip: int
    max_broadcast_fps: int


class FRVidPlayer(VideoPlayer):
    '''Facial recognition video player using InsightFace and Voyager vector index.'''

    def __init__(self, width: int, height: int, fps: int, inference_width: int, inference_height: int) -> None:
        # This is the size of self.frame (which is broadcast)
        super().__init__(width=width, height=height, fps=fps)
        self.fr_settings = self._load_settings()

        warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface\.utils\.transform")
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface\.utils\.face_align")

        provider = "CUDAExecutionProvider" if is_cuda_available() else "CPUExecutionProvider"
        self.model = FaceAnalysis(
            name="buffalo_l",
            providers=[provider],
            allowed_modules=["detection", "recognition"],
        )

        # This is the size that images are rescaled to before inference is run
        self.INFERENCE_WIDTH = inference_width
        self.INFERENCE_HEIGHT = inference_height
        self.model.prepare(
            ctx_id=0, 
            det_size=(inference_width, inference_height), 
            det_thresh=0.5,
        )

        # Index state
        self.vector_index = Index(Space.Cosine, num_dimensions=512)
        self.idx_to_name = []
        self.embeddings_list = []
        
        # Other state
        self.old_detections: deque[RecentDetection] = deque(maxlen=self.fr_settings["q_max_size"])
        self.embeddings_loaded = False
        self.embeddings_loading = False

        # Capture 
        self.capture_lock = threading.Lock()
        self.latest_frame_bytes: bytes | None = None
        self.latest_target_detection: dict | None = None

        # Inference
        self.inference_lock = threading.Lock()
        self.inferenceThread = None
        self.fr_results = []

        log_info(f"FR Model initialised! Input: {width}x{height}, {fps}fps; Inference: {inference_width}x{inference_height}")

    def adjust_values(self, new_settings: FRSettings) -> FRSettings:
        '''Adjust settings to new_settings'''

        previous_settings = self.fr_settings or self._load_settings()
        self.fr_settings = {**previous_settings, **new_settings} # Merge new_settings with previous
        with open(FR_SETTINGS_PATH, 'w') as f:
            json.dump(self.fr_settings, f)
        return self.fr_settings
    
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


    # ─────────────────────────── Index Management ────────────────────────────
    def load_embeddings(self) -> None:
        '''Resets index, reloads from cache (recomputes cache if not found)'''
        self.embeddings_loading = True
        self.embeddings_loaded = False

        try:
            self._reset_index()
            self.old_detections: deque[RecentDetection] = deque(maxlen=self.fr_settings["q_max_size"])
            self._load_captures()

            self.embeddings_loading = False
            self.embeddings_loaded = True
            
        except:
            self.embeddings_loading = False
            self.embeddings_loaded = False

            log_info("Failed to load embeddings!")

    def _reset_index(self) -> None:
        self.idx_to_name = []
        self.embeddings_list = []
        self.vector_index = Index(Space.Cosine, num_dimensions=512)

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
                self.idx_to_name.append(name)
                self.embeddings_list.append(emb)
                self.vector_index.add_item(emb)
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
    
    # ───────────────────────────── Capture ─────────────────────────────────
    def capture_unknown(self, name: str) -> dict:
        '''Captures the latest target, saves image and updates the vector index'''

        name = (name or "").strip()
        if not name or name == "":
            return {"ok": False, "message": "Name is required."}

        with self.capture_lock:
            detection, frame = self.latest_target_detection, self.latest_frame_bytes

        if not detection or not frame:
            return {"ok": False, "message": "No target face available."}
        
        # Process name: remove spaces, non-legal characters, converts to uppercase
        safe_name = "".join(c for c in name if c.isalnum() or c in "_- ").strip().replace(" ", "_").upper()
        data_dir = os.path.join("data", "captures", safe_name)
        os.makedirs(data_dir, exist_ok=True)

        try:
            self._save_capture_files(safe_name, data_dir, frame, detection["bbox"])
            self._recompute_cached_embedding(data_dir)
            updated_emb = self._load_cached_embedding(data_dir)
            
            if safe_name in self.idx_to_name:
                idx = self.idx_to_name.index(safe_name)
                self.embeddings_list[idx] = updated_emb
                self.vector_index.add_item(updated_emb, idx)
            else:
                self.idx_to_name.append(safe_name)
                self.embeddings_list.append(updated_emb)
                self.vector_index.add_item(updated_emb)

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
        display_name = os.path.basename(person_dir).strip().upper()

        if not (target_path == captures_root or target_path.startswith(captures_root + os.sep)):
            return {"ok": False, "message": "Invalid image_path"}

        if not os.path.isfile(target_path):
            return {"ok": False, "message": "Image not found"}

        os.remove(target_path)

        emb = self._recompute_cached_embedding(person_dir)
        idx = self.idx_to_name.index(display_name)

        if emb is None or not emb.size:
            # Person has been completely removedl index needs to be reset
            self.idx_to_name.pop(idx)
            self.embeddings_list.pop(idx)
            self.vector_index = Index(Space.Cosine, num_dimensions=512)
            self.vector_index.add_items(self.embeddings_list)

        else:
            # Image removed but others still exist; update embeddings
            self.embeddings_list[idx] = emb
            self.vector_index.add_item(emb, idx)

        log_info(f"Removed image: {image_path}")
        return {"ok": True, "message": "Success!"}


    def _save_capture_files(self, name: str, folder: str, frame: bytes, bbox: list[float]):
        # Save snapshot
        img = np.frombuffer(frame, dtype=np.uint8).reshape(
            (self.height, self.width, 3)
        )

        l = max(int(bbox[0] * self.width), 0)
        t = max(int(bbox[1] * self.height), 0)
        r = min(int(bbox[2] * self.width), self.width)
        b = min(int(bbox[3] * self.height), self.height)

        # Expand bbox (70%) while staying within frame
        bw, bh = max(r - l, 0), max(b - t, 0)
        pad_x = int(bw * 0.7)
        pad_y = int(bh * 0.7)
        l = max(l - pad_x, 0)
        r = min(r + pad_x, self.width)
        t = max(t - pad_y, 0)
        b = min(b + pad_y, self.height)

        crop = img[t:b, l:r]
        crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR) # convert to BGR before using cv2.imwrite
        img_path = os.path.join(folder, f"{name}_{datetime.now():%Y%m%d_%H%M%S}.jpg")
        cv2.imwrite(img_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 100])

    def _update_capture_target(self, frame: bytes, embeddings: list, labels: list[str], bboxes: list) -> int | None:
        target_idx = None
        max_area = -1
        for i, label in enumerate(labels):
            if label != "Unknown" or i >= len(bboxes):
                continue
            box = bboxes[i]
            area = max(0, (box[2] - box[0]) * (box[3] - box[1]))
            if area > max_area:
                max_area, target_idx = area, i

        with self.capture_lock:
            self.latest_frame_bytes = memoryview(frame)
            self.latest_target_detection = (
                {
                    "bbox": bboxes[target_idx], 
                    "embedding": np.asarray(embeddings[target_idx], dtype=np.float32)
                }
                if target_idx is not None and target_idx < len(embeddings) else None
            )
        return target_idx

    # ──────────────────────────── Inference ────────────────────────────────

    def infer(self, frame_bytes: bytes) -> list[FRResult]:
        if not frame_bytes:
            return []

        try:
            # Read frame, run model
            frame = np.frombuffer(frame_bytes, np.uint8).reshape(
                (self.height, self.width, 3)
            )
            resized_frame = cv2.resize(frame, (self.INFERENCE_WIDTH, self.INFERENCE_HEIGHT))
            faces = self.model.get(resized_frame)
            faces.sort(key=lambda x: x.det_score)
            faces = faces[:self.fr_settings["max_detections"]] # Limit to top few detections

            # No current detections; update and return
            if not faces or len(faces) == 0:
                self._update_old_detections([])
                return []
            
            embeddings = [f.normed_embedding for f in faces]
            bboxes = [self._frac_bbox(self.INFERENCE_WIDTH, self.INFERENCE_HEIGHT, f.bbox.tolist()) for f in faces]

            if not self.idx_to_name:
                labels = ["Unknown"] * len(faces)
                dists = [[1.0]] * len(faces)
            else:
                k = min(2 if self.fr_settings["use_differentiator"] else 1, len(self.idx_to_name))
                neighbors, dists = self.vector_index.query(embeddings, k=k)
                labels = self._match_labels(embeddings, neighbors, dists, bboxes)

            target_idx = self._update_capture_target(frame_bytes, embeddings, labels, bboxes)
            
            # Store all non-unknown detections
            self._update_old_detections([
                {
                    "name": labels[i], 
                    "bbox": bboxes[i], 
                    "norm_embed": embeddings[i], 
                    "last_seen": time.monotonic()
                }
                for i in range(len(faces)) if labels[i] != "Unknown"
            ])

            # Return a list of FRResult
            return [
                {
                    "bbox": bboxes[i],
                    "label": labels[i],
                    "score": float(dists[i][0]),
                    "last_seen": time.monotonic(),
                    "is_target": i == target_idx
                }
                for i in range(len(faces))
            ]

        except Exception as e:
            log_info(f"Infer error: {e}\n{traceback.format_exc()}")
            return []

    def _match_labels(self, embeddings, neighbors, dists, bboxes) -> list[str]:
        '''Matches detections to embeddings, optionally using the differentiator and persistor mechanics'''
        labels = []
        s = self.fr_settings
        for i, dist in enumerate(dists):
            match = None
            if dist[0] < s["threshold"]:
                match = self.idx_to_name[neighbors[i][0]]
            elif s["use_differentiator"] and dist[0] < s["threshold_lenient_diff"] and len(dist) > 1 and (dist[1] - dist[0]) > s["similarity_gap"]:
                match = self.idx_to_name[neighbors[i][0]]
            elif s["use_persistor"]:
                match = self._catch_recent(embeddings[i], dist[0], bboxes[i])
            labels.append(match or "Unknown")
        return labels

    def _catch_recent(self, emb_norm: np.ndarray, score: float, bbox: list[float]) -> str | None:
        '''Implements the persistor mechanic'''
        
        best_sim, match = 0, None
        for r in self.old_detections:
            if calc_iou(r["bbox"], bbox) < self.fr_settings["threshold_iou"]:
                continue
            sim = float(np.dot(emb_norm, r["norm_embed"]))
            if sim > best_sim:
                best_sim, match = sim, r["name"]

        if best_sim > (1 - self.fr_settings["threshold_prev"]) and score < self.fr_settings["threshold_lenient_pers"]:
            # log_info(f"Persistor hit! {score}")
            return match
        return None

    def _update_old_detections(self, current_detections: list[RecentDetection]) -> None:
        '''
        Updates old_detections based on the latest detections
        '''
        for r in current_detections:
            self.old_detections.append(r)

    @staticmethod
    def _norm(emb: np.ndarray) -> np.ndarray:
        emb = np.asarray(emb, dtype=np.float32)
        n = np.linalg.norm(emb)
        return emb / n if n else emb

    @staticmethod
    def _frac_bbox(w: int, h: int, bbox: list[float]) -> list[float]:
        return [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]

    # ──────────────────────── Inference Thread ─────────────────────────────

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
            
            last_send_time = now
            yield json.dumps({"data": results}) + '\n'

    def cleanup(self, *_args) -> None:
        '''Sets event to trigger termination of inference and streaming processees'''

        log_info("CLEANING UP...")

        self.end_stream()

        # Join thread briefly to allow clean exitW
        try:
            if self.streamThread and self.streamThread.is_alive():
                self.streamThread.join(timeout=1)
            if self.inferenceThread and self.inferenceThread.is_alive():
                self.inferenceThread.join(timeout=1)
        except Exception:
            pass

        self._reset_index()
        self.old_detections: deque[RecentDetection] = deque(maxlen=self.fr_settings["q_max_size"])

        force_exit = bool(_args)
        if not force_exit:
            return None

        time.sleep(0.5)

        # Final safety: exit if anything lingers
        try:
            sys.exit(0)
        except SystemExit:
            # In case sys.exit is swallowed by threads, force exit
            import os
            os._exit(0)

        return None
    

    def _loop_inference(self) -> None:
        '''Thread to continuously run inference on latest frames'''

        last_id, frame_cnt = -1, 0

        # Perf logging
        perf_interval_s = 5.0
        last_perf_log = time.monotonic()
        infer_calls = 0
        infer_time_s_sum = 0.0
        skipped_same_frame = 0
        skipped_for_performance = 0

        while not self.end_event.is_set():
            # If stream is dead, exit inference loop
            if not self.streamThread or not self.streamThread.is_alive():
                break

            try:
                with self.vid_lock.read_lock():
                    curr_id, frame = self.frame_id, self.frame_bytes

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
                if self.fr_settings["perf_logging"]:
                    t0 = time.monotonic()
                    results = self.infer(frame)
                    infer_time_s_sum += (time.monotonic() - t0)
                    infer_calls += 1
                else:
                    results = self.infer(frame)

                # Write inference results
                with self.inference_lock:
                    self.fr_results = results

                if frame_cnt == 1:
                    log_info("Successfully performed inference on first RGB frame")

                # Write perf log if it has been perf_interval_s since the last log
                now = time.monotonic()
                if self.fr_settings["perf_logging"] and (now - last_perf_log) >= perf_interval_s:
                    interval = max(now - last_perf_log, 1e-9)
                    fps_infer = infer_calls / interval
                    avg_ms = (infer_time_s_sum / max(infer_calls, 1)) * 1000.0
                    skip_val = self.fr_settings["frame_skip"]
                    wait_ratio = skipped_same_frame / max(skipped_same_frame + infer_calls, 1)
                    log_info(f"[PERF] infer:{fps_infer:.1f}fps/{avg_ms:.0f}ms skip:{skip_val} wait:{wait_ratio:.1%}")
                    last_perf_log = now
                    infer_calls = 0
                    infer_time_s_sum = 0.0
                    skipped_same_frame = 0
                    skipped_for_performance = 0

            except Exception as e:
                log_info(f"Inference loop error: {e}")
        
        # When loop breaks, reset index and state
        self._reset_index()
        self.old_detections: deque[RecentDetection] = deque(maxlen=self.fr_settings["q_max_size"])