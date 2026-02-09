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
import numpy as np
from PIL import Image
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
    "use_brute_force": False,
    "perf_logging": False, 
    "frame_skip": 1, 
    "max_broadcast_fps": 50,

    "use_differentiator": True, 
    "threshold_lenient_diff": 0.55,
    "similarity_gap": 0.10, 

    "use_persistor": True, 
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
    bboxes: list[float]
    labels: str
    score: float
    is_target: bool


class RecentDetection(TypedDict):
    name: str
    bbox: list[float]
    norm_embed: np.ndarray
    last_seen: datetime


class FRSettings(TypedDict):
    threshold: float
    holding_time: int
    use_brute_force: bool
    perf_logging: bool
    use_differentiator: bool
    threshold_lenient_diff: float
    similarity_gap: float
    use_persistor: bool
    threshold_prev: float
    threshold_iou: float
    threshold_lenient_pers: float
    frame_skip: int
    max_broadcast_fps: int


class FRVidPlayer(VideoPlayer):
    """Facial recognition video player using InsightFace and Voyager vector index."""

    def __init__(self) -> None:
        super().__init__()
        self._suppress_warnings()

        provider = "CUDAExecutionProvider" if is_cuda_available() else "CPUExecutionProvider"
        self.model = FaceAnalysis(
            name="buffalo_l",
            providers=[provider],
            allowed_modules=["detection", "recognition"],
        )
        self.model.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

        self._reset_index()
        self.current_detections: list[RecentDetection] = []
        self.old_detections: list[RecentDetection] = []

        # Capture state
        self.capture_lock = threading.Lock()
        self.latest_frame_bytes: bytes | None = None
        self.latest_target: dict | None = None

        # Settings
        self.fr_settings = self._load_settings()
        self.embeddings_loaded = False
        self.embeddings_loading = False

        # Inference
        self.inference_lock = threading.Lock()
        self.inferenceThread = None
        self.fr_results = []

        log_info("FR Model initialised!")

    @staticmethod
    def _suppress_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface\.utils\.transform")
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface\.utils\.face_align")

    def _load_settings(self) -> FRSettings:
        ''''Loads settings from FR_SETTINGS_PATH json file, uses defaults and writes back if not specified'''
        
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

    def adjust_values(self, new_settings: FRSettings) -> FRSettings:
        previous_settings = self.fr_settings or self._load_settings()
        self.fr_settings = {**previous_settings, **new_settings} # Merge new_settings with previous
        with open(FR_SETTINGS_PATH, 'w') as f:
            json.dump(self.fr_settings, f)
        return self.fr_settings

    # ─────────────────────────── Index Management ────────────────────────────
    def load_embeddings(self) -> None:
        # Full reload from disk (captures + cache)
        self.embeddings_loading = True
        self.embeddings_loaded = False
        success = False
        try:
            self._reset_index()
            self.current_detections = []
            self.old_detections = []
            self._load_captures()
            success = True
        finally:
            self.embeddings_loading = False
            self.embeddings_loaded = success

    def _reset_index(self) -> None:
        self.name_list = []
        self.vector_index = Index(Space.Cosine, num_dimensions=512)
        self.embeddings_array = np.array([])
        self.db_embeddings_normalized = np.array([])

    def _add_embedding(self, name: str, embedding: np.ndarray) -> None:
        self.name_list.append(name)
        self.vector_index.add_item(embedding)
        self.embeddings_array = (
            np.array([embedding]) if self.embeddings_array.size == 0
            else np.vstack([self.embeddings_array, embedding])
        )
        norms = np.linalg.norm(self.embeddings_array, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.db_embeddings_normalized = self.embeddings_array / norms

    def _rebuild_index_from_embeddings(self) -> None:
        # Rebuild ANN index from in-memory embeddings
        self.vector_index = Index(Space.Cosine, num_dimensions=512)
        if self.embeddings_array.size == 0:
            self.db_embeddings_normalized = np.array([])
            return

        for emb in self.embeddings_array:
            self.vector_index.add_item(emb)

        norms = np.linalg.norm(self.embeddings_array, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.db_embeddings_normalized = self.embeddings_array / norms

    def _upsert_embedding(self, name: str, embedding: np.ndarray | None) -> None:
        # Insert/update/remove a single identity in-memory, then rebuild index
        if name in self.name_list:
            idx = self.name_list.index(name)
            if embedding is None or not embedding.size:
                self.name_list.pop(idx)
                if self.embeddings_array.size:
                    self.embeddings_array = np.delete(self.embeddings_array, idx, axis=0)
                    if self.embeddings_array.size == 0:
                        self.embeddings_array = np.array([])
            else:
                self.embeddings_array[idx] = embedding
        else:
            if embedding is None or not embedding.size:
                return
            self.name_list.append(name)
            self.embeddings_array = (
                np.array([embedding]) if self.embeddings_array.size == 0
                else np.vstack([self.embeddings_array, embedding])
            )

        self._rebuild_index_from_embeddings()

    def _load_captures(self) -> int:
        root = os.path.join("data", "captures")
        if not os.path.isdir(root):
            return 0

        added = 0
        scanned = 0

        for name in sorted(os.listdir(root)):
            person_dir = os.path.join(root, name)
            if not os.path.isdir(person_dir):
                continue

            display_name = str(name).strip().upper()
            images = self._list_capture_images(person_dir)
            scanned += len(images)

            if not images:
                continue

            # Prefer cached average; recompute if missing
            emb = self._load_cached_embedding(person_dir)
            if emb is None:
                emb = self._recompute_cached_embedding(person_dir, images)

            if emb is not None and emb.size:
                self._add_embedding(display_name, emb)
                added += 1

        log_info(f"Loaded {added} capture embeddings from {scanned} images.")
        return added

    def _avg_embedding(self, folder: str, images: list[str]) -> np.ndarray:
        embeddings = []
        no_face = 0
        for img_name in images:
            try:
                img_path = os.path.join(folder, img_name)
                img = Image.open(img_path).convert("RGB")
                faces = self.model.get(np.array(img))
                if faces:
                    embeddings.append(faces[0].normed_embedding)
                else:
                    no_face += 1
            except Exception as e:
                log_info(f"Capture image load failed: {img_name} ({e})")
        if no_face and not embeddings:
            log_info(f"No face detected in {no_face} capture image(s) from {folder}.")
        return np.mean(embeddings, axis=0) if embeddings else np.array([])

    @staticmethod
    def _embedding_cache_path(folder: str) -> str:
        return os.path.join(folder, "embedding_avg.npy")

    def _load_cached_embedding(self, folder: str) -> np.ndarray | None:
        cache_path = self._embedding_cache_path(folder)
        if not os.path.exists(cache_path):
            return None
        try:
            return np.load(cache_path)
        except Exception as e:
            log_info(f"Embedding cache load failed: {cache_path} ({e})")
            return None

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

    def _save_cached_embedding(self, folder: str, emb: np.ndarray) -> None:
        try:
            np.save(self._embedding_cache_path(folder), emb)
        except Exception as e:
            log_info(f"Embedding cache save failed: {folder} ({e})")

    def _recompute_cached_embedding(self, folder: str, images: list[str] | None = None) -> np.ndarray | None:
        # Recalculate average embedding from images and update cache
        images = images if images is not None else self._list_capture_images(folder)
        cache_path = self._embedding_cache_path(folder)

        if not images:
            if os.path.exists(cache_path):
                os.remove(cache_path)
            return None

        emb = self._avg_embedding(folder, images)
        if emb.size:
            self._save_cached_embedding(folder, emb)
            return emb

        if os.path.exists(cache_path):
            os.remove(cache_path)
        return None

    def remove_capture_image(self, image_path: str) -> dict:
        '''Delete image then refresh cached + in-memory embedding'''

        relative_path = image_path.lstrip("/\\")
        captures_root = os.path.realpath(os.path.join(os.getcwd(), "data", "captures"))
        target_path = os.path.realpath(os.path.join(captures_root, relative_path))

        if not (target_path == captures_root or target_path.startswith(captures_root + os.sep)):
            return {"ok": False, "message": "Invalid image_path"}

        if not os.path.isfile(target_path):
            return {"ok": False, "message": "Image not found"}

        os.remove(target_path)

        person_dir = os.path.dirname(target_path)
        display_name = os.path.basename(person_dir).strip().upper()
        remaining_images = self._list_capture_images(person_dir)
        emb = self._recompute_cached_embedding(person_dir, remaining_images)
        self._upsert_embedding(display_name, emb)
        log_info(f"Removed image: {image_path}")
        return {"ok": True, "message": "Success!"}

    # ───────────────────────────── Capture ─────────────────────────────────

    def capture_unknown(self, name: str) -> dict:
        name = (name or "").strip().upper()
        if not name:
            return {"ok": False, "message": "Name is required."}

        with self.capture_lock:
            target, frame = self.latest_target, self.latest_frame_bytes

        if not target or not frame:
            return {"ok": False, "message": "No target face available."}

        emb = np.asarray(target["embedding"], dtype=np.float32)
        safe_name = "".join(c for c in name if c.isalnum() or c in "_- ").strip().replace(" ", "_")
        data_dir = os.path.join("data", "captures", safe_name)
        os.makedirs(data_dir, exist_ok=True)

        try:
            self._save_capture_files(safe_name, data_dir, frame, target["bbox"])
            self._update_cached_embedding(data_dir, emb)
            updated_emb = self._load_cached_embedding(data_dir)
            self._upsert_embedding(safe_name, updated_emb)
            log_info(f"Captured face for {name}")
        except Exception as e:
            log_info(f"Capture failed for {name}: {e}")
            return {"ok": False, "message": "Failed to save capture."}

        return {"ok": True, "message": f"Captured {name} successfully."}

    def _save_capture_files(self, name: str, folder: str, frame: bytes, bbox: list[float]):
        # Save snapshot
        img = Image.frombytes("RGB", (self.width, self.height), frame)
        l = max(int(bbox[0] * self.width), 0)
        t = max(int(bbox[1] * self.height), 0)
        r = min(int(bbox[2] * self.width), self.width)
        b = min(int(bbox[3] * self.height), self.height)

        # Expand bbox (70%) while staying within frame
        bw, bh = max(r - l, 0), max(b - t, 0)
        pad_x = int(bw * 0.7)
        pad_y = int(bh * 0.7)
        l = max(l - pad_x, 0)
        t = max(t - pad_y, 0)
        r = min(r + pad_x, self.width)
        b = min(b + pad_y, self.height)
        crop = img.crop((l, t, r, b)) if r > l and b > t else img
        crop.save(os.path.join(folder, f"{name}_{datetime.now():%Y%m%d_%H%M%S}.jpg"), quality=95)

    def _update_cached_embedding(self, folder: str, emb: np.ndarray) -> None:
        cache_path = self._embedding_cache_path(folder)
        try:
            image_count = self._count_capture_images(folder)
            prev_count = max(image_count - 1, 0)

            if os.path.exists(cache_path):
                prev = np.load(cache_path)
                new_avg = (prev * prev_count + emb) / max(prev_count + 1, 1)
            else:
                new_avg = emb

            np.save(cache_path, new_avg)
        except Exception as e:
            log_info(f"Embedding cache update failed: {folder} ({e})")

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
            self.latest_frame_bytes = frame
            self.latest_target = (
                {"bbox": bboxes[target_idx], "embedding": np.asarray(embeddings[target_idx], dtype=np.float32)}
                if target_idx is not None and target_idx < len(embeddings) else None
            )
        return target_idx

    # ──────────────────────────── Inference ────────────────────────────────

    def infer(self, frame_bytes: bytes) -> list[FRResult]:
        if not frame_bytes:
            return []

        try:
            img = np.array(Image.frombytes("RGB", (self.width, self.height), frame_bytes))
            faces = self.model.get(img)
            if not faces:
                self._update_recent([])
                return [{"label": l["name"]} for l in self.old_detections]

            embeddings = [f.normed_embedding for f in faces]
            bboxes = [self._frac_bbox(img.shape[1], img.shape[0], f.bbox.tolist()) for f in faces]

            if not self.name_list:
                labels = ["Unknown"] * len(faces)
                dists = [[1.0]] * len(faces)
            else:
                k = min(2 if self.fr_settings.get("use_differentiator", False) else 1, len(self.name_list))
                neighbors, dists = (
                    self._brute_search(embeddings, k) if self.fr_settings.get("use_brute_force", False)
                    else self.vector_index.query(embeddings, k=k)
                )
                labels = self._match_labels(embeddings, neighbors, dists, bboxes)

            target_idx = self._update_capture_target(frame_bytes, embeddings, labels, bboxes)
            self._update_recent([
                {
                    "name": labels[i], 
                    "bbox": bboxes[i], 
                    "norm_embed": self._norm(embeddings[i]), 
                    "last_seen": datetime.now()
                }
                for i in range(len(faces)) if labels[i] != "Unknown"
            ])

            return [
                {"bbox": bboxes[i], "label": labels[i], "score": float(dists[i][0]), "is_target": i == target_idx}
                for i in range(len(faces))
            ] + [{"label": l["name"]} for l in self.old_detections]

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
                match = self.name_list[neighbors[i][0]]
            elif s["use_differentiator"] and dist[0] < s["threshold_lenient_diff"] and len(dist) > 1 and (dist[1] - dist[0]) > s["similarity_gap"]:
                match = self.name_list[neighbors[i][0]]
            elif s["use_persistor"]:
                match = self._catch_recent(embeddings[i], dist[0], bboxes[i])
            labels.append(match or "Unknown")
        return labels

    def _catch_recent(self, emb: np.ndarray, score: float, bbox: list[float]) -> str | None:
        '''Implements the persistor mechanic'''
        emb_norm = self._norm(emb)
        best_sim, match = 0, None
        for r in self.current_detections + self.old_detections:
            if calc_iou(r["bbox"], bbox) < self.fr_settings["threshold_iou"]:
                continue
            sim = float(np.dot(emb_norm, r["norm_embed"]))
            if sim > best_sim:
                best_sim, match = sim, r["name"]
        if best_sim > (1 - self.fr_settings["threshold_prev"]) and score < self.fr_settings["threshold_lenient_pers"]:
            return match
        return None

    def _update_recent(self, updated: list[RecentDetection]) -> list[str]:
        '''
        Updates self.current_detections based on the latest detections
        Preserves previous detections in self.old_detections for holding_time
        '''
        self.current_detections = updated

        preserved = []
        updated_names = {d["name"] for d in updated}
        cutoff = datetime.now() - timedelta(seconds=self.fr_settings["holding_time"])
        for d in self.current_detections + self.old_detections:
            if d["name"] not in updated_names and d["last_seen"] > cutoff:
                preserved.append(d)

        self.old_detections = preserved

    def _brute_search(self, queries: list[np.ndarray], k: int) -> tuple[list, list]:
        if not self.db_embeddings_normalized.size:
            return [], []
        neighbors, dists = [], []
        for q in queries:
            q_norm = self._norm(q)
            cos_dist = 1 - np.dot(self.db_embeddings_normalized, q_norm)
            idx = np.argpartition(cos_dist, min(k, len(cos_dist) - 1))[:k]
            idx = idx[np.argsort(cos_dist[idx])]
            neighbors.append(idx.tolist())
            dists.append(cos_dist[idx].tolist())
        return neighbors, dists

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
        with self.inference_lock:
            self.fr_results = []
        self.inferenceThread = threading.Thread(target=self._loop_inference, daemon=True)
        self.inferenceThread.start()

    def _loop_inference(self) -> None:
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
                    time.sleep(0.001)
                    continue

                last_id = curr_id
                frame_cnt += 1
                
                # Enforced frame skipping
                skip = self.fr_settings.get("frame_skip", 1)
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

                # Write perf log if it has been perf_interval_s since the last log
                now = time.monotonic()
                if self.fr_settings["perf_logging"] and (now - last_perf_log) >= perf_interval_s:
                    interval = max(now - last_perf_log, 1e-9)
                    fps_infer = infer_calls / interval
                    avg_ms = (infer_time_s_sum / max(infer_calls, 1)) * 1000.0
                    skip_val = self.fr_settings.get("frame_skip", 1)
                    wait_ratio = skipped_same_frame / max(skipped_same_frame + infer_calls, 1)
                    log_info(f"[PERF] infer:{fps_infer:.1f}fps/{avg_ms:.0f}ms skip:{skip_val} wait:{wait_ratio:.1%}")
                    last_perf_log = now
                    infer_calls = 0
                    infer_time_s_sum = 0.0
                    skipped_same_frame = 0
                    skipped_for_performance = 0

            except Exception as e:
                log_info(f"Inference loop error: {e}")

        self._reset_index()
        self.current_detections = []
        self.old_detections = []

    def start_detection_broadcast(self) -> Generator[str, None, None]:
        last_hash, last_send_time = None, 0
        while self.inferenceThread and self.inferenceThread.is_alive():
            # Cap rate of broadcast to MAX_BROADCAST_FPS
            now = time.monotonic()
            if now - last_send_time < 1 / self.fr_settings.get("max_broadcast_fps", 50):
                time.sleep(0.001)
                continue

            with self.inference_lock:
                results = self.fr_results
            
            # If results are the same, don't broadcast
            h = hash(str(results))
            if h == last_hash:
                time.sleep(0.001)
                continue

            last_hash, last_send_time = h, now
            yield json.dumps({"data": results}) + '\n'

    def cleanup(self, *_args) -> None:
        """Sets event to trigger termination of inference and streaming processees"""

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
        self.current_detections = []
        self.old_detections = []

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