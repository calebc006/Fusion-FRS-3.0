import os
import sys
import threading
import time
import traceback
import warnings
import json
import hashlib
from datetime import datetime, timedelta
from typing import Generator, TypedDict
import subprocess
import logging

import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from voyager import Index, Space
from tqdm import tqdm

from fr import VideoPlayer
from sql_db import get_db, recreate_table, fetch_records, save_record
from utils import calc_iou, log_info

logging.getLogger('insightface').setLevel(logging.ERROR)
os.environ['ORT_LOGGING_LEVEL'] = '3'

FR_SETTINGS_PATH = 'settings.json'
EMBEDDINGS_CACHE_FP = 'embeddings_cache.json'
FR_DEFAULT_SETTINGS = {
    "threshold": 0.45, 
    "holding_time": 3, 
    "use_brute_force": False,
    "perf_logging": False, 
    "frame_skip": 1, 
    "max_broadcast_fps": 50,

    "use_differentiator": True, 
    "threshold_lenient_diff": 0.55,
    "similarity_gap": 0.10, 

    "use_persistor": True, 
    "threshold_prev": 0.3,
    "threshold_iou": 0.2, 
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

        # Settings
        self.fr_settings = self._load_settings()
        self.embeddings_loaded = False
        self.embeddings_loading = False

        # Inference
        self.inference_lock = threading.Lock()
        self.inferenceThread = None
        self.fr_results = []
        self.broadcast_package = []

        log_info("FR Model initialised!")

    @staticmethod
    def _suppress_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface\.utils\.transform")
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface\.utils\.face_align")

    def _load_settings(self) -> FRSettings:
        '''Loads settings from FR_SETTINGS_PATH json file, uses defaults and writes back if not specified'''

        defaults: FRSettings = {**FR_DEFAULT_SETTINGS}
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
        self.fr_settings = {**previous_settings, **new_settings}
        with open(FR_SETTINGS_PATH, 'w') as f:
            json.dump(self.fr_settings, f)
        return self.fr_settings

    # ─────────────────────────── Index Management ────────────────────────────

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
        """Rebuild ANN index from in-memory embeddings"""
        self.vector_index = Index(Space.Cosine, num_dimensions=512)
        if self.embeddings_array.size == 0:
            self.db_embeddings_normalized = np.array([])
            return

        for emb in self.embeddings_array:
            self.vector_index.add_item(emb)

        norms = np.linalg.norm(self.embeddings_array, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.db_embeddings_normalized = self.embeddings_array / norms

    # ─────────────────────── Embedding Loading (Namelist + Cache) ────────────

    def load_embeddings(self, data_file: str | None) -> None:
        """
        Loads embeddings from DB cache or regenerates from namelist JSON

        Arguments
        - data_file: file path (relative to './data' folder) to json file linking names to pictures
        """
        self.embeddings_loading = True
        self.embeddings_loaded = False
        success = False
        try:
            self._reset_index()
            self.current_detections = []
            self.old_detections = []

            if not data_file:
                self._fetch_embeddings()
            else:
                data_file = data_file.strip()
                if self._should_regenerate_embeddings(data_file):
                    self._form_embeddings(data_file)
                else:
                    self._fetch_embeddings()
            success = True
        finally:
            self.embeddings_loading = False
            self.embeddings_loaded = success

    @staticmethod
    def _compute_file_hash(filepath: str) -> str:
        """Compute SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    @staticmethod
    def _load_cache_info() -> dict:
        """Load embedding cache information"""
        if os.path.exists(EMBEDDINGS_CACHE_FP):
            with open(EMBEDDINGS_CACHE_FP, 'r') as f:
                return json.load(f)
        return {}

    @staticmethod
    def _save_cache_info(data_file: str, file_hash: str) -> None:
        """Save embedding cache information"""
        cache_info = {
            "data_file": data_file,
            "file_hash": file_hash,
            "timestamp": datetime.now().isoformat()
        }
        with open(EMBEDDINGS_CACHE_FP, 'w') as f:
            json.dump(cache_info, f, indent=2)

    def _should_regenerate_embeddings(self, data_file: str) -> bool:
        """Check if embeddings need to be regenerated based on file hash"""
        current_hash = self._compute_file_hash(data_file)
        cache_info = self._load_cache_info()

        if cache_info.get("data_file") == data_file and cache_info.get("file_hash") == current_hash:
            with get_db() as conn:
                records = fetch_records(conn)
                if len(records) > 0:
                    log_info(f"Using cached embeddings (file hash matches: {current_hash[:8]}...)")
                    return False
        
        log_info("Regenerating embeddings (file changed or no cache)")
        return True

    def _fetch_embeddings(self) -> None:
        """Load embeddings from SQLite database into in-memory index"""
        log_info("Loading embeddings from database...")

        with get_db() as conn:
            records = fetch_records(conn)

        for record in records:
            self._add_embedding(record["name"], record["ave_embedding"])

        log_info(f"Loaded {len(self.name_list)} embeddings from db")

    def _form_embeddings(self, data_file: str) -> None:
        """Create embeddings from namelist JSON and save to SQLite database"""

        if not data_file.endswith(".json"):
            raise ValueError("Please provide a json file with the .json extension.")

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"{data_file} does not exist!")

        with open(data_file, "r") as file:
            data_dict = json.load(file)

        # Resolve img_folder_path relative to the data file's directory
        data_dir = os.path.dirname(data_file)
        img_folder_path = os.path.join(data_dir, data_dict["img_folder_path"])

        with get_db() as conn:
            recreate_table(conn)
            log_info("Extracting embeddings from images...")
            self._reset_index()

            for entry in tqdm(data_dict["details"]):
                name = entry["name"]
                ave_embedding = self._avg_embedding(img_folder_path, entry["images"])

                if ave_embedding.size == 0:
                    continue

                save_record(conn, name, ave_embedding)
                self._add_embedding(name, ave_embedding)

        # Save cache info after successful generation
        file_hash = self._compute_file_hash(data_file)
        self._save_cache_info(data_file, file_hash)

    def _avg_embedding(self, folder: str, images: list[str]) -> np.ndarray:
        """Extract the average embedding representation of a person's face"""
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
                log_info(f"Error processing image: {img_name} ({e})")
        if no_face and not embeddings:
            log_info(f"No face detected in {no_face} image(s) from {folder}.")
        return np.mean(embeddings, axis=0) if embeddings else np.array([])

    # ──────────────────────────── Inference ────────────────────────────────

    def infer(self, frame_bytes: bytes) -> list[FRResult]:
        if not frame_bytes:
            return []

        try:
            img = np.array(Image.frombytes("RGB", (self.width, self.height), frame_bytes))
            faces = self.model.get(img)
            if not faces:
                self._update_recent([])
                return [{"label": l["name"], "held": True} for l in self.old_detections]

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
                {"bbox": bboxes[i], "label": labels[i], "score": float(dists[i][0]), "held": False}
                for i in range(len(faces))
            ] + [{"label": l["name"], "held": True} for l in self.old_detections]

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

        preserved = []
        updated_names = {d["name"] for d in updated}
        cutoff = datetime.now() - timedelta(seconds=self.fr_settings["holding_time"])
        for d in self.current_detections + self.old_detections:
            if d["name"] not in updated_names and d["last_seen"] > cutoff:
                preserved.append(d)

        self.current_detections = updated
        self.old_detections = preserved

    def _merge_with_held(self, results: list[FRResult]) -> list[FRResult]:
        """Append held detections that are still within holding_time."""
        merged = list(results)
        result_labels = {r["label"] for r in merged}
        cutoff = datetime.now() - timedelta(seconds=self.fr_settings["holding_time"])
        for d in self.old_detections:
            if d["name"] not in result_labels and d["last_seen"] > cutoff:
                merged.append({"label": d["name"], "held": True})
                result_labels.add(d["name"])
        return merged

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

                # Skip inference if repeated frame
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
                    self.broadcast_package = self._merge_with_held(results)

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
        last_send_time = 0
        while self.inferenceThread and self.inferenceThread.is_alive():
            # Cap rate of broadcast to max_broadcast_fps
            now = time.monotonic()
            if now - last_send_time < 1 / self.fr_settings.get("max_broadcast_fps", 50):
                time.sleep(0.001)
                continue

            with self.inference_lock:
                results = list(self.broadcast_package)
            
            last_send_time = now
            yield json.dumps({"data": results}) + '\n'

    def cleanup(self, *_args) -> None:
        """Sets event to trigger termination of inference and streaming processes"""

        log_info("CLEANING UP...")

        self.end_stream()

        # Join thread briefly to allow clean exit
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
            os._exit(0)

        return None
