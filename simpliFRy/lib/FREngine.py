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
import hashlib
from tqdm import tqdm
import numpy as np
import cv2
from insightface.app import FaceAnalysis
import voyager

from lib.VideoPlayer import VideoPlayer, StreamState
from lib.utils import calc_iou, log_info
from sql_db import get_db, recreate_table, fetch_records, save_record

FR_SETTINGS_PATH = 'settings.json'
EMBEDDINGS_CACHE_FP = 'embeddings_cache.json'
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
        if len(self.embeddings_list) > 0:
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

        # Other state
        self.persisted_detections: deque[FRResult] = deque(maxlen=self.fr_settings["q_max_size"])
        self.embeddings_loaded = False
        self.embeddings_loading = False

        log_info(f"FR Model initialised! Input: {self.width}x{self.height}, {self.fps}fps; Inference: {inference_width}x{inference_height}")

    
    # ───────────────────────────── Embeddings ─────────────────────────────────

    def load_embeddings(self, data_file: str | None) -> None:
        '''
        Loads embeddings from DB cache or regenerates from namelist JSON

        Arguments
        - data_file: file path (relative to './data' folder) to json file linking names to pictures
        '''
        self.embeddings_loading = True
        self.embeddings_loaded = False
        data_file = data_file.strip()

        try:
            self.embedding_index.reset_index()
            self.persisted_detections.clear()

            if not data_file or not self._should_regenerate_embeddings(data_file):
                log_info("Loading embeddings from database...")
                with get_db() as conn:
                    records = fetch_records(conn)
                for record in records:
                    self.embedding_index.add_item(record["name"], record["ave_embedding"])
                log_info(f"Loaded {self.embedding_index.size()} embeddings from db")
            else:
                log_info("Generating embeddings...")
                self._generate_embeddings(data_file)

            self.embeddings_loading = False
            self.embeddings_loaded = True
            
        except:
            self.embeddings_loading = False
            self.embeddings_loaded = False

            log_info("Failed to load embeddings!")

    def _generate_embeddings(self, data_file: str) -> None:
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
            self.embedding_index.reset_index()

            for entry in tqdm(data_dict["details"]):
                name = entry["name"]
                ave_embedding = self._avg_embedding(img_folder_path, entry["images"])

                if ave_embedding.size == 0:
                    continue

                save_record(conn, name, ave_embedding)
                self.embedding_index.add_item(name, ave_embedding)

        # Save cache info after successful generation
        file_hash = self._compute_file_hash(data_file)
        self._save_cache_info(data_file, file_hash)

    def _avg_embedding(self, folder_path: str, image_list: list[str]) -> np.ndarray:
        """Extract the average embedding from images in image_list"""
        embeddings = []
        no_face = 0
        for img_name in image_list:
            try:
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB after using cv2.imread
                faces = self.model.get(img)
                if faces:
                    embeddings.append(faces[0].normed_embedding)
                else:
                    no_face += 1
            except Exception as e:
                log_info(f"Error processing image: {img_name} ({e})")
        if no_face and not embeddings:
            log_info(f"No face detected in {no_face} image(s) from {folder_path}.")
        return np.mean(embeddings, axis=0) if embeddings else np.array([])

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

    # ───────────────────────────── Settings ─────────────────────────────────

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

            current_detections = [
                FRResult({
                    "label": labels[i],
                    "bbox": bboxes[i],
                    "norm_embed": embeddings[i], 
                    "score": float(dists[i][0]),
                    "last_seen": time.monotonic(),
                })
                for i in range(len(preds))
            ]

            # Persist non-unknown detections
            for r in current_detections:
                if r["label"] != "Unknown":
                    self.persisted_detections.append(r)

            t3 = time.monotonic() # t3-t2 is the time it takes to update persistor state

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


    
