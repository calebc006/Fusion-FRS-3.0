import time
import json
import os
import io
import threading
import traceback
import hashlib
import warnings
from datetime import datetime, timedelta
from typing import Generator, TypedDict
import subprocess

import numpy as np
from insightface.app import FaceAnalysis
from voyager import Index, Space
from PIL import Image
from tqdm import tqdm

from fr import VideoPlayer
from sql_db import get_db, recreate_table, fetch_records, save_record
from utils import calc_iou, log_info


def is_cuda_available():
    try:
        subprocess.check_output(['nvidia-smi'])
        return True
    except (Exception, FileNotFoundError):
        return False


FR_SETTINGS_FP = 'settings.json'
EMBEDDINGS_CACHE_FP = 'embeddings_cache.json'


class FRResult(TypedDict):
    """Detection results from FR for an individual"""

    bboxes: list[float]
    labels: str
    score: float


class RecentDetection(TypedDict):
    """Recent detection results from FR for an individual"""

    name: str
    bbox: list[float]
    norm_embed: np.ndarray
    last_seen: datetime


class FRSettings(TypedDict):
    """Adjustable parameteres for FR algorithm"""

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


class FRVidPlayer(VideoPlayer):
    """
    Class for handling facial recognition conducted on ffmpeg stream
    """

    def __init__(self) -> None:
        """Initialises the class"""

        super().__init__()

        # Reduce noise from upstream dependencies (does not affect behavior)
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module=r"insightface\.utils\.transform",
        )
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module=r"insightface\.utils\.face_align",
        )

        # Last-call timing (seconds) for logging
        self._last_search_time_s = 0.0
        self._last_search_backend = "none"

        provider = "CUDAExecutionProvider" if is_cuda_available() else "CPUExecutionProvider"

        # For FR algorithm
        # Buffalo_L with optimized settings for RTX A5000
        self.model = FaceAnalysis(
            name="buffalo_l",
            providers=[provider],
            allowed_modules=["detection", "recognition"],  # Skip unnecessary modules (age, gender, etc.)
        )
        # det_size: Smaller = faster detection. 640x640 is default, 480x480 or 320x320 for speed
        # det_thresh: Higher = fewer false positives, faster (skips low-confidence faces)
        self.model.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

        self.vector_index = Index(Space.Cosine, num_dimensions=512)
        self.name_list = []
        self.embeddings_array = np.array([])  # For brute force search
        self.db_norms = np.array([]) # Pre-computed norms for brute force search
        self.db_embeddings_normalized = np.array([])  # Pre-computed normalized embeddings

        self.recent_detections: list[RecentDetection] = []

        # For settings
        if not os.path.exists(FR_SETTINGS_FP):
            fr_settings = {}
            with open(FR_SETTINGS_FP, 'w') as file:
                json.dump(fr_settings, file)
        else:
            with open(FR_SETTINGS_FP, 'r') as file:
                fr_settings = json.load(file)

        self.fr_settings: FRSettings = {
            "threshold": fr_settings.get("threshold", 0.45),
            "holding_time": fr_settings.get("holding_time", 3),
            "use_brute_force": fr_settings.get("use_brute_force", False),
            "perf_logging": fr_settings.get("perf_logging", False),
            "use_differentiator": fr_settings.get("use_differentiator", True),
            "threshold_lenient_diff": fr_settings.get("threshold_lenient_diff", 0.55),
            "similarity_gap": fr_settings.get("similarity_gap", 0.10),
            "use_persistor": fr_settings.get("use_persistor", True),
            "threshold_prev": fr_settings.get("threshold_prev", 0.3),
            "threshold_iou": fr_settings.get("threshold_iou", 0.2),        
            "threshold_lenient_pers": fr_settings.get("threshold_lenient_pers", 0.60),
            "frame_skip": fr_settings.get("frame_skip", 1),  # Process every Nth frame (1 = no skip)
        }

        with open(FR_SETTINGS_FP, 'w') as file:
            json.dump(self.fr_settings, file)

        # For threading
        self.inference_lock = threading.Lock()
        self.fr_results = []

        # Mirror FR setting onto base class (used by VideoPlayer stream perf logs)
        self.perf_logging = bool(self.fr_settings.get("perf_logging", False))

        log_info("FR Model initialised!")

        pass

    def _compute_file_hash(self, filepath: str) -> str:
        """Compute SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _load_cache_info(self) -> dict:
        """Load embedding cache information"""
        if os.path.exists(EMBEDDINGS_CACHE_FP):
            with open(EMBEDDINGS_CACHE_FP, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache_info(self, data_file: str, file_hash: str) -> None:
        """Save embedding cache information"""
        cache_info = {
            "data_file": data_file,
            "file_hash": file_hash,
            "timestamp": datetime.now().isoformat()
        }
        with open(EMBEDDINGS_CACHE_FP, 'w') as f:
            json.dump(cache_info, f, indent=2)

    def _should_regenerate_embeddings(self, data_file: str) -> bool:
        """
        Check if embeddings need to be regenerated
        
        Returns True if embeddings should be regenerated, False if cached embeddings can be used
        """
        data_file_path = os.path.join('data', data_file)
        
        # Compute current file hash
        current_hash = self._compute_file_hash(data_file_path)
        
        # Load cache info
        cache_info = self._load_cache_info()
        
        # Check if we have valid cache
        if cache_info.get("data_file") == data_file and cache_info.get("file_hash") == current_hash:
            # Check if database has embeddings
            with get_db() as conn:
                records = fetch_records(conn)
                if len(records) > 0:
                    log_info(f"Using cached embeddings (file hash matches: {current_hash[:8]}...)")
                    return False
        
        log_info(f"Regenerating embeddings (file changed or no cache)")
        return True

    def load_embeddings(self, data_file: str | None) -> None:
        """
        Loads embeddings

        Arguments
        - data_file: file path (relative to './data' folder) to json file linking names to pictures
        """

        if not data_file:
            self._fetch_embeddings()
        else:
            data_file = data_file.strip()
            if self._should_regenerate_embeddings(data_file):
                self._form_embeddings(data_file)
            else:
                self._fetch_embeddings()

    def adjust_values(self, new_settings: FRSettings) -> FRSettings:
        """
        Adjusts adjustable FR parameters based on form submission from settings page and update to FR settings json file

        Arguments
        - new_settings: new setting parameters submitted by user (typed dictionary)

        Returns
        - new setting parameters
        """

        self.fr_settings = new_settings

        # Keep stream/infer logging in sync with settings
        self.perf_logging = bool(new_settings.get("perf_logging", False))

        with open(FR_SETTINGS_FP, 'w') as file:
            json.dump(new_settings, file)

        return self.fr_settings

    def _extract_ave_embedding(self, img_folder_path: str, images: list[str]) -> np.ndarray:
        """
        Extract the average embedding representation of a person's face based on the picture(s) of the person

        Arguments
        - img_folder_path: path to image folder
        - images: name of the image files bearing the person's picture

        Returns
        - Average embedding representation of the person's face
        """

        embedding_list = []
        for img_name in images:
            img_fp = os.path.join(img_folder_path, img_name)

            try: 
                img = Image.open(img_fp).convert("RGB")
                img_arr = np.array(img)
            except Exception:
                log_info(f"Error processing {img_name}")
                continue

            faces = self.model.get(img_arr)

            if not len(faces): 
                log_info(f"{img_name} contains no detectable faces!")
                continue

            embedding_list.append(faces[0].embedding)

        if len(embedding_list) == 0:
            return np.array([])

        return sum(embedding_list) / len(embedding_list)

    def _reset_vector_index(self) -> None:
        """Reset vector index, embeddings array, and name list"""

        self.name_list = []
        self.vector_index = Index(Space.Cosine, num_dimensions=512)
        self.embeddings_array = np.array([])
        self.db_norms = np.array([])
        self.db_embeddings_normalized = np.array([])

    def _fetch_embeddings(self) -> None:
        """Load embeddings from SQLite database"""

        log_info("Loading embeddings...")

        if self.name_list:
            log_info("Embeddings already loaded!")
            return None

        with get_db() as conn:
            records = fetch_records(conn)

        embeddings_list = []
        for record in records:
            self.name_list.append(record["name"])
            self.vector_index.add_item(record["ave_embedding"])
            embeddings_list.append(record["ave_embedding"])
        
        if embeddings_list:
            self.embeddings_array = np.array(embeddings_list)
            # Pre-compute norms and normalized embeddings
            self.db_norms = np.linalg.norm(self.embeddings_array, axis=1, keepdims=True)
            self.db_norms[self.db_norms == 0] = 1
            self.db_embeddings_normalized = self.embeddings_array / self.db_norms

        log_info(f"Loaded {len(self.name_list)} embeddings from db")       

    def _form_embeddings(self, data_file: str) -> None:
        """
        Create embeddings for SQLite database from data provided
        
        Arguments
        - data_file: file path (relative to './data' folder) to json file linking names to pictures
        """

        if not data_file.endswith(".json"):
            raise ValueError("Please provide a json file with the .json extension.")

        data_file_path = os.path.join('data', data_file)

        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"{data_file_path} does not exists!")

        with open(data_file_path, "r") as file:
            data_dict = json.load(file)

        img_folder_path = os.path.join('data', data_dict["img_folder_path"])

        with get_db() as conn:
            recreate_table(conn)
            log_info("Extracting embeddings from images...")
            self._reset_vector_index()
            embeddings_list = []
            for entry in tqdm(data_dict["details"]):
                name = entry["name"]
                ave_embedding = self._extract_ave_embedding(
                    img_folder_path, entry["images"]
                )

                if ave_embedding.shape == (0,): 
                    continue

                save_record(conn, name, ave_embedding)

                self.name_list.append(name)
                self.vector_index.add_item(ave_embedding)
                embeddings_list.append(ave_embedding)
            
            if embeddings_list:
                self.embeddings_array = np.array(embeddings_list)
                # Pre-compute norms and normalized embeddings for brute-force search
                self.db_norms = np.linalg.norm(self.embeddings_array, axis=1, keepdims=True)
                self.db_norms[self.db_norms == 0] = 1
                self.db_embeddings_normalized = self.embeddings_array / self.db_norms
        
        # Save cache info after successful generation
        data_file_path = os.path.join('data', data_file)
        file_hash = self._compute_file_hash(data_file_path)
        self._save_cache_info(data_file, file_hash)

    def _brute_force_search(self, query_embeddings: list[np.ndarray], k: int) -> tuple[list[list[int]], list[list[float]]]:
        """
        Perform brute force cosine distance search
        
        Arguments
        - query_embeddings: list of query embedding vectors
        - k: number of nearest neighbors to return
        
        Returns
        - tuple of (neighbours indices, distances)
        """
        if self.embeddings_array.size == 0:
            return [], []
        
        neighbours = []
        distances = []
        
        # Use pre-computed normalized embeddings
        if self.db_embeddings_normalized.size == 0 and self.embeddings_array.size > 0:
             # Fallback if somehow not computed (safety)
            self.db_norms = np.linalg.norm(self.embeddings_array, axis=1, keepdims=True)
            self.db_norms[self.db_norms == 0] = 1
            self.db_embeddings_normalized = self.embeddings_array / self.db_norms
        
        for query_embed in query_embeddings:
            # Normalize query embedding (ensure float32)
            query_embed = np.asarray(query_embed, dtype=np.float32)
            query_norm = np.linalg.norm(query_embed)
            if query_norm == 0:
                query_norm = 1
            query_normalized = query_embed / query_norm
            
            # Compute cosine similarities (optimized with float32)
            cosine_similarities = np.dot(self.db_embeddings_normalized, query_normalized)
            
            # Convert to cosine distances (1 - similarity)
            cosine_distances = 1 - cosine_similarities
            
            # Get k nearest neighbors using argpartition (faster than full sort for small k)
            if k >= len(cosine_distances):
                nearest_indices = np.arange(len(cosine_distances))
            else:
                # argpartition is O(n) vs argsort O(n log n)
                partition_indices = np.argpartition(cosine_distances, k)[:k]
                nearest_indices = partition_indices[np.argsort(cosine_distances[partition_indices])]
            
            nearest_distances = cosine_distances[nearest_indices]
            
            neighbours.append(nearest_indices.tolist())
            distances.append(nearest_distances.tolist())
        
        return neighbours, distances

    @staticmethod
    def _fractionalise_bbox(
        img_width: int, img_height: int, bbox: list[float]
    ) -> list[float]:
        """
        Convert bounding box values to fractions of the image

        Arguments
        - img_width: width of image (pixels)
        - img_height: height of image (pixels)
        - bbox: bounding box in xyxy format (pixels)

        Returns
        - bounding box in xyxy format (fraction)
        """

        return [
            float(bbox[0] / img_width),
            float(bbox[1] / img_height),
            float(bbox[2] / img_width),
            float(bbox[3] / img_height),
        ]

    @staticmethod
    def _normalise_embed(embed: np.ndarray) -> np.ndarray:
        """
        Normalise embeddings 

        Arguments
        - embed: raw embedding represenatation of a person's face

        Returns
        - normalised embedding representation of a person's face
        """

        # Make sure its type float32
        embed = np.asarray(embed, dtype=np.float32)
        norm = np.linalg.norm(embed)
        if norm == 0:
            return embed
        return embed / norm

    def _catch_recent(self, embed: np.ndarray, score: np.float32, bbox: list[float]) -> tuple[str, np.ndarray]:
        """
        Implementation of persistor mechanic
        If unrecognised by vanilla FR and differentiator, persistor mechanic checks if the bounding box is close to a face previously recognised. To do this 3 criteria must be fulfilled
        1. The face sufficiently resembles the previously recognised face (threshold_prev)
        2. The face somewhat resembles a face in the database (threshold_lenient_pers)
        3. The face occupies a similar area in the image as the previously recognised  (threshold_iou)
        
        Arguments
        - embed: embeddings of the yet-to-be recognised face
        - score: furthest distance (cosine similarity) to a face in the database
        - bbox: bounding box of the yet-to-be recognised face

        Returns
        - If recognised, name of the person bearing the recognised face, else "Unknown"
        - Normalised embedding of the yet-to-be recognised face
        """

        embed = FRVidPlayer._normalise_embed(embed)
        max_sim: float = 0
        closest_match = None

        for recent_detection in self.recent_detections:
            iou = calc_iou(recent_detection["bbox"], bbox)
            if iou < self.fr_settings["threshold_iou"]: 
                continue  

            similarity = np.dot(embed, recent_detection["norm_embed"])
            if float(similarity) > max_sim:
                max_sim = similarity
                closest_match = recent_detection["name"]

        if max_sim/512 > (1-self.fr_settings["threshold_prev"]) and (score < self.fr_settings['threshold_lenient_pers']):
            return (closest_match, embed)

        return ("Unknown", embed)

    def _update_recent_detections(
        self, updated: list[RecentDetection]
    ) -> list[str]:
        """
        Part of persistor mechanic
        Updates the self.recent_detections with latest detection

        Arguments:
        - updated: latest detections (from latest frame)

        Returns
        - names of those recently recognised (within holding time) but recognised in the latest frame
        """

        preserved_labels = []
        updated_names = [d["name"] for d in updated]

        curr_time = datetime.now()
        holding_time = timedelta(seconds=self.fr_settings["holding_time"])

        for detection in self.recent_detections:
            if detection["name"] in updated_names:
                continue

            time_diff = curr_time - detection["last_seen"]
            if time_diff > holding_time:
                continue

            updated.append(detection)
            preserved_labels.append(detection["name"])

        self.recent_detections = updated
        return preserved_labels

    def _log_if(self, name:str) -> None:
        """
        Log detection if name is not in recent detections (to minimise unnecessary logs)

        Arguments
        - name: name of recognised person
        """

        recent_names = [detection["name"] for detection in self.recent_detections]
        
        if name not in recent_names:
            log_info(f"{name} detected")

    def infer(self, frame_bytes: bytes) -> list[FRResult]:
        """
        Conducts FR inference on provided frame.
        Uses insightface for detecting faces and encoding them in embedding representation and uses Spotify's Voyager for a vector index search; includes self-implemented differentiator and persistor mechanics with adjustable parameters to improve accuracy of algorithm

        Arguments:
        - frame_bytes: image (in bytes) which FR is conducted on

        Returns
        - list of recognised faces, their scores and bounding boxes (typed dictionary)    
        """

        try:
            if not frame_bytes:
                return [{"label": "Unknown"}]

            img = Image.frombytes("RGB", (self.width, self.height), frame_bytes)

            width, height = img.size
            img = np.array(img)

            faces = self.model.get(img)
            embeddings_list = [face.embedding for face in faces]

            if len(embeddings_list) == 0:
                extra_labels = self._update_recent_detections([])
                return [{"label": label} for label in extra_labels]

            if len(self.name_list) == 0:
                # No embeddings loaded, return empty results
                extra_labels = self._update_recent_detections([])
                return [{"label": label} for label in extra_labels]

            # Use brute force or Voyager search based on settings
            # Optimize k: only need k=2 if differentiator is enabled, otherwise k=1 is sufficient
            k = 2 if self.fr_settings["use_differentiator"] else 1
            k = min(k, len(self.name_list))
            
            t_search0 = time.monotonic() if self.fr_settings.get("perf_logging", False) else None
            if self.fr_settings["use_brute_force"]:
                neighbours, distances = self._brute_force_search(
                    embeddings_list, k=k
                )
                self._last_search_backend = "brute"
            else:
                neighbours, distances = self.vector_index.query(
                    embeddings_list, k=k
                )
                self._last_search_backend = "voyager"
            if t_search0 is not None:
                self._last_search_time_s = time.monotonic() - t_search0
        except Exception as e:
            log_info(f"Error in infer method: {e}")
            log_info(traceback.format_exc())
            # Return empty results instead of crashing
            extra_labels = self._update_recent_detections([])
            return [{"label": label} for label in extra_labels]

        labels = []
        updated_recent_detections : list[RecentDetection] = []  # Same format as recent_detections

        bboxes = [FRVidPlayer._fractionalise_bbox(
                    width, height, face["bbox"].tolist()
                ) for face in faces]

        for i, dist in enumerate(distances):
            if dist[0] < self.fr_settings["threshold"] or (
                self.fr_settings["use_differentiator"]
                and dist[0] < self.fr_settings["threshold_lenient_diff"]
                and len(dist) > 1
                and (dist[1] - dist[0]) > self.fr_settings["similarity_gap"]
            ):
                name = self.name_list[neighbours[i][0]]
                latest_embedding = FRVidPlayer._normalise_embed(embeddings_list[i])
                self._log_if(name)

            elif self.fr_settings["use_persistor"]:
                name, latest_embedding = self._catch_recent(embeddings_list[i], dist[0], bboxes[i])
            
            else: 
                name = "Unknown"

            labels.append(name)

            if name == "Unknown":
                continue
                
            updated_recent_detections.append({
                "name": name,
                "bbox": bboxes[i],
                "norm_embed": latest_embedding,
                "last_seen": datetime.now(),
            })
            

        extra_labels = self._update_recent_detections(updated_recent_detections)

        return [
            {
                "bbox": bboxes[i],
                "label": labels[i],
                "score": float(distances[i][0]),
            }
            for i in range(len(faces))
        ] + [{"label": label} for label in extra_labels]

    def _loopInference(self) -> None:
        """Repeatedly conducts inference on the latest frame from the ffmpeg video stream"""

        last_processed_id = -1
        frame_counter = 0

        # Perf logging
        perf_interval_s = 5.0
        last_perf_log = time.monotonic()
        infer_calls = 0
        infer_time_s_sum = 0.0
        search_time_s_sum = 0.0
        search_calls = 0
        skipped_same_frame = 0
        skipped_for_performance = 0

        while self.streamThread.is_alive() and not self.end_event.is_set():
            try:
                # Acquire read lock - doesn't block other readers (broadcast thread)
                self.vid_lock.acquire_read()
                current_id = self.frame_id
                frame_bytes = self.frame_bytes
                self.vid_lock.release_read()
                
                # Skip if we've already processed this frame
                if current_id == last_processed_id:
                    skipped_same_frame += 1
                    time.sleep(0.001) # Short cool-down
                    continue
                
                last_processed_id = current_id
                frame_counter += 1
                
                # Frame skipping for performance (process every Nth frame)
                frame_skip = self.fr_settings.get("frame_skip", 1)
                if frame_skip > 1 and (frame_counter % frame_skip) != 0:
                    skipped_for_performance += 1
                    continue

                if self.fr_settings.get("perf_logging", False):
                    t0 = time.monotonic()
                    results = self.infer(frame_bytes)
                    infer_time_s_sum += (time.monotonic() - t0)
                    infer_calls += 1

                    search_time_s_sum += float(getattr(self, "_last_search_time_s", 0.0))
                    search_calls += 1
                else:
                    results = self.infer(frame_bytes)

                with self.inference_lock:
                    self.fr_results = results

                now = time.monotonic()
                if self.fr_settings.get("perf_logging", False) and (now - last_perf_log) >= perf_interval_s:
                    interval = max(now - last_perf_log, 1e-9)
                    fps_infer = infer_calls / interval
                    avg_ms = (infer_time_s_sum / max(infer_calls, 1)) * 1000.0
                    avg_search_ms = (search_time_s_sum / max(search_calls, 1)) * 1000.0
                    skip = self.fr_settings.get("frame_skip", 1)
                    wait_ratio = skipped_same_frame / max(skipped_same_frame + infer_calls, 1)
                    log_info(f"[PERF] infer:{fps_infer:.1f}fps/{avg_ms:.0f}ms search:{avg_search_ms:.1f}ms skip:{skip} wait:{wait_ratio:.1%}")
                    last_perf_log = now
                    infer_calls = 0
                    infer_time_s_sum = 0.0
                    search_time_s_sum = 0.0
                    search_calls = 0
                    skipped_same_frame = 0
                    skipped_for_performance = 0
            except Exception as e:
                log_info(f"Error in inference loop: {e}")
                log_info(traceback.format_exc())
                # Continue processing instead of crashing
                continue
        else:
            self._reset_vector_index()
            self.recent_detections = []

    def start_inference(self) -> None:
        """Starts FR inference on ffmpeg video stream in a separate thread"""

        self.inferenceThread = threading.Thread(target=self._loopInference)
        self.inferenceThread.daemon = True
        self.inferenceThread.start()
        return None

    def start_detection_broadcast(self) -> Generator[list[FRResult], any, any]:
        """
        Starts broadcast of detection results from FR inferencing on ffmpeg video stream
        
        Returns
        - Generator yielding FR detection results (list of typed dictionary)
        """
        
        last_results_hash = None
        min_interval = 0.033  # ~30 FPS max for detection updates (reduces network overhead)
        last_send_time = 0.0

        while self.inferenceThread is not None and self.inferenceThread.is_alive():
            try:
                now = time.monotonic()
                
                # Throttle to avoid flooding client
                if (now - last_send_time) < min_interval:
                    time.sleep(0.005)
                    continue
                
                with self.inference_lock:
                    results = self.fr_results
                
                # Only send if results changed (reduces redundant updates)
                results_hash = hash(str(results))
                if results_hash == last_results_hash:
                    time.sleep(0.005)
                    continue
                
                last_results_hash = results_hash
                last_send_time = now
                yield json.dumps({"data": results}) + '\n'
                
            except Exception as e:
                log_info(f"Error in detection broadcast: {e}")
                log_info(traceback.format_exc())
                # Yield empty results instead of crashing
                yield json.dumps({"data": []}) + '\n'
