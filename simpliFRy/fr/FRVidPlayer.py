import time
import json
import os
import threading
import traceback
from datetime import datetime, timedelta
from typing import Generator, TypedDict

import numpy as np
from torch import cuda
from insightface.app import FaceAnalysis
from voyager import Index, Space
from PIL import Image
from tqdm import tqdm

from fr import VideoPlayer
from sql_db import get_db, recreate_table, fetch_records, save_record
from utils import calc_iou, log_info


FR_SETTINGS_FP = 'settings.json'


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


        provider: str = (
            "CUDAExecutionProvider" if cuda.is_available() else "CPUExecutionProvider"
        )

        # For FR algorithm
        self.model = FaceAnalysis(providers=[provider])
        self.model.prepare(ctx_id=0)

        self.vector_index = Index(Space.Cosine, num_dimensions=512)
        self.name_list = []

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
            "holding_time": fr_settings.get("holding_time", 15),
            "use_differentiator": fr_settings.get("use_differentiator", True),
            "threshold_lenient_diff": fr_settings.get("threshold_lenient_diff", 0.55),
            "similarity_gap": fr_settings.get("similarity_gap", 0.10),
            "use_persistor": fr_settings.get("use_persistor", True),
            "threshold_prev": fr_settings.get("threshold_prev", 0.3),
            "threshold_iou": fr_settings.get("threshold_iou", 0.2),        
            "threshold_lenient_pers": fr_settings.get("threshold_lenient_pers", 0.60),
        }

        with open(FR_SETTINGS_FP, 'w') as file:
            json.dump(self.fr_settings, file)

        # For threading
        self.inference_lock = threading.Lock()
        self.fr_results = []

        log_info("FR Model initialised!")

        pass

    def load_embeddings(self, data_file: str | None) -> None:
        """
        Loads embeddings

        Arguments
        - data_file: file path (relative to './data' folder) to json file linking names to pictures
        """

        if not data_file:
            self._fetch_embeddings()
        else:
            self._form_embeddings(data_file.strip())

    def adjust_values(self, new_settings: FRSettings) -> FRSettings:
        """
        Adjusts adjustable FR parameters based on form submission from settings page and update to FR settings json file

        Arguments
        - new_settings: new setting parameters submitted by user (typed dictionary)

        Returns
        - new setting parameters
        """

        self.fr_settings = new_settings

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
        """Reset vector index and name list"""

        self.name_list = []
        self.vector_index = Index(Space.Cosine, num_dimensions=512)

    def _fetch_embeddings(self) -> None:
        """Load embeddings from SQLite database"""

        log_info("Loading embeddings...")

        if self.name_list:
            log_info("Embeddings already loaded!")
            return None

        with get_db() as conn:
            records = fetch_records(conn)

        for record in records:
            self.name_list.append(record["name"])
            self.vector_index.add_item(record["ave_embedding"])

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

            neighbours, distances = self.vector_index.query(
                embeddings_list, k=min(2, len(self.name_list))
            )
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

        while self.streamThread.is_alive() and not self.end_event.is_set():
            try:
                with self.vid_lock:
                    frame_bytes = self.frame_bytes
                results = self.infer(frame_bytes)
                with self.inference_lock:
                    self.fr_results = results
            except AttributeError as e:
                log_info(f"Error in inference loop: {e}")
                continue
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

        while hasattr(self, 'inferenceThread') and self.inferenceThread.is_alive():
            try:
                with self.inference_lock:
                    yield json.dumps({"data": self.fr_results}) + '\n'
                    time.sleep(0.01) # cap FPS at 100
            except Exception as e:
                log_info(f"Error in detection broadcast: {e}")
                log_info(traceback.format_exc())
                # Yield empty results instead of crashing
                yield json.dumps({"data": []}) + '\n'
