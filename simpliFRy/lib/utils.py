import sys
import subprocess
import logging
import os
from datetime import datetime

logger = None

def init_logger():
    # Set internal FLASK logs level to error
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    global logger
    
    log_folder = os.path.join('data', 'logs')

    os.makedirs(log_folder, exist_ok=True)

    curr_time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')

    log_filename = os.path.join(log_folder, f"Logs {curr_time}.logs")

    logger = logging.getLogger('detections')
    logger.setLevel(logging.INFO)
    logger_handler = logging.FileHandler(log_filename)
    logger_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)

    print(f"\nLogging to file: {log_filename}\n")
    

def log_info(message: str) -> None:
    """
    Logs a message in the logs file at INFO level

    Arguments
    - message: message to be logged
    """
    
    logger.info(message)


def calc_box_area(bbox: list[float]) -> float:
    """
    Calculates the area of a bounding box

    Arguments
    - bbox: bounding box in xyxy format 

    Returns
    - Area of bounding box
    """

    x_min, y_min, x_max, y_max = bbox
    return max(0, x_max - x_min) * max(0, y_max - y_min)

def calc_iou(bbox1: list[float], bbox2: list[float]) -> float:
    """
    Calculate Intersection-Over-Union value of 2 bounding boxes

    Arguments
    - bbox1: bounding box in xyxy format
    - bbox2: bounding box in xyxy format

    Returns
    - intersection-over-union value
    """
    
    # Calculate intersection area bounding box values
    x_min = max(bbox1[0], bbox2[0])
    y_min = max(bbox1[1], bbox2[1])
    x_max = min(bbox1[2], bbox2[2])
    y_max = min(bbox1[3], bbox2[3])

    # If bounding boxes do not intersect, return 0
    if x_min >= x_max or y_min >= y_max: 
        return 0.0

    inter_area = calc_box_area([x_min, y_min, x_max, y_max])
    union_area = calc_box_area(bbox1) + calc_box_area(bbox2) - inter_area
    
    return inter_area / union_area

def list_camera_devices() -> list[str]:
    '''
    List available camera device names on the system.
    Returns a list of device name strings.
    '''
    if sys.platform.startswith("win"):
        try:
            result = subprocess.run(
                ["ffmpeg", "-f", "dshow", "-list_devices", "true", "-i", "dummy"],
                capture_output=True, text=True, timeout=5
            )
            lines = result.stderr.split('\n')
            video_devices = []
            for line in lines:
                if '(video)' in line and '"' in line:
                    start = line.find('"') + 1
                    end = line.find('"', start)
                    if start > 0 and end > start:
                        name = line[start:end]
                        if not name.startswith('@'):
                            video_devices.append(name)
            return video_devices
        except Exception as e:
            log_info(f"Error listing cameras: {e}")
            return []
    else:
        # Linux/macOS: list /dev/video* devices
        import glob
        devices = sorted(glob.glob("/dev/video*"))
        log_info(f"Detected {len(devices)} camera(s): {devices}")
        return devices