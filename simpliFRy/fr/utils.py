import numpy as np
import subprocess

def fractionalize_bbox(
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

def normalize_embed(embed: np.ndarray) -> np.ndarray:
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

def is_cuda_available():
    try:
        subprocess.check_output(['nvidia-smi'])
        return True
    except (Exception, FileNotFoundError):
        return False
    
