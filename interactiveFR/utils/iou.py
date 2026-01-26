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
