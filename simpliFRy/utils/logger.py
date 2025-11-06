import logging
import os
from datetime import datetime

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
