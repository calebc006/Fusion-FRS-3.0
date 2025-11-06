import subprocess
import threading
import time
from typing import Generator

import cv2
import numpy as np

from utils import log_info


class VideoPlayer:
    """
    Class for streaming video from ffmpeg
    """

    def __init__(self) -> None:
        """Initialises the class"""

        # For modifiables
        self.vid_lock = threading.Lock()

        # Thread event
        self.end_event = threading.Event()

        # Set resolution of input video
        self.width = 1280
        self.height = 720

        # Printing
        self.in_error = False

        # Check if ffmpeg has been initialised (used to block secondary requests)
        self.is_started = False

        log_info("Video Player initialised!")
        pass

    # def _print_error_notif(self) -> None:
    #     if not self.in_error:
    #         self.in_error = True
    #         print("Error reading frame!!!")

    # def _print_continue_notif(self) -> None:
    #     if self.in_error:
    #         self.in_error = False
    #         print("Error resolved, continue ")

    def _handle_stream_end(self) -> None:
        log_info("ENDING FFMPEG SUBPROCESS")
        self.is_started = False

    def _handleRTSP(self, stream_src:str) -> None:
        """
        Opens ffmpeg subprocess and processes the frame to bytes
        
        Arguments
        - stream_src: url to RTSP video stream or source to VCC
        """

        command = [
            "ffmpeg",
            "-rtsp_transport", "tcp", # Force TCP (for testing)
            "-i", stream_src.strip(),
            "-vsync", "0",
            "-copyts",
            "-an",
            "-sn",
            "-f", "rawvideo",  # Video format is raw video
            "-s", "1280x720",
            "-pix_fmt", "bgr24",  # bgr24 pixel format matches OpenCV default pixels format.
            "-probesize", "32",
            "-analyzeduration", "0",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-tune", "zerolatency",
            "-b:v", "500k",
            "-buffer_size", "1000k",
            "-",
        ]

        try:
            ffmpeg_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except Exception as e:
            log_info("An error occured:", e)
            self._handle_stream_end()

        while not self.end_event.is_set():
            # Read width*height*3 bytes from stdout (1 frame)
            raw_frame = ffmpeg_process.stdout.read(self.width * self.height * 3)

            # If error, ends ffmpeg subprocess
            if len(raw_frame) != (self.width * self.height * 3):
                self.end_event.set()
                continue

            # Convert the bytes read into a NumPy array, and reshape it to video frame dimensions
            frame = np.frombuffer(raw_frame, np.uint8).reshape(
                (self.height, self.width, 3)
            )
            _, buffer = cv2.imencode(".jpg", frame)

            with self.vid_lock:
                self.frame_bytes = buffer.tobytes()

        else:
            self._handle_stream_end()

            #ffmpeg_process.stdout.close()  # Closing stdout terminates FFmpeg sub-process.
            ffmpeg_process.terminate()
            ffmpeg_process.wait()  # Wait for FFmpeg sub-process to finish
            exit(0)

    def cleanup(self, sig, f) -> None:
        """Sets event to trigger termination of ffmpeg subprocess"""

        log_info("CLEANING UP...")
        self.end_event.set()
        time.sleep(0.5)
        exit(0)
        return None

    def start_stream(self, stream_src: str) -> None:
        """
        Starts ffmpeg video stream in a separate thread
        
        Arguments
        - stream_src: url to RTSP video stream or source to VCC
        """

        self.is_started = True
        self.end_event = threading.Event()
        self.streamThread = threading.Thread(target=self._handleRTSP, args=(stream_src,))
        self.streamThread.daemon = True
        self.streamThread.start()

        return None
    
    def end_stream(self) -> None:
        """Ends ffmpeg video stream"""\
        
        self.end_event.set()
        
        return None
        
    def start_broadcast(self) -> Generator[bytes, any, any]:
        """
        Broadcasts ffmpeg video stream

        Returns
        - Generator yielding video frames proccessed from ffmpeg
        """

        while self.streamThread.is_alive():
            with self.vid_lock:
                frame_bytes = self.frame_bytes
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )