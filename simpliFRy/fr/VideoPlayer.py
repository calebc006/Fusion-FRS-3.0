import subprocess
import threading
import time
import sys
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
        self.frame_bytes = b""  # Initialize frame_bytes to avoid AttributeError
        self.ffmpeg_process = None
        self.streamThread = None
        self.inferenceThread = None

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

    def _handle_stream_end(self) -> None:
        log_info("ENDING FFMPEG SUBPROCESS")
        self.is_started = False

    def _handleRTSP(self, stream_src:str) -> None:
        """
        Opens ffmpeg subprocess and processes the frame to bytes
        
        Arguments
        - stream_src: url to RTSP video stream or source to VCC
        """

        log_info(f"Starting FFmpeg stream from: {stream_src}")

        # command = [
        #     "ffmpeg",
        #     "-rtsp_transport", "tcp", 
        #     "-i", stream_src.strip(),
        #     "-vsync", "0",
        #     "-copyts",
        #     "-an",
        #     "-sn",
        #     "-f", "rawvideo",  # Video format is raw video
        #     "-s", "1280x720",
        #     "-pix_fmt", "bgr24",  # bgr24 pixel format matches OpenCV default pixels format.
        #     "-probesize", "32",
        #     "-analyzeduration", "0",
        #     "-fflags", "nobuffer",
        #     "-flags", "low_delay",
        #     "-tune", "zerolatency",
        #     "-b:v", "500k",
        #     "-buffer_size", "1000k",
        #     "-",
        # ]
        command = [
            "ffmpeg",

            # Low-latency RTSP settings
            "-rtsp_transport", "tcp",
            "-fflags", "nobuffer+discardcorrupt",
            "-flags", "low_delay",
            "-avioflags", "direct",

            # Must be BEFORE -i
            "-probesize", "32",
            "-analyzeduration", "0",
            "-thread_queue_size", "8",

            "-i", stream_src.strip(),

            # Video processing
            "-vf", f"scale={self.width}:{self.height}",
            "-an",
            "-sn",

            # MJPEG output
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-q:v", "3",
            "-pix_fmt", "yuv420p",

            # Buffering
            "-buffer_size", "64k",

            # Output to stdout
            "pipe:1",
        ]


        try:
            ffmpeg_process = subprocess.Popen(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                bufsize=0  # Unbuffered
            )
            log_info("FFmpeg process started")
            # Keep a handle so we can stop it from other methods
            self.ffmpeg_process = ffmpeg_process
        except Exception as e:
            log_info(f"An error occured starting ffmpeg: {e}")
            self._handle_stream_end()
            return

        # Give ffmpeg a brief moment to start and check if it failed immediately
        time.sleep(0.1)  # Reduced from 0.5s for faster startup
        if ffmpeg_process.poll() is not None:
            # Process has already terminated
            try:
                stderr_output = ffmpeg_process.stderr.read().decode('utf-8', errors='ignore')
                log_info(f"FFmpeg process terminated immediately. Error output:\n{stderr_output}")
            except Exception as e:
                log_info(f"FFmpeg process terminated immediately. Could not read stderr: {e}")
            self._handle_stream_end()
            return
        
        log_info("FFmpeg process is running, starting frame processing")

        # Capture stderr lines for reporting on failure (no verbose logging)
        stderr_lines = []
        def read_stderr():
            try:
                for line in iter(ffmpeg_process.stderr.readline, b''):
                    if line:
                        decoded_line = line.decode('utf-8', errors='ignore').strip()
                        stderr_lines.append(decoded_line)
            except Exception as e:
                log_info(f"Error reading stderr: {e}")
        
        threading.Thread(target=read_stderr, daemon=True).start()

        buffer_bytes = bytearray()
        frames_processed = 0
        last_data_time = None  # Will be set when we first receive data

        while not self.end_event.is_set():
            # Check if process has terminated
            if ffmpeg_process.poll() is not None:
                try:
                    remaining_stderr = ffmpeg_process.stderr.read().decode('utf-8', errors='ignore')
                    all_stderr = "\n".join(stderr_lines) + "\n" + remaining_stderr
                    log_info(f"FFmpeg process terminated unexpectedly. Processed {frames_processed} frames. Error output:\n{all_stderr}")
                except Exception as e:
                    log_info(f"FFmpeg process terminated unexpectedly. Processed {frames_processed} frames. Stderr lines: {stderr_lines[-10:]}")
                self.end_event.set()
                break

            # Read from stdout (this will block, but we check timeout after)
            # Increased chunk size to reduce syscalls (was 4096)
            try:
                chunk = ffmpeg_process.stdout.read(65536)  # 64KB chunks for better throughput
            except Exception as e:
                log_info(f"Error reading from stdout: {e}")
                self.end_event.set()
                break

            if not chunk:
                # Empty chunk - could mean EOF or just no data yet
                if ffmpeg_process.poll() is not None:
                    # Process terminated
                    try:
                        remaining_stderr = ffmpeg_process.stderr.read().decode('utf-8', errors='ignore')
                        all_stderr = "\n".join(stderr_lines) + "\n" + remaining_stderr
                        log_info(f"FFmpeg process terminated. Processed {frames_processed} frames. Error output:\n{all_stderr}")
                    except Exception as e:
                        log_info(f"FFmpeg process terminated. Processed {frames_processed} frames. Stderr lines: {stderr_lines[-10:]}")
                    self.end_event.set()
                    break
                else:
                    # Process alive but no data yet; minimal sleep to avoid busy loop
                    time.sleep(0.01)  # Reduced from 0.1s for faster response
                continue

            # We got data!
            last_data_time = time.time()

            buffer_bytes.extend(chunk)

            while True:
                start = buffer_bytes.find(b"\xff\xd8")  # JPEG start marker
                end = buffer_bytes.find(b"\xff\xd9", start + 2) if start != -1 else -1

                if start == -1 or end == -1:
                    break

                jpg_bytes = buffer_bytes[start : end + 2]
                del buffer_bytes[: end + 2]

                # Pass through JPEG directly without decoding/re-encoding to eliminate latency
                # Only decode if we need to check dimensions (which we skip if already correct)
                # This eliminates ~50-100ms of encoding/decoding overhead per frame
                with self.vid_lock:
                    self.frame_bytes = jpg_bytes
                
                frames_processed += 1
                if frames_processed == 1:
                    log_info("Successfully processed first frame from FFmpeg stream")

        else:
            self._handle_stream_end()

            #ffmpeg_process.stdout.close()  # Closing stdout terminates FFmpeg sub-process.
            if ffmpeg_process.poll() is None:
                ffmpeg_process.terminate()
                try:
                    ffmpeg_process.wait(timeout=2)  # Wait for FFmpeg sub-process to finish
                except subprocess.TimeoutExpired:
                    ffmpeg_process.kill()
            return

    def _stop_ffmpeg(self) -> None:
        """Terminate ffmpeg process safely and close pipes"""
        try:
            if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
                self.ffmpeg_process.terminate()
                try:
                    self.ffmpeg_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.ffmpeg_process.kill()
            if self.ffmpeg_process:
                if self.ffmpeg_process.stdout:
                    try:
                        self.ffmpeg_process.stdout.close()
                    except Exception:
                        pass
                if self.ffmpeg_process.stderr:
                    try:
                        self.ffmpeg_process.stderr.close()
                    except Exception:
                        pass
        except Exception:
            pass

    def cleanup(self, sig, f) -> None:
        """Sets event to trigger termination of ffmpeg subprocess"""

        log_info("CLEANING UP...")
        self.end_stream()
        time.sleep(0.5)
        # Final safety: exit if anything lingers
        try:
            sys.exit(0)
        except SystemExit:
            # In case sys.exit is swallowed by threads, force exit
            import os
            os._exit(0)
        return None

    def start_stream(self, stream_src: str) -> None:
        """
        Starts ffmpeg video stream in a separate thread
        
        Arguments
        - stream_src: url to RTSP video stream or source to VCC
        """

        log_info(f"start_stream called with source: {stream_src}")
        self.is_started = True
        self.end_event = threading.Event()
        self.streamThread = threading.Thread(target=self._handleRTSP, args=(stream_src,))
        self.streamThread.daemon = True
        self.streamThread.start()
        log_info(f"Stream thread started. Thread alive: {self.streamThread.is_alive()}")

        return None
    
    def end_stream(self) -> None:
        """Ends ffmpeg video stream"""\
        
        self.end_event.set()
        self._stop_ffmpeg()

        # Join threads briefly to allow clean exit
        try:
            if self.streamThread and self.streamThread.is_alive():
                self.streamThread.join(timeout=1)
            if self.inferenceThread and self.inferenceThread.is_alive():
                self.inferenceThread.join(timeout=1)
        except Exception:
            pass
        
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
            if frame_bytes:  # Only yield if we have a frame
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            else:
                # Very small sleep if no frame yet to prevent CPU spinning
                time.sleep(0.001)