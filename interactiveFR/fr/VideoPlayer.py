import subprocess
import threading
import time
import sys
from contextlib import contextmanager
from typing import Generator
import numpy as np
import cv2

from utils import log_info

class RWLock:
    """
    Read-Write Lock: allows multiple simultaneous readers OR one exclusive writer.
    Readers don't block each other, only writers need exclusive access.
    Used for protecting frame_bytes and frame_id, allowing multiple readers (broadcast and inference)
    """

    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0

    def acquire_read(self):
        """Acquire a read lock. Multiple readers can hold this simultaneously."""
        with self._read_ready:
            self._readers += 1

    def release_read(self):
        """Release a read lock."""
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()

    def acquire_write(self):
        """Acquire a write lock. Blocks until all readers release."""
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def release_write(self):
        """Release a write lock."""
        self._read_ready.release()

    @contextmanager
    def read_lock(self):
        """Context manager for read lock (exception-safe)."""
        self.acquire_read()
        try:
            yield
        finally:
            self.release_read()

    @contextmanager
    def write_lock(self):
        """Context manager for write lock (exception-safe)."""
        self.acquire_write()
        try:
            yield
        finally:
            self.release_write()

class StreamState:
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"

class VideoSource:
    '''
    Class to build ffmpeg command based on video source string
    '''
    
    def __init__(self, width: int, height: int, fps: int) -> None:
        self.width = width
        self.height = height
        self.fps = fps
    
    @staticmethod
    def list_cameras() -> list[str]:
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

    def build_ffmpeg_command(self, src: str) -> list[str]:
        src = (src or "").strip()

        # Default: RTSP/IP camera
        if src.upper().startswith("RTSP://"):
            return [
                "ffmpeg",

                # Hardware acceleration
                "-hwaccel", "auto",

                # Low-latency RTSP settings
                "-rtsp_transport", "tcp",
                "-fflags", "nobuffer+discardcorrupt",
                "-flags", "low_delay",
                "-avioflags", "direct",

                # Must be BEFORE -i
                "-probesize", "32",
                "-analyzeduration", "0",
                "-thread_queue_size", "8",

                "-i", src,

                # Video processing
                "-vf", f"fps={self.fps}, scale={self.width}:{self.height}",
                "-an",
                "-sn",

                # RGB frames output
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",

                # Buffering
                "-buffer_size", "64k",

                # Output to stdout
                "pipe:1",
            ]
        
        # Camera device (starts with 'camera:')
        if src.startswith("camera:"):
            device_name = src.split(":", 1)[1].strip()

            if not device_name:
                log_info("No camera device name provided")
                return None

            if sys.platform.startswith("win"):
                log_info(f"Using DirectShow camera: {device_name}")
                return [
                    "ffmpeg",

                    # Hardware acceleration
                    "-hwaccel", "auto",

                    # Low-latency DirectShow settings
                    "-f", "dshow",
                    "-fflags", "nobuffer",
                    "-flags", "low_delay",
                    "-probesize", "32",
                    "-analyzeduration", "0",
                    "-thread_queue_size", "1",

                    # Input device and scaling
                    "-i", f"video={device_name}",
                    "-vf", f"fps={self.fps}, scale={self.width}:{self.height}",
                    "-an",
                    "-sn",

                    # RGB frames output
                    "-f", "rawvideo",
                    "-pix_fmt", "rgb24",

                    # Buffering
                    "-buffer_size", "64k",

                    # Output to stdout
                    "-loglevel", "error",
                    "pipe:1",
                ]
            else:
                log_info(f"Using v4l2 camera: {device_name}")
                return [
                    "ffmpeg",

                    # Low-latency v4l2 settings
                    "-f", "dshow",
                    "-fflags", "nobuffer",
                    "-flags", "low_delay",
                    "-probesize", "32",
                    "-analyzeduration", "0",
                    "-thread_queue_size", "1",

                    # Input device and scaling
                    "-i", device_name,
                    "-vf", f"fps={self.fps}, scale={self.width}:{self.height}",
                    "-an",
                    "-sn",

                    # RGB frames output
                    "-f", "rawvideo",
                    "-pix_fmt", "rgb24",

                    # Buffering
                    "-buffer_size", "64k",

                    # Output to stdout
                    "-loglevel", "error",
                    "pipe:1",
                ]

class VideoPlayer:
    """
    Class for streaming video from ffmpeg
    """

    def __init__(self, width, height, fps) -> None:
        # For modifiables
        self.vid_lock = RWLock()  # RWLock for concurrent read access
        self.frame_bytes = b"" 
        self.frame_id = 0 # Counter to track new frames
        self.perf_logging = False
        self.ffmpeg_process = None
        self.streamThread = None

        # Thread event
        self.end_event = threading.Event()

        # Set resolution and framerate that ffmpeg will convert video source to
        # This is the resolution and framerate that will be broadcast
        self.width = width
        self.height = height
        self.frame_size = self.width * self.height * 3
        self.fps = fps
        
        self.stream_state = StreamState.IDLE
        self.last_error = None

        log_info("Video Player initialised!")

    # -------- Public API ---------

    @property
    def is_started(self) -> bool:
        return self.stream_state == StreamState.RUNNING
    
    def start_stream(self, stream_src: str) -> None:
        """
        Starts ffmpeg video stream in a separate thread
        
        Arguments
        - stream_src: url to RTSP video stream or 'camera:<device_name>'
        """

        log_info(f"Starting FFmpeg stream from {stream_src}")

        if self.stream_state in (StreamState.STARTING, StreamState.RUNNING, StreamState.STOPPING):
            log_info(f"Stream already active (state={self.stream_state}); start ignored")
            return None

        ffmpeg_command = VideoSource(self.width, self.height, self.fps).build_ffmpeg_command(stream_src)
        if not ffmpeg_command:
            log_info("Failed to build FFmpeg command; stream not started")
            self.end_event.set()
            self.stream_state = StreamState.FAILED
            return None

        self.end_event = threading.Event()
        self.last_error = None
        self.stream_state = StreamState.STARTING
        self.streamThread = threading.Thread(target=self._handleFFmpegStream, args=(ffmpeg_command,), daemon=True)
        self.streamThread.start()
        log_info(f"Stream thread started. Thread alive: {self.streamThread.is_alive()}")
    
    def end_stream(self) -> None:
        """Ends ffmpeg video stream"""

        log_info("Ending stream...")
        self._shutdown_stream()
        self.stream_state = StreamState.IDLE
        
    def start_broadcast(self) -> Generator[bytes, any, any]:
        """
        Generator yielding video frames proccessed from ffmpeg, for broadcast 
        """

        last_frame_id = 0

        while self.streamThread is not None and self.streamThread.is_alive():
            with self.vid_lock.read_lock():
                current_frame_id = self.frame_id
                frame_bytes = self.frame_bytes

            if current_frame_id <= last_frame_id:
                time.sleep(0.01)
                continue
            last_frame_id = current_frame_id

            if not frame_bytes or len(frame_bytes) != self.frame_size:
                time.sleep(0.01)
                continue

            # Reshape, convert to JPEG before streaming
            frame = np.frombuffer(frame_bytes, np.uint8).reshape(
                (self.height, self.width, 3)
            )
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

    # -------- Internal Methods ---------

    def _handleFFmpegStream(self, ffmpeg_command: list) -> None:
        """
        Thread to open ffmpeg subprocess and processes the frame to bytes
        
        Arguments
        - ffmpeg_command: FFmpeg command in list form for streaming to RGB bytes.
        """

        # log_info(f"FFmpeg command: {ffmpeg_command}")
        ffmpeg_process = None

        # Main try/except 
        try:
            ffmpeg_process = subprocess.Popen(
                ffmpeg_command,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
                bufsize=0  # Unbuffered
            )
            log_info("FFmpeg process started")

            # Keep a handle so we can stop it from other methods
            self.ffmpeg_process = ffmpeg_process

            # Give ffmpeg a brief moment to start and check if it failed immediately
            time.sleep(0.1)

            if ffmpeg_process.poll() is not None:
                # Process has already terminated (stderr drain thread will log details)
                log_info("FFmpeg process terminated immediately")
                self.stream_state = StreamState.FAILED
                self.end_event.set()
                return

            self.stream_state = StreamState.RUNNING
            buffer_bytes = bytearray()
            frames_processed = 0

            # MAIN READ LOOP
            while not self.end_event.is_set():
                # Check if process has terminated
                if ffmpeg_process.poll() is not None:
                    log_info(f"FFmpeg process terminated unexpectedly. Processed {frames_processed} frames.")
                    self.stream_state = StreamState.FAILED
                    self.end_event.set()
                    break

                # Read chunk of data
                try:
                    chunk = ffmpeg_process.stdout.read(65536)  # 64KB chunks for better throughput
                except Exception as e:
                    log_info(f"Error reading from stdout: {e}")
                    self.stream_state = StreamState.FAILED
                    self.end_event.set()
                    break

                # If no data yet, wait first
                if not chunk:
                    continue                    

                # We got data!
                buffer_bytes.extend(chunk)

                # Cap buffer to prevent unbounded memory growth (~10 frames max)
                max_buffer = self.frame_size * 10
                if len(buffer_bytes) > max_buffer:
                    del buffer_bytes[:len(buffer_bytes) - max_buffer]

                while len(buffer_bytes) >= self.frame_size:
                    frame_bytes = buffer_bytes[:self.frame_size]
                    del buffer_bytes[:self.frame_size]

                    with self.vid_lock.write_lock():
                        self.frame_bytes = frame_bytes
                        self.frame_id += 1

                    frames_processed += 1
                    if frames_processed == 1:
                        log_info("Successfully processed first RGB frame from FFmpeg stream")
        
        except Exception as e:
            log_info(f"Unhandled error in FFmpeg stream thread: {e}")
            self.stream_state = StreamState.FAILED

            self.end_event.set()
        
        if self.stream_state != StreamState.FAILED:
            log_info(f"Stream ended peacefully")
            self.stream_state = StreamState.IDLE

        return

    def _shutdown_stream(self):
        """Ends streaming loop, safely stops ffmpeg process and resets state"""

        if self.stream_state not in (StreamState.IDLE, StreamState.FAILED):
            self.stream_state = StreamState.STOPPING

        # Kill the read loop
        self.end_event.set()
    
        # If ffmpeg_process is still alive: terminate, wait, kill
        if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
            self.ffmpeg_process.terminate()
            try:
                self.ffmpeg_process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.ffmpeg_process.kill()
