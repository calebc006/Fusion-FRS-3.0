import subprocess
import threading
import time
import sys
from typing import Generator

import numpy as np
import cv2

from utils import log_info


class RWLock:
    """
    Read-Write Lock: allows multiple simultaneous readers OR one exclusive writer.
    Readers don't block each other, only writers need exclusive access.
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



class VideoSource:
    '''
    Class to build ffmpeg command based on video source string
    '''
    
    def __init__(self, source: str, width: int, height: int) -> None:
        self.source = source
        self.width = width
        self.height = height
    
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
                log_info(f"Detected {len(video_devices)} camera(s): {video_devices}")
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

    def _get_windows_camera_device_name(self, camera_index: int) -> str | None:
        '''
        Get the Windows camera device name by index using DirectShow. 
        If index is out of range, return the first device detected. If no devices found, return None.
        '''

        # Don't force video_size/framerate - let FFmpeg auto-detect camera capabilities
        # and scale output to desired resolution
        device_name = None
        try:
            result = subprocess.run(
                ["ffmpeg", "-f", "dshow", "-list_devices", "true", "-i", "dummy"],
                capture_output=True, text=True, timeout=5
            )
            # Parse stderr for video devices (device names appear after [dshow])
            lines = result.stderr.split('\n')
            video_devices = []
            for line in lines:
                # Look for lines with "(video)" to identify video devices
                if '(video)' in line and '"' in line:
                    # Extract device name between quotes
                    start = line.find('"') + 1
                    end = line.find('"', start)
                    if start > 0 and end > start:
                        name = line[start:end]
                        # Skip alternative names (they start with @)
                        if not name.startswith('@'):
                            video_devices.append(name)
                            log_info(f"Found video device: {name}")
            
            log_info(f"Found {len(video_devices)} video devices: {video_devices}")
            
            if camera_index < len(video_devices):
                device_name = video_devices[camera_index]
            elif video_devices:
                device_name = video_devices[0]
                log_info(f"Camera index {camera_index} out of range, using: {device_name}")
            
            return device_name
        
        except Exception as e:
            log_info(f"Error listing DirectShow devices: {e}")
            return None

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
                "-vf", f"fps=25, scale={self.width}:{self.height}",
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
                    "-f", "dshow",
                    "-video_size", f"{self.width}x{self.height}",
                    "-vcodec", "mjpeg",
                    "-i", f"video={device_name}",
                    "-vf", "fps=25",
                    "-an",
                    "-sn",
                    "-f", "rawvideo",
                    "-pix_fmt", "rgb24",
                    "-buffer_size", "64k",
                    "pipe:1",
                ]
            else:
                log_info(f"Using v4l2 camera: {device_name}")
                return [
                    "ffmpeg",
                    "-f", "v4l2",
                    "-i", device_name,
                    "-vf", f"scale={self.width}:{self.height}",
                    "-r", "25",
                    "-an",
                    "-sn",
                    "-f", "rawvideo",
                    "-pix_fmt", "rgb24",
                    "-buffer_size", "64k",
                    "pipe:1",
                ]

        

class VideoPlayer:
    """
    Class for streaming video from ffmpeg
    """

    def __init__(self) -> None:
        """Initialises the class"""

        # For modifiables
        self.vid_lock = RWLock()  # RWLock for concurrent read access
        self.frame_bytes = b"" 
        self.frame_id = 0 # Counter to track new frames
        self.perf_logging = False
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

        self.video_source = None

        log_info("Video Player initialised!")
        pass

    def _handle_stream_end(self) -> None:
        log_info("ENDING FFMPEG SUBPROCESS")
        self.is_started = False

    def _handleFFmpegStream(self, stream_src:str) -> None:
        """
        Opens ffmpeg subprocess and processes the frame to bytes
        
        Arguments
        - stream_src: url to RTSP video stream or source to VCC
        """

        log_info(f"Starting FFmpeg stream from: {stream_src}")

        self.video_source = VideoSource(stream_src, self.width, self.height)
        command = self.video_source.build_ffmpeg_command(stream_src)

        log_info(f"FFmpeg command: {command}")

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
        time.sleep(0.1) 
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

        # MAIN READ LOOP
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
            buffer_bytes.extend(chunk)
            frame_size = self.width * self.height * 3

            while len(buffer_bytes) >= frame_size:
                frame_bytes = buffer_bytes[:frame_size]
                del buffer_bytes[:frame_size]

                self.vid_lock.acquire_write()
                self.frame_bytes = frame_bytes
                self.frame_id += 1
                self.vid_lock.release_write()

                frames_processed += 1
                if frames_processed == 1:
                    log_info("Successfully processed first RGB frame from FFmpeg stream")

        self._handle_stream_end()

        # ffmpeg_process.stdout.close()  # Closing stdout terminates FFmpeg sub-process.
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

    def cleanup(self, sig=None, f=None) -> None:
        """Sets event to trigger termination of ffmpeg subprocess"""

        log_info("CLEANING UP...")
        self._shutdown_stream(force_exit=True)
        return None

    def _shutdown_stream(self, force_exit: bool = False) -> None:
        """Centralized shutdown for stream threads and ffmpeg process."""

        self.is_started = False
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

        if not force_exit:
            return None

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

        self.is_started = True
        self.end_event = threading.Event()
        self.streamThread = threading.Thread(target=self._handleFFmpegStream, args=(stream_src,))
        self.streamThread.daemon = True
        self.streamThread.start()
        log_info(f"Stream thread started. Thread alive: {self.streamThread.is_alive()}")

        return None
    
    def end_stream(self) -> None:
        """Ends ffmpeg video stream"""
        return self._shutdown_stream(force_exit=False)
        
    def start_broadcast(self) -> Generator[bytes, any, any]:
        """
        Broadcasts ffmpeg video stream

        Returns
        - Generator yielding video frames proccessed from ffmpeg
        """

        frame_size = self.width * self.height * 3
        last_frame_id = 0

        while self.streamThread is not None and self.streamThread.is_alive():
            # Acquire read lock - multiple readers can hold simultaneously
            self.vid_lock.acquire_read()
            current_frame_id = self.frame_id
            if current_frame_id <= last_frame_id:
                self.vid_lock.release_read()
                time.sleep(0.01)
                continue
            frame_bytes = self.frame_bytes  # bytes are immutable, so this is safe
            last_frame_id = current_frame_id
            self.vid_lock.release_read()

            if not frame_bytes or len(frame_bytes) != frame_size:
                time.sleep(0.01)
                continue

            # Do expensive operations outside the lock
            frame = np.frombuffer(frame_bytes, np.uint8).reshape(
                (self.height, self.width, 3)
            )

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )