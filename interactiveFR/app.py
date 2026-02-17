import argparse
import json
import os
import signal
import time
import atexit
from types import SimpleNamespace

from dotenv import load_dotenv
from flask import Flask, Response, render_template, request, redirect, url_for, send_from_directory
from flask_cors import CORS
from waitress import serve

from lib import FREngine, VideoPlayer, StreamState
from lib.utils import list_camera_devices, log_info, init_logger

load_dotenv()
app = Flask(__name__)
CORS(app) 

init_logger()
log_info("Starting FR Session")

videoplayer = VideoPlayer(
    width=int(os.getenv("WIDTH", "1920")),
    height=int(os.getenv("HEIGHT", "1080")),
    fps=int(os.getenv("FPS", "25")),
)

fr_instance = FREngine(
    videoplayer=videoplayer,
    inference_width=int(os.getenv("INFERENCE_WIDTH", "640")),
    inference_height=int(os.getenv("INFERENCE_HEIGHT", "480")),
)

# Config from environment 
config = SimpleNamespace(
    ip=os.getenv("APP_IP", "0.0.0.0"),
    port=int(os.getenv("APP_PORT", "1334")),
    env=os.getenv("APP_ENV", "production").lower(), # "production" or "development"
)

# Ensure fr_instance runs .cleanup() before process termination
atexit.register(lambda sig, frame: fr_instance.cleanup(force_exit=True))

# ───────────────────────────── API Routes ─────────────────────────────────

@app.route("/api/start_stream", methods=["POST"])
def start_stream():
    """API for frontend to start videoplayer stream"""
    payload = request.get_json(silent=True)
    stream_src = payload.get("stream_src")

    if videoplayer.is_started():
        return _json({"stream": False, "message": "Stream already started!"})

    videoplayer.start_stream(stream_src)

    # Poll the stream thread until it starts (if not assume it failed)
    for delay in [0, 0.1, 0.2, 0.5, 1]:
        time.sleep(delay)
        if videoplayer.streamThread and videoplayer.streamThread.is_alive():
            break
        log_info(f"Stream thread not started, checking again in {delay}s...")

    if not videoplayer.streamThread or not videoplayer.streamThread.is_alive():
        log_info("Stream thread couldn't start")
        fr_instance.cleanup()
        return _json({"stream": False, "message": "Failed to start video stream. Check logs for details."})
    
    return _json({"stream": True, "message": "Success!"})


@app.route("/api/start_fr", methods=["POST"])
def start_fr():
    """For frontend to load FR embeddings and start inference"""
    try:
        fr_instance.load_embeddings()
    except (ValueError, FileNotFoundError) as e:
        fr_instance.cleanup()
        log_info(f"Couldn't load embeddings, {e}")
        return _json({"inference": False, "message": "Failed to load embeddings. Check logs for details."})

    fr_instance.start_inference()
    return _json({"inference": True, "message": "Success!"})


@app.route("/api/end", methods=["POST"])
def end():
    """API for frontend to end FR"""

    if not videoplayer.is_started():
        return _json({"stream": False, "message": "Stream not started!"})

    fr_instance.cleanup()
    return _json({"stream": True, "message": "Success!"})


@app.route("/api/status", methods=["GET"])
def status():
    """API to check stream/inference status and last error"""

    return _json({
        "stream_state": videoplayer.stream_state,
        "last_error": videoplayer.last_error,
        "embeddings_loaded": getattr(fr_instance, "embeddings_loaded", False),
        "embeddings_loading": getattr(fr_instance, "embeddings_loading", False),
    })

@app.route("/api/vidFeed", methods=["GET"])
def video_feed():
    return Response(videoplayer.start_broadcast(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/frResults", methods=["GET"])
def fr_results():
    return Response(fr_instance.start_detection_broadcast(), mimetype="application/json")

prev_target = None
@app.route("/api/capture", methods=["POST"])
def capture():
    payload = request.get_json(silent=True)
    name = payload.get("name")
    allow_duplicate = payload.get("allow_duplicate") 
    result = fr_instance.capture_unknown(name, allow_duplicate)
    
    # Check if capture_unknown gave us a target crop; if so we need user confirmation
    global prev_target
    prev_target = result.get("crop", None)
    if prev_target is not None:
        return _json({"ok": result.get("ok"), "message": result.get("message")}, 202)
    
    return _json(result, 200 if result.get("ok") else 400)

@app.route("/api/capture/confirm", methods=["POST"])
def confirm_capture():
    payload = request.get_json(silent=True)
    name = payload.get("name")
    allow = payload.get("allow_duplicate") # Flask auto parses JSON boolean!

    if allow: 
        result = fr_instance.capture_unknown(name, allow_duplicate=True, target_image=prev_target)
        return _json(result, 200 if result.get("ok") else 400)
    return _json({"ok": True, "message": "User cancelled capture operation"}, 200)


@app.route("/api/submit_settings", methods=["POST"])
def submit_settings():
    s = fr_instance.fr_settings
    new = {
        "threshold": float(request.form.get("threshold", s["threshold"])),
        "holding_time": float(request.form.get("holding_time", s["holding_time"])),
        "max_detections": int(request.form.get("max_detections", s["max_detections"])),
        "perf_logging": "perf_logging" in request.form,
        "frame_skip": int(request.form.get("frame_skip", s["frame_skip"])),
        "max_broadcast_fps": int(request.form.get("max_broadcast_fps", s["max_broadcast_fps"])),
        "video_width": int(request.form.get("video_width", s["video_width"])),
        "video_height": int(request.form.get("video_height", s["video_height"])),

        "use_differentiator": "use_differentiator" in request.form,
        "threshold_lenient_diff": float(request.form.get("threshold_lenient_diff", s["threshold_lenient_diff"])),
        "similarity_gap": float(request.form.get("similarity_gap", s["similarity_gap"])),
        
        "use_persistor": "use_persistor" in request.form,
        "q_max_size": int(request.form.get("q_max_size", s["q_max_size"])),
        "threshold_iou": float(request.form.get("threshold_iou", s["threshold_iou"])),
        "threshold_sim": float(request.form.get("threshold_sim", s["threshold_sim"])),
        "threshold_lenient_pers": float(request.form.get("threshold_lenient_pers", s["threshold_lenient_pers"])),
    }
    fr_instance.adjust_values(new)
    return redirect(url_for('settings'))


@app.route("/api/get_settings", methods=["GET"])
def get_settings():
    return _json(fr_instance.fr_settings)


@app.route("/api/get_reference_images", methods=["GET"])
def get_reference_images():
    images = _collect_reference_images()
    return _json([{"name": n, "images": imgs} for n, imgs in sorted(images.items())])


@app.route("/api/get_reference_names", methods=["GET"])
def get_reference_names():
    names = []

    # From captures
    captures_path = os.path.join("data", "captures")
    if os.path.isdir(captures_path):
        for name in os.listdir(captures_path):
            names.append(name)

    return _json(names)


@app.route("/api/remove_image", methods=["POST"])
def remove_image():
    payload = request.get_json(silent=True) or {}
    image_path = payload.get("image_path")

    if not image_path:
        return _json({"message": "Missing image_path"}, 400)

    drive, _ = os.path.splitdrive(image_path)
    if drive:
        return _json({"message": "Invalid image_path"}, 400)

    result = fr_instance.remove_capture_image(image_path)
    if result.get("ok"):
        return _json({"message": result.get("message", "Success!")})
    return _json({"message": result.get("message", "Invalid image_path")}, 400)


@app.route("/api/listCameras", methods=["GET"])
def list_cameras():
    """API to list available camera devices"""
    response = _json(list_camera_devices())
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/api/get_performance", methods=["GET"])
def get_performance():
    """API to get backend performance"""
    if fr_instance.last_perf is not None:
        return _json(fr_instance.last_perf)
    return _json({})
# ───────────────────────────── Pages ──────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/interactive")
def interactive():
    return render_template("interactive.html")


@app.route("/references")
def references():
    return render_template("references.html")


@app.route("/settings")
def settings():
    return render_template("settings.html", **fr_instance.fr_settings) # Render with fr_settings as context


@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory('data', filename)


# ───────────────────────────── Helpers ────────────────────────────────────

def _json(data, status: int | None = 200, headers: dict | None = None) -> Response:
    '''Wrapper around json.dumps. Takes in data (Python list/dict) and optional status and headers'''
    response = Response(json.dumps(data), status=status, mimetype='application/json')
    if headers:
        for key, value in headers.items():
            response.headers[key] = value
    return response


def _collect_reference_images() -> dict:
    images = {}

    # From captures
    captures = os.path.join("data", "captures")
    if os.path.isdir(captures):
        for name in os.listdir(captures):
            person = os.path.join(captures, name)
            if os.path.isdir(person):
                imgs = [f"/data/captures/{name}/{i}" for i in os.listdir(person) if i.lower().endswith((".jpg", ".jpeg", ".png"))]
                if imgs:
                    images[name] = images.get(name, []) + imgs

    return images


# ───────────────────────────── Main ───────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Facial Recognition")
    parser.add_argument("-ip", "--ipaddress", type=str)
    parser.add_argument("-p", "--port", type=int)
    parser.add_argument("--env", type=str, choices=["development", "production"])
    parser.add_argument("--prod", action="store_true")
    args = parser.parse_args()

    if args.ipaddress:
        config.ip = args.ipaddress
    if args.port:
        config.port = args.port
    if args.env:
        config.env = args.env
    if args.prod:
        config.env = "production"

    # Cleanup upon interruption
    signal.signal(signal.SIGINT, lambda sig, frame: fr_instance.cleanup(force_exit=True))

    if config.env == "production":
        try:
            log_info(f"Production mode on {config.ip}:{config.port}")
            serve(app, host=config.ip, port=config.port)
        except Exception as e:
            log_info(f"Waitress failed ({e}), using Flask")
            app.run(host=config.ip, port=config.port, use_reloader=False)
    else:
        log_info(f"Development mode on {config.ip}:{config.port}")
        app.run(debug=True, host=config.ip, port=config.port, use_reloader=False)
