import argparse
import json
import os
import signal
import time
import atexit
from types import SimpleNamespace

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, redirect, url_for, send_from_directory
from flask_cors import CORS

from fr import FRVidPlayer, VideoSource
from utils import log_info

load_dotenv()
app = Flask(__name__)
CORS(app)

log_info("Starting FR Session")
fr_instance = FRVidPlayer()

# Config from environment (can be overridden by argparse)
config = SimpleNamespace(
    ip=os.getenv("APP_IP", "0.0.0.0"),
    port=int(os.getenv("APP_PORT", "1333")),
    video=os.getenv("APP_VIDEO", "true").lower() == "true",
    env=os.getenv("APP_ENV", "development").lower(),
)

# Ensure fr_instance runs .cleanup() before process termination
atexit.register(fr_instance.cleanup)


# ───────────────────────────── API Routes ─────────────────────────────────

@app.route("/start", methods=["POST"])
def start():
    """API for frontend to start FR"""

    if fr_instance.is_started:
        return _json({"stream": False, "message": "Stream already started!"})

    stream_src = request.form.get("stream_src")
    data_file = request.form.get("data_file")

    fr_instance.start_stream(stream_src)

    # Give the stream thread a moment to start and check if it's still alive
    time.sleep(0.3)
    if not fr_instance.streamThread or not fr_instance.streamThread.is_alive():
        log_info("Stream thread died immediately after starting")
        fr_instance.end_stream()
        return _json({"stream": False, "message": "Failed to start video stream. Check logs for details."})

    try:
        fr_instance.load_embeddings(data_file)
    except (ValueError, FileNotFoundError) as e:
        fr_instance.end_stream()
        return _json({"stream": False, "message": str(e)})

    fr_instance.start_inference()
    return _json({"stream": True, "message": "Success!"})


@app.route("/end", methods=["POST"])
def end():
    """API for frontend to end FR"""

    if not fr_instance.is_started:
        return _json({"stream": False, "message": "Stream not started!"})

    fr_instance.end_stream()
    return _json({"stream": True, "message": "Success!"})


@app.route("/streamStatus", methods=["GET"])
def stream_status():
    """API to check stream state and last error"""

    return _json({
        "stream_state": fr_instance.stream_state,
        "last_error": fr_instance.last_error,
        "embeddings_loaded": getattr(fr_instance, "embeddings_loaded", False),
        "embeddings_loading": getattr(fr_instance, "embeddings_loading", False),
    })


@app.route("/vidFeed", methods=["GET"])
def video_feed():
    if not config.video:
        return Response("Video disabled", status=405)
    return Response(fr_instance.start_broadcast(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/frResults", methods=["GET"])
def fr_results():
    return Response(fr_instance.start_detection_broadcast(), mimetype="application/json")


@app.route("/submit_settings", methods=["POST"])
def submit_settings():
    s = fr_instance.fr_settings
    new = {
        "threshold": float(request.form.get("threshold", s["threshold"])),
        "holding_time": float(request.form.get("holding_time", s["holding_time"])),
        "use_brute_force": "use_brute_force" in request.form,
        "perf_logging": "perf_logging" in request.form,
        "use_differentiator": "use_differentiator" in request.form,
        "threshold_lenient_diff": float(request.form.get("threshold_lenient_diff", s["threshold_lenient_diff"])),
        "similarity_gap": float(request.form.get("similarity_gap", s["similarity_gap"])),
        "use_persistor": "use_persistor" in request.form,
        "threshold_prev": float(request.form.get("threshold_prev", s["threshold_prev"])),
        "threshold_iou": float(request.form.get("threshold_iou", s["threshold_iou"])),
        "threshold_lenient_pers": float(request.form.get("threshold_lenient_pers", s["threshold_lenient_pers"])),
        "frame_skip": int(request.form.get("frame_skip", s["frame_skip"])),
        "max_broadcast_fps": int(request.form.get("max_broadcast_fps", s["max_broadcast_fps"])),
    }
    fr_instance.adjust_values(new)
    return redirect(url_for('settings'))


@app.route("/listCameras")
def list_cameras():
    """API to list available camera devices"""
    response = jsonify(VideoSource.list_cameras())
    response.headers["Cache-Control"] = "no-store"
    return response


# ───────────────────────────── Pages ──────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/welcome")
def welcome():
    return render_template("welcome.html")


@app.route("/seats")
def seats():
    return render_template("seats.html")


@app.route("/old_layout")
def old_layout():
    return render_template("old_layout.html")


@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory('data', filename)


@app.route("/settings")
def settings():
    return render_template("settings.html", **fr_instance.fr_settings)


# ───────────────────────────── Helpers ────────────────────────────────────

def _json(data, status=200, headers=None):
    response = Response(json.dumps(data), status=status, mimetype='application/json')
    if headers:
        for key, value in headers.items():
            response.headers[key] = value
    return response


# ───────────────────────────── Main ───────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Facial Recognition")
    parser.add_argument("-ip", "--ipaddress", type=str)
    parser.add_argument("-p", "--port", type=int)
    parser.add_argument("-v", "--video", type=str)
    parser.add_argument("--env", type=str, choices=["development", "production"])
    parser.add_argument("--prod", action="store_true")
    args = parser.parse_args()

    if args.ipaddress:
        config.ip = args.ipaddress
    if args.port:
        config.port = args.port
    if args.video:
        config.video = args.video.lower() == "true"
    if args.env:
        config.env = args.env
    if args.prod:
        config.env = "production"

    # Cleanup upon interruption
    signal.signal(signal.SIGINT, fr_instance.cleanup)

    if config.env == "production":
        try:
            from waitress import serve
            log_info(f"Production mode on {config.ip}:{config.port}")
            serve(app, host=config.ip, port=config.port)
        except Exception as e:
            log_info(f"Waitress failed ({e}), using Flask")
            app.run(host=config.ip, port=config.port, use_reloader=False)
    else:
        log_info(f"Development mode on {config.ip}:{config.port}")
        app.run(debug=True, host=config.ip, port=config.port, use_reloader=False)