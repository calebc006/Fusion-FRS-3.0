import argparse
import json
import os
import signal
import time

from flask import Flask, Response, render_template, request, redirect, url_for, send_from_directory
from flask_cors import CORS

from fr import FRVidPlayer
from utils import log_info

app = Flask(__name__)
CORS(app)

log_info("Starting FR Session")

fr_instance = FRVidPlayer()


@app.route("/start", methods=["POST"])
def start():
    """API for frontend to start FR"""

    if fr_instance.is_started:        
        response_msg = json.dumps({"stream": False, "message": "Stream already started!"})
        return Response(response_msg, status=200, mimetype='application/json')

    stream_src = request.form.get("stream_src", None)
    data_file = request.form.get("data_file", None)

    fr_instance.start_stream(stream_src)
    
    # Give the stream thread a moment to start and check if it's still alive
    time.sleep(1.5)
    if not fr_instance.streamThread.is_alive():
        log_info("Stream thread died immediately after starting")
        fr_instance.end_event.set()
        response_msg = json.dumps({"stream": False, "message": "Failed to start video stream. Check logs for details."})
        return Response(response_msg, status=200, mimetype='application/json')

    try:
        fr_instance.load_embeddings(data_file)
    except (ValueError, FileNotFoundError) as err:
        fr_instance.end_event.set()
        response_msg = json.dumps({"stream": False, "message": str(err)})
        return Response(response_msg, status=200, mimetype='application/json')

    fr_instance.start_inference()
    response_msg = json.dumps({"stream": True, "message": "Success!"})
    return Response(response_msg, status=200, mimetype='application/json')


@app.route("/end", methods=["POST"])
def end():
    """API for frontend to end FR"""

    if not fr_instance.is_started:
        response_msg = json.dumps({"stream": False, "message": "Stream not started!"})
        return Response(response_msg, status=200, mimetype='application/json')
    
    fr_instance.end_stream()

    response_msg = json.dumps({"stream": True, "message": "Success!"})
    return Response(response_msg, status=200, mimetype='application/json')


@app.route("/checkAlive") 
def check_alive():
    """API to check if FR has started"""

    try:
        if fr_instance.streamThread.is_alive():
            response = "Yes"
        else: 
            response = "No"
    except AttributeError: 
        response = "No"

    return Response(response, status=200, mimetype='application/json')

@app.route("/vidFeed")
def video_feed():
    """Returns a HTTP streaming response of the video feed from FFMPEG"""
    vid_enabled = args.video == "true"

    if vid_enabled:
        return Response(
            fr_instance.start_broadcast(), mimetype="multipart/x-mixed-replace; boundary=frame"
        )
    else:
        Response("Video stream is not enabled", status=405, mimetype='application/json')


@app.route("/frResults")
def fr_results():
    """Returns a HTTP streaming response of the recently detected names, their scores, and bounding boxes"""

    return Response(
        fr_instance.start_detection_broadcast(), mimetype="application/json"
    )


@app.route("/submit_settings", methods=["POST"])
def submit():
    """Handles form submission to adjust FR settings, subsequently redirects to settings page"""

    new_settings = {
        "threshold": float(request.form.get(
            "threshold", fr_instance.fr_settings["threshold"]
        )),
        "holding_time": float(
            request.form.get("holding_time", fr_instance.fr_settings["holding_time"]))
        ,
        "use_differentiator": "use_differentiator" in request.form,
        "threshold_lenient_diff": float(request.form.get(
            "threshold_lenient_diff", fr_instance.fr_settings["threshold_lenient_diff"]
        )),
        "similarity_gap": float(request.form.get(
            "similarity_gap", fr_instance.fr_settings["similarity_gap"]
        )),
        "use_persistor": "use_persistor" in request.form,
        "threshold_prev": float(request.form.get(
            "threshold_prev", fr_instance.fr_settings["threshold_prev"]
        )),
        "threshold_iou": float(request.form.get(
            "threshold_iou", fr_instance.fr_settings["threshold_iou"]
        )),
        "threshold_lenient_pers": float(request.form.get(
            "threshold_lenient_pers", fr_instance.fr_settings["threshold_lenient_pers"]
        ))
    }

    fr_instance.adjust_values(new_settings)
    return redirect(url_for('settings'))


@app.route("/")
def index():
    """Renders home page which includes the live feed (with bounding boxes) and a detection list"""

    return render_template("index.html")

@app.route("/seats")
def seats():
    """Renders a page with the seating plan, with seats that light up upon detection."""

    return render_template("seats.html")

@app.route("/old_layout")
def oldlayout():
    """Renders the old layout with video on the left and detection list on the right"""

    return render_template("old_layout.html")

@app.route('/data/<path:filename>')
def serve_data(filename):
    """Serve files from the data directory"""
    return send_from_directory('data', filename)

@app.route("/settings")
def settings():
    """Renders settings page"""

    return render_template(
        "settings.html",
        threshold=fr_instance.fr_settings["threshold"],
        holding_time=fr_instance.fr_settings["holding_time"],
        use_differentiator=fr_instance.fr_settings["use_differentiator"],
        threshold_lenient_diff=fr_instance.fr_settings["threshold_lenient_diff"],
        similarity_gap=fr_instance.fr_settings["similarity_gap"],
        use_persistor=fr_instance.fr_settings["use_persistor"],
        threshold_prev=fr_instance.fr_settings["threshold_prev"],
        threshold_iou=fr_instance.fr_settings["threshold_iou"],
        threshold_lenient_pers=fr_instance.fr_settings["threshold_lenient_pers"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Facial Recognition Program")

    # Arguments
    parser.add_argument(
        "-ip",
        "--ipaddress",
        type=str,
        help="IP address to host the app from",
        required=False,
        default="0.0.0.0",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=str,
        help="Port to host the app from",
        required=False,
        default="1333",
    )
    parser.add_argument(
        "-v",
        "--video",
        type=str,
        help="Enable the video feed (default true)",
        required=False,
        default="true",
    )

    args = parser.parse_args()

    signal.signal(signal.SIGINT, fr_instance.cleanup)
    app.run(debug=True, host=args.ipaddress, port=args.port, use_reloader=False)
    