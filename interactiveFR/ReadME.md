# InteractiveFR

![Project Logo](static/images/fusionlogo.png)

---

## Table of Contents

- [InteractiveFR](#interactivefr)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Install and Run](#install-and-run)
    - [Prerequisites](#prerequisites)
    - [Docker](#docker)
    - [Local](#local)
  - [Pages](#pages)
  - [Data](#data)
    - [`.env`](#env)

---

## Description

**InteractiveFR** is the interactive UI for real-time facial recognition. It runs on Flask, streams an RTSP or local camera feed, shows live detections, and lets you capture new faces into `data/captures`.

---

## Install and Run

### Prerequisites

- [Python 3.10](https://www.python.org/downloads/) (support not confirmed for later versions)
- [FFmpeg 8.0.1](https://www.ffmpeg.org/download.html) (support not confirmed for later versions)
- [Docker](https://www.docker.com/products/docker-desktop/) (optional)

### Docker

```bash
docker compose build

docker compose up 
```

### Local

```bash
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python app.py
```
Open <http://localhost:1334>.


## Pages

- `/` (home / stream selection)
- `/interactive` (live detections + capture)
- `/references` (browse captures)
- `/settings` (FR tuning)

---

## Data
- Captures are stored in `data/captures/<NAME>/` and served at `/data/...`
- Logs are written to `data/logs/` per session
- The whole `/data` directory is a docker volume so its contents are persisted if using docker

### `.env`
Refer to .env.example
```py
APP_IP=0.0.0.0
APP_PORT=1334
APP_ENV=production # "production" or "development"

# Video config (what ffmpeg uses, and what is streamed to /vidFeed)
WIDTH=1920 
HEIGHT=1080
FPS=25

# Input resolution for inferencez` 
INFERENCE_WIDTH=640
INFERENCE_HEIGHT=480

```
