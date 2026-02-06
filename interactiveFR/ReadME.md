# InteractiveFR

![Project Logo](static/images/fusionlogo.png)

---

## Table of Contents

- [InteractiveFR](#interactivefr)
    - [Table of Contents](#table-of-contents)
    - [Description](#description)
    - [Install](#install)
        - [Prerequisites](#prerequisites)
        - [Docker](#docker)
        - [Local](#local)
    - [Run](#run)
        - [Docker](#docker-1)
        - [Local](#local-1)
    - [Pages](#pages)
    - [Data](#data)
        - [`.env`](#env)

---

## Description

**InteractiveFR** is the interactive UI for real-time facial recognition. It runs on Flask, streams an RTSP or local camera feed, shows live detections, and lets you capture new faces into `data/captures`.

---

## Install

### Prerequisites

- [Python 3.10](https://www.python.org/downloads/)
- [FFmpeg 8.0.1](https://www.ffmpeg.org/download.html)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (optional)

### Docker

```bash
docker compose build
```

### Local

```bash
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## Run

### Docker

```bash
docker compose up --no-build
```

### Local

```bash
python app.py
```

Open <http://127.0.0.1:1334>.

---

## Pages

- `/` (home / stream selection)
- `/interactive` (live detections + capture)
- `/references` (browse captures)
- `/settings` (FR tuning)

---

## Data

- Captures are stored in `data/captures/<NAME>/` and served at `/data/...`.
- Logs are written to `data/logs/` per session.

### `.env`

```
APP_IP=0.0.0.0
APP_PORT=1334
APP_VIDEO=true
APP_ENV=production
```
