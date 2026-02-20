<!-- omit from toc -->
# InteractiveFR

<img src="static/images/fusionlogo.png" width="200" alt="Project Logo" />

---

<!-- omit from toc -->
## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Pages](#pages)
- [Data](#data)
- [`.env`](#env)
- [Settings](#settings)

---

## Description

**InteractiveFR** is the interactive UI for real-time facial recognition. It runs on Flask, streams an RTSP or local camera feed, shows live detections, and lets you capture new faces into `data/captures`.

---

## Installation

### Prerequisites

- [Python 3.10](https://www.python.org/downloads/) (or later)
- [FFmpeg 8.0.1](https://www.ffmpeg.org/download.html) (or later)
- [Docker](https://www.docker.com/products/docker-desktop/) (optional)
- [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for GPU usage within docker container)

The `buffalo_l` model will auto-install if it is not found on your system. Alternatively, you can install it manually by following the instructions [here](https://github.com/deepinsight/insightface/blob/master/python-package/README.md#model-zoo). A copy of `buffalo_l.zip` can be found in the root directory of this repository.

### Download Source Code

1. Clone the repository:

    ```bash
    git clone https://github.com/calebc006/Fusion-FRS-3.0.git
    ```

2. Navigate to simpliFRy directory

    ```bash
    cd simpliFRy
    ```

### Installation via Docker

1. Build Docker Image
   
    ```bash
    docker compose build
    ```

### Installation via Virtual Environment

1. First, create a Python virtual environment in the `interactiveFR` directory

    ```bash
    py -m venv venv
    ```

2. Activate it

    ```bash
    venv\Scripts\activate # use source venv/bin/activate for linux and macOS
    ```

3. Install the requirements with pip

    ```bash
    pip install -r requirements.txt # this may take some time!
    ```

---

## Usage

### Docker

To start up a new container from the command line, run the following command from the `simpliFRy` directory

```bash
docker compose up
```

If you already have an existing container, you can simply start by pressing the play button in the Docker Desktop application.

### Virtual Environment

Activate the Python virtual environment and run the `app.py` script

```bash
venv\Scripts\activate # use source venv/bin/activate for linux and macOS
```

```bash
py app.py
```

Access the application at <http://localhost:1334> (preferably using a Chromium browser)



## Pages

- `/` (stream selection)
- `/interactive` (live video feed + capture UI + reference modal)
- `/references` (browse references in separate page)
- `/settings` (FR tuning)

---

## Data
- Captured images are stored in `data/captures/<NAME>/<capture>.jpg`
- Average embedding for each person is stored in `data/captures/<NAME>/embedding_avg.npy`
- Logs for each session are written to `data/logs` 
- The whole `/data` directory is a docker volume so its contents are persisted when using docker

## `.env`
Refer to `.env.example`
```py
APP_IP=0.0.0.0
APP_PORT=1334
APP_ENV=production # "production" or "development"

# Out of 100. Higher is better, but costs more latency and bandwidth
STREAM_JPG_QUALITY=90 

# Input resolution for inference; 4:3 aspect ratio recommended for HIKVISION cameras
INFERENCE_WIDTH=640
INFERENCE_HEIGHT=480
```

## Settings

To adjust parameters used in the FR algorithm, go to <http://localhost:1334/settings>

![simpliFRy settings page](../simpliFRy/assets/settings_page.png)

For more information on the parameters, click [here](../Developer%20Guide.md#configuration)

