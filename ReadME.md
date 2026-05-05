# Real-time Facial Recognition System 3

<img src="assets/fusionlogo.png" width="200" alt="Project Logo" />


## Description

This repository contains all the source code for Fusion FRS 3 (initially released Feb 2026). 

FRS 3 is a web application for real-time facial recognition, used for attendance taking during events and army/unit showcases.

`/simplifry` contains the core FRS software used for deployment. **SimpliFRy** is a locally-hosted web application built using Python 3.10 and [Flask](https://github.com/pallets/flask). It makes use of the [`buffalo_l`](https://github.com/deepinsight/insightface/blob/master/python-package/README.md#model-zoo) model by [InsightFace](https://github.com/deepinsight/insightface) for face detection and generation of embeddings as well as Spotify's [Voyager](https://github.com/spotify/voyager) for approximate-nearest-neighbor search.

`/gotendance` is the companion to **SimpliFRy**. It is a lightweight attendance-tracking application built in Go that processes real-time results from SimpliFRy for automated attendance taking.

Also check out [InteractiveFR](google.com), a showcase version of FRS.

---

## Table of Contents

- [Features](#features)
- [Installation & Setup](#installation--setup-simplifry--gotendance)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Architecture](#architecture)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features

As compared to previous iterations, **FRS 3** has the following key improvements:

- Massively improved back-end performance and reliability via code optimizations. Eliminated previous issues of lag, crashing and instability.
- Added optional input configuration for users: description, table number (for new seating feature), sorting index (for priority of display), filter tag(s).
- Lightweight, containerized deployment with Docker Compose.
- Easy integration between SimpliFRy (recognition) and Gotendance (attendance tracking).

For a detailed list of changes, refer to the [changelog](./changelog.md).

---

## Installation & Setup (SimpliFRy + Gotendance)

### Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop/) and [Docker Compose](https://docs.docker.com/compose/install/)
- [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for GPU support in Docker)
- [Python 3.10+](https://www.python.org/downloads/) (for local development without Docker)
- [Go 1.23.0+](https://go.dev/doc/install) (for building Gotendance locally)
- [FFmpeg 8.0.1+ ](https://ffmpeg.org/download.html) 

### Quick Start with Docker Compose (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/calebc006/Fusion-FRS-3.git
   cd Fusion-FRS-3
   ```

2. Build the Docker images:
   ```bash
   docker compose build
   ```

3. Start both services:
   ```bash
   docker compose up
   ```

4. Access the applications:
   - **SimpliFRy**: http://localhost:1333
   - **Gotendance**: http://localhost:1500

### Docker Desktop View:

![alt text](./assets/docker.png)
Check that both containers are running! The containers can also be started from the Docker Desktop app.

### Local Development Setup

**SimpliFRy:**
```bash
cd simpliFRy
python -m venv venv
venv\Scripts\activate  # On macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python app.py
# Access at http://localhost:1333
```

**Gotendance:**
```bash
cd gotendance
go build
./gotendance.exe  # On Linux/macOS: ./gotendance
# Access at http://localhost:1500
```

---

## Usage


Once both containers are running,

#### SimpliFRy (Port 1333)
1. Open http://localhost:1333 in your browser
2. Load a personnel JSON file (see [Data Preparation](#data-preparation))
3. Follow instructions on the UI to connect to a video stream
4. Begin facial recognition!

#### Gotendance (Port 1500)
1. Open http://localhost:1500 in your browser
2. Load the same personnel dataset
3. Connect to SimpliFRy's result stream:
   - **URL**: `http://<my.ip.address>:1333/api/frResults`
   - Multiple result streams can be connected to Gotendance.
4. Gotendance will automatically track attendance based on detections from the connected SimpliFRy result streams

#### Connecting Services

Multiple instances of SimpliFRy (running of different computers) can be connected to the same instance of Gotendance, for centralized attendance tracking. 

| Scenario | URL to use in Gotendance |
|----------|-------------------------|
| Both in Docker on the same machine | `http://simplifry:1333/api/frResults` OR `http://<host.ip.address>:1333/api/frResults` |
| SimpliFRy on external machine | `http://<external.ip.address>:1333/api/frResults` |

---

## Data Preparation

To set up facial recognition, you need to prepare a JSON file mapping images to people. In the `simpliFRy/data` folder, 

1. Create a new folder (any name) containing all the images.
2. Create a JSON file (any name) that maps images to people (more details below).

### Example Directory Structure

```
simpliFRy/
├── data/
│   ├── logs/              # Auto-generated logs
│   ├── flags/             # Optional: flag images
│   ├── pictures/          # Your reference images
│   │   ├── john_doe1.jpg
│   │   ├── john_doe2.png
│   │   └── caleb.png
│   └── namelist.json      # Personnel mapping file
└── other files
```

### JSON Format
**Minimal format (required):**
```json
{
    "img_folder_path": "pictures",
    "details": [
        {
            "name": "John Doe",
            "images": ["john_doe1.jpg", "john_doe2.png"]
        },
        {
            "name": "3SG CALEB CHIA",
            "images": ["caleb.png"]
        }
    ]
}
```

**Extended format (optional features):**
```json
{
    "img_folder_path": "pictures",
    "flag_folder_path": "flags",
    "details": [
        {
            "name": "John Doe",
            "images": ["john_doe1.jpg", "john_doe2.png"],
            "description": "someone",
            "country_flag": "singapore_flag.png",
            "table": "T1",
            "tags": ["Army", "DIS"],
            "priority": 2
        }
    ]
}
```

| Parameter | Type | Description | Required? |
|-----------|--------|--------------------------|-------|
|`img_folder_path`| String | The path to the folder with all the user images relative to `/data`| Y |
|`flag_folder_path`| String | The path to the folder with all the country flags relative to `/data`| N |
|`name`| String | The display name of the user (must be unique!) | Y |
|`images`| List[String] | List of image names within `img_folder_path`| Y |
|`description`| List[String] | Optional description to be displayed alongside `name` | N |
|`country_flag`| String | Image name for the flag to be displayed on `/welcome` page | N |
|`table`| String | The table name that is used for the `/seats` table lighting functionality | N |
|`tags`| List[String] | Optional list of filter tags used in Gotendance | N |
|`priority`| Integer | Determines the sorting order in detection lists (lower number = shown first). If two people have the same priority or no priority is set, they are sorted alphabetically | N |

### Loading Data

1. **Via SimpliFRy UI**: Upload JSON file in settings page
2. **Via Gotendance UI**: Upload the same JSON file in home page
3. Both services use the same format, so you only need one JSON file

---

## Architecture

### SimpliFRy
- **Backend**: Python 3.10 + Flask
- **Face Detection**: InsightFace (buffalo_l model)
- **Embedding Search**: Spotify Voyager (ANN)
- **Output**: HTTP streaming JSON responses at `/api/frResults`

### Gotendance
- **Backend**: Go 1.23.0
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Input**: HTTP streaming from SimpliFRy
- **Output**: Attendance records in JSON format
- **No external runtime dependencies** (standalone binary)


---


## License 

This project is [licensed](./LICENSE) under Apache 2.0.

&copy; Fusion Company, 11C4I Battalion, Singapore Armed Forces 

---

## Acknowledgements

We acknowledge the past versions of FRS, built by our seniors in Fusion Coy; much in this project is owed to their efforts.
- v2.2: https://github.com/Cooleststar/FUSION-FR
- v2.1: https://github.com/plainpotato/FR-FUSION
- v2.0: https://github.com/CJBuzz/Real-time-FRS-2.0
- v1.0: https://github.com/CJBuzz/FRS

Special thanks to:
- [**Cooleststar**](https://github.com/cooleststar): The long-time maintainer of FRS v2.0-v2.2 and one of the main developers that supported the FRS v3.0. Without his efforts this project could not have survived for my generation to see. He was also the one who developed the Showcase version of the FRS.
- [**Ruihongc**](https://github.com/ruihongc): Suggesting the use of Spotify's Voyager which led to much faster embedding search compared to previous methods.
- [**BabyWaffles**](https://github.com/BabyWaffles): Dockerization of simpliFRy was made possible by his extensive help.
- [**CJBuzz**](https://github.com/CJBuzz): One of the first few developers of this project.
- [**plainpotato**](https://github.com/plainpotato): Provided support for a big part of the project.
- [**bryannyp**](https://github.com/bryannyp): Built the UI for the Gotendance app
