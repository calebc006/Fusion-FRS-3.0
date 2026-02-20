<!-- omit from toc -->
# SimpliFRy

![Project Logo](static/images/favicon.png)

---

<!-- omit from toc -->
## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Pages](#pages)
- [`/data`](#data)
- [`.env`](#env)
- [Settings](#settings)
- [Data Preparation](#data-preparation)

---

## Description

**SimpliFRy** is the core component for deployment of **FRS 3**. It is a locally-hosted web application built using Python 3.10 and [Flask](https://github.com/pallets/flask), and makes use of the [InsightFace](https://github.com/deepinsight/insightface) library by deepinsight for face detection and generation of embeddings as well as the [Voyager](https://github.com/spotify/voyager) library by Spotify for ANN search.

If you are a developer and would like to understand more about how SimpliFRy works, refer to the [Developer Guide](../Developer%20Guide.md)

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

1. First, create a Python virtual environment in the `simpliFRy` directory

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

Access the application at <http://localhost:1333> (preferably using a Chromium browser)


## Pages

The pages in this application are:

- `localhost:1333`
- `localhost:1333/seats`
- `localhost:1333/old_layout`
- `localhost:1333/settings`


## `/data` 

The `data` directory in `simpliFRy` is a **volume mount** as it is volume mounted to the `/app/data` directory within the docker container. Hence, it is the primary way to pass information to and from the container.

Everytime the app is started, a new `.logs` file will be created in the `/data/logs` directory. It will list key actions undertaken by the simpliFRy app (and any error messages) in that session.

## `.env` 

To configure the Python web server, create a file named `.env` in the base directory. Its format should be:

```
APP_IP = 0.0.0.0
APP_PORT = 1333
APP_VIDEO = true
APP_ENV = production
```

## Settings

To adjust parameters used in the FR algorithm, go to <http://localhost:1333/settings>

![simpliFRy settings page](assets/settings_page.png)

For more information on the parameters, click [here](../Developer%20Guide.md#configuration)


## Data Preparation

To conduct facial recognition, you need to load images of people you wish to be recognised into simpliFRy. Each person can have 1 or more pictures.

1. From the `simpliFRy/data` folder (created automatically when starting the app), create a new directory with all the images of the people you wish to be detected.

2. In the `simpliFRy/data` folder, create a JSON file (name it whatever you want) that maps the image file name with the name of the person to be recognised.

For example, if `john_doe1.jpg` and `john_doe2.png` are pictures of 'John Doe' while `jane_smith.png` is a picture of 'Jane Smith', and all images are in a folder called `pictures`, this is the directory structure.

```
simpliFRy/
├── data/
|   ├── logs/
|   ├── flags/
|   |   ├── singapore_flag.png
|   ├── pictures/
|   |   ├── john_doe1.jpg
|   |   ├── john_doe2.png
|   |   └── caleb.png
|   └── namelist.json
└── other files and folders
```

`namelist.json` would look like this

```json
{
    "img_folder_path": "pictures",
    "flag_folder_path": "flags",
    "details": [
        {
            "name": "John Doe",
            "images": ["john_doe1.jpg", "john_doe2.png"],
            "country_flag": "singapore_flag.png",
            "description": "someone",
            "table": "T1",
            "tags": ["Army", "DIS"],
            "priority": 2
        },
        {
            "name": "3SG CALEB CHIA",
            "images": ["caleb.png"],
            "country_flag": "singapore_flag.png",
            "description": "someone else",
            "table": "VIP",
            "tags": ["Air Force"],
            "priority": 1
        }
    ]
}
```

- `img_folder_path` (optional) will the path to the folder with all the faces relative to `/data`

- `flag_folder_path` (optional) will the path to the folder with all the country flags relative to `/data`

- `priority` (optional) determines the sorting order in detection lists (lower number = shown first). If two people have the same priority or no priority is set, they are sorted alphabetically.

- **The fields `flag_folder_path`, `country_flag`, `description`, `table`, `tags`, `priority` can be omitted if the deployment does not require these information.**