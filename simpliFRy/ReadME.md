# SimpliFRy

![Project Logo](static/favicon.png)

---

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

---

## Description

**SimpliFRy** is the core component of [Real-time FRS 2.0](https://github.com/CJBuzz/Real-time-FRS-2.0) and is the software that handles Real-time Facial Recognition. It is a locally-hosted web application built using python 3.10 and [Flask](https://github.com/pallets/flask), and makes use of the [insightface](https://github.com/deepinsight/insightface) library by deepinsight for face detection and generation of embeddings as well as the [voyager](https://github.com/spotify/voyager) library by Spotify for K-Nearest Neighbour search.

If you are a developer and would like to understand more about how simpliFRy works, refer to the [Developer Guide](Developer%20Guide.md)

---

## Installation

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for GPU usage within docker container, which is very recommended; install via WSL)

### Installation via Docker

1. Clone the repository:

   ```bash
   git clone https://github.com/CJBuzz/Real-time-FRS-2.0.git
   ```

2. Navigate to simpliFRy directory

   ```bash
   cd simpliFRy
   ```

3. Build Docker Image
   ```bash
   docker compose build
   ```

### Installation by other means

It is highly recommended to install and run simpliFRy via Docker, else there is a need to install dependencies such as CUDA and cuDNN separately. It is quite troublesome to achieve version compatibility for CUDA, cuDNN, pyTorch and onnxruntime. However, if you insist on refusing to use Docker, below are the versions that worked for me.

- CUDA 11.8
- cuDNN 8.9.2.26
- pyTorch 2.1.2+cu118
- onnxruntime 1.18.1

In addition, if you are using windows, there is a need to install CMake and Microsoft Visual Studio C++ built tools separately.

Afterwards, you can create a virtual environment in the `simpliFRy` directory

```bash
py -m venv venv
```

Activate it

```bash
venv\Scripts\activate # on windows, use source venv/bin/activate for linux and macOS
```

Install the requirements with pip

```bash
pip install -r requirements.txt
```

---

## Usage

### Docker

1. Start Docker Desktop application

2. Run the container

If you wish to create a new container, run the following command

```bash
docker run -p 1333:1333 -v "C:\Users\Admin\Desktop\FUSION-FR\simpliFRy\data:/app/data" --gpus all simplifry-simplifry
```

Alternatively, you can run docker compose from the `simpliFRy` directory

```bash
docker compose up --no-build
```

Access the web UI at <http://127.0.0.1:1333> (preferably using a Chromium browser)

If you want to run multiple containers, run one container at with the port argument `1333:1333` (port 1333) and another with the argument `2000:1333` (this means port 2000, you can use another port number).

This means the 2nd container can be accessed at <http://127.0.0.1:2000>.

Alternatively, you can edit `docker-compose.yml` and run

```bash
docker compose up --no-build
```

If you already have an existing container, you can simply start it from the Docker Desktop application.

### Without Docker (for development)

Run the `app.py` script in the simpliFRy virtual environment

```bash
py app.py
```

### Data Preparation

To conduct facial recognition, you need to load images of people you wish to be recognised into simpliFRy. Each person can have 1 or more pictures.

1. From the `simpliFRy/data` folder (created automatically when starting the app), create a new directory with all the images of the people you wish to be detected.

2. In the `simpliFRy/data` folder, create a JSON file (name it whatever you want) that maps the image file name with the name of the person to be recognised. Format it as shown below:

```json
// example.json
{
  "img_folder_path": "path/to/image/folder",
  "flag_folder_path": "path/to/flag/folder",
  "details": [
    {
      "name": "Person One",
      "images": ["image1.jpg", "image2.png"],
      "country_flag": "singapore_flag.png",
      "description": "Chief of Army, Redland Armed Forces",
      "table": "T1"
    },
    {
      "name": "Person Two",
      "images": ["image3.jpg", "image4.png"],
      "country_flag": "brunei_flag.png",
      "description": "Janitor",
      "table": "T2"
    }
    // Other similar entries as above
  ]
}
```

`img_folder_path` will the path to the folder with all the faces relative to `/data`
`flag_folder_path` will the path to the folder with all the country flags relative to `/data`


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
|   |   └── jane_smith.png
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
      "table": "T1"
    },
    {
      "name": "Jane Smith",
      "images": ["jane_smith.png"],
      "country_flag": "singapore_flag.png",
      "description": "someone else",
      "table": "T2"
    }
  ]
}
```

**The fields `flag_folder_path`, `country_flag`, `description` and `table` can be omitted if the deployment does not require these information.**

### `/data` Directory

The `data` directory in `simpliFRy` is a **volume mount** as it is volume mounted to the `/app/data` directory within the docker container. Hence, it is the primary way to pass information to (e.g. folder of images, refer to [Data Preparation](#data-preparation) for more info) and from (logs) the container.

Everytime the app is started, a new `.logs` file will be created in the `/data/logs` directory. It will list key actions undertaken by the simpliFRy app (and any error messages) in that session.

## Pages

The pages in this application are:
- `localhost:1333`
- `localhost:1333/seats`
- `localhost:1333/old_layout`
- `localhost:1333/settings`

### `.env` file

To configure the Python web server, create a file named `.env` in the base directory. Its format should be:

```
APP_IP = 0.0.0.0
APP_PORT = 1333
APP_VIDEO = true
APP_ENV = production
```

### Settings

To adjust parameters used in the FR algorithm, go to <http://127.0.0.1:1333/settings>.

![simpliFRy settings page](assets/settings_page.png)

For more information on the parameters, click [here](Developer%20Guide.md#fr-settings).
