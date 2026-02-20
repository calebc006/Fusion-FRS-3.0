# Real-time Facial Recognition System 3

## Description

This repository contains all the source code for Fusion FRS 3 (initially released Feb 2026). 

FRS 3.0 is a web application for real-time facial recognition. Used for attendance taking for events and army/unit showcases.

`/simplifry` contains the core FRS software used for deployment. **SimpliFRy** is a locally-hosted web application built using python 3.10 and [Flask](https://github.com/pallets/flask). It makes use of the [`buffalo_l`](https://github.com/deepinsight/insightface/blob/master/python-package/README.md#model-zoo) model by [InsightFace](https://github.com/deepinsight/insightface) for face detection and generation of embeddings, while Spotify's [Voyager](https://github.com/spotify/voyager) is used for approximate-nearest-neighbor search. 

`/gotendance` is the companion to **SimpliFRy**. It is an attendance-tracking app, intended as a companion to simpliFRy for attendance taking. 

`/interactiveFR` is a showcase version of FRS. Much of the backend is similar to **SimpliFRy**, but it allows users to capture their faces and add themselves to the database while the app is running.

For more technical details, refer to the [Developer Guide](./Developer%20Guide.md)

We acknowledge the past versions of FRS, built by our seniors in Fusion Coy; much in this project is owed to their efforts. (unfortunately some of these links are now inactive)
- v2.2: https://github.com/Cooleststar/FUSION-FR
- v2.1: https://github.com/plainpotato/FR-FUSION
- v2.0: https://github.com/CJBuzz/Real-time-FRS-2.0
- v1.0: https://github.com/CJBuzz/FRS

---

## New Features

As compared to previous iterations, **FRS 3** has the following key improvements:

- Massively improved back-end performance and reliability via code optimizations. Eliminated previous issues of lag, crashing and instability.
- Added optional input configuration for users: description, table number (for new seating feature), sorting index (for priority of display), filter tag(s).
- Introduced InteractiveFR, using a similar backend as SimpliFRy.

For a detailed list of changes, refer to [changelog.md](./changelog.md) 

## Installation

For more information on installation, refer to the following:
- [SimpliFRy](./simpliFRy/ReadME.md#installation)
- [Gotendance](./gotendance/ReadME.md#installation)
- [InteractiveFR](./interactiveFR/ReadME.md#install-and-run)



## License 

This project is [licensed](./LICENSE) under Apache 2.0.

&copy; Fusion Company, 11C4I Battalion, Singapore Armed Forces 


## Acknowledgements

- [**Cooleststar**](https://github.com/cooleststar): The long-time maintainer of FRS v2.0-v2.2. Without his efforts this project could not have survived for my generation to see.
- [**Ruihongc**](https://github.com/ruihongc): Suggesting the use of Spotify's Voyager which led to much faster embedding search compared to previous methods.
- [**BabyWaffles**](https://github.com/BabyWaffles): Dockerization of simpliFRy was made possible by his extensive help.
- [**CJBuzz**](https://github.com/CJBuzz): One of the first few developers of this project.
- [**plainpotato**](https://github.com/plainpotato): Provided support for a big part of the project.
- [**bryannyp**](https://github.com/bryannyp): Built the UI for the Gotendance app
