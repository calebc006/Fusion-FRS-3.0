# Real-time Facial Recognition System 3.0


## Table of Contents

- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Description

This repository contains all the source code for Fusion FRS 3.0 (released Feb 2026). 

FRS 3.0 is a web application for real-time facial recognition. Applications include attendance taking for events and and army/unit showcases.

The core FRS software is found in the `/simplifry` directory. **SimpliFRy** is a locally-hosted web application built using python 3.10 and [Flask](https://github.com/pallets/flask). It makes use of the [insightface](https://github.com/deepinsight/insightface) `buffalo_l` model by deepinsight for face detection and generation of embeddings and the [voyager](https://github.com/spotify/voyager) library by Spotify for approximate-KNN search. More details can be found in the SimpliFRy [ReadME](./simpliFRy/ReadME.md).

Alongside simpliFRy is `/gotendance`. **gotendance** is also a locally-hosted web application, built using Golang. It is an attendance-tracking app, intended as a companion to simpliFRy for attendance taking. More details can be found in the Gotendance [ReadME](./gotendance/ReadME.md).

Additionally, there is a showcase version of SimpliFRy found in the `/interactive` directory. Much of the backend is similar, but this application allows users to capture their faces and add themselves to the database while the it is running.

We acknowledge the past versions of FRS, built by our seniors in Fusion Coy; much in this project is owed to their efforts.
- Ver 2.2: https://github.com/Cooleststar/FUSION-FR
- Ver 2.1: https://github.com/plainpotato/FR-FUSION
- Ver 2.0: https://github.com/CJBuzz/Real-time-FRS-2.0
- Ver 1.0: https://github.com/CJBuzz/FRS

---

## Features

As compared to previous iterations, **FRS 3.0** has the following improvements:

- 


## Installation

For more information on installation, refer to the following:
- [simpliFRy](./simpliFRy/ReadME.md#installation)
- [gotendance](./gotendance/ReadME.md#installation)



## License 

This project is [licensed](./LICENSE) under Apache 2.0.

Copyright 2026 Fusion Company, 11C4I Battalion, Singapore Armed Forces 


## Acknowledgements

I would like to extend my gratitude to the following people and resources:
- [**Cooleststar**](https://github.com/cooleststar): The long-time maintainer of FRS v2.0-v2.2. Without his efforts this project could not have survived for my generation to see.
- [**Ruihongc**](https://github.com/ruihongc): Suggesting the use of Spotify's Voyager led to much better faster embedding search compared to previous methods.
- [**BabyWaffles**](https://github.com/BabyWaffles): Dockerization of simpliFRy was made possible by his extensive help.
- [**Tabler Icons**](https://tabler.io/icons): Multiple icons from Tabler were used in the UI of both simpliFRy and Gotendance.
- [**CJBuzz**](https://github.com/CJBuzz): The first few developers that made this project happen.
- [**plainpotato**](https://github.com/plainpotato): Provided support for a big part of the project.
- [**bryannyp**](https://github.com/bryannyp): Build the UI for the Gotendance app
