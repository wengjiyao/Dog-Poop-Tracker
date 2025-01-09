# Dog Pooping Detector

[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

A small system (Python-based) to monitor your dog in a backyard via a camera feed.  
It automatically detects:
1. Whether there is a dog in the frame (using YOLOv8).
2. Whether the dog is pooping (using a MobileNetV2 classifier).
3. If pooping is detected, the system flashes the screen, saves a snapshot to disk, and briefly freezes the frame.  
   This helps you locate and clean up after your dog.  
   
**This project is open-sourced under the Apache 2.0 License.**

---

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Training Your Own Model](#training-your-own-model)
6. [License](#license)

---

## Features
- **Real-time Dog Detection** using [Ultralytics YOLOv8](https://docs.ultralytics.com).
- **Poop vs. Not-Poop Classification** using a custom MobileNetV2 model.
- **Screenshot & Alert**: Automatically captures a snapshot and plays a camera shutter sound when a poop event is detected.
- **Cooldown Mechanism** to avoid repeated false positives in a short period.

---

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/wengjiyao/Dog-Poop-Tracker.git
cd dog-pooping-detector
```
## Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

---

## Usage
python track_n_detect.py

---

## Dog Poop Detection Model
Dog Poop Detection uses MobileNetV2. For training:
Jupyter Notebook Notebook [dogpoop_mobilenetv2](https://www.kaggle.com/code/wengjiyao/dogpoop-mobilenetv2)
Dataset [Dog Poop Dataset](https://www.kaggle.com/datasets/wengjiyao/dog-poop-dataset]

---

## More Projects
[More Dog-Inspired Projects](https://dogumentary.pro)