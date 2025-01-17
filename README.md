# Dog Pooping Detector

[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

A Python application that monitors your dog in the backyard using a camera feed, detecting and tracking when your dog poops.
It automatically detects:
1. Whether there is a dog in the frame (using YOLOv8).
2. Whether the dog is pooping (using a MobileNetV2 classifier).
3. If pooping is detected, the system flashes the screen, saves a snapshot to disk, and briefly freezes the frame.  
   This helps you locate and clean up after your dog.  
   
**This project is open-sourced under the Apache 2.0 License.**

---

## Table of Contents
1. [Features](#features)
2. [Clone This Repository](#clone-this-repository)
3. [Install Dependencies](#install-dependencies)
4. [Usage](#usage)
5. [Dog Poop Detection Model](#dog-poop-detection-model)
6. [Related Article](#related-article)
7. [More Dog-Inspired Projects](#more-dog-inspired-projects)
7. [License](#license)

---

## Features
- **Real-time Dog Detection** using [Ultralytics YOLOv8](https://docs.ultralytics.com).
- **Poop vs. Not-Poop Classification** using a custom MobileNetV2 model.
- **Screenshot & Alert**: Automatically captures a snapshot and plays a camera shutter sound when a poop event is detected.
- **Cooldown Mechanism** to avoid repeated false positives in a short period.

---

## Clone This Repository

```bash
git clone https://github.com/wengjiyao/Dog-Poop-Tracker.git
cd dog-pooping-detector
```

## Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage
```bash
python track_n_detect.py
```
---

## Dog Poop Detection Model
Dog Poop Detection uses MobileNetV2. For training:
Jupyter Notebook [dogpoop_mobilenetv2](https://www.kaggle.com/code/wengjiyao/dogpoop-mobilenetv2)
Dataset [Dog Poop Dataset](https://www.kaggle.com/datasets/wengjiyao/dog-poop-dataset)

---

## Related Article
Medium Article [Tracking Dog Poop in the Backyard: A Computer Vision Solution](https://medium.com/@j.y.weng/tracking-dog-poop-in-the-backyard-a-computer-vision-solution-12f0237affcf)

---

## More Dog-Inspired Projects
[More Dog-Inspired Projects](https://dogumentary.pro)

---

## License
This project is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
