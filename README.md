# Adjustable-Resolution Neural Style Transfer and Depth Mapping Realtime Video Feedback

## Overview

This repository contains code for live video NST and depth mapping at adjustable resolutions. It includes:
- **High-resolution video processing** for both foreground and background.
- **Low-resolution video processing** for both foreground and background.
- **Demo application** to dynamically adjust video feedback resolution through a simple user interface.

## Features

- **`test.py`**: Processes video with high-resolution for both the foreground and background.
- **`demo/app.py`**: A user-friendly interface to adjust and generate NST & DM video feedback with dynamic resolution changes.

## Installation and Usage
(on Linux)

### Clone the Repository
1. Download .zip file and extract
2. cd to extracted folder

OR

```sh
git clone https://github.com/SophiaClifton/computer-vision-project.git
cd computer-vision-project
```

### Install Dependencies  
```sh
pip install -r requirements.txt
```

### Running Tests  
```sh
cd demo
python3 demo.py N forground_res background_res

or 

python3 demo.py N forground_res background_res camera

N: FPS processed
forground_res: resolution of forground (high or low)
background_res: resolution of background (high or low)
camera: specify recording device

example: python3 demo.py 3 high low
```

### Running the Demo  
```sh
cd demo
python3 app.py
```
Open **localhost** link in terminal to open demo page on browser.

---


