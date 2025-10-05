# 🔥 YOLO Fire Detection System

Real-time fire and smoke detection using YOLOv8, trained on a custom dataset for automated monitoring in forests, warehouses, and industrial facilities.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLO](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)

## 🎯 Overview

This project uses YOLOv8 object detection to identify fire and smoke in real-time. The model is trained on a custom Roboflow dataset and achieves high-speed inference suitable for production deployment.

**Key Features:**
- ⚡ Real-time detection with YOLOv8s
- 🎯 Custom-trained on fire/smoke dataset
- 📊 80 training epochs, 640×640 image resolution
- 🚀 Easy deployment and inference

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/yolofire-detection.git
cd yolofire-detection
pip install -r requirements.txt
```

**Requirements:** `ultralytics`, `roboflow`, `torch>=2.0`, `opencv-python`

### Dataset Setup

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("ai-8pzau").project("continuous_fire-pqda2")
dataset = project.version(1).download("yolov8")
```

Get your API key from [Roboflow](https://roboflow.com/).

## 💻 Usage

### Training

```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
model.train(
    data='continuous_fire-1/data.yaml',
    epochs=80,
    imgsz=640,
    plots=True
)
```

Weights saved to: `runs/detect/train/weights/best.pt`

### Inference

```python
model = YOLO('runs/detect/train3/weights/best.pt')

# Single image
results = model.predict(source='image.jpg', conf=0.25, save=True)

# Video
results = model.predict(source='video.mp4', conf=0.25, save=True)

# Webcam (real-time)
results = model.predict(source=0, conf=0.25, show=True)
```

Results saved to: `runs/detect/predict/`

## 📊 Results

Training outputs include:
- Loss and mAP curves
- Confusion matrix
- Precision-Recall metrics
- Sample predictions

View results:
```python
from IPython.display import Image
Image(filename='runs/detect/train/results.png', width=600)
```

## 📁 Project Structure

```
yolofire-detection/
├── yolofire.ipynb          # Main training notebook
├── continuous_fire-1/       # Dataset (auto-downloaded)
└── runs/detect/
    ├── train/weights/      # Model weights
    └── predict/            # Inference results
```

## 🔮 Future Improvements

- Deploy as REST API
- Real-time alert system
- Mobile/edge device optimization
- Multi-camera integration

