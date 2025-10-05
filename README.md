# 🔥 YOLO Fire Detection System

A real-time fire and smoke detection system using YOLOv8 (You Only Look Once) object detection model, trained on a custom dataset for proactive monitoring in critical environments such as forests, warehouses, and industrial facilities.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLO](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Project Overview

This project leverages the YOLOv8 architecture from Ultralytics to detect fire and smoke in real-time, enabling early warning systems for fire prevention and safety monitoring. The model is trained on a custom dataset from Roboflow, optimized for deployment in various environments.

### Key Applications
- 🌲 **Forest Fire Detection** - Early warning systems for wildfire prevention
- 🏭 **Industrial Safety** - Warehouse and factory fire monitoring
- 🏢 **Building Security** - Automated fire detection in commercial spaces
- 📹 **Surveillance Integration** - Real-time video stream analysis

## ✨ Key Features

- ⚡ **Real-Time Detection** - High-speed inference for immediate fire/smoke identification
- 🎯 **High Accuracy** - YOLOv8 state-of-the-art object detection architecture
- 🔧 **Custom Training Pipeline** - Complete workflow from data preparation to deployment
- 📊 **Comprehensive Metrics** - Training visualization and performance evaluation
- 🚀 **Easy Deployment** - Simple inference API for integration into existing systems

## 📊 Model Performance

- **Model:** YOLOv8s (Small variant)
- **Training Epochs:** 80
- **Image Size:** 640×640
- **Dataset:** Custom fire/smoke dataset from Roboflow
- **Framework:** Ultralytics YOLOv8

Training results include visualizations of:
- mAP (mean Average Precision) curves
- Loss reduction over epochs
- Precision-Recall curves
- Sample detection outputs

## 🚀 Getting Started

### Prerequisites

- **Python:** 3.8 or higher
- **GPU:** CUDA-enabled GPU recommended (e.g., Tesla T4, RTX series)
- **Operating System:** Windows, Linux, or macOS
- **RAM:** Minimum 8GB (16GB+ recommended for training)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/yolofire-detection.git
cd yolofire-detection
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Core dependencies include:**
- `ultralytics>=8.0.0` - YOLOv8 framework
- `roboflow>=1.0.0` - Dataset management
- `torch>=2.0.0` - Deep learning backend
- `opencv-python` - Image processing
- `matplotlib` - Visualization

3. **Verify installation:**
```bash
python -c "from ultralytics import YOLO; print('✓ Installation successful!')"
```

## 📁 Dataset Setup

The dataset is automatically downloaded from Roboflow during the training process.

### Automatic Download (Recommended)

1. **Obtain your Roboflow API Key:**
   - Sign up at [Roboflow](https://roboflow.com/)
   - Navigate to your workspace settings
   - Copy your API key

2. **Run the dataset preparation cells in `yolofire.ipynb`:**

```python
from roboflow import Roboflow

# Initialize Roboflow with your API key
rf = Roboflow(api_key="YOUR_API_KEY_HERE")

# Load the fire detection project
project = rf.workspace("ai-8pzau").project("continuous_fire-pqda2")
version = project.version(1)

# Download dataset in YOLOv8 format
dataset = version.download("yolov8")
```

The dataset will be automatically organized in the correct structure for training.

## 💻 Usage

All implementation details are documented in the `yolofire.ipynb` notebook.

### 1. Training the Model

**Basic Training:**
```python
from ultralytics import YOLO

# Load pre-trained YOLOv8s model
model = YOLO('yolov8s.pt')

# Train on custom fire detection dataset
model.train(
    data='/content/continuous_fire-1/data.yaml',
    epochs=80,
    imgsz=640,
    plots=True  # Generate training visualizations
)
```

**Output Location:**
- Trained weights are saved to: `runs/detect/train[x]/weights/best.pt`
- Training plots and metrics: `runs/detect/train[x]/results.png`

### 2. Validation

Evaluate model performance on the validation set:

```python
# Validate the trained model
metrics = model.val()

print(f"mAP@0.5: {metrics.box.map50:.3f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
```

### 3. Inference & Prediction

**Single Image Detection:**
```python
from ultralytics import YOLO

# Load your trained model
model = YOLO('runs/detect/train3/weights/best.pt')

# Run prediction on a single image
results = model.predict(
    source='path/to/test/image.jpg',
    conf=0.25,  # Confidence threshold
    save=True   # Save annotated results
)
```

**Batch Detection (Multiple Images):**
```python
# Predict on all images in a folder
results = model.predict(
    source='path/to/images/folder/',
    conf=0.25,
    save=True
)
```

**Video Detection:**
```python
# Detect fire in video streams
results = model.predict(
    source='path/to/video.mp4',
    conf=0.25,
    save=True,
    stream=True  # Process frame by frame
)
```

**Real-Time Webcam Detection:**
```python
# Use webcam for live detection
results = model.predict(
    source=0,      # Webcam device ID
    conf=0.25,
    show=True      # Display live results
)
```

**Output Location:**
- Prediction results are saved to: `runs/detect/predict[x]/`
- Annotated images with bounding boxes and labels

## 📂 Project Structure

```
yolofire-detection/
│
├── yolofire.ipynb           # Main notebook with full pipeline
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
│
├── continuous_fire-1/      # Dataset (auto-downloaded)
│   ├── data.yaml          # Dataset configuration
│   ├── train/             # Training images & labels
│   ├── valid/             # Validation images & labels
│   └── test/              # Test images & labels
│
└── runs/                   # Training & inference outputs
    └── detect/
        ├── train/         # Training results
        │   └── weights/   # Model weights (best.pt, last.pt)
        └── predict/       # Inference results
```

## 📊 Viewing Training Results

After training, view comprehensive metrics:

```python
from IPython.display import Image

# Display training results
Image(filename='runs/detect/train/results.png', width=800)
```

Training visualizations include:
- Loss curves (box, cls, dfl)
- Precision-Recall curves
- Confusion matrix
- mAP metrics over epochs

## 🔧 Advanced Configuration

### Custom Training Parameters

```python
model.train(
    data='continuous_fire-1/data.yaml',
    epochs=80,
    imgsz=640,
    batch=16,           # Batch size (adjust based on GPU memory)
    patience=20,        # Early stopping patience
    optimizer='Adam',   # Optimizer choice
    lr0=0.01,          # Initial learning rate
    device=0,          # GPU device (0 for first GPU, 'cpu' for CPU)
    workers=8,         # Number of data loader workers
    cache=True,        # Cache images for faster training
    save_period=10     # Save checkpoint every N epochs
)
```

### Inference Parameters

```python
results = model.predict(
    source='image.jpg',
    conf=0.25,         # Confidence threshold (0-1)
    iou=0.45,          # NMS IoU threshold
    max_det=100,       # Maximum detections per image
    classes=[0, 1],    # Filter by class IDs
    save=True,         # Save results
    save_txt=True,     # Save results as .txt
    save_conf=True,    # Save confidence scores
    line_width=2,      # Bounding box line width
    show_labels=True,  # Show class labels
    show_conf=True     # Show confidence scores
)
```

## 🎯 Use Cases

### 1. Forest Fire Monitoring
Deploy on surveillance cameras in forests for early wildfire detection.

### 2. Industrial Safety
Integrate with factory CCTV systems for automated fire alerts.

### 3. Smart Building Systems
Connect to building management systems for enhanced fire safety.

### 4. Emergency Response
Provide real-time detection data to fire departments and emergency services.

## 🔮 Future Improvements

- [ ] Deploy as REST API using Flask/FastAPI
- [ ] Create web-based demo interface
- [ ] Add mobile app integration
- [ ] Implement real-time alert system (SMS/Email)
- [ ] Optimize model for edge devices (Jetson Nano, Raspberry Pi)
- [ ] Add smoke intensity classification
- [ ] Multi-camera system integration
- [ ] Cloud deployment (AWS, Azure, GCP)

## 🐛 Troubleshooting

**CUDA Out of Memory:**
- Reduce batch size: `batch=8` or `batch=4`
- Use smaller image size: `imgsz=416`

**Slow Training:**
- Enable caching: `cache=True`
- Increase workers: `workers=8`
- Use mixed precision: `amp=True`

**Low Detection Accuracy:**
- Increase training epochs
- Adjust confidence threshold
- Add more training data
- Use data augmentation

## 📝 License

This project is available for educational and portfolio purposes.

## 🙏 Acknowledgments

- **Ultralytics** - YOLOv8 framework
- **Roboflow** - Dataset management and annotation
- **Community** - Open-source computer vision community

## 📧 Contact

For questions, suggestions, or collaboration opportunities:
- **GitHub:** [@yourusername](https://github.com/yourusername)
- **Email:** your.email@example.com
- **LinkedIn:** [Your Name](https://linkedin.com/in/yourprofile)

---

⭐ **Star this repository if you find it helpful!**

Made with ❤️ for fire safety and computer vision
