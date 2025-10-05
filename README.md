🔥 YOLO Fire Detection System: Real-Time Fire & Smoke Detection1. Project OverviewThis repository hosts a high-performance YOLO (You Only Look Once) model dedicated to real-time detection of fire and smoke.The system is built using the Ultralytics framework and is ideal for integration into proactive monitoring systems for areas prone to fire hazards (e.g., remote forests, industrial warehouses).Key FeaturesReal-time Performance: Optimized for high-speed inference on live video streams.Custom Training Workflow: Complete training process documented in the yolofire.ipynb notebook.Roboflow Integration: Demonstrates efficient dataset management and preparation via the Roboflow API.2. Getting StartedPrerequisitesPython 3.8+GPU Access: A dedicated GPU (e.g., NVIDIA T4, as used in the notebook) is strongly recommended for rapid model training.InstallationClone the repository and install the required dependencies:git clone <YOUR-REPOSITORY-URL>
cd yolofire-detection
pip install -r requirements.txt
Inferred Dependencies: ultralytics, roboflow, torch>=2.0Data SetupThe model is trained on a custom dataset managed through Roboflow.Obtain your Roboflow API Key.Execute the initial cells of the yolofire.ipynb notebook. The script will automatically download the dataset and structure it in the Ultralytics format (data.yaml).3. UsageThe complete workflow, from data preparation to detection, is executed within the yolofire.ipynb notebook.3.1. Training the ModelThe training process uses the downloaded dataset configuration:from ultralytics import YOLO

# Initialize the desired base model (e.g., YOLOv8s)
model = YOLO('yolov8s.pt')

# Start training
model.train(
    data='/content/continuous_fire-1/data.yaml',  # Path to Roboflow data configuration
    epochs=50,
    imgsz=640,
)
Output: The trained weights (best.pt) are saved within the /runs/detect/train[x]/weights/ directory.3.2. Running Inference (Detection)Load your custom trained weights and run predictions on new images or videos:# Load your custom trained model weights
model = YOLO('/content/runs/detect/train3/weights/best.pt')

# Run inference on a test source (image, folder, or video file)
results = model.predict(
    source='path/to/test/image_or_folder',
    conf=0.25  # Minimum confidence threshold for bounding boxes
)
Output: Inference results (images/videos with detection bounding boxes) are saved in the /runs/detect/predict/ directory.LicenseThis project is licensed under the MIT License.
