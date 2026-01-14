# Drone Object Detection & Tracking

Real-time object detection and multi-object tracking based on YOLOv8 + ByteTrack

---

## Installation

### 1. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux
venv\Scripts\activate      # Windows

### 2. Install PyTorch with CUDA
Follow the official instructions:  
https://pytorch.org/get-started/locally/

### 3. Install dependencies
pip install -r requirements.txt

---

## Usage

### Webcam
python main.py --source 0

### Video file
python main.py --source path/to/video.mp4

### Drone RTSP stream
python main.py --source rtsp://DRONE_IP:PORT/stream

### Advanced options
```
--model WEIGHTS
    Any YOLO weight name or path (default: yolov8s.pt)
    Examples: yolov8s.pt, yolo26s.pt, yolo11n.pt, custom.pt

--img-size SIZE
    Inference image size (default: 640)
    Higher = more accurate but slower

--conf THRESHOLD
    Confidence threshold (default: 0.35)
    Range: 0.0 to 1.0
```

### Examples
```
# CPU-friendly (fast)
python main.py --source 0 --model yolov8n.pt --img-size 480

# Balanced (default, recommended)
python main.py --source 0 --model yolov8s.pt --img-size 640

# High accuracy (requires GPU)
python main.py --source 0 --model yolov8x.pt --img-size 1280

# Test the new YOLO26 (edge-optimized)
python main.py --source 0 --model yolo26s.pt --img-size 640

# Custom local weight
python main.py --source 0 --model /path/to/your/custom.pt --img-size 640

# Custom settings
python main.py --source video.mp4 --model yolov8m.pt --img-size 800 --conf 0.5
```

---

## Tracking
Each detected object is assigned a persistent ID across frames.
This enables:
- object following
- trajectory analysis
- counting
- behavior analysis

Tracking is handled by ByteTrack, integrated directly into YOLOv8.

---

## Detected objects
The model is pre-trained on the COCO dataset (80 classes), including:
- people
- vehicles
- animals
- common objects

---

## Exit
Press Q to quit.

---

## Notes
- Default model: yolov8s.pt (balanced accuracy/speed, works on CPU)
- You can pass any Ultralytics-compatible .pt weight (yolov8/11/26 or custom trained)
- For higher FPS, reduce img_size or use n/s variants; x variants are heaviest and prefer GPU.