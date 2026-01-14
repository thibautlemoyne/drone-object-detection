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
--model {yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt}
    Model size (default: yolov8n.pt)
    - yolov8n: Nano (fastest, least accurate)
    - yolov8s: Small (recommended for CPU)
    - yolov8m: Medium
    - yolov8l: Large
    - yolov8x: XLarge (most accurate, requires GPU)

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
- Default model: YOLOv8x (maximum accuracy)
- For higher FPS, reduce img_size or use YOLOv8l / YOLOv8n