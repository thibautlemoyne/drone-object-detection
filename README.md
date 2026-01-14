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