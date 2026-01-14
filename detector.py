import torch
from ultralytics import YOLO


class ObjectDetector:
    """
    YOLOv8 object detector with built-in multi-object tracking (ByteTrack).
    """

    def __init__(
        self,
        model_name="yolov8x.pt",
        conf_threshold=0.35,
        img_size=1280
    ):
        """
        Parameters
        ----------
        model_name : str
            YOLOv8 model file
        conf_threshold : float
            Minimum confidence score
        img_size : int
            Inference image size
        """
        # Use CUDA if available, otherwise use CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = YOLO(model_name).to(self.device)
        self.conf_threshold = conf_threshold
        self.img_size = img_size

    def track(self, frame):
        """
        Run object detection + tracking on a frame.

        Each object receives a persistent ID across frames.

        Parameters
        ----------
        frame : numpy.ndarray
            BGR image from OpenCV

        Returns
        -------
        ultralytics.yolo.engine.results.Results
        """
        results = self.model.track(
            frame,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            device=self.device,
            persist=True,
            tracker="bytetrack.yaml"
        )
        return results