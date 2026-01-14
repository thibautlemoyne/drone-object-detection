import argparse
import cv2

from video_source import open_video_source
from detector import ObjectDetector


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time Drone Object Detection and Tracking (YOLOv8 + ByteTrack)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: webcam index (0), video file path, or RTSP URL"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="Path or name of a YOLO weight file (.pt), e.g., yolov8s.pt, yolo26s.pt, or a custom path"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Inference image size (default: 640). Larger = more accurate but slower"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold for detections (default: 0.35)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Convert webcam index if needed
    source = int(args.source) if args.source.isdigit() else args.source

    cap = open_video_source(source)

    detector = ObjectDetector(
        model_name=args.model,
        conf_threshold=args.conf,
        img_size=args.img_size
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.track(frame)

        # Draw bounding boxes, labels, confidence scores and tracking IDs
        annotated_frame = results[0].plot(
            line_width=2,
            font_size=0.8
        )

        cv2.imshow("Drone Object Detection + Tracking", annotated_frame)

        # Press 'Q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()