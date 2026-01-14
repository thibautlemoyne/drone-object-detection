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
    return parser.parse_args()


def main():
    args = parse_args()

    # Convert webcam index if needed
    source = int(args.source) if args.source.isdigit() else args.source

    cap = open_video_source(source)

    detector = ObjectDetector(
        model_name="yolov8x.pt",
        conf_threshold=0.35,
        img_size=1280
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