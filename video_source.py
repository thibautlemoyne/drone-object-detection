import cv2


def open_video_source(source):
    """
    Open a video source.

    Parameters
    ----------
    source : int or str
        - int (0, 1, ...)           -> webcam
        - str (.mp4, .avi)         -> video file
        - str (rtsp://...)         -> drone RTSP stream

    Returns
    -------
    cv2.VideoCapture
    """
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")

    return cap