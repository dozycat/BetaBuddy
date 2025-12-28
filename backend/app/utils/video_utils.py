import cv2
from pathlib import Path
from typing import Optional
import numpy as np

from app.config import settings


class VideoProcessor:
    def __init__(self, video_path: str | Path):
        self.video_path = Path(video_path)
        self.cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[dict] = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {self.video_path}")

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def metadata(self) -> dict:
        if self._metadata is None:
            self._metadata = self.extract_metadata()
        return self._metadata

    def extract_metadata(self) -> dict:
        if self.cap is None:
            self.open()

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        return {
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "duration": duration,
        }

    def read_frame(self, frame_number: Optional[int] = None) -> Optional[np.ndarray]:
        if self.cap is None:
            self.open()

        if frame_number is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = self.cap.read()
        if not ret:
            return None

        return frame

    def read_frames(self, start: int = 0, end: Optional[int] = None, step: int = 1):
        if self.cap is None:
            self.open()

        total_frames = self.metadata["total_frames"]
        if end is None:
            end = total_frames

        for frame_num in range(start, min(end, total_frames), step):
            frame = self.read_frame(frame_num)
            if frame is not None:
                yield frame_num, frame

    def get_frame_rgb(self, frame_number: int) -> Optional[np.ndarray]:
        frame = self.read_frame(frame_number)
        if frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None


def get_video_metadata(video_path: str | Path) -> dict:
    with VideoProcessor(video_path) as processor:
        return processor.metadata


def is_valid_video_format(filename: str) -> bool:
    suffix = Path(filename).suffix.lower()
    return suffix in settings.supported_video_formats


def generate_thumbnail(video_path: str | Path, output_path: str | Path, frame_number: int = 0) -> bool:
    try:
        with VideoProcessor(video_path) as processor:
            frame = processor.read_frame(frame_number)
            if frame is not None:
                cv2.imwrite(str(output_path), frame)
                return True
    except Exception:
        pass
    return False
