import cv2
import logging
import random
import shutil
import subprocess
from pathlib import Path
from typing import Optional
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


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


def get_ffmpeg_path() -> Optional[str]:
    """Get ffmpeg executable path, checking system and imageio-ffmpeg."""
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def generate_gif_thumbnail(
    video_path: str | Path,
    output_path: str | Path,
    num_frames: int = 10,
    width: int = 320,
    fps: int = 5,
) -> bool:
    """
    Generate an animated GIF thumbnail from randomly sampled video frames.

    Args:
        video_path: Path to source video
        output_path: Path for output GIF
        num_frames: Number of frames to sample (default 10)
        width: Output GIF width in pixels (height auto-calculated)
        fps: Output GIF frame rate

    Returns:
        True if successful, False otherwise
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    ffmpeg_path = get_ffmpeg_path()
    if not ffmpeg_path:
        logger.warning("ffmpeg not found, cannot generate GIF thumbnail")
        return False

    # Get total frame count
    try:
        with VideoProcessor(video_path) as processor:
            total_frames = processor.metadata.get("total_frames", 0)
            if total_frames < 1:
                logger.error("Video has no frames")
                return False

            if total_frames <= num_frames:
                # If video has fewer frames than requested, use evenly spaced frames
                frame_numbers = list(range(0, total_frames, max(1, total_frames // num_frames)))[:num_frames]
            else:
                # Randomly sample frames spread across the video
                # Divide video into segments and pick one random frame from each
                segment_size = total_frames // num_frames
                frame_numbers = []
                for i in range(num_frames):
                    start = i * segment_size
                    end = start + segment_size
                    frame_numbers.append(random.randint(start, min(end - 1, total_frames - 1)))
                frame_numbers.sort()
    except Exception as e:
        logger.error(f"Failed to get video metadata: {e}")
        return False

    if not frame_numbers:
        logger.error("No frames to sample")
        return False

    # Build ffmpeg select filter
    select_expr = "+".join([f"eq(n\\,{f})" for f in frame_numbers])

    # Build ffmpeg command
    cmd = [
        ffmpeg_path,
        "-y",  # Overwrite output
        "-i", str(video_path),
        "-vf", f"select='{select_expr}',setpts=N/{fps}/TB,scale={width}:-1:flags=lanczos",
        "-r", str(fps),
        "-loop", "0",  # Infinite loop
        "-loglevel", "warning",
        str(output_path)
    ]

    try:
        logger.info(f"Generating GIF thumbnail with {len(frame_numbers)} frames from {video_path.name}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            logger.error(f"ffmpeg error: {result.stderr}")
            return False

        if output_path.exists():
            size_kb = output_path.stat().st_size / 1024
            logger.info(f"GIF thumbnail generated: {size_kb:.1f}KB")
            return True

        return False

    except subprocess.TimeoutExpired:
        logger.error("GIF generation timed out")
        return False
    except Exception as e:
        logger.error(f"GIF generation failed: {e}")
        return False
