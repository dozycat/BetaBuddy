"""
Video annotation module for drawing keypoints, skeleton, and metrics on video frames.
"""
import cv2
import numpy as np
import subprocess
import shutil
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import logging
from PIL import Image, ImageDraw, ImageFont

from app.core.pose_estimator import KeypointData, LANDMARK_NAMES

logger = logging.getLogger(__name__)

# Path to Chinese font
CHINESE_FONT_PATH = Path(__file__).parent.parent / "fonts" / "NotoSansCJKsc-Regular.otf"

# Cache for loaded fonts
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


def get_chinese_font(size: int = 20) -> ImageFont.FreeTypeFont:
    """Get Chinese font with caching."""
    if size not in _font_cache:
        if CHINESE_FONT_PATH.exists():
            _font_cache[size] = ImageFont.truetype(str(CHINESE_FONT_PATH), size)
        else:
            logger.warning(f"Chinese font not found at {CHINESE_FONT_PATH}, using default")
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def put_chinese_text(
    img: np.ndarray,
    text: str,
    position: tuple[int, int],
    font_size: int = 20,
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Draw Chinese text on OpenCV image using PIL.

    Args:
        img: OpenCV image (BGR format)
        text: Text to draw (supports Chinese)
        position: (x, y) position for text
        font_size: Font size in pixels
        color: Text color in BGR format

    Returns:
        Image with text drawn
    """
    # Convert BGR to RGB for PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # Get font
    font = get_chinese_font(font_size)

    # Convert BGR color to RGB for PIL
    rgb_color = (color[2], color[1], color[0])

    # Draw text
    draw.text(position, text, font=font, fill=rgb_color)

    # Convert back to BGR for OpenCV
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def get_ffmpeg_path() -> Optional[str]:
    """Get ffmpeg executable path, checking system and imageio-ffmpeg."""
    # Check system ffmpeg first
    if shutil.which("ffmpeg"):
        return "ffmpeg"

    # Try imageio-ffmpeg bundled version
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    return None


def compress_video_h264(input_path: Path, output_path: Path, crf: int = 28, preset: str = "slow") -> bool:
    """
    Re-encode video with H.264 for better compression.

    Args:
        input_path: Path to input video (uncompressed/poorly compressed)
        output_path: Path to output video (H.264 compressed)
        crf: Constant Rate Factor (0-51, lower = better quality).
             18 = visually lossless, 23 = default, 28 = good for web, 32 = smaller file
        preset: Encoding speed/quality tradeoff.
                ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
                Slower presets = better compression efficiency at same quality

    Returns:
        True if successful, False otherwise
    """
    # Check if ffmpeg is available (system or bundled)
    ffmpeg_path = get_ffmpeg_path()
    if not ffmpeg_path:
        logger.warning("ffmpeg not found (install ffmpeg or imageio-ffmpeg), skipping compression")
        return False

    try:
        # Use ffmpeg to re-encode with H.264
        cmd = [
            ffmpeg_path,
            "-y",  # Overwrite output
            "-i", str(input_path),
            "-c:v", "libx264",  # H.264 codec
            "-preset", preset,  # Slower = better compression
            "-crf", str(crf),  # Quality (28 = good balance for web)
            "-pix_fmt", "yuv420p",  # Ensure compatibility with all players
            "-profile:v", "high",  # H.264 High profile for better compression
            "-level", "4.1",  # Compatibility level
            "-tune", "film",  # Optimize for real-world video content
            "-c:a", "aac",  # Audio codec (if any)
            "-b:a", "128k",  # Audio bitrate
            "-movflags", "+faststart",  # Web optimization (metadata at start)
            "-loglevel", "warning",
            str(output_path)
        ]

        logger.info(f"Compressing video with CRF={crf}, preset={preset}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"ffmpeg error: {result.stderr}")
            return False

        # Log compression ratio
        if output_path.exists():
            input_size = input_path.stat().st_size
            output_size = output_path.stat().st_size
            ratio = input_size / output_size if output_size > 0 else 0
            logger.info(f"Compression: {input_size/1024/1024:.1f}MB -> {output_size/1024/1024:.1f}MB (ratio: {ratio:.1f}x)")

        return output_path.exists()
    except Exception as e:
        logger.error(f"Compression failed: {e}")
        return False


# MediaPipe Pose skeleton connections
SKELETON_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),  # Left eye
    (0, 4), (4, 5), (5, 6), (6, 8),  # Right eye
    (9, 10),  # Mouth
    # Torso
    (11, 12),  # Shoulders
    (11, 23), (12, 24),  # Shoulder to hip
    (23, 24),  # Hips
    # Left arm
    (11, 13), (13, 15),  # Shoulder -> elbow -> wrist
    (15, 17), (15, 19), (15, 21),  # Wrist to fingers
    (17, 19),  # Pinky to index
    # Right arm
    (12, 14), (14, 16),  # Shoulder -> elbow -> wrist
    (16, 18), (16, 20), (16, 22),  # Wrist to fingers
    (18, 20),  # Pinky to index
    # Left leg
    (23, 25), (25, 27),  # Hip -> knee -> ankle
    (27, 29), (27, 31), (29, 31),  # Ankle to foot
    # Right leg
    (24, 26), (26, 28),  # Hip -> knee -> ankle
    (28, 30), (28, 32), (30, 32),  # Ankle to foot
]

# Color scheme (BGR format for OpenCV)
COLORS = {
    "high_confidence": (0, 255, 0),      # Green
    "medium_confidence": (0, 255, 255),  # Yellow
    "low_confidence": (0, 0, 255),       # Red
    "skeleton": (255, 200, 100),         # Light blue
    "com": (255, 0, 255),                # Magenta
    "com_trajectory": (200, 100, 255),   # Pink
    "text_bg": (0, 0, 0),                # Black
    "text": (255, 255, 255),             # White
}


@dataclass
class AnnotationConfig:
    """Configuration for video annotation."""
    draw_keypoints: bool = True
    draw_skeleton: bool = True
    draw_com: bool = True
    draw_com_trajectory: bool = True
    draw_metrics_overlay: bool = True
    draw_movements: bool = True  # Draw movement technique labels
    keypoint_radius: int = 5
    skeleton_thickness: int = 2
    com_radius: int = 8
    trajectory_thickness: int = 2
    visibility_threshold: float = 0.5
    font_scale: float = 0.6
    font_thickness: int = 1
    movement_font_scale: float = 0.7
    movement_display_frames: int = 45  # Frames to show label after movement starts


class VideoAnnotator:
    """Annotates video frames with pose keypoints, skeleton, and metrics."""

    def __init__(self, config: Optional[AnnotationConfig] = None):
        self.config = config or AnnotationConfig()

    def get_keypoint_color(self, visibility: float) -> tuple:
        """Get color based on keypoint visibility/confidence."""
        if visibility >= 0.8:
            return COLORS["high_confidence"]
        elif visibility >= self.config.visibility_threshold:
            return COLORS["medium_confidence"]
        else:
            return COLORS["low_confidence"]

    def draw_keypoints(
        self,
        frame: np.ndarray,
        keypoints: list[KeypointData],
        frame_height: int,
        frame_width: int,
    ) -> np.ndarray:
        """Draw colored circles at keypoint positions."""
        for kp in keypoints:
            if kp.visibility < self.config.visibility_threshold:
                continue

            x = int(kp.x * frame_width)
            y = int(kp.y * frame_height)
            color = self.get_keypoint_color(kp.visibility)

            cv2.circle(frame, (x, y), self.config.keypoint_radius, color, -1)
            cv2.circle(frame, (x, y), self.config.keypoint_radius, (0, 0, 0), 1)

        return frame

    def draw_skeleton(
        self,
        frame: np.ndarray,
        keypoints: list[KeypointData],
        frame_height: int,
        frame_width: int,
    ) -> np.ndarray:
        """Draw bone connections between keypoints."""
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            if start_idx >= len(keypoints) or end_idx >= len(keypoints):
                continue

            kp1 = keypoints[start_idx]
            kp2 = keypoints[end_idx]

            # Skip if either keypoint has low visibility
            if (kp1.visibility < self.config.visibility_threshold or
                kp2.visibility < self.config.visibility_threshold):
                continue

            x1 = int(kp1.x * frame_width)
            y1 = int(kp1.y * frame_height)
            x2 = int(kp2.x * frame_width)
            y2 = int(kp2.y * frame_height)

            cv2.line(frame, (x1, y1), (x2, y2), COLORS["skeleton"],
                     self.config.skeleton_thickness)

        return frame

    def draw_center_of_mass(
        self,
        frame: np.ndarray,
        com: tuple[float, float],
        frame_height: int,
        frame_width: int,
        trajectory: Optional[list[tuple[float, float]]] = None,
    ) -> np.ndarray:
        """Draw center of mass point and optional trajectory line."""
        # Draw trajectory if provided
        if trajectory and len(trajectory) > 1 and self.config.draw_com_trajectory:
            points = []
            for point in trajectory[-100:]:  # Last 100 points
                x = int(point[0] * frame_width)
                y = int(point[1] * frame_height)
                points.append((x, y))

            for i in range(1, len(points)):
                alpha = i / len(points)  # Fade effect
                color = tuple(int(c * alpha) for c in COLORS["com_trajectory"])
                cv2.line(frame, points[i-1], points[i], color,
                         self.config.trajectory_thickness)

        # Draw current COM
        x = int(com[0] * frame_width)
        y = int(com[1] * frame_height)

        cv2.circle(frame, (x, y), self.config.com_radius, COLORS["com"], -1)
        cv2.circle(frame, (x, y), self.config.com_radius, (255, 255, 255), 2)

        return frame

    def draw_metrics_overlay(
        self,
        frame: np.ndarray,
        metrics: dict,
        frame_height: int,
        frame_width: int,
    ) -> np.ndarray:
        """Draw text overlay with stability, angles, velocity."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        x_offset = 10
        line_height = 25

        # Background rectangle for text
        bg_height = line_height * 6
        cv2.rectangle(frame, (5, 5), (280, bg_height + 10),
                      COLORS["text_bg"], -1)
        cv2.rectangle(frame, (5, 5), (280, bg_height + 10),
                      COLORS["text"], 1)

        lines = []

        # Frame info
        if "frame_number" in metrics:
            lines.append(f"Frame: {metrics['frame_number']}")

        # Center of mass
        if "center_of_mass" in metrics:
            com = metrics["center_of_mass"]
            lines.append(f"CoM: ({com[0]:.3f}, {com[1]:.3f})")

        # Velocity
        if "velocity" in metrics and metrics["velocity"]:
            vel = metrics["velocity"]
            vel_mag = np.sqrt(vel[0]**2 + vel[1]**2)
            lines.append(f"Velocity: {vel_mag:.2f}")

        # Acceleration
        if "acceleration" in metrics and metrics["acceleration"]:
            acc = metrics["acceleration"]
            acc_mag = np.sqrt(acc[0]**2 + acc[1]**2)
            lines.append(f"Acceleration: {acc_mag:.2f}")

        for i, line in enumerate(lines):
            cv2.putText(frame, line, (x_offset, y_offset + i * line_height),
                        font, self.config.font_scale, COLORS["text"],
                        self.config.font_thickness)

        return frame

    def draw_joint_angles_overlay(
        self,
        frame: np.ndarray,
        joint_angles: dict[str, float],
        frame_height: int,
        frame_width: int,
    ) -> np.ndarray:
        """Draw joint angles panel on the right side of the frame."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        x_offset = frame_width - 200
        y_offset = 30
        line_height = 20

        # Background rectangle
        num_lines = len(joint_angles) + 1
        bg_height = line_height * num_lines
        cv2.rectangle(frame, (x_offset - 5, 5), (frame_width - 5, bg_height + 10),
                      COLORS["text_bg"], -1)
        cv2.rectangle(frame, (x_offset - 5, 5), (frame_width - 5, bg_height + 10),
                      COLORS["text"], 1)

        cv2.putText(frame, "Joint Angles", (x_offset, y_offset),
                    font, 0.5, COLORS["text"], 1)

        name_map = {
            "left_elbow": "L.Elbow",
            "right_elbow": "R.Elbow",
            "left_shoulder": "L.Shoulder",
            "right_shoulder": "R.Shoulder",
            "left_hip": "L.Hip",
            "right_hip": "R.Hip",
            "left_knee": "L.Knee",
            "right_knee": "R.Knee",
        }

        for i, (joint, angle) in enumerate(joint_angles.items()):
            short_name = name_map.get(joint, joint)
            text = f"{short_name}: {angle:.0f}"
            cv2.putText(frame, text, (x_offset, y_offset + (i + 1) * line_height),
                        font, 0.4, COLORS["text"], 1)

        return frame

    def draw_movement_labels(
        self,
        frame: np.ndarray,
        active_movements: list[dict],
        frame_height: int,
        frame_width: int,
        y_start_offset: int = 200,
    ) -> np.ndarray:
        """
        Draw movement technique labels on the RIGHT side of the frame.

        Args:
            frame: Video frame to annotate
            active_movements: List of movements active at this frame
            frame_height: Frame height in pixels
            frame_width: Frame width in pixels
            y_start_offset: Y offset from top to start drawing (below joint angles)

        Returns:
            Annotated frame
        """
        if not active_movements:
            return frame

        line_height = 140  # 28 * 5
        x_offset = 20  # Move to left side for larger text
        y_offset = y_start_offset

        # Draw header using Chinese text
        y_offset += 110  # 22 * 5

        # Show at most 3 movements to avoid clutter with larger font
        for i, movement in enumerate(active_movements[:3]):
            name = movement.get("movement_name_cn", "")
            side_cn = movement.get("side_cn", "")
            is_challenging = movement.get("is_challenging", False)

            if not name:
                continue

            # Build display text
            if side_cn and side_cn != "双侧":
                display_text = f"{name}({side_cn})"
            else:
                display_text = name

            # Add star for challenging moves
            if is_challenging:
                display_text = "★ " + display_text

            # Color based on difficulty
            if is_challenging:
                text_color = (0, 215, 255)  # Gold for challenging (BGR)
            else:
                text_color = (0, 255, 200)  # Cyan for normal (BGR)

            frame = put_chinese_text(
                frame,
                display_text,
                (x_offset, y_offset + i * line_height),
                font_size=80,  # 16 * 5
                color=text_color,
            )

        return frame

    def draw_movement_subtitle(
        self,
        frame: np.ndarray,
        active_movements: list[dict],
        frame_height: int,
        frame_width: int,
    ) -> np.ndarray:
        """
        Draw movement technical details as centered subtitle at bottom.

        Args:
            frame: Video frame to annotate
            active_movements: List of movements active at this frame
            frame_height: Frame height in pixels
            frame_width: Frame width in pixels

        Returns:
            Annotated frame
        """
        if not active_movements:
            return frame

        font_size = 40  # Reduced for bottom subtitle

        # Build subtitle text from first active movement
        movement = active_movements[0]
        name = movement.get("movement_name_cn", "")
        side_cn = movement.get("side_cn", "")
        is_challenging = movement.get("is_challenging", False)
        key_angles = movement.get("key_angles", {})

        if not name:
            return frame

        # Build detailed subtitle
        parts = [name]
        if side_cn and side_cn != "双侧":
            parts.append(f"({side_cn})")

        # Add key angle details
        angle_parts = []
        if "elbow" in key_angles:
            angle_parts.append(f"肘{key_angles['elbow']:.0f}°")
        if "knee" in key_angles:
            angle_parts.append(f"膝{key_angles['knee']:.0f}°")
        if "hip" in key_angles:
            angle_parts.append(f"髋{key_angles['hip']:.0f}°")
        if "shoulder" in key_angles:
            angle_parts.append(f"肩{key_angles['shoulder']:.0f}°")

        if angle_parts:
            parts.append(" | " + " ".join(angle_parts))

        if is_challenging:
            parts.append(" [高难度]")

        subtitle_text = "".join(parts)

        # Calculate text size using PIL font
        pil_font = get_chinese_font(font_size)
        # Use getbbox for text size (returns (left, top, right, bottom))
        bbox = pil_font.getbbox(subtitle_text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Position: centered at bottom
        x = (frame_width - text_width) // 2
        y = frame_height - 60  # Adjusted for 40px font

        # Draw semi-transparent background
        padding = 12
        bg_x1 = x - padding
        bg_y1 = y - padding
        bg_x2 = x + text_width + padding
        bg_y2 = y + text_height + padding

        # Create overlay for semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw border
        border_color = (0, 215, 255) if is_challenging else (200, 200, 200)
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), border_color, 2)  # Thicker border

        # Draw text using PIL for Chinese support
        frame = put_chinese_text(
            frame,
            subtitle_text,
            (x, y),
            font_size=font_size,
            color=(255, 255, 255),
        )

        return frame

    def annotate_frame(
        self,
        frame: np.ndarray,
        keypoints: list[KeypointData],
        metrics: Optional[dict] = None,
        trajectory: Optional[list[tuple[float, float]]] = None,
    ) -> np.ndarray:
        """Full annotation: keypoints + skeleton + CoM + metrics."""
        frame_height, frame_width = frame.shape[:2]
        annotated = frame.copy()

        # Draw skeleton first (below keypoints)
        if self.config.draw_skeleton:
            annotated = self.draw_skeleton(annotated, keypoints, frame_height, frame_width)

        # Draw keypoints
        if self.config.draw_keypoints:
            annotated = self.draw_keypoints(annotated, keypoints, frame_height, frame_width)

        # Draw center of mass
        if self.config.draw_com and metrics and "center_of_mass" in metrics:
            annotated = self.draw_center_of_mass(
                annotated,
                metrics["center_of_mass"],
                frame_height,
                frame_width,
                trajectory,
            )

        # Draw metrics overlay
        if self.config.draw_metrics_overlay and metrics:
            annotated = self.draw_metrics_overlay(annotated, metrics, frame_height, frame_width)

            # Draw joint angles if available
            if "joint_angles" in metrics and metrics["joint_angles"]:
                annotated = self.draw_joint_angles_overlay(
                    annotated, metrics["joint_angles"], frame_height, frame_width
                )

        return annotated


class AnnotatedVideoGenerator:
    """Generates annotated video from analysis results."""

    def __init__(
        self,
        input_path: str,
        output_path: str,
        config: Optional[AnnotationConfig] = None,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.config = config or AnnotationConfig()
        self.annotator = VideoAnnotator(self.config)

    def generate(
        self,
        frame_data: list[dict],
        com_trajectory: Optional[list[tuple[float, float]]] = None,
    ) -> bool:
        """
        Generate annotated video from analysis results.

        Args:
            frame_data: List of frame metrics from analysis
            com_trajectory: Optional center of mass trajectory

        Returns:
            True if successful, False otherwise
        """
        from app.core.pose_estimator import PoseEstimator

        cap = cv2.VideoCapture(str(self.input_path))
        if not cap.isOpened():
            return False

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temporary file first, then compress with ffmpeg
        # Try multiple codecs in order of preference
        codecs_to_try = [
            ('XVID', '.avi'),   # Widely available
            ('MJPG', '.avi'),   # Always available, larger files
            ('mp4v', '.mp4'),   # MPEG-4
        ]

        out = None
        temp_path = self.output_path.with_suffix('.temp.avi')
        for codec, ext in codecs_to_try:
            temp_path = self.output_path.with_suffix(f'.temp{ext}')
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))
            if out.isOpened():
                logger.info(f"Using codec: {codec}")
                break
            else:
                logger.info(f"Codec {codec} not available, trying next...")
                out = None

        if out is None or not out.isOpened():
            logger.error("Failed to create video writer - no compatible codec found")
            cap.release()
            return False

        # Create lookup for frame data
        frame_lookup = {fd["frame_number"]: fd for fd in frame_data}

        # Process frames with pose estimation (VIDEO mode + GPU for speed)
        frame_num = 0
        trajectory_so_far = []

        with PoseEstimator(video_mode=True, use_gpu=True) as pose_estimator:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to RGB for pose estimation
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Calculate timestamp in milliseconds for VIDEO mode
                timestamp_ms = int((frame_num / fps) * 1000)
                keypoints = pose_estimator.process_frame(frame_rgb, timestamp_ms=timestamp_ms)

                if keypoints and frame_num in frame_lookup:
                    metrics = frame_lookup[frame_num]

                    # Build trajectory up to this point
                    if "center_of_mass" in metrics:
                        trajectory_so_far.append(metrics["center_of_mass"])

                    # Annotate frame
                    frame = self.annotator.annotate_frame(
                        frame,
                        keypoints,
                        metrics,
                        trajectory_so_far if self.config.draw_com_trajectory else None,
                    )

                out.write(frame)
                frame_num += 1

        cap.release()
        out.release()

        # Compress with H.264 for much smaller file size
        if temp_path.exists():
            if compress_video_h264(temp_path, self.output_path):
                temp_path.unlink()  # Delete temp file
                logger.info(f"Compressed video saved to {self.output_path}")
            else:
                # Fallback: rename temp to output if compression fails
                temp_path.rename(self.output_path)
                logger.warning("ffmpeg compression failed, using uncompressed video")

        return self.output_path.exists()

    def generate_from_existing_keypoints(
        self,
        frame_data: list[dict],
        movement_data: Optional[list[dict]] = None,
    ) -> bool:
        """
        Generate annotated video using stored keypoint data (faster, no re-detection).

        Args:
            frame_data: List of frame data including keypoints
            movement_data: Optional list of detected movements for overlay

        Returns:
            True if successful, False otherwise
        """
        cap = cv2.VideoCapture(str(self.input_path))
        if not cap.isOpened():
            logger.error(f"Failed to open input video: {self.input_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Input video: {width}x{height} @ {fps}fps")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temporary file first, then compress with ffmpeg
        temp_path = self.output_path.with_suffix('.temp.avi')

        # Try multiple codecs in order of preference
        codecs_to_try = [
            ('XVID', '.avi'),   # Widely available
            ('MJPG', '.avi'),   # Always available, larger files
            ('mp4v', '.mp4'),   # MPEG-4
        ]

        out = None
        for codec, ext in codecs_to_try:
            temp_path = self.output_path.with_suffix(f'.temp{ext}')
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))
            if out.isOpened():
                logger.info(f"Using codec: {codec}")
                break
            else:
                logger.info(f"Codec {codec} not available, trying next...")
                out = None

        if out is None or not out.isOpened():
            logger.error("Failed to create video writer - no compatible codec found")
            cap.release()
            return False

        # Create lookup for frame data
        frame_lookup = {fd["frame_number"]: fd for fd in frame_data}
        logger.info(f"Frame data contains {len(frame_data)} frames with keypoints")

        # Build movement frame lookup if movement data is provided
        movement_by_frame = {}
        if movement_data and self.config.draw_movements:
            for m in movement_data:
                # Show movement label from start to end frame, plus some extra frames
                start = m.get("start_frame", 0)
                end = m.get("end_frame", start) + self.config.movement_display_frames
                for f in range(start, end + 1):
                    if f not in movement_by_frame:
                        movement_by_frame[f] = []
                    # Avoid duplicates
                    if not any(existing.get("movement_type") == m.get("movement_type") and
                              existing.get("side") == m.get("side")
                              for existing in movement_by_frame[f]):
                        movement_by_frame[f].append(m)
            logger.info(f"Movement data contains {len(movement_data)} movements, frames with overlays: {len(movement_by_frame)}")

        frame_num = 0
        trajectory_so_far = []
        annotated_count = 0
        movement_frames_drawn = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num in frame_lookup:
                data = frame_lookup[frame_num]

                # Build trajectory
                if "center_of_mass" in data:
                    trajectory_so_far.append(data["center_of_mass"])

                # Convert keypoints dict to KeypointData objects if needed
                if "keypoints" in data and data["keypoints"]:
                    keypoints = []
                    for kp_dict in data["keypoints"]:
                        keypoints.append(KeypointData(
                            x=kp_dict["x"],
                            y=kp_dict["y"],
                            z=kp_dict.get("z", 0.0),
                            visibility=kp_dict.get("visibility", 1.0),
                            name=kp_dict.get("name", ""),
                        ))

                    frame = self.annotator.annotate_frame(
                        frame,
                        keypoints,
                        data,
                        trajectory_so_far if self.config.draw_com_trajectory else None,
                    )
                    annotated_count += 1

            # Draw movement overlays if enabled and present for this frame
            if self.config.draw_movements and frame_num in movement_by_frame:
                active_movements = movement_by_frame[frame_num]
                movement_frames_drawn += 1

                # Draw movement labels on right side (below joint angles)
                frame = self.annotator.draw_movement_labels(
                    frame,
                    active_movements,
                    height,
                    width,
                    y_start_offset=200,  # Below joint angles panel
                )

                # Draw technical details as subtitle at bottom
                frame = self.annotator.draw_movement_subtitle(
                    frame,
                    active_movements,
                    height,
                    width,
                )

            out.write(frame)
            frame_num += 1

        cap.release()
        out.release()

        logger.info(f"Annotated {annotated_count} frames, movements on {movement_frames_drawn} frames, total {frame_num}")

        # Compress with H.264 for much smaller file size
        if temp_path.exists():
            if compress_video_h264(temp_path, self.output_path):
                temp_path.unlink()  # Delete temp file
                logger.info(f"Compressed video saved to {self.output_path}")
            else:
                # Fallback: move temp to output if compression fails (handles overwrite)
                if self.output_path.exists():
                    self.output_path.unlink()  # Remove existing file first
                shutil.move(str(temp_path), str(self.output_path))
                logger.warning("ffmpeg compression failed, using uncompressed video")

        return self.output_path.exists()
