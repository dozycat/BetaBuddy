"""
Video annotation module for drawing keypoints, skeleton, and metrics on video frames.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from app.core.pose_estimator import KeypointData, LANDMARK_NAMES


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
    keypoint_radius: int = 5
    skeleton_thickness: int = 2
    com_radius: int = 8
    trajectory_thickness: int = 2
    visibility_threshold: float = 0.5
    font_scale: float = 0.6
    font_thickness: int = 1


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

        # Stability
        if "stability_score" in metrics:
            stability = metrics["stability_score"] * 100
            lines.append(f"Stability: {stability:.1f}%")

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

        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))

        # Create lookup for frame data
        frame_lookup = {fd["frame_number"]: fd for fd in frame_data}

        # Process frames with pose estimation
        frame_num = 0
        trajectory_so_far = []

        with PoseEstimator() as pose_estimator:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to RGB for pose estimation
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                keypoints = pose_estimator.process_frame(frame_rgb)

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

        return self.output_path.exists()

    def generate_from_existing_keypoints(
        self,
        frame_data: list[dict],
    ) -> bool:
        """
        Generate annotated video using stored keypoint data (faster, no re-detection).

        Args:
            frame_data: List of frame data including keypoints

        Returns:
            True if successful, False otherwise
        """
        import logging
        logger = logging.getLogger(__name__)

        cap = cv2.VideoCapture(str(self.input_path))
        if not cap.isOpened():
            logger.error(f"Failed to open input video: {self.input_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Input video: {width}x{height} @ {fps}fps")

        # Try different codecs for compatibility
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use mp4v codec for .mp4 files
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))

        if not out.isOpened():
            # Fallback to XVID
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_avi = str(self.output_path).replace('.mp4', '.avi')
            out = cv2.VideoWriter(output_avi, fourcc, fps, (width, height))
            if not out.isOpened():
                logger.error("Failed to create video writer")
                cap.release()
                return False

        # Create lookup for frame data
        frame_lookup = {fd["frame_number"]: fd for fd in frame_data}
        logger.info(f"Frame data contains {len(frame_data)} frames with keypoints")

        frame_num = 0
        trajectory_so_far = []
        annotated_count = 0

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

            out.write(frame)
            frame_num += 1

        cap.release()
        out.release()

        logger.info(f"Annotated {annotated_count} frames out of {frame_num} total")
        return self.output_path.exists()
