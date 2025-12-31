import numpy as np
from typing import Optional
from dataclasses import dataclass, field

from app.core.pose_estimator import KeypointData
from app.core.physics_engine import (
    calculate_center_of_mass,
    calculate_all_joint_angles,
    calculate_trajectory_length,
    calculate_distance,
)


@dataclass
class FrameMetrics:
    frame_number: int
    timestamp: float
    center_of_mass: tuple[float, float]
    joint_angles: dict[str, float]
    velocity: Optional[tuple[float, float]] = None
    acceleration: Optional[tuple[float, float]] = None


@dataclass
class AnalysisSummary:
    total_frames: int
    duration: float
    avg_efficiency: float
    max_velocity: float
    max_acceleration: float
    dyno_count: int
    total_distance: float
    joint_angle_stats: dict[str, dict[str, float]] = field(default_factory=dict)


class ClimbingMetrics:
    def __init__(self):
        self.com_history: list[tuple[float, float]] = []
        self.frame_metrics: list[FrameMetrics] = []
        self.velocity_history: list[tuple[float, float]] = []
        self.acceleration_history: list[tuple[float, float]] = []

    def reset(self):
        """Reset all accumulated data."""
        self.com_history = []
        self.frame_metrics = []
        self.velocity_history = []
        self.acceleration_history = []

    def process_frame(
        self,
        frame_number: int,
        timestamp: float,
        keypoints: list[KeypointData],
    ) -> FrameMetrics:
        """
        Process a single frame and compute metrics.

        Args:
            frame_number: Current frame index
            timestamp: Time in seconds
            keypoints: List of detected keypoints

        Returns:
            FrameMetrics for this frame
        """
        # Calculate center of mass
        com = calculate_center_of_mass(keypoints)
        self.com_history.append(com)

        # Calculate joint angles
        joint_angles = calculate_all_joint_angles(keypoints)

        # Calculate velocity/acceleration if we have history
        velocity = None
        acceleration = None

        if len(self.com_history) >= 2:
            prev_com = self.com_history[-2]
            dt = 1.0 / 30.0  # Assuming 30 fps, will be adjusted
            vx = (com[0] - prev_com[0]) / dt
            vy = (com[1] - prev_com[1]) / dt
            velocity = (vx, vy)
            self.velocity_history.append(velocity)

            if len(self.velocity_history) >= 2:
                prev_vel = self.velocity_history[-2]
                ax = (velocity[0] - prev_vel[0]) / dt
                ay = (velocity[1] - prev_vel[1]) / dt
                acceleration = (ax, ay)
                self.acceleration_history.append(acceleration)

        metrics = FrameMetrics(
            frame_number=frame_number,
            timestamp=timestamp,
            center_of_mass=com,
            joint_angles=joint_angles,
            velocity=velocity,
            acceleration=acceleration,
        )

        self.frame_metrics.append(metrics)
        return metrics

    def calculate_efficiency(self) -> float:
        """
        Calculate movement efficiency: direct distance / actual path length.

        Returns:
            Efficiency score from 0 to 1
        """
        if len(self.com_history) < 2:
            return 1.0

        trajectory = np.array(self.com_history)
        actual_distance = calculate_trajectory_length(trajectory)
        direct_distance = calculate_distance(trajectory[0], trajectory[-1])

        if actual_distance == 0:
            return 1.0

        return min(1.0, direct_distance / actual_distance)

    def detect_dyno(self, threshold: float = 30.0) -> list[int]:
        """
        Detect dynamic moves (dynos) based on acceleration peaks.

        Args:
            threshold: Acceleration threshold for dyno detection

        Returns:
            List of frame numbers where dynos were detected
        """
        dyno_frames = []

        for i, acc in enumerate(self.acceleration_history):
            if acc is not None:
                magnitude = np.sqrt(acc[0] ** 2 + acc[1] ** 2)
                if magnitude > threshold:
                    # Map acceleration index back to frame number
                    frame_num = i + 2  # Account for diff operations
                    dyno_frames.append(frame_num)

        return dyno_frames

    def get_summary(self, fps: float = 30.0) -> AnalysisSummary:
        """
        Generate analysis summary statistics.

        Args:
            fps: Video frame rate

        Returns:
            AnalysisSummary with aggregated metrics
        """
        if not self.frame_metrics:
            return AnalysisSummary(
                total_frames=0,
                duration=0.0,
                avg_efficiency=0.0,
                max_velocity=0.0,
                max_acceleration=0.0,
                dyno_count=0,
                total_distance=0.0,
            )

        total_frames = len(self.frame_metrics)
        duration = total_frames / fps

        # Calculate velocity/acceleration magnitudes
        velocity_magnitudes = []
        for v in self.velocity_history:
            if v is not None:
                velocity_magnitudes.append(np.sqrt(v[0] ** 2 + v[1] ** 2))

        acceleration_magnitudes = []
        for a in self.acceleration_history:
            if a is not None:
                acceleration_magnitudes.append(np.sqrt(a[0] ** 2 + a[1] ** 2))

        # Calculate joint angle statistics
        joint_angle_stats = {}
        for joint_name in ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
                           "left_hip", "right_hip", "left_knee", "right_knee"]:
            angles = [fm.joint_angles.get(joint_name, 0) for fm in self.frame_metrics if fm.joint_angles.get(joint_name, 0) > 0]
            if angles:
                joint_angle_stats[joint_name] = {
                    "min": float(np.min(angles)),
                    "max": float(np.max(angles)),
                    "avg": float(np.mean(angles)),
                }

        return AnalysisSummary(
            total_frames=total_frames,
            duration=duration,
            avg_efficiency=self.calculate_efficiency(),
            max_velocity=float(max(velocity_magnitudes)) if velocity_magnitudes else 0.0,
            max_acceleration=float(max(acceleration_magnitudes)) if acceleration_magnitudes else 0.0,
            dyno_count=len(self.detect_dyno()),
            total_distance=calculate_trajectory_length(np.array(self.com_history)),
            joint_angle_stats=joint_angle_stats,
        )

    def get_com_trajectory(self) -> list[tuple[float, float]]:
        """Get the center of mass trajectory."""
        return list(self.com_history)
