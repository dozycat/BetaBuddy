import numpy as np
from typing import Optional
from scipy.signal import savgol_filter

from app.core.pose_estimator import KeypointData, LANDMARK_NAMES


# Body segment mass ratios (proportion of total body mass)
BODY_MASS_RATIOS = {
    "head": 0.08,
    "trunk": 0.50,
    "upper_arm_left": 0.028,
    "upper_arm_right": 0.028,
    "forearm_left": 0.016,
    "forearm_right": 0.016,
    "hand_left": 0.006,
    "hand_right": 0.006,
    "thigh_left": 0.10,
    "thigh_right": 0.10,
    "shank_left": 0.0465,
    "shank_right": 0.0465,
    "foot_left": 0.0145,
    "foot_right": 0.0145,
}

# Climbing-relevant joint angle definitions
# Each tuple contains (point1_idx, vertex_idx, point2_idx)
CLIMBING_ANGLES = {
    "left_elbow": (11, 13, 15),    # shoulder-elbow-wrist
    "right_elbow": (12, 14, 16),
    "left_shoulder": (13, 11, 23),  # elbow-shoulder-hip
    "right_shoulder": (14, 12, 24),
    "left_hip": (11, 23, 25),       # shoulder-hip-knee
    "right_hip": (12, 24, 26),
    "left_knee": (23, 25, 27),      # hip-knee-ankle
    "right_knee": (24, 26, 28),
}


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate the angle formed by three points where p2 is the vertex.

    Args:
        p1, p2, p3: Points as numpy arrays [x, y] or [x, y, z]

    Returns:
        Angle in degrees
    """
    v1 = np.array(p1[:2]) - np.array(p2[:2])
    v2 = np.array(p3[:2]) - np.array(p2[:2])

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    return float(np.degrees(angle))


def calculate_all_joint_angles(keypoints: list[KeypointData]) -> dict[str, float]:
    """
    Calculate all climbing-relevant joint angles.

    Args:
        keypoints: List of 33 KeypointData objects

    Returns:
        Dictionary mapping joint names to angles in degrees
    """
    angles = {}
    kp_array = np.array([[kp.x, kp.y] for kp in keypoints])

    for joint_name, (idx1, idx2, idx3) in CLIMBING_ANGLES.items():
        if (keypoints[idx1].visibility > 0.5 and
            keypoints[idx2].visibility > 0.5 and
            keypoints[idx3].visibility > 0.5):
            angles[joint_name] = calculate_angle(
                kp_array[idx1],
                kp_array[idx2],
                kp_array[idx3],
            )
        else:
            angles[joint_name] = 0.0

    return angles


def calculate_center_of_mass(keypoints: list[KeypointData]) -> tuple[float, float]:
    """
    Calculate the center of mass using body segment mass distribution.

    Args:
        keypoints: List of 33 KeypointData objects

    Returns:
        (x, y) coordinates of the center of mass (normalized 0-1)
    """
    kp_array = np.array([[kp.x, kp.y, kp.visibility] for kp in keypoints])

    # Define segment positions as midpoints or key points
    segments = {
        "head": kp_array[0, :2],  # nose approximates head
        "trunk": np.mean([
            kp_array[11, :2],  # left_shoulder
            kp_array[12, :2],  # right_shoulder
            kp_array[23, :2],  # left_hip
            kp_array[24, :2],  # right_hip
        ], axis=0),
        "upper_arm_left": np.mean([kp_array[11, :2], kp_array[13, :2]], axis=0),
        "upper_arm_right": np.mean([kp_array[12, :2], kp_array[14, :2]], axis=0),
        "forearm_left": np.mean([kp_array[13, :2], kp_array[15, :2]], axis=0),
        "forearm_right": np.mean([kp_array[14, :2], kp_array[16, :2]], axis=0),
        "hand_left": kp_array[15, :2],  # left_wrist
        "hand_right": kp_array[16, :2],  # right_wrist
        "thigh_left": np.mean([kp_array[23, :2], kp_array[25, :2]], axis=0),
        "thigh_right": np.mean([kp_array[24, :2], kp_array[26, :2]], axis=0),
        "shank_left": np.mean([kp_array[25, :2], kp_array[27, :2]], axis=0),
        "shank_right": np.mean([kp_array[26, :2], kp_array[28, :2]], axis=0),
        "foot_left": kp_array[27, :2],  # left_ankle
        "foot_right": kp_array[28, :2],  # right_ankle
    }

    weighted_x = 0.0
    weighted_y = 0.0
    total_weight = 0.0

    for part, coord in segments.items():
        weight = BODY_MASS_RATIOS.get(part, 0.01)
        weighted_x += coord[0] * weight
        weighted_y += coord[1] * weight
        total_weight += weight

    if total_weight == 0:
        return 0.5, 0.5

    return float(weighted_x / total_weight), float(weighted_y / total_weight)


def calculate_velocity(
    positions: np.ndarray,
    fps: float = 30.0,
    smooth: bool = True,
) -> np.ndarray:
    """
    Calculate velocity from position time series.

    Args:
        positions: Array of shape (N, 2) containing x, y positions
        fps: Frames per second
        smooth: Whether to apply Savitzky-Golay smoothing

    Returns:
        Velocity array of shape (N-1, 2)
    """
    dt = 1.0 / fps
    velocity = np.diff(positions, axis=0) / dt

    if smooth and len(velocity) >= 5:
        window_length = min(5, len(velocity) if len(velocity) % 2 == 1 else len(velocity) - 1)
        if window_length >= 3:
            velocity = savgol_filter(velocity, window_length, polyorder=2, axis=0)

    return velocity


def calculate_acceleration(
    velocity: np.ndarray,
    fps: float = 30.0,
) -> np.ndarray:
    """
    Calculate acceleration from velocity time series.

    Args:
        velocity: Array of shape (N, 2) containing vx, vy
        fps: Frames per second

    Returns:
        Acceleration array of shape (N-1, 2)
    """
    dt = 1.0 / fps
    return np.diff(velocity, axis=0) / dt


def calculate_kinematics(
    positions: np.ndarray,
    fps: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate both velocity and acceleration from positions.

    Args:
        positions: Array of shape (N, 2)
        fps: Frames per second

    Returns:
        Tuple of (velocity, acceleration) arrays
    """
    velocity = calculate_velocity(positions, fps)
    acceleration = calculate_acceleration(velocity, fps)
    return velocity, acceleration


def get_support_points(keypoints: list[KeypointData]) -> np.ndarray:
    """
    Get contact points (hands and feet) for stability calculation.

    Args:
        keypoints: List of KeypointData

    Returns:
        Array of shape (N, 2) with visible contact points
    """
    contact_indices = [
        15, 16,  # wrists (hands)
        27, 28,  # ankles (feet)
        31, 32,  # foot indices
    ]

    points = []
    for idx in contact_indices:
        if keypoints[idx].visibility > 0.5:
            points.append([keypoints[idx].x, keypoints[idx].y])

    return np.array(points) if points else np.array([[0.5, 0.5]])


def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points."""
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def calculate_trajectory_length(positions: np.ndarray) -> float:
    """Calculate the total length of a trajectory."""
    if len(positions) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
