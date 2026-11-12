"""
Unit conversion module for converting normalized coordinates to real-world meters.

Uses user-provided height and arm span to calculate a conversion factor
based on detected pose keypoints.
"""

import math
from typing import Optional
from app.core.pose_estimator import KeypointData


def get_keypoint_by_name(
    keypoints: list[KeypointData], name: str
) -> Optional[KeypointData]:
    """Get a keypoint by its landmark name."""
    for kp in keypoints:
        if kp.name == name:
            return kp
    return None


def calculate_distance(p1: KeypointData, p2: KeypointData) -> float:
    """Calculate Euclidean distance between two keypoints in normalized coords."""
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    return math.sqrt(dx * dx + dy * dy)


def estimate_height_in_units(
    keypoints: list[KeypointData], min_visibility: float = 0.5
) -> Optional[float]:
    """
    Estimate person's height in normalized units using nose and ankle keypoints.

    Uses the average of left and right ankle positions if both are visible.

    Returns:
        Height in normalized units, or None if keypoints not sufficiently visible.
    """
    nose = get_keypoint_by_name(keypoints, "nose")
    left_ankle = get_keypoint_by_name(keypoints, "left_ankle")
    right_ankle = get_keypoint_by_name(keypoints, "right_ankle")

    if not nose or nose.visibility < min_visibility:
        return None

    # Use average ankle position if both visible, otherwise use whichever is visible
    ankle_y = None
    if left_ankle and left_ankle.visibility >= min_visibility:
        if right_ankle and right_ankle.visibility >= min_visibility:
            ankle_y = (left_ankle.y + right_ankle.y) / 2
        else:
            ankle_y = left_ankle.y
    elif right_ankle and right_ankle.visibility >= min_visibility:
        ankle_y = right_ankle.y

    if ankle_y is None:
        return None

    # Height is the vertical distance (y increases downward in image coords)
    height = abs(ankle_y - nose.y)
    return height if height > 0.1 else None  # Sanity check: at least 10% of frame


def estimate_arm_span_in_units(
    keypoints: list[KeypointData], min_visibility: float = 0.5
) -> Optional[float]:
    """
    Estimate person's arm span in normalized units using wrist keypoints.

    Returns:
        Arm span in normalized units, or None if wrists not sufficiently visible.
    """
    left_wrist = get_keypoint_by_name(keypoints, "left_wrist")
    right_wrist = get_keypoint_by_name(keypoints, "right_wrist")

    if not left_wrist or left_wrist.visibility < min_visibility:
        return None
    if not right_wrist or right_wrist.visibility < min_visibility:
        return None

    span = calculate_distance(left_wrist, right_wrist)
    return span if span > 0.1 else None  # Sanity check: at least 10% of frame


def calculate_meters_per_unit(
    keypoints: list[KeypointData],
    user_height_m: Optional[float] = None,
    user_arm_span_m: Optional[float] = None,
    min_visibility: float = 0.5,
) -> Optional[float]:
    """
    Calculate the conversion factor from normalized units to meters.

    Uses both height and arm span measurements when available and averages
    them for better accuracy.

    Args:
        keypoints: List of pose keypoints from a frame
        user_height_m: User's height in meters
        user_arm_span_m: User's arm span in meters
        min_visibility: Minimum visibility threshold for keypoints

    Returns:
        meters_per_unit conversion factor, or None if calculation not possible.
        Multiply normalized coordinates by this factor to get meters.
    """
    if not user_height_m and not user_arm_span_m:
        return None

    conversion_factors = []

    # Calculate from height
    if user_height_m and user_height_m > 0:
        height_units = estimate_height_in_units(keypoints, min_visibility)
        if height_units:
            factor = user_height_m / height_units
            conversion_factors.append(factor)

    # Calculate from arm span
    if user_arm_span_m and user_arm_span_m > 0:
        span_units = estimate_arm_span_in_units(keypoints, min_visibility)
        if span_units:
            factor = user_arm_span_m / span_units
            conversion_factors.append(factor)

    if not conversion_factors:
        return None

    # Average the conversion factors for better accuracy
    return sum(conversion_factors) / len(conversion_factors)


def calculate_meters_per_unit_from_frames(
    frames_keypoints: list[list[KeypointData]],
    user_height_m: Optional[float] = None,
    user_arm_span_m: Optional[float] = None,
    min_visibility: float = 0.5,
    max_frames: int = 30,
) -> Optional[float]:
    """
    Calculate conversion factor using multiple frames for robustness.

    Analyzes up to max_frames and averages the conversion factors
    from frames where calculation was successful.

    Args:
        frames_keypoints: List of keypoint lists from multiple frames
        user_height_m: User's height in meters
        user_arm_span_m: User's arm span in meters
        min_visibility: Minimum visibility threshold
        max_frames: Maximum number of frames to analyze

    Returns:
        Average meters_per_unit conversion factor, or None if not calculable.
    """
    if not user_height_m and not user_arm_span_m:
        return None

    factors = []
    for i, keypoints in enumerate(frames_keypoints[:max_frames]):
        if not keypoints:
            continue
        factor = calculate_meters_per_unit(
            keypoints, user_height_m, user_arm_span_m, min_visibility
        )
        if factor:
            factors.append(factor)

    if not factors:
        return None

    return sum(factors) / len(factors)
