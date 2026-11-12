"""
Movement detection module for identifying climbing techniques from pose data.
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MovementType(str, Enum):
    """Types of climbing movements that can be detected."""
    SIDE_PULL = "side_pull"
    UNDERCLING = "undercling"
    GASTON = "gaston"
    HEEL_HOOK = "heel_hook"
    TOE_HOOK = "toe_hook"
    FLAG = "flag"
    DROP_KNEE = "drop_knee"
    DYNO = "dyno"


# Chinese names for each movement type
MOVEMENT_NAMES_CN = {
    MovementType.SIDE_PULL: "侧拉",
    MovementType.UNDERCLING: "反提",
    MovementType.GASTON: "推撑",
    MovementType.HEEL_HOOK: "跟勾",
    MovementType.TOE_HOOK: "趾勾",
    MovementType.FLAG: "旗杆",
    MovementType.DROP_KNEE: "埃及步",
    MovementType.DYNO: "动态跳跃",
}

# Side names in Chinese
SIDE_NAMES_CN = {
    "left": "左侧",
    "right": "右侧",
    "both": "双侧",
}

# MediaPipe keypoint indices
KEYPOINT_INDICES = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


@dataclass
class DetectedMovement:
    """Represents a detected climbing movement."""
    movement_type: MovementType
    start_frame: int
    end_frame: int
    start_timestamp: float = 0.0
    end_timestamp: float = 0.0
    side: str = "both"  # "left", "right", or "both"
    confidence: float = 0.0
    is_challenging: bool = False
    key_angles: dict = field(default_factory=dict)
    peak_frame: int = 0
    description_cn: Optional[str] = None

    @property
    def movement_name_cn(self) -> str:
        """Get Chinese name for this movement."""
        return MOVEMENT_NAMES_CN.get(self.movement_type, str(self.movement_type))

    @property
    def side_cn(self) -> str:
        """Get Chinese name for the side."""
        return SIDE_NAMES_CN.get(self.side, self.side)

    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        return self.end_timestamp - self.start_timestamp

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "movement_type": self.movement_type.value,
            "movement_name_cn": self.movement_name_cn,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "side": self.side,
            "side_cn": self.side_cn,
            "confidence": self.confidence,
            "is_challenging": self.is_challenging,
            "key_angles": self.key_angles,
            "peak_frame": self.peak_frame,
            "description_cn": self.description_cn,
        }


class MovementDetector:
    """Detects climbing movements from analysis frame data."""

    def __init__(
        self,
        min_duration_frames: int = 5,
        confidence_threshold: float = 0.6,
        merge_gap_frames: int = 3,
    ):
        """
        Initialize the movement detector.

        Args:
            min_duration_frames: Minimum frames for a valid movement
            confidence_threshold: Minimum confidence to report
            merge_gap_frames: Merge detections within this gap
        """
        self.min_duration_frames = min_duration_frames
        self.confidence_threshold = confidence_threshold
        self.merge_gap_frames = merge_gap_frames

    def detect_movements(
        self,
        frame_data: list[dict],
        fps: float = 30.0,
    ) -> list[DetectedMovement]:
        """
        Detect all climbing movements in the video.

        Args:
            frame_data: List of frame analysis data
            fps: Video frame rate

        Returns:
            List of detected movements sorted by start_frame
        """
        if not frame_data:
            return []

        all_movements = []

        # Detect each movement type
        all_movements.extend(self._detect_side_pulls(frame_data))
        all_movements.extend(self._detect_underclings(frame_data))
        all_movements.extend(self._detect_gastons(frame_data))
        all_movements.extend(self._detect_heel_hooks(frame_data))
        all_movements.extend(self._detect_toe_hooks(frame_data))
        all_movements.extend(self._detect_flags(frame_data))
        all_movements.extend(self._detect_drop_knees(frame_data))
        all_movements.extend(self._detect_dynos(frame_data))

        # Filter by confidence
        all_movements = [
            m for m in all_movements
            if m.confidence >= self.confidence_threshold
        ]

        # Sort by start frame
        all_movements.sort(key=lambda m: m.start_frame)

        # Calculate timestamps
        for m in all_movements:
            m.start_timestamp = m.start_frame / fps
            m.end_timestamp = m.end_frame / fps

        logger.info(f"Detected {len(all_movements)} movements")
        return all_movements

    def _get_keypoints_dict(self, frame: dict) -> dict[str, dict]:
        """Convert keypoints list to name-indexed dict."""
        keypoints = frame.get("keypoints", [])
        result = {}
        for kp in keypoints:
            if kp.get("visibility", 0) > 0.5:
                result[kp["name"]] = kp
        return result

    def _get_keypoint_by_index(self, frame: dict, index: int) -> Optional[dict]:
        """Get a keypoint by its MediaPipe index."""
        keypoints = frame.get("keypoints", [])
        if index < len(keypoints):
            kp = keypoints[index]
            if kp.get("visibility", 0) > 0.5:
                return kp
        return None

    def _merge_detections(
        self,
        detections: list[tuple],
        movement_type: MovementType,
    ) -> list[DetectedMovement]:
        """
        Merge consecutive frame detections into movements.

        Args:
            detections: List of (frame_num, side, confidence, angles, is_challenging)
            movement_type: Type of movement

        Returns:
            List of merged DetectedMovement objects
        """
        if not detections:
            return []

        # Group by side
        by_side = {"left": [], "right": [], "both": []}
        for det in detections:
            frame_num, side, confidence, angles, is_challenging = det
            by_side[side].append((frame_num, confidence, angles, is_challenging))

        movements = []

        for side, side_detections in by_side.items():
            if not side_detections:
                continue

            # Sort by frame number
            side_detections.sort(key=lambda x: x[0])

            # Merge consecutive detections
            current_start = side_detections[0][0]
            current_frames = [side_detections[0]]

            for i in range(1, len(side_detections)):
                frame_num = side_detections[i][0]
                prev_frame = side_detections[i - 1][0]

                if frame_num - prev_frame <= self.merge_gap_frames + 1:
                    # Continue current segment
                    current_frames.append(side_detections[i])
                else:
                    # End current segment and start new one
                    if len(current_frames) >= self.min_duration_frames:
                        movements.append(self._create_movement(
                            current_frames, side, movement_type
                        ))
                    current_start = frame_num
                    current_frames = [side_detections[i]]

            # Don't forget the last segment
            if len(current_frames) >= self.min_duration_frames:
                movements.append(self._create_movement(
                    current_frames, side, movement_type
                ))

        return movements

    def _create_movement(
        self,
        frames: list[tuple],
        side: str,
        movement_type: MovementType,
    ) -> DetectedMovement:
        """Create a DetectedMovement from a list of frame detections."""
        start_frame = frames[0][0]
        end_frame = frames[-1][0]

        # Average confidence
        avg_confidence = sum(f[1] for f in frames) / len(frames)

        # Find peak frame (highest confidence)
        peak_idx = max(range(len(frames)), key=lambda i: frames[i][1])
        peak_frame = frames[peak_idx][0]
        peak_angles = frames[peak_idx][2]

        # Check if any frame was challenging
        is_challenging = any(f[3] for f in frames)

        return DetectedMovement(
            movement_type=movement_type,
            start_frame=start_frame,
            end_frame=end_frame,
            side=side,
            confidence=avg_confidence,
            is_challenging=is_challenging,
            key_angles=peak_angles,
            peak_frame=peak_frame,
        )

    def _detect_side_pulls(self, frame_data: list[dict]) -> list[DetectedMovement]:
        """
        Detect side pull movements.

        Criteria:
        - Wrist far from shoulder horizontally
        - Elbow angle 70-140 degrees
        - Body leaning opposite to pulling arm
        """
        detections = []

        for frame in frame_data:
            frame_num = frame.get("frame_number", 0)
            angles = frame.get("joint_angles", {})
            com = frame.get("center_of_mass", [0.5, 0.5])

            for side in ["left", "right"]:
                wrist = self._get_keypoint_by_index(
                    frame, KEYPOINT_INDICES[f"{side}_wrist"]
                )
                shoulder = self._get_keypoint_by_index(
                    frame, KEYPOINT_INDICES[f"{side}_shoulder"]
                )
                elbow_angle = angles.get(f"{side}_elbow", 0)

                if not wrist or not shoulder:
                    continue

                # Check horizontal displacement
                h_displacement = abs(wrist["x"] - shoulder["x"])

                # Check elbow angle
                elbow_ok = 70 <= elbow_angle <= 140

                # Check if pulling to the side (wrist to the side of shoulder)
                pulling_side = h_displacement > 0.12

                # Check body lean (CoM shifted opposite to pulling side)
                if side == "left":
                    body_lean = com[0] > shoulder["x"]  # Body leaning right
                else:
                    body_lean = com[0] < shoulder["x"]  # Body leaning left

                if pulling_side and elbow_ok:
                    # Calculate confidence
                    confidence = min(1.0, h_displacement / 0.2) * 0.5
                    if body_lean:
                        confidence += 0.3
                    if 80 <= elbow_angle <= 120:
                        confidence += 0.2

                    # Challenging if deep pull (small elbow angle)
                    is_challenging = elbow_angle < 90

                    detections.append((
                        frame_num,
                        side,
                        confidence,
                        {"elbow": elbow_angle, "h_displacement": h_displacement},
                        is_challenging,
                    ))

        return self._merge_detections(detections, MovementType.SIDE_PULL)

    def _detect_underclings(self, frame_data: list[dict]) -> list[DetectedMovement]:
        """
        Detect undercling movements.

        Criteria:
        - Wrist below shoulder
        - Elbow angle 60-130 degrees
        - Arm pulling upward
        """
        detections = []

        for frame in frame_data:
            frame_num = frame.get("frame_number", 0)
            angles = frame.get("joint_angles", {})

            for side in ["left", "right"]:
                wrist = self._get_keypoint_by_index(
                    frame, KEYPOINT_INDICES[f"{side}_wrist"]
                )
                shoulder = self._get_keypoint_by_index(
                    frame, KEYPOINT_INDICES[f"{side}_shoulder"]
                )
                hip = self._get_keypoint_by_index(
                    frame, KEYPOINT_INDICES[f"{side}_hip"]
                )
                elbow_angle = angles.get(f"{side}_elbow", 0)
                shoulder_angle = angles.get(f"{side}_shoulder", 0)

                if not wrist or not shoulder or not hip:
                    continue

                # Wrist below shoulder (higher y = lower position)
                wrist_below = wrist["y"] > shoulder["y"] + 0.05

                # Elbow bent for pulling
                elbow_ok = 60 <= elbow_angle <= 130

                # Shoulder angle indicating arm reaching down
                shoulder_reaching = shoulder_angle > 100

                if wrist_below and elbow_ok:
                    # Calculate confidence
                    vertical_drop = wrist["y"] - shoulder["y"]
                    confidence = min(1.0, vertical_drop / 0.2) * 0.4
                    if shoulder_reaching:
                        confidence += 0.3
                    if 70 <= elbow_angle <= 110:
                        confidence += 0.3

                    # Challenging if very low undercling
                    is_challenging = wrist["y"] > hip["y"]

                    detections.append((
                        frame_num,
                        side,
                        confidence,
                        {"elbow": elbow_angle, "shoulder": shoulder_angle},
                        is_challenging,
                    ))

        return self._merge_detections(detections, MovementType.UNDERCLING)

    def _detect_gastons(self, frame_data: list[dict]) -> list[DetectedMovement]:
        """
        Detect gaston movements.

        Criteria:
        - Elbow pointed outward
        - Pushing motion (elbow outside shoulder line)
        - Elbow angle 80-150 degrees
        """
        detections = []

        for frame in frame_data:
            frame_num = frame.get("frame_number", 0)
            angles = frame.get("joint_angles", {})

            for side in ["left", "right"]:
                wrist = self._get_keypoint_by_index(
                    frame, KEYPOINT_INDICES[f"{side}_wrist"]
                )
                shoulder = self._get_keypoint_by_index(
                    frame, KEYPOINT_INDICES[f"{side}_shoulder"]
                )
                elbow = self._get_keypoint_by_index(
                    frame, KEYPOINT_INDICES[f"{side}_elbow"]
                )
                elbow_angle = angles.get(f"{side}_elbow", 0)

                if not wrist or not shoulder or not elbow:
                    continue

                # Check if elbow is outside the shoulder line (pushing outward)
                if side == "left":
                    elbow_out = elbow["x"] < shoulder["x"] - 0.05
                else:
                    elbow_out = elbow["x"] > shoulder["x"] + 0.05

                # Elbow angle for pushing
                elbow_ok = 80 <= elbow_angle <= 150

                # Wrist between shoulder and elbow horizontally (characteristic gaston position)
                if side == "left":
                    wrist_between = elbow["x"] < wrist["x"] < shoulder["x"]
                else:
                    wrist_between = shoulder["x"] < wrist["x"] < elbow["x"]

                if elbow_out and elbow_ok:
                    # Calculate confidence
                    confidence = 0.4
                    if wrist_between:
                        confidence += 0.3
                    if 90 <= elbow_angle <= 130:
                        confidence += 0.3

                    # Challenging if extended gaston
                    is_challenging = elbow_angle > 120

                    detections.append((
                        frame_num,
                        side,
                        confidence,
                        {"elbow": elbow_angle},
                        is_challenging,
                    ))

        return self._merge_detections(detections, MovementType.GASTON)

    def _detect_heel_hooks(self, frame_data: list[dict]) -> list[DetectedMovement]:
        """
        Detect heel hook movements.

        Criteria:
        - Heel at or above hip level
        - Knee bent
        - Hip angle elevated
        """
        detections = []

        for frame in frame_data:
            frame_num = frame.get("frame_number", 0)
            angles = frame.get("joint_angles", {})

            for side in ["left", "right"]:
                heel = self._get_keypoint_by_index(
                    frame, KEYPOINT_INDICES[f"{side}_heel"]
                )
                hip = self._get_keypoint_by_index(
                    frame, KEYPOINT_INDICES[f"{side}_hip"]
                )
                shoulder = self._get_keypoint_by_index(
                    frame, KEYPOINT_INDICES[f"{side}_shoulder"]
                )
                knee_angle = angles.get(f"{side}_knee", 180)
                hip_angle = angles.get(f"{side}_hip", 0)

                if not heel or not hip:
                    continue

                # Heel at or above hip level (lower y = higher position)
                heel_high = heel["y"] < hip["y"] + 0.05

                # Knee bent for hooking
                knee_bent = knee_angle < 140

                # Hip raised
                hip_raised = hip_angle > 70

                if heel_high and knee_bent:
                    # Calculate confidence
                    height_diff = hip["y"] - heel["y"]
                    confidence = min(1.0, height_diff / 0.15 + 0.3) * 0.5
                    if hip_raised:
                        confidence += 0.25
                    if knee_angle < 120:
                        confidence += 0.25

                    # Challenging if heel at or above shoulder
                    is_challenging = shoulder and heel["y"] < shoulder["y"]

                    detections.append((
                        frame_num,
                        side,
                        confidence,
                        {"knee": knee_angle, "hip": hip_angle},
                        is_challenging,
                    ))

        return self._merge_detections(detections, MovementType.HEEL_HOOK)

    def _detect_toe_hooks(self, frame_data: list[dict]) -> list[DetectedMovement]:
        """
        Detect toe hook movements.

        Criteria:
        - Toe elevated
        - Foot rotated (toe position different from ankle)
        - Hip angle indicating leg reach
        """
        detections = []

        for frame in frame_data:
            frame_num = frame.get("frame_number", 0)
            angles = frame.get("joint_angles", {})

            for side in ["left", "right"]:
                toe = self._get_keypoint_by_index(
                    frame, KEYPOINT_INDICES[f"{side}_foot_index"]
                )
                ankle = self._get_keypoint_by_index(
                    frame, KEYPOINT_INDICES[f"{side}_ankle"]
                )
                knee = self._get_keypoint_by_index(
                    frame, KEYPOINT_INDICES[f"{side}_knee"]
                )
                hip = self._get_keypoint_by_index(
                    frame, KEYPOINT_INDICES[f"{side}_hip"]
                )
                hip_angle = angles.get(f"{side}_hip", 0)

                if not toe or not ankle or not knee:
                    continue

                # Toe elevated (above knee level indicates hooking)
                toe_high = toe["y"] < knee["y"]

                # Foot rotated (significant horizontal difference between toe and ankle)
                foot_rotated = abs(toe["x"] - ankle["x"]) > 0.08

                if toe_high and foot_rotated:
                    # Calculate confidence
                    confidence = 0.4
                    if toe["y"] < ankle["y"]:  # Toe above ankle
                        confidence += 0.3
                    if 70 <= hip_angle <= 130:
                        confidence += 0.3

                    # Challenging if toe at hip level or higher
                    is_challenging = hip and toe["y"] < hip["y"]

                    detections.append((
                        frame_num,
                        side,
                        confidence,
                        {"hip": hip_angle},
                        is_challenging,
                    ))

        return self._merge_detections(detections, MovementType.TOE_HOOK)

    def _detect_flags(self, frame_data: list[dict]) -> list[DetectedMovement]:
        """
        Detect flag movements.

        Criteria:
        - One leg extended (knee angle > 150 degrees)
        - Foot far from body centerline
        - Leg not on hold (foot below hip)
        """
        detections = []

        for frame in frame_data:
            frame_num = frame.get("frame_number", 0)
            angles = frame.get("joint_angles", {})
            com = frame.get("center_of_mass", [0.5, 0.5])

            for side in ["left", "right"]:
                ankle = self._get_keypoint_by_index(
                    frame, KEYPOINT_INDICES[f"{side}_ankle"]
                )
                hip = self._get_keypoint_by_index(
                    frame, KEYPOINT_INDICES[f"{side}_hip"]
                )
                knee_angle = angles.get(f"{side}_knee", 0)

                # Check other leg's knee angle for asymmetry
                other_side = "right" if side == "left" else "left"
                other_knee_angle = angles.get(f"{other_side}_knee", 0)

                if not ankle or not hip:
                    continue

                # Leg extended
                leg_extended = knee_angle > 145

                # Foot below hip (hanging)
                foot_hanging = ankle["y"] > hip["y"]

                # Foot far from center
                lateral_extension = abs(ankle["x"] - com[0])

                # Asymmetric stance (other leg bent)
                asymmetric = other_knee_angle < 140 and knee_angle > other_knee_angle + 20

                if leg_extended and foot_hanging and lateral_extension > 0.15:
                    # Calculate confidence
                    confidence = min(1.0, lateral_extension / 0.3) * 0.4
                    if asymmetric:
                        confidence += 0.35
                    if knee_angle > 160:
                        confidence += 0.25

                    # Challenging if large extension
                    is_challenging = lateral_extension > 0.25

                    detections.append((
                        frame_num,
                        side,
                        confidence,
                        {"knee": knee_angle, "extension": lateral_extension},
                        is_challenging,
                    ))

        return self._merge_detections(detections, MovementType.FLAG)

    def _detect_drop_knees(self, frame_data: list[dict]) -> list[DetectedMovement]:
        """
        Detect drop knee (Egyptian) movements.

        Criteria:
        - Knee dropped inward relative to hip
        - Asymmetric hip angles
        - Hip rotation
        """
        detections = []

        for frame in frame_data:
            frame_num = frame.get("frame_number", 0)
            angles = frame.get("joint_angles", {})

            for side in ["left", "right"]:
                knee = self._get_keypoint_by_index(
                    frame, KEYPOINT_INDICES[f"{side}_knee"]
                )
                hip = self._get_keypoint_by_index(
                    frame, KEYPOINT_INDICES[f"{side}_hip"]
                )
                hip_angle = angles.get(f"{side}_hip", 0)
                knee_angle = angles.get(f"{side}_knee", 180)

                other_side = "right" if side == "left" else "left"
                other_hip_angle = angles.get(f"{other_side}_hip", 0)

                if not knee or not hip:
                    continue

                # Knee dropped inward (toward center)
                if side == "left":
                    knee_inward = knee["x"] > hip["x"] + 0.03
                else:
                    knee_inward = knee["x"] < hip["x"] - 0.03

                # Knee below hip
                knee_dropped = knee["y"] > hip["y"]

                # Asymmetric hip angles
                hip_asymmetry = abs(hip_angle - other_hip_angle) > 25

                # Hip angle indicating turn-in
                hip_turned = hip_angle < 100

                if knee_inward and knee_dropped and hip_turned:
                    # Calculate confidence
                    confidence = 0.35
                    if hip_asymmetry:
                        confidence += 0.35
                    if knee_angle < 100:
                        confidence += 0.3

                    # Challenging if deep drop knee
                    is_challenging = knee_angle < 70

                    detections.append((
                        frame_num,
                        side,
                        confidence,
                        {"hip": hip_angle, "knee": knee_angle},
                        is_challenging,
                    ))

        return self._merge_detections(detections, MovementType.DROP_KNEE)

    def _detect_dynos(self, frame_data: list[dict]) -> list[DetectedMovement]:
        """
        Detect dynamic (dyno) movements.

        Criteria:
        - High acceleration
        - Rapid vertical movement
        """
        detections = []
        acceleration_threshold = 12.0

        for frame in frame_data:
            frame_num = frame.get("frame_number", 0)
            accel = frame.get("acceleration")
            velocity = frame.get("velocity")

            if not accel:
                continue

            # Calculate acceleration magnitude
            accel_mag = (accel[0] ** 2 + accel[1] ** 2) ** 0.5

            if accel_mag > acceleration_threshold:
                # Calculate confidence based on acceleration
                confidence = min(1.0, accel_mag / 20.0)

                # Check velocity direction for upward movement
                vel_mag = 0
                if velocity:
                    vel_mag = (velocity[0] ** 2 + velocity[1] ** 2) ** 0.5

                # Challenging if high acceleration
                is_challenging = accel_mag > 18.0

                detections.append((
                    frame_num,
                    "both",
                    confidence,
                    {"acceleration": accel_mag, "velocity": vel_mag},
                    is_challenging,
                ))

        return self._merge_detections(detections, MovementType.DYNO)


def get_movement_summary(movements: list[DetectedMovement]) -> dict:
    """
    Generate a summary of detected movements.

    Args:
        movements: List of detected movements

    Returns:
        Summary dictionary
    """
    if not movements:
        return {
            "total_movements": 0,
            "by_type": {},
            "challenging_count": 0,
            "total_duration": 0.0,
        }

    by_type = {}
    for m in movements:
        type_name = m.movement_name_cn
        if type_name not in by_type:
            by_type[type_name] = 0
        by_type[type_name] += 1

    return {
        "total_movements": len(movements),
        "by_type": by_type,
        "challenging_count": sum(1 for m in movements if m.is_challenging),
        "total_duration": sum(m.duration for m in movements),
    }
