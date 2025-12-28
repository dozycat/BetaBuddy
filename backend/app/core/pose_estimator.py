import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import urllib.request
import logging

from app.config import settings

logger = logging.getLogger(__name__)

# Model configuration
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "pose_landmarker_full.task"


def _ensure_model_exists() -> None:
    """Download the pose landmarker model if it doesn't exist."""
    if MODEL_PATH.exists():
        return

    logger.info(f"Downloading pose landmarker model to {MODEL_PATH}...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    logger.info("Model download complete.")


# MediaPipe Pose landmark indices (same as legacy API)
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


@dataclass
class KeypointData:
    x: float
    y: float
    z: float
    visibility: float
    name: str

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "visibility": self.visibility,
            "name": self.name,
        }

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


class PoseEstimator:
    def __init__(
        self,
        model_complexity: int = settings.pose_model_complexity,
        min_detection_confidence: float = settings.min_detection_confidence,
        min_tracking_confidence: float = settings.min_tracking_confidence,
    ):
        # Ensure model is downloaded
        _ensure_model_exists()

        # New MediaPipe Tasks API
        base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self.pose = vision.PoseLandmarker.create_from_options(options)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.pose.close()

    def process_frame(self, frame_rgb: np.ndarray) -> Optional[list[KeypointData]]:
        """
        Process a single RGB frame and extract pose landmarks.

        Args:
            frame_rgb: RGB image as numpy array (H, W, 3)

        Returns:
            List of 33 KeypointData objects or None if no pose detected
        """
        # Convert numpy array to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        results = self.pose.detect(mp_image)

        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None

        # Get the first detected pose
        landmarks = results.pose_landmarks[0]

        keypoints = []
        for idx, landmark in enumerate(landmarks):
            keypoints.append(
                KeypointData(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z,
                    visibility=landmark.visibility if hasattr(landmark, 'visibility') else 1.0,
                    name=LANDMARK_NAMES[idx],
                )
            )

        return keypoints

    def process_frame_world(self, frame_rgb: np.ndarray) -> Optional[tuple[list[KeypointData], list[KeypointData]]]:
        """
        Process frame and return both normalized and world coordinates.

        Returns:
            Tuple of (normalized_keypoints, world_keypoints) or None
        """
        # Convert numpy array to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        results = self.pose.detect(mp_image)

        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None

        # Get the first detected pose
        landmarks = results.pose_landmarks[0]

        normalized = []
        for idx, landmark in enumerate(landmarks):
            normalized.append(
                KeypointData(
                    x=landmark.x,
                    y=landmark.y,
                    z=landmark.z,
                    visibility=landmark.visibility if hasattr(landmark, 'visibility') else 1.0,
                    name=LANDMARK_NAMES[idx],
                )
            )

        world = []
        if results.pose_world_landmarks and len(results.pose_world_landmarks) > 0:
            world_landmarks = results.pose_world_landmarks[0]
            for idx, landmark in enumerate(world_landmarks):
                world.append(
                    KeypointData(
                        x=landmark.x,
                        y=landmark.y,
                        z=landmark.z,
                        visibility=landmark.visibility if hasattr(landmark, 'visibility') else 1.0,
                        name=LANDMARK_NAMES[idx],
                    )
                )

        return normalized, world if world else normalized

    @staticmethod
    def filter_by_visibility(
        keypoints: list[KeypointData],
        min_visibility: float = 0.5,
    ) -> list[KeypointData]:
        """Filter keypoints by visibility threshold."""
        return [kp for kp in keypoints if kp.visibility >= min_visibility]

    @staticmethod
    def keypoints_to_array(keypoints: list[KeypointData]) -> np.ndarray:
        """Convert keypoints to numpy array of shape (33, 4) - x, y, z, visibility."""
        return np.array([[kp.x, kp.y, kp.z, kp.visibility] for kp in keypoints])

    @staticmethod
    def get_landmark_index(name: str) -> int:
        """Get the index of a landmark by name."""
        return LANDMARK_NAMES.index(name)


# Convenience function for single frame processing
def detect_pose(frame_rgb: np.ndarray) -> Optional[list[KeypointData]]:
    """Detect pose in a single frame (creates new estimator each time)."""
    with PoseEstimator() as estimator:
        return estimator.process_frame(frame_rgb)
