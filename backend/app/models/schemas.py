from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


# Enums
class AnalysisStatusEnum(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# Keypoint schemas
class Keypoint(BaseModel):
    x: float
    y: float
    z: Optional[float] = None
    visibility: float
    name: str


class KeypointFrame(BaseModel):
    keypoints: list[Keypoint]
    timestamp: float


# Frame analysis schemas
class FrameAnalysis(BaseModel):
    frame_number: int
    timestamp: float
    keypoints: list[Keypoint]
    center_of_mass: tuple[float, float]
    joint_angles: dict[str, float]
    velocity: Optional[tuple[float, float]] = None
    acceleration: Optional[tuple[float, float]] = None


# Video schemas
class VideoBase(BaseModel):
    original_filename: str


class VideoCreate(VideoBase):
    pass


class VideoMetadata(BaseModel):
    duration: Optional[float] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    total_frames: Optional[int] = None


class VideoResponse(BaseModel):
    id: str
    filename: str
    original_filename: str
    file_path: str
    file_size: int
    duration: Optional[float] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    total_frames: Optional[int] = None
    created_at: datetime
    preview_url: Optional[str] = None
    thumbnail_url: Optional[str] = None  # GIF thumbnail URL

    class Config:
        from_attributes = True


class ThumbnailResponse(BaseModel):
    """Response after generating/regenerating thumbnail."""
    video_id: str
    thumbnail_url: str
    message: str


class VideoListResponse(BaseModel):
    videos: list[VideoResponse]
    total: int


# Analysis task schemas
class AnalysisTaskCreate(BaseModel):
    video_id: str


class AnalysisTaskResponse(BaseModel):
    id: str
    video_id: str
    status: AnalysisStatusEnum
    progress: float
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    websocket_url: Optional[str] = None

    class Config:
        from_attributes = True


# Analysis result schemas
class JointAngleStats(BaseModel):
    left_elbow: dict[str, float]  # min, max, avg
    right_elbow: dict[str, float]
    left_shoulder: dict[str, float]
    right_shoulder: dict[str, float]
    left_hip: dict[str, float]
    right_hip: dict[str, float]
    left_knee: dict[str, float]
    right_knee: dict[str, float]


class SummaryStats(BaseModel):
    avg_efficiency: float
    max_velocity: float
    max_acceleration: float
    dyno_count: int
    total_distance: float
    meters_per_unit: Optional[float] = None  # Conversion factor from units to meters


class AnalysisResultResponse(BaseModel):
    id: str
    task_id: str
    total_frames_analyzed: int
    avg_efficiency: Optional[float] = None
    max_acceleration: Optional[float] = None
    dyno_detected: int = 0
    summary_stats: Optional[dict] = None
    joint_angle_stats: Optional[dict] = None
    com_trajectory: Optional[list[list[float]]] = None
    beta_suggestion: Optional[str] = None
    annotated_video_url: Optional[str] = None
    meters_per_unit: Optional[float] = None  # Conversion factor from units to meters
    created_at: datetime

    class Config:
        from_attributes = True


class AnalysisStartRequest(BaseModel):
    """Request parameters for starting video analysis."""
    height_m: Optional[float] = Field(None, gt=0, le=3, description="User height in meters")
    arm_span_m: Optional[float] = Field(None, gt=0, le=4, description="User arm span in meters")


class AnnotateVideoRequest(BaseModel):
    """Request to generate annotated video."""
    draw_keypoints: bool = True
    draw_skeleton: bool = True
    draw_com: bool = True
    draw_com_trajectory: bool = True
    draw_metrics_overlay: bool = True
    draw_movements: bool = True  # Draw detected movement labels


class AnnotateVideoResponse(BaseModel):
    """Response after generating annotated video."""
    video_id: str
    annotated_video_url: str
    message: str


class VideoAnalysisResult(BaseModel):
    video_id: str
    total_frames: int
    fps: float
    duration: float
    frames: list[FrameAnalysis]
    summary: dict
    beta_suggestion: Optional[str] = None


# Beta suggestion schemas
class BetaSuggestionRequest(BaseModel):
    video_id: str
    metrics_summary: Optional[dict] = None


class BetaSuggestionResponse(BaseModel):
    video_id: str
    suggestion: str
    metrics_used: dict
    generated_at: datetime


# WebSocket message schemas
class WSMessage(BaseModel):
    type: str  # progress, keypoints, metrics, complete, error
    data: dict


class WSProgressMessage(BaseModel):
    type: str = "progress"
    progress: float
    current_frame: int
    total_frames: int


class WSKeypointsMessage(BaseModel):
    type: str = "keypoints"
    frame_number: int
    keypoints: list[Keypoint]
    center_of_mass: tuple[float, float]


class WSMetricsMessage(BaseModel):
    type: str = "metrics"
    frame_number: int
    joint_angles: dict[str, float]
    velocity: Optional[tuple[float, float]] = None


class WSCompleteMessage(BaseModel):
    type: str = "complete"
    task_id: str
    result_id: str
    summary: dict


class WSErrorMessage(BaseModel):
    type: str = "error"
    message: str
    details: Optional[str] = None


# Movement detection schemas
class MovementDetectionRequest(BaseModel):
    """Request parameters for movement detection."""
    min_duration_frames: int = Field(5, ge=1, le=30, description="Minimum frames to consider a movement")
    confidence_threshold: float = Field(0.6, ge=0.0, le=1.0, description="Minimum confidence to report")
    generate_descriptions: bool = Field(True, description="Whether to generate LLM descriptions")


class DetectedMovementSchema(BaseModel):
    """Schema for a detected climbing movement."""
    movement_type: str
    movement_name_cn: str
    start_frame: int
    end_frame: int
    start_timestamp: float
    end_timestamp: float
    side: str  # "left", "right", or "both"
    side_cn: str
    confidence: float
    is_challenging: bool
    key_angles: dict[str, float]
    peak_frame: int
    description_cn: Optional[str] = None


class TimelineEntry(BaseModel):
    """Entry in the movement timeline."""
    timestamp: float
    frame: int
    movements: list[str]  # Movement names active at this point


class MovementSummary(BaseModel):
    """Summary of detected movements."""
    total_movements: int
    by_type: dict[str, int]  # Movement type -> count
    challenging_count: int
    total_duration: float


class MovementDetectionResponse(BaseModel):
    """Response from movement detection endpoint."""
    video_id: str
    total_movements: int
    challenging_count: int
    movements: list[DetectedMovementSchema]
    timeline: list[TimelineEntry]
    summary: MovementSummary
    annotated_video_url: Optional[str] = None
