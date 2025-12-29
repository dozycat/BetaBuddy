"""
Movement detection API routes.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import logging

from app.config import settings
from app.models.database import get_db, Video, AnalysisTask, AnalysisResult, AnalysisStatus
from app.models.schemas import (
    MovementDetectionRequest,
    MovementDetectionResponse,
    DetectedMovementSchema,
    TimelineEntry,
    MovementSummary,
)
from app.core.movement_detector import MovementDetector, get_movement_summary
from app.core.llm_client import generate_movement_descriptions
from app.core.annotator import AnnotatedVideoGenerator, AnnotationConfig

logger = logging.getLogger(__name__)

router = APIRouter()


def build_timeline(movements: list[dict], duration: float) -> list[TimelineEntry]:
    """
    Build a timeline of movement events.

    Args:
        movements: List of detected movements
        duration: Video duration in seconds

    Returns:
        List of timeline entries
    """
    if not movements:
        return []

    # Collect all unique timestamps where movements start
    events = {}
    for m in movements:
        start = m.get("start_timestamp", 0)
        start_frame = m.get("start_frame", 0)
        name = m.get("movement_name_cn", "")

        if start not in events:
            events[start] = {"frame": start_frame, "movements": []}
        if name not in events[start]["movements"]:
            events[start]["movements"].append(name)

    # Sort by timestamp and create timeline entries
    timeline = [
        TimelineEntry(
            timestamp=ts,
            frame=data["frame"],
            movements=data["movements"],
        )
        for ts, data in sorted(events.items())
    ]

    return timeline


@router.post("/videos/{video_id}/detect-movements", response_model=MovementDetectionResponse)
async def detect_movements(
    video_id: str,
    request: MovementDetectionRequest = MovementDetectionRequest(),
    db: AsyncSession = Depends(get_db),
):
    """
    Detect climbing movements from completed video analysis.

    This endpoint analyzes the stored frame data from a completed analysis
    to identify common climbing techniques like side pulls, heel hooks,
    flags, drop knees, etc.

    Args:
        video_id: ID of the video to analyze
        request: Detection parameters (min_duration, confidence, generate_descriptions)

    Returns:
        MovementDetectionResponse with detected movements and timeline
    """
    # Verify video exists
    result = await db.execute(select(Video).where(Video.id == video_id))
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )

    # Get the latest completed analysis
    result = await db.execute(
        select(AnalysisTask)
        .where(AnalysisTask.video_id == video_id)
        .where(AnalysisTask.status == AnalysisStatus.COMPLETED)
        .order_by(AnalysisTask.completed_at.desc())
    )
    task = result.scalars().first()

    if not task:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No completed analysis found. Please run analysis first.",
        )

    # Get the analysis result with frame data
    result = await db.execute(
        select(AnalysisResult).where(AnalysisResult.task_id == task.id)
    )
    analysis_result = result.scalar_one_or_none()

    if not analysis_result or not analysis_result.frame_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Analysis data not found. Please re-run analysis.",
        )

    # Run movement detection
    logger.info(f"Running movement detection for video {video_id}")
    detector = MovementDetector(
        min_duration_frames=request.min_duration_frames,
        confidence_threshold=request.confidence_threshold,
    )

    fps = video.fps or 30.0
    detected_movements = detector.detect_movements(
        frame_data=analysis_result.frame_data,
        fps=fps,
    )

    # Convert to dict for JSON serialization and LLM processing
    movements_data = [m.to_dict() for m in detected_movements]

    # Generate LLM descriptions if requested
    if request.generate_descriptions and movements_data:
        logger.info("Generating movement descriptions")
        movements_data = await generate_movement_descriptions(movements_data)

    # Store in database
    analysis_result.movement_data = movements_data
    await db.commit()

    # Regenerate annotated video with movement overlays
    if movements_data and analysis_result.frame_data:
        logger.info(f"Regenerating annotated video with {len(movements_data)} movements")
        output_filename = f"{video_id}_annotated.mp4"
        output_path = settings.upload_dir / output_filename

        config = AnnotationConfig(
            draw_keypoints=True,
            draw_skeleton=True,
            draw_com=True,
            draw_com_trajectory=True,
            draw_metrics_overlay=True,
            draw_movements=True,
        )

        generator = AnnotatedVideoGenerator(
            input_path=video.file_path,
            output_path=str(output_path),
            config=config,
        )

        success = generator.generate_from_existing_keypoints(
            frame_data=analysis_result.frame_data,
            movement_data=movements_data,
        )

        if success:
            analysis_result.annotated_video_path = str(output_path)
            await db.commit()
            logger.info(f"Annotated video regenerated: {output_path}")
        else:
            logger.warning("Failed to regenerate annotated video")

    # Build timeline
    duration = video.duration or (len(analysis_result.frame_data) / fps)
    timeline = build_timeline(movements_data, duration)

    # Get summary
    summary_data = get_movement_summary(detected_movements)
    summary = MovementSummary(
        total_movements=summary_data["total_movements"],
        by_type=summary_data["by_type"],
        challenging_count=summary_data["challenging_count"],
        total_duration=summary_data["total_duration"],
    )

    # Convert to response schema
    movements_response = [
        DetectedMovementSchema(
            movement_type=m["movement_type"],
            movement_name_cn=m["movement_name_cn"],
            start_frame=m["start_frame"],
            end_frame=m["end_frame"],
            start_timestamp=m["start_timestamp"],
            end_timestamp=m["end_timestamp"],
            side=m["side"],
            side_cn=m["side_cn"],
            confidence=m["confidence"],
            is_challenging=m["is_challenging"],
            key_angles=m["key_angles"],
            peak_frame=m["peak_frame"],
            description_cn=m.get("description_cn"),
        )
        for m in movements_data
    ]

    # Build annotated video URL
    annotated_video_url = None
    if analysis_result.annotated_video_path:
        annotated_video_url = f"/uploads/{video_id}_annotated.mp4"

    return MovementDetectionResponse(
        video_id=video_id,
        total_movements=len(movements_response),
        challenging_count=sum(1 for m in movements_response if m.is_challenging),
        movements=movements_response,
        timeline=timeline,
        summary=summary,
        annotated_video_url=annotated_video_url,
    )


@router.get("/videos/{video_id}/movements", response_model=MovementDetectionResponse)
async def get_movements(
    video_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get previously detected movements for a video.

    This returns cached movement data without re-running detection.
    Use POST /detect-movements to run new detection.

    Args:
        video_id: ID of the video

    Returns:
        MovementDetectionResponse with cached movements or empty list
    """
    # Verify video exists
    result = await db.execute(select(Video).where(Video.id == video_id))
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )

    # Get the latest completed analysis
    result = await db.execute(
        select(AnalysisTask)
        .where(AnalysisTask.video_id == video_id)
        .where(AnalysisTask.status == AnalysisStatus.COMPLETED)
        .order_by(AnalysisTask.completed_at.desc())
    )
    task = result.scalars().first()

    if not task:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No completed analysis found.",
        )

    # Get the analysis result
    result = await db.execute(
        select(AnalysisResult).where(AnalysisResult.task_id == task.id)
    )
    analysis_result = result.scalar_one_or_none()

    if not analysis_result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Analysis result not found.",
        )

    # Get cached movement data
    movements_data = analysis_result.movement_data or []

    # Build response
    fps = video.fps or 30.0
    duration = video.duration or 0
    timeline = build_timeline(movements_data, duration)

    # Calculate summary from cached data
    by_type = {}
    challenging_count = 0
    total_duration = 0.0

    for m in movements_data:
        type_cn = m.get("movement_name_cn", "")
        by_type[type_cn] = by_type.get(type_cn, 0) + 1
        if m.get("is_challenging"):
            challenging_count += 1
        total_duration += m.get("end_timestamp", 0) - m.get("start_timestamp", 0)

    summary = MovementSummary(
        total_movements=len(movements_data),
        by_type=by_type,
        challenging_count=challenging_count,
        total_duration=total_duration,
    )

    movements_response = [
        DetectedMovementSchema(
            movement_type=m["movement_type"],
            movement_name_cn=m["movement_name_cn"],
            start_frame=m["start_frame"],
            end_frame=m["end_frame"],
            start_timestamp=m["start_timestamp"],
            end_timestamp=m["end_timestamp"],
            side=m["side"],
            side_cn=m["side_cn"],
            confidence=m["confidence"],
            is_challenging=m["is_challenging"],
            key_angles=m["key_angles"],
            peak_frame=m["peak_frame"],
            description_cn=m.get("description_cn"),
        )
        for m in movements_data
    ]

    # Build annotated video URL
    annotated_video_url = None
    if analysis_result.annotated_video_path:
        annotated_video_url = f"/uploads/{video_id}_annotated.mp4"

    return MovementDetectionResponse(
        video_id=video_id,
        total_movements=len(movements_response),
        challenging_count=challenging_count,
        movements=movements_response,
        timeline=timeline,
        summary=summary,
        annotated_video_url=annotated_video_url,
    )
