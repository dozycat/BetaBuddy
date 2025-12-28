import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.config import settings
from app.models.database import get_db, Video, AnalysisTask, AnalysisResult, AnalysisStatus
from app.models.schemas import (
    VideoResponse,
    VideoListResponse,
    AnnotateVideoRequest,
    AnnotateVideoResponse,
)
from app.utils.video_utils import get_video_metadata, is_valid_video_format

router = APIRouter()


@router.post("/upload", response_model=VideoResponse, status_code=status.HTTP_201_CREATED)
async def upload_video(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided",
        )

    if not is_valid_video_format(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video format. Supported formats: {settings.supported_video_formats}",
        )

    # Generate unique filename
    video_id = str(uuid.uuid4())
    suffix = Path(file.filename).suffix
    filename = f"{video_id}{suffix}"
    file_path = settings.upload_dir / filename

    # Save file
    try:
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}",
        )

    # Get file size
    file_size = file_path.stat().st_size
    if file_size > settings.max_upload_size:
        file_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.max_upload_size / 1024 / 1024}MB",
        )

    # Extract video metadata
    try:
        metadata = get_video_metadata(file_path)
    except Exception as e:
        file_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process video: {str(e)}",
        )

    # Create database record
    video = Video(
        id=video_id,
        filename=filename,
        original_filename=file.filename,
        file_path=str(file_path),
        file_size=file_size,
        duration=metadata.get("duration"),
        fps=metadata.get("fps"),
        width=metadata.get("width"),
        height=metadata.get("height"),
        total_frames=metadata.get("total_frames"),
    )

    db.add(video)
    await db.commit()
    await db.refresh(video)

    return VideoResponse(
        id=video.id,
        filename=video.filename,
        original_filename=video.original_filename,
        file_path=video.file_path,
        file_size=video.file_size,
        duration=video.duration,
        fps=video.fps,
        width=video.width,
        height=video.height,
        total_frames=video.total_frames,
        created_at=video.created_at,
        preview_url=f"/uploads/{video.filename}",
    )


@router.get("", response_model=VideoListResponse)
async def list_videos(
    skip: int = 0,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Video).order_by(Video.created_at.desc()).offset(skip).limit(limit)
    )
    videos = result.scalars().all()

    count_result = await db.execute(select(Video))
    total = len(count_result.scalars().all())

    return VideoListResponse(
        videos=[
            VideoResponse(
                id=v.id,
                filename=v.filename,
                original_filename=v.original_filename,
                file_path=v.file_path,
                file_size=v.file_size,
                duration=v.duration,
                fps=v.fps,
                width=v.width,
                height=v.height,
                total_frames=v.total_frames,
                created_at=v.created_at,
                preview_url=f"/uploads/{v.filename}",
            )
            for v in videos
        ],
        total=total,
    )


@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(
    video_id: str,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Video).where(Video.id == video_id))
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )

    return VideoResponse(
        id=video.id,
        filename=video.filename,
        original_filename=video.original_filename,
        file_path=video.file_path,
        file_size=video.file_size,
        duration=video.duration,
        fps=video.fps,
        width=video.width,
        height=video.height,
        total_frames=video.total_frames,
        created_at=video.created_at,
        preview_url=f"/uploads/{video.filename}",
    )


@router.delete("/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_video(
    video_id: str,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Video).where(Video.id == video_id))
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )

    # Delete file
    file_path = Path(video.file_path)
    if file_path.exists():
        file_path.unlink()

    # Delete database record
    await db.execute(delete(Video).where(Video.id == video_id))
    await db.commit()


@router.post("/{video_id}/annotate", response_model=AnnotateVideoResponse)
async def generate_annotated_video(
    video_id: str,
    request: AnnotateVideoRequest = AnnotateVideoRequest(),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate an annotated video with keypoints, skeleton, and metrics overlay.

    Requires a completed analysis for the video.
    """
    from app.core.annotator import AnnotatedVideoGenerator, AnnotationConfig

    # Check if video exists
    result = await db.execute(select(Video).where(Video.id == video_id))
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )

    # Find the latest completed analysis
    result = await db.execute(
        select(AnalysisTask)
        .where(AnalysisTask.video_id == video_id)
        .where(AnalysisTask.status == AnalysisStatus.COMPLETED)
        .order_by(AnalysisTask.completed_at.desc())
    )
    task = result.scalar_one_or_none()

    if not task:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No completed analysis found. Please analyze the video first.",
        )

    # Get the analysis result
    result = await db.execute(
        select(AnalysisResult).where(AnalysisResult.task_id == task.id)
    )
    analysis_result = result.scalar_one_or_none()

    if not analysis_result or not analysis_result.frame_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Analysis result or frame data not found.",
        )

    # Generate output path
    output_filename = f"{video_id}_annotated.mp4"
    output_path = settings.upload_dir / output_filename

    # Create annotation config from request
    config = AnnotationConfig(
        draw_keypoints=request.draw_keypoints,
        draw_skeleton=request.draw_skeleton,
        draw_com=request.draw_com,
        draw_com_trajectory=request.draw_com_trajectory,
        draw_metrics_overlay=request.draw_metrics_overlay,
    )

    # Generate annotated video using stored keypoints
    generator = AnnotatedVideoGenerator(
        input_path=video.file_path,
        output_path=str(output_path),
        config=config,
    )

    # Use generate_from_existing_keypoints for faster processing
    success = generator.generate_from_existing_keypoints(
        frame_data=analysis_result.frame_data,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate annotated video.",
        )

    # Update analysis result with annotated video path
    analysis_result.annotated_video_path = str(output_path)
    await db.commit()

    return AnnotateVideoResponse(
        video_id=video_id,
        annotated_video_url=f"/uploads/{output_filename}",
        message="Annotated video generated successfully",
    )


@router.get("/{video_id}/annotated")
async def get_annotated_video(
    video_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get the URL for the annotated video if it exists.
    """
    # Find the latest analysis result with annotated video
    result = await db.execute(
        select(AnalysisTask)
        .where(AnalysisTask.video_id == video_id)
        .where(AnalysisTask.status == AnalysisStatus.COMPLETED)
        .order_by(AnalysisTask.completed_at.desc())
    )
    task = result.scalar_one_or_none()

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No completed analysis found for this video.",
        )

    result = await db.execute(
        select(AnalysisResult).where(AnalysisResult.task_id == task.id)
    )
    analysis_result = result.scalar_one_or_none()

    if not analysis_result or not analysis_result.annotated_video_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Annotated video not found. Generate it first using POST /{video_id}/annotate",
        )

    # Check if file exists
    if not Path(analysis_result.annotated_video_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Annotated video file not found on disk.",
        )

    output_filename = f"{video_id}_annotated.mp4"
    return {
        "video_id": video_id,
        "annotated_video_url": f"/uploads/{output_filename}",
    }
