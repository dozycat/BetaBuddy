import asyncio
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import json

from app.config import settings
from app.models.database import get_db, Video, AnalysisTask, AnalysisResult, AnalysisStatus
from app.models.schemas import (
    AnalysisTaskResponse,
    AnalysisResultResponse,
    AnalysisStatusEnum,
)
from app.core.pose_estimator import PoseEstimator
from app.core.metrics import ClimbingMetrics
from app.utils.video_utils import VideoProcessor


router = APIRouter()

# Store for active analysis tasks and their connections
active_tasks: dict[str, dict] = {}


async def run_analysis(
    task_id: str,
    video_id: str,
    video_path: str,
    fps: float,
    db_session_factory,
):
    """Background task to run video analysis."""
    async with db_session_factory() as db:
        try:
            # Update task status
            result = await db.execute(
                select(AnalysisTask).where(AnalysisTask.id == task_id)
            )
            task = result.scalar_one_or_none()
            if not task:
                return

            task.status = AnalysisStatus.PROCESSING
            task.started_at = datetime.utcnow()
            await db.commit()

            # Initialize analysis components
            metrics = ClimbingMetrics()
            frame_data = []

            with VideoProcessor(video_path) as video, PoseEstimator() as pose_estimator:
                total_frames = video.metadata["total_frames"]
                video_fps = video.metadata["fps"] or fps

                for frame_num, frame in video.read_frames():
                    # Convert BGR to RGB
                    import cv2
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Detect pose
                    keypoints = pose_estimator.process_frame(frame_rgb)

                    if keypoints:
                        timestamp = frame_num / video_fps
                        frame_metrics = metrics.process_frame(
                            frame_number=frame_num,
                            timestamp=timestamp,
                            keypoints=keypoints,
                        )

                        # Store frame data (subset for storage efficiency)
                        frame_data.append({
                            "frame_number": frame_metrics.frame_number,
                            "timestamp": frame_metrics.timestamp,
                            "center_of_mass": frame_metrics.center_of_mass,
                            "joint_angles": frame_metrics.joint_angles,
                            "stability_score": frame_metrics.stability_score,
                        })

                    # Update progress
                    progress = (frame_num + 1) / total_frames * 100
                    task.progress = progress

                    # Notify WebSocket clients
                    if task_id in active_tasks:
                        active_tasks[task_id]["progress"] = progress
                        active_tasks[task_id]["current_frame"] = frame_num

                    # Yield control periodically
                    if frame_num % 10 == 0:
                        await db.commit()
                        await asyncio.sleep(0)

            # Generate summary
            summary = metrics.get_summary(fps=video_fps)

            # Create analysis result
            analysis_result = AnalysisResult(
                task_id=task_id,
                total_frames_analyzed=summary.total_frames,
                avg_stability_score=summary.avg_stability_score,
                avg_efficiency=summary.avg_efficiency,
                max_acceleration=summary.max_acceleration,
                dyno_detected=summary.dyno_count,
                frame_data=frame_data,
                summary_stats={
                    "duration": summary.duration,
                    "avg_stability_score": summary.avg_stability_score,
                    "min_stability_score": summary.min_stability_score,
                    "max_stability_score": summary.max_stability_score,
                    "avg_efficiency": summary.avg_efficiency,
                    "max_velocity": summary.max_velocity,
                    "max_acceleration": summary.max_acceleration,
                    "dyno_count": summary.dyno_count,
                    "total_distance": summary.total_distance,
                },
                joint_angle_stats=summary.joint_angle_stats,
                com_trajectory=metrics.get_com_trajectory(),
            )

            db.add(analysis_result)

            # Update task status
            task.status = AnalysisStatus.COMPLETED
            task.progress = 100.0
            task.completed_at = datetime.utcnow()
            await db.commit()

            # Notify completion
            if task_id in active_tasks:
                active_tasks[task_id]["status"] = "completed"
                active_tasks[task_id]["result_id"] = analysis_result.id

        except Exception as e:
            # Update task with error
            result = await db.execute(
                select(AnalysisTask).where(AnalysisTask.id == task_id)
            )
            task = result.scalar_one_or_none()
            if task:
                task.status = AnalysisStatus.FAILED
                task.error_message = str(e)
                await db.commit()

            if task_id in active_tasks:
                active_tasks[task_id]["status"] = "failed"
                active_tasks[task_id]["error"] = str(e)


@router.get("/videos/{video_id}/analyze", response_model=AnalysisTaskResponse)
async def start_analysis(
    video_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Start video analysis task."""
    # Check if video exists
    result = await db.execute(select(Video).where(Video.id == video_id))
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )

    # Check for existing pending/processing task
    result = await db.execute(
        select(AnalysisTask)
        .where(AnalysisTask.video_id == video_id)
        .where(AnalysisTask.status.in_([AnalysisStatus.PENDING, AnalysisStatus.PROCESSING]))
    )
    existing_task = result.scalar_one_or_none()

    if existing_task:
        return AnalysisTaskResponse(
            id=existing_task.id,
            video_id=existing_task.video_id,
            status=AnalysisStatusEnum(existing_task.status.value),
            progress=existing_task.progress,
            error_message=existing_task.error_message,
            created_at=existing_task.created_at,
            started_at=existing_task.started_at,
            completed_at=existing_task.completed_at,
            websocket_url=f"/ws/analysis/{existing_task.id}",
        )

    # Create new analysis task
    task = AnalysisTask(video_id=video_id)
    db.add(task)
    await db.commit()
    await db.refresh(task)

    # Initialize task tracking
    active_tasks[task.id] = {
        "status": "pending",
        "progress": 0,
        "current_frame": 0,
    }

    # Start background analysis
    from app.models.database import async_session
    background_tasks.add_task(
        run_analysis,
        task.id,
        video_id,
        video.file_path,
        video.fps or settings.default_fps,
        async_session,
    )

    return AnalysisTaskResponse(
        id=task.id,
        video_id=task.video_id,
        status=AnalysisStatusEnum(task.status.value),
        progress=task.progress,
        error_message=task.error_message,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        websocket_url=f"/ws/analysis/{task.id}",
    )


@router.get("/videos/{video_id}/results", response_model=AnalysisResultResponse)
async def get_analysis_results(
    video_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get analysis results for a video."""
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
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No completed analysis found for this video",
        )

    # Get the analysis result
    result = await db.execute(
        select(AnalysisResult).where(AnalysisResult.task_id == task.id)
    )
    analysis_result = result.scalar_one_or_none()

    if not analysis_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis result not found",
        )

    return AnalysisResultResponse(
        id=analysis_result.id,
        task_id=analysis_result.task_id,
        total_frames_analyzed=analysis_result.total_frames_analyzed,
        avg_stability_score=analysis_result.avg_stability_score,
        avg_efficiency=analysis_result.avg_efficiency,
        max_acceleration=analysis_result.max_acceleration,
        dyno_detected=analysis_result.dyno_detected,
        summary_stats=analysis_result.summary_stats,
        joint_angle_stats=analysis_result.joint_angle_stats,
        com_trajectory=analysis_result.com_trajectory,
        beta_suggestion=analysis_result.beta_suggestion,
        created_at=analysis_result.created_at,
    )


@router.get("/tasks/{task_id}", response_model=AnalysisTaskResponse)
async def get_task_status(
    task_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get the status of an analysis task."""
    result = await db.execute(
        select(AnalysisTask).where(AnalysisTask.id == task_id)
    )
    task = result.scalar_one_or_none()

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )

    return AnalysisTaskResponse(
        id=task.id,
        video_id=task.video_id,
        status=AnalysisStatusEnum(task.status.value),
        progress=task.progress,
        error_message=task.error_message,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        websocket_url=f"/ws/analysis/{task.id}",
    )
