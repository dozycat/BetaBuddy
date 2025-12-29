from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.database import get_db, Video, AnalysisTask, AnalysisResult, AnalysisStatus
from app.models.schemas import BetaSuggestionRequest, BetaSuggestionResponse
from app.core.llm_client import generate_beta_suggestion, OllamaClient

router = APIRouter()


@router.post("/suggest", response_model=BetaSuggestionResponse)
async def get_beta_suggestion(
    request: BetaSuggestionRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generate AI beta suggestion for a video analysis."""
    # Get the video
    result = await db.execute(select(Video).where(Video.id == request.video_id))
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )

    # Get the latest completed analysis
    result = await db.execute(
        select(AnalysisTask)
        .where(AnalysisTask.video_id == request.video_id)
        .where(AnalysisTask.status == AnalysisStatus.COMPLETED)
        .order_by(AnalysisTask.completed_at.desc())
    )
    task = result.scalars().first()

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No completed analysis found. Please run analysis first.",
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

    # Use provided metrics or get from analysis result
    if request.metrics_summary:
        metrics_summary = request.metrics_summary
    else:
        metrics_summary = analysis_result.summary_stats or {}
        metrics_summary["joint_angle_stats"] = analysis_result.joint_angle_stats or {}

    # Get movement data from analysis result
    movements = analysis_result.movement_data or []

    # Generate beta suggestion with movement data
    suggestion = await generate_beta_suggestion(metrics_summary, movements=movements)

    if suggestion is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to generate beta suggestion. LLM service may be unavailable.",
        )

    # Store the suggestion
    analysis_result.beta_suggestion = suggestion
    await db.commit()

    return BetaSuggestionResponse(
        video_id=request.video_id,
        suggestion=suggestion,
        metrics_used=metrics_summary,
        generated_at=datetime.utcnow(),
    )


@router.get("/status")
async def check_llm_status():
    """Check if the LLM service is available."""
    client = OllamaClient()
    is_available = await client.is_available()

    return {
        "available": is_available,
        "model": client.model,
        "base_url": client.base_url,
    }
