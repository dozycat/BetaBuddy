from math import sqrt
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import AnalysisResult, AnalysisStatus, AnalysisTask, Video, get_db
from app.models.schemas import (
    ComparedVideo,
    MetricComparison,
    PhaseComparison,
    VideoComparisonRequest,
    VideoComparisonResponse,
)

router = APIRouter()


async def get_latest_analysis(
    video_id: str,
    db: AsyncSession,
) -> tuple[Video, AnalysisResult]:
    video_result = await db.execute(select(Video).where(Video.id == video_id))
    video = video_result.scalar_one_or_none()

    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video {video_id} not found",
        )

    task_result = await db.execute(
        select(AnalysisTask)
        .where(AnalysisTask.video_id == video_id)
        .where(AnalysisTask.status == AnalysisStatus.COMPLETED)
        .order_by(AnalysisTask.completed_at.desc())
    )
    task = task_result.scalars().first()

    if not task:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No completed analysis found for {video.original_filename}. Run analysis first.",
        )

    analysis_result = await db.execute(
        select(AnalysisResult).where(AnalysisResult.task_id == task.id)
    )
    analysis = analysis_result.scalar_one_or_none()

    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Analysis result not found for {video.original_filename}.",
        )

    return video, analysis


def annotated_url(analysis: AnalysisResult) -> Optional[str]:
    if not analysis.annotated_video_path:
        return None
    return f"/uploads/{Path(analysis.annotated_video_path).name}"


def compared_video(video: Video, analysis: AnalysisResult) -> ComparedVideo:
    return ComparedVideo(
        id=video.id,
        original_filename=video.original_filename,
        preview_url=f"/uploads/{video.filename}",
        annotated_video_url=annotated_url(analysis),
    )


def number(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def compare_metric(
    metric: str,
    baseline: Optional[float],
    comparison: Optional[float],
    unit: str,
    higher_is_better: bool,
) -> MetricComparison:
    delta = None
    percent_change = None
    interpretation = "No comparable data"

    if baseline is not None and comparison is not None:
        delta = comparison - baseline
        if baseline != 0:
            percent_change = (delta / abs(baseline)) * 100

        if abs(delta) < 0.0001:
            interpretation = "No meaningful change"
        elif (delta > 0 and higher_is_better) or (delta < 0 and not higher_is_better):
            interpretation = "Improved"
        else:
            interpretation = "Needs attention"

    return MetricComparison(
        metric=metric,
        baseline=baseline,
        comparison=comparison,
        delta=delta,
        percent_change=percent_change,
        unit=unit,
        interpretation=interpretation,
    )


def point_distance(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def path_length(points: list[list[float]]) -> float:
    return sum(point_distance(points[index - 1], points[index]) for index in range(1, len(points)))


def vertical_gain(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    return points[0][1] - points[-1][1]


def phase_points(points: list[list[float]], start_percent: float, end_percent: float) -> list[list[float]]:
    if not points:
        return []
    start = int((len(points) - 1) * start_percent)
    end = max(start + 1, int((len(points) - 1) * end_percent))
    return points[start : end + 1]


def build_phase_comparisons(
    baseline_points: list[list[float]],
    comparison_points: list[list[float]],
) -> list[PhaseComparison]:
    phases = [
        ("Setup", 0.0, 0.33),
        ("Crux", 0.33, 0.66),
        ("Finish", 0.66, 1.0),
    ]
    comparisons = []

    for phase, start_percent, end_percent in phases:
        baseline_phase = phase_points(baseline_points, start_percent, end_percent)
        comparison_phase = phase_points(comparison_points, start_percent, end_percent)
        baseline_path = path_length(baseline_phase)
        comparison_path = path_length(comparison_phase)
        baseline_gain = vertical_gain(baseline_phase)
        comparison_gain = vertical_gain(comparison_phase)
        path_delta = comparison_path - baseline_path
        gain_delta = comparison_gain - baseline_gain

        if abs(path_delta) < 0.01 and abs(gain_delta) < 0.01:
            insight = "Movement path is similar in this phase."
        elif path_delta > 0:
            insight = "Comparison attempt travels farther through this phase; look for extra repositioning."
        else:
            insight = "Comparison attempt uses a more direct center-of-mass path in this phase."

        comparisons.append(
            PhaseComparison(
                phase=phase,
                start_percent=start_percent,
                end_percent=end_percent,
                baseline_path_length=baseline_path,
                comparison_path_length=comparison_path,
                path_delta=path_delta,
                baseline_vertical_gain=baseline_gain,
                comparison_vertical_gain=comparison_gain,
                vertical_gain_delta=gain_delta,
                insight=insight,
            )
        )

    return comparisons


def movement_summary(baseline: AnalysisResult, comparison: AnalysisResult) -> dict:
    baseline_movements = baseline.movement_data or []
    comparison_movements = comparison.movement_data or []

    return {
        "baseline_total": len(baseline_movements),
        "comparison_total": len(comparison_movements),
        "baseline_challenging": sum(1 for movement in baseline_movements if movement.get("is_challenging")),
        "comparison_challenging": sum(1 for movement in comparison_movements if movement.get("is_challenging")),
        "baseline_types": movement_counts(baseline_movements),
        "comparison_types": movement_counts(comparison_movements),
    }


def movement_counts(movements: list[dict]) -> dict[str, int]:
    counts = {}
    for movement in movements:
        name = movement.get("movement_name_cn") or movement.get("movement_type") or "unknown"
        counts[name] = counts.get(name, 0) + 1
    return counts


def build_coaching_notes(
    metrics: list[MetricComparison],
    phases: list[PhaseComparison],
    movements: dict,
) -> list[str]:
    notes = []
    metric_lookup = {metric.metric: metric for metric in metrics}

    efficiency = metric_lookup.get("Average efficiency")
    if efficiency and efficiency.delta is not None:
        if efficiency.delta < 0:
            notes.append("Efficiency dropped; reduce pauses and keep hips closer before initiating the next move.")
        elif efficiency.delta > 0:
            notes.append("Efficiency improved; preserve the smoother pacing and body tension from the comparison attempt.")

    distance = metric_lookup.get("Total CoM distance")
    if distance and distance.delta is not None and distance.delta > 0:
        notes.append("The comparison attempt covers more center-of-mass distance; search for a straighter hip path.")

    acceleration = metric_lookup.get("Max acceleration")
    if acceleration and acceleration.delta is not None and acceleration.delta > 0:
        notes.append("Peak acceleration increased; control the catch or drive earlier from the legs instead of yanking late.")

    worst_phase = max(phases, key=lambda phase: phase.path_delta, default=None)
    if worst_phase and worst_phase.path_delta > 0.01:
        notes.append(f"Focus on the {worst_phase.phase.lower()} phase first; it adds the most extra movement.")

    if movements["comparison_challenging"] > movements["baseline_challenging"]:
        notes.append("More challenging movement flags appeared in the comparison attempt; review those labels on the annotated video.")

    if not notes:
        notes.append("The two attempts are biomechanically close; use the phase deltas to tune timing rather than changing beta.")

    return notes


@router.post("/auto-beta", response_model=VideoComparisonResponse)
async def compare_auto_beta(
    request: VideoComparisonRequest,
    db: AsyncSession = Depends(get_db),
):
    if request.baseline_video_id == request.comparison_video_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Choose two different videos to compare.",
        )

    baseline_video, baseline = await get_latest_analysis(request.baseline_video_id, db)
    comparison_video, comparison = await get_latest_analysis(request.comparison_video_id, db)

    baseline_summary = baseline.summary_stats or {}
    comparison_summary = comparison.summary_stats or {}
    metric_deltas = [
        compare_metric(
            "Average efficiency",
            number(baseline_summary.get("avg_efficiency") or baseline.avg_efficiency),
            number(comparison_summary.get("avg_efficiency") or comparison.avg_efficiency),
            "",
            True,
        ),
        compare_metric(
            "Duration",
            number(baseline_summary.get("duration") or baseline_video.duration),
            number(comparison_summary.get("duration") or comparison_video.duration),
            "s",
            False,
        ),
        compare_metric(
            "Max velocity",
            number(baseline_summary.get("max_velocity")),
            number(comparison_summary.get("max_velocity")),
            "units/s",
            False,
        ),
        compare_metric(
            "Max acceleration",
            number(baseline_summary.get("max_acceleration") or baseline.max_acceleration),
            number(comparison_summary.get("max_acceleration") or comparison.max_acceleration),
            "units/s²",
            False,
        ),
        compare_metric(
            "Dyno count",
            number(baseline_summary.get("dyno_count") or baseline.dyno_detected),
            number(comparison_summary.get("dyno_count") or comparison.dyno_detected),
            "",
            False,
        ),
        compare_metric(
            "Total CoM distance",
            number(baseline_summary.get("total_distance")),
            number(comparison_summary.get("total_distance")),
            "units",
            False,
        ),
    ]

    joint_angle_deltas = {}
    baseline_angles = baseline.joint_angle_stats or {}
    comparison_angles = comparison.joint_angle_stats or {}
    for joint in sorted(set(baseline_angles) | set(comparison_angles)):
        joint_angle_deltas[joint] = compare_metric(
            joint.replace("_", " ").title(),
            number((baseline_angles.get(joint) or {}).get("avg")),
            number((comparison_angles.get(joint) or {}).get("avg")),
            "°",
            True,
        )

    baseline_points = baseline.com_trajectory or []
    comparison_points = comparison.com_trajectory or []
    phase_comparisons = build_phase_comparisons(baseline_points, comparison_points)
    movements = movement_summary(baseline, comparison)

    alignment_notes = [
        "Attempts are normalized by percent progress through each center-of-mass trajectory.",
        "Setup, crux, and finish phases split each attempt into thirds; run movement detection first for richer movement labels.",
    ]

    return VideoComparisonResponse(
        baseline_video=compared_video(baseline_video, baseline),
        comparison_video=compared_video(comparison_video, comparison),
        metric_deltas=metric_deltas,
        joint_angle_deltas=joint_angle_deltas,
        phase_comparisons=phase_comparisons,
        movement_summary=movements,
        coaching_notes=build_coaching_notes(metric_deltas, phase_comparisons, movements),
        alignment_notes=alignment_notes,
    )
