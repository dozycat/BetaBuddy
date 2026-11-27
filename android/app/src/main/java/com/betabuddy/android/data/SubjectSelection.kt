package com.betabuddy.android.data

import kotlinx.serialization.Serializable
import kotlin.math.abs

/** Upright video coordinates, normalized to its displayed width and height. */
@Serializable
data class SubjectBounds(val x: Double, val y: Double, val width: Double, val height: Double) {
    val valid: Boolean get() = listOf(x, y, width, height).all { it.isFinite() } && x >= 0 && y >= 0 && width > 0 && height > 0 && x + width <= 1.0001 && y + height <= 1.0001
}
@Serializable
data class SubjectCandidate(val id: String, val bounds: SubjectBounds, val appearance: List<Double>, val pose: PoseFrame, val tracking: SubjectTracking? = null)
@Serializable
data class SubjectTracking(val token: String, val confidence: Double)
@Serializable
data class SubjectPosition(val time: Double, val bounds: SubjectBounds)
@Serializable
data class SubjectSample(val time: Double, val candidates: List<SubjectCandidate>)
@Serializable
data class SubjectAnchor(val time: Double, val bounds: SubjectBounds, val appearance: List<Double>) {
    constructor(time: Double, candidate: SubjectCandidate) : this(time, candidate.bounds, candidate.appearance)
}
@Serializable
data class SubjectPick(val time: Double, val personID: String? = null)
@Serializable
data class SubjectResolution(val schema: Int, val picks: List<SubjectPick>, val manual: Boolean, val unresolved: Int)

/** Candidate ids from the private policy are the only source of selected poses. */
fun PoseAnalysis.selecting(result: SubjectResolution): PoseAnalysis {
    val samples = requireNotNull(people)
    require(result.schema == 1 && result.picks.size == samples.size)
    val selected = samples.zip(result.picks).mapNotNull { (sample, pick) ->
        require(abs(sample.time-pick.time) < 0.0001)
        pick.personID?.let { id ->
            requireNotNull(sample.candidates.firstOrNull { it.id == id }) { "Unknown selected candidate" }.also {
                require(it.bounds.valid && it.pose.time.isFinite() && abs(it.pose.time-sample.time) < 0.0001)
            }
        }
    }
    return copy(frames = selected.map { it.pose }.filter { it.hasReliableTorso }, subject = result, subjectPositions = selected.map { SubjectPosition(it.pose.time,it.bounds) })
}
fun PoseAnalysis.subjectSample(time: Double): SubjectSample? {
    val samples = people ?: return null
    var low = 0; var high = samples.size
    while (low < high) {
        val mid = (low+high)/2
        if (samples[mid].time < time) low = mid+1 else high = mid
    }
    return listOf(low-1,low).filter { it in samples.indices }.map { samples[it] }
        .minByOrNull { abs(it.time-time) }?.takeIf { abs(it.time-time) <= .16 }
}
