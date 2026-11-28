package com.betabuddy.android.data

import kotlinx.serialization.Serializable
import kotlin.math.abs

/** Upright video coordinates, normalized to its displayed width and height. */
@Serializable
data class SubjectBounds(val x: Double, val y: Double, val width: Double, val height: Double) {
    val valid: Boolean get() = listOf(x, y, width, height).all { it.isFinite() } && x >= 0 && y >= 0 && width > 0 && height > 0 && x + width <= 1.0001 && y + height <= 1.0001
    fun expanded(amount: Double): SubjectBounds {
        val left = maxOf(0.0,x-width*amount); val top = maxOf(0.0,y-height*amount)
        return SubjectBounds(left,top,minOf(1.0,x+width*(1+amount))-left,minOf(1.0,y+height*(1+amount))-top)
    }
    fun overlap(other: SubjectBounds): Double {
        val area = maxOf(0.0,minOf(x+width,other.x+other.width)-maxOf(x,other.x))*maxOf(0.0,minOf(y+height,other.y+other.height)-maxOf(y,other.y))
        return area/(width*height+other.width*other.height-area).coerceAtLeast(.0000001)
    }
}
@Serializable
data class SubjectCandidate(val id: String, val bounds: SubjectBounds, val appearance: List<Double>, val pose: PoseFrame, val tracking: SubjectTracking? = null)
@Serializable
data class SubjectTracking(val token: String, val confidence: Double, val engine: String? = null)
@Serializable
data class SubjectPosition(val time: Double, val bounds: SubjectBounds)
@Serializable
data class SubjectSample(val time: Double, val candidates: List<SubjectCandidate>)
@Serializable
data class SubjectAnchor(val time: Double, val bounds: SubjectBounds, val appearance: List<Double>, val region: Boolean = false, val candidateID: String? = null) {
    constructor(time: Double, candidate: SubjectCandidate) : this(time, candidate.bounds, candidate.appearance, candidate.tracking != null, candidate.id)
}
@Serializable
data class SubjectPick(val time: Double, val personID: String? = null)
@Serializable
data class SubjectResolution(val schema: Int, val picks: List<SubjectPick>, val manual: Boolean, val unresolved: Int, val recoveryRecommended: Boolean? = null)

/** Display-only coverage of actual selections; never fills missing observations. */
class SubjectTimeline(picks: List<SubjectPick>, duration: Double) {
    data class Span(val start: Double, val end: Double) { val midpoint get() = (start + end) / 2 }
    val selected: List<Span>
    val gaps: List<Span>
    init {
        val selectedSpans = mutableListOf<Span>()
        val missing = mutableListOf<Span>()
        if (duration.isFinite() && duration > 0) {
            val samples = picks.filter { it.time.isFinite() && it.time >= 0 && it.time < duration }.sortedBy { it.time }
            val steps = samples.zipWithNext { a,b -> b.time-a.time }.filter { it > 0 }.sorted()
            val cadence = minOf(.32, if (steps.isEmpty()) .1 else steps[steps.size / 2])
            samples.forEachIndexed { index, sample ->
                if (sample.personID != null) {
                    var start = maxOf(0.0,sample.time-cadence/2); var end = minOf(duration,sample.time+cadence/2)
                    if (index == 0 && sample.time <= cadence) start = 0.0
                    if (index == samples.lastIndex && duration-sample.time <= cadence*1.5) end = duration
                    val last = selectedSpans.lastOrNull()
                    if (last != null && start <= last.end+.000001) selectedSpans[selectedSpans.lastIndex] = last.copy(end=maxOf(last.end,end))
                    else selectedSpans += Span(start,end)
                }
            }
            var cursor = 0.0
            for (span in selectedSpans) {
                if (span.start > cursor+.000001) missing += Span(cursor,span.start)
                cursor = span.end
            }
            if (cursor < duration-.000001) missing += Span(cursor,duration)
        }
        selected = selectedSpans; gaps = missing
    }
    fun nextGap(after: Double): Double? = (gaps.firstOrNull { it.midpoint > after+.05 } ?: gaps.firstOrNull())?.midpoint
}

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
