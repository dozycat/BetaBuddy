package com.betabuddy.android.data

import com.betabuddy.android.pose.PoseMath
import java.util.Locale
import java.util.UUID
import kotlin.math.abs
import kotlinx.serialization.Serializable

// The archive mirrors the iOS app's JSON field for field (ios/BetaBuddy/Core/Models.swift), so a
// record written on one platform can be read on the other. Coordinates are top-left normalized.

@Serializable
data class Joint(val x: Double, val y: Double, val confidence: Double) {
    val valid: Boolean
        get() = x.isFinite() && y.isFinite() && confidence >= 0.6 && x >= 0.0 && x <= 1.0 && y >= 0.0 && y <= 1.0
}

@Serializable
data class PoseFrame(val time: Double, val joints: List<Joint>) {
    val hasReliableTorso: Boolean
        get() = joints.size == 33 && joints[11].valid && joints[12].valid && joints[23].valid && joints[24].valid

    /** Image-space hip midpoint; never presented as physical center of mass. */
    val hip: Joint?
        get() {
            if (joints.size != 33 || !joints[23].valid || !joints[24].valid) return null
            return Joint(
                x = (joints[23].x + joints[24].x) / 2,
                y = (joints[23].y + joints[24].y) / 2,
                confidence = minOf(joints[23].confidence, joints[24].confidence),
            )
        }
}

@Serializable
data class Moment(
    val id: String = newId(),
    val time: Double,
    /** Stored in the source language (Simplified Chinese), like iOS; shown only for unknown kinds. */
    val title: String,
    val detail: String,
    /** What triggered the cue. The UI shows text in the current language for known kinds. */
    val kind: String? = null,
) {
    companion object {
        const val RAISED_HEEL = "raisedHeel"
        const val FAST_MOVE = "fastMove"
    }
}

@Serializable
data class PoseAnalysis(
    val frames: List<PoseFrame>,
    val sampledFrames: Int,
    val duration: Double,
    val aspectRatio: Double,
    val processingSeconds: Double,
    val moments: List<Moment>,
) {
    /** Share of sampled frames in which both hips are reliable. Not a model accuracy or a score. */
    val coverage: Double
        get() = if (sampledFrames > 0) frames.count { it.hip != null }.toDouble() / sampledFrames else 0.0

    /** Aspect-corrected 2D projected angle, averaged over both elbows. Not a joint angle. */
    val averageElbowAngle: Double?
        get() {
            var sum = 0.0
            var count = 0
            for (frame in frames) {
                PoseMath.angle(frame, 11, 13, 15, aspectRatio)?.let { sum += it; count++ }
                PoseMath.angle(frame, 12, 14, 16, aspectRatio)?.let { sum += it; count++ }
            }
            return if (count == 0) null else sum / count
        }

    /** Binary search. A stale skeleton is never drawn across missing detections. */
    fun nearestFrame(time: Double): PoseFrame? {
        var low = 0
        var high = frames.size
        while (low < high) {
            val mid = (low + high) / 2
            if (frames[mid].time < time) low = mid + 1 else high = mid
        }
        val best = listOf(low - 1, low).filter { it in frames.indices }.map { frames[it] }
            .minByOrNull { abs(it.time - time) } ?: return null
        return if (abs(best.time - time) <= 0.16 && best.hasReliableTorso) best else null
    }
}

@Serializable
data class Climb(
    val id: String,
    /** Seconds since 2001-01-01T00:00:00Z, the encoding iOS uses for dates. */
    val date: Double,
    val title: String,
    val grade: String,
    val sent: Boolean,
    val note: String,
    val videoName: String,
    val thumbnailName: String,
    val analysis: PoseAnalysis,
    val gymId: String? = null,
    val gymName: String? = null,
    val climbedAt: Double? = null,
    val folderId: String? = null,
) {
    val climbedAtMillis: Long get() = (((climbedAt ?: date) + REFERENCE_DATE_OFFSET_SECONDS) * 1000).toLong()
    val dateMillis: Long get() = ((date + REFERENCE_DATE_OFFSET_SECONDS) * 1000).toLong()

    companion object {
        const val REFERENCE_DATE_OFFSET_SECONDS = 978_307_200.0
        fun dateFromMillis(millis: Long): Double = millis / 1000.0 - REFERENCE_DATE_OFFSET_SECONDS
    }
}

@Serializable
data class ClimbFolder(val id: String = newId(), val name: String)

@Serializable
data class Archive(
    val version: Int = 1,
    val climbs: List<Climb> = emptyList(),
    val folders: List<ClimbFolder> = emptyList(),
    // Archives written by other builds may carry other keys; decoding ignores them.
)

/** Stored sentinel for a climb without a grade. Kept as is so archives stay readable on both platforms. */
const val UNGRADED_VALUE = "未标级"

/** Uppercase, as iOS writes UUIDs. */
fun newId(): String = UUID.randomUUID().toString().uppercase(Locale.ROOT)

/** Timecode is always mm:ss, in every language. */
fun clockText(time: Double): String {
    val seconds = maxOf(0, (if (time.isFinite()) time else 0.0).toInt())
    return String.format(Locale.ROOT, "%02d:%02d", seconds / 60, seconds % 60)
}
