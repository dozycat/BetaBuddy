package com.betabuddy.android.data

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
data class PoseFrame(
    val time: Double,
    val joints: List<Joint>,
    /**
     * The landmarker's world landmarks for the same sample: 33 × `[x, y, z, visibility]`.
     * Absent in records written before they were kept; those need pose recognition again
     * before the pack's analysis page can use them.
     */
    val world: List<List<Double>>? = null,
) {
    val hasReliableTorso: Boolean
        get() = joints.size == 33 && joints[11].valid && joints[12].valid && joints[23].valid && joints[24].valid

    val hasWorldLandmarks: Boolean
        get() = joints.size == 33 && world?.size == 33 && world.all { it.size == 4 }
}

/**
 * Pose recognition output for one clip. Fields older builds wrote and this one no longer uses
 * (`processingSeconds`, `moments`) are ignored when an archive is read.
 */
@Serializable
data class PoseAnalysis(
    val frames: List<PoseFrame>,
    val sampledFrames: Int,
    val duration: Double,
    val aspectRatio: Double,
) {
    /** False for records analysed before world landmarks were kept, and for clips not analysed yet. */
    val hasWorldLandmarks: Boolean
        get() = frames.any { it.hasWorldLandmarks }

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
    val pendingAnalysis: String? = null,
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

/** Four decimal places are enough for stored landmarks and keep the archive small. */
fun rounded4(value: Double): Double = Math.round(value * 10_000) / 10_000.0

/** Uppercase, as iOS writes UUIDs. */
fun newId(): String = UUID.randomUUID().toString().uppercase(Locale.ROOT)

/** Timecode is always mm:ss, in every language. */
fun clockText(time: Double): String {
    val seconds = maxOf(0, (if (time.isFinite()) time else 0.0).toInt())
    return String.format(Locale.ROOT, "%02d:%02d", seconds / 60, seconds % 60)
}
