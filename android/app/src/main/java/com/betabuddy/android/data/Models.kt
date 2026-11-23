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
    /** Set for records made with the in-app camera: what was chosen and recognised while framing. */
    val capture: CaptureInfo? = null,
) {
    val climbedAtMillis: Long get() = (((climbedAt ?: date) + REFERENCE_DATE_OFFSET_SECONDS) * 1000).toLong()
    val dateMillis: Long get() = ((date + REFERENCE_DATE_OFFSET_SECONDS) * 1000).toLong()

    companion object {
        const val REFERENCE_DATE_OFFSET_SECONDS = 978_307_200.0
        fun dateFromMillis(millis: Long): Double = millis / 1000.0 - REFERENCE_DATE_OFFSET_SECONDS
    }
}

/**
 * A problem of the pack's own that the capture page took a recording for. A guess of the page's;
 * [name] is in the interface language, [nameZh] the problem's own name; with the rest they are
 * whatever the page said about it, and are only ever shown.
 */
@Serializable
data class CaptureRoute(
    val id: String,
    val confidence: Double? = null,
    val name: String? = null,
    val nameZh: String? = null,
    val grade: String? = null,
    val colour: String? = null,
    val wall: String? = null,
) {
    /** What to call it on screen: its name in the page's language, else its own name, else its id. */
    val title: String get() = name?.takeIf { it.isNotBlank() } ?: nameZh?.takeIf { it.isNotBlank() } ?: id

    /**
     * What is kept with a record: the id and the problem's own name. The name and grade the page
     * gave in the interface language of the moment are not stored; they are looked up when shown.
     */
    fun stored(): CaptureRoute = CaptureRoute(id = id, confidence = confidence, nameZh = nameZh)

    /**
     * The grade as one of the values the logbook uses (V0–V12), or null when the page's text is
     * anything else. Gyms write grades in their own ways; only this form is the same in every language.
     */
    val albumGrade: String?
        get() {
            val text = grade ?: return null
            val number = text.removePrefix("V").toIntOrNull() ?: return null
            return text.takeIf { number in 0..12 && it == "V$number" }
        }
}

/**
 * What the capture page had decided when recording started. Kept with the record for display; the
 * plan itself is the page's own format and is stored next to the clip untouched.
 */
@Serializable
data class CaptureSummary(
    val color: String? = null,
    val rgb: List<Int>? = null,
    val holds: Int = 0,
    val startHands: Int = 0,
    val startFeet: Int = 0,
    val finish: Boolean = false,
    val ok: Boolean = false,
    val issues: List<String> = emptyList(),
    val route: CaptureRoute? = null,
) {
    /** As it is written into the archive: without anything in an interface language. */
    fun stored(): CaptureSummary = copy(route = route?.stored())
}

/**
 * What a record made with the in-app camera keeps about how it was framed. Imported records, and
 * every record written before the camera existed, have none. Same fields as iOS `CaptureInfo`.
 */
@Serializable
data class CaptureInfo(
    /** The colour the user chose on the capture screen (the pack's key for it), and how it looked. */
    val colorKey: String? = null,
    val colorRGB: List<Int>? = null,
    val summary: CaptureSummary? = null,
    /** The page's capture plan is stored next to the clip and is handed to the analysis page. */
    val hasPlan: Boolean = false,
    /** Version of the pack whose capture page was used; null when the build had no capture page. */
    val packVersion: String? = null,
    /** The analysis asked for when the clip was saved has not produced a result yet. */
    val analysisQueued: Boolean? = null,
    /** The pack's own problem the capture page recognised and the user did not reject: its id, and its own name (not a translation) when the page gave one. */
    val routeId: String? = null,
    val routeName: String? = null,
    /**
     * What the analysis page said it did with the capture plan, as it said it. Null until an
     * analysis has said. Nothing may present the marks as used while `used` is not true.
     */
    val planUse: CapturePlanUse? = null,
)

/**
 * Whether the analysis page could use the capture plan, as it says with its result. A plan is not
 * used when the camera was moved between framing and recording; the analysis is then as without
 * one. Same fields as iOS `CapturePlanUse`.
 */
@Serializable
data class CapturePlanUse(
    val used: Boolean,
    /** Why not: `mismatch`, `unreadable`, or something a newer page says. */
    val reason: String? = null,
    /** Whose start and finish the result went by: `user`, `venue`, `plan` or `video`. */
    val start: String? = null,
    val finish: String? = null,
)

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
