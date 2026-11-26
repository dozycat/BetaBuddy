package com.betabuddy.android.edit

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.doubleOrNull

// The analysis page's plan for an automatic edit, as the app reads it. The plan says what to show
// and when; nothing here decides anything about climbing. Times are seconds in the source clip,
// coordinates are 0–1 in the upright picture with the origin top-left. No Android classes.

/** `[x, y, w, h]`, top-left origin, fractions of the picture. */
data class Box(val x: Double, val y: Double, val w: Double, val h: Double) {
    val right: Double get() = x + w
    val bottom: Double get() = y + h
    val area: Double get() = w * h

    /** Grown by [dx] to the left and right and [dy] up and down, kept inside the picture. */
    fun grown(dx: Double, dy: Double): Box {
        val left = (x - dx).coerceIn(0.0, 1.0)
        val top = (y - dy).coerceIn(0.0, 1.0)
        return Box(left, top, (right + dx).coerceIn(0.0, 1.0) - left, (bottom + dy).coerceIn(0.0, 1.0) - top)
    }

    fun intersects(other: Box): Boolean = x < other.right && other.x < right && y < other.bottom && other.y < bottom

    fun union(other: Box): Box {
        val left = minOf(x, other.x)
        val top = minOf(y, other.y)
        return Box(left, top, maxOf(right, other.right) - left, maxOf(bottom, other.bottom) - top)
    }

    /** Intersection over union: 0 for boxes apart, 1 for the same box. */
    fun overlap(other: Box): Double {
        val width = minOf(right, other.right) - maxOf(x, other.x)
        val height = minOf(bottom, other.bottom) - maxOf(y, other.y)
        if (width <= 0 || height <= 0) return 0.0
        val shared = width * height
        return shared / (area + other.area - shared)
    }
}

/** The visible window at [t]: centred on ([cx], [cy]), `1/s` of the picture wide and high. */
data class CameraKey(val t: Double, val cx: Double, val cy: Double, val s: Double)

/** One source range of the output. [camera] is the segment's own, or null to use the plan's. */
data class EditSegment(val t0: Double, val t1: Double, val speed: Double, val camera: List<CameraKey>? = null)

/** The boxes that must stay live at [t]. */
data class KeepSample(val t: Double, val boxes: List<Box>)

/** What passer-by removal needs: where the climber is, and which pictures the clean plate is made of. */
data class EditKeep(val feather: Double, val samples: List<KeepSample>, val plateTimes: List<Double>)

data class EditPlan(
    val duration: Double,
    /** The attempt: what is shown at normal speed when slow motion is switched off. */
    val trim: Pair<Double, Double>,
    val segments: List<EditSegment>,
    val camera: List<CameraKey>,
    val keep: EditKeep?,
    /**
     * The plan's `cameraMode` is `"track"`: the camera keys (the plan's and every segment's) are a
     * dense, already smoothed track of the subject, to be followed in straight lines from key to
     * key. Otherwise the keys are a few set positions with an eased move between them.
     */
    val cameraTrack: Boolean = false,
) {
    /** The plan has more to say about time than "the attempt at normal speed". */
    val hasSlowMotion: Boolean
        get() = segments.size != 1 || segments[0].speed != 1.0 || segments[0].t0 != trim.first || segments[0].t1 != trim.second

    val hasCamera: Boolean get() = camera.isNotEmpty() || segments.any { !it.camera.isNullOrEmpty() }

    val hasKeep: Boolean get() = keep != null

    companion object {
        const val MAX_SEGMENTS = 40
        const val MIN_SPEED = 0.2
        const val MAX_SPEED = 4.0
        /** A camera that follows a climber far from the lens needs to come this close. */
        const val MAX_ZOOM = 8.0
        const val MAX_PLATE_TIMES = 25
        /** A median of fewer pictures than this does not remove anybody. */
        const val MIN_PLATE_TIMES = 3
        /** The shortest source range kept; anything shorter may not hold a single frame. */
        const val MIN_RANGE = 0.05
        private const val MAX_LENGTH = 4 * 1024 * 1024
        private const val MAX_SAMPLES = 20_000

        /**
         * Reads a stored plan. Null for anything that is not a usable version-1 plan; unknown fields
         * are ignored, and numbers a little outside their range are brought back into it.
         */
        fun parse(text: String?): EditPlan? {
            if (text == null || text.length > MAX_LENGTH) return null
            val body = try { Json.parseToJsonElement(text) as? JsonObject } catch (error: Exception) { null } ?: return null
            if (body.number("v") != 1.0) return null
            val duration = body.number("duration")?.takeIf { it > 0 } ?: return null

            val segments = (body["segments"] as? JsonArray).orEmpty().mapNotNull { item ->
                val segment = item as? JsonObject ?: return@mapNotNull null
                val t0 = segment.number("t0")?.coerceIn(0.0, duration) ?: return@mapNotNull null
                val t1 = segment.number("t1")?.coerceIn(0.0, duration) ?: return@mapNotNull null
                val speed = segment.number("speed") ?: return@mapNotNull null
                if (t1 - t0 < MIN_RANGE || speed <= 0) return@mapNotNull null
                EditSegment(t0, t1, speed.coerceIn(MIN_SPEED, MAX_SPEED), (segment["camera"] as? JsonArray)?.let(::cameraKeys))
            }.take(MAX_SEGMENTS)
            if (segments.isEmpty()) return null

            val trim = (body["trim"] as? JsonArray)?.takeIf { it.size == 2 }?.let { range ->
                val start = range[0].number()?.coerceIn(0.0, duration)
                val end = range[1].number()?.coerceIn(0.0, duration)
                if (start != null && end != null && end - start >= MIN_RANGE) start to end else null
            } ?: (segments.minOf { it.t0 } to segments.maxOf { it.t1 })

            // Any other value, or none, is the eased camera.
            val track = (body["cameraMode"] as? JsonPrimitive)?.takeIf { it.isString }?.content == "track"
            return EditPlan(duration, trim, segments, cameraKeys(body["camera"] as? JsonArray), keep(body, duration), track)
        }

        private fun cameraKeys(array: JsonArray?): List<CameraKey> = array.orEmpty().mapNotNull { item ->
            val key = item as? JsonObject ?: return@mapNotNull null
            val t = key.number("t") ?: return@mapNotNull null
            val cx = key.number("cx") ?: return@mapNotNull null
            val cy = key.number("cy") ?: return@mapNotNull null
            val s = key.number("s") ?: return@mapNotNull null
            CameraKey(t, cx.coerceIn(0.0, 1.0), cy.coerceIn(0.0, 1.0), s.coerceIn(1.0, MAX_ZOOM))
        }.sortedBy { it.t }

        private fun keep(body: JsonObject, duration: Double): EditKeep? {
            val keep = body["keep"] as? JsonObject ?: return null
            val samples = (keep["boxes"] as? JsonArray).orEmpty().take(MAX_SAMPLES).mapNotNull { item ->
                val sample = item as? JsonObject ?: return@mapNotNull null
                val t = sample.number("t") ?: return@mapNotNull null
                val boxes = (sample["b"] as? JsonArray).orEmpty().mapNotNull { box ->
                    val values = (box as? JsonArray)?.takeIf { it.size == 4 }?.map { it.number() ?: return@mapNotNull null } ?: return@mapNotNull null
                    val x = values[0].coerceIn(0.0, 1.0)
                    val y = values[1].coerceIn(0.0, 1.0)
                    val w = minOf(values[2], 1.0 - x)
                    val h = minOf(values[3], 1.0 - y)
                    if (w > 0 && h > 0) Box(x, y, w, h) else null
                }
                KeepSample(t, boxes)
            }.sortedBy { it.t }
            val times = ((body["plate"] as? JsonObject)?.get("times") as? JsonArray).orEmpty()
                .mapNotNull { it.number()?.takeIf { time -> time in 0.0..duration } }.distinct().sorted().take(MAX_PLATE_TIMES)
            if (samples.isEmpty() || times.size < MIN_PLATE_TIMES) return null
            return EditKeep((keep.number("feather") ?: 0.0).coerceIn(0.0, 0.2), samples, times)
        }

        private fun JsonElement.number(): Double? = (this as? JsonPrimitive)?.takeIf { !it.isString }?.doubleOrNull?.takeIf { it.isFinite() }
        private fun JsonObject.number(name: String): Double? = this[name]?.number()
    }
}
