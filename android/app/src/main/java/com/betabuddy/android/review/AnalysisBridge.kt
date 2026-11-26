package com.betabuddy.android.review

import com.betabuddy.android.data.CapturePlanShift
import com.betabuddy.android.data.CapturePlanUse
import com.betabuddy.android.data.Discipline
import com.betabuddy.android.data.PoseFrame
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.add
import kotlinx.serialization.json.addJsonArray
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray

// What crosses between the app and the pack's analysis page. The app is only the host: it hands
// the page frames and poses, keeps what the page hands back, and never looks inside the result.
// No Android classes here, so the unit tests run on a JVM.

/** Page → app messages of the analysis page, posted to the same `gym` handler as the gym page's. */
sealed interface AnalysisMessage {
    /** The page's script is up and `window.BetaBuddyAnalysis` exists. */
    data object Ready : AnalysisMessage

    /**
     * A finished result. [json] is the page's own format and is stored exactly as received.
     * [climber] is the JSON text of the body measurements the page used, to start the next analysis from.
     */
    data class Result(
        val id: String, val json: String, val climber: String?, val capture: CapturePlanUse? = null,
        /** The page's plan for an automatic edit, as the JSON text it sent, or null when it sent none. Stored as it is. */
        val edit: String? = null,
        /** What the page judged the climb to be (one of the `Discipline` values), or null when it could not tell. */
        val discipline: String? = null,
    ) : AnalysisMessage

    /** The user asked to see this moment of the clip. Seconds. */
    data class Seek(val time: Double) : AnalysisMessage

    /**
     * The page could not analyse the clip and is already showing why. [discipline] is what it still
     * made of the climbing style (one of the `Discipline` values), from packs that send it. [edit] is
     * a plan for an automatic edit it could still make, as the JSON text it sent, stored as it is.
     */
    data class Error(val code: String, val discipline: String? = null, val edit: String? = null) : AnalysisMessage

    companion object {
        // A result carries a picture and runs to a couple of hundred kilobytes; anything far beyond that is not one.
        private const val MAX_LENGTH = 8 * 1024 * 1024
        private const val MAX_CLIMBER_LENGTH = 2_000
        // A plan is a list of times and boxes: tens of kilobytes for a long clip.
        private const val MAX_EDIT_LENGTH = 4 * 1024 * 1024

        /**
         * Parses the JSON text of one message. Progress reports (the page draws its own), unknown
         * types and malformed input give null; extra fields are ignored.
         */
        fun parse(json: String?): AnalysisMessage? {
            if (json == null || json.length > MAX_LENGTH) return null
            val body = try { Json.parseToJsonElement(json) as? JsonObject } catch (error: Exception) { null } ?: return null
            fun string(name: String) = (body[name] as? JsonPrimitive)?.takeIf { it.isString }?.content
            return when (string("type")) {
                "analysisReady" -> Ready
                "analysisResult" -> {
                    val result = string("json")?.takeIf { it.isNotEmpty() } ?: return null
                    val climber = (body["climber"] as? JsonObject)?.toString()?.takeIf { it.length <= MAX_CLIMBER_LENGTH }
                    val capture = (body["capture"] as? JsonObject)?.let { use ->
                        val used = (use["used"] as? JsonPrimitive)?.takeIf { !it.isString }?.booleanOrNull ?: return@let null
                        fun text(name: String) = (use[name] as? JsonPrimitive)?.takeIf { it.isString }?.content?.takeIf { it.isNotEmpty() }?.take(40)
                        // What the page says it did with the capture plan it was given; only a newer page says so.
                        val shift = (use["shift"] as? JsonObject)?.let { moved ->
                            fun number(name: String) = (moved[name] as? JsonPrimitive)?.takeIf { !it.isString }?.doubleOrNull?.takeIf { it.isFinite() }
                            CapturePlanShift(number("dx"), number("dy"), number("rot"), number("scale")).takeIf { it != CapturePlanShift() }
                        }
                        CapturePlanUse(used, text("reason"), text("start"), text("finish"), shift)
                    }
                    val edit = (body["edit"] as? JsonObject)?.toString()?.takeIf { it.length <= MAX_EDIT_LENGTH }
                    val discipline = ((body["discipline"] as? JsonObject)?.get("kind") as? JsonPrimitive)?.takeIf { it.isString }?.content?.takeIf(Discipline::valid)
                    Result(string("id").orEmpty(), result, climber, capture, edit, discipline)
                }
                "analysisSeek" -> {
                    val time = (body["t"] as? JsonPrimitive)?.takeIf { !it.isString }?.doubleOrNull
                    if (time != null && time.isFinite()) Seek(maxOf(0.0, time)) else null
                }
                "analysisError" -> Error(
                    string("code").orEmpty().take(60),
                    ((body["discipline"] as? JsonObject)?.get("kind") as? JsonPrimitive)?.takeIf { it.isString }?.content?.takeIf(Discipline::valid),
                    (body["edit"] as? JsonObject)?.toString()?.takeIf { it.length <= MAX_EDIT_LENGTH },
                )
                else -> null
            }
        }
    }
}

/** App → page calls: `window.BetaBuddyAnalysis.begin / addFrame / addTrackFrame / addPoses / run / show`. */
object AnalysisScript {
    /**
     * [id] is the record's id, [width] × [height] the clip as displayed, [climber] the JSON text an
     * earlier result came with. [capture] is the capture page's plan for a record made with the
     * in-app camera, handed back as the text it was stored as. [discipline] is what the user chose
     * for this record, if they did; the page then does not judge it again. [duration] is how long
     * the clip really is, in seconds.
     */
    fun begin(id: String, width: Int, height: Int, climber: String?, capture: String? = null, discipline: String? = null, duration: Double? = null): String {
        val measurements = climber?.let { text -> try { Json.parseToJsonElement(text) as? JsonObject } catch (error: Exception) { null } }
        val meta = buildJsonObject {
            put("id", id)
            put("w", width)
            put("h", height)
            if (measurements != null) put("climber", measurements)
            if (!capture.isNullOrEmpty()) put("capture", capture)
            if (Discipline.valid(discipline)) put("discipline", discipline)
            if (duration != null && duration.isFinite() && duration > 0) put("duration", duration)
        }
        return call("begin", literal(meta.toString()))
    }

    fun addFrame(index: Int, time: Double, jpegBase64: String): String =
        call("addFrame", "$index,${JsonPrimitive(finite(time))},${string(jpegBase64)}")

    /**
     * One of the small, frequent pictures of the whole clip that newer packs take besides the
     * stills: seconds into the clip, and the JPEG. The call evaluates to `false` on a page that has
     * no such function (see [refused]); nothing more needs sending then.
     */
    fun addTrackFrame(time: Double, jpegBase64: String): String =
        call("addTrackFrame", "${JsonPrimitive(finite(time))},${string(jpegBase64)}")

    /** True when what a call evaluated to says the page has no such function. */
    fun refused(answer: String?): Boolean = answer?.trim() == "false"

    /** [batch] is the JSON text of an array from [AnalysisFeed.poseBatches]; the page parses it. */
    fun addPoses(batch: String): String = call("addPoses", string(batch))

    fun run(): String = call("run", "")

    /** [saved] is a stored result, handed back as the text it arrived as. */
    fun show(saved: String): String = call("show", string(saved))

    private fun call(function: String, arguments: String) =
        "window.BetaBuddyAnalysis && typeof window.BetaBuddyAnalysis.$function === 'function' ? (window.BetaBuddyAnalysis.$function($arguments), true) : false"

    /** A JavaScript string literal. JSON encoding does the escaping; nothing is quoted by hand. */
    private fun string(value: String) = literal(JsonPrimitive(value).toString())

    // JSON text is a JavaScript expression, except that older engines end a line at these two characters.
    private fun literal(json: String) = json.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
}

/** What the page is given for a fresh analysis, in the shapes its contract names. */
object AnalysisFeed {
    const val FRAME_COUNT = 60
    const val FRAME_WIDTH = 540
    const val POSE_BATCH = 40

    /** The small pictures of the whole clip (`addTrackFrame`): this many a second, this size, this JPEG quality, at most this many. */
    const val TRACK_RATE = 3.0
    const val TRACK_SHORT_SIDE = 270
    const val TRACK_JPEG_QUALITY = 50
    const val TRACK_MAX_FRAMES = 900

    /** The page needs at least this many pictures to work with. */
    const val MIN_FRAMES = 5

    /**
     * Times of the pictures taken from a clip of [duration] seconds: the middle of each of up to
     * [FRAME_COUNT] equal slices, and never closer together than a tenth of a second.
     */
    fun frameTimes(duration: Double): List<Double> {
        if (!duration.isFinite() || duration <= 0) return emptyList()
        val count = (duration * 10).toInt().coerceIn(1, FRAME_COUNT)
        return List(count) { index -> (index + 0.5) * duration / count }
    }

    /**
     * Times of the small pictures of the whole clip: evenly spread, about [TRACK_RATE] a second, and
     * fewer a second once that would be more than [TRACK_MAX_FRAMES].
     */
    fun trackTimes(duration: Double): List<Double> {
        if (!duration.isFinite() || duration <= 0) return emptyList()
        val count = Math.ceil(duration * TRACK_RATE).toInt().coerceIn(1, TRACK_MAX_FRAMES)
        return List(count) { index -> (index + 0.5) * duration / count }
    }

    /**
     * Size of a picture for the page: the shorter side is [shortSide] (never upscaled), same shape
     * as the displayed clip, so a landscape clip keeps as much detail as a portrait one.
     */
    fun frameSize(displayWidth: Int, displayHeight: Int, shortSide: Int = FRAME_WIDTH): Pair<Int, Int> {
        val shorter = minOf(displayWidth, displayHeight).coerceAtLeast(1)
        val scale = minOf(1.0, shortSide.toDouble() / shorter)
        return maxOf(1, Math.round(displayWidth * scale).toInt()) to maxOf(1, Math.round(displayHeight * scale).toInt())
    }

    /**
     * The stored poses as JSON arrays of `{t, img, lm}`, [POSE_BATCH] samples each: `img` is the 33
     * picture landmarks as `[x, y, visibility]` (the landmarker's own visibility), `lm` the 33 world landmarks. Samples saved without
     * world landmarks are left out.
     */
    fun poseBatches(frames: List<PoseFrame>, batchSize: Int = POSE_BATCH): List<String> =
        frames.filter { it.hasWorldLandmarks }.chunked(batchSize).map { batch ->
            buildJsonArray {
                for (frame in batch) add(buildJsonObject {
                    put("t", finite(frame.time))
                    putJsonArray("img") {
                        // Visibility as the landmarker reported it (the world landmark carries the same value), so both platforms hand the page the same numbers.
                        frame.joints.forEachIndexed { index, joint ->
                            val visibility = frame.world?.getOrNull(index)?.getOrNull(3) ?: joint.confidence
                            addJsonArray { add(finite(joint.x)); add(finite(joint.y)); add(finite(visibility)) }
                        }
                    }
                    putJsonArray("lm") {
                        for (point in frame.world.orEmpty()) addJsonArray { for (value in point) add(finite(value)) }
                    }
                })
            }.toString()
        }
}

private fun finite(value: Double): Double = if (value.isFinite()) value else 0.0
