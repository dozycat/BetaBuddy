package com.betabuddy.android.review

import com.betabuddy.android.data.PoseFrame
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.add
import kotlinx.serialization.json.addJsonArray
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
    data class Result(val id: String, val json: String, val climber: String?) : AnalysisMessage

    /** The user asked to see this moment of the clip. Seconds. */
    data class Seek(val time: Double) : AnalysisMessage

    /** The page could not analyse the clip and is already showing why. */
    data class Error(val code: String) : AnalysisMessage

    companion object {
        // A result carries a picture and runs to a couple of hundred kilobytes; anything far beyond that is not one.
        private const val MAX_LENGTH = 8 * 1024 * 1024
        private const val MAX_CLIMBER_LENGTH = 2_000

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
                    Result(string("id").orEmpty(), result, climber)
                }
                "analysisSeek" -> {
                    val time = (body["t"] as? JsonPrimitive)?.takeIf { !it.isString }?.doubleOrNull
                    if (time != null && time.isFinite()) Seek(maxOf(0.0, time)) else null
                }
                "analysisError" -> Error(string("code").orEmpty().take(60))
                else -> null
            }
        }
    }
}

/** App → page calls: `window.BetaBuddyAnalysis.begin / addFrame / addPoses / run / show`. */
object AnalysisScript {
    /** [id] is the record's id, [width] × [height] the clip as displayed, [climber] the JSON text an earlier result came with. */
    fun begin(id: String, width: Int, height: Int, climber: String?): String {
        val measurements = climber?.let { text -> try { Json.parseToJsonElement(text) as? JsonObject } catch (error: Exception) { null } }
        val meta = buildJsonObject {
            put("id", id)
            put("w", width)
            put("h", height)
            if (measurements != null) put("climber", measurements)
        }
        return call("begin", literal(meta.toString()))
    }

    fun addFrame(index: Int, time: Double, jpegBase64: String): String =
        call("addFrame", "$index,${JsonPrimitive(finite(time))},${string(jpegBase64)}")

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
     * Size of a picture for the page: the shorter side is [FRAME_WIDTH] (never upscaled), same shape
     * as the displayed clip, so a landscape clip keeps as much detail as a portrait one.
     */
    fun frameSize(displayWidth: Int, displayHeight: Int): Pair<Int, Int> {
        val shorter = minOf(displayWidth, displayHeight).coerceAtLeast(1)
        val scale = minOf(1.0, FRAME_WIDTH.toDouble() / shorter)
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
