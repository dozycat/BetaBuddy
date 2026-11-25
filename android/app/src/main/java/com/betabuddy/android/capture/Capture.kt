package com.betabuddy.android.capture

import com.betabuddy.android.data.CaptureRoute
import com.betabuddy.android.data.CaptureSummary
import com.betabuddy.android.pose.ImportRules
import kotlin.math.asin
import kotlin.math.atan
import kotlin.math.atan2
import kotlin.math.roundToInt
import kotlin.math.sqrt
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.add
import kotlinx.serialization.json.addJsonArray
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.put

// The capture page of a gym pack: the app shows the camera and draws everything, a page without
// any interface of its own looks at preview frames and answers with what it thinks of the camera
// position and what it recognises. What it computes is the pack's business; this file is only the
// shape of the messages, which the app treats leniently: unknown types, codes and extra fields are
// ignored, missing ones fall back to the cautious reading. Counterpart of ios/BetaBuddy/Core/Capture.swift.
// No Android classes here, so the unit tests run on a JVM.

/** What a hold is to the problem being recorded. */
enum class CaptureRole(val wire: String) {
    START_HAND("start_hand"),
    START_FOOT("start_foot"),
    FINISH("finish"),
    HOLD("hold"),
    /** Only ever sent by the app: "this one is not part of the problem". */
    NONE("none");

    companion object {
        fun of(wire: String?): CaptureRole? = entries.firstOrNull { it.wire == wire }
    }
}

/** A position in a preview frame: `x` to the right, `y` down, both 0–1. */
data class CapturePoint(val x: Double, val y: Double)

data class CaptureBox(val x: Double, val y: Double, val width: Double, val height: Double) {
    val midX: Double get() = x + width / 2
}

/** One reply of the capture page to a preview frame. */
data class CaptureAssessment(
    val time: Double,
    val ok: Boolean,
    /** 0–1, only for an indicator. Null when the page gave none. */
    val score: Double?,
    val steady: Boolean,
    val issues: List<Issue>,
    /** The one thing to tell the user now. */
    val hint: Issue?,
    /** `"none"`, `"partial"`, `"full"` or `"unknown"`. */
    val person: String,
    val wall: CaptureBox?,
    val palette: List<Swatch>,
    val color: String?,
    val holds: List<Hold>,
    val startHands: List<String>,
    val startFeet: List<String>,
    val finish: String?,
    /** A problem of the pack's own that the page thinks this is. A guess like everything else here. */
    val route: CaptureRoute?,
) {
    data class Issue(
        val code: String,
        /** True for `severity: "block"`: analysis will most likely not work from here. */
        val blocking: Boolean,
        val move: String?,
    )

    data class Swatch(val key: String, val rgb: List<Int>, val count: Int)

    data class Hold(
        val id: String,
        val box: CaptureBox,
        /** Outline to draw. Empty when the page gave none; the box is drawn instead. */
        val poly: List<CapturePoint>,
        val role: CaptureRole,
        val confidence: Double?,
        /** The role was set by the user with a tap. */
        val user: Boolean,
    )

    /**
     * The user has said where the problem starts, and where it ends. Until then both are the
     * page's guesses, and the start in particular is wrong more often than right.
     */
    val startConfirmed: Boolean get() = holds.any { it.role == CaptureRole.START_HAND && it.user }
    val finishConfirmed: Boolean get() = holds.any { it.role == CaptureRole.FINISH && it.user }

    /** The camera position can be used and the picture has stopped moving. */
    val ready: Boolean get() = ok && steady

    companion object {
        const val MAX_HOLDS = 300
        const val MAX_POLYGON_POINTS = 96

        fun from(body: JsonObject): CaptureAssessment {
            val issues = body.array("issues").take(32).mapNotNull(::issue)
            // A page that says nothing about `ok` is read from its issues; one that says nothing about
            // `steady` has not said the picture is still.
            val ok = body.boolean("ok") ?: issues.none { it.blocking }
            // The hint carries no severity of its own; it is the first issue.
            val hint = issue(body["hint"])?.let { hint -> hint.copy(blocking = issues.firstOrNull { it.code == hint.code }?.blocking ?: !ok) }
                ?: issues.firstOrNull()
            val wall = (body["wall"] as? JsonObject)?.let { wall ->
                val top = wall.number("top"); val bottom = wall.number("bottom"); val left = wall.number("left"); val right = wall.number("right")
                if (top != null && bottom != null && left != null && right != null && bottom > top && right > left) CaptureBox(left, top, right - left, bottom - top) else null
            }
            val palette = body.array("palette").take(16).mapNotNull { item ->
                val entry = item as? JsonObject ?: return@mapNotNull null
                val key = entry.string("key")?.takeIf { it.isNotEmpty() && it.length <= 40 } ?: return@mapNotNull null
                // The contract names the count `count`; a draft of it said `n`.
                Swatch(key, rgb(entry["rgb"]) ?: listOf(128, 128, 128), (entry.number("count") ?: entry.number("n") ?: 0.0).toInt().coerceAtLeast(0))
            }
            val holds = body.array("holds").take(MAX_HOLDS).mapNotNull { item ->
                val entry = item as? JsonObject ?: return@mapNotNull null
                val id = entry.string("id")?.takeIf { it.isNotEmpty() && it.length <= 40 } ?: return@mapNotNull null
                val poly = entry.array("poly").take(MAX_POLYGON_POINTS).mapNotNull { point ->
                    val pair = point as? JsonArray ?: return@mapNotNull null
                    val x = pair.getOrNull(0).number(); val y = pair.getOrNull(1).number()
                    if (x != null && y != null) CapturePoint(x, y) else null
                }
                val numbers = entry.array("box").mapNotNull { it.number() }
                val box = when {
                    numbers.size == 4 && numbers[2] >= 0 && numbers[3] >= 0 -> CaptureBox(numbers[0], numbers[1], numbers[2], numbers[3])
                    poly.size >= 3 -> CaptureBox(poly.minOf { it.x }, poly.minOf { it.y }, poly.maxOf { it.x } - poly.minOf { it.x }, poly.maxOf { it.y } - poly.minOf { it.y })
                    else -> return@mapNotNull null   // Nothing to draw.
                }
                // A role this build has never heard of is still a hold of the problem.
                val role = CaptureRole.of(entry.string("role"))?.takeIf { it != CaptureRole.NONE } ?: CaptureRole.HOLD
                Hold(id, box, if (poly.size >= 3) poly else emptyList(), role, entry.number("conf"), entry.boolean("user") ?: false)
            }
            val start = body["start"] as? JsonObject
            return CaptureAssessment(
                time = body.number("t") ?: 0.0,
                ok = ok,
                score = body.number("score")?.coerceIn(0.0, 1.0),
                steady = body.boolean("steady") ?: false,
                issues = issues,
                hint = hint,
                person = (body.string("person") ?: "unknown").take(16),
                wall = wall,
                palette = palette,
                color = body.string("color")?.takeIf { it.isNotEmpty() }?.take(40),
                holds = holds,
                startHands = ids(start?.get("hands")),
                startFeet = ids(start?.get("feet")),
                finish = (body["finish"] as? JsonObject)?.string("id")?.takeIf { it.isNotEmpty() }?.take(40),
                route = route(body["route"]),
            )
        }

        private fun ids(value: JsonElement?): List<String> =
            (value as? JsonArray).orEmpty().take(8).mapNotNull { it.string()?.takeIf(String::isNotEmpty)?.take(40) }

        internal fun issue(value: JsonElement?): Issue? {
            val item = value as? JsonObject ?: return null
            val code = item.string("code")?.takeIf { it.isNotEmpty() } ?: return null
            // Anything that is not plainly a warning is taken as blocking: the user is then asked before recording.
            return Issue(code.take(40), (item.string("severity") ?: "block") != "warn", item.string("move")?.takeIf { it.isNotEmpty() }?.take(40))
        }

        internal fun rgb(value: JsonElement?): List<Int>? =
            (value as? JsonArray).orEmpty().mapNotNull { it.number()?.roundToInt()?.coerceIn(0, 255) }.takeIf { it.size == 3 }

        /** Null for `null`, for anything without an id, and for an object this build cannot read. */
        internal fun route(value: JsonElement?): CaptureRoute? {
            val body = value as? JsonObject ?: return null
            val id = body.string("id")?.takeIf { it.isNotEmpty() } ?: return null
            fun text(name: String) = body.string(name)?.takeIf { it.isNotEmpty() }?.take(120)
            return CaptureRoute(
                id = id.take(120), confidence = body.number("conf"), name = text("name"), nameZh = text("nameZh"),
                grade = text("grade"), colour = text("colour") ?: text("color"), wall = text("wall"),
            )
        }

        internal fun summary(body: JsonObject): CaptureSummary {
            val start = body["start"] as? JsonObject
            return CaptureSummary(
                color = body.string("color")?.takeIf { it.isNotEmpty() }?.take(40),
                rgb = rgb(body["rgb"]),
                holds = (body.number("holds") ?: 0.0).toInt().coerceAtLeast(0),
                startHands = (start?.number("hands") ?: 0.0).toInt().coerceAtLeast(0),
                startFeet = (start?.number("feet") ?: 0.0).toInt().coerceAtLeast(0),
                finish = body.boolean("finish") ?: false,
                ok = body.boolean("ok") ?: false,
                issues = body.array("issues").take(32).mapNotNull { it.string()?.take(40) },
                route = route(body["route"]),
            )
        }
    }
}

/** Capture page → app messages. They arrive on the same `gym` handler, from the capture page's own web view. */
sealed interface CaptureMessage {
    data class Ready(val packVersion: String) : CaptureMessage
    data class Assessment(val assessment: CaptureAssessment) : CaptureMessage
    /** [plan] is null when the page gave none that can be kept. */
    data class Locked(val id: String, val plan: String?, val summary: CaptureSummary) : CaptureMessage
    /** The page cannot assess. Recording goes on without it. */
    data class Failed(val code: String) : CaptureMessage
    /** A script error. Before `ready` it means the page never came up. */
    data class PageError(val message: String) : CaptureMessage

    companion object {
        const val MAX_PLAN_LENGTH = 4_000_000
        // An assessment with a few hundred outlined holds runs to tens of kilobytes; a plan to a few hundred.
        private const val MAX_LENGTH = 6 * 1024 * 1024

        /** Parses the JSON text of one message. Unknown types and malformed input give null; extra fields are ignored. */
        fun parse(json: String?): CaptureMessage? {
            if (json == null || json.length > MAX_LENGTH) return null
            val body = try { Json.parseToJsonElement(json) as? JsonObject } catch (error: Exception) { null } ?: return null
            return when (body.string("type")) {
                "captureReady" -> Ready((body.string("packVersion") ?: "").take(40))
                "captureAssessment" -> Assessment(CaptureAssessment.from(body))
                "captureLocked" -> {
                    // The plan is a string; a page that hands over an object instead is still kept.
                    val plan = when (val value = body["plan"]) {
                        is JsonPrimitive -> value.takeIf { it.isString }?.content
                        is JsonObject, is JsonArray -> value.toString()
                        else -> null
                    }?.takeIf { it.isNotEmpty() && it.toByteArray(Charsets.UTF_8).size <= MAX_PLAN_LENGTH }
                    Locked((body.string("id") ?: "").take(80), plan, CaptureAssessment.summary(body["summary"] as? JsonObject ?: JsonObject(emptyMap())))
                }
                "captureError" -> Failed((body.string("code") ?: "").take(80))
                "error" -> PageError((body.string("message") ?: "").take(300))
                else -> null
            }
        }
    }
}

/** What the app knows about a preview frame besides the picture. Every part is optional. */
data class CaptureFrameInfo(
    /** Elevation of the camera's optical axis above the horizon, degrees, up positive. */
    val pitch: Double? = null,
    /** Rotation of the phone about the optical axis, degrees, clockwise positive as the user looks at the screen; upright portrait is 0. */
    val roll: Double? = null,
    /** Root mean square of the angular velocity since the previous frame, rad/s. */
    val shake: Double? = null,
    /** 33 × `[x, y, visibility]` when a person was detected in this frame. */
    val pose: List<List<Double>>? = null,
) {
    val isEmpty: Boolean get() = pitch == null && roll == null && shake == null && pose == null
}

/** App → capture page calls on `window.BetaBuddyCapture`. */
object CaptureScript {
    fun begin(id: String, width: Int, height: Int, fieldOfView: Double?, venue: String?): String {
        val meta = buildJsonObject {
            put("id", id)
            put("w", width)
            put("h", height)
            finite(fieldOfView)?.let { put("fovH", it) }
            if (!venue.isNullOrEmpty()) put("venue", venue)
        }
        return call("begin", literal(meta.toString()))
    }

    fun addFrame(time: Double, jpegBase64: String, info: CaptureFrameInfo): String {
        val arguments = StringBuilder().append(JsonPrimitive(finite(time, 1000.0) ?: 0.0)).append(',').append(string(jpegBase64))
        // The third argument is left out entirely when there is nothing to say.
        if (!info.isEmpty) {
            val body = buildJsonObject {
                finite(info.pitch)?.let { put("pitch", it) }
                finite(info.roll)?.let { put("roll", it) }
                finite(info.shake, 10_000.0)?.let { put("shake", it) }
                info.pose?.let { pose ->
                    put("pose", buildJsonArray { for (point in pose) addJsonArray { for (value in point) add(if (value.isFinite()) round(value, 10_000.0) else 0.0) } })
                }
            }
            arguments.append(',').append(literal(body.toString()))
        }
        return call("addFrame", arguments.toString())
    }

    fun setColor(key: String?): String = call("setColor", key?.let(::string) ?: "null")

    /** Answers with the id of the hold the page found there, or an empty string. */
    fun tap(x: Double, y: Double, role: CaptureRole): String =
        call("tap", "${JsonPrimitive(round(x.coerceIn(0.0, 1.0), 10_000.0))},${JsonPrimitive(round(y.coerceIn(0.0, 1.0), 10_000.0))},${string(role.wire)}")

    fun resetTaps(): String = call("resetTaps", "")

    fun lock(): String = call("lock", "")

    /** Only a newer page has it; an older one answers `false` and nothing happens. */
    fun rejectRoute(id: String): String = call("rejectRoute", string(id))

    private fun call(function: String, arguments: String): String {
        val invocation = "window.BetaBuddyCapture.$function($arguments)"
        // `tap` answers with the id of the hold it found, or nothing; the other calls only say that they were made.
        val made = if (function == "tap") "($invocation || '')" else "($invocation, true)"
        return "window.BetaBuddyCapture && typeof window.BetaBuddyCapture.$function === 'function' ? $made : false"
    }

    /** What `evaluateJavascript` hands back for [tap]: the hold's id, empty for "no hold here", null when the page has no `tap`. */
    fun tapAnswer(result: String?): String? {
        val value = try { Json.parseToJsonElement(result ?: return null) } catch (error: Exception) { return null }
        return (value as? JsonPrimitive)?.takeIf { it.isString }?.content
    }

    /** Whether a call other than [tap] was made. */
    fun made(result: String?): Boolean = result?.trim() == "true"

    /** A JavaScript string literal. JSON encoding does the escaping; nothing is quoted by hand. */
    private fun string(value: String) = literal(JsonPrimitive(value).toString())

    // JSON text is a JavaScript expression, except that older engines end a line at these two characters.
    private fun literal(json: String) = json.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")

    private fun round(value: Double, places: Double) = Math.round(value * places) / places
    private fun finite(value: Double?, places: Double = 100.0): Double? = value?.takeIf { it.isFinite() }?.let { round(it, places) }
}

/**
 * The codes and moves this build has a sentence for. A newer pack may send others: an unknown
 * code is shown as the generic "try another position", an unknown move is not shown at all.
 */
object CaptureGuidance {
    enum class Code(val wire: String) {
        NO_WALL("no_wall"), TOO_DARK("too_dark"), BACKLIT("backlit"), BLURRY("blurry"), SHAKING("shaking"), TILTED("tilted"),
        TOO_CLOSE("too_close"), TOO_FAR("too_far"),
        CUT_TOP("cut_top"), CUT_BOTTOM("cut_bottom"), CUT_LEFT("cut_left"), CUT_RIGHT("cut_right"),
        OBLIQUE("oblique"), PITCH_HIGH("pitch_high"), OCCLUDED("occluded"), NO_HOLDS("no_holds"),
        START_UNCLEAR("start_unclear"), FINISH_UNCLEAR("finish_unclear");

        companion object {
            fun of(wire: String?): Code? = entries.firstOrNull { it.wire == wire }
        }
    }

    enum class Move(val wire: String) {
        BACK("back"), CLOSER("closer"), LEFT("left"), RIGHT("right"), UP("up"), DOWN("down"),
        TILT_UP("tilt_up"), TILT_DOWN("tilt_down"), LEVEL("level"), HOLD_STILL("hold_still");

        companion object {
            fun of(wire: String?): Move? = entries.firstOrNull { it.wire == wire }
        }
    }
}

/**
 * Angles of the phone from a unit vector pointing down, in the device's own axes (x to the right
 * of the screen, y to its top, z out of the screen towards the user). The back camera looks along −z.
 * Android's gravity sensor reports the opposite vector, in m/s²; see [down].
 */
object CaptureMotion {
    /** The unit "down" vector from what `TYPE_GRAVITY` (or a resting accelerometer) reports. */
    fun down(x: Double, y: Double, z: Double): Triple<Double, Double, Double>? {
        val length = sqrt(x * x + y * y + z * z)
        if (!length.isFinite() || length < 1e-3) return null
        return Triple(-x / length, -y / length, -z / length)
    }

    /** Elevation of the optical axis above the horizon, degrees: 0 with the phone upright, positive when the camera looks up. */
    fun pitch(x: Double, y: Double, z: Double): Double? {
        if (!x.isFinite() || !y.isFinite() || !z.isFinite()) return null
        return Math.toDegrees(asin(z.coerceIn(-1.0, 1.0)))
    }

    /**
     * Rotation about the optical axis, degrees, clockwise positive as seen on the screen. With the
     * phone lying nearly flat there is no "upright" to measure from: the answer is 0, and the pitch
     * says what is wrong.
     */
    fun roll(x: Double, y: Double, z: Double): Double? {
        if (!x.isFinite() || !y.isFinite() || !z.isFinite()) return null
        if (sqrt(x * x + y * y) <= 0.3) return 0.0
        return Math.toDegrees(atan2(x, -y))
    }

    /**
     * Horizontal field of view, in degrees, of an upright picture [frameLong] × [frameShort] pixels
     * taken from a sensor area [sensorLong] × [sensorShort] millimetres behind a lens of [focal]
     * millimetres. A stream is the largest rectangle of its own shape that fits the sensor: a 16:9
     * stream from a 4:3 sensor keeps the sensor's full length and loses height, a squarer one keeps
     * the height. The upright picture's width is the stream's short side.
     */
    fun streamFieldOfView(sensorLong: Double, sensorShort: Double, focal: Double, frameLong: Int, frameShort: Int): Double? {
        if (!(sensorLong > 0) || !(sensorShort > 0) || !(focal > 0) || frameLong <= 0 || frameShort <= 0) return null
        val long = maxOf(sensorLong, sensorShort)
        val short = minOf(sensorLong, sensorShort)
        val shape = maxOf(frameLong, frameShort).toDouble() / minOf(frameLong, frameShort)
        val usedShort = if (shape >= long / short) long / shape else short
        return fieldOfView(usedShort, focal)
    }

    /** Field of view in degrees across a sensor dimension of [millimetres] at this focal length. */
    fun fieldOfView(millimetres: Double, focalLength: Double): Double? {
        if (!(millimetres > 0) || !(focalLength > 0)) return null
        return Math.toDegrees(2 * atan(millimetres / (2 * focalLength)))
    }
}

object CaptureLimits {
    /** Twenty minutes, as on iOS: what pose recognition takes of a camera record ([ImportRules.CAMERA_MAX_SECONDS]), less a little so the finished file stays under it. */
    const val MAX_SECONDS = ImportRules.CAMERA_MAX_SECONDS - 2
    /** That ceiling in whole minutes, for the sentence that says a recording stopped at it. */
    val MAX_MINUTES: Int get() = (ImportRules.CAMERA_MAX_SECONDS / 60).toInt()
    /** Shorter than this there is nothing to review. */
    const val MIN_SECONDS = 1.0
    /** Free space needed to start recording, and the level at which a running recording is closed. */
    const val STARTING_FREE_BYTES = 600L * 1024 * 1024
    const val STOPPING_FREE_BYTES = 250L * 1024 * 1024
    /** Shorter side of the pictures handed to the page. */
    const val SHORT_SIDE = 360

    /**
     * Size of the picture handed to the page for a source frame of this upright size: the shorter
     * side scaled down to [SHORT_SIDE] (never up), both sides even.
     */
    fun frameSize(width: Int, height: Int, shortSide: Int = SHORT_SIDE): Pair<Int, Int> {
        val scale = minOf(1.0, shortSide.toDouble() / minOf(width, height).coerceAtLeast(1))
        fun even(value: Int) = maxOf(2, Math.round(value * scale).toInt() and 1.inv())
        return even(width) to even(height)
    }
}

/** Why a recording ended. */
enum class CaptureStop { USER, LIMIT, STORAGE, INTERRUPTED, OVERHEATED }

private fun JsonElement?.string(): String? = (this as? JsonPrimitive)?.takeIf { it.isString }?.content
/** A finite JSON number. A JSON boolean, or a number written as a string, is not one. */
private fun JsonElement?.number(): Double? = (this as? JsonPrimitive)?.takeIf { !it.isString && this !is JsonNull }?.doubleOrNull?.takeIf { it.isFinite() }
private fun JsonObject.string(name: String): String? = this[name].string()
private fun JsonObject.number(name: String): Double? = this[name].number()
private fun JsonObject.boolean(name: String): Boolean? = (this[name] as? JsonPrimitive)?.takeIf { !it.isString }?.booleanOrNull
private fun JsonObject.array(name: String): List<JsonElement> = (this[name] as? JsonArray).orEmpty()
