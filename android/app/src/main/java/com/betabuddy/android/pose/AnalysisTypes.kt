package com.betabuddy.android.pose

import java.security.MessageDigest
import kotlin.math.floor
import kotlin.math.roundToInt

/** The bundled pose model: MediaPipe Pose Landmarker Full, float16, official version 1. Same file as iOS. */
object PoseModel {
    const val ASSET = "pose_landmarker_full.task"
    const val SHA256 = "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1"
    const val BYTES = 9_398_198L

    fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
}

/** Why an analysis or import could not finish. The UI turns each into a message the user can act on. */
enum class AnalysisError { MISSING_MODEL, INVALID_VIDEO, TOO_LONG, TOO_LARGE, NO_POSE, OVERHEATED }

class AnalysisException(val error: AnalysisError, cause: Throwable? = null) : Exception(error.name, cause)

/** Same Full weights in both; only sampling and preprocessing differ. Mirrors iOS `AnalysisQuality`. */
enum class AnalysisQuality(val fps: Double, val maxDimension: Int) {
    BALANCED(10.0, 720),
    DETAILED(15.0, 960),
}

/** Limits on an imported clip: 0.5–180 s, at most 500 MB, with a video track. */
object ImportRules {
    const val MAX_BYTES = 500L * 1024 * 1024
    const val MIN_SECONDS = 0.5
    const val MAX_SECONDS = 180.0

    /** Returns null when the clip is acceptable. Same order of checks as iOS. */
    fun check(sizeBytes: Long, durationSeconds: Double?, hasVideoTrack: Boolean): AnalysisError? = when {
        sizeBytes > MAX_BYTES -> AnalysisError.TOO_LARGE
        sizeBytes <= 0 || !hasVideoTrack -> AnalysisError.INVALID_VIDEO
        durationSeconds == null || !durationSeconds.isFinite() || durationSeconds < MIN_SECONDS -> AnalysisError.INVALID_VIDEO
        durationSeconds > MAX_SECONDS -> AnalysisError.TOO_LONG
        else -> null
    }
}

/**
 * Decides which decoded frames are analysed. Frames are dropped before any colour conversion,
 * scaling or inference; at most [fps] frames per second of video are kept. Same rule as iOS.
 */
class FrameSampler(private val fps: Double) {
    private var nextTime = 0.0
    private var lastTimestampMs = -1L

    fun shouldSample(time: Double): Boolean {
        if (!time.isFinite() || time + TOLERANCE < nextTime) return false
        nextTime = (floor((time + TOLERANCE) * fps) + 1) / fps
        return true
    }

    private companion object {
        // Decoder timestamps are whole microseconds, so a frame due at 2/15 s arrives as 0.133333.
        // Half a millisecond absorbs that; it is far below one frame interval of any real clip.
        const val TOLERANCE = 0.0005
    }

    /** Strictly increasing millisecond timestamps, as the landmarker's video mode requires. */
    fun timestampMs(time: Double): Long {
        val timestamp = maxOf(lastTimestampMs + 1, (time * 1000).toLong())
        lastTimestampMs = timestamp
        return timestamp
    }
}

/** Size and orientation of the picture handed to the model, so it matches what playback shows. */
object FrameGeometry {
    fun normalizeRotation(degrees: Int): Int = (((degrees % 360) + 360) % 360).let { if (it % 90 == 0) it else 0 }

    /** Display size of a coded picture after its rotation metadata is applied. */
    fun displaySize(width: Int, height: Int, rotationDegrees: Int): Pair<Int, Int> =
        if (normalizeRotation(rotationDegrees) % 180 == 90) height to width else width to height

    /** Never upscales; the long edge is capped at [maxDimension]. */
    fun targetSize(displayWidth: Int, displayHeight: Int, maxDimension: Int): Pair<Int, Int> {
        val scale = minOf(1.0, maxDimension.toDouble() / maxOf(displayWidth, displayHeight))
        return maxOf(1, (displayWidth * scale).roundToInt()) to maxOf(1, (displayHeight * scale).roundToInt())
    }
}

/**
 * Converts one decoded YUV 4:2:0 picture straight into a rotated, downscaled ARGB picture in a
 * single pass, so no full-size RGB copy is ever made. Planes are given as byte arrays with their
 * row and pixel strides (as `android.media.Image` reports them). Pure Kotlin: testable on a JVM.
 */
object YuvConverter {
    class Plane(val data: ByteArray, val rowStride: Int, val pixelStride: Int)

    /**
     * @param cropLeft,cropTop,cropWidth,cropHeight the visible rectangle inside the coded picture
     * @param rotationDegrees clockwise rotation needed to display the picture upright (0/90/180/270)
     * @param outWidth,outHeight size of [out] in display orientation
     * @param bt709 BT.709 matrix when true, BT.601 otherwise
     * @param fullRange full-range (0–255) samples when true, video range (16–235) otherwise
     */
    fun convert(
        y: Plane, u: Plane, v: Plane,
        cropLeft: Int, cropTop: Int, cropWidth: Int, cropHeight: Int,
        rotationDegrees: Int, outWidth: Int, outHeight: Int,
        bt709: Boolean, fullRange: Boolean, out: IntArray,
    ) {
        val rotation = FrameGeometry.normalizeRotation(rotationDegrees)
        val swapped = rotation % 180 == 90
        // Size of the output expressed in source orientation.
        val unrotatedWidth = if (swapped) outHeight else outWidth
        val unrotatedHeight = if (swapped) outWidth else outHeight
        val stepX = cropWidth.toDouble() / unrotatedWidth
        val stepY = cropHeight.toDouble() / unrotatedHeight
        // Box-average luma when shrinking by 2× or more, to limit aliasing of thin limbs.
        val box = minOf(4, maxOf(1, minOf(stepX, stepY).toInt()))
        val kr = if (bt709) 0.2126 else 0.299
        val kb = if (bt709) 0.0722 else 0.114
        val kg = 1 - kr - kb
        val yScale = if (fullRange) 1.0 else 255.0 / 219.0
        val cScale = if (fullRange) 1.0 else 255.0 / 224.0
        val yOffset = if (fullRange) 0 else 16
        val rv = 2 * (1 - kr) * cScale
        val bu = 2 * (1 - kb) * cScale
        val gu = -2 * (1 - kb) * kb / kg * cScale
        val gv = -2 * (1 - kr) * kr / kg * cScale
        for (oy in 0 until outHeight) {
            for (ox in 0 until outWidth) {
                // Position in the unrotated (source-oriented) output grid.
                val ux: Int
                val uy: Int
                when (rotation) {
                    90 -> { ux = oy; uy = unrotatedHeight - 1 - ox }
                    180 -> { ux = unrotatedWidth - 1 - ox; uy = unrotatedHeight - 1 - oy }
                    270 -> { ux = unrotatedWidth - 1 - oy; uy = ox }
                    else -> { ux = ox; uy = oy }
                }
                val sx = cropLeft + minOf(cropWidth - 1, ((ux + 0.5) * stepX).toInt())
                val sy = cropTop + minOf(cropHeight - 1, ((uy + 0.5) * stepY).toInt())
                var luma: Int
                if (box == 1) {
                    luma = y.data[sy * y.rowStride + sx * y.pixelStride].toInt() and 0xFF
                } else {
                    var sum = 0
                    var count = 0
                    val x0 = maxOf(cropLeft, sx - box / 2)
                    val y0 = maxOf(cropTop, sy - box / 2)
                    val x1 = minOf(cropLeft + cropWidth, x0 + box)
                    val y1 = minOf(cropTop + cropHeight, y0 + box)
                    for (yy in y0 until y1) {
                        val row = yy * y.rowStride
                        for (xx in x0 until x1) { sum += y.data[row + xx * y.pixelStride].toInt() and 0xFF; count++ }
                    }
                    luma = sum / count
                }
                val cx = sx / 2
                val cy = sy / 2
                val cb = (u.data[cy * u.rowStride + cx * u.pixelStride].toInt() and 0xFF) - 128
                val cr = (v.data[cy * v.rowStride + cx * v.pixelStride].toInt() and 0xFF) - 128
                val l = (luma - yOffset) * yScale
                val r = clamp(l + rv * cr)
                val g = clamp(l + gu * cb + gv * cr)
                val b = clamp(l + bu * cb)
                out[oy * outWidth + ox] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
    }

    private fun clamp(value: Double): Int = if (value <= 0) 0 else if (value >= 255) 255 else (value + 0.5).toInt()
}
