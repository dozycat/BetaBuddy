package com.betabuddy.android.edit

// How a plan becomes an output: which source ranges play, how fast, and what the picture shows at
// each source moment. Pure arithmetic on the plan's numbers; no Android classes.

/** The three switches on the edit page. Each one only matters when the plan has the data for it. */
data class EditOptions(val removePassersBy: Boolean = true, val slowMotion: Boolean = true, val cameraMoves: Boolean = true) {
    companion object {
        /**
         * Passer-by removal is built and tested but not offered yet: it has not been tried on real
         * footage with passers-by. The edit page shows no switch for it and starts with it off.
         */
        const val REMOVAL_OFFERED = false

        /** What the edit page starts from. */
        val offered: EditOptions get() = EditOptions(removePassersBy = REMOVAL_OFFERED)
    }
}

/**
 * One stretch of the output: source [t0]…[t1] played at [speed], seen through [camera]. With
 * [cameraLinear] the window moves in straight lines between the keys (a plan whose camera follows
 * the subject); otherwise it eases from one key to the next.
 */
data class EditClip(val t0: Double, val t1: Double, val speed: Double, val camera: List<CameraKey>, val cameraLinear: Boolean = false) {
    val outputDuration: Double get() = (t1 - t0) / speed
}

/** Where the picture is looking: centre and zoom, already inside the frame. */
data class CameraWindow(val cx: Double, val cy: Double, val s: Double) {
    companion object { val FULL = CameraWindow(0.5, 0.5, 1.0) }
}

object EditTimeline {
    /**
     * The clips of the output for [plan] with these switches, for a clip that really lasts
     * [sourceDuration] seconds. Slow motion off is the attempt alone at normal speed; camera moves
     * off is the full frame throughout. Ranges are cut to the clip's real length.
     */
    fun clips(plan: EditPlan, options: EditOptions, sourceDuration: Double): List<EditClip> {
        val end = if (sourceDuration.isFinite() && sourceDuration > 0) minOf(plan.duration, sourceDuration) else plan.duration
        fun camera(own: List<CameraKey>?) = if (options.cameraMoves) own ?: plan.camera else emptyList()
        val segments = if (options.slowMotion) plan.segments else listOf(EditSegment(plan.trim.first, plan.trim.second, 1.0))
        val clips = segments.mapNotNull { segment ->
            val t1 = minOf(segment.t1, end)
            if (t1 - segment.t0 < EditPlan.MIN_RANGE) null else EditClip(segment.t0, t1, segment.speed, camera(segment.camera), plan.cameraTrack)
        }
        // A plan made for a longer clip than this one may leave nothing; the whole clip is then shown as it is.
        return clips.ifEmpty { listOf(EditClip(0.0, end, 1.0, camera(null), plan.cameraTrack)) }
    }

    fun outputDuration(clips: List<EditClip>): Double = clips.sumOf { it.outputDuration }

    /** When clip [index] starts in the output. */
    fun outputStart(clips: List<EditClip>, index: Int): Double = clips.take(index).sumOf { it.outputDuration }

    /** The clip showing at [outputTime] and the source moment it shows. Times outside the output are held at its ends. */
    fun sourceTime(clips: List<EditClip>, outputTime: Double): Pair<Int, Double> {
        var start = 0.0
        for ((index, clip) in clips.withIndex()) {
            val end = start + clip.outputDuration
            if (outputTime < end || index == clips.lastIndex) return index to sourceTime(clip, outputTime - start)
            start = end
        }
        return 0 to 0.0
    }

    /** The source moment [elapsed] seconds of output into [clip]. */
    fun sourceTime(clip: EditClip, elapsed: Double): Double = (clip.t0 + elapsed * clip.speed).coerceIn(clip.t0, clip.t1)

    /** The output moment at which [clip], starting at [outputStart], shows source time [t]. */
    fun outputTime(clip: EditClip, outputStart: Double, t: Double): Double = outputStart + (t.coerceIn(clip.t0, clip.t1) - clip.t0) / clip.speed
}

object CameraPath {
    /**
     * The window at source time [t]: between two keyframes the centre and zoom ease in and out
     * (smoothstep), or with [linear] move at an even pace (keys that already are a smooth track);
     * before the first and after the last they hold, and the window never leaves the frame. No
     * keyframes is the full frame.
     */
    fun window(keys: List<CameraKey>, t: Double, linear: Boolean = false): CameraWindow {
        if (keys.isEmpty()) return CameraWindow.FULL
        val next = keys.indexOfFirst { it.t > t }
        val raw = when (next) {
            0 -> keys.first()
            -1 -> keys.last()
            else -> {
                val a = keys[next - 1]
                val b = keys[next]
                val u = ((t - a.t) / (b.t - a.t)).coerceIn(0.0, 1.0)
                val w = if (linear) u else u * u * (3 - 2 * u)
                CameraKey(t, a.cx + (b.cx - a.cx) * w, a.cy + (b.cy - a.cy) * w, a.s + (b.s - a.s) * w)
            }
        }
        return clamp(raw.cx, raw.cy, raw.s)
    }

    fun clamp(cx: Double, cy: Double, s: Double): CameraWindow {
        val zoom = s.coerceIn(1.0, EditPlan.MAX_ZOOM)
        val half = 0.5 / zoom
        return CameraWindow(cx.coerceIn(half, 1 - half), cy.coerceIn(half, 1 - half), zoom)
    }
}

object KeepPath {
    /**
     * The boxes to keep live at source time [t]: each box moves in a straight line between the two
     * neighbouring samples when they have the same number of boxes; otherwise the nearer sample is
     * used as it is. Outside the samples the first or last one holds.
     */
    fun boxes(samples: List<KeepSample>, t: Double): List<Box> {
        if (samples.isEmpty()) return emptyList()
        val next = samples.indexOfFirst { it.t > t }
        if (next == 0) return samples.first().boxes
        if (next == -1) return samples.last().boxes
        val a = samples[next - 1]
        val b = samples[next]
        val u = (t - a.t) / (b.t - a.t)
        if (a.boxes.size != b.boxes.size) return if (u < 0.5) a.boxes else b.boxes
        return a.boxes.indices.map { index -> lerp(a.boxes[index], b.boxes[index], u) }
    }

    internal fun lerp(a: Box, b: Box, u: Double): Box =
        Box(a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u, a.w + (b.w - a.w) * u, a.h + (b.h - a.h) * u)
}

/** The size the edit is rendered at: the upright picture, long side at most 1920, both sides even. */
object RenderSize {
    const val MAX_LONG_SIDE = 1920

    fun of(displayWidth: Int, displayHeight: Int, maxLongSide: Int = MAX_LONG_SIDE): Pair<Int, Int> {
        val long = maxOf(displayWidth, displayHeight).coerceAtLeast(1)
        val scale = minOf(1.0, maxLongSide.toDouble() / long)
        fun even(value: Double) = maxOf(2, (Math.round(value / 2) * 2).toInt())
        return even(displayWidth * scale) to even(displayHeight * scale)
    }
}
