package com.betabuddy.android.pose

import com.betabuddy.android.data.Moment
import com.betabuddy.android.data.PoseFrame
import kotlin.math.PI
import kotlin.math.acos
import kotlin.math.hypot

/**
 * Pure geometry, independent of inference and UI. Coordinates are top-left normalized.
 * A line-for-line port of ios/BetaBuddy/Core/PoseMath.swift: the same input gives the same output.
 */
object PoseMath {
    val connections: List<Pair<Int, Int>> = listOf(
        11 to 12, 11 to 13, 13 to 15, 12 to 14, 14 to 16, 11 to 23, 12 to 24, 23 to 24, 23 to 25,
        25 to 27, 24 to 26, 26 to 28, 27 to 29, 29 to 31, 27 to 31, 28 to 30, 30 to 32, 28 to 32,
    )

    /** 2D projected angle at [b] in degrees, corrected for the picture's aspect ratio (width / height). */
    fun angle(frame: PoseFrame, a: Int, b: Int, c: Int, aspect: Double): Double? {
        if (!(aspect > 0) || !aspect.isFinite()) return null
        for (index in intArrayOf(a, b, c)) {
            if (index !in frame.joints.indices || !frame.joints[index].valid) return null
        }
        val p = frame.joints[a]
        val q = frame.joints[b]
        val r = frame.joints[c]
        val ux = (p.x - q.x) * aspect
        val uy = p.y - q.y
        val vx = (r.x - q.x) * aspect
        val vy = r.y - q.y
        val length = hypot(ux, uy) * hypot(vx, vy)
        if (!(length > 0.000001)) return null
        return acos(((ux * vx + uy * vy) / length).coerceIn(-1.0, 1.0)) * 180 / PI
    }

    // Stored with the record in the source language, exactly as iOS stores them; the UI shows the
    // current language by `kind` and only falls back to these for kinds it does not know.
    private const val RAISED_TITLE = "高抬脚 · 候选片段"
    private const val RAISED_DETAIL = "画面中脚跟高于髋部。回看是否为跟勾或高踩；模型无法确认岩点接触。"
    private const val FAST_TITLE = "快速移动 · 回看一下"
    private const val FAST_DETAIL = "髋部在画面中的移动较快。可能是动态动作、下落或镜头移动，请结合原片判断。"

    fun moments(frames: List<PoseFrame>, aspect: Double): List<Moment> {
        val result = mutableListOf<Moment>()
        var lastRaised = -10.0
        var lastFast = -10.0
        for ((index, frame) in frames.withIndex()) {
            if (frame.joints.size != 33) continue
            val hip = frame.hip ?: continue
            // A high heel alone cannot establish contact with a hold: explicitly a candidate.
            val raised = (frame.joints[29].valid && frame.joints[29].y < hip.y - 0.04) ||
                (frame.joints[30].valid && frame.joints[30].y < hip.y - 0.04)
            if (raised && frame.time - lastRaised >= 3) {
                result += Moment(time = frame.time, title = RAISED_TITLE, detail = RAISED_DETAIL, kind = Moment.RAISED_HEEL)
                lastRaised = frame.time
            }
            // No velocity across tracking gaps: only consecutive samples at most 0.3 s apart.
            val previous = if (index > 0) frames[index - 1].hip else null
            if (previous != null) {
                val dt = frame.time - frames[index - 1].time
                if (dt > 0 && dt <= 0.3) {
                    val speed = hypot((hip.x - previous.x) * aspect, hip.y - previous.y) / dt
                    if (speed > 0.45 && frame.time - lastFast >= 3) {
                        result += Moment(time = frame.time, title = FAST_TITLE, detail = FAST_DETAIL, kind = Moment.FAST_MOVE)
                        lastFast = frame.time
                    }
                }
            }
        }
        return result.sortedBy { it.time }.take(24)
    }
}
