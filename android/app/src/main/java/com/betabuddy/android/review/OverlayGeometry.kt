package com.betabuddy.android.review

import com.betabuddy.android.data.Joint
import com.betabuddy.android.data.PoseFrame
import com.betabuddy.android.pose.PoseMath

/** Where things go on the replay, with no Android dependency so the rules can be unit-tested. */
object OverlayGeometry {
    /**
     * Size of the largest box with the picture's [aspect] (width / height) that fits the container
     * (aspect-fit, i.e. letterboxed or pillarboxed). The video surface and the skeleton canvas are
     * both given exactly this box, so normalized landmarks map onto the picture without offsets.
     */
    fun fit(containerWidth: Float, containerHeight: Float, aspect: Double): Pair<Float, Float> {
        if (!(containerWidth > 0) || !(containerHeight > 0) || !(aspect > 0) || !aspect.isFinite()) return 0f to 0f
        val width = minOf(containerWidth.toDouble(), containerHeight * aspect)
        return width.toFloat() to (width / aspect).toFloat()
    }

    /** Bones whose two ends are both reliable and inside the picture. Others are not drawn. */
    fun bones(frame: PoseFrame): List<Pair<Joint, Joint>> {
        if (frame.joints.size != 33) return emptyList()
        return PoseMath.connections.mapNotNull { (a, b) ->
            val start = frame.joints[a]
            val end = frame.joints[b]
            if (start.valid && end.valid) start to end else null
        }
    }

    /** Body points (shoulders down; the face points are left out) that are reliable. */
    fun points(frame: PoseFrame): List<Joint> = frame.joints.drop(11).filter { it.valid }
}
