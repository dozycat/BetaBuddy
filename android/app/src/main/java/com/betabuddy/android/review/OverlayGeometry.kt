package com.betabuddy.android.review

/** Where things go on the replay, with no Android dependency so the rules can be unit-tested. */
object OverlayGeometry {
    /**
     * Size of the largest box with the picture's [aspect] (width / height) that fits the container
     * (aspect-fit, i.e. letterboxed or pillarboxed). The video surface is given exactly this box.
     */
    fun fit(containerWidth: Float, containerHeight: Float, aspect: Double): Pair<Float, Float> {
        if (!(containerWidth > 0) || !(containerHeight > 0) || !(aspect > 0) || !aspect.isFinite()) return 0f to 0f
        val width = minOf(containerWidth.toDouble(), containerHeight * aspect)
        return width.toFloat() to (width / aspect).toFloat()
    }
}
