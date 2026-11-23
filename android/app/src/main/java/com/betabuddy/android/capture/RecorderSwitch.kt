package com.betabuddy.android.capture

import java.nio.ByteBuffer

/**
 * Which of two recorders writes a clip. The camera library's own recorder is tried first. It
 * takes the camera's timestamps to be on the system's uptime clock, and a camera that counts from
 * somewhere else gives it nothing it will keep: it then runs and writes no video at all. When that
 * is seen, the app's own encoder takes over, fed with the pictures the app receives anyway and
 * stamped with the app's own clock. A device that needed it once starts on it the next time.
 *
 * Pure, so the rule can be tested on a JVM.
 */
class RecorderSwitch(private val graceMillis: Long = GRACE_MILLIS, remembered: Boolean = false) {
    enum class Path {
        /** The camera library's recorder. */
        LIBRARY,
        /** The app's own encoder. */
        OWN,
    }

    var path: Path = if (remembered) Path.OWN else Path.LIBRARY
        private set

    /** The library's recorder has written video: it works on this device, and is never taken off a clip it is writing. */
    var libraryWrote: Boolean = false
        private set

    /** The change-over happened during this recording (and is worth remembering for the next one). */
    var switched: Boolean = false
        private set

    fun libraryWrote() {
        if (path == Path.LIBRARY) libraryWrote = true
    }

    /** True exactly once: when the library's recorder has had its time and has written nothing. */
    fun shouldSwitch(elapsedMillis: Long): Boolean {
        if (path != Path.LIBRARY || libraryWrote || elapsedMillis < graceMillis) return false
        return change()
    }

    /**
     * The library's recorder gave up by itself without having written anything. When the user did
     * not ask for that ([stopRequested] false), the app's encoder takes over at once.
     */
    fun libraryEndedEmpty(stopRequested: Boolean): Boolean {
        if (path != Path.LIBRARY || libraryWrote || stopRequested) return false
        return change()
    }

    private fun change(): Boolean {
        path = Path.OWN
        switched = true
        return true
    }

    companion object {
        /** How long the library's recorder gets to write its first video. */
        const val GRACE_MILLIS = 3_000L
    }
}

/** Copying one plane of a YUV 4:2:0 picture into another buffer with other strides. Pure. */
object YuvPlanes {
    /**
     * Copies a [width] × [height] block that starts at ([left], [top]) of the source plane into the
     * destination plane from its origin. Row and pixel strides are as `android.media.Image` reports
     * them; neither buffer's position is changed.
     */
    fun copy(
        source: ByteBuffer, sourceRowStride: Int, sourcePixelStride: Int, left: Int, top: Int,
        target: ByteBuffer, targetRowStride: Int, targetPixelStride: Int, width: Int, height: Int, row: ByteArray,
    ): ByteArray {
        val span = (width - 1) * sourcePixelStride + 1
        val line = if (row.size >= maxOf(span, width)) row else ByteArray(maxOf(span, width))
        val from = source.duplicate()
        val to = target.duplicate()
        for (y in 0 until height) {
            from.position((top + y) * sourceRowStride + left * sourcePixelStride)
            from.get(line, 0, span)
            if (sourcePixelStride != 1) for (x in 1 until width) line[x] = line[x * sourcePixelStride]
            val start = y * targetRowStride
            if (targetPixelStride == 1) {
                to.position(start)
                to.put(line, 0, width)
            } else {
                for (x in 0 until width) to.put(start + x * targetPixelStride, line[x])
            }
        }
        return line
    }
}
