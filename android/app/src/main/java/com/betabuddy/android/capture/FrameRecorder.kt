package com.betabuddy.android.capture

import android.graphics.Rect
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.util.Log
import java.io.File
import java.nio.ByteBuffer

/**
 * The app's own recorder: H.264 from the YUV pictures the app receives anyway, into an MP4, with
 * the times the app gives them. It stands in where the camera library's recorder writes nothing
 * (see [RecorderSwitch]). The pictures go in as the camera delivers them, lying on their side;
 * the file says by how much to turn them, as a phone's camera app does.
 *
 * One recording per instance. [append] comes from the camera's frame thread, [finish] from anywhere.
 */
internal class FrameRecorder(private val file: File, private val width: Int, private val height: Int, rotationDegrees: Int) {
    /** One plane of a picture: its bytes and how they are laid out. */
    class Plane(val buffer: ByteBuffer, val rowStride: Int, val pixelStride: Int)

    private val codec: MediaCodec
    private val muxer: MediaMuxer
    private val info = MediaCodec.BufferInfo()
    private var track = -1
    private var firstUs = -1L
    private var lastUs = -1L
    private var written = 0
    private var closed = false
    private var row = ByteArray(0)

    /** Pictures that reached the file. */
    val frames: Int @Synchronized get() = written
    /** Seconds from the first picture to the last. */
    val seconds: Double @Synchronized get() = if (firstUs >= 0 && lastUs > firstUs) (lastUs - firstUs) / 1e6 else 0.0

    init {
        val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
            setInteger(MediaFormat.KEY_BIT_RATE, bitRate(width, height))
            setInteger(MediaFormat.KEY_FRAME_RATE, 30)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        }
        codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
        try {
            codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            codec.start()
            muxer = MediaMuxer(file.path, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            muxer.setOrientationHint(((rotationDegrees % 360) + 360) % 360)
        } catch (error: Exception) {
            runCatching { codec.release() }
            throw error
        }
    }

    /**
     * One picture: its three planes and the visible part of it ([crop], [width] × [height]), at
     * [timeNanos] on a clock that only has to run forwards. A picture the encoder has no room for
     * is left out rather than waited for. Returns the seconds recorded so far.
     */
    @Synchronized
    fun append(y: Plane, u: Plane, v: Plane, crop: Rect, timeNanos: Long): Double {
        if (closed) return seconds
        try {
            drain(false)
            val timeUs = timeNanos / 1_000
            if (timeUs <= lastUs) return seconds
            val index = codec.dequeueInputBuffer(0)
            if (index < 0) return seconds
            val image = codec.getInputImage(index)
            if (image == null) {
                codec.queueInputBuffer(index, 0, 0, 0, 0)
                return seconds
            }
            val planes = image.planes
            row = YuvPlanes.copy(y.buffer, y.rowStride, y.pixelStride, crop.left, crop.top, planes[0].buffer, planes[0].rowStride, planes[0].pixelStride, width, height, row)
            row = YuvPlanes.copy(u.buffer, u.rowStride, u.pixelStride, crop.left / 2, crop.top / 2, planes[1].buffer, planes[1].rowStride, planes[1].pixelStride, width / 2, height / 2, row)
            row = YuvPlanes.copy(v.buffer, v.rowStride, v.pixelStride, crop.left / 2, crop.top / 2, planes[2].buffer, planes[2].rowStride, planes[2].pixelStride, width / 2, height / 2, row)
            if (firstUs < 0) firstUs = timeUs
            codec.queueInputBuffer(index, 0, width * height * 3 / 2, timeUs - firstUs, 0)
            lastUs = timeUs
            drain(false)
        } catch (error: Exception) {
            Log.e(TAG, "own encoder: a picture was not taken", error)
        }
        return seconds
    }

    private fun drain(untilEnd: Boolean) {
        var idle = 0
        while (true) {
            val index = codec.dequeueOutputBuffer(info, if (untilEnd) 10_000 else 0)
            when {
                index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> { track = muxer.addTrack(codec.outputFormat); muxer.start() }
                index >= 0 -> {
                    val data = codec.getOutputBuffer(index)
                    if (data != null && info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0 && info.size > 0 && track >= 0) {
                        muxer.writeSampleData(track, data, info)
                        written += 1
                    }
                    val end = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                    codec.releaseOutputBuffer(index, false)
                    if (end) return
                }
                !untilEnd -> return
                ++idle > 300 -> return   // about three seconds without the end showing up
            }
        }
    }

    /** Closes the file with what was written. False when there is nothing in it that can be played. */
    @Synchronized
    fun finish(): Boolean {
        if (closed) return written > 0
        closed = true
        var playable = false
        try {
            val index = codec.dequeueInputBuffer(200_000)
            if (index >= 0) codec.queueInputBuffer(index, 0, 0, maxOf(0, lastUs - firstUs) + 1, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
            drain(true)
            playable = written > 0 && track >= 0
        } catch (error: Exception) {
            Log.e(TAG, "own encoder: not closed cleanly", error)
        }
        runCatching { codec.stop() }
        runCatching { codec.release() }
        if (track >= 0) playable = runCatching { muxer.stop() }.isSuccess && playable
        runCatching { muxer.release() }
        if (!playable) file.delete()
        return playable
    }

    companion object {
        private const val TAG = "capture"   // Local logcat only.

        /** About what a phone's own camera spends on a picture of this size. */
        fun bitRate(width: Int, height: Int): Int = (width.toLong() * height * 6.5).toInt().coerceIn(2_000_000, 12_000_000)
    }
}
