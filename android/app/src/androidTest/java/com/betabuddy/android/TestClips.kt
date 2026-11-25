package com.betabuddy.android

import android.graphics.Canvas
import android.graphics.Rect
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import java.io.File

/** Short synthetic H.264 clips for device tests, drawn frame by frame on the encoder's input surface. */
internal object TestClips {
    /**
     * Writes [frames] frames at 30 fps to [target], each drawn by [draw], with [rotation] as the
     * orientation hint. Frames are stamped by the clock, so this takes as long as the clip.
     */
    fun encode(target: File, width: Int, height: Int, frames: Int, rotation: Int = 0, draw: (Canvas, Int) -> Unit): File {
        val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
            setInteger(MediaFormat.KEY_BIT_RATE, 4_000_000)
            setInteger(MediaFormat.KEY_FRAME_RATE, 30)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        }
        val encoder = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
        encoder.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        val surface = encoder.createInputSurface()
        encoder.start()
        val muxer = MediaMuxer(target.path, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        muxer.setOrientationHint(rotation)
        val info = MediaCodec.BufferInfo()
        var track = -1
        var firstTime = -1L
        fun drain(untilEnd: Boolean) {
            while (true) {
                val index = encoder.dequeueOutputBuffer(info, if (untilEnd) 10_000 else 0)
                when {
                    index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> { track = muxer.addTrack(encoder.outputFormat); muxer.start() }
                    index >= 0 -> {
                        val data = encoder.getOutputBuffer(index)!!
                        if (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0 && info.size > 0 && track >= 0) {
                            if (firstTime < 0) firstTime = info.presentationTimeUs
                            info.presentationTimeUs -= firstTime
                            muxer.writeSampleData(track, data, info)
                        }
                        val end = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                        encoder.releaseOutputBuffer(index, false)
                        if (end) return
                    }
                    !untilEnd -> return
                }
            }
        }
        repeat(frames) { frame ->
            val canvas = surface.lockCanvas(Rect(0, 0, width, height))
            draw(canvas, frame)
            surface.unlockCanvasAndPost(canvas)
            Thread.sleep(33)
            drain(untilEnd = false)
        }
        encoder.signalEndOfInputStream()
        drain(untilEnd = true)
        encoder.stop(); encoder.release(); surface.release()
        muxer.stop(); muxer.release()
        return target
    }
}
