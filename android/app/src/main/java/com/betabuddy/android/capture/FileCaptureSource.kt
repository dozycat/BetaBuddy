package com.betabuddy.android.capture

import android.content.Context
import android.graphics.Bitmap
import android.media.Image
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMuxer
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.view.View
import android.widget.ImageView
import androidx.lifecycle.LifecycleOwner
import com.betabuddy.android.pose.FrameGeometry
import com.betabuddy.android.pose.VideoAnalyzer
import com.betabuddy.android.pose.YuvConverter
import java.io.File
import java.nio.ByteBuffer
import kotlin.concurrent.thread

/**
 * Plays a video file in a loop, at its own pace, in place of the camera. Only reachable from
 * device tests and, in a debug build, with the `debugCaptureVideo` launch extra: it is how the
 * capture screen is exercised where there is no camera worth pointing at anything.
 *
 * "Recording" copies the file's own encoded pictures, from the next key frame on, into a new clip
 * for as long as it runs, so what is saved is a real video of the right length.
 */
class FileCaptureSource(private val file: File, override val horizontalFieldOfView: Double? = null) : CaptureSource {
    override var onFrame: ((CaptureFrame) -> Unit)? = null
    override var onInterruption: (() -> Unit)? = null
    override val needsCameraPermission: Boolean get() = false

    private val main = Handler(Looper.getMainLooper())
    @Volatile private var running = false
    @Volatile private var recording: Copy? = null
    private var player: Thread? = null
    private var preview: ImageView? = null

    override fun createPreview(context: Context): View = ImageView(context).apply {
        scaleType = ImageView.ScaleType.FIT_CENTER
        preview = this
    }

    override suspend fun start(owner: LifecycleOwner) {
        val info = VideoAnalyzer.probe(file)
        if (!file.isFile || !info.hasVideo) throw CaptureSourceException(CaptureSourceException.Reason.UNAVAILABLE)
        if (running) return
        running = true
        player = thread(name = "capture-file", isDaemon = true) {
            try { play(info.rotation) } catch (error: Exception) { Log.e(TAG, "file source stopped", error) }
            recording?.let { finish(it, ClipEnd.SOURCE) }
        }
    }

    override fun stop() {
        running = false
    }

    /** Lets a test end the pictures the way another app taking the camera would. */
    fun interrupt() {
        running = false
        main.post { onInterruption?.invoke() }
    }

    override fun startRecording(file: File, maxSeconds: Double, onProgress: (Double) -> Unit, onFinished: (ClipResult) -> Unit): ClipRecording? {
        if (!running || recording != null) return null
        val copy = Copy(file, (maxSeconds * 1_000_000).toLong(), onProgress, onFinished)
        recording = copy
        return object : ClipRecording {
            override fun stop() { copy.stopRequested = true }
        }
    }

    private inner class Copy(val target: File, val limitUs: Long, val onProgress: (Double) -> Unit, val onFinished: (ClipResult) -> Unit) {
        @Volatile var stopRequested = false
        var muxer: MediaMuxer? = null
        var track = -1
        var firstUs = -1L
        var lastUs = 0L
        var samples = 0
        var reported = 0L
    }

    private fun finish(copy: Copy, end: ClipEnd) {
        if (recording !== copy) return
        recording = null
        val seconds = if (copy.samples > 0) (copy.lastUs - copy.firstUs) / 1e6 else 0.0
        val written = copy.samples > 0 && runCatching { copy.muxer?.stop() }.isSuccess
        runCatching { copy.muxer?.release() }
        if (!written) copy.target.delete()
        main.post { copy.onFinished(ClipResult(if (written) copy.target else null, if (written) seconds else 0.0, end)) }
    }

    /** One encoded picture of the file, at [timeUs] on the source's running clock. Player thread. */
    private fun record(copy: Copy, format: MediaFormat, rotation: Int, buffer: ByteBuffer, size: Int, timeUs: Long, key: Boolean) {
        if (copy.muxer == null) {
            if (!key) return   // A clip has to begin with a picture that can be decoded on its own.
            val muxer = MediaMuxer(copy.target.path, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            muxer.setOrientationHint(rotation)
            copy.track = muxer.addTrack(format)
            muxer.start()
            copy.muxer = muxer
            copy.firstUs = timeUs
        }
        val info = MediaCodec.BufferInfo()
        info.set(0, size, timeUs - copy.firstUs, if (key) MediaCodec.BUFFER_FLAG_KEY_FRAME else 0)
        copy.muxer?.writeSampleData(copy.track, buffer, info)
        copy.samples += 1
        copy.lastUs = maxOf(copy.lastUs, timeUs)
        if (copy.lastUs - copy.reported >= 250_000) {
            copy.reported = copy.lastUs
            val seconds = (copy.lastUs - copy.firstUs) / 1e6
            main.post { copy.onProgress(seconds) }
        }
    }

    private fun play(rotation: Int) {
        val began = SystemClock.elapsedRealtimeNanos()
        var offsetUs = 0L
        val copies = PlaneCopies()
        var shown = -1L
        while (running) {
            val extractor = MediaExtractor()
            var codec: MediaCodec? = null
            var lastUs = 0L
            try {
                extractor.setDataSource(file.path)
                val track = (0 until extractor.trackCount).firstOrNull { extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true } ?: return
                extractor.selectTrack(track)
                val format = extractor.getTrackFormat(track)
                val mime = format.getString(MediaFormat.KEY_MIME) ?: return
                val decoding = MediaFormat(format)
                decoding.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
                if (Build.VERSION.SDK_INT >= 31) decoding.setInteger(MediaFormat.KEY_COLOR_TRANSFER_REQUEST, MediaFormat.COLOR_TRANSFER_SDR_VIDEO)
                val decoder = MediaCodec.createDecoderByType(mime)
                codec = decoder
                decoder.configure(decoding, null, null, 0)
                decoder.start()
                var bt709 = maxOf(format.getInteger(MediaFormat.KEY_WIDTH), format.getInteger(MediaFormat.KEY_HEIGHT)) >= 1280
                var fullRange = false
                val bufferInfo = MediaCodec.BufferInfo()
                var inputDone = false
                var idle = 0
                while (running) {
                    recording?.let { copy ->
                        if (copy.stopRequested) finish(copy, ClipEnd.STOPPED)
                        else if (copy.samples > 0 && copy.lastUs - copy.firstUs >= copy.limitUs) finish(copy, ClipEnd.LIMIT)
                    }
                    if (!inputDone) {
                        val index = decoder.dequeueInputBuffer(TIMEOUT_US)
                        if (index >= 0) {
                            val input = decoder.getInputBuffer(index) ?: return
                            val size = extractor.readSampleData(input, 0)
                            if (size < 0) {
                                decoder.queueInputBuffer(index, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                                inputDone = true
                            } else {
                                val timeUs = extractor.sampleTime
                                recording?.let { copy ->
                                    val key = extractor.sampleFlags and MediaExtractor.SAMPLE_FLAG_SYNC != 0
                                    runCatching { record(copy, format, rotation, input, size, offsetUs + timeUs, key) }.onFailure { finish(copy, ClipEnd.FAILED) }
                                    input.position(0)
                                }
                                decoder.queueInputBuffer(index, 0, size, timeUs, 0)
                                extractor.advance()
                            }
                        }
                    }
                    val index = decoder.dequeueOutputBuffer(bufferInfo, TIMEOUT_US)
                    if (index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                        val output = decoder.outputFormat
                        if (output.containsKey(MediaFormat.KEY_COLOR_STANDARD)) bt709 = output.getInteger(MediaFormat.KEY_COLOR_STANDARD) == MediaFormat.COLOR_STANDARD_BT709
                        if (output.containsKey(MediaFormat.KEY_COLOR_RANGE)) fullRange = output.getInteger(MediaFormat.KEY_COLOR_RANGE) == MediaFormat.COLOR_RANGE_FULL
                    } else if (index >= 0) {
                        idle = 0
                        val end = bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                        if (bufferInfo.size > 0) {
                            lastUs = maxOf(lastUs, bufferInfo.presentationTimeUs)
                            val timeUs = offsetUs + bufferInfo.presentationTimeUs
                            // The file's own pace: a picture is not handed on before its time.
                            val wait = timeUs * 1000 - (SystemClock.elapsedRealtimeNanos() - began)
                            if (wait > 0) Thread.sleep(wait / 1_000_000, (wait % 1_000_000).toInt())
                            decoder.getOutputImage(index)?.use { image ->
                                val frame = FileFrame(image, rotation, timeUs / 1e6, bt709, fullRange, copies)
                                onFrame?.invoke(frame)
                                if (timeUs - shown >= PREVIEW_INTERVAL_US) { shown = timeUs; show(frame) }
                            }
                        }
                        decoder.releaseOutputBuffer(index, false)
                        if (end) break
                    } else if (inputDone && ++idle > 500) {
                        break
                    }
                }
            } finally {
                runCatching { codec?.stop() }
                runCatching { codec?.release() }
                runCatching { extractor.release() }
            }
            if (lastUs <= 0) return   // Nothing could be read; do not spin.
            offsetUs += lastUs + 33_333
        }
    }

    /** The preview is the file's own pictures, a few a second, drawn into an image view. */
    private fun show(frame: FileFrame) {
        val view = preview ?: return
        val (width, height) = CaptureLimits.frameSize(frame.width, frame.height, PREVIEW_SHORT_SIDE)
        val pixels = IntArray(width * height)
        frame.render(width, height, pixels)
        val bitmap = Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
        main.post { view.setImageBitmap(bitmap) }
    }

    private class FileFrame(
        private val image: Image, private val rotation: Int, override val time: Double,
        private val bt709: Boolean, private val fullRange: Boolean, private val copies: PlaneCopies,
    ) : CaptureFrame {
        private val crop = image.cropRect
        private val size = FrameGeometry.displaySize(crop.width(), crop.height(), rotation)
        override val width: Int get() = size.first
        override val height: Int get() = size.second

        override fun render(outWidth: Int, outHeight: Int, out: IntArray) {
            val planes = image.planes.take(3).mapIndexed { index, plane -> copies.copy(index, plane.buffer, plane.rowStride, plane.pixelStride) }
            YuvConverter.convert(planes[0], planes[1], planes[2], crop.left, crop.top, crop.width(), crop.height(), rotation, outWidth, outHeight, bt709, fullRange, out)
        }
    }

    private companion object {
        const val TAG = "capture"   // Local logcat only.
        const val TIMEOUT_US = 5_000L
        const val PREVIEW_INTERVAL_US = 100_000L
        const val PREVIEW_SHORT_SIDE = 360
    }
}
