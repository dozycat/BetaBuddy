package com.betabuddy.android.pose

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Rect
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.media.MediaMuxer
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import kotlin.math.abs
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Device tests for decoding and inference, mirroring ios/BetaBuddyTests/VideoAnalyzerTests.swift.
 * They need a device or emulator with an H.264 encoder and decoder. The fixture is Google's
 * MediaPipe sample picture pose.jpg (not a climb, not an accuracy benchmark).
 *
 * NOT RUN YET: written without an Android SDK on the machine. Treat failures here as unverified
 * test code first, product code second.
 */
@RunWith(AndroidJUnit4::class)
class VideoAnalyzerInstrumentedTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext
    private val created = mutableListOf<File>()

    @After fun cleanUp() { created.forEach { it.delete() } }

    private fun fixture(): Bitmap =
        instrumentation.context.assets.open("pose.jpg").use { BitmapFactory.decodeStream(it) }

    /** One second of 30 fps H.264 showing [image] (or black), with the given rotation metadata. */
    private fun makeVideo(image: Bitmap?, rotation: Int = 0, width: Int = 640, height: Int = 480): File {
        val file = File(context.cacheDir, "smoke-${System.nanoTime()}.mp4").also { created += it }
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
        val muxer = MediaMuxer(file.path, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
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
        repeat(30) {
            val canvas = surface.lockCanvas(null)
            canvas.drawColor(Color.BLACK)
            image?.let { canvas.drawBitmap(it, null, Rect(0, 0, width, height), null) }
            surface.unlockCanvasAndPost(canvas) // timestamped with the clock, so pace the frames
            Thread.sleep(33)
            drain(untilEnd = false)
        }
        encoder.signalEndOfInputStream()
        drain(untilEnd = true)
        encoder.stop(); encoder.release(); surface.release()
        muxer.stop(); muxer.release()
        return file
    }

    @Test fun bundledModelIsThePinnedFile() {
        assertEquals(PoseModel.BYTES.toInt(), VideoAnalyzer.loadModel(context).capacity())
    }

    @Test fun bundledModelProcessesVideoOffline() = runBlocking {
        val video = makeVideo(fixture())
        var lastProgress = 0.0
        val result = VideoAnalyzer(context).analyze(video, AnalysisQuality.BALANCED) { lastProgress = it }
        assertTrue("coverage ${result.coverage}", result.coverage > 0.5)
        assertTrue(result.frames.size > 2)
        assertEquals(33, result.frames.first().joints.size)
        assertEquals(640.0 / 480, result.aspectRatio, 0.01)
        assertTrue("sampled ${result.sampledFrames}", result.sampledFrames <= 12)
        assertTrue(result.frames.all { it.hasReliableTorso })
        assertEquals(1.0, lastProgress, 0.0)
    }

    // Compare the pixels handed to the model with the frame the platform shows for the same clip,
    // using asymmetric content so that a 180° or a 90° mix-up cannot pass.
    @Test fun inferenceOrientationMatchesPlayback() = runBlocking {
        for (rotation in listOf(0, 90, 180, 270)) {
            val video = makeVideo(fixture(), rotation)
            var actual: Bitmap? = null
            val analyzer = VideoAnalyzer(context)
            analyzer.frameObserver = { if (actual == null) actual = it.copy(Bitmap.Config.ARGB_8888, false) }
            val result = analyzer.analyze(video, AnalysisQuality.BALANCED) { }
            val retriever = MediaMetadataRetriever()
            retriever.setDataSource(video.path)
            val expected = retriever.getFrameAtTime(0, MediaMetadataRetriever.OPTION_CLOSEST)!! // upright, rotation applied
            retriever.release()
            val seen = actual!!
            assertEquals("rotation $rotation", expected.width.toDouble() / expected.height, seen.width.toDouble() / seen.height, 0.02)
            assertEquals("rotation $rotation", expected.width.toDouble() / expected.height, result.aspectRatio, 0.02)
            fun pixels(bitmap: Bitmap): IntArray {
                val small = Bitmap.createScaledBitmap(bitmap, 32, 32, true)
                return IntArray(32 * 32).also { small.getPixels(it, 0, 32, 0, 0, 32, 32) }
            }
            val a = pixels(seen)
            val b = pixels(expected)
            var error = 0.0
            for (i in a.indices) for (shift in intArrayOf(16, 8, 0)) error += abs((a[i] shr shift and 0xFF) - (b[i] shr shift and 0xFF))
            val meanError = error / (32 * 32 * 3)
            assertTrue("Orientation mismatch at $rotation°, MAE $meanError", meanError < 12)
        }
    }

    @Test fun noPersonReturnsActionableError() = runBlocking {
        val video = makeVideo(null)
        try {
            VideoAnalyzer(context).analyze(video, AnalysisQuality.BALANCED) { }
            fail("A blank video must not receive invented results")
        } catch (error: AnalysisException) {
            assertEquals(AnalysisError.NO_POSE, error.error)
        }
    }

    @Test fun unreadableFileIsAnInvalidVideo() = runBlocking {
        val file = File(context.cacheDir, "broken.mp4").also { created += it; it.writeText("not a video") }
        try {
            VideoAnalyzer(context).analyze(file, AnalysisQuality.BALANCED) { }
            fail()
        } catch (error: AnalysisException) {
            assertEquals(AnalysisError.INVALID_VIDEO, error.error)
        }
    }

    @Test fun cancellationDoesNotReturnPartialAnalysis() = runBlocking {
        val video = makeVideo(null)
        val work = async { VideoAnalyzer(context).analyze(video, AnalysisQuality.DETAILED) { } }
        work.cancel()
        try { work.await(); fail("Cancelled analysis returned results") } catch (expected: CancellationException) { }
    }

    @Test fun thumbnailIsASmallJpeg() = runBlocking {
        val bytes = VideoAnalyzer(context).thumbnail(makeVideo(fixture(), rotation = 90))
        val decoded = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        assertTrue(maxOf(decoded.width, decoded.height) <= 600)
        assertTrue("portrait after rotation", decoded.height > decoded.width)
    }
}
