package com.betabuddy.android.pose

import android.content.Context
import android.graphics.Bitmap
import android.media.Image
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.os.Build
import android.os.PowerManager
import android.os.SystemClock
import androidx.annotation.VisibleForTesting
import com.betabuddy.android.data.Joint
import com.betabuddy.android.data.PoseAnalysis
import com.betabuddy.android.data.PoseFrame
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.util.concurrent.Executors
import kotlin.coroutines.coroutineContext
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * On-device pose analysis of one clip. Counterpart of ios/BetaBuddy/Core/VideoAnalyzer.swift.
 *
 * - MediaPipe Pose Landmarker Full (float16, 33 points), bundled in assets and hash-checked.
 * - One landmarker at a time, video running mode, on one dedicated worker thread; never on main.
 * - `MediaExtractor` + `MediaCodec` decode the clip in order. A frame that is not sampled is
 *   released without colour conversion; a sampled frame is rotated and scaled in one pass into a
 *   single reused bitmap. The clip is never expanded into a list of bitmaps.
 * - Rotation metadata is baked into the model input, so its aspect ratio matches playback.
 * - CPU delegate is the baseline. GPU is available as an option and is not the default.
 * - Cancellation and the thermal state are checked between frames.
 */
class VideoAnalyzer(context: Context) {
    private val context = context.applicationContext

    /** Receives each picture exactly as the model is about to see it. For the orientation test only. */
    @VisibleForTesting
    internal var frameObserver: ((Bitmap) -> Unit)? = null

    /**
     * @param progress called from the worker thread, at most once per five sampled frames
     * @throws AnalysisException with a reason the UI can explain
     * @throws CancellationException when the caller cancels; no partial result is returned
     */
    suspend fun analyze(
        video: File, quality: AnalysisQuality, useGpu: Boolean = false, progress: (Double) -> Unit,
    ): PoseAnalysis = serial.withLock {
        withContext(worker) {
            try {
                run(video, quality, useGpu, progress)
            } catch (error: CancellationException) {
                throw error
            } catch (error: AnalysisException) {
                throw error
            } catch (error: Exception) {
                // Extractor, codec and model errors on a clip that passed the import checks.
                throw AnalysisException(AnalysisError.INVALID_VIDEO, error)
            }
        }
    }

    private suspend fun run(video: File, quality: AnalysisQuality, useGpu: Boolean, progress: (Double) -> Unit): PoseAnalysis {
        coroutineContext.ensureActive()
        val start = SystemClock.elapsedRealtime()
        val model = loadModel(context)
        val info = probe(video)
        ImportRules.check(video.length(), info.durationSeconds, info.hasVideo)?.let { throw AnalysisException(it) }
        val (displayWidth, displayHeight) = FrameGeometry.displaySize(info.width, info.height, info.rotation)
        val aspect = displayWidth.toDouble() / displayHeight
        if (!(aspect > 0) || !aspect.isFinite()) throw AnalysisException(AnalysisError.INVALID_VIDEO)
        val duration = info.durationSeconds!!

        val extractor = MediaExtractor()
        var openCodec: MediaCodec? = null
        var openLandmarker: PoseLandmarker? = null
        var openBitmap: Bitmap? = null
        try {
            extractor.setDataSource(video.path)
            val track = (0 until extractor.trackCount).firstOrNull {
                extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true
            } ?: throw AnalysisException(AnalysisError.INVALID_VIDEO)
            extractor.selectTrack(track)
            val format = extractor.getTrackFormat(track)
            val mime = format.getString(MediaFormat.KEY_MIME) ?: throw AnalysisException(AnalysisError.INVALID_VIDEO)
            // Byte-buffer output in a flexible YUV 4:2:0 layout, read through android.media.Image.
            format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
            if (Build.VERSION.SDK_INT >= 31) {
                // Ask for SDR output when the clip is HDR (HLG / PQ); the model expects ordinary 8-bit colour.
                format.setInteger(MediaFormat.KEY_COLOR_TRANSFER_REQUEST, MediaFormat.COLOR_TRANSFER_SDR_VIDEO)
            }

            val baseOptions = BaseOptions.builder()
                .setModelAssetBuffer(model)
                // CPU: works on emulators and gives a reproducible baseline for speed and heat.
                .setDelegate(if (useGpu) Delegate.GPU else Delegate.CPU)
                .build()
            val landmarker = PoseLandmarker.createFromOptions(
                context,
                PoseLandmarker.PoseLandmarkerOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setRunningMode(RunningMode.VIDEO)
                    .setNumPoses(1)
                    // MediaPipe defaults: raising these can miss the climber and start video
                    // tracking on a high-contrast background false positive (see the iOS regression notes).
                    .setMinPoseDetectionConfidence(0.5f)
                    .setMinPosePresenceConfidence(0.5f)
                    .setMinTrackingConfidence(0.5f)
                    .build(),
            )
            openLandmarker = landmarker

            val codec = MediaCodec.createDecoderByType(mime)
            openCodec = codec
            codec.configure(format, null, null, 0)
            codec.start()

            val power = context.getSystemService(Context.POWER_SERVICE) as? PowerManager
            val sampler = FrameSampler(quality.fps)
            val frames = ArrayList<PoseFrame>()
            val buffers = PlaneBuffers()
            var pixels = IntArray(0)
            var colour = ColourInfo.guess(info.width, info.height)
            val bufferInfo = MediaCodec.BufferInfo()
            var sampled = 0
            var inputDone = false
            var outputDone = false
            var idleRounds = 0

            while (!outputDone) {
                coroutineContext.ensureActive()
                if (Build.VERSION.SDK_INT >= 29 && power != null &&
                    power.currentThermalStatus >= PowerManager.THERMAL_STATUS_SEVERE
                ) throw AnalysisException(AnalysisError.OVERHEATED)

                if (!inputDone) {
                    val index = codec.dequeueInputBuffer(TIMEOUT_US)
                    if (index >= 0) {
                        val input = codec.getInputBuffer(index) ?: throw AnalysisException(AnalysisError.INVALID_VIDEO)
                        val size = extractor.readSampleData(input, 0)
                        if (size < 0) {
                            codec.queueInputBuffer(index, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(index, 0, size, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }

                val index = codec.dequeueOutputBuffer(bufferInfo, TIMEOUT_US)
                if (index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    colour = ColourInfo.from(codec.outputFormat, colour)
                } else if (index >= 0) {
                    idleRounds = 0
                    val endOfStream = bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                    val time = bufferInfo.presentationTimeUs / 1_000_000.0
                    // Skip before colour conversion, scaling and inference. Never queue decoded pictures.
                    val wanted = bufferInfo.size > 0 && sampler.shouldSample(time)
                    var converted: Bitmap? = null
                    if (wanted) {
                        val image = codec.getOutputImage(index) ?: throw AnalysisException(AnalysisError.INVALID_VIDEO)
                        try {
                            val crop = image.cropRect
                            val (frameWidth, frameHeight) = FrameGeometry.displaySize(crop.width(), crop.height(), info.rotation)
                            val (outWidth, outHeight) = FrameGeometry.targetSize(frameWidth, frameHeight, quality.maxDimension)
                            val existing = openBitmap
                            val target = if (existing != null && existing.width == outWidth && existing.height == outHeight) {
                                existing
                            } else {
                                existing?.recycle()
                                pixels = IntArray(outWidth * outHeight)
                                Bitmap.createBitmap(outWidth, outHeight, Bitmap.Config.ARGB_8888).also { openBitmap = it }
                            }
                            val planes = buffers.read(image)
                            YuvConverter.convert(
                                planes[0], planes[1], planes[2],
                                crop.left, crop.top, crop.width(), crop.height(),
                                info.rotation, outWidth, outHeight, colour.bt709, colour.fullRange, pixels,
                            )
                            target.setPixels(pixels, 0, outWidth, 0, 0, outWidth, outHeight)
                            converted = target
                        } finally {
                            image.close()
                        }
                    }
                    codec.releaseOutputBuffer(index, false)
                    if (converted != null) {
                        frameObserver?.invoke(converted)
                        // Synchronous in video mode, so the single bitmap can be reused for the next frame.
                        val result = landmarker.detectForVideo(BitmapImageBuilder(converted).build(), sampler.timestampMs(time))
                        sampled += 1
                        val points = result.landmarks().firstOrNull()
                        if (points != null && points.size == 33) {
                            val joints = points.map { point ->
                                val x = point.x().toDouble()
                                val y = point.y().toDouble()
                                val confidence = minOf(point.visibility().orElse(0f), point.presence().orElse(0f)).toDouble()
                                if (x.isFinite() && y.isFinite() && confidence.isFinite()) Joint(x, y, confidence) else Joint(0.0, 0.0, 0.0)
                            }
                            val frame = PoseFrame(time, joints)
                            // Keep only samples whose shoulders and hips are reliable and inside the picture.
                            if (frame.hasReliableTorso) frames.add(frame)
                        }
                        if (sampled % 5 == 0) progress(minOf(0.99, time / duration))
                    }
                    if (endOfStream) outputDone = true
                } else if (inputDone && ++idleRounds > MAX_IDLE_ROUNDS) {
                    throw AnalysisException(AnalysisError.INVALID_VIDEO) // decoder never signalled the end
                }
            }

            coroutineContext.ensureActive()
            if (frames.count { it.hip != null } < 3) throw AnalysisException(AnalysisError.NO_POSE)
            progress(1.0)
            return PoseAnalysis(
                frames = frames, sampledFrames = sampled, duration = duration, aspectRatio = aspect,
                processingSeconds = (SystemClock.elapsedRealtime() - start) / 1000.0,
                moments = PoseMath.moments(frames, aspect),
            )
        } finally {
            runCatching { openCodec?.stop() }
            runCatching { openCodec?.release() }
            runCatching { extractor.release() }
            runCatching { openLandmarker?.close() }
            runCatching { openBitmap?.recycle() }
        }
    }

    /** First frame, upright, at most 600 px on the long edge, as JPEG. */
    suspend fun thumbnail(video: File): ByteArray = withContext(worker) {
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(video.path)
            val frame = (if (Build.VERSION.SDK_INT >= 27) {
                retriever.getScaledFrameAtTime(0, MediaMetadataRetriever.OPTION_CLOSEST_SYNC, 600, 600)
            } else {
                retriever.getFrameAtTime(0, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
            }) ?: throw AnalysisException(AnalysisError.INVALID_VIDEO)
            val (width, height) = FrameGeometry.targetSize(frame.width, frame.height, 600)
            val scaled = if (width == frame.width) frame else Bitmap.createScaledBitmap(frame, width, height, true)
            val output = ByteArrayOutputStream()
            if (!scaled.compress(Bitmap.CompressFormat.JPEG, 80, output)) throw AnalysisException(AnalysisError.INVALID_VIDEO)
            if (scaled !== frame) scaled.recycle()
            frame.recycle()
            output.toByteArray()
        } catch (error: AnalysisException) {
            throw error
        } catch (error: Exception) {
            throw AnalysisException(AnalysisError.INVALID_VIDEO, error)
        } finally {
            runCatching { retriever.release() }
        }
    }

    class VideoInfo(val hasVideo: Boolean, val durationSeconds: Double?, val width: Int, val height: Int, val rotation: Int)

    /** Reusable copies of the three planes of a decoded picture. */
    private class PlaneBuffers {
        private val arrays = arrayOfNulls<ByteArray>(3)
        fun read(image: Image): List<YuvConverter.Plane> = image.planes.take(3).mapIndexed { index, plane ->
            val buffer = plane.buffer
            buffer.rewind()
            val size = buffer.remaining()
            val array = arrays[index]?.takeIf { it.size >= size } ?: ByteArray(size).also { arrays[index] = it }
            buffer.get(array, 0, size)
            YuvConverter.Plane(array, plane.rowStride, plane.pixelStride)
        }
    }

    private class ColourInfo(val bt709: Boolean, val fullRange: Boolean) {
        companion object {
            /** Common convention when the stream does not say: HD is BT.709, SD is BT.601, video range. */
            fun guess(width: Int, height: Int) = ColourInfo(bt709 = maxOf(width, height) >= 1280 || minOf(width, height) > 576, fullRange = false)

            fun from(format: MediaFormat, fallback: ColourInfo): ColourInfo {
                val standard = if (format.containsKey(MediaFormat.KEY_COLOR_STANDARD)) format.getInteger(MediaFormat.KEY_COLOR_STANDARD) else 0
                val range = if (format.containsKey(MediaFormat.KEY_COLOR_RANGE)) format.getInteger(MediaFormat.KEY_COLOR_RANGE) else 0
                return ColourInfo(
                    bt709 = when (standard) {
                        MediaFormat.COLOR_STANDARD_BT709 -> true
                        MediaFormat.COLOR_STANDARD_BT601_PAL, MediaFormat.COLOR_STANDARD_BT601_NTSC -> false
                        else -> fallback.bt709
                    },
                    fullRange = when (range) {
                        MediaFormat.COLOR_RANGE_FULL -> true
                        MediaFormat.COLOR_RANGE_LIMITED -> false
                        else -> fallback.fullRange
                    },
                )
            }
        }
    }

    companion object {
        private const val TIMEOUT_US = 10_000L
        private const val MAX_IDLE_ROUNDS = 500 // about five seconds without output after the last input

        // One analysis at a time, always on the same thread: decoder, bitmap and model are owned by it.
        private val serial = Mutex()
        private val worker = Executors.newSingleThreadExecutor { task ->
            Thread(task, "pose-analysis").apply { isDaemon = true }
        }.asCoroutineDispatcher()

        /** Reads the bundled model and refuses anything but the pinned file. */
        fun loadModel(context: Context): ByteBuffer {
            val bytes = try {
                context.assets.open(PoseModel.ASSET).use { it.readBytes() }
            } catch (error: IOException) {
                throw AnalysisException(AnalysisError.MISSING_MODEL, error)
            }
            if (PoseModel.sha256(bytes) != PoseModel.SHA256) throw AnalysisException(AnalysisError.MISSING_MODEL)
            // MediaPipe requires a direct (or mapped) buffer.
            return ByteBuffer.allocateDirect(bytes.size).apply { put(bytes); rewind() }
        }

        /** Container-level facts about a clip. Never throws: an unreadable file has no video. */
        fun probe(video: File): VideoInfo {
            val extractor = MediaExtractor()
            try {
                extractor.setDataSource(video.path)
                for (index in 0 until extractor.trackCount) {
                    val format = extractor.getTrackFormat(index)
                    if (format.getString(MediaFormat.KEY_MIME)?.startsWith("video/") != true) continue
                    val width = if (format.containsKey(MediaFormat.KEY_WIDTH)) format.getInteger(MediaFormat.KEY_WIDTH) else 0
                    val height = if (format.containsKey(MediaFormat.KEY_HEIGHT)) format.getInteger(MediaFormat.KEY_HEIGHT) else 0
                    val duration = if (format.containsKey(MediaFormat.KEY_DURATION)) format.getLong(MediaFormat.KEY_DURATION) / 1_000_000.0 else null
                    val rotation = if (format.containsKey(MediaFormat.KEY_ROTATION)) format.getInteger(MediaFormat.KEY_ROTATION) else retrieverRotation(video)
                    return VideoInfo(width > 0 && height > 0, duration, width, height, FrameGeometry.normalizeRotation(rotation))
                }
            } catch (error: Exception) {
                // fall through: not a readable video
            } finally {
                runCatching { extractor.release() }
            }
            return VideoInfo(false, null, 0, 0, 0)
        }

        private fun retrieverRotation(video: File): Int {
            val retriever = MediaMetadataRetriever()
            return try {
                retriever.setDataSource(video.path)
                retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull() ?: 0
            } catch (error: Exception) {
                0
            } finally {
                runCatching { retriever.release() }
            }
        }
    }
}
