package com.betabuddy.android.capture

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.util.Size
import android.view.View
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.CameraState
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FallbackStrategy
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.Observer
import com.betabuddy.android.pose.FrameGeometry
import com.betabuddy.android.pose.YuvConverter
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlinx.coroutines.suspendCancellableCoroutine

/** Why a source cannot deliver pictures. */
class CaptureSourceException(val reason: Reason) : Exception(reason.name) {
    enum class Reason { DENIED, UNAVAILABLE }
}

/**
 * One picture from a source, in the orientation the user sees. Only valid during the call it is
 * handed over in: whoever wants to keep it renders it first.
 */
interface CaptureFrame {
    val width: Int
    val height: Int
    /** Seconds on the source's own clock; increases from frame to frame. */
    val time: Double

    /** The picture, upright, scaled to [outWidth] × [outHeight], as ARGB. Scaled on the processor. */
    fun render(outWidth: Int, outHeight: Int, out: IntArray)
}

/** How a recording ended, as far as the source can tell. */
enum class ClipEnd { STOPPED, LIMIT, STORAGE, SOURCE, FAILED }

/** [file] is null when nothing usable was written. */
data class ClipResult(val file: File?, val seconds: Double, val end: ClipEnd)

interface ClipRecording {
    /** Closes the clip with what has been written. The result arrives through the callback given at the start. */
    fun stop()
}

/**
 * Where the capture screen's pictures come from, and what writes them to a clip. On a phone that
 * is the back camera; tests and debug runs play a file instead. Everything after this point (the
 * checks, saving, the queue) is the same code for both.
 */
interface CaptureSource {
    /** Upright frames with increasing times, on the source's own thread. */
    var onFrame: ((CaptureFrame) -> Unit)?
    /** The source stopped delivering by itself: another app took the camera, an error. Main thread. */
    var onInterruption: (() -> Unit)?
    /** Horizontal field of view of the upright frames in degrees, when known. */
    val horizontalFieldOfView: Double?
    /** Whether [start] needs the camera permission to have been granted. */
    val needsCameraPermission: Boolean

    /** A view showing what the source delivers, uncropped. Main thread. */
    fun createPreview(context: Context): View

    /** Starts delivering. Throws [CaptureSourceException]. Main thread. */
    suspend fun start(owner: LifecycleOwner)
    fun stop()

    /**
     * Starts writing what the source delivers into [file], video only. [onProgress] (seconds
     * written so far) and [onFinished] arrive on the main thread; [onFinished] exactly once, also
     * when the recording ends by itself. Null when recording cannot start.
     */
    fun startRecording(file: File, maxSeconds: Double, onProgress: (Double) -> Unit, onFinished: (ClipResult) -> Unit): ClipRecording?
}

/** Reusable copies of the three planes of a YUV 4:2:0 picture. */
internal class PlaneCopies {
    private val arrays = arrayOfNulls<ByteArray>(3)
    fun copy(index: Int, buffer: java.nio.ByteBuffer, rowStride: Int, pixelStride: Int): YuvConverter.Plane {
        val source = buffer.duplicate()
        source.rewind()
        val size = source.remaining()
        val array = arrays[index]?.takeIf { it.size >= size } ?: ByteArray(size).also { arrays[index] = it }
        source.get(array, 0, size)
        return YuvConverter.Plane(array, rowStride, pixelStride)
    }
}

/**
 * The back camera through CameraX: a preview, 1080p video to a file and small frames for the
 * checks, all 16:9 so the three show the same picture. Video only: no audio source is ever
 * attached, so the microphone is not asked for.
 */
class CameraSource(context: Context) : CaptureSource {
    private val context = context.applicationContext
    override var onFrame: ((CaptureFrame) -> Unit)? = null
    override var onInterruption: (() -> Unit)? = null
    override var horizontalFieldOfView: Double? = null
        private set
    override val needsCameraPermission: Boolean get() = true

    private val main = Handler(Looper.getMainLooper())
    private var provider: ProcessCameraProvider? = null
    private var preview: Preview? = null
    private var video: VideoCapture<Recorder>? = null
    private var previewView: PreviewView? = null
    private var frames: ExecutorService? = null
    private var stateObserver: Observer<CameraState>? = null
    private var stateSource: androidx.lifecycle.LiveData<CameraState>? = null
    private val planes = PlaneCopies()

    override fun createPreview(context: Context): View = PreviewView(context).apply {
        // The whole picture, never cropped: what is drawn over it is positioned against the full frame.
        scaleType = PreviewView.ScaleType.FIT_CENTER
        implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        previewView = this
        preview?.surfaceProvider = surfaceProvider
    }

    override suspend fun start(owner: LifecycleOwner) {
        // Asked before permission: where there is no camera at all, there is nothing to ask for.
        if (!context.packageManager.hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY)) throw CaptureSourceException(CaptureSourceException.Reason.UNAVAILABLE)
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            throw CaptureSourceException(CaptureSourceException.Reason.DENIED)
        }
        val provider = try { cameraProvider() } catch (error: Exception) {
            Log.e(TAG, "camera provider unavailable", error)
            throw CaptureSourceException(CaptureSourceException.Reason.UNAVAILABLE)
        }
        val selector = CameraSelector.DEFAULT_BACK_CAMERA
        val present = try { provider.hasCamera(selector) } catch (error: Exception) { false }
        if (!present) throw CaptureSourceException(CaptureSourceException.Reason.UNAVAILABLE)

        val wide = AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY
        val preview = Preview.Builder().setResolutionSelector(ResolutionSelector.Builder().setAspectRatioStrategy(wide).build()).build()
        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(
                ResolutionSelector.Builder().setAspectRatioStrategy(wide)
                    .setResolutionStrategy(ResolutionStrategy(Size(1280, 720), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER)).build(),
            )
            // A late frame is dropped rather than held: the checks only need a few frames a second.
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .build()
        val recorder = Recorder.Builder()
            .setQualitySelector(QualitySelector.from(Quality.FHD, FallbackStrategy.lowerQualityOrHigherThan(Quality.FHD)))
            .setAspectRatio(AspectRatio.RATIO_16_9)
            .build()
        val video = VideoCapture.withOutput(recorder)
        val executor = Executors.newSingleThreadExecutor { task -> Thread(task, "capture-frames").apply { isDaemon = true } }
        analysis.setAnalyzer(executor) { image -> image.use { onFrame?.invoke(CameraFrame(it, planes)) } }

        val camera = try {
            provider.unbindAll()
            provider.bindToLifecycle(owner, selector, preview, analysis, video)
        } catch (error: Exception) {
            // No camera that can do all three at once, or the camera could not be opened.
            Log.e(TAG, "camera cannot be bound", error)
            executor.shutdown()
            throw CaptureSourceException(CaptureSourceException.Reason.UNAVAILABLE)
        }
        this.provider = provider
        this.preview = preview
        this.video = video
        frames = executor
        previewView?.let { preview.surfaceProvider = it.surfaceProvider }
        horizontalFieldOfView = backCameraFieldOfView()
        val observer = Observer<CameraState> { state ->
            // Errors the camera will not get over by itself while this screen is up.
            when (state.error?.code) {
                CameraState.ERROR_CAMERA_IN_USE, CameraState.ERROR_MAX_CAMERAS_IN_USE, CameraState.ERROR_CAMERA_DISABLED,
                CameraState.ERROR_CAMERA_FATAL_ERROR, CameraState.ERROR_DO_NOT_DISTURB_MODE_ENABLED -> onInterruption?.invoke()
                else -> Unit
            }
        }
        stateObserver = observer
        stateSource = camera.cameraInfo.cameraState.also { it.observeForever(observer) }
    }

    override fun stop() {
        stateObserver?.let { stateSource?.removeObserver(it) }
        stateObserver = null
        stateSource = null
        runCatching { provider?.unbindAll() }
        provider = null
        preview = null
        video = null
        frames?.shutdown()
        frames = null
    }

    @SuppressLint("MissingPermission")   // No audio is enabled, so only the camera permission is involved; it was checked in start.
    override fun startRecording(file: File, maxSeconds: Double, onProgress: (Double) -> Unit, onFinished: (ClipResult) -> Unit): ClipRecording? {
        val video = video ?: return null
        var seconds = 0.0
        val recording: Recording = try {
            // `withAudioEnabled` is never called: the clip has a video track and nothing else.
            video.output.prepareRecording(context, FileOutputOptions.Builder(file).setDurationLimitMillis((maxSeconds * 1000).toLong()).build())
                .start(ContextCompat.getMainExecutor(context)) { event ->
                    when (event) {
                        is VideoRecordEvent.Status -> {
                            seconds = event.recordingStats.recordedDurationNanos / 1e9
                            onProgress(seconds)
                        }
                        is VideoRecordEvent.Finalize -> {
                            seconds = maxOf(seconds, event.recordingStats.recordedDurationNanos / 1e9)
                            val end = when (event.error) {
                                VideoRecordEvent.Finalize.ERROR_NONE -> ClipEnd.STOPPED
                                VideoRecordEvent.Finalize.ERROR_DURATION_LIMIT_REACHED, VideoRecordEvent.Finalize.ERROR_FILE_SIZE_LIMIT_REACHED -> ClipEnd.LIMIT
                                VideoRecordEvent.Finalize.ERROR_INSUFFICIENT_STORAGE -> ClipEnd.STORAGE
                                VideoRecordEvent.Finalize.ERROR_SOURCE_INACTIVE -> ClipEnd.SOURCE
                                else -> ClipEnd.FAILED
                            }
                            // Of these, only the last kind leaves no file that can be played.
                            val usable = end != ClipEnd.FAILED && file.isFile && file.length() > 0
                            if (!usable) Log.e(TAG, "recording ended with error ${event.error}", event.cause)
                            onFinished(ClipResult(if (usable) file else null, if (usable) seconds else 0.0, end))
                        }
                        else -> Unit
                    }
                }
        } catch (error: Exception) {
            Log.e(TAG, "recording could not start", error)
            return null
        }
        return object : ClipRecording {
            override fun stop() { main.post { runCatching { recording.stop() } } }
        }
    }

    private suspend fun cameraProvider(): ProcessCameraProvider = suspendCancellableCoroutine { continuation ->
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            try { continuation.resume(future.get()) } catch (error: Exception) { continuation.resumeWithException(error) }
        }, ContextCompat.getMainExecutor(context))
    }

    /**
     * From the sensor's size and the lens: the field of view across the sensor's long side, then
     * across the short side of a 16:9 picture, which is the horizontal one once the frame is upright.
     */
    private fun backCameraFieldOfView(): Double? = runCatching {
        val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val id = manager.cameraIdList.firstOrNull { manager.getCameraCharacteristics(it).get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK } ?: return null
        val characteristics = manager.getCameraCharacteristics(id)
        val sensor = characteristics.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE) ?: return null
        val focal = characteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull() ?: return null
        val long = CaptureMotion.fieldOfView(maxOf(sensor.width, sensor.height).toDouble(), focal.toDouble()) ?: return null
        CaptureMotion.uprightFieldOfView(long, 16.0, 9.0)
    }.getOrNull()

    private class CameraFrame(private val image: ImageProxy, private val copies: PlaneCopies) : CaptureFrame {
        private val rotation = FrameGeometry.normalizeRotation(image.imageInfo.rotationDegrees)
        private val crop = image.cropRect
        override val width: Int = if (rotation % 180 == 90) crop.height() else crop.width()
        override val height: Int = if (rotation % 180 == 90) crop.width() else crop.height()
        override val time: Double = image.imageInfo.timestamp / 1e9

        override fun render(outWidth: Int, outHeight: Int, out: IntArray) {
            val planes = image.planes.take(3).mapIndexed { index, plane -> copies.copy(index, plane.buffer, plane.rowStride, plane.pixelStride) }
            // Camera frames are full-range BT.601 by Android's convention for YUV_420_888 from a camera.
            YuvConverter.convert(planes[0], planes[1], planes[2], crop.left, crop.top, crop.width(), crop.height(), rotation, outWidth, outHeight, bt709 = false, fullRange = true, out = out)
        }
    }

    private companion object {
        const val TAG = "capture"   // Local logcat only.
    }
}
