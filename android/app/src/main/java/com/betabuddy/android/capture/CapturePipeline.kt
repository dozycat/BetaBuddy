package com.betabuddy.android.capture

import android.content.Context
import android.graphics.Bitmap
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.os.PowerManager
import android.os.SystemClock
import android.util.Base64
import com.betabuddy.android.pose.VideoAnalyzer
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import java.io.ByteArrayOutputStream
import java.util.Locale
import kotlin.math.sqrt

/**
 * The phone's attitude and how much it is moving, from the gravity sensor (or, where there is
 * none, a smoothed accelerometer) and the gyroscope. All of it is optional help for the capture page.
 */
class CaptureMotionMonitor(context: Context) : SensorEventListener {
    private val manager = context.applicationContext.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
    private var thread: HandlerThread? = null
    private val lock = Any()
    private var gravity: DoubleArray? = null
    private var smoothed = false
    private var squares = 0.0
    private var samples = 0

    /** Stands in for the sensors where there are none; only ever set in debug builds and tests. */
    @Volatile var fixed: CaptureFrameInfo? = null

    fun start() {
        val manager = manager ?: return
        if (thread != null) return
        val worker = HandlerThread("capture-motion").also { it.start() }
        thread = worker
        val handler = Handler(worker.looper)
        val down = manager.getDefaultSensor(Sensor.TYPE_GRAVITY)
        smoothed = down == null
        (down ?: manager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER))?.let { manager.registerListener(this, it, SAMPLE_US, handler) }
        manager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)?.let { manager.registerListener(this, it, SAMPLE_US, handler) }
    }

    fun stop() {
        manager?.unregisterListener(this)
        thread?.quitSafely()
        thread = null
        synchronized(lock) { gravity = null; squares = 0.0; samples = 0 }
    }

    override fun onSensorChanged(event: SensorEvent) {
        val x = event.values[0].toDouble(); val y = event.values[1].toDouble(); val z = event.values[2].toDouble()
        synchronized(lock) {
            if (event.sensor.type == Sensor.TYPE_GYROSCOPE) {
                squares += x * x + y * y + z * z
                samples += 1
            } else {
                val last = gravity
                // An accelerometer also feels the hand moving; most of that is averaged away here.
                gravity = if (smoothed && last != null) doubleArrayOf(last[0] * 0.9 + x * 0.1, last[1] * 0.9 + y * 0.1, last[2] * 0.9 + z * 0.1) else doubleArrayOf(x, y, z)
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    /** Attitude now and shake since the last call. All null without motion sensors. */
    fun take(): CaptureFrameInfo = synchronized(lock) {
        val down = gravity?.let { CaptureMotion.down(it[0], it[1], it[2]) }
        val shake = if (samples > 0) sqrt(squares / samples) else null
        squares = 0.0
        samples = 0
        if (down == null) return fixed ?: CaptureFrameInfo(shake = shake)
        CaptureFrameInfo(pitch = CaptureMotion.pitch(down.first, down.second, down.third), roll = CaptureMotion.roll(down.first, down.second, down.third), shake = shake)
    }

    private companion object {
        const val SAMPLE_US = 33_333
    }
}

/**
 * The bundled pose model on preview frames. It is optional help for the capture page, so it is
 * dropped for the rest of the session as soon as it cannot keep up. Used from one thread only.
 */
internal class LivePose(private val context: Context) {
    private var landmarker: PoseLandmarker? = null
    private var lastTimestamp = -1L
    private var total = 0.0
    private var samples = 0
    @Volatile var enabled = true
        private set
    private val power = context.getSystemService(Context.POWER_SERVICE) as? PowerManager

    /** 33 × `[x, y, visibility]`, or null when nobody was found. The time spent is returned either way. */
    fun detect(bitmap: Bitmap, time: Double): Pair<List<List<Double>>?, Double> {
        if (!enabled) return null to 0.0
        val model = landmarker ?: try {
            PoseLandmarker.createFromOptions(
                context,
                PoseLandmarker.PoseLandmarkerOptions.builder()
                    .setBaseOptions(BaseOptions.builder().setModelAssetBuffer(VideoAnalyzer.loadModel(context)).setDelegate(Delegate.CPU).build())
                    .setRunningMode(RunningMode.VIDEO).setNumPoses(1).build(),
            ).also { landmarker = it }
        } catch (error: Throwable) {
            enabled = false
            return null to 0.0
        }
        val began = SystemClock.elapsedRealtimeNanos()
        val timestamp = maxOf(lastTimestamp + 1, (time * 1000).toLong())
        lastTimestamp = timestamp
        val result = runCatching { model.detectForVideo(BitmapImageBuilder(bitmap).build(), timestamp) }.getOrNull()
        val seconds = (SystemClock.elapsedRealtimeNanos() - began) / 1e9
        // The first run also loads the model; it does not say how fast the following ones are.
        if (samples > 0) total += seconds
        samples += 1
        val hot = Build.VERSION.SDK_INT >= 29 && (power?.currentThermalStatus ?: 0) >= PowerManager.THERMAL_STATUS_SEVERE
        if (samples >= 6 && (total / (samples - 1) > BUDGET_SECONDS || hot)) close()
        val points = result?.landmarks()?.firstOrNull()?.takeIf { it.size == 33 } ?: return null to seconds
        return points.map { listOf(it.x().toDouble(), it.y().toDouble(), it.visibility().orElse(0f).toDouble()) } to seconds
    }

    fun close() {
        enabled = false
        runCatching { landmarker?.close() }
        landmarker = null
    }

    companion object {
        /** Average time per frame above which live pose is given up. */
        const val BUDGET_SECONDS = 0.15
    }
}

/** What handing preview frames to the capture page costs, for the local log and for tests. */
data class CaptureMetrics(
    val frames: Int = 0,
    /** Scaling to the hand-over size, JPEG encoding and base64. */
    val prepareSeconds: Double = 0.0,
    val poseSeconds: Double = 0.0,
    val poseFrames: Int = 0,
    val posesFound: Int = 0,
    /** From the call into the page until it returns. */
    val handOverSeconds: Double = 0.0,
    val jpegBytes: Long = 0,
    val poseDropped: Boolean = false,
    /** Replies from the page, and the time between the first and the last of them. */
    val assessments: Int = 0,
    val assessmentSpan: Double = 0.0,
) {
    val assessmentsPerSecond: Double get() = if (assessments > 1 && assessmentSpan > 0) (assessments - 1) / assessmentSpan else 0.0
    private fun average(total: Double, count: Int) = if (count > 0) total / count else 0.0
    val summary: String
        get() = String.format(
            Locale.ROOT, "frames %d, prepare %.1f ms, pose %.1f ms over %d (found %d%s), hand-over %.1f ms, jpeg %d B, %d replies at %.2f per second",
            frames, average(prepareSeconds, frames) * 1000, average(poseSeconds, poseFrames) * 1000, poseFrames, posesFound,
            if (poseDropped) ", dropped" else "", average(handOverSeconds, frames) * 1000, if (frames > 0) jpegBytes / frames else 0, assessments, assessmentsPerSecond,
        )
}

/**
 * Sits between the source and the capture page: a few frames a second are scaled down on the
 * processor, encoded, and handed over with what the sensors know. Recording is the source's own
 * business and does not pass through here.
 */
class CapturePipeline(context: Context, val motion: CaptureMotionMonitor = CaptureMotionMonitor(context)) {
    class Frame(
        val time: Double, val jpegBase64: String, val width: Int, val height: Int, val info: CaptureFrameInfo,
        val prepareSeconds: Double, val poseSeconds: Double?, val bytes: Int,
    )

    /** Called on the main thread. The second argument must be called when the frame has been dealt with. */
    var deliver: ((Frame, () -> Unit) -> Unit)? = null
    /** Upright pixel size of the source's frames, whenever it changes, on the source's thread. */
    var onFrameSize: ((Int, Int) -> Unit)? = null

    private val main = Handler(Looper.getMainLooper())
    private val pose = LivePose(context.applicationContext)
    private val lock = Any()
    private var busy = false
    private var feeding = false
    private var wantsPose = true
    private var interval = 0.5
    private var lastFed = Double.NEGATIVE_INFINITY
    private var origin: Double? = null
    private var size: Pair<Int, Int>? = null
    private var pixels = IntArray(0)
    private var bitmap: Bitmap? = null
    private val output = ByteArrayOutputStream(64 * 1024)

    val poseDropped: Boolean get() = !pose.enabled

    /** [interval] is the least time between two frames handed to the page. */
    fun setFeeding(on: Boolean, interval: Double? = null, pose: Boolean? = null) = synchronized(lock) {
        feeding = on
        if (interval != null) this.interval = interval
        if (pose != null) wantsPose = pose
    }

    /** When the source last delivered a picture, on the system's elapsed-time clock; 0 before the first. */
    @Volatile var lastFrameNanos = 0L

    /** The source's frame callback. Everything a frame needs is done before this returns, while the frame is still valid. */
    fun handle(frame: CaptureFrame) {
        lastFrameNanos = SystemClock.elapsedRealtimeNanos()
        val seconds = frame.time
        val feed: Boolean
        val wantsPose: Boolean
        val elapsed: Double
        var changed: Pair<Int, Int>? = null
        synchronized(lock) {
            if (size?.first != frame.width || size?.second != frame.height) { size = frame.width to frame.height; changed = size }
            if (origin == null) origin = seconds
            feed = feeding && !busy && seconds - lastFed >= interval - 0.005
            if (feed) { busy = true; lastFed = seconds }
            wantsPose = this.wantsPose
            elapsed = seconds - (origin ?: seconds)
        }
        changed?.let { onFrameSize?.invoke(it.first, it.second) }
        if (!feed) return
        try {
            val began = SystemClock.elapsedRealtimeNanos()
            val (width, height) = CaptureLimits.frameSize(frame.width, frame.height)
            if (pixels.size != width * height) pixels = IntArray(width * height)
            val target = bitmap?.takeIf { it.width == width && it.height == height }
                ?: Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also { bitmap?.recycle(); bitmap = it }
            frame.render(width, height, pixels)
            target.setPixels(pixels, 0, width, 0, 0, width, height)
            output.reset()
            if (!target.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, output)) { release(); return }
            val bytes = output.size()
            val base64 = Base64.encodeToString(output.toByteArray(), Base64.NO_WRAP)
            val prepared = (SystemClock.elapsedRealtimeNanos() - began) / 1e9
            var info = motion.take()
            var poseSeconds: Double? = null
            if (wantsPose && pose.enabled) {
                val (found, spent) = pose.detect(target, elapsed)
                info = info.copy(pose = found)
                poseSeconds = spent
            }
            val prepared2 = Frame(elapsed, base64, width, height, info, prepared, poseSeconds, bytes)
            main.post {
                val deliver = deliver
                if (deliver == null) release() else deliver(prepared2) { release() }
            }
        } catch (error: Throwable) {
            release()
        }
    }

    private fun release() = synchronized(lock) { busy = false }

    /** Call when the source has stopped for good. */
    fun close() {
        pose.close()
    }

    companion object {
        const val JPEG_QUALITY = 70

        /** Free space where clips are written, or null when it cannot be told. */
        fun freeBytes(directory: java.io.File): Long? = runCatching { directory.usableSpace }.getOrNull()?.takeIf { it > 0 || directory.exists() }
    }
}
