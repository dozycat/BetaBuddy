package com.betabuddy.android.capture

import android.content.Context
import android.os.Build
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import android.webkit.WebView
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.LifecycleOwner
import com.betabuddy.android.data.CaptureInfo
import com.betabuddy.android.data.CaptureRoute
import com.betabuddy.android.data.CaptureSummary
import com.betabuddy.android.data.ClimbStore
import com.betabuddy.android.data.PoseAnalysis
import com.betabuddy.android.data.UNGRADED_VALUE
import com.betabuddy.android.data.newId
import com.betabuddy.android.gym.GymMessage
import com.betabuddy.android.gym.GymPack
import com.betabuddy.android.gym.PackWebView
import com.betabuddy.android.journal.AnalysisService
import com.betabuddy.android.journal.CameraGate
import com.betabuddy.android.journal.JournalGraph
import com.betabuddy.android.pose.FrameGeometry
import com.betabuddy.android.pose.VideoAnalyzer
import java.io.File
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * One visit to the capture screen: frames from the source go to the pack's capture page a few
 * times a second; the page's replies are published for the screen to draw. When recording stops
 * the clip is written into the logbook first, and only then is anything analysed. Without a
 * capture page the camera records and saves all the same. Counterpart of ios/BetaBuddy/Core/CaptureSession.swift.
 *
 * Main thread only. It has its own scope, so that closing the screen while a clip is being
 * written does not cut the save short.
 */
class CaptureSession(
    context: Context,
    pack: GymPack?,
    val source: CaptureSource,
    /** The language the page is loaded in (it only affects the page's debug text). */
    private val language: String,
    /** Title of a record saved without one, in the language shown at save time. */
    private val defaultTitle: () -> String,
    private val packFiles: PackWebView.Files? = null,
    private val store: suspend () -> ClimbStore = { withContext(Dispatchers.IO) { JournalGraph.store(context) } },
    private val staging: File = JournalGraph.stagingDirectory(context),
) {
    /** The capture page. UNAVAILABLE: this build's pack has none (or there is no pack). */
    enum class Page { UNAVAILABLE, LOADING, READY, FAILED }
    enum class Camera { STARTING, RUNNING, DENIED, UNAVAILABLE }
    sealed interface Phase {
        data object Framing : Phase
        data object Recording : Phase
        data object Saving : Phase
        /** The record is in the logbook. */
        data class Saved(val id: String) : Phase
        /** The clip was recorded but could not be written into the logbook; it is still held. */
        data class SaveFailed(val message: String) : Phase
    }
    /** Something the user should be told once, as a type so the screen picks the words. */
    enum class Notice { TOO_SHORT, LOW_STORAGE, RECORDER_FAILED }

    private val context = context.applicationContext
    private val pack: GymPack? = pack?.takeIf { it.supportsCapture }
    /** The pack can analyse the clip afterwards, so saving asks for that too. */
    private val analysisAvailable = pack?.analysisUrl(language) != null
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val pipeline = CapturePipeline(this.context)

    var page: Page by mutableStateOf(if (this.pack == null) Page.UNAVAILABLE else Page.LOADING)
        private set
    var camera: Camera by mutableStateOf(Camera.STARTING)
        private set
    var phase: Phase by mutableStateOf<Phase>(Phase.Framing)
        private set
    var assessment: CaptureAssessment? by mutableStateOf<CaptureAssessment?>(null)
        private set
    /** The colour the user wants to climb, as one of the page's palette keys. */
    var color: String? by mutableStateOf<String?>(null)
        private set
    /** Seconds written so far while recording. */
    var recordedSeconds: Double by mutableStateOf(0.0)
        private set
    var stopReason: CaptureStop? by mutableStateOf<CaptureStop?>(null)
        private set
    var summary: CaptureSummary? by mutableStateOf<CaptureSummary?>(null)
        private set
    /** Upright pixel size of the source's frames; the picture on screen has the same shape. */
    var frameSize: Pair<Int, Int> by mutableStateOf(9 to 16)
        private set
    var webView: WebView? by mutableStateOf<WebView?>(null)
        private set
    var notice: Notice? by mutableStateOf<Notice?>(null)
    /** The last tap on the picture found no hold there. Cleared by itself after a moment. */
    var tapMissed: Boolean by mutableStateOf(false)
        private set
    /** Problems the page recognised and the user said are not the one they are climbing, by id. */
    var rejectedRoutes: Set<String> by mutableStateOf(emptySet())
        private set
    var gymId: String? by mutableStateOf<String?>(null)
        private set
    var gymName: String by mutableStateOf("")
        private set
    var grade: String by mutableStateOf(UNGRADED_VALUE)

    private var host: PackWebView? = null
    private var started = false
    private var captureId = newId()
    private var begun: Begun? = null
    private var pageVersion: String? = null
    private var plan: String? = null
    private var recording: ClipRecording? = null
    private var recordingStartedMillis = 0L
    private var recordingFile: File? = null
    private var held: HeldClip? = null
    private var watch: Job? = null
    private var missed: Job? = null
    private var firstAssessment = 0L
    /** When the page last answered (or was last asked to start over). A page that has gone quiet must not leave an old judgement on the screen. */
    private var lastReply = 0L
    private var gymChosen = false

    var metrics = CaptureMetrics()
        private set
    var quietLimitSeconds = 5.0
    /** Seconds between frames handed to the page while framing, and while recording. */
    var framingInterval = 0.5
    var recordingInterval = 1.0
    var maxSeconds = CaptureLimits.MAX_SECONDS
    var watchIntervalMillis = 3_000L
    /** Free space in bytes where clips are written; replaced in tests. */
    var freeBytes: () -> Long? = { CapturePipeline.freeBytes(staging) }
    /** Lets a test see how the phone is held without sensors. */
    val motion: CaptureMotionMonitor get() = pipeline.motion

    private data class Begun(val id: String, val width: Int, val height: Int, val venue: String?)
    private class HeldClip(val file: File, val startedMillis: Long)

    // Lifetime

    /** Starts the page and the pictures. [owner] is what the camera follows: it stops when the owner does. */
    fun start(owner: LifecycleOwner) {
        if (started) return
        started = true
        // Pose recognition of earlier clips waits while the camera is in use.
        CameraGate.set(true)
        pipeline.deliver = { frame, done -> hand(frame, done) }
        pipeline.onFrameSize = { width, height ->
            scope.launch {
                if (!started) return@launch
                frameSize = width to height
                if (camera == Camera.STARTING) camera = Camera.RUNNING   // Only now is there something to record.
            }
        }
        source.onFrame = pipeline::handle
        source.onInterruption = { stop(CaptureStop.INTERRUPTED) }
        loadPage()
        scope.launch {
            // The gym of the last record is where the next one most likely is; it is a default, never a choice made for the user.
            runCatching { store().lastGym }.getOrNull()?.let { (id, name) -> if (!gymChosen) { gymId = id; gymName = name } }
            try {
                staging.mkdirs()
                source.start(owner)
                if (!started) { source.stop(); return@launch }
                pipeline.motion.start()
                pipeline.setFeeding(page != Page.UNAVAILABLE, framingInterval)
            } catch (error: CaptureSourceException) {
                camera = if (error.reason == CaptureSourceException.Reason.DENIED) Camera.DENIED else Camera.UNAVAILABLE
            } catch (error: Exception) {
                Log.e(TAG, "source did not start", error)
                camera = Camera.UNAVAILABLE
            }
        }
    }

    /** After the user has answered the system's question about the camera: try the source again. */
    fun retry(owner: LifecycleOwner) {
        if (!started || (camera != Camera.DENIED && camera != Camera.UNAVAILABLE)) return
        camera = Camera.STARTING
        scope.launch {
            try {
                source.start(owner)
                pipeline.motion.start()
                pipeline.setFeeding(page != Page.UNAVAILABLE, framingInterval)
            } catch (error: CaptureSourceException) {
                camera = if (error.reason == CaptureSourceException.Reason.DENIED) Camera.DENIED else Camera.UNAVAILABLE
            } catch (error: Exception) {
                camera = Camera.UNAVAILABLE
            }
        }
    }

    /** The app left the foreground: a running recording is closed and saved with what it has. */
    fun onBackground() = stop(CaptureStop.INTERRUPTED)

    /** Leaves the screen. A recording still running is closed and saved first. */
    fun close() {
        if (!started) return
        if (phase == Phase.Recording) stop(CaptureStop.USER)
        started = false
        watch?.cancel()
        pipeline.setFeeding(false)
        pipeline.motion.stop()
        host?.dispose()
        host = null
        webView = null
        Log.i(TAG, "capture closed: ${metrics.copy(poseDropped = pipeline.poseDropped).summary}")
        // A clip still being closed is written first; the source and the gate are let go after it.
        if (phase == Phase.Saving) return
        release()
    }

    private fun release() {
        source.onFrame = null
        source.onInterruption = null
        source.stop()
        pipeline.close()
        held?.file?.delete()
        held = null
        CameraGate.set(false)
        AnalysisService.start(context)
    }

    // Page

    private fun loadPage() {
        val pack = pack ?: return
        val url = pack.captureUrl(language) ?: return
        page = Page.LOADING
        val created = try {
            PackWebView(context, pack, Listener(), packFiles)
        } catch (error: Exception) {
            Log.e(TAG, "web view unavailable", error)   // No usable system WebView on this device.
            page = Page.FAILED
            return
        }
        host = created
        webView = created.view
        created.load(url)
    }

    private val venue: String? get() = pack?.manifest?.venueId(gymId, gymName)

    /** The user chose a gym. The page is told about it only when it is one of the pack's own; another gym starts its judgement over. */
    fun selectGym(id: String?, name: String) {
        gymChosen = true
        gymId = id
        gymName = name
        val begun = begun ?: return
        if (phase == Phase.Framing && begun.venue != venue) this.begun = null
    }

    private fun hand(frame: CapturePipeline.Frame, done: () -> Unit) {
        val host = host
        if (!started || host == null || (page != Page.READY && page != Page.FAILED)) { done(); return }
        val now = SystemClock.elapsedRealtime()
        val current = begun
        if (current == null || current.id != captureId || current.width != frame.width || current.height != frame.height) {
            if (phase != Phase.Framing) { done(); return }   // A locked judgement is not started over.
            val venue = venue
            begun = Begun(captureId, frame.width, frame.height, venue)
            rejectedRoutes = emptySet()   // The page forgets what was rejected when it starts over; so does the app.
            assessment = null
            lastReply = now
            host.evaluate(CaptureScript.begin(captureId, frame.width, frame.height, source.horizontalFieldOfView, venue))
            color?.let { host.evaluate(CaptureScript.setColor(it)) }
        }
        val began = SystemClock.elapsedRealtimeNanos()
        host.evaluate(CaptureScript.addFrame(frame.time, frame.jpegBase64, frame.info)) {
            // The call has returned: the page has the picture (it answers with a message later).
            val found = frame.info.pose != null
            metrics = metrics.copy(
                frames = metrics.frames + 1, handOverSeconds = metrics.handOverSeconds + (SystemClock.elapsedRealtimeNanos() - began) / 1e9,
                prepareSeconds = metrics.prepareSeconds + frame.prepareSeconds, jpegBytes = metrics.jpegBytes + frame.bytes,
                poseFrames = metrics.poseFrames + if (frame.poseSeconds != null) 1 else 0, poseSeconds = metrics.poseSeconds + (frame.poseSeconds ?: 0.0),
                posesFound = metrics.posesFound + if (found) 1 else 0, poseDropped = pipeline.poseDropped,
            )
            done()
        }
        if (phase == Phase.Framing && page == Page.READY && now - lastReply > quietLimitSeconds * 1000) {
            Log.e(TAG, "capture page has not answered for ${quietLimitSeconds.toInt()} s")
            page = Page.FAILED
            assessment = null
        }
    }

    private inner class Listener : PackWebView.Listener {
        override fun onMessage(json: String?) {
            when (val message = CaptureMessage.parse(json)) {
                is CaptureMessage.Ready -> {
                    pageVersion = message.packVersion.ifEmpty { pack?.manifest?.packVersion }
                    begun = null
                    page = Page.READY
                }
                is CaptureMessage.Assessment -> {
                    if (phase != Phase.Framing || begun?.id != captureId) return
                    if (page == Page.FAILED) page = Page.READY
                    val now = SystemClock.elapsedRealtime()
                    lastReply = now
                    assessment = message.assessment
                    if (firstAssessment == 0L) firstAssessment = now
                    metrics = metrics.copy(assessments = metrics.assessments + 1, assessmentSpan = (now - firstAssessment) / 1000.0)
                }
                is CaptureMessage.Locked -> {
                    if (message.id.isNotEmpty() && message.id != captureId) return
                    summary = message.summary
                    message.plan?.let { plan = it }
                    // The page may answer, or answer again with a fuller plan, after the clip is in the logbook.
                    val saved = (phase as? Phase.Saved)?.id
                    val late = message.plan
                    if (saved != null && late != null) scope.launch {
                        runCatching { val store = store(); withContext(Dispatchers.IO) { store.saveCapturePlan(saved, late, message.summary) } }
                            .onFailure { Log.e(TAG, "late plan not kept", it) }
                    }
                }
                // Some of these are passing (one frame it could not read, a venue file it could not get) and
                // the page carries on. Whether it still answers is what decides what the screen says, see `hand`.
                is CaptureMessage.Failed -> Log.i(TAG, "capture page reported: ${message.code}")
                is CaptureMessage.PageError -> {
                    Log.e(TAG, "capture page error (${if (page == Page.LOADING) "before" else "after"} ready): ${message.message}")
                    if (page == Page.LOADING) page = Page.FAILED
                }
                null -> if (page == Page.LOADING && GymMessage.parse(json) is GymMessage.Error) page = Page.FAILED
            }
        }

        override fun onDocumentLoaded(url: String?) = Unit
        override fun onLoadFailed(description: String) = pageFailed(description)
        override fun onRendererGone() {
            host = null
            webView = null
            pageFailed("renderer gone")
        }
    }

    /** The page is gone for this visit; frames are no longer prepared for it. */
    private fun pageFailed(text: String) {
        Log.e(TAG, "capture page: $text")
        page = Page.FAILED
        assessment = null
        pipeline.setFeeding(false)
    }

    // What the user chooses

    val palette: List<CaptureAssessment.Swatch> get() = assessment?.palette.orEmpty()

    /** Null, or the colour already chosen, clears the choice. */
    fun selectColor(key: String?) {
        if (phase != Phase.Framing) return
        color = if (key == color) null else key
        host?.evaluate(CaptureScript.setColor(color))
    }

    /**
     * The user says what the hold at this spot of the picture is. The page finds the hold; what
     * changed shows with its next reply. When it finds none there, the screen says so for a moment.
     */
    fun correct(x: Double, y: Double, role: CaptureRole) {
        if (phase != Phase.Framing || page != Page.READY || color == null) return
        host?.evaluate(CaptureScript.tap(x, y, role)) { result ->
            // An empty answer is "no hold here"; `false` means the page has no `tap`.
            missed?.cancel()
            if (CaptureScript.tapAnswer(result)?.isEmpty() == true) {
                tapMissed = true
                missed = scope.launch { delay(3_000); tapMissed = false }
            } else {
                tapMissed = false
            }
        }
    }

    fun resetCorrections() {
        if (phase == Phase.Framing && page == Page.READY) host?.evaluate(CaptureScript.resetTaps())
    }

    /** The problem the page recognised, unless the user has said it is not that one. */
    val recognised: CaptureRoute? get() = (summary?.route ?: assessment?.route)?.takeIf { it.id !in rejectedRoutes }

    /**
     * "Not this one". A newer page is told, and comes back with its next candidate or with none; an
     * older one has no call for it and keeps naming the same problem, which is then simply not
     * shown again for this attempt. Either way a rejected problem is not stored with the record.
     */
    fun rejectRoute() {
        val route = recognised ?: return
        rejectedRoutes = rejectedRoutes + route.id
        host?.evaluate(CaptureScript.rejectRoute(route.id))
    }

    /** The hint blocks: the user is asked before recording from this position. */
    val needsConfirmation: Boolean get() = page == Page.READY && assessment?.ok == false

    // Recording

    fun startRecording() {
        if (phase != Phase.Framing || camera != Camera.RUNNING) return
        val free = freeBytes()
        if (free != null && free < CaptureLimits.STARTING_FREE_BYTES) { notice = Notice.LOW_STORAGE; return }
        staging.mkdirs()
        val file = File(staging, "betabuddy-capture-${newId()}.mp4")
        val began = System.currentTimeMillis()
        val started = source.startRecording(file, maxSeconds, onProgress = { seconds -> if (phase == Phase.Recording) recordedSeconds = seconds }, onFinished = ::finished)
        if (started == null) { notice = Notice.RECORDER_FAILED; return }
        recording = started
        recordingFile = file
        recordingStartedMillis = began
        plan = null; summary = null; stopReason = null; notice = null
        recordedSeconds = 0.0
        phase = Phase.Recording
        // The judgement of this moment is what the analysis will be given. Frames keep going to the page at a slower pace.
        pipeline.setFeeding(page == Page.READY, recordingInterval)
        if (page == Page.READY) host?.evaluate(CaptureScript.lock())
        watch?.cancel()
        val power = context.getSystemService(Context.POWER_SERVICE) as? PowerManager
        watch = scope.launch {
            while (isActive) {
                delay(watchIntervalMillis)
                if (phase != Phase.Recording) return@launch
                val left = freeBytes()
                if (left != null && left < CaptureLimits.STOPPING_FREE_BYTES) stop(CaptureStop.STORAGE)
                else if (Build.VERSION.SDK_INT >= 29 && (power?.currentThermalStatus ?: 0) >= PowerManager.THERMAL_STATUS_CRITICAL) stop(CaptureStop.OVERHEATED)
            }
        }
    }

    /**
     * Closes the clip with what has been recorded; it is then written into the logbook, and only
     * after that may analysis start. Safe to call from several places: only the first call acts.
     */
    fun stop(reason: CaptureStop) {
        if (phase != Phase.Recording) return
        phase = Phase.Saving
        stopReason = reason
        watch?.cancel()
        pipeline.setFeeding(false)
        recording?.stop()
    }

    /** The source has closed the clip: because it was told to, or by itself. */
    private fun finished(result: ClipResult) {
        if (phase == Phase.Recording) {
            // Nobody asked: the limit, a full disk, or the camera went away.
            phase = Phase.Saving
            watch?.cancel()
            pipeline.setFeeding(false)
            stopReason = when (result.end) {
                ClipEnd.LIMIT -> CaptureStop.LIMIT
                ClipEnd.STORAGE -> CaptureStop.STORAGE
                else -> CaptureStop.INTERRUPTED
            }
        } else if (result.end == ClipEnd.LIMIT) {
            stopReason = CaptureStop.LIMIT
        }
        if (phase != Phase.Saving) return
        recording = null
        Log.i(TAG, "recorded ${"%.1f".format(result.seconds)} s, end ${result.end}, stop $stopReason")
        val file = result.file
        if (file == null || result.seconds < CaptureLimits.MIN_SECONDS) {
            (file ?: recordingFile)?.delete()
            notice = if (file == null && result.end == ClipEnd.FAILED) Notice.RECORDER_FAILED else Notice.TOO_SHORT
            if (started) startFraming() else release()
            return
        }
        held = HeldClip(file, recordingStartedMillis)
        save()
    }

    /** Writes the held clip into the logbook. Also the retry after a failed save. */
    fun save() {
        val clip = held ?: return
        phase = Phase.Saving
        scope.launch {
            try {
                val analysis = withContext(Dispatchers.IO) {
                    val info = VideoAnalyzer.probe(clip.file)
                    val duration = info.durationSeconds
                    if (!info.hasVideo || duration == null) throw IllegalStateException("The recording cannot be read")
                    val (width, height) = FrameGeometry.displaySize(info.width, info.height, info.rotation)
                    PoseAnalysis(emptyList(), 0, duration, width.toDouble() / height)
                }
                val thumbnail = VideoAnalyzer(context).thumbnail(clip.file)
                val swatch = assessment?.palette?.firstOrNull { it.key == color }
                val route = recognised
                val info = CaptureInfo(
                    colorKey = color, colorRGB = swatch?.rgb ?: summary?.rgb?.takeIf { summary?.color == color }, summary = summary,
                    packVersion = if (pack == null) null else pageVersion, analysisQueued = if (analysisAvailable) true else null,
                    routeId = route?.id, routeName = route?.displayName(language)?.takeIf { it != route.id },
                )
                // One write: the clip, its thumbnail, the plan, what was chosen and the request for
                // analysis. Nothing is analysed until this has succeeded.
                val store = store()
                val climb = withContext(Dispatchers.IO) {
                    store.save(
                        title = "", defaultTitle = defaultTitle(), grade = grade, sent = false, note = "", stagedVideo = clip.file, thumbnail = thumbnail, analysis = analysis,
                        gymId = gymId, gymName = gymName, climbedAtMillis = clip.startedMillis, pendingAnalysis = PENDING_QUALITY, capture = info, capturePlan = plan,
                    )
                }
                held = null
                phase = Phase.Saved(climb.id)
                if (started) {
                    CameraGate.set(false)
                    AnalysisService.start(context)
                } else {
                    release()
                }
            } catch (error: Exception) {
                Log.e(TAG, "clip not saved", error)
                phase = Phase.SaveFailed(error.localizedMessage ?: error.javaClass.simpleName)
                // The screen is gone and nobody can retry: the clip stays in the staging directory until the next launch clears it.
                if (!started) { held = null; release() }
            }
        }
    }

    /** Gives up a clip that could not be saved. */
    fun discardHeldClip() {
        held?.file?.delete()
        held = null
        startFraming()
    }

    /** Back to framing for another attempt, with the same gym, grade and colour. */
    fun frameAgain() {
        if (phase == Phase.Recording || phase == Phase.Saving) return
        startFraming()
    }

    private fun startFraming() {
        CameraGate.set(true)
        captureId = newId()
        begun = null
        assessment = null
        plan = null; summary = null
        recordedSeconds = 0.0
        rejectedRoutes = emptySet()
        phase = Phase.Framing
        pipeline.setFeeding(page != Page.UNAVAILABLE && started, framingInterval)
    }

    private companion object {
        const val TAG = "capture"   // Local logcat only.
        const val PENDING_QUALITY = "balanced"
    }
}
