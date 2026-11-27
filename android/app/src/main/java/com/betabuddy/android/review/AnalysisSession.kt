package com.betabuddy.android.review

import android.content.Context
import android.util.Log
import android.webkit.WebView
import androidx.annotation.StringRes
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.betabuddy.android.R
import com.betabuddy.android.data.PoseFrame
import com.betabuddy.android.gym.GymMessage
import com.betabuddy.android.gym.GymPack
import com.betabuddy.android.gym.PackWebView
import com.betabuddy.android.journal.JournalViewModel
import com.betabuddy.android.pose.FrameGeometry
import com.betabuddy.android.pose.VideoAnalyzer
import java.io.File
import kotlin.coroutines.resume
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

/**
 * Hosts the pack's analysis page for one climb, for as long as the page is on screen. The app's
 * part is to prepare data, hand it to the page, keep what the page hands back and answer its
 * messages; what the page computes and shows is the pack's own.
 *
 * - The page says it is ready → a saved result is shown as it is; without one the page is given
 *   pictures taken from the clip right now and the stored poses, and told to run.
 * - A result arrives (again after every adjustment the user makes in the page) → saved over the last
 *   one, together with the plan for an automatic edit and the discipline the page judged, when it sent them.
 * - The page asks for a moment of the clip → [onSeek]. An error → the page explains; nothing is saved.
 *
 * Main thread only. [scope] must run on the main thread and end with the screen.
 */
internal class AnalysisSession(
    private val context: Context,
    private val pack: GymPack,
    private val url: String,
    internal val climbId: String,
    private val video: File,
    private val model: JournalViewModel,
    private val scope: CoroutineScope,
    private val poses: () -> List<PoseFrame>,
    private val onSeek: (Double) -> Unit,
    private val packFiles: PackWebView.Files? = null,
    /**
     * Told once how a fresh analysis ended: true when its result has been saved, false when the
     * page could not analyse or did not come up. Only the queue that runs the page out of sight listens.
     */
    private val onOutcome: ((Boolean) -> Unit)? = null,
) {
    /** Why the page is covered. The screen turns these into text in the current language. */
    sealed interface Failure {
        /** The page's script failed before it was ready. */
        data object Page : Failure
        data class Load(val description: String) : Failure
        data object RendererGone : Failure
        /** The clip gave no pictures to analyse. */
        data class Video(@StringRes val message: Int) : Failure
    }

    var webView: WebView? by mutableStateOf<WebView?>(null)
        private set

    /** Until the page's script is up. */
    var loading: Boolean by mutableStateOf(true)
        private set

    /** While pictures are being taken from the clip. */
    var preparing: Boolean by mutableStateOf(false)
        private set

    var failure: Failure? by mutableStateOf<Failure?>(null)
        private set

    private var host: PackWebView? = null
    private var ready = false
    private var feed: Job? = null
    private var subjectRevision: Int? = null
    private val saving = Mutex() // results are written in the order they arrived

    fun open() {
        if (host != null) return
        loading = true
        ready = false
        failure = null
        val created = try {
            PackWebView(context, pack, Page(), packFiles)
        } catch (error: Exception) {
            // No usable system WebView on this device.
            Log.e(TAG, "web view unavailable", error)
            failure = Failure.Load(error.message ?: error.javaClass.simpleName)
            onOutcome?.invoke(false)
            return
        }
        host = created
        webView = created.view
        created.load(url)
    }

    /** The reload button on the failure card: a new page, starting over. */
    fun reload() {
        close()
        open()
    }

    fun close() {
        feed?.cancel()
        feed = null
        preparing = false
        host?.dispose()
        host = null
        webView = null
    }

    private inner class Page : PackWebView.Listener {
        override fun onMessage(json: String?) {
            when (val message = AnalysisMessage.parse(json)) {
                AnalysisMessage.Ready -> {
                    ready = true
                    loading = false
                    feed?.cancel()
                    feed = scope.launch { start() }
                }
                // After showing a saved result the page no longer names the record; it is still this one.
                is AnalysisMessage.Result -> if (message.id.isEmpty() || message.id == climbId) {
                    val revision = subjectRevision ?: return
                    scope.launch {
                        val saved = saving.withLock { model.saveAnalysisResult(climbId, message.json, message.climber, message.capture, message.edit, message.discipline, revision) }
                            .onFailure { Log.e(TAG, "result not saved", it) }
                        onOutcome?.invoke(saved.isSuccess)
                    }
                }
                is AnalysisMessage.Seek -> onSeek(message.time)
                // The page is already showing why. Nothing is saved and the page stays open.
                is AnalysisMessage.Error -> {
                    val revision = subjectRevision ?: return
                    Log.i(TAG, "analysis page could not analyse: ${message.code}")
                    // No result, but what the page made of the climbing style is kept as a guess.
                    message.discipline?.let { guess -> scope.launch { model.setAutomaticDiscipline(climbId, guess, revision) } }
                    // Nor is a plan for an automatic edit lost for want of a result. An error without one
                    // leaves a plan stored earlier where it is.
                    message.edit?.let { plan -> scope.launch { model.saveEditPlan(climbId, plan, revision) } }
                    onOutcome?.invoke(false)
                }
                // Anything else is ignored, except a script error before the page ever came up.
                null -> if (!ready && GymMessage.parse(json) is GymMessage.Error) { failure = Failure.Page; onOutcome?.invoke(false) }
            }
        }

        override fun onDocumentLoaded(url: String?) = Unit

        override fun onLoadFailed(description: String) {
            failure = Failure.Load(description)
            onOutcome?.invoke(false)
        }

        override fun onRendererGone() {
            feed?.cancel()
            preparing = false
            host = null
            webView = null
            failure = Failure.RendererGone
            onOutcome?.invoke(false)
        }
    }

    private suspend fun start() {
        val snapshot = model.currentClimb(climbId) ?: return
        subjectRevision = snapshot.subjectRevision ?: 0
        val selected = snapshot.analysis.subject != null
        val saved = model.analysisResult(climbId)
        if (saved != null) {
            host?.evaluate(AnalysisScript.show(saved))
            return
        }
        preparing = true
        try {
            if (!withContext(Dispatchers.IO) { video.isFile }) {
                failure = Failure.Video(R.string.journal_review_video_missing)
                onOutcome?.invoke(false)
                return
            }
            val info = withContext(Dispatchers.IO) { VideoAnalyzer.probe(video) }
            val duration = info.durationSeconds
            if (!info.hasVideo || duration == null) {
                failure = Failure.Video(R.string.journal_error_invalid_video)
                onOutcome?.invoke(false)
                return
            }
            val (width, height) = FrameGeometry.displaySize(info.width, info.height, info.rotation)
            val (frameWidth, frameHeight) = AnalysisFeed.frameSize(width, height)
            // A record made with the in-app camera brings the capture page's plan; any other has none.
            host?.evaluate(AnalysisScript.begin(climbId, width, height, model.analysisClimber(), model.capturePlan(climbId), model.chosenDiscipline(climbId), duration, subjectLocked = selected))
            var index = 0
            val delivered = AnalysisFrames.extract(video, AnalysisFeed.frameTimes(duration), frameWidth, frameHeight) { time, jpeg ->
                withContext(Dispatchers.Main) { host?.evaluate(AnalysisScript.addFrame(index++, time, jpeg)) }
            }
            if (delivered < AnalysisFeed.MIN_FRAMES) {
                failure = Failure.Video(R.string.journal_error_invalid_video)
                onOutcome?.invoke(false)
                return
            }
            if (!selected) sendTrackFrames(duration, width, height)
            for (batch in snapshot.analysis.subjectPositions.orEmpty().chunked(40)) {
                if (!taken(AnalysisScript.addSubjectPositions(Json.encodeToString(batch)))) {
                    failure = Failure.Page
                    onOutcome?.invoke(false)
                    return
                }
            }
            val stored = if (selected) snapshot.analysis.frames else poses()
            for (batch in withContext(Dispatchers.Default) { AnalysisFeed.poseBatches(stored) }) host?.evaluate(AnalysisScript.addPoses(batch))
            host?.evaluate(AnalysisScript.run())
        } finally {
            preparing = false
        }
    }

    /**
     * The small, frequent pictures of the whole clip that newer packs take after the stills: upright
     * like them, made off the main thread and kept only in memory. A page without the call says so
     * on the first one, and no more are made. If they cannot be made the page simply gets none: they
     * are an extra, and the analysis goes on without them.
     */
    private suspend fun sendTrackFrames(duration: Double, width: Int, height: Int) {
        val (trackWidth, trackHeight) = AnalysisFeed.frameSize(width, height, AnalysisFeed.TRACK_SHORT_SIDE)
        try {
            AnalysisFrames.extract(video, AnalysisFeed.trackTimes(duration), trackWidth, trackHeight, AnalysisFeed.TRACK_JPEG_QUALITY) { time, jpeg ->
                val taken = withContext(Dispatchers.Main) { taken(AnalysisScript.addTrackFrame(time, jpeg)) }
                if (!taken) throw NotWanted()
            }
        } catch (error: CancellationException) {
            throw error
        } catch (error: NotWanted) {
            // An older pack: nothing more to send.
        } catch (error: Exception) {
            Log.i(TAG, "no track frames: ${error.javaClass.simpleName}")
        }
    }

    private class NotWanted : Exception()

    /** Runs [script] in the page and tells whether the page took the call. A page that is gone took nothing. */
    private suspend fun taken(script: String): Boolean = suspendCancellableCoroutine { continuation ->
        val page = host
        if (page == null) continuation.resume(false)
        else page.evaluate(script) { answer -> if (continuation.isActive) continuation.resume(!AnalysisScript.refused(answer)) }
    }

    private companion object {
        const val TAG = "analysis"   // Local logcat only.
    }
}
