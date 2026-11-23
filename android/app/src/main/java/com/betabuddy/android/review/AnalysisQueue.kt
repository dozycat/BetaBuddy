package com.betabuddy.android.review

import android.content.Context
import android.view.ViewGroup
import android.webkit.WebView
import android.widget.FrameLayout
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.betabuddy.android.R
import com.betabuddy.android.data.Climb
import com.betabuddy.android.gym.GymPack
import com.betabuddy.android.gym.PackWebView
import com.betabuddy.android.i18n.AppLanguage
import com.betabuddy.android.journal.AnalysisQueuePolicy
import com.betabuddy.android.journal.CameraGate
import com.betabuddy.android.journal.JournalViewModel
import com.betabuddy.android.journal.UiMessage
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/**
 * Runs the pack's analysis for records that asked for it when they were saved: the ones made with
 * the in-app camera. Pose recognition comes first, in its own service; when a record has its
 * poses, the analysis page is loaded out of sight, given the clip and the capture plan, and its
 * result is kept exactly as when the user opens the page. One record at a time, never while the
 * camera is in use or the app is in the background. The request is part of the record, so it is
 * picked up again after the app has been closed. Counterpart of ios/BetaBuddy/Core/AnalysisQueue.swift.
 *
 * Main thread only.
 */
internal class AnalysisQueue(
    private val context: Context,
    private val model: JournalViewModel,
    private val pack: GymPack?,
    private val url: String?,
    private val scope: CoroutineScope,
    private val packFiles: PackWebView.Files? = null,
) {
    /** The page being used, for the host to keep in the window where nobody sees it. */
    var webView: WebView? by mutableStateOf<WebView?>(null)
        private set
    var running: String? by mutableStateOf<String?>(null)
        private set

    private var session: AnalysisSession? = null
    private var timeout: Job? = null
    private var mirror: Job? = null
    private val settled = mutableSetOf<String>()
    /** Long enough for the longest clip on a slow phone; a page that takes longer is given up on. */
    var patienceMillis = 600_000L

    /** Looks at what there is to do. Called whenever the records, the results, the camera or the foreground change. */
    fun advance(climbs: List<Climb>, results: Set<String>, paused: Boolean) {
        if (paused) { halt(); return }
        if (session != null) return
        when (val step = AnalysisQueuePolicy.next(climbs, results, settled, paused = false, hasPage = pack != null && url != null)) {
            AnalysisQueuePolicy.Step.Wait -> Unit
            is AnalysisQueuePolicy.Step.Settle -> {
                settled += step.id
                scope.launch { model.settleQueuedAnalysis(step.id, null) }
            }
            is AnalysisQueuePolicy.Step.Run -> run(step.climb)
        }
    }

    /** The user opened the analysis page for this record themselves; that visit does the work. */
    fun cancel(id: String) {
        if (running == id) halt()
        settled += id
        scope.launch { model.settleQueuedAnalysis(id, null) }
    }

    fun halt() {
        timeout?.cancel(); timeout = null
        mirror?.cancel(); mirror = null
        session?.close()
        session = null
        webView = null
        running = null
    }

    private fun run(climb: Climb) {
        val pack = pack ?: return
        val url = url ?: return
        val id = climb.id
        val created = AnalysisSession(
            context = context, pack = pack, url = url, climbId = id, video = model.videoFile(climb), model = model, scope = scope,
            poses = { climb.analysis.frames }, onSeek = {}, packFiles = packFiles, onOutcome = { succeeded -> finish(id, succeeded) },
        )
        session = created
        running = id
        created.open()
        webView = created.webView
        timeout = scope.launch {
            delay(patienceMillis)
            finish(id, false)
        }
    }

    private fun finish(id: String, succeeded: Boolean) {
        if (running != id) return
        halt()
        settled += id
        scope.launch {
            model.settleQueuedAnalysis(id, UiMessage(if (succeeded) R.string.capture_analysis_done else R.string.capture_analysis_failed))
        }
    }
}

/**
 * A scope that is on the main thread whatever the composition runs on, and ends with the caller.
 * The pages' web views may only be touched from the main thread.
 */
@Composable
internal fun rememberMainScope(): CoroutineScope {
    val scope = remember { CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate) }
    DisposableEffect(scope) { onDispose { scope.cancel() } }
    return scope
}

/**
 * Keeps the queue going while the app's home is on screen, and its page in the window, behind
 * everything and at full size, so its scripts run as on a visible page.
 */
@Composable
internal fun AnalysisQueueHost(model: JournalViewModel, pack: GymPack?, packFiles: PackWebView.Files? = null, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val scope = rememberMainScope()
    val url = pack?.analysisUrl(AppLanguage.current())
    val queue = remember(pack, url) { AnalysisQueue(context, model, pack, url, scope, packFiles) }
    val climbs by model.climbs.collectAsState()
    val results by model.analysisResults.collectAsState()
    val cameraOpen by CameraGate.open.collectAsState()
    val owner = LocalLifecycleOwner.current
    var foreground by remember { mutableStateOf(owner.lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED)) }
    DisposableEffect(owner, queue) {
        val observer = LifecycleEventObserver { _, _ -> foreground = owner.lifecycle.currentState.isAtLeast(Lifecycle.State.STARTED) }
        owner.lifecycle.addObserver(observer)
        onDispose { owner.lifecycle.removeObserver(observer); queue.halt() }
    }
    // `running` is read so that the next record is looked at when one is done.
    LaunchedEffect(queue, climbs, results, cameraOpen, foreground, queue.running) {
        climbs?.let { queue.advance(it, results, paused = cameraOpen || !foreground) }
    }
    queue.webView?.let { webView ->
        key(webView) {
            AndroidView(
                factory = { viewContext ->
                    FrameLayout(viewContext).apply {
                        (webView.parent as? ViewGroup)?.removeView(webView)
                        addView(webView, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
                        // Nobody sees or touches this page.
                        importantForAccessibility = android.view.View.IMPORTANT_FOR_ACCESSIBILITY_NO_HIDE_DESCENDANTS
                    }
                },
                onRelease = { container -> container.removeAllViews() },
                modifier = modifier.fillMaxSize().clearAndSetSemantics { },
            )
        }
    }
}
