package com.betabuddy.android.review

import android.content.Context
import android.view.TextureView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.*
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LifecycleEventEffect
import com.betabuddy.android.R
import com.betabuddy.android.data.*
import com.betabuddy.android.journal.AnalysisService
import com.betabuddy.android.journal.JournalViewModel
import com.betabuddy.android.journal.OutlineButton
import com.betabuddy.android.journal.SubjectGate
import com.betabuddy.android.pose.AnalysisQuality
import com.betabuddy.android.pose.SubjectResolver
import com.betabuddy.android.pose.VideoAnalyzer
import com.betabuddy.android.ui.common.PaperPage
import com.betabuddy.android.ui.theme.BB
import kotlinx.coroutines.*
import kotlin.math.abs

/** Draft selection owns its jobs. Cancellation never writes to the saved record. */
internal class SubjectSelectionSession(
    private val context: Context, private val climb: Climb,
    private val model: JournalViewModel, private val scope: CoroutineScope,
) {
    var analysis by mutableStateOf<PoseAnalysis?>(null); private set
    var anchors by mutableStateOf(climb.subjectAnchors.orEmpty()); private set
    var busy by mutableStateOf(false); private set
    var progress by mutableDoubleStateOf(0.0); private set
    var error by mutableStateOf<String?>(null); private set
    private var job: Job? = null
    private var generation = 0
    private var interrupted = false

    fun prepare() = work {
        val cached = model.subjectAnalysis(climb.id)
        val observations = if (cached?.people?.any { it.candidates.isNotEmpty() } == true) cached else
            VideoAnalyzer(context).analyze(model.videoFile(climb),AnalysisQuality.BALANCED,collectSubjects = true) { value ->
                scope.launch { progress = value }
            }
        SubjectResolver(context).resolve(observations,anchors)
    }

    fun choose(candidate: SubjectCandidate, time: Double) = resolve(
        (anchors.filter { abs(it.time-time) > .2 } + SubjectAnchor(time,candidate)).sortedBy { it.time })
    fun automatic() = resolve(emptyList())
    private fun resolve(next: List<SubjectAnchor>) {
        val original = analysis ?: return
        work { SubjectResolver(context).resolve(original,next).also { anchors = next } }
    }
    private fun work(action: suspend () -> PoseAnalysis) {
        job?.cancel(); generation += 1
        val serial = generation
        busy = true; error = null; progress = 0.0
        job = scope.launch {
            try { val result = action(); ensureActive(); if (serial == generation) analysis = result }
            catch (cancel: CancellationException) { throw cancel }
            catch (failure: Exception) { if (serial == generation) error = failure.localizedMessage ?: failure.javaClass.simpleName }
            finally { if (serial == generation) busy = false }
        }
    }
    fun stop() { generation += 1; job?.cancel(); busy = false }
    fun suspend() { interrupted = busy; stop() }
    fun resume() { if (interrupted) { interrupted = false; prepare() } }
}

/** A light outline on the original video. Scrub and tap again to correct another moment. */
@Composable
internal fun SubjectSelectionScreen(climb: Climb, model: JournalViewModel, onDone: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val session = remember(climb.id) { SubjectSelectionSession(context,climb,model,scope) }
    val playback = remember(climb.id) { Playback(context) }
    var saving by remember { mutableStateOf(false) }
    var failure by remember { mutableStateOf<String?>(null) }
    val analysis = session.analysis
    val sample = analysis?.subjectSample(playback.time)
    val selected = analysis?.subject?.picks?.firstOrNull { sample != null && abs(it.time-sample.time) < .0001 }?.personID
    val duration = maxOf(.1,climb.analysis.duration)
    DisposableEffect(climb.id) {
        SubjectGate.set(true); playback.load(model.videoFile(climb)); session.prepare()
        onDispose {
            session.stop(); playback.release(); SubjectGate.set(false)
            if (model.climbs.value?.any { it.pendingAnalysis != null } == true) AnalysisService.start(context)
        }
    }
    LaunchedEffect(playback) { while (true) { playback.refreshTime(); delay(33) } }
    LifecycleEventEffect(Lifecycle.Event.ON_STOP) { playback.pause(); session.suspend() }
    LifecycleEventEffect(Lifecycle.Event.ON_START) { session.resume() }
    Dialog(onDismissRequest = onDone, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        PaperPage {
            Row(Modifier.fillMaxWidth(),horizontalArrangement = Arrangement.SpaceBetween) {
                OutlineButton(stringResource(R.string.journal_cancel),onClick = onDone)
                OutlineButton(stringResource(R.string.subject_save),Modifier.testTag("saveSubject"),enabled = !session.busy && !saving && analysis?.hasSelectedSubject == true) {
                    val result = session.analysis ?: return@OutlineButton
                    saving = true
                    scope.launch {
                        model.selectSubject(climb,result,session.anchors).onSuccess { onDone() }.onFailure { failure = it.localizedMessage }
                        saving = false
                    }
                }
            }
            Text(stringResource(R.string.subject_title),style = BB.Headline)
            Text(stringResource(R.string.subject_instruction),style = BB.Caption,color = BB.Muted)
            BoxWithConstraints(Modifier.fillMaxWidth().height(340.dp).background(Color.Black).testTag("subjectStage")) {
                val aspect = playback.videoAspect ?: analysis?.aspectRatio ?: climb.analysis.aspectRatio
                val (width,height) = OverlayGeometry.fit(maxWidth.value,maxHeight.value,aspect)
                Box(Modifier.align(Alignment.Center).size(width.dp,height.dp)) {
                    AndroidView(factory = { TextureView(it).also(playback::attach) },onRelease = playback::detach,modifier = Modifier.fillMaxSize())
                    sample?.candidates?.forEachIndexed { index, candidate ->
                        val bounds = candidate.bounds
                        val chosen = selected == candidate.id
                        val label = stringResource(R.string.subject_person,index+1)
                        val w = maxOf(44.0,width*bounds.width+12).dp
                        val h = maxOf(44.0,height*bounds.height+12).dp
                        Canvas(Modifier.offset((width*(bounds.x+bounds.width/2)).dp-w/2,(height*(bounds.y+bounds.height/2)).dp-h/2)
                            .size(w,h).testTag("subject-${candidate.id}")
                            .semantics { contentDescription = label; this.selected = chosen }
                            .clickable(enabled = !session.busy,role = Role.Button) { playback.pause(); session.choose(candidate,sample.time) }) {
                            drawOval(Color.Black.copy(alpha = .75f),style = Stroke(if (chosen) 5.dp.toPx() else 3.dp.toPx()))
                            drawOval(if (chosen) BB.Lime else Color.White,style = Stroke(if (chosen) 3.dp.toPx() else 1.dp.toPx()))
                        }
                    }
                }
            }
            val sliderLabel = stringResource(R.string.journal_review_position_cd)
            Slider(value = playback.time.coerceIn(0.0,duration).toFloat(),onValueChange = { playback.pause(); playback.seek(it.toDouble()) },
                valueRange = 0f..duration.toFloat(),modifier = Modifier.semantics { contentDescription = sliderLabel })
            Row(Modifier.fillMaxWidth(),horizontalArrangement = Arrangement.spacedBy(12.dp),verticalAlignment = Alignment.CenterVertically) {
                OutlineButton(stringResource(if (playback.playing) R.string.subject_pause else R.string.subject_play)) { playback.toggle(duration) }
                Text(clockText(playback.time),style = BB.Caption,modifier = Modifier.weight(1f))
                OutlineButton(stringResource(R.string.subject_auto),enabled = analysis != null && !session.busy) { playback.pause(); session.automatic() }
            }
            val status = when {
                session.busy -> R.string.subject_working
                analysis?.hasSelectedSubject != true -> R.string.subject_empty
                selected == null -> R.string.subject_missing
                session.anchors.isEmpty() -> R.string.subject_automatic
                else -> R.string.subject_manual
            }
            Text(stringResource(status),style = BB.Caption,color = BB.Muted,modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite })
            if (analysis?.hasSelectedSubject == true && analysis.subject?.picks?.any { it.personID == null } == true)
                Text(stringResource(R.string.subject_gaps),style = BB.Caption,color = BB.Muted)
            if ((analysis?.subject?.unresolved ?: 0) > 0) Text(stringResource(R.string.subject_unresolved),style = BB.Caption,color = BB.Muted)
            Text(stringResource(R.string.subject_reanalyze),style = BB.Caption,color = BB.Muted)
            (failure ?: session.error)?.let {
                Text(it,style = BB.Caption)
                OutlineButton(stringResource(R.string.subject_retry),enabled = !session.busy) { session.prepare() }
            }
        }
    }
}
