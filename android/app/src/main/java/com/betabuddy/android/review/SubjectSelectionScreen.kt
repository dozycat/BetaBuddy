package com.betabuddy.android.review

import android.content.Context
import android.view.TextureView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.*
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.compose.ui.zIndex
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LifecycleEventEffect
import com.betabuddy.android.R
import com.betabuddy.android.data.*
import com.betabuddy.android.journal.AnalysisService
import com.betabuddy.android.journal.JournalViewModel
import com.betabuddy.android.journal.OutlineButton
import com.betabuddy.android.journal.SubjectGate
import com.betabuddy.android.pose.AnalysisQuality
import com.betabuddy.android.pose.AnalysisError
import com.betabuddy.android.pose.AnalysisException
import com.betabuddy.android.pose.SubjectRegionAnalyzer
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
    private var lastAction: (suspend () -> Pair<PoseAnalysis,List<SubjectAnchor>>)? = null
    private var resumeAction: (suspend () -> Pair<PoseAnalysis,List<SubjectAnchor>>)? = null

    fun prepare() = work {
        val serial = generation
        val cached = model.subjectAnalysis(climb.id)
        val usedCache=cached?.people?.any { it.candidates.isNotEmpty() } == true
        val observations = if (usedCache) requireNotNull(cached) else
            try { SubjectResolver(context).analyze(model.videoFile(climb),AnalysisQuality.BALANCED,anchors=anchors) { value ->
                scope.launch { if (serial == generation) progress = value }
            } } catch (failure: AnalysisException) {
                if (failure.error != AnalysisError.NO_POSE) throw failure
                climb.analysis.copy(people = emptyList(),peopleReference = null)
            }
        val resolver=SubjectResolver(context)
        var selected=resolver.resolve(observations,anchors)
        if(usedCache && anchors.isEmpty() && !selected.hasSelectedSubject) selected=resolver.recover(model.videoFile(climb),selected) { value ->
            scope.launch { if(serial==generation) progress=value }
        }
        selected to anchors
    }

    fun choose(candidate: SubjectCandidate, time: Double) = resolve(
        (anchors.filter { abs(it.time-time) > .2 } + SubjectAnchor(time,candidate)).sortedBy { it.time })
    fun automatic() = resolve(emptyList())
    private fun resolve(next: List<SubjectAnchor>) {
        val original = analysis ?: return
        work {
            val serial=generation;val resolver=SubjectResolver(context)
            var selected=resolver.resolve(original,next)
            if(next.isEmpty() && !selected.hasSelectedSubject) selected=resolver.recover(model.videoFile(climb),selected) { value ->
                scope.launch { if(serial==generation) progress=value }
            }
            selected to next
        }
    }
    fun refine(bounds: SubjectBounds,time: Double) {
        val original = analysis ?: climb.analysis
        work {
            val refined = SubjectRegionAnalyzer(context).refine(model.videoFile(climb),original,bounds,time) { value ->
                withContext(Dispatchers.Main) { ensureActive(); progress = value }
            }
            val next = (anchors.filter { abs(it.time-refined.anchor.time) > .2 }+refined.anchor).sortedBy { it.time }
            SubjectResolver(context).resolve(refined.analysis,next) to next
        }
    }
    private fun work(action: suspend () -> Pair<PoseAnalysis,List<SubjectAnchor>>) {
        stop(); lastAction = action
        val serial = generation
        busy = true; error = null; progress = 0.0
        job = scope.launch {
            try { val result = action(); ensureActive(); if (serial == generation) { analysis = result.first; anchors = result.second } }
            catch (cancel: CancellationException) { throw cancel }
            catch (failure: Exception) { if (serial == generation) error = failure.localizedMessage ?: failure.javaClass.simpleName }
            finally { if (serial == generation) busy = false }
        }
    }
    fun retry() { lastAction?.let(::work) }
    fun stop() { generation += 1; job?.cancel(); busy = false; resumeAction = null }
    fun suspend() { if (busy) { val action = lastAction; stop(); resumeAction = action } }
    fun resume() { resumeAction?.let(::work) }
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
    var drawing by remember { mutableStateOf(false) }
    var adjusting by remember { mutableStateOf(false) }
    var draft by remember { mutableStateOf(SubjectBounds(.35,.25,.30,.50)) }
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
        BoxWithConstraints(Modifier.fillMaxSize().background(BB.Paper).windowInsetsPadding(WindowInsets.safeDrawing)) {
            val videoHeight = minOf(340.dp,maxHeight*(if (adjusting) .34f else .43f))
            Column {
                Row(Modifier.fillMaxWidth().padding(horizontal=24.dp,vertical=12.dp),horizontalArrangement = Arrangement.SpaceBetween) {
                    OutlineButton(stringResource(R.string.journal_cancel),onClick = onDone)
                    OutlineButton(stringResource(R.string.subject_save),Modifier.testTag("saveSubject"),enabled = !drawing && !session.busy && !saving && analysis?.hasSelectedSubject == true) {
                        val result = session.analysis ?: return@OutlineButton
                        saving = true
                        scope.launch {
                            model.selectSubject(climb,result,session.anchors).onSuccess { onDone() }.onFailure { failure = it.localizedMessage }
                            saving = false
                        }
                    }
                }
                Text(stringResource(R.string.subject_title),style = BB.Headline,modifier=Modifier.padding(horizontal=24.dp,vertical=8.dp))
                BoxWithConstraints(Modifier.padding(horizontal=24.dp).fillMaxWidth().height(videoHeight).background(Color.Black).testTag("subjectStage")) {
                    val aspect = playback.videoAspect ?: analysis?.aspectRatio ?: climb.analysis.aspectRatio
                    val (width,height) = OverlayGeometry.fit(maxWidth.value,maxHeight.value,aspect)
                    Box(Modifier.align(Alignment.Center).size(width.dp,height.dp)) {
                        AndroidView(factory = { TextureView(it).also(playback::attach) },onRelease = playback::detach,modifier = Modifier.fillMaxSize())
                        (if (drawing) null else sample)?.candidates?.forEachIndexed { index, candidate ->
                            val bounds = candidate.bounds
                            val chosen = selected == candidate.id
                            val label = stringResource(R.string.subject_person,index+1)
                            val w = maxOf(44.0,width*bounds.width+12).dp
                            val h = maxOf(44.0,height*bounds.height+12).dp
                            Canvas(Modifier.offset((width*(bounds.x+bounds.width/2)).dp-w/2,(height*(bounds.y+bounds.height/2)).dp-h/2)
                                .size(w,h).testTag("subject-${candidate.id}")
                                .zIndex(if (chosen) 1f else 0f)
                                .semantics { contentDescription = label; this.selected = chosen }
                                .clickable(enabled = !session.busy,role = Role.Button) { playback.pause(); session.choose(candidate,candidate.pose.time) }) {
                                if (selected == null || chosen) {
                                    drawOval(Color.Black.copy(alpha = .75f),style = Stroke(if (chosen) 5.dp.toPx() else 3.dp.toPx()))
                                    drawOval(if (chosen) BB.Lime else Color.White,style = Stroke(if (chosen) 3.dp.toPx() else 1.dp.toPx()))
                                }
                            }
                        }
                        if (drawing) {
                            val regionLabel = stringResource(R.string.subject_draw_instruction)
                            Canvas(Modifier.fillMaxSize().testTag("subjectRegionCanvas")
                                .semantics { contentDescription = regionLabel }
                                .pointerInput(Unit) {
                                    var start = Offset.Zero
                                    detectDragGestures(orientationLock = null,onDragStart = { down, _, _ -> start = down.position; playback.pause() }) { change, _ ->
                                        change.consume()
                                        val x = (minOf(start.x,change.position.x)/size.width).coerceIn(0f,1f).toDouble()
                                        val y = (minOf(start.y,change.position.y)/size.height).coerceIn(0f,1f).toDouble()
                                        val right = (maxOf(start.x,change.position.x)/size.width).coerceIn(0f,1f).toDouble()
                                        val bottom = (maxOf(start.y,change.position.y)/size.height).coerceIn(0f,1f).toDouble()
                                        if (right-x >= .02 && bottom-y >= .02) draft = SubjectBounds(x,y,right-x,bottom-y)
                                    }
                                }) {
                                val offset = Offset((draft.x*size.width).toFloat(),(draft.y*size.height).toFloat())
                                val ellipse = androidx.compose.ui.geometry.Size((draft.width*size.width).toFloat(),(draft.height*size.height).toFloat())
                                drawOval(Color.Black.copy(alpha=.75f),topLeft=offset,size=ellipse,style=Stroke(5.dp.toPx()))
                                drawOval(BB.Lime,topLeft=offset,size=ellipse,style=Stroke(3.dp.toPx()))
                            }
                        }
                    }
                }
                PaperPage(Modifier.weight(1f),insets=WindowInsets(0,0,0,0),spacing=16.dp) {
                    if (!drawing) Text(stringResource(R.string.subject_instruction),style = BB.Caption,color = BB.Muted)
                    OutlineButton(stringResource(if (drawing) R.string.subject_draw_cancel else R.string.subject_draw),Modifier.testTag("drawSubjectRegion"),enabled = !saving) {
                        playback.pause(); session.stop(); drawing = !drawing; adjusting = false
                        if (!drawing && session.analysis == null) session.prepare()
                    }
                    if (drawing) {
                        OutlineButton(stringResource(R.string.subject_draw_adjust),Modifier.testTag("adjustSubjectRegion")) { adjusting = !adjusting }
                        if (adjusting) {
                            @Composable fun adjustment(label: Int,value: Double,range: ClosedFloatingPointRange<Float>,change: (Float) -> Unit) {
                                val text = stringResource(label)
                                Text(text,style=BB.Caption)
                                Slider(value=value.toFloat().coerceIn(range.start,range.endInclusive),onValueChange=change,valueRange=range,modifier=Modifier.semantics { contentDescription=text })
                            }
                            adjustment(R.string.subject_draw_horizontal,draft.x,0f..(1-draft.width).toFloat()) { draft=draft.copy(x=it.toDouble()) }
                            adjustment(R.string.subject_draw_vertical,draft.y,0f..(1-draft.height).toFloat()) { draft=draft.copy(y=it.toDouble()) }
                            adjustment(R.string.subject_draw_width,draft.width,.02f..1f) { draft=draft.copy(x=minOf(draft.x,1-it.toDouble()),width=it.toDouble()) }
                            adjustment(R.string.subject_draw_height,draft.height,.02f..1f) { draft=draft.copy(y=minOf(draft.y,1-it.toDouble()),height=it.toDouble()) }
                        }
                        OutlineButton(stringResource(R.string.subject_draw_track),Modifier.testTag("trackSubjectRegion"),enabled = !session.busy) {
                            playback.pause(); drawing = false; session.refine(draft,playback.time)
                        }
                        Text(stringResource(R.string.subject_draw_instruction),style = BB.Caption,color = BB.Muted)
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
                    if (session.busy) LinearProgressIndicator(progress = { session.progress.coerceIn(0.0,1.0).toFloat() },modifier = Modifier.fillMaxWidth())
                    if (analysis?.hasSelectedSubject == true && analysis.subject?.picks?.any { it.personID == null } == true)
                        Text(stringResource(R.string.subject_gaps),style = BB.Caption,color = BB.Muted)
                    if ((analysis?.subject?.unresolved ?: 0) > 0) Text(stringResource(R.string.subject_unresolved),style = BB.Caption,color = BB.Muted)
                    Text(stringResource(R.string.subject_reanalyze),style = BB.Caption,color = BB.Muted)
                    (failure ?: session.error)?.let {
                        Text(it,style = BB.Caption)
                        OutlineButton(stringResource(R.string.subject_retry),enabled = !session.busy) { session.retry() }
                    }
                }
            }
        }
    }
}
