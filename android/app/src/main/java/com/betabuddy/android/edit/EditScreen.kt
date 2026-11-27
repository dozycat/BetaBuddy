package com.betabuddy.android.edit

import android.content.Context
import android.view.TextureView
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.compose.ui.window.DialogWindowProvider
import androidx.core.view.WindowCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LifecycleEventEffect
import androidx.media3.common.Player
import com.betabuddy.android.R
import com.betabuddy.android.journal.DangerColor
import com.betabuddy.android.journal.OutlineButton
import com.betabuddy.android.review.OverlayGeometry
import com.betabuddy.android.review.Playback
import com.betabuddy.android.ui.common.BigButton
import com.betabuddy.android.ui.theme.BB
import java.io.File
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.launch

/**
 * The automatic edit of one record, for as long as its page is open: which switches are on, and
 * the file rendered for them. A change of switch renders again; what was found about the clip the
 * first time (who walks through it, the clean plate) is kept, so only the export is repeated.
 * The file is in the cache and goes when the page closes. Main thread only.
 */
internal class EditSession(
    private val context: Context,
    private val climbId: String,
    video: File,
    val plan: EditPlan,
    private val scope: CoroutineScope,
    newFinder: () -> PeopleFinder = { ModelPeopleFinder(context) },
) {
    sealed interface State {
        data class Working(val progress: Double) : State
        data class Ready(val file: File) : State
        /** [reason] is what the system said; it has no translation. */
        data class Failed(val reason: String) : State
    }

    var options by mutableStateOf(EditOptions.offered)
        private set
    var state by mutableStateOf<State>(State.Working(0.0))
        private set

    private val renderer = EditRenderer(context, video, plan, newFinder)
    private var job: Job? = null
    private var serial = 0

    fun open() { if (job == null) render() }

    fun set(next: EditOptions) {
        if (next == options) return
        options = next
        render()
    }

    fun retry() = render()

    private fun render() {
        val previous = job
        val wanted = options
        job = scope.launch {
            previous?.cancelAndJoin()
            state = State.Working(0.0)
            val directory = EditFiles.directory(context, climbId)
            val target = File(directory, "BetaBuddy-${++serial}.mp4")
            try {
                renderer.render(wanted, target) { state = State.Working(it.coerceIn(0.0, 1.0)) }
                // The file of the switches before these is no longer shown.
                directory.listFiles()?.filter { it != target }?.forEach { it.delete() }
                state = State.Ready(target)
            } catch (error: CancellationException) {
                target.delete()
                throw error
            } catch (error: Exception) {
                android.util.Log.e("edit", "render failed", error)   // Local logcat only.
                target.delete()
                state = State.Failed(EditRenderer.reason(error))
            }
        }
    }

    fun close() {
        job?.cancel()
        job = null
        EditFiles.clear(context, climbId)
    }
}

/**
 * The edit page, full screen over the replay: the rendered clip playing in a loop, a chip for each
 * switch the plan has something for, sharing, and Done. System back closes it, like Done.
 */
@Composable
internal fun EditScreen(session: EditSession, onDone: () -> Unit) {
    val context = LocalContext.current
    DisposableEffect(session) {
        session.open()
        onDispose { session.close() }
    }
    Dialog(
        onDismissRequest = onDone,
        properties = DialogProperties(usePlatformDefaultWidth = false, decorFitsSystemWindows = false, dismissOnClickOutside = false),
    ) {
        val view = LocalView.current
        SideEffect {
            (view.parent as? DialogWindowProvider)?.window?.let { window ->
                window.setDimAmount(0f)
                WindowCompat.getInsetsController(window, view).apply {
                    isAppearanceLightStatusBars = true
                    isAppearanceLightNavigationBars = true
                }
            }
        }
        val state = session.state
        val shareTitle = stringResource(R.string.journal_edit_share)
        Column(
            Modifier.fillMaxSize().background(BB.Paper).windowInsetsPadding(WindowInsets.safeDrawing).padding(horizontal = 16.dp).testTag("editScreen"),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text(stringResource(R.string.journal_edit_open), style = BB.Headline, modifier = Modifier.weight(1f))
                Box(
                    Modifier.defaultMinSize(minWidth = 48.dp, minHeight = 48.dp).clip(CircleShape)
                        .clickable(role = Role.Button, onClick = onDone).padding(horizontal = 12.dp).testTag("editDone"),
                    contentAlignment = Alignment.Center,
                ) { Text(stringResource(if (state is EditSession.State.Ready) R.string.journal_import_done else R.string.journal_cancel), style = BB.Subheadline, fontWeight = FontWeight.Bold) }
            }
            Box(Modifier.weight(1f).fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(Color.Black), contentAlignment = Alignment.Center) {
                when (state) {
                    is EditSession.State.Ready -> key(state.file) { LoopingClip(state.file) }
                    is EditSession.State.Working -> Column(
                        Modifier.padding(32.dp).testTag("editWorking"), verticalArrangement = Arrangement.spacedBy(12.dp), horizontalAlignment = Alignment.CenterHorizontally,
                    ) {
                        BelayProgress(state.progress)
                    }
                    is EditSession.State.Failed -> Column(
                        Modifier.padding(24.dp), verticalArrangement = Arrangement.spacedBy(12.dp), horizontalAlignment = Alignment.CenterHorizontally,
                    ) {
                        Text(
                            stringResource(R.string.journal_edit_failed, state.reason), style = BB.Subheadline, color = BB.Paper,
                            modifier = Modifier.testTag("editFailed").semantics { liveRegion = LiveRegionMode.Polite },
                        )
                        OutlineButton(stringResource(R.string.capture_location_retry), Modifier.testTag("editRetry"), color = BB.Paper, onClick = session::retry)
                    }
                }
            }
            val options = session.options
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                if (EditOptions.REMOVAL_OFFERED && session.plan.hasKeep) EditChip(stringResource(R.string.journal_edit_remove_passers), options.removePassersBy, Modifier.weight(1f).testTag("editRemovePassers")) {
                    session.set(options.copy(removePassersBy = it))
                }
                if (session.plan.hasSlowMotion) EditChip(stringResource(R.string.journal_edit_slow), options.slowMotion, Modifier.weight(1f).testTag("editSlow")) {
                    session.set(options.copy(slowMotion = it))
                }
                if (session.plan.hasCamera) EditChip(stringResource(R.string.journal_edit_camera), options.cameraMoves, Modifier.weight(1f).testTag("editCamera")) {
                    session.set(options.copy(cameraMoves = it))
                }
            }
            BigButton(
                shareTitle, { (state as? EditSession.State.Ready)?.let { EditFiles.share(context, it.file, shareTitle) } },
                Modifier.testTag("editShare"), enabled = state is EditSession.State.Ready,
            )
        }
    }
}

/** A switch as a chip: filled when on. At least 48 dp to touch. */
@Composable
private fun EditChip(label: String, on: Boolean, modifier: Modifier = Modifier, onChange: (Boolean) -> Unit) {
    Box(
        modifier.defaultMinSize(minHeight = 48.dp).clip(CircleShape)
            .then(if (on) Modifier.background(BB.Ink) else Modifier.border(1.dp, BB.Ink.copy(alpha = 0.35f), CircleShape))
            .toggleable(value = on, role = Role.Switch, onValueChange = onChange).padding(horizontal = 10.dp, vertical = 8.dp),
        contentAlignment = Alignment.Center,
    ) { Text(label, style = BB.Caption, fontWeight = FontWeight.Bold, color = if (on) BB.Paper else BB.Ink, maxLines = 2) }
}

/** The rendered file, playing over and over while the page is in front. */
@Composable
private fun LoopingClip(file: File) {
    val context = LocalContext.current
    val playback = remember(file) { Playback(context) }
    DisposableEffect(playback) {
        playback.load(file)
        playback.player.repeatMode = Player.REPEAT_MODE_ALL
        playback.player.play()
        onDispose { playback.release() }
    }
    LifecycleEventEffect(Lifecycle.Event.ON_STOP) { playback.pause() }
    LifecycleEventEffect(Lifecycle.Event.ON_START) { playback.player.play() }
    BoxWithConstraints(Modifier.fillMaxSize().testTag("editPreview")) {
        val (width, height) = OverlayGeometry.fit(maxWidth.value, maxHeight.value, playback.videoAspect ?: (9.0 / 16))
        Box(Modifier.align(Alignment.Center).size(width.dp, height.dp)) {
            AndroidView(
                factory = { viewContext -> TextureView(viewContext).also { playback.attach(it) } },
                onRelease = { playback.detach(it) },
                modifier = Modifier.fillMaxSize(),
            )
            playback.failure?.let { Text(stringResource(it), style = BB.Caption, color = DangerColor, modifier = Modifier.align(Alignment.Center)) }
        }
    }
}
