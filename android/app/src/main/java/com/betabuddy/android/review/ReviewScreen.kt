package com.betabuddy.android.review

import android.content.Intent
import android.view.TextureView
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LifecycleEventEffect
import com.betabuddy.android.R
import com.betabuddy.android.data.Climb
import com.betabuddy.android.data.clockText
import com.betabuddy.android.journal.CardColor
import com.betabuddy.android.journal.DangerColor
import com.betabuddy.android.journal.FilledButton
import com.betabuddy.android.journal.Glyph
import com.betabuddy.android.journal.GlyphButton
import com.betabuddy.android.journal.JournalViewModel
import com.betabuddy.android.journal.KeepScreenOn
import com.betabuddy.android.journal.OutlineButton
import com.betabuddy.android.journal.ProgressBar
import com.betabuddy.android.journal.SwitchRow
import com.betabuddy.android.journal.UiMessage
import com.betabuddy.android.journal.appLocale
import com.betabuddy.android.journal.OrganizeDialog
import com.betabuddy.android.journal.dateText
import com.betabuddy.android.journal.displayDetail
import com.betabuddy.android.journal.displayTitle
import com.betabuddy.android.journal.gradeText
import com.betabuddy.android.journal.percentText
import com.betabuddy.android.journal.reportText
import com.betabuddy.android.journal.text
import com.betabuddy.android.ui.common.BBIcons
import com.betabuddy.android.ui.common.CircleIconButton
import com.betabuddy.android.ui.common.ColorCard
import com.betabuddy.android.ui.common.PaperPage
import com.betabuddy.android.ui.common.SectionLabel
import com.betabuddy.android.ui.common.Stamp
import com.betabuddy.android.ui.common.Stat
import com.betabuddy.android.ui.theme.BB
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/**
 * Replay of one climb: original or skeleton, hip trail, timeline, 0.1 s steps, 0.5× slow motion,
 * candidate cues that seek and pause, then the note for next time. Counterpart of ReviewView on iOS.
 */
@Composable
internal fun ReviewScreen(climb: Climb, model: JournalViewModel, onBack: () -> Unit) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val analysis = climb.analysis
    val playback = remember(climb.id) { Playback(context) }
    val reanalysisState by model.reanalysis.collectAsState()
    val reanalysis = reanalysisState.takeIf { it.climbId == climb.id }
    var skeleton by rememberSaveable(climb.id) { mutableStateOf(true) }
    var slow by rememberSaveable(climb.id) { mutableStateOf(false) }
    var note by rememberSaveable(climb.id) { mutableStateOf(climb.note) }
    var sent by rememberSaveable(climb.id) { mutableStateOf(climb.sent) }
    var savedNotice by remember(climb.id) { mutableStateOf(false) }
    var organize by remember(climb.id) { mutableStateOf(false) }
    if (organize) OrganizeDialog(climb, model) { organize = false }
    var confirmDelete by remember(climb.id) { mutableStateOf(false) }
    var failure by remember(climb.id) { mutableStateOf<UiMessage?>(null) }

    DisposableEffect(playback) {
        playback.load(model.videoFile(climb))
        onDispose { playback.release() }
    }
    LaunchedEffect(playback) {
        while (true) { playback.refreshTime(); delay(33) }
    }
    // Nothing keeps running once the app leaves the foreground.
    LifecycleEventEffect(Lifecycle.Event.ON_STOP) { playback.pause(); model.cancelReanalysis() }
    DisposableEffect(climb.id) {
        onDispose { model.cancelReanalysis(); model.clearReanalysisMessage() }
    }
    if (reanalysis?.busy == true) KeepScreenOn()

    val status = stringResource(if (climb.sent) R.string.journal_filter_sent else R.string.journal_filter_projecting)
    val coverageText = percentText(analysis.coverage, appLocale())
    val report = reportText(
        title = climb.title, grade = gradeText(climb.grade), status = status, clock = clockText(analysis.duration),
        coverageLine = stringResource(R.string.journal_share_coverage, coverageText),
        note = climb.note, caveat = stringResource(R.string.journal_share_caveat),
    )
    val shareTitle = stringResource(R.string.journal_share)

    PaperPage {
        Row(verticalAlignment = Alignment.CenterVertically) {
            CircleIconButton(BBIcons.ArrowBack, stringResource(R.string.journal_back), onBack, Modifier.testTag("reviewBack"))
            Spacer(Modifier.width(12.dp))
            Text(stringResource(R.string.journal_review_title), style = BB.Headline)
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Stamp("THE REPLAY")
            Spacer(Modifier.weight(1f))
            Text(gradeText(climb.grade), style = BB.display(28.sp), maxLines = 1)
        }
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(climb.title, style = BB.Title)
            Text(dateText(climb.climbedAtMillis, withYear = true), style = BB.Caption, color = BB.Muted)
            if (!climb.gymName.isNullOrBlank()) Text(climb.gymName, style = BB.Caption, color = BB.Muted)
            OutlineButton(stringResource(R.string.album_organize), onClick = { organize = true })
        }

        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            VideoStage(playback, climb, skeleton)
            val duration = maxOf(0.1, analysis.duration)
            val positionLabel = stringResource(R.string.journal_review_position_cd)
            Slider(
                value = playback.time.coerceIn(0.0, duration).toFloat(),
                onValueChange = { playback.pause(); playback.seek(it.toDouble()) },
                valueRange = 0f..duration.toFloat(),
                colors = SliderDefaults.colors(thumbColor = BB.Ink, activeTrackColor = BB.Ink, inactiveTrackColor = BB.Line),
                modifier = Modifier.fillMaxWidth().semantics {
                    contentDescription = positionLabel
                    stateDescription = clockText(playback.time) + " / " + clockText(analysis.duration)
                },
            )
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    clockText(playback.time) + " / " + clockText(analysis.duration),
                    style = BB.mono(11.sp, FontWeight.Medium, 0.sp), modifier = Modifier.weight(1f),
                )
                GlyphButton(Glyph.StepBack, stringResource(R.string.journal_review_back_step_cd), {
                    playback.pause(); playback.seek(maxOf(0.0, playback.time - 0.1))
                })
                GlyphButton(
                    if (playback.playing) Glyph.Pause else Glyph.Play,
                    stringResource(if (playback.playing) R.string.journal_review_pause else R.string.journal_review_play),
                    { playback.toggle(analysis.duration) }, background = BB.Lime,
                )
                GlyphButton(Glyph.StepForward, stringResource(R.string.journal_review_forward_step_cd), {
                    playback.pause(); playback.seek(minOf(analysis.duration, playback.time + 0.1))
                })
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.weight(1f)) { SwitchRow(stringResource(R.string.journal_review_skeleton), skeleton) { skeleton = it } }
                Spacer(Modifier.width(16.dp))
                Text(
                    stringResource(if (slow) R.string.journal_review_slow else R.string.journal_review_normal_speed),
                    style = BB.Caption, fontWeight = FontWeight.Bold,
                    modifier = Modifier.clip(CircleShape).background(BB.Lavender)
                        .clickable(role = Role.Button) { slow = !slow; playback.setSlow(slow) }
                        .padding(horizontal = 14.dp, vertical = 14.dp),
                )
            }
        }
        playback.failure?.let { Text(stringResource(it), style = BB.Caption, color = DangerColor) }

        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            if (reanalysis?.busy == true) {
                ProgressBar(reanalysis.progress)
                OutlineButton(stringResource(R.string.journal_reanalyze_stop)) { model.cancelReanalysis() }
            } else {
                OutlineButton(stringResource(R.string.journal_reanalyze)) { playback.pause(); model.reanalyze(climb) }
            }
            reanalysis?.message?.let {
                Text(it.text(), style = BB.Caption, color = BB.Muted, modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite })
            }
        }

        // Shares of sampled frames and projected angles. No scores.
        Row {
            Stat(coverageText, stringResource(R.string.journal_stat_coverage), Modifier.weight(1f))
            Stat(analysis.averageElbowAngle?.let { "${it.toInt()}°" } ?: "—", stringResource(R.string.journal_stat_elbow), Modifier.weight(1f))
        }
        Text(stringResource(R.string.journal_stat_caveat), style = BB.Caption, color = BB.Muted)

        SectionLabel(stringResource(R.string.journal_cues_title), "REPLAY CUES")
        if (analysis.moments.isEmpty()) {
            Text(stringResource(R.string.journal_cues_empty), style = BB.Subheadline, color = BB.Muted)
        }
        for (moment in analysis.moments) {
            Row(
                Modifier.fillMaxWidth().clip(RoundedCornerShape(18.dp)).background(CardColor)
                    .clickable(role = Role.Button) { playback.pause(); playback.seek(moment.time) }.padding(16.dp),
                verticalAlignment = Alignment.Top,
            ) {
                Text(
                    clockText(moment.time), style = BB.mono(12.sp, FontWeight.Bold, 0.sp),
                    modifier = Modifier.clip(RoundedCornerShape(8.dp)).background(BB.Lime).padding(8.dp),
                )
                Spacer(Modifier.width(14.dp))
                Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
                    Text(moment.displayTitle(), style = BB.Subheadline, fontWeight = FontWeight.Bold)
                    Text(moment.displayDetail(), style = BB.Caption, color = BB.Muted)
                }
            }
        }

        ColorCard(BB.Lavender, cornerRadius = 22.dp, padding = 20.dp, spacing = 16.dp) {
            SectionLabel(stringResource(R.string.journal_note_title), "YOUR NOTE")
            OutlinedTextField(
                value = note, onValueChange = { note = it.take(1000); savedNotice = false }, minLines = 3, maxLines = 6,
                placeholder = { Text(stringResource(R.string.journal_note_hint), color = BB.Muted) },
                modifier = Modifier.fillMaxWidth().testTag("reviewNote"),
            )
            SwitchRow(stringResource(R.string.journal_sent_toggle), sent) { sent = it; savedNotice = false }
            FilledButton(
                stringResource(if (savedNotice) R.string.journal_note_saved else R.string.journal_note_save),
                Modifier.testTag("saveNote").semantics { liveRegion = LiveRegionMode.Polite },
            ) {
                scope.launch {
                    model.saveNote(climb, note, sent)
                        .onSuccess { savedNotice = true; failure = null }
                        .onFailure { failure = UiMessage(R.string.journal_note_save_failed, it.localizedMessage ?: it.javaClass.simpleName) }
                }
            }
        }

        OutlineButton(shareTitle, Modifier.fillMaxWidth().testTag("shareReport")) {
            // Plain text through the system share sheet; the video never leaves the app this way.
            val send = Intent(Intent.ACTION_SEND).setType("text/plain").putExtra(Intent.EXTRA_TEXT, report)
            context.startActivity(Intent.createChooser(send, shareTitle))
        }
        failure?.let { Text(it.text(), style = BB.Caption, color = DangerColor, modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite }) }
        OutlineButton(stringResource(R.string.journal_delete), Modifier.fillMaxWidth().testTag("deleteClimb"), color = DangerColor) { confirmDelete = true }
        val seconds = analysis.processingSeconds.let { if (it.isFinite()) it.toInt() else 0 }
        Text(
            pluralStringResource(R.plurals.journal_model_line, analysis.sampledFrames, analysis.sampledFrames, seconds),
            style = BB.mono(9.sp, FontWeight.Normal, 0.sp), color = BB.Muted,
        )
    }

    if (confirmDelete) {
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            containerColor = BB.Paper,
            text = { Text(stringResource(R.string.journal_delete_confirm), style = BB.Body, color = BB.Ink) },
            confirmButton = {
                TextButton(onClick = {
                    confirmDelete = false
                    playback.pause()
                    scope.launch {
                        // On success the record disappears from the list and the logbook returns to it.
                        model.delete(climb).onFailure {
                            failure = UiMessage(R.string.journal_delete_failed, it.localizedMessage ?: it.javaClass.simpleName)
                        }
                    }
                }) { Text(stringResource(R.string.journal_delete), color = DangerColor) }
            },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text(stringResource(R.string.journal_cancel), color = BB.Ink) } },
        )
    }
}

/**
 * The picture and the skeleton share one box with the picture's own aspect ratio, centred in a
 * black stage (aspect-fit). The player draws into a TextureView that fills that box, with rotation
 * already applied by the decoder, so overlay coordinates and pixels line up by construction.
 */
@Composable
private fun VideoStage(playback: Playback, climb: Climb, skeleton: Boolean) {
    val analysis = climb.analysis
    BoxWithConstraints(
        Modifier.fillMaxWidth().height(340.dp).clip(RoundedCornerShape(20.dp)).background(Color.Black),
    ) {
        // Until the player reports its size, use the aspect ratio the analysis saw (same orientation rule).
        val aspect = playback.videoAspect ?: analysis.aspectRatio
        val (width, height) = OverlayGeometry.fit(maxWidth.value, maxHeight.value, aspect)
        Box(Modifier.align(Alignment.Center).size(width.dp, height.dp)) {
            AndroidView(
                factory = { viewContext -> TextureView(viewContext).also { playback.attach(it) } },
                onRelease = { playback.detach(it) },
                modifier = Modifier.fillMaxSize(),
            )
            if (skeleton) SkeletonOverlay(analysis, playback.time, Modifier.fillMaxSize())
        }
        val badge = when {
            !skeleton -> "ORIGINAL"
            analysis.nearestFrame(playback.time) == null -> stringResource(R.string.journal_review_no_pose_badge)
            else -> "33 POINTS / LOCAL"
        }
        Text(
            badge, style = BB.mono(10.sp, FontWeight.Bold),
            modifier = Modifier.align(Alignment.TopStart).padding(12.dp).clip(CircleShape).background(BB.Lime).padding(9.dp),
        )
    }
}
