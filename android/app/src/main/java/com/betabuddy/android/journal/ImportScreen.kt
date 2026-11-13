package com.betabuddy.android.journal

import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LifecycleEventEffect
import com.betabuddy.android.R
import com.betabuddy.android.data.UNGRADED_VALUE
import com.betabuddy.android.pose.AnalysisQuality
import com.betabuddy.android.ui.common.BBIcons
import com.betabuddy.android.ui.common.BigButton
import com.betabuddy.android.ui.common.ColorCard
import com.betabuddy.android.ui.common.PaperPage
import com.betabuddy.android.ui.common.SectionLabel
import com.betabuddy.android.ui.common.Stamp
import com.betabuddy.android.ui.theme.BB

/** Stored grade values: the "no grade" sentinel, then V0–V12. Set by the user; never inferred. */
internal val gradeValues: List<String> = listOf(UNGRADED_VALUE) + (0..12).map { "V$it" }

/**
 * Pick one video with the system photo picker → tag it → choose the detail level → analyse on
 * this phone. Counterpart of ImportView on iOS.
 */
@Composable
internal fun ImportScreen(model: JournalViewModel, onClose: () -> Unit, onSaved: (String) -> Unit) {
    val state by model.import.collectAsState()
    var title by rememberSaveable { mutableStateOf("") }
    var grade by rememberSaveable { mutableStateOf(UNGRADED_VALUE) }
    var sent by rememberSaveable { mutableStateOf(false) }
    var note by rememberSaveable { mutableStateOf("") }
    var gymId by rememberSaveable { mutableStateOf<String?>(null) }
    var gymName by rememberSaveable { mutableStateOf("") }
    var climbedOn by rememberSaveable { mutableStateOf(System.currentTimeMillis()) }
    var folderId by rememberSaveable { mutableStateOf<String?>(null) }
    var organizing by rememberSaveable { mutableStateOf(false) }
    var quality by rememberSaveable { mutableStateOf(AnalysisQuality.BALANCED) }
    val defaultTitle = stringResource(R.string.journal_default_title) // in the language shown at save time

    // The system photo picker returns only the item the user chose; no storage permission is requested.
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri -> model.onVideoPicked(uri) }

    val locked = state.busy || state.importing
    // As on iOS, the flow cannot be dismissed while it is copying or analysing; stop first.
    BackHandler { if (!locked) onClose() }
    // Leaving the foreground stops the analysis. There is no background continuation.
    LifecycleEventEffect(Lifecycle.Event.ON_STOP) { model.cancelAnalysis() }
    LaunchedEffect(state.savedId) { state.savedId?.let(onSaved) }
    if (state.busy) KeepScreenOn()

    PaperPage {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(stringResource(R.string.journal_import_nav_title), style = BB.Headline, modifier = Modifier.weight(1f))
            OutlineButton(stringResource(R.string.journal_import_done), Modifier.testTag("importDone"), enabled = !locked, onClick = onClose)
        }
        Stamp("A NEW ATTEMPT")
        Text(
            stringResource(if (state.busy) R.string.journal_import_headline_busy else R.string.journal_import_headline),
            style = BB.display(29.sp),
        )
        if (state.busy) {
            Processing(state.progress) { model.cancelAnalysis() }
        } else {
            Column(verticalArrangement = Arrangement.spacedBy(22.dp)) {
                ColorCard(BB.Lime, cornerRadius = 22.dp, padding = 20.dp, spacing = 14.dp) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(if (state.staged == null) BBIcons.Plus else BBIcons.Check, contentDescription = null, tint = BB.Ink, modifier = Modifier.size(28.dp))
                        Spacer(Modifier.width(12.dp))
                        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                            Text(
                                stringResource(
                                    when {
                                        state.importing -> R.string.journal_import_fetching
                                        state.staged == null -> R.string.journal_import_pick_title
                                        else -> R.string.journal_import_ready
                                    },
                                ),
                                style = BB.Headline, modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite },
                            )
                            Text(stringResource(R.string.journal_import_requirements), style = BB.Caption, color = BB.Muted)
                        }
                    }
                    FilledButton(stringResource(R.string.journal_import_pick_action), Modifier.testTag("pickVideo"), enabled = !state.importing) {
                        picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly))
                    }
                }
                Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
                    Text(stringResource(R.string.journal_import_name_label), style = BB.Caption, fontWeight = FontWeight.Bold)
                    OutlinedTextField(
                        value = title, onValueChange = { title = it.take(60) }, singleLine = true,
                        placeholder = { Text(stringResource(R.string.journal_import_name_hint), color = BB.Muted) },
                        modifier = Modifier.fillMaxWidth().testTag("climbTitle"),
                    )
                    GradePicker(grade) { grade = it }
                    TextButton(onClick = { organizing = !organizing }) { Text(stringResource(R.string.album_optional_tags)) }
                    if (organizing) MetadataFields(model, gymId, gymName, climbedOn, folderId,
                        { id, name -> gymId = id; gymName = name }, { climbedOn = it }, { folderId = it })
                    SwitchRow(stringResource(R.string.journal_sent_toggle), sent) { sent = it }
                    OutlinedTextField(
                        value = note, onValueChange = { note = it.take(1000) }, minLines = 2, maxLines = 4,
                        placeholder = { Text(stringResource(R.string.journal_import_note_hint), color = BB.Muted) },
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
                Column(Modifier.selectableGroup(), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    SectionLabel(stringResource(R.string.journal_quality_label), "SAME FULL MODEL")
                    QualityOption(R.string.journal_quality_balanced, R.string.journal_quality_balanced_detail, quality == AnalysisQuality.BALANCED) { quality = AnalysisQuality.BALANCED }
                    QualityOption(R.string.journal_quality_detailed, R.string.journal_quality_detailed_detail, quality == AnalysisQuality.DETAILED) { quality = AnalysisQuality.DETAILED }
                }
                BigButton(
                    stringResource(R.string.journal_import_start),
                    { model.startAnalysis(title, grade, sent, note, quality, defaultTitle, gymId, gymName, climbedOn, folderId) },
                    Modifier.testTag("startAnalysis"),
                    enabled = state.staged != null && !state.importing,
                )
                Text(stringResource(R.string.journal_import_foreground_note), style = BB.Caption, color = BB.Muted)
            }
        }
        state.error?.let { error ->
            Row(Modifier.testTag("importError").semantics { liveRegion = LiveRegionMode.Polite }, verticalAlignment = Alignment.Top) {
                Icon(BBIcons.Alert, contentDescription = null, tint = DangerColor, modifier = Modifier.size(20.dp))
                Spacer(Modifier.width(8.dp))
                Text(error.text(), style = BB.Subheadline, color = DangerColor)
            }
        }
    }
}

@Composable
private fun Processing(progress: Double, onStop: () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(24.dp)) {
        Box(Modifier.fillMaxWidth().clip(RoundedCornerShape(24.dp)).background(BB.Lime).padding(24.dp)) {
            ClimbingArt(Modifier.fillMaxWidth().height(220.dp))
        }
        Row(verticalAlignment = Alignment.Bottom) {
            Text(percentText(progress, appLocale()), style = BB.display(54.sp), modifier = Modifier.weight(1f))
            Text("ON YOUR PHONE", style = BB.mono(10.sp, FontWeight.Bold)) // brand mark, not translated
        }
        ProgressBar(progress)
        Text(stringResource(R.string.journal_import_steps), style = BB.Subheadline, color = BB.Muted)
        OutlineButton(stringResource(R.string.journal_import_stop), Modifier.fillMaxWidth().testTag("stopAnalysis"), onClick = onStop)
    }
}

@Composable
internal fun GradePicker(grade: String, onSelect: (String) -> Unit) {
    var open by remember { mutableStateOf(false) }
    Row(
        Modifier.fillMaxWidth().defaultMinSize(minHeight = 48.dp).clickable(role = Role.DropdownList) { open = true },
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(stringResource(R.string.journal_import_grade_label), style = BB.Subheadline, modifier = Modifier.weight(1f))
        Box {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(gradeText(grade), style = BB.Subheadline, fontWeight = FontWeight.Bold)
                Text(" ▾", style = BB.Subheadline)
            }
            DropdownMenu(expanded = open, onDismissRequest = { open = false }, containerColor = BB.Paper) {
                for (value in gradeValues) {
                    DropdownMenuItem(text = { Text(gradeText(value), color = BB.Ink) }, onClick = { onSelect(value); open = false })
                }
            }
        }
    }
}

@Composable
internal fun SwitchRow(label: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(
        Modifier.fillMaxWidth().defaultMinSize(minHeight = 48.dp).toggleable(value = checked, role = Role.Switch, onValueChange = onChange),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(label, style = BB.Subheadline, modifier = Modifier.weight(1f))
        Switch(
            checked = checked, onCheckedChange = null,
            colors = SwitchDefaults.colors(checkedThumbColor = BB.Paper, checkedTrackColor = BB.Ink, uncheckedThumbColor = BB.Ink, uncheckedTrackColor = BB.Paper, uncheckedBorderColor = BB.Ink),
        )
    }
}

@Composable
private fun QualityOption(title: Int, detail: Int, selected: Boolean, onSelect: () -> Unit) {
    Row(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(16.dp)).background(if (selected) BB.Lavender else CardColor)
            .selectable(selected = selected, role = Role.RadioButton, onClick = onSelect).padding(16.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(stringResource(title), style = BB.Subheadline, fontWeight = FontWeight.Bold)
            Text(stringResource(detail), style = BB.Caption, color = BB.Muted)
        }
        // Selection is a mark as well as a colour.
        Text(if (selected) "●" else "○", style = BB.Headline)
    }
}

/** Analysis only runs in the foreground, so the screen is kept awake while it does. */
@Composable
internal fun KeepScreenOn() {
    val view = LocalView.current
    DisposableEffect(view) {
        view.keepScreenOn = true
        onDispose { view.keepScreenOn = false }
    }
}
