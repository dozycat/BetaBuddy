package com.betabuddy.android.journal

import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.selection.toggleable
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.betabuddy.android.R
import com.betabuddy.android.data.UNGRADED_VALUE
import com.betabuddy.android.pose.AnalysisQuality
import com.betabuddy.android.ui.common.BBIcons
import com.betabuddy.android.ui.common.BigButton
import com.betabuddy.android.ui.common.ColorCard
import com.betabuddy.android.ui.common.PaperPage
import com.betabuddy.android.ui.common.Stamp
import com.betabuddy.android.ui.theme.BB

/** Stored grade values: the "no grade" sentinel, then V0–V12. Set by the user; never inferred. */
internal val gradeValues: List<String> = listOf(UNGRADED_VALUE) + (0..12).map { "V$it" }

/**
 * Pick a video, edit its time and tags, then save it. Pose recognition waits for “Analyse this attempt”.
 */
@Composable
internal fun ImportScreen(model: JournalViewModel, onClose: () -> Unit, onSaved: (String) -> Unit) {
    val state by model.import.collectAsState()
    var title by rememberSaveable { mutableStateOf("") }
    var grade by rememberSaveable { mutableStateOf(UNGRADED_VALUE) }
    var sent by rememberSaveable { mutableStateOf(false) }
    var gymId by rememberSaveable { mutableStateOf<String?>(null) }
    var gymName by rememberSaveable { mutableStateOf("") }
    var climbedOn by rememberSaveable { mutableStateOf(System.currentTimeMillis()) }
    var folderId by rememberSaveable { mutableStateOf<String?>(null) }
    var organizing by rememberSaveable { mutableStateOf(false) }
    var appliedVideo by rememberSaveable { mutableStateOf<String?>(null) }
    val defaultTitle = stringResource(R.string.journal_default_title) // in the language shown at save time

    // The system photo picker returns only the item the user chose; no storage permission is requested.
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri -> model.onVideoPicked(uri) }

    val locked = state.busy || state.importing
    // Lock only while copying or committing the video. Inference runs after saving.
    BackHandler { if (!locked) onClose() }
    LaunchedEffect(state.staged) {
        val path = state.staged?.path
        if (path != null && path != appliedVideo) {
            climbedOn = state.recordingDateMillis ?: System.currentTimeMillis()
            appliedVideo = path
        }
    }
    LaunchedEffect(state.savedId) { state.savedId?.let(onSaved) }

    PaperPage {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(stringResource(R.string.journal_import_nav_title), style = BB.Headline, modifier = Modifier.weight(1f))
            OutlineButton(stringResource(R.string.journal_import_done), Modifier.testTag("importDone"), enabled = !locked, onClick = onClose)
        }
        Stamp("A NEW ATTEMPT")
        Text(
            stringResource(if (state.busy) R.string.journal_saving_video else R.string.journal_import_headline),
            style = BB.display(29.sp),
        )
        if (state.busy) {
            androidx.compose.material3.CircularProgressIndicator(color = BB.Ink)
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
                    ClimbDateTimeField(climbedOn) { climbedOn = it }
                    Text(stringResource(if (state.recordingDateMillis != null) R.string.journal_recording_date_found else R.string.journal_recording_date_missing), style = BB.Caption, color = BB.Muted)
                    TextButton(onClick = { organizing = !organizing }) { Text(stringResource(R.string.album_optional_tags)) }
                    if (organizing) MetadataFields(model, gymId, gymName, climbedOn, folderId,
                        { id, name -> gymId = id; gymName = name }, { climbedOn = it }, { folderId = it }, showDate = false)
                    SwitchRow(stringResource(R.string.journal_sent_toggle), sent) { sent = it }
                }
                BigButton(
                    stringResource(R.string.journal_save_climb),
                    { model.saveVideo(title, grade, sent, "", AnalysisQuality.BALANCED, defaultTitle, gymId, gymName, climbedOn, folderId) },
                    Modifier.testTag("startAnalysis"),
                    enabled = state.staged != null && !state.importing,
                )
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

/** Notification permission is optional; refusing it never blocks local analysis. */
@Composable
internal fun rememberAnalysisNotifications(): () -> Unit {
    val context = androidx.compose.ui.platform.LocalContext.current
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { }
    return {
        if (android.os.Build.VERSION.SDK_INT >= 33 && context.checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS) != android.content.pm.PackageManager.PERMISSION_GRANTED) {
            launcher.launch(android.Manifest.permission.POST_NOTIFICATIONS)
        }
    }
}
