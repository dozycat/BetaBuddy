package com.betabuddy.android.journal

import android.content.Intent
import android.net.Uri
import android.provider.Settings
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.betabuddy.android.R
import com.betabuddy.android.ui.common.ColorCard
import com.betabuddy.android.ui.common.PaperPage
import com.betabuddy.android.ui.theme.BB

@Composable
internal fun LibraryImportHost(model: LibraryImport) {
    val owner = LocalLifecycleOwner.current
    val camera by CameraGate.open.collectAsState()
    val subject by SubjectGate.open.collectAsState()
    val analysis by AnalysisService.state.collectAsState()
    val desk by AnalysisDesk.open.collectAsState()
    val busy = camera || subject || analysis.busy || desk.isNotEmpty()
    DisposableEffect(owner,model,busy) {
        fun update() { model.setActive(owner.lifecycle.currentState.isAtLeast(Lifecycle.State.RESUMED),busy) }
        val observer = LifecycleEventObserver { _, _ -> update() }
        owner.lifecycle.addObserver(observer); update()
        onDispose { owner.lifecycle.removeObserver(observer); model.setActive(false,busy) }
    }
}

@Composable
internal fun LibraryImportScreen(model: LibraryImport, onClose: () -> Unit) {
    val context = LocalContext.current
    val state by model.state.collectAsState()
    val enabledLabel = stringResource(R.string.library_enable)
    val permissions = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { model.refresh() }
    BackHandler(onBack = onClose)
    PaperPage {
        Row(Modifier.fillMaxWidth(),horizontalArrangement = Arrangement.SpaceBetween) {
            Text(stringResource(R.string.library_title),style = BB.Title3,modifier = Modifier.weight(1f))
            TextButton(onClick = onClose) { Text(stringResource(R.string.album_done)) }
        }
        ColorCard(BB.Lime) {
            Text(stringResource(R.string.library_heading),style = BB.Title)
            Text(stringResource(R.string.library_description),style = BB.Body)
            Row(Modifier.fillMaxWidth(),verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
                Text(enabledLabel,modifier = Modifier.weight(1f))
                Switch(state.enabled,onCheckedChange = { enabled ->
                    model.setEnabled(enabled)
                    if (enabled && state.access == LibraryAccess.NONE) permissions.launch(LibraryPermission.request())
                },modifier = Modifier.testTag("libraryImportToggle").semantics { contentDescription = enabledLabel })
            }
            if (state.enabled && state.access == LibraryAccess.NONE) {
                Text(stringResource(R.string.library_access_error))
                TextButton(onClick = { context.startActivity(Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:${context.packageName}"))) }) { Text(stringResource(R.string.library_settings)) }
            }
            if (state.enabled && state.access == LibraryAccess.SELECTED) {
                Text(stringResource(R.string.library_selected))
                TextButton(onClick = { permissions.launch(LibraryPermission.request()) }) { Text(stringResource(R.string.library_reselect)) }
            }
        }
        if (state.enabled && state.access != LibraryAccess.NONE) {
            Column(Modifier.fillMaxWidth().testTag("libraryImportStatus"),verticalArrangement = Arrangement.spacedBy(12.dp)) {
                if (state.running) {
                    Text(stringResource(R.string.library_running),style = BB.Headline)
                    LinearProgressIndicator(progress = { state.checked.toFloat()/maxOf(1,state.total) },modifier = Modifier.fillMaxWidth())
                    Text(stringResource(R.string.library_progress,state.checked,state.total))
                    TextButton(onClick = model::stop) { Text(stringResource(R.string.library_stop)) }
                } else {
                    Text(stringResource(R.string.library_imported,state.imported),style = BB.Headline)
                    TextButton(onClick = model::retry,modifier = Modifier.testTag("libraryScan")) { Text(stringResource(R.string.library_scan)) }
                }
                if (state.unavailable > 0) Text(stringResource(R.string.library_unavailable,state.unavailable),style = BB.Caption,color = BB.Muted)
                state.error?.let { Text(stringResource(it),color = DangerColor) }
            }
        }
        Text(stringResource(R.string.library_grouping),style = BB.Caption,color = BB.Muted)
        Text(stringResource(R.string.library_deletion),style = BB.Caption,color = BB.Muted)
        Text(GymDirectory.COORDINATE_CREDIT,style = BB.Caption,color = BB.Muted)
    }
}
