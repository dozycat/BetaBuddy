package com.betabuddy.android.capture

import android.Manifest
import android.content.Intent
import android.net.Uri
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.betabuddy.android.R
import com.betabuddy.android.journal.DirectoryGym
import com.betabuddy.android.journal.GymDirectory
import com.betabuddy.android.journal.GymPicker
import com.betabuddy.android.ui.common.BBIcons
import com.betabuddy.android.ui.theme.BB

/**
 * Recommends gyms by distance when the position is known, and leads to the full directory.
 * Nothing here selects a gym by itself: the user taps one.
 */
@Composable
internal fun CaptureGymSheet(
    gymId: String?, gymName: String, locator: GymLocator, directory: List<DirectoryGym>,
    onSelect: (String?, String) -> Unit, onClose: () -> Unit,
) {
    var all by remember { mutableStateOf(false) }
    val context = LocalContext.current
    val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { granted -> locator.onPermissionResult(granted.values.any { it }) }
    LaunchedEffect(Unit) { locator.locateIfAllowed() }
    if (all) {
        // A choice made in the full directory closes the whole sheet.
        GymPicker(gymName, onSelect = onSelect, onClose = { all = false })
        return
    }
    AlertDialog(
        onDismissRequest = onClose, containerColor = BB.Paper,
        title = { Text(stringResource(R.string.album_choose_gym), style = BB.Title3, color = BB.Ink) },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState()).testTag("captureGymSheet"), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                val note = stringResource(R.string.capture_gym_location_note)
                when (locator.access) {
                    GymLocator.Access.ALLOWED -> {
                        Header(stringResource(R.string.capture_gym_near))
                        // Waiting for a position, having none, and having one with no gym near it are three different things to be told.
                        when {
                            locator.locating -> Row(
                                Modifier.padding(vertical = 8.dp).testTag("locationWaiting"),
                                horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically,
                            ) {
                                CircularProgressIndicator(color = BB.Ink, trackColor = BB.Line, strokeWidth = 2.dp, modifier = Modifier.size(20.dp))
                                Text(stringResource(R.string.capture_location_waiting), style = BB.Subheadline, color = BB.Muted)
                            }
                            locator.fix == null -> {
                                Text(stringResource(R.string.capture_location_unavailable), style = BB.Subheadline, color = BB.Muted, modifier = Modifier.padding(top = 8.dp).testTag("locationUnavailable"))
                                TextButton(onClick = { locator.retry() }, modifier = Modifier.testTag("locationRetry")) { Text(stringResource(R.string.capture_location_retry)) }
                            }
                            else -> {
                                val suggestions = locator.suggestions(directory)
                                for (item in suggestions.ranked) GymRow(item.gym, captureDistance(item), item.gym.id == gymId, "nearbyGym-${item.gym.id}") { onSelect(item.gym.id, item.gym.name) }
                                if (suggestions.ranked.isEmpty()) Text(stringResource(R.string.capture_gym_none_near), style = BB.Subheadline, color = BB.Muted, modifier = Modifier.padding(vertical = 8.dp).testTag("noGymNear"))
                                Text(note, style = BB.Caption, color = BB.Muted)
                                // Where distances from the directory's coordinates are shown, their source is named.
                                if (directory.any { it.coordinate != null }) Text(GymDirectory.COORDINATE_CREDIT, style = BB.Caption, color = BB.Muted, modifier = Modifier.testTag("coordinateCredit"))
                                // Gyms the directory cannot place more exactly than their district or city: listed, never measured.
                                if (suggestions.sameCity.isNotEmpty()) {
                                    Header(stringResource(R.string.capture_gym_same_city), Modifier.padding(top = 10.dp))
                                    for (gym in suggestions.sameCity) GymRow(gym, null, gym.id == gymId, "cityGym-${gym.id}") { onSelect(gym.id, gym.name) }
                                }
                            }
                        }
                    }
                    GymLocator.Access.NOT_ASKED -> {
                        Row(
                            Modifier.fillMaxWidth().defaultMinSize(minHeight = 48.dp)
                                .clickable(role = Role.Button) { permission.launch(GymLocator.PERMISSIONS) }.testTag("askLocation"),
                            horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Icon(BBIcons.Pin, contentDescription = null, tint = BB.Ink, modifier = Modifier.size(20.dp))
                            Text(stringResource(R.string.capture_gym_ask_location), style = BB.Headline, color = BB.Ink)
                        }
                        Text(note, style = BB.Caption, color = BB.Muted)
                    }
                    // Refused: said so, with where to change it.
                    GymLocator.Access.REFUSED -> {
                        Text(stringResource(R.string.capture_location_refused), style = BB.Subheadline, color = BB.Muted, modifier = Modifier.testTag("locationRefused"))
                        TextButton(onClick = {
                            runCatching { context.startActivity(Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", context.packageName, null))) }
                        }) { Text(stringResource(R.string.capture_open_settings)) }
                    }
                }
                Row(
                    Modifier.fillMaxWidth().padding(top = 10.dp).defaultMinSize(minHeight = 48.dp).clickable(role = Role.Button) { all = true }.testTag("allGyms"),
                    horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(stringResource(R.string.capture_gym_all), style = BB.Headline, color = BB.Ink, modifier = Modifier.weight(1f))
                    Icon(BBIcons.ChevronRight, contentDescription = null, tint = BB.Ink, modifier = Modifier.size(18.dp))
                }
            }
        },
        confirmButton = { TextButton(onClick = onClose) { Text(stringResource(R.string.album_cancel)) } },
    )
}

@Composable
private fun Header(text: String, modifier: Modifier = Modifier) {
    Text(text, style = BB.Caption, fontWeight = FontWeight.Bold, color = BB.Muted, modifier = modifier.semantics { heading() })
}

@Composable
private fun GymRow(gym: DirectoryGym, distance: String?, selected: Boolean, tag: String, onClick: () -> Unit) {
    Row(
        Modifier.fillMaxWidth().defaultMinSize(minHeight = 48.dp).clickable(role = Role.Button, onClick = onClick).padding(vertical = 6.dp).testTag(tag),
        horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
            Text(gym.name, style = BB.Subheadline, fontWeight = FontWeight.SemiBold, color = BB.Ink)
            val place = listOf(gym.city.ifBlank { gym.province }, gym.address.ifBlank { gym.district }).filter(String::isNotBlank).distinct().joinToString(" · ")
            if (place.isNotEmpty()) Text(place, style = BB.Caption, color = BB.Muted)
        }
        if (distance != null) Text(distance, style = BB.mono(12.sp, FontWeight.Medium, 0.sp), color = BB.Muted)
        if (selected) Icon(BBIcons.Check, contentDescription = null, tint = BB.Ink, modifier = Modifier.size(16.dp))
    }
}
