package com.betabuddy.android.journal

import android.app.DatePickerDialog
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.betabuddy.android.R
import com.betabuddy.android.data.Climb
import com.betabuddy.android.data.ClimbFolder
import com.betabuddy.android.ui.theme.BB
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

@Serializable
internal data class DirectoryGym(val id: String, val name: String, val province: String = "", val city: String = "", val address: String = "", val sourceURL: String = "", val district: String = "", val aliases: List<String> = emptyList())
@Serializable
private data class GymDirectory(val gyms: List<DirectoryGym> = emptyList())
private val directoryJson = Json { ignoreUnknownKeys = true }

@Composable
internal fun directoryGyms(): List<DirectoryGym> {
    val context = LocalContext.current
    val gyms by produceState<List<DirectoryGym>>(emptyList()) {
        value = withContext(Dispatchers.IO) {
            runCatching { context.assets.open("gyms-cn.json").bufferedReader().use { directoryJson.decodeFromString<GymDirectory>(it.readText()).gyms } }.getOrDefault(emptyList())
        }
    }
    return gyms
}

internal fun DirectoryGym.displayLabel(): String = listOf(name, city.ifBlank { province }, address.ifBlank { district }).filter(String::isNotBlank).distinct().joinToString(" · ")

@Composable
internal fun FolderStrip(folders: List<ClimbFolder>, selected: String?, onSelect: (String?) -> Unit, onManage: () -> Unit) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
        Row(Modifier.weight(1f).horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            FilterChip(selected == null, { onSelect(null) }, { Text(stringResource(R.string.journal_filter_all)) })
            FilterChip(selected == "", { onSelect("") }, { Text(stringResource(R.string.album_unfiled)) })
            folders.forEach { folder -> FilterChip(selected == folder.id, { onSelect(folder.id) }, { Text(folder.name) }) }
        }
        TextButton(onClick = onManage, modifier = Modifier.defaultMinSize(minWidth = 48.dp, minHeight = 48.dp)) {
            Text(stringResource(R.string.album_manage))
        }
    }
}

@Composable
internal fun FolderManager(model: JournalViewModel, onClose: () -> Unit) {
    val folders by model.folders.collectAsState()
    val scope = rememberCoroutineScope()
    var editing by remember { mutableStateOf<String?>(null) }
    var name by rememberSaveable { mutableStateOf("") }
    var removing by remember { mutableStateOf<ClimbFolder?>(null) }
    var error by remember { mutableStateOf(false) }
    var busy by remember { mutableStateOf(false) }
    AlertDialog(onDismissRequest = { if (!busy) onClose() }, containerColor = BB.Paper,
        title = { Text(stringResource(R.string.album_folders)) },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                folders.forEach { folder ->
                    Column {
                        Text(folder.name, style = BB.Headline)
                        Row {
                            TextButton(enabled = !busy, onClick = { editing = folder.id; name = folder.name; error = false }) { Text(stringResource(R.string.album_rename)) }
                            TextButton(enabled = !busy, onClick = { removing = folder }) { Text(stringResource(R.string.album_delete)) }
                        }
                    }
                }
                OutlinedTextField(name, { name = it.take(40); error = false }, label = { Text(stringResource(if (editing == null) R.string.album_new_folder else R.string.album_rename)) }, singleLine = true)
                if (error) Text(stringResource(R.string.album_folder_error), color = DangerColor)
                FilledButton(stringResource(R.string.album_save), enabled = !busy && name.isNotBlank()) {
                    busy = true
                    scope.launch { model.saveFolder(name, editing).fold(onSuccess = { name = ""; editing = null; error = false }, onFailure = { error = true }); busy = false }
                }
                if (editing != null) TextButton(onClick = { editing = null; name = "" }) { Text(stringResource(R.string.album_cancel)) }
            }
        }, confirmButton = { TextButton(enabled = !busy, onClick = onClose) { Text(stringResource(R.string.album_done)) } })
    removing?.let { folder ->
        AlertDialog(onDismissRequest = { if (!busy) removing = null }, containerColor = BB.Paper,
            title = { Text(stringResource(R.string.album_delete)) }, text = { Text(stringResource(R.string.album_delete_detail)) },
            confirmButton = { TextButton(enabled = !busy, onClick = {
                busy = true; scope.launch { model.deleteFolder(folder.id).fold(onSuccess = { if (editing == folder.id) { editing = null; name = "" }; removing = null }, onFailure = { error = true; removing = null }); busy = false }
            }) { Text(stringResource(R.string.album_delete)) } },
            dismissButton = { TextButton(enabled = !busy, onClick = { removing = null }) { Text(stringResource(R.string.album_cancel)) } })
    }
}

@Composable
internal fun ChoiceRow(label: String, value: String, choices: List<Pair<String, String>>, onSelect: (String) -> Unit) {
    var open by remember { mutableStateOf(false) }
    Box {
        TextButton(onClick = { open = true }, modifier = Modifier.fillMaxWidth()) { Text("$label · $value") }
        DropdownMenu(expanded = open, onDismissRequest = { open = false }, containerColor = BB.Paper) {
            choices.forEach { (key, title) -> DropdownMenuItem(text = { Text(title) }, onClick = { onSelect(key); open = false }) }
        }
    }
}

@Composable
internal fun ClimbDateField(label: String, millis: Long?, onChange: (Long) -> Unit) {
    val context = LocalContext.current
    TextButton(onClick = {
        val day = Instant.ofEpochMilli(millis ?: System.currentTimeMillis()).atZone(ZoneId.systemDefault()).toLocalDate()
        DatePickerDialog(context, { _, y, m, d -> onChange(LocalDate.of(y, m + 1, d).atStartOfDay(ZoneId.systemDefault()).toInstant().toEpochMilli()) }, day.year, day.monthValue - 1, day.dayOfMonth).apply { datePicker.maxDate = System.currentTimeMillis() }.show()
    }) { Text("$label · ${millis?.let { dateText(it, withYear = true) } ?: stringResource(R.string.album_any)}") }
}

@Composable
internal fun MetadataFields(model: JournalViewModel, gymId: String?, gymName: String, date: Long, folderId: String?, onGym: (String?, String) -> Unit, onDate: (Long) -> Unit, onFolder: (String?) -> Unit) {
    val folders by model.folders.collectAsState()
    var pickGym by remember { mutableStateOf(false) }
    val unfiled = stringResource(R.string.album_unfiled)
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        TextButton(onClick = { pickGym = true }, modifier = Modifier.fillMaxWidth()) { Text("${stringResource(R.string.album_gym)} · ${gymName.ifBlank { stringResource(R.string.album_choose_gym) }}") }
        ClimbDateField(stringResource(R.string.album_climbed_on), date, onDate)
        ChoiceRow(stringResource(R.string.album_folder), folders.firstOrNull { it.id == folderId }?.name ?: unfiled,
            listOf("" to unfiled) + folders.map { it.id to it.name }) { onFolder(it.takeIf(String::isNotEmpty)) }
    }
    if (pickGym) GymPicker(gymName, onSelect = { id, name -> onGym(id, name); pickGym = false }, onClose = { pickGym = false })
}

@Composable
private fun GymPicker(initialName: String, onSelect: (String?, String) -> Unit, onClose: () -> Unit) {
    var query by rememberSaveable { mutableStateOf(initialName) }
    val gyms = directoryGyms()
    val matches = remember(gyms, query) { gyms.filter { gym -> query.trim().split(Regex("\\s+")).all { token -> (listOf(gym.name, gym.city, gym.province, gym.address, gym.district) + gym.aliases).any { it.contains(token, true) } } } }
    AlertDialog(onDismissRequest = onClose, containerColor = BB.Paper, title = { Text(stringResource(R.string.album_choose_gym)) }, text = {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            OutlinedTextField(query, { query = it.take(120) }, label = { Text(stringResource(R.string.album_gym_search)) }, singleLine = true)
            Text(stringResource(R.string.album_directory_note), style = BB.Caption, color = BB.Muted)
            LazyColumn(Modifier.heightIn(max = 280.dp)) {
                items(matches, key = { it.id }) { gym ->
                    Column(Modifier.fillMaxWidth().clickable { onSelect(gym.id, gym.name) }.padding(vertical = 12.dp)) {
                        Text(gym.name, style = BB.Headline)
                        Text(listOf(gym.province, gym.city, gym.address.ifBlank { gym.district }).filter(String::isNotBlank).distinct().joinToString(" · "), style = BB.Caption, color = BB.Muted)
                    }
                }
            }
            if (query.isNotBlank()) TextButton(onClick = { onSelect(null, query.trim()) }) { Text(stringResource(R.string.album_use_typed, query.trim())) }
            TextButton(onClick = { onSelect(null, "") }) { Text(stringResource(R.string.album_no_gym)) }
        }
    }, confirmButton = { TextButton(onClick = onClose) { Text(stringResource(R.string.album_cancel)) } })
}

@Composable
internal fun OrganizeDialog(climb: Climb, model: JournalViewModel, onClose: () -> Unit) {
    var gymId by rememberSaveable { mutableStateOf(climb.gymId) }
    var gymName by rememberSaveable { mutableStateOf(climb.gymName.orEmpty()) }
    var date by rememberSaveable { mutableStateOf(climb.climbedAtMillis) }
    var folder by rememberSaveable { mutableStateOf(climb.folderId) }
    var grade by rememberSaveable { mutableStateOf(climb.grade) }
    var error by remember { mutableStateOf(false) }
    var busy by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()
    AlertDialog(onDismissRequest = { if (!busy) onClose() }, containerColor = BB.Paper, title = { Text(stringResource(R.string.album_organize)) }, text = {
        Column(Modifier.verticalScroll(rememberScrollState())) {
            GradePicker(grade) { grade = it }
            MetadataFields(model, gymId, gymName, date, folder, { id, name -> gymId = id; gymName = name }, { date = it }, { folder = it })
            if (error) Text(stringResource(R.string.album_save_error), color = DangerColor)
        }
    }, confirmButton = { TextButton(enabled = !busy, onClick = {
        busy = true; scope.launch { model.organize(climb, grade, gymId, gymName, date, folder).fold(onSuccess = { onClose() }, onFailure = { error = true }); busy = false }
    }) { Text(stringResource(R.string.album_save)) } }, dismissButton = { TextButton(enabled = !busy, onClick = onClose) { Text(stringResource(R.string.album_cancel)) } })
}

@Composable
internal fun AlbumFilterDialog(climbs: List<Climb>, initial: AlbumFilters, onApply: (AlbumFilters) -> Unit, onClose: () -> Unit) {
    var gym by rememberSaveable { mutableStateOf(initial.gymKey) }
    var grade by rememberSaveable { mutableStateOf(initial.grade) }
    var from by remember { mutableStateOf(initial.from) }
    var through by remember { mutableStateOf(initial.through) }
    val all = stringResource(R.string.album_any)
    val unknown = stringResource(R.string.album_no_gym)
    val directory = directoryGyms().associateBy { it.id }
    val gyms = listOf("*" to all, "" to unknown) + climbs.filter { it.gymKey.isNotEmpty() }.distinctBy { it.gymKey }.sortedBy { it.gymName }.map { it.gymKey to (directory[it.gymId]?.displayLabel() ?: it.gymName.orEmpty()) }
    AlertDialog(onDismissRequest = onClose, containerColor = BB.Paper, title = { Text(stringResource(R.string.album_filter)) }, text = {
        Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            ChoiceRow(stringResource(R.string.album_gym), gyms.firstOrNull { it.first == gym }?.second ?: all, gyms) { gym = it.takeUnless { it == "*" } }
            ChoiceRow(stringResource(R.string.journal_import_grade_label), grade?.let { gradeText(it) } ?: all, listOf("*" to all) + (gradeValues + climbs.map { it.grade }).distinct().map { it to gradeText(it) }) { grade = it.takeUnless { it == "*" } }
            Text(stringResource(R.string.album_climbed_on), style = BB.Headline)
            TextButton(onClick = { from = null; through = null }) { Text(all) }
            Row {
                TextButton(onClick = { through = LocalDate.now(); from = through!!.minusDays(6) }) { Text(stringResource(R.string.album_last_week)) }
                TextButton(onClick = { through = LocalDate.now(); from = through!!.minusDays(29) }) { Text(stringResource(R.string.album_last_month)) }
            }
            ClimbDateField(stringResource(R.string.album_from), from?.atStartOfDay(ZoneId.systemDefault())?.toInstant()?.toEpochMilli()) { from = Instant.ofEpochMilli(it).atZone(ZoneId.systemDefault()).toLocalDate() }
            ClimbDateField(stringResource(R.string.album_through), through?.atStartOfDay(ZoneId.systemDefault())?.toInstant()?.toEpochMilli()) { through = Instant.ofEpochMilli(it).atZone(ZoneId.systemDefault()).toLocalDate() }
            if (from == null && through == null) Text(all, style = BB.Caption, color = BB.Muted)
            if (from != null && through != null && from!! > through!!) Text(stringResource(R.string.album_date_error), color = DangerColor)
        }
    }, confirmButton = { TextButton(enabled = from == null || through == null || from!! <= through!!, onClick = { onApply(initial.copy(gymKey = gym, grade = grade, from = from, through = through)) }) { Text(stringResource(R.string.album_apply)) } }, dismissButton = { TextButton(onClick = onClose) { Text(stringResource(R.string.album_cancel)) } })
}
