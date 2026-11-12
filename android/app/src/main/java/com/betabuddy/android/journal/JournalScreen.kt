package com.betabuddy.android.journal

import android.graphics.BitmapFactory
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.betabuddy.android.R
import com.betabuddy.android.data.Climb
import com.betabuddy.android.data.UNGRADED_VALUE
import com.betabuddy.android.data.clockText
import com.betabuddy.android.review.ReviewScreen
import com.betabuddy.android.ui.common.BBIcons
import com.betabuddy.android.ui.common.BigButton
import com.betabuddy.android.ui.common.CircleIconButton
import com.betabuddy.android.ui.common.ColorCard
import com.betabuddy.android.ui.common.PaperPage
import com.betabuddy.android.ui.theme.BB
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

private const val ROUTE_LIST = "list"
private const val ROUTE_IMPORT = "import"
private const val ROUTE_REVIEW = "review:"

/**
 * The logbook: the user's own climbs. List → import → replay, all inside this one entry.
 * Counterpart of JournalView / ImportView / ReviewView on iOS.
 */
@Composable
fun JournalScreen(onOpenSettings: () -> Unit) {
    val model: JournalViewModel = viewModel()
    var route by rememberSaveable { mutableStateOf(ROUTE_LIST) }
    val climbs by model.climbs.collectAsState()
    val unreadable by model.storeUnreadable.collectAsState()

    when {
        route == ROUTE_IMPORT -> ImportScreen(
            model = model,
            onClose = { model.closeImport(); route = ROUTE_LIST },
            onSaved = { id -> model.closeImport(); route = ROUTE_REVIEW + id },
        )
        route.startsWith(ROUTE_REVIEW) -> {
            val id = route.removePrefix(ROUTE_REVIEW)
            val list = climbs
            val climb = list?.firstOrNull { it.id == id }
            BackHandler { route = ROUTE_LIST }
            when {
                climb != null -> ReviewScreen(climb = climb, model = model, onBack = { route = ROUTE_LIST })
                list != null -> LaunchedEffect(id) { route = ROUTE_LIST } // deleted, or gone after a restore
                else -> Box(Modifier.fillMaxSize().background(BB.Paper))
            }
        }
        else -> JournalList(
            climbs = climbs, unreadable = unreadable, model = model,
            onOpenSettings = onOpenSettings,
            onImport = { route = ROUTE_IMPORT },
            onOpen = { route = ROUTE_REVIEW + it.id },
        )
    }
}

@Composable
private fun JournalList(
    climbs: List<Climb>?, unreadable: Boolean, model: JournalViewModel,
    onOpenSettings: () -> Unit, onImport: () -> Unit, onOpen: (Climb) -> Unit,
) {
    var filter by rememberSaveable { mutableStateOf(JournalFilter.ALL) }
    var search by rememberSaveable { mutableStateOf("") }
    val ungraded = stringResource(R.string.journal_ungraded)

    PaperPage {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(BBIcons.BrandMark, contentDescription = null, tint = BB.Ink, modifier = Modifier.size(30.dp))
            Spacer(Modifier.width(6.dp))
            Text("betabuddy", style = BB.display(23.sp), modifier = Modifier.weight(1f)) // brand mark, not translated
            CircleIconButton(BBIcons.Sliders, stringResource(R.string.journal_about_cd), onOpenSettings, Modifier.testTag("about"))
        }
        Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
            Text(stringResource(R.string.journal_title), style = BB.display(34.sp))
            Text(stringResource(R.string.journal_tagline), style = BB.Body, color = BB.Muted)
        }
        if (unreadable) {
            Text(stringResource(R.string.journal_store_unreadable), style = BB.Subheadline, color = DangerColor)
        }
        when {
            climbs == null -> Unit // index still loading: show neither the empty state nor a list
            climbs.isEmpty() -> {
                // No demo records are injected: the empty state leads to the first import.
                ColorCard(BB.Lavender) {
                    Text(stringResource(R.string.journal_empty_title), style = BB.Title)
                    Text(stringResource(R.string.journal_empty_body), style = BB.Body, color = BB.Muted)
                    BigButton(stringResource(R.string.journal_empty_action), onImport, Modifier.testTag("importClimb"), icon = BBIcons.Plus)
                }
                LocalNote()
            }
            else -> {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    BigButton(stringResource(R.string.journal_import_action), onImport, Modifier.testTag("importClimb"), icon = BBIcons.Plus)
                    LocalNote()
                }
                FilterBar(filter) { filter = it }
                SearchField(search) { search = it }
                val visible = filterClimbs(climbs, filter, search) { if (it == UNGRADED_VALUE) ungraded else it }
                if (visible.isEmpty()) {
                    ColorCard(BB.Lavender, spacing = 12.dp) {
                        Text(stringResource(R.string.journal_no_match_title), style = BB.Title3)
                        Text(stringResource(R.string.journal_no_match_body), style = BB.Body, color = BB.Muted)
                    }
                }
                for (climb in visible) {
                    ClimbRow(climb, model.thumbnailFile(climb)) { onOpen(climb) }
                }
            }
        }
    }
}

@Composable
private fun LocalNote() {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Icon(BBIcons.Lock, contentDescription = null, tint = BB.Muted, modifier = Modifier.size(14.dp))
        Spacer(Modifier.width(6.dp))
        Text(stringResource(R.string.journal_local_note), style = BB.Caption, fontWeight = FontWeight.Medium, color = BB.Muted)
    }
}

/** All / Sent / Projecting. Selection is shown by fill and weight, and announced as a tab-like choice. */
@Composable
private fun FilterBar(selected: JournalFilter, onSelect: (JournalFilter) -> Unit) {
    val description = stringResource(R.string.journal_filter_cd)
    Row(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(BB.Line).padding(3.dp)
            .selectableGroup().semantics { contentDescription = description },
    ) {
        val labels = listOf(
            JournalFilter.ALL to R.string.journal_filter_all,
            JournalFilter.SENT to R.string.journal_filter_sent,
            JournalFilter.PROJECTING to R.string.journal_filter_projecting,
        )
        for ((value, label) in labels) {
            val active = value == selected
            Box(
                Modifier.weight(1f).defaultMinSize(minHeight = 44.dp).clip(RoundedCornerShape(11.dp))
                    .background(if (active) BB.Paper else Color.Transparent)
                    .selectable(selected = active, role = Role.Tab, onClick = { onSelect(value) })
                    .padding(horizontal = 6.dp, vertical = 8.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    stringResource(label), style = BB.Subheadline, maxLines = 1, overflow = TextOverflow.Ellipsis,
                    fontWeight = if (active) FontWeight.Bold else FontWeight.Normal,
                )
            }
        }
    }
}

@Composable
private fun SearchField(value: String, onChange: (String) -> Unit) {
    val hint = stringResource(R.string.journal_search_hint)
    Box(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(CardColor).padding(horizontal = 14.dp, vertical = 14.dp),
        contentAlignment = Alignment.CenterStart,
    ) {
        if (value.isEmpty()) Text(hint, style = BB.Body, color = BB.Muted)
        BasicTextField(
            value = value, onValueChange = { onChange(it.take(80)) }, singleLine = true,
            textStyle = BB.Body.copy(color = BB.Ink), cursorBrush = SolidColor(BB.Ink),
            modifier = Modifier.fillMaxWidth().testTag("journalSearch").semantics { contentDescription = hint },
        )
    }
}

@Composable
private fun ClimbRow(climb: Climb, thumbnail: File, onClick: () -> Unit) {
    val image by produceState<ImageBitmap?>(null, thumbnail.path) {
        value = withContext(Dispatchers.IO) { runCatching { BitmapFactory.decodeFile(thumbnail.path)?.asImageBitmap() }.getOrNull() }
    }
    Row(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(22.dp)).background(CardColor)
            .clickable(role = Role.Button, onClick = onClick).padding(14.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(Modifier.size(width = 84.dp, height = 106.dp).clip(RoundedCornerShape(14.dp)).background(BB.Lavender)) {
            image?.let { Image(it, contentDescription = null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize()) }
        }
        Spacer(Modifier.width(16.dp))
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(9.dp)) {
            Text(climb.title, style = BB.Headline, maxLines = 2, overflow = TextOverflow.Ellipsis)
            Text("${gradeText(climb.grade)}  /  ${clockText(climb.analysis.duration)}", style = BB.mono(12.sp, FontWeight.Bold, 0.sp))
            Text(dateText(climb.dateMillis, withYear = false), style = BB.Caption, color = BB.Muted)
            // The status is words plus a mark, never colour alone.
            Text(
                stringResource(if (climb.sent) R.string.journal_row_sent else R.string.journal_row_projecting),
                style = BB.Caption, fontWeight = FontWeight.Bold,
            )
        }
        Icon(BBIcons.ArrowUpRight, contentDescription = null, tint = BB.Ink, modifier = Modifier.size(18.dp))
    }
}
