package com.betabuddy.android.settings

import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalResources
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.betabuddy.android.BuildConfig
import com.betabuddy.android.R
import com.betabuddy.android.gym.GymNoticeList
import com.betabuddy.android.gym.GymPackStatus
import com.betabuddy.android.i18n.AppLanguage
import com.betabuddy.android.ui.common.BBIcons
import com.betabuddy.android.ui.common.CircleIconButton
import com.betabuddy.android.ui.common.Hairline
import com.betabuddy.android.ui.common.PaperPage
import com.betabuddy.android.ui.common.SectionLabel
import com.betabuddy.android.ui.theme.BB
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

private const val SUPPORT_URL = "https://betabuddy-support.dozycfs.chatgpt.site"
private const val SUPPORT_MAIL = "hi@dozycat.ai"

/** A settings page: back button and title on top, then the content. Fills the whole window. */
@Composable
private fun SettingsPage(title: String, onBack: () -> Unit, content: @Composable ColumnScope.() -> Unit) {
    PaperPage(insets = WindowInsets.safeDrawing) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(14.dp)) {
            CircleIconButton(
                icon = BBIcons.ArrowBack,
                contentDescription = stringResource(R.string.action_back),
                onClick = onBack,
                modifier = Modifier.testTag("back"),
            )
            Text(title, style = BB.Title2, modifier = Modifier.weight(1f).semantics { heading() })
        }
        content()
    }
}

/** One row that leads somewhere: a label and a chevron. */
@Composable
private fun SettingsRow(label: String, onClick: () -> Unit, modifier: Modifier = Modifier) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .clickable(role = Role.Button, onClick = onClick)
            .heightIn(min = 56.dp)
            .padding(vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(label, style = BB.Headline, modifier = Modifier.weight(1f))
        Icon(BBIcons.ChevronRight, contentDescription = null, tint = BB.Muted, modifier = Modifier.size(18.dp))
    }
}

/**
 * Follow the system, or one of the five languages. Each language is written in itself, so a wrong
 * choice can always be undone. Picking one applies at once (the activity is recreated in the new
 * language and stays on this page) and is remembered.
 */
@Composable
private fun LanguagePicker() {
    var selected by remember { mutableStateOf(AppLanguage.selected()) }
    val options: List<Pair<String?, String>> =
        listOf<Pair<String?, String>>(null to stringResource(R.string.language_follow_system)) +
            AppLanguage.supported.map { it to AppLanguage.nativeName(it) }
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text(stringResource(R.string.settings_language), style = BB.Title3, modifier = Modifier.semantics { heading() })
        Column(Modifier.clip(RoundedCornerShape(18.dp)).background(Color.White.copy(alpha = 0.6f)).selectableGroup()) {
            options.forEachIndexed { index, (code, name) ->
                val isSelected = selected == code
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .selectable(selected = isSelected, role = Role.RadioButton) {
                            if (!isSelected) {
                                selected = code
                                AppLanguage.select(code)
                            }
                        }
                        .heightIn(min = 52.dp)
                        .padding(horizontal = 16.dp, vertical = 8.dp)
                        .testTag("language-${code ?: "system"}"),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(name, style = BB.Body, modifier = Modifier.weight(1f))
                    if (isSelected) Icon(BBIcons.Check, contentDescription = null, tint = BB.Ink, modifier = Modifier.size(18.dp))
                }
                if (index != options.lastIndex) Hairline(Modifier.padding(start = 16.dp))
            }
        }
    }
}

/**
 * "Settings and About", one page as on iOS: the language first, then what the app is, what it does
 * not claim, and what stays on the phone. Mirrors ios/BetaBuddy/Views/AboutView.swift.
 */
@Composable
fun SettingsScreen(gym: GymPackStatus, onBack: () -> Unit, onOpenPrivacy: () -> Unit, onOpenLicenses: () -> Unit) {
    val context = LocalContext.current
    SettingsPage(stringResource(R.string.settings_title), onBack) {
        LanguagePicker()
        Hairline()
        Text("YOUR BETA.\nYOUR DEVICE.", style = BB.display(37.sp))
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Icon(BBIcons.Lock, contentDescription = null, tint = BB.Ink, modifier = Modifier.size(24.dp))
            Text(stringResource(R.string.about_on_device), style = BB.Title3)
        }
        Text(stringResource(R.string.about_free), style = BB.Subheadline.copy(fontWeight = FontWeight.Bold))
        Text(stringResource(R.string.about_local), style = BB.Body, color = BB.Muted)
        Hairline()
        SectionLabel(title = stringResource(R.string.about_limits_title), trailing = "KEEP IT HONEST")
        Text(stringResource(R.string.about_limits_body), style = BB.Body)
        Text(stringResource(R.string.about_video_requirements), style = BB.Body, color = BB.Muted)
        Hairline()
        if (gym is GymPackStatus.Ready) {
            val manifest = gym.pack.manifest
            GymNoticeList()
            Text(
                text = stringResource(
                    R.string.about_gym_pack_line,
                    manifest.displayName(AppLanguage.current()), manifest.routeCount, manifest.packVersion, manifest.generatedDate,
                ),
                style = BB.Caption,
                color = BB.Muted,
            )
        } else {
            SectionLabel(title = stringResource(R.string.tab_gym), trailing = "NOT IN THIS BUILD")
            Text(stringResource(R.string.about_gym_missing), style = BB.Body, color = BB.Muted)
        }
        Hairline()
        Text(stringResource(R.string.about_records_title), style = BB.Headline, modifier = Modifier.semantics { heading() })
        Text(stringResource(R.string.about_records_body), style = BB.Body, color = BB.Muted)
        Column {
            SettingsRow(stringResource(R.string.privacy_title), onOpenPrivacy, Modifier.testTag("privacyPolicy"))
            Hairline()
            SettingsRow(stringResource(R.string.licenses_title), onOpenLicenses, Modifier.testTag("licenses"))
            Hairline()
            // These two leave the app: the system opens them in the browser or mail app the user has.
            SettingsRow(stringResource(R.string.about_support), { context.view(Uri.parse(SUPPORT_URL)) })
            Hairline()
            Text(
                text = SUPPORT_MAIL,
                style = BB.Headline.copy(textDecoration = TextDecoration.Underline),
                modifier = Modifier
                    .clip(RoundedCornerShape(14.dp))
                    .clickable(role = Role.Button) { context.view(Uri.parse("mailto:$SUPPORT_MAIL")) }
                    .heightIn(min = 56.dp)
                    .padding(vertical = 16.dp),
            )
        }
        Text("BETABUDDY / ${BuildConfig.VERSION_NAME}\nBuilt for the next attempt.", style = BB.mono(12.sp, tracking = 0.sp))
    }
}

private fun Context.view(uri: Uri) {
    try {
        startActivity(Intent(Intent.ACTION_VIEW, uri).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
    } catch (error: ActivityNotFoundException) {
        // No browser or mail app on this device; nothing to open.
    }
}

/**
 * The privacy policy, bundled and read offline. Simplified and Traditional Chinese and English
 * are bundled (res/raw with language qualifiers); other languages read the English text.
 */
@Composable
fun PrivacyPolicyScreen(onBack: () -> Unit) {
    val resources = LocalResources.current
    val language = AppLanguage.current()
    val policy = remember(resources, language) {
        try {
            resources.openRawResource(R.raw.privacy_policy).use { it.readBytes().toString(Charsets.UTF_8) }.trim()
        } catch (error: Exception) {
            null
        }
    }
    SettingsPage(stringResource(R.string.privacy_title), onBack) {
        SelectionContainer {
            Text(policy ?: stringResource(R.string.privacy_unreadable), style = BB.Body.copy(lineHeight = 28.sp), modifier = Modifier.testTag("privacyPolicyText"))
        }
    }
}

/** The files under assets/licenses, in the order they are shown. */
private val LICENSE_FILES = listOf("components.txt", "Apache-2.0.txt", "MediaPipe-NOTICE.txt")

/**
 * The licenses of the third-party software in the app, bundled and read offline: the list of
 * components, the Apache License, and the notices of the code inside MediaPipe. The text is long
 * (the notices alone are over a megabyte), so it is read off the main thread and shown as a lazy
 * list of paragraphs. Licenses stay in their original English.
 */
@Composable
fun LicensesScreen(onBack: () -> Unit) {
    val assets = LocalContext.current.assets
    val paragraphs by produceState<List<String>?>(initialValue = null, assets) {
        value = withContext(Dispatchers.IO) {
            try {
                LICENSE_FILES.flatMap { name ->
                    val text = assets.open("licenses/$name").use { it.readBytes().toString(Charsets.UTF_8) }
                    listOf("— $name —") + text.replace("\r\n", "\n").split(Regex("\n\\s*\n")).map { it.trimEnd() }.filter { it.isNotBlank() }
                }
            } catch (error: Exception) {
                emptyList()
            }
        }
    }
    Column(Modifier.fillMaxSize().background(BB.Paper).windowInsetsPadding(WindowInsets.safeDrawing)) {
        Row(
            modifier = Modifier.padding(start = 24.dp, top = 24.dp, end = 24.dp, bottom = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            CircleIconButton(BBIcons.ArrowBack, stringResource(R.string.action_back), onBack, Modifier.testTag("back"))
            Text(stringResource(R.string.licenses_title), style = BB.Title2, modifier = Modifier.weight(1f).semantics { heading() })
        }
        val list = paragraphs
        when {
            list == null -> Unit
            list.isEmpty() -> Text(stringResource(R.string.licenses_unreadable), style = BB.Body, modifier = Modifier.padding(24.dp))
            else -> SelectionContainer {
                LazyColumn(
                    modifier = Modifier.fillMaxSize().testTag("licensesText"),
                    contentPadding = PaddingValues(start = 24.dp, end = 24.dp, bottom = 48.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    items(list.size) { index ->
                        val text = list[index]
                        val isTitle = text.startsWith("— ")
                        Text(text, style = if (isTitle) BB.Headline else BB.mono(11.sp, FontWeight.Normal, 0.sp), color = if (isTitle) BB.Ink else BB.Muted)
                    }
                }
            }
        }
    }
}
