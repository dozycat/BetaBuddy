package com.betabuddy.android.ui.nav

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.WindowInsetsSides
import androidx.compose.foundation.layout.consumeWindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.only
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.saveable.rememberSaveableStateHolder
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.betabuddy.android.R
import com.betabuddy.android.gym.GymScreen
import com.betabuddy.android.gym.GymSession
import com.betabuddy.android.journal.JournalScreen
import com.betabuddy.android.settings.PrivacyPolicyScreen
import com.betabuddy.android.settings.SettingsScreen
import com.betabuddy.android.ui.common.BBIcons
import com.betabuddy.android.ui.theme.BB

/** The two entries, in order. The app opens on the gym. */
object Tabs {
    const val GYM = 0
    const val JOURNAL = 1
}

private object Routes {
    const val HOME = "home"
    const val SETTINGS = "settings"
    const val PRIVACY = "settings/privacy"
}

/**
 * The app's frame: the two-tab home, and "Settings and About" on top of it.
 * System back (with the predictive back animation) pops settings pages, then returns from the
 * logbook to the gym, then leaves the app.
 */
@Composable
fun AppShell(gym: GymSession, initialTab: Int = Tabs.GYM) {
    val navigation = rememberNavController()
    val openSettings = { navigation.open(Routes.SETTINGS) }

    // The gym page can ask for settings from its own menu (`openSettings`).
    if (gym.settingsRequested) {
        LaunchedEffect(Unit) {
            gym.consumeSettingsRequest()
            openSettings()
        }
    }

    NavHost(navController = navigation, startDestination = Routes.HOME, modifier = Modifier.fillMaxSize().background(BB.Paper)) {
        composable(Routes.HOME) {
            HomeScreen(gym = gym, initialTab = initialTab, onOpenSettings = openSettings)
        }
        composable(Routes.SETTINGS) {
            SettingsScreen(gym = gym.status, onBack = { navigation.popBackStack() }, onOpenPrivacy = { navigation.open(Routes.PRIVACY) })
        }
        composable(Routes.PRIVACY) { PrivacyPolicyScreen(onBack = { navigation.popBackStack() }) }
    }
}

private fun NavHostController.open(route: String) = navigate(route) { launchSingleTop = true }

@Composable
private fun HomeScreen(gym: GymSession, initialTab: Int, onOpenSettings: () -> Unit) {
    var tab by rememberSaveable { mutableIntStateOf(initialTab.coerceIn(Tabs.GYM, Tabs.JOURNAL)) }
    var barHeight by rememberSaveable { mutableIntStateOf(0) }
    val journalState = rememberSaveableStateHolder()

    // Registered before the logbook's own back handlers, so those win while it has somewhere to go back to.
    BackHandler(enabled = tab != Tabs.GYM) { tab = Tabs.GYM }

    Column(Modifier.fillMaxSize().background(BB.Paper)) {
        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                // The tab bar sits on the navigation area, so content above it has that much less
                // bottom inset left to deal with (navigation bar, and the part of the keyboard the bar covers).
                .consumeWindowInsets(PaddingValues(bottom = with(LocalDensity.current) { barHeight.toDp() })),
        ) {
            // Always composed: the gym page keeps its scene and selection while the logbook is in front.
            GymScreen(session = gym, visible = tab == Tabs.GYM, onOpenSettings = onOpenSettings)
            if (tab == Tabs.JOURNAL) {
                journalState.SaveableStateProvider("journal") {
                    Box(
                        Modifier
                            .fillMaxSize()
                            .background(BB.Paper)
                            .windowInsetsPadding(WindowInsets.safeDrawing.only(WindowInsetsSides.Top + WindowInsetsSides.Horizontal)),
                    ) {
                        JournalScreen(onOpenSettings = onOpenSettings)
                    }
                }
            }
        }
        TabBar(selected = tab, onSelect = { tab = it }, onOpenSettings = onOpenSettings, modifier = Modifier.onSizeChanged { barHeight = it.height })
    }
}

/** The two entries share the width; the gear at the right end opens settings from anywhere, the 3D gym page included. */
@Composable
private fun TabBar(selected: Int, onSelect: (Int) -> Unit, onOpenSettings: () -> Unit, modifier: Modifier = Modifier) {
    Column(modifier.fillMaxWidth().background(BB.Paper)) {
        Box(Modifier.fillMaxWidth().height(1.dp).background(BB.Line))
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .windowInsetsPadding(WindowInsets.safeDrawing.only(WindowInsetsSides.Bottom + WindowInsetsSides.Horizontal))
                .padding(top = 10.dp, bottom = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Row(Modifier.weight(1f).selectableGroup()) {
                TabItem(stringResource(R.string.tab_gym), BBIcons.Map, selected == Tabs.GYM, "tabGym") { onSelect(Tabs.GYM) }
                TabItem(stringResource(R.string.tab_journal), BBIcons.Stack, selected == Tabs.JOURNAL, "tabJournal") { onSelect(Tabs.JOURNAL) }
            }
            val settings = stringResource(R.string.settings_title)
            Box(
                modifier = Modifier
                    .width(64.dp)
                    .height(52.dp)
                    .clip(CircleShape)
                    .clickable(role = Role.Button, onClickLabel = settings, onClick = onOpenSettings)
                    .testTag("settings"),
                contentAlignment = Alignment.Center,
            ) {
                Icon(BBIcons.Gear, contentDescription = settings, tint = BB.Muted, modifier = Modifier.size(22.dp))
            }
        }
    }
}

@Composable
private fun RowScope.TabItem(title: String, icon: ImageVector, selected: Boolean, tag: String, onClick: () -> Unit) {
    val color = if (selected) BB.Ink else BB.Muted
    Column(
        modifier = Modifier
            .weight(1f)
            .selectable(selected = selected, role = Role.Tab, onClick = onClick)
            .padding(vertical = 2.dp)
            .testTag(tag),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(5.dp),
    ) {
        Box(
            modifier = Modifier.width(54.dp).height(30.dp).clip(CircleShape).background(if (selected) BB.Lime else Color.Transparent),
            contentAlignment = Alignment.Center,
        ) {
            Icon(icon, contentDescription = null, tint = color, modifier = Modifier.size(20.dp))
        }
        Text(
            text = title,
            style = BB.Caption.copy(fontSize = 11.sp, fontWeight = if (selected) FontWeight.Bold else FontWeight.Medium),
            color = color,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
    }
}
