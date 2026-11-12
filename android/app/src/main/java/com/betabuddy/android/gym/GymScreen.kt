package com.betabuddy.android.gym

import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.WindowInsetsSides
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.only
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.key
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LifecycleStartEffect
import com.betabuddy.android.R
import com.betabuddy.android.i18n.AppLanguage
import com.betabuddy.android.ui.common.BBIcons
import com.betabuddy.android.ui.common.BigButton
import com.betabuddy.android.ui.common.CircleIconButton
import com.betabuddy.android.ui.common.ColorCard
import com.betabuddy.android.ui.common.PaperPage
import com.betabuddy.android.ui.common.SectionLabel
import com.betabuddy.android.ui.common.Stamp
import com.betabuddy.android.ui.theme.BB

/** What the gym feature is honest about. Shown before first entry and kept in About. */
val GymNoticeItems: List<Int> = listOf(
    R.string.gym_notice_1,
    R.string.gym_notice_2,
    R.string.gym_notice_3,
    R.string.gym_notice_4,
    R.string.gym_notice_5,
)

/**
 * The gym tab. [visible] is false while the logbook is in front: the page stays alive behind it
 * (so coming back does not reload it) but is hidden, paused and out of reach of touches.
 */
@Composable
fun GymScreen(session: GymSession, visible: Boolean, onOpenSettings: () -> Unit, modifier: Modifier = Modifier) {
    when (val status = session.status) {
        GymPackStatus.Missing -> if (visible) {
            GymUnavailable(
                title = stringResource(R.string.gym_missing_title),
                detail = stringResource(R.string.gym_missing_detail),
                onOpenSettings = onOpenSettings,
                modifier = modifier,
            )
        }
        is GymPackStatus.Invalid -> if (visible) {
            GymUnavailable(
                title = stringResource(R.string.gym_invalid_title),
                detail = describe(status.error),
                onOpenSettings = onOpenSettings,
                modifier = modifier,
            )
        }
        is GymPackStatus.Ready -> when {
            session.noticeSeen -> GymStage(session, visible, modifier)
            visible -> GymIntro(status.pack, onEnter = session::acknowledgeNotice, onOpenSettings = onOpenSettings, modifier = modifier)
        }
    }
}

@Composable
private fun describe(error: GymPackError): String = when (error) {
    GymPackError.UnreadableManifest -> stringResource(R.string.gym_error_manifest)
    is GymPackError.UnsupportedSchema -> stringResource(R.string.gym_error_schema, error.schema)
    is GymPackError.UnsafePath, is GymPackError.MissingFile -> stringResource(R.string.gym_error_files)
}

/** The stamp on the left, the settings entry on the right. */
@Composable
private fun GymHeader(stamp: String, onOpenSettings: () -> Unit) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Stamp(text = stamp)
        Spacer(Modifier.weight(1f))
        CircleIconButton(
            icon = BBIcons.Sliders,
            contentDescription = stringResource(R.string.settings_title),
            onClick = onOpenSettings,
            modifier = Modifier.testTag("openSettings"),
        )
    }
}

@Composable
private fun GymUnavailable(title: String, detail: String, onOpenSettings: () -> Unit, modifier: Modifier = Modifier) {
    PaperPage(modifier) {
        GymHeader("GYM / NOT IN THIS BUILD", onOpenSettings)
        ColorCard(BB.Lavender) {
            Icon(BBIcons.Map, contentDescription = null, tint = BB.Ink, modifier = Modifier.size(44.dp))
            Text(title, style = BB.Title, modifier = Modifier.testTag("gymUnavailable").semantics { heading() })
            Text(detail, style = BB.Body, color = BB.Muted)
        }
    }
}

@Composable
private fun GymIntro(pack: GymPack, onEnter: () -> Unit, onOpenSettings: () -> Unit, modifier: Modifier = Modifier) {
    val manifest = pack.manifest
    PaperPage(modifier) {
        GymHeader("GYM / UNOFFICIAL", onOpenSettings)
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(
                text = manifest.displayName(AppLanguage.current()),
                style = BB.display(26.sp),
                modifier = Modifier.testTag("gymName").semantics { heading() },
            )
            Text(
                text = stringResource(R.string.gym_pack_summary, manifest.routeCount, manifest.packVersion),
                style = BB.mono(12.sp, FontWeight.Bold, 0.sp),
            )
        }
        ColorCard(BB.Lime, cornerRadius = 22.dp, padding = 22.dp, spacing = 10.dp) {
            Text(stringResource(R.string.gym_intro_headline), style = BB.Title2)
            Text(stringResource(R.string.gym_intro_body), style = BB.Subheadline)
        }
        GymNoticeList()
        BigButton(title = stringResource(R.string.gym_enter), onClick = onEnter, modifier = Modifier.testTag("enterGym"))
    }
}

/** The five honesty notes, numbered. Full-size text; they are part of the screen, not fine print. */
@Composable
fun GymNoticeList(modifier: Modifier = Modifier) {
    Column(modifier, verticalArrangement = Arrangement.spacedBy(14.dp)) {
        SectionLabel(title = stringResource(R.string.gym_notice_title), trailing = "KEEP IT HONEST")
        GymNoticeItems.forEachIndexed { index, item ->
            Row(horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                Text(text = "%02d".format(index + 1), style = BB.display(15.sp), color = BB.Muted, modifier = Modifier.width(24.dp))
                Text(text = stringResource(item), style = BB.Subheadline, modifier = Modifier.weight(1f))
            }
        }
    }
}

@Composable
private fun GymStage(session: GymSession, visible: Boolean, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val language = AppLanguage.current()

    DisposableEffect(session, context) {
        session.attach(context)
        onDispose { session.detach(context) }
    }
    // Opens the page the first time; reloads it with the new `lang` after a language change.
    LaunchedEffect(session, language) { session.start(language) }
    LifecycleStartEffect(session, visible) {
        session.setActive(visible)
        onStopOrDispose { session.setActive(false) }
    }

    val phase = session.phase
    Box(modifier.fillMaxSize().background(BB.Paper)) {
        session.webView?.let { webView ->
            key(webView) {
                AndroidView(
                    factory = { viewContext ->
                        FrameLayout(viewContext).apply {
                            (webView.parent as? ViewGroup)?.removeView(webView)
                            addView(webView, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
                        }
                    },
                    update = { container -> container.visibility = if (visible) View.VISIBLE else View.INVISIBLE },
                    onRelease = { container -> container.removeAllViews() },
                    // Android WebView reports safe areas unevenly across versions, so the page is simply
                    // laid out inside the status bar and cutouts; the paper behind them is the same colour
                    // as the page's own top bar. The tab bar below already covers the navigation area.
                    modifier = Modifier
                        .fillMaxSize()
                        .windowInsetsPadding(WindowInsets.safeDrawing.only(WindowInsetsSides.Top + WindowInsetsSides.Horizontal))
                        .testTag("gymPage"),
                )
            }
        }
        if (visible) {
            when (phase) {
                GymSession.Phase.Idle, GymSession.Phase.Loading -> GymLoading()
                is GymSession.Phase.Failed -> GymFailed(phase.failure, onReload = { session.reload(language) })
                GymSession.Phase.Ready -> Unit
            }
        }
    }
}

/** Covers the page until it is ready, so there is no white flash and no half-drawn scene. */
@Composable
private fun GymLoading() {
    Column(
        modifier = Modifier.fillMaxSize().background(BB.Paper).blockTouches(),
        verticalArrangement = Arrangement.spacedBy(18.dp, Alignment.CenterVertically),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        CircularProgressIndicator(color = BB.Ink, trackColor = BB.Line, strokeWidth = 3.dp, modifier = Modifier.size(32.dp))
        Text(stringResource(R.string.gym_loading), style = BB.Headline)
        Text("ON YOUR PHONE", style = BB.mono(10.sp, FontWeight.Bold), color = BB.Muted)
    }
}

@Composable
private fun GymFailed(failure: GymFailure, onReload: () -> Unit) {
    val message = when (failure) {
        is GymFailure.Page -> failure.message.ifEmpty { stringResource(R.string.gym_failed_page) }
        is GymFailure.Load -> stringResource(R.string.gym_failed_load, failure.description)
        GymFailure.RendererGone -> stringResource(R.string.gym_failed_renderer)
    }
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(BB.Paper)
            .blockTouches()
            .windowInsetsPadding(WindowInsets.safeDrawing.only(WindowInsetsSides.Top + WindowInsetsSides.Horizontal))
            .padding(24.dp),
        contentAlignment = Alignment.Center,
    ) {
        ColorCard(BB.Lavender) {
            Icon(BBIcons.Alert, contentDescription = null, tint = BB.Ink, modifier = Modifier.size(44.dp))
            Text(stringResource(R.string.gym_failed_title), style = BB.Title, modifier = Modifier.semantics { heading() })
            Text(message, style = BB.Body, color = BB.Muted, modifier = Modifier.testTag("gymError"))
            BigButton(title = stringResource(R.string.action_reload), icon = BBIcons.Reload, onClick = onReload)
        }
    }
}

/** Keeps touches on a covering layer from reaching the web view underneath. */
private fun Modifier.blockTouches(): Modifier = pointerInput(Unit) {
    awaitPointerEventScope {
        while (true) awaitPointerEvent()
    }
}
