package com.betabuddy.android.review

import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.key
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.compose.ui.window.DialogWindowProvider
import androidx.core.view.WindowCompat
import com.betabuddy.android.R
import com.betabuddy.android.journal.AnalysisDesk
import com.betabuddy.android.ui.common.BBIcons
import com.betabuddy.android.ui.common.BigButton
import com.betabuddy.android.ui.common.ColorCard
import com.betabuddy.android.ui.theme.BB

/**
 * The pack's analysis page, full screen over the replay (tab bar included). The app's own part is
 * one bar: what it is doing while it takes pictures from the clip, and Done. Everything below the
 * bar is the page. System back closes it, like Done.
 *
 * As with the gym page, the web view sits inside the system bars and is told nothing about them.
 */
@Composable
internal fun AnalysisScreen(session: AnalysisSession, onDone: () -> Unit) {
    DisposableEffect(session) {
        // From here on this visit does the work for this record: the queue that analyses camera
        // records out of sight stops its own run of it, so one record is never analysed twice at once.
        AnalysisDesk.enter(session.climbId)
        session.open()
        onDispose {
            session.close()
            AnalysisDesk.leave(session.climbId)
        }
    }
    Dialog(
        onDismissRequest = onDone,
        properties = DialogProperties(usePlatformDefaultWidth = false, decorFitsSystemWindows = false, dismissOnClickOutside = false),
    ) {
        // The dialog is a window of its own: paper under both bars, dark icons, nothing dimmed behind it.
        val view = LocalView.current
        SideEffect {
            (view.parent as? DialogWindowProvider)?.window?.let { window ->
                window.setDimAmount(0f)
                WindowCompat.getInsetsController(window, view).apply {
                    isAppearanceLightStatusBars = true
                    isAppearanceLightNavigationBars = true
                }
            }
        }
        Column(Modifier.fillMaxSize().background(BB.Paper).windowInsetsPadding(WindowInsets.safeDrawing).testTag("analysisScreen")) {
            Row(Modifier.fillMaxWidth().padding(start = 16.dp, end = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                Text(
                    if (session.preparing) stringResource(R.string.journal_analysis_preparing) else "",
                    style = BB.Caption, color = BB.Muted,
                    modifier = Modifier.weight(1f).semantics { liveRegion = LiveRegionMode.Polite },
                )
                Box(
                    Modifier.defaultMinSize(minWidth = 48.dp, minHeight = 48.dp).clip(CircleShape)
                        .clickable(role = Role.Button, onClick = onDone).padding(horizontal = 12.dp).testTag("analysisDone"),
                    contentAlignment = Alignment.Center,
                ) { Text(stringResource(R.string.journal_import_done), style = BB.Subheadline, fontWeight = FontWeight.Bold) }
            }
            Box(Modifier.weight(1f).fillMaxWidth()) {
                session.webView?.let { webView ->
                    key(webView) {
                        AndroidView(
                            factory = { viewContext ->
                                FrameLayout(viewContext).apply {
                                    (webView.parent as? ViewGroup)?.removeView(webView)
                                    addView(webView, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
                                }
                            },
                            onRelease = { container -> container.removeAllViews() },
                            modifier = Modifier.fillMaxSize().testTag("analysisPage"),
                        )
                    }
                }
                val failure = session.failure
                when {
                    failure != null -> AnalysisFailed(failure, onReload = session::reload)
                    // Covers the page until its script is up, so there is no white flash.
                    session.loading -> Box(Modifier.fillMaxSize().background(BB.Paper).blockTouches(), contentAlignment = Alignment.Center) {
                        CircularProgressIndicator(color = BB.Ink, trackColor = BB.Line, strokeWidth = 3.dp, modifier = Modifier.size(32.dp))
                    }
                }
            }
        }
    }
}

@Composable
private fun AnalysisFailed(failure: AnalysisSession.Failure, onReload: () -> Unit) {
    // What the system or the page said about a failed load is shown as it is; it has no translation.
    val detail = when (failure) {
        AnalysisSession.Failure.Page, AnalysisSession.Failure.RendererGone -> ""
        is AnalysisSession.Failure.Load -> failure.description
        is AnalysisSession.Failure.Video -> stringResource(failure.message)
    }
    Box(Modifier.fillMaxSize().background(BB.Paper).blockTouches().padding(24.dp), contentAlignment = Alignment.Center) {
        ColorCard(BB.Lavender) {
            Icon(BBIcons.Alert, contentDescription = null, tint = BB.Ink, modifier = Modifier.size(44.dp))
            Text(
                stringResource(R.string.journal_analysis_failed_title), style = BB.Title,
                modifier = Modifier.testTag("analysisFailed").semantics { heading(); liveRegion = LiveRegionMode.Polite },
            )
            if (detail.isNotEmpty()) Text(detail, style = BB.Body, color = BB.Muted, modifier = Modifier.testTag("analysisError"))
            // A clip that gives no pictures will not give any on a second try; the page itself can be loaded again.
            if (failure !is AnalysisSession.Failure.Video) {
                BigButton(title = stringResource(R.string.action_reload), icon = BBIcons.Reload, onClick = onReload)
            }
        }
    }
}

/** Keeps touches on a covering layer from reaching the web view underneath. */
private fun Modifier.blockTouches(): Modifier = pointerInput(Unit) {
    awaitPointerEventScope {
        while (true) awaitPointerEvent()
    }
}
