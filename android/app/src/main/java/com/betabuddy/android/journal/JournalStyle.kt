package com.betabuddy.android.journal

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.betabuddy.android.ui.theme.BB

// Logbook-only pieces that the shared design system (ui/theme, ui/common) does not have.
// Colours, type, PaperPage, Stamp, BigButton, SectionLabel, Stat and ColorCard come from there.

/** White at 60 %: rows and fields on paper, as on iOS. */
internal val CardColor = Color.White.copy(alpha = 0.6f)
internal val DangerColor = Color(0xFFB3261E)

/** Secondary action: outlined, at least 48 dp tall. */
@Composable
internal fun OutlineButton(title: String, modifier: Modifier = Modifier, color: Color = BB.Ink, enabled: Boolean = true, onClick: () -> Unit) {
    Box(
        modifier = modifier.defaultMinSize(minHeight = 48.dp).alpha(if (enabled) 1f else 0.4f).clip(RoundedCornerShape(14.dp))
            .border(1.dp, color.copy(alpha = 0.35f), RoundedCornerShape(14.dp))
            .clickable(enabled = enabled, role = Role.Button, onClick = onClick).padding(horizontal = 16.dp, vertical = 12.dp),
        contentAlignment = Alignment.Center,
    ) { Text(title, style = BB.Subheadline, fontWeight = FontWeight.SemiBold, color = color) }
}

/** Primary action inside a card: filled ink. */
@Composable
internal fun FilledButton(title: String, modifier: Modifier = Modifier, enabled: Boolean = true, onClick: () -> Unit) {
    Box(
        modifier = modifier.defaultMinSize(minHeight = 48.dp).alpha(if (enabled) 1f else 0.4f).clip(RoundedCornerShape(14.dp)).background(BB.Ink)
            .clickable(enabled = enabled, role = Role.Button, onClick = onClick).padding(horizontal = 16.dp, vertical = 12.dp),
        contentAlignment = Alignment.Center,
    ) { Text(title, style = BB.Subheadline, fontWeight = FontWeight.SemiBold, color = BB.Paper) }
}

/** Determinate progress. The percentage is read from the text next to it; the bar is decoration. */
@Composable
internal fun ProgressBar(progress: Double, modifier: Modifier = Modifier) {
    Box(modifier.fillMaxWidth().height(6.dp).clip(CircleShape).background(BB.Line).clearAndSetSemantics { }) {
        Box(Modifier.fillMaxHeight().fillMaxWidth(progress.coerceIn(0.0, 1.0).toFloat()).background(BB.Ink))
    }
}

internal enum class Glyph { Play, Pause, StepBack, StepForward }

/** Transport controls drawn locally: play, pause and the 0.1 s steps. At least 48 dp to touch. */
@Composable
internal fun GlyphButton(
    glyph: Glyph, description: String, onClick: () -> Unit, modifier: Modifier = Modifier,
    size: Dp = 48.dp, background: Color = Color.Transparent,
) {
    Box(
        modifier = modifier.size(size).clip(CircleShape).background(background)
            .clickable(role = Role.Button, onClick = onClick).semantics { contentDescription = description },
        contentAlignment = Alignment.Center,
    ) {
        Canvas(Modifier.size(16.dp)) {
            val w = this.size.width
            val h = this.size.height
            fun triangle(base: Float, tip: Float) = Path().apply {
                moveTo(base, h * 0.08f); lineTo(tip, h * 0.5f); lineTo(base, h * 0.92f); close()
            }
            when (glyph) {
                Glyph.Play -> drawPath(triangle(w * 0.18f, w * 0.95f), BB.Ink)
                Glyph.Pause -> {
                    drawRect(BB.Ink, Offset(w * 0.16f, h * 0.08f), Size(w * 0.24f, h * 0.84f))
                    drawRect(BB.Ink, Offset(w * 0.60f, h * 0.08f), Size(w * 0.24f, h * 0.84f))
                }
                Glyph.StepForward -> {
                    drawPath(triangle(w * 0.08f, w * 0.70f), BB.Ink)
                    drawRect(BB.Ink, Offset(w * 0.78f, h * 0.08f), Size(w * 0.14f, h * 0.84f))
                }
                Glyph.StepBack -> {
                    drawPath(triangle(w * 0.92f, w * 0.30f), BB.Ink)
                    drawRect(BB.Ink, Offset(w * 0.08f, h * 0.08f), Size(w * 0.14f, h * 0.84f))
                }
            }
        }
    }
}
