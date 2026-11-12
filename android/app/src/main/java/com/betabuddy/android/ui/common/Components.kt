package com.betabuddy.android.ui.common

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.WindowInsetsSides
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.only
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.betabuddy.android.ui.theme.BB

/**
 * A scrolling paper page: 24 dp margins, 28 dp between blocks. It keeps clear of the status bar,
 * cutouts and side insets by default; the bottom is the caller's (the tab bar already sits there).
 */
@Composable
fun PaperPage(
    modifier: Modifier = Modifier,
    insets: WindowInsets = WindowInsets.safeDrawing.only(WindowInsetsSides.Top + WindowInsetsSides.Horizontal),
    spacing: Dp = 28.dp,
    content: @Composable ColumnScope.() -> Unit,
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .background(BB.Paper)
            .windowInsetsPadding(insets)
            .verticalScroll(rememberScrollState())
            .padding(PaddingValues(start = 24.dp, top = 24.dp, end = 24.dp, bottom = 48.dp)),
        verticalArrangement = Arrangement.spacedBy(spacing),
        content = content,
    )
}

/** A small outlined capsule with an English brand mark in it. Not translated. */
@Composable
fun Stamp(text: String, modifier: Modifier = Modifier, color: Color = BB.Ink) {
    Text(
        text = text,
        style = BB.mono(10.sp, FontWeight.Bold, 1.3.sp),
        color = color,
        maxLines = 1,
        overflow = TextOverflow.Ellipsis,
        modifier = modifier
            .border(BorderStroke(1.dp, color.copy(alpha = 0.5f)), CircleShape)
            .padding(horizontal = 9.dp, vertical = 6.dp),
    )
}

/** The primary action: an ink block with paper text and an icon on the right. */
@Composable
fun BigButton(
    title: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    icon: ImageVector = BBIcons.ArrowUpRight,
    enabled: Boolean = true,
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(if (enabled) BB.Ink else BB.Ink.copy(alpha = 0.35f))
            .clickable(enabled = enabled, role = Role.Button, onClick = onClick)
            .padding(20.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(text = title, style = BB.Headline, color = BB.Paper, modifier = Modifier.weight(1f))
        Spacer(Modifier.width(12.dp))
        Icon(imageVector = icon, contentDescription = null, tint = BB.Paper, modifier = Modifier.size(22.dp))
    }
}

/** A section heading with a small English mark on the right. */
@Composable
fun SectionLabel(title: String, trailing: String, modifier: Modifier = Modifier) {
    Row(modifier = modifier.fillMaxWidth(), verticalAlignment = Alignment.Bottom) {
        Text(text = title, style = BB.Title3, modifier = Modifier.weight(1f).semantics { heading() })
        Spacer(Modifier.width(12.dp))
        Text(text = trailing, style = BB.mono(10.sp), color = BB.Muted, maxLines = 1)
    }
}

/** A big number with a caption under it. */
@Composable
fun Stat(value: String, label: String, modifier: Modifier = Modifier) {
    Column(modifier = modifier, verticalArrangement = Arrangement.spacedBy(5.dp)) {
        Text(text = value, style = BB.display(30.sp))
        Text(text = label, style = BB.Caption, color = BB.Muted)
    }
}

/** A round outlined icon button, 48 dp: the settings entry, back, and the like. */
@Composable
fun CircleIconButton(
    icon: ImageVector,
    contentDescription: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Box(
        modifier = modifier
            .size(48.dp)
            .clip(CircleShape)
            .border(BorderStroke(1.dp, BB.Line), CircleShape)
            .clickable(role = Role.Button, onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Icon(imageVector = icon, contentDescription = contentDescription, tint = BB.Ink, modifier = Modifier.size(22.dp))
    }
}

/** A filled block with rounded corners: lime for the invitation, lavender for states and notes. */
@Composable
fun ColorCard(
    color: Color,
    modifier: Modifier = Modifier,
    cornerRadius: Dp = 24.dp,
    padding: Dp = 24.dp,
    spacing: Dp = 18.dp,
    content: @Composable ColumnScope.() -> Unit,
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(cornerRadius))
            .background(color)
            .padding(padding),
        verticalArrangement = Arrangement.spacedBy(spacing),
        content = content,
    )
}

/** A one-pixel ink line. */
@Composable
fun Hairline(modifier: Modifier = Modifier) {
    Box(modifier.fillMaxWidth().height(1.dp).background(BB.Line))
}
