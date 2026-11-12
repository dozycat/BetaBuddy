package com.betabuddy.android.ui.theme

import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp

/**
 * The BetaBuddy palette and type, the same values as `BB` in ios/BetaBuddy/Views/DesignSystem.swift.
 * Screens use these directly; Material 3 is only the base the components sit on.
 */
object BB {
    val Paper = Color(0xFFF5F2E8)
    val Ink = Color(0xFF1A1F1A)
    val Lime = Color(0xFFD1F24A)
    val Lavender = Color(0xFFD1CCF0)
    val Muted = Color(0xFF616659)
    val Line = Ink.copy(alpha = 0.13f)

    /** Extra-heavy display type for the big statements. */
    fun display(size: TextUnit): TextStyle =
        TextStyle(fontSize = size, lineHeight = size * 1.08f, fontWeight = FontWeight.Black, letterSpacing = (-0.01).em)

    /** Small monospaced labels: stamps, trailing section marks, version lines. They stay in English. */
    fun mono(size: TextUnit, weight: FontWeight = FontWeight.Medium, tracking: TextUnit = 1.sp): TextStyle =
        TextStyle(fontSize = size, lineHeight = size * 1.5f, fontWeight = weight, fontFamily = FontFamily.Monospace, letterSpacing = tracking)

    val Title = TextStyle(fontSize = 28.sp, lineHeight = 34.sp, fontWeight = FontWeight.Bold)
    val Title2 = TextStyle(fontSize = 22.sp, lineHeight = 28.sp, fontWeight = FontWeight.Bold)
    val Title3 = TextStyle(fontSize = 20.sp, lineHeight = 25.sp, fontWeight = FontWeight.Bold)
    val Headline = TextStyle(fontSize = 17.sp, lineHeight = 22.sp, fontWeight = FontWeight.SemiBold)
    val Body = TextStyle(fontSize = 17.sp, lineHeight = 24.sp, fontWeight = FontWeight.Normal)
    val Subheadline = TextStyle(fontSize = 15.sp, lineHeight = 22.sp, fontWeight = FontWeight.Normal)
    val Caption = TextStyle(fontSize = 12.sp, lineHeight = 16.sp, fontWeight = FontWeight.Normal)
}

private val Colors = lightColorScheme(
    primary = BB.Ink,
    onPrimary = BB.Paper,
    primaryContainer = BB.Lime,
    onPrimaryContainer = BB.Ink,
    secondary = BB.Ink,
    onSecondary = BB.Paper,
    secondaryContainer = BB.Lavender,
    onSecondaryContainer = BB.Ink,
    tertiary = BB.Ink,
    onTertiary = BB.Paper,
    tertiaryContainer = BB.Lavender,
    onTertiaryContainer = BB.Ink,
    background = BB.Paper,
    onBackground = BB.Ink,
    surface = BB.Paper,
    onSurface = BB.Ink,
    surfaceVariant = BB.Paper,
    onSurfaceVariant = BB.Muted,
    surfaceTint = BB.Paper,
    outline = BB.Ink.copy(alpha = 0.5f),
    outlineVariant = BB.Line,
)

private val Type = Typography().run {
    copy(
        headlineLarge = BB.display(34.sp),
        headlineMedium = BB.Title,
        headlineSmall = BB.Title2,
        titleLarge = BB.Title3,
        titleMedium = BB.Headline,
        bodyLarge = BB.Body,
        bodyMedium = BB.Subheadline,
        bodySmall = BB.Caption,
        labelLarge = BB.Headline,
    )
}

/** Light only, like the iOS app: warm paper, ink text. */
@Composable
fun BetaBuddyTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = Colors, typography = Type) {
        CompositionLocalProvider(LocalContentColor provides BB.Ink, content = content)
    }
}
