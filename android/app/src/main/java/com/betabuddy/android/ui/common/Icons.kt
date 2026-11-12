package com.betabuddy.android.ui.common

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathFillType
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.vector.PathBuilder
import androidx.compose.ui.graphics.vector.path
import androidx.compose.ui.unit.dp

/**
 * The app's own small icon set, drawn as 24 × 24 outlines so no icon font or library is needed.
 * Tint them with `Icon(tint = …)`.
 */
object BBIcons {
    private fun outline(name: String, strokeWidth: Float = 2f, block: PathBuilder.() -> Unit): ImageVector =
        ImageVector.Builder(name = name, defaultWidth = 24.dp, defaultHeight = 24.dp, viewportWidth = 24f, viewportHeight = 24f)
            .path(
                fill = null,
                stroke = SolidColor(Color.Black),
                strokeLineWidth = strokeWidth,
                strokeLineCap = StrokeCap.Round,
                strokeLineJoin = StrokeJoin.Round,
                pathBuilder = block,
            )
            .build()

    /** A folded map: the gym tab. */
    val Map: ImageVector by lazy {
        outline("Map") {
            moveTo(3f, 6.5f); lineTo(9f, 4f); lineTo(15f, 6.5f); lineTo(21f, 4f)
            lineTo(21f, 17.5f); lineTo(15f, 20f); lineTo(9f, 17.5f); lineTo(3f, 20f); close()
            moveTo(9f, 4f); lineTo(9f, 17.5f)
            moveTo(15f, 6.5f); lineTo(15f, 20f)
        }
    }

    /** A stack of cards: the logbook tab. */
    val Stack: ImageVector by lazy {
        outline("Stack") {
            moveTo(8.5f, 3f); lineTo(15.5f, 3f)
            moveTo(7f, 6f); lineTo(17f, 6f)
            moveTo(7.5f, 9f); lineTo(16.5f, 9f)
            arcTo(2f, 2f, 0f, false, true, 18.5f, 11f)
            lineTo(18.5f, 19f)
            arcTo(2f, 2f, 0f, false, true, 16.5f, 21f)
            lineTo(7.5f, 21f)
            arcTo(2f, 2f, 0f, false, true, 5.5f, 19f)
            lineTo(5.5f, 11f)
            arcTo(2f, 2f, 0f, false, true, 7.5f, 9f)
            close()
        }
    }

    /** Three sliders: settings. */
    val Sliders: ImageVector by lazy {
        outline("Sliders", strokeWidth = 1.8f) {
            moveTo(4f, 7f); lineTo(13f, 7f)
            moveTo(17f, 7f); lineTo(20f, 7f)
            moveTo(17f, 7f); arcTo(2f, 2f, 0f, true, true, 13f, 7f); arcTo(2f, 2f, 0f, true, true, 17f, 7f)
            moveTo(4f, 12f); lineTo(6f, 12f)
            moveTo(10f, 12f); lineTo(20f, 12f)
            moveTo(10f, 12f); arcTo(2f, 2f, 0f, true, true, 6f, 12f); arcTo(2f, 2f, 0f, true, true, 10f, 12f)
            moveTo(4f, 17f); lineTo(12f, 17f)
            moveTo(16f, 17f); lineTo(20f, 17f)
            moveTo(16f, 17f); arcTo(2f, 2f, 0f, true, true, 12f, 17f); arcTo(2f, 2f, 0f, true, true, 16f, 17f)
        }
    }

    /** A gear: settings, at the right end of the tab bar. */
    val Gear: ImageVector by lazy {
        outline("Gear", strokeWidth = 1.8f) {
            // Eight teeth around a ring, and the hole in the middle.
            moveTo(12f, 2.5f); lineTo(12f, 5f)
            moveTo(12f, 19f); lineTo(12f, 21.5f)
            moveTo(2.5f, 12f); lineTo(5f, 12f)
            moveTo(19f, 12f); lineTo(21.5f, 12f)
            moveTo(5.3f, 5.3f); lineTo(7.05f, 7.05f)
            moveTo(16.95f, 16.95f); lineTo(18.7f, 18.7f)
            moveTo(5.3f, 18.7f); lineTo(7.05f, 16.95f)
            moveTo(16.95f, 7.05f); lineTo(18.7f, 5.3f)
            moveTo(19f, 12f); arcTo(7f, 7f, 0f, true, true, 5f, 12f); arcTo(7f, 7f, 0f, true, true, 19f, 12f)
            moveTo(15f, 12f); arcTo(3f, 3f, 0f, true, true, 9f, 12f); arcTo(3f, 3f, 0f, true, true, 15f, 12f)
        }
    }

    val ArrowUpRight: ImageVector by lazy {
        outline("ArrowUpRight", strokeWidth = 2.4f) {
            moveTo(7f, 17f); lineTo(17f, 7f)
            moveTo(8.5f, 7f); lineTo(17f, 7f); lineTo(17f, 15.5f)
        }
    }

    val ArrowBack: ImageVector by lazy {
        outline("ArrowBack", strokeWidth = 2.2f) {
            moveTo(20f, 12f); lineTo(4.5f, 12f)
            moveTo(11f, 5.5f); lineTo(4.5f, 12f); lineTo(11f, 18.5f)
        }
    }

    val ChevronRight: ImageVector by lazy {
        outline("ChevronRight", strokeWidth = 2.2f) {
            moveTo(9.5f, 5.5f); lineTo(16f, 12f); lineTo(9.5f, 18.5f)
        }
    }

    val Reload: ImageVector by lazy {
        outline("Reload", strokeWidth = 2.2f) {
            moveTo(19.5f, 12f)
            arcTo(7.5f, 7.5f, 0f, true, true, 17.3f, 6.7f)
            moveTo(17.5f, 3f); lineTo(17.5f, 7f); lineTo(13.5f, 7f)
        }
    }

    val Check: ImageVector by lazy {
        outline("Check", strokeWidth = 2.4f) {
            moveTo(5f, 12.5f); lineTo(10f, 17.5f); lineTo(19f, 7f)
        }
    }

    val Plus: ImageVector by lazy {
        outline("Plus", strokeWidth = 2.4f) {
            moveTo(12f, 5f); lineTo(12f, 19f)
            moveTo(5f, 12f); lineTo(19f, 12f)
        }
    }

    /** A circle with an exclamation mark: something did not open. */
    val Alert: ImageVector by lazy {
        outline("Alert", strokeWidth = 1.6f) {
            moveTo(21f, 12f); arcTo(9f, 9f, 0f, true, true, 3f, 12f); arcTo(9f, 9f, 0f, true, true, 21f, 12f)
            moveTo(12f, 7.5f); lineTo(12f, 13f)
            moveTo(12f, 16.4f); lineTo(12f, 16.6f)
        }
    }

    /** A padlock: stays on this phone. */
    val Lock: ImageVector by lazy {
        outline("Lock", strokeWidth = 1.8f) {
            moveTo(7f, 11f); lineTo(17f, 11f)
            arcTo(2f, 2f, 0f, false, true, 19f, 13f)
            lineTo(19f, 19f)
            arcTo(2f, 2f, 0f, false, true, 17f, 21f)
            lineTo(7f, 21f)
            arcTo(2f, 2f, 0f, false, true, 5f, 19f)
            lineTo(5f, 13f)
            arcTo(2f, 2f, 0f, false, true, 7f, 11f)
            close()
            moveTo(8.5f, 11f); lineTo(8.5f, 7.5f)
            arcTo(3.5f, 3.5f, 0f, false, true, 15.5f, 7.5f)
            lineTo(15.5f, 11f)
        }
    }

    /** The BetaBuddy mark, an open climbing link on a 100 × 100 grid (ios/BetaBuddy/Views/BrandMark.swift). */
    val BrandMark: ImageVector by lazy {
        ImageVector.Builder(name = "BrandMark", defaultWidth = 24.dp, defaultHeight = 24.dp, viewportWidth = 100f, viewportHeight = 100f)
            .path(fill = SolidColor(Color.Black), pathFillType = PathFillType.NonZero) {
                moveTo(33f, 88f)
                curveTo(16f, 77f, 10f, 63f, 20f, 47f)
                lineTo(43f, 14f)
                curveTo(51f, 2f, 65f, 1f, 75f, 8f)
                lineTo(88f, 17f)
                curveTo(100f, 25f, 101f, 38f, 94f, 49f)
                lineTo(82f, 66f)
                lineTo(64f, 54f)
                lineTo(76f, 37f)
                quadTo(78f, 34f, 74f, 32f)
                lineTo(63f, 25f)
                quadTo(59f, 22f, 57f, 26f)
                lineTo(37f, 56f)
                quadTo(33f, 61f, 39f, 65f)
                lineTo(48f, 71f)
                quadTo(52f, 74f, 55f, 69f)
                lineTo(58f, 65f)
                lineTo(76f, 77f)
                lineTo(72f, 83f)
                curveTo(65f, 95f, 52f, 101f, 40f, 93f)
                close()
            }
            .build()
    }
}
