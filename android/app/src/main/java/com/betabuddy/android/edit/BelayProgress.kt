package com.betabuddy.android.edit

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.ProgressBarRangeInfo
import androidx.compose.ui.semantics.progressBarRangeInfo
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.unit.dp
import com.betabuddy.android.R
import com.betabuddy.android.ui.theme.BB
import java.text.NumberFormat

/** The hold rises with the renderer's actual progress; no looping or invented progress. */
@Composable
internal fun BelayProgress(progress: Double) {
    val fraction = if (progress.isFinite()) progress.coerceIn(0.0, 1.0).toFloat() else 0f
    val label = stringResource(R.string.journal_edit_working)
    BoxWithConstraints {
        val ropeHeight = (maxHeight - (70 * LocalDensity.current.fontScale).dp).coerceIn(0.dp, 210.dp)
        Column(
            Modifier.semantics(mergeDescendants = true) { contentDescription = label; progressBarRangeInfo = ProgressBarRangeInfo(fraction, 0f..1f) },
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(18.dp),
        ) {
            if (ropeHeight >= 64.dp) Canvas(Modifier.size(140.dp, ropeHeight)) {
                val top = 24.dp.toPx()
                val bottom = size.height - 26.dp.toPx()
                val left = size.width * .35f
                val right = size.width * .65f
                val y = bottom - (bottom - top) * fraction
                val rope = Path().apply {
                    moveTo(left, bottom)
                    lineTo(left, top)
                    quadraticTo(size.width / 2, -4.dp.toPx(), right, top)
                    lineTo(right, bottom)
                }
                drawPath(rope, BB.Paper.copy(alpha = .28f), style = Stroke(width = 3.dp.toPx(), cap = StrokeCap.Round))
                drawLine(BB.Lime, Offset(left, bottom), Offset(left, y), 3.dp.toPx(), StrokeCap.Round)
                drawCircle(BB.Paper, 3.5.dp.toPx(), Offset(size.width / 2, 10.dp.toPx()))
                val scale = 18.dp.toPx()
                val hold = Path().apply {
                    moveTo(left - scale, y - scale * .25f)
                    quadraticTo(left - scale * .8f, y - scale, left, y - scale)
                    quadraticTo(left + scale * .8f, y - scale, left + scale * .85f, y - scale * .15f)
                    quadraticTo(left + scale, y + scale * .65f, left + scale * .1f, y + scale)
                    quadraticTo(left - scale, y + scale, left - scale, y - scale * .25f)
                    close()
                }
                drawPath(hold, BB.Lime)
                drawCircle(BB.Ink, 3.dp.toPx(), Offset(left, y))
            }
            Text(NumberFormat.getPercentInstance().format(fraction), style = BB.Headline, color = BB.Paper)
        }
    }
}
