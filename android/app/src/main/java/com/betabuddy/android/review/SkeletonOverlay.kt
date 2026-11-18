package com.betabuddy.android.review

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.unit.dp
import com.betabuddy.android.data.Joint
import com.betabuddy.android.data.PoseAnalysis
import com.betabuddy.android.ui.theme.BB

/**
 * Skeleton for the frame nearest to [time]. The canvas must have exactly the bounds of the
 * displayed picture (see [OverlayGeometry.fit]); then (x, y) in 0…1 is (x·w, y·h).
 * Nothing is drawn when there is no reliable frame within 0.16 s.
 */
@Composable
internal fun SkeletonOverlay(analysis: PoseAnalysis, time: Double, modifier: Modifier = Modifier) {
    Canvas(modifier.clearAndSetSemantics { }) {
        val frame = analysis.nearestFrame(time) ?: return@Canvas
        fun point(joint: Joint) = Offset((joint.x * size.width).toFloat(), (joint.y * size.height).toFloat())
        for ((start, end) in OverlayGeometry.bones(frame)) {
            drawLine(Color.Black.copy(alpha = 0.6f), point(start), point(end), 5.dp.toPx(), StrokeCap.Round)
            drawLine(BB.Lime, point(start), point(end), 2.5.dp.toPx(), StrokeCap.Round)
        }
        for (joint in OverlayGeometry.points(frame)) drawCircle(Color.White, 3.dp.toPx(), point(joint))
    }
}
