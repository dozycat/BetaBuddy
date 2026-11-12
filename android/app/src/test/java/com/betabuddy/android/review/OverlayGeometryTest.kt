package com.betabuddy.android.review

import com.betabuddy.android.data.Joint
import com.betabuddy.android.data.PoseAnalysis
import com.betabuddy.android.data.PoseFrame
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class OverlayGeometryTest {
    private fun frame(time: Double, hipY: Double = 0.5, hipConfidence: Double = 1.0) = PoseFrame(
        time, List(33) { index -> if (index == 23 || index == 24) Joint(0.5, hipY, hipConfidence) else Joint(0.5, 0.5, 1.0) },
    )
    private fun analysis(vararg frames: PoseFrame) = PoseAnalysis(frames.toList(), frames.size, 10.0, 0.5625, 1.0, emptyList())

    @Test fun pictureIsFittedInsideTheStage() {
        // Portrait clip in a wide stage: pillarboxed, full height.
        val (w, h) = OverlayGeometry.fit(360f, 340f, 9.0 / 16)
        assertEquals(340f, h, 0.01f)
        assertEquals(191.25f, w, 0.01f)
        // Landscape clip: letterboxed, full width.
        val (w2, h2) = OverlayGeometry.fit(360f, 340f, 16.0 / 9)
        assertEquals(360f, w2, 0.01f)
        assertEquals(202.5f, h2, 0.01f)
        assertEquals(0f to 0f, OverlayGeometry.fit(360f, 340f, Double.NaN))
        assertEquals(0f to 0f, OverlayGeometry.fit(0f, 340f, 1.0))
    }

    @Test fun lowConfidenceAndOutOfFramePointsAreNotDrawn() {
        val full = frame(0.0)
        assertEquals(18, OverlayGeometry.bones(full).size)
        assertEquals(22, OverlayGeometry.points(full).size) // 33 minus the 11 face points
        val joints = full.joints.toMutableList()
        joints[15] = Joint(0.5, 0.5, 0.3)   // left wrist unsure
        joints[28] = Joint(0.5, 1.2, 1.0)   // right ankle below the picture
        val partial = PoseFrame(0.0, joints)
        assertEquals(18 - 1 - 3, OverlayGeometry.bones(partial).size) // 13–15, and 26–28, 28–30, 28–32
        assertEquals(20, OverlayGeometry.points(partial).size)
        assertTrue(OverlayGeometry.bones(PoseFrame(0.0, emptyList())).isEmpty())
    }

    @Test fun trailCoversTheLastTwoSecondsOnly() {
        val result = analysis(frame(0.0), frame(0.1), frame(2.0), frame(2.1), frame(2.2), frame(3.0))
        val trail = OverlayGeometry.hipTrail(result, 2.2)
        assertEquals(listOf(3), trail.map { it.size })
    }

    @Test fun trailBreaksAtGapsAndUnreliableHips() {
        val result = analysis(
            frame(1.0), frame(1.1), frame(1.2),
            frame(1.3, hipConfidence = 0.2),     // occluded: breaks the line
            frame(1.4), frame(1.5),
            frame(2.0), frame(2.1),              // 0.5 s without samples: breaks the line
        )
        val trail = OverlayGeometry.hipTrail(result, 2.1)
        assertEquals(listOf(3, 2, 2), trail.map { it.size })
        assertTrue(OverlayGeometry.hipTrail(result, 0.5).isEmpty())
        // Samples up to 0.3 s apart still connect, as on iOS.
        assertEquals(listOf(2), OverlayGeometry.hipTrail(analysis(frame(1.0), frame(1.25)), 1.25).map { it.size })
    }
}
