package com.betabuddy.android.pose

import com.betabuddy.android.data.ClimbStore
import com.betabuddy.android.data.Joint
import com.betabuddy.android.data.PoseAnalysis
import com.betabuddy.android.data.PoseFrame
import com.betabuddy.android.data.clockText
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Rules for the recognised poses and the saved analysis. */
class PoseMathTest {
    private fun frame(time: Double = 0.0) = PoseFrame(time, List(33) { Joint(0.5, 0.5, 1.0) })

    private fun PoseFrame.with(index: Int, change: (Joint) -> Joint) =
        copy(joints = joints.mapIndexed { i, joint -> if (i == index) change(joint) else joint })

    private fun analysis(frames: List<PoseFrame>, sampled: Int, aspect: Double = 1.0, duration: Double = 1.0) =
        PoseAnalysis(frames, sampled, duration, aspect)

    @Test fun partialBackgroundDetectionIsNotDrawn() {
        val pose = frame().with(23) { it.copy(y = -0.02) }
        val result = analysis(listOf(pose), 1)
        assertFalse(pose.hasReliableTorso)
        assertNull(result.nearestFrame(0.0))
    }

    @Test fun skeletonIsNotCarriedAcrossMissingDetections() {
        val result = analysis(listOf(frame(), frame(1.0)), 10)
        assertNull(result.nearestFrame(0.5))
        assertEquals(1.0, result.nearestFrame(0.91)!!.time, 0.0)
        assertEquals(0.0, result.nearestFrame(0.05)!!.time, 0.0)
    }

    @Test fun archiveRoundTripPreservesAnalysis() {
        val original = analysis(listOf(frame()), 3, aspect = 0.5625, duration = 0.7)
        val decoded = ClimbStore.json.decodeFromString(
            PoseAnalysis.serializer(), ClimbStore.json.encodeToString(PoseAnalysis.serializer(), original))
        assertEquals(original, decoded)
        assertEquals(33, decoded.frames.first().joints.size)
        assertEquals("01:01", clockText(61.0))
        assertEquals("00:00", clockText(Double.NaN))
    }

    @Test fun confidenceThresholdAndFrameBounds() {
        assertTrue(Joint(0.0, 1.0, 0.6).valid)
        assertFalse(Joint(0.5, 0.5, 0.599).valid)
        assertFalse(Joint(0.5, 1.0001, 1.0).valid)
        assertFalse(Joint(0.5, 0.5, Double.NaN).valid)
        assertFalse(Joint(Double.NaN, 0.5, 1.0).valid)
        assertFalse(PoseFrame(0.0, List(32) { Joint(0.5, 0.5, 1.0) }).hasReliableTorso)
        assertFalse(PoseFrame(0.0, emptyList()).hasReliableTorso)
    }

    @Test fun nearestFrameStopsAtTheGapLimit() {
        val result = analysis(listOf(frame(1.0)), 1)
        assertNotNull(result.nearestFrame(1.15))
        assertNull(result.nearestFrame(1.17))
        assertNotNull(result.nearestFrame(0.85))
        assertNull(result.nearestFrame(0.83))
        assertNull(analysis(emptyList(), 0).nearestFrame(0.0))
    }

    @Test fun timecodeIsAlwaysMinutesAndSeconds() {
        assertEquals("00:00", clockText(0.0))
        assertEquals("00:09", clockText(9.99))
        assertEquals("03:00", clockText(180.0))
        assertEquals("100:05", clockText(6005.0))
        assertEquals("00:00", clockText(-3.0))
        assertEquals("00:00", clockText(Double.NEGATIVE_INFINITY))
    }
}
