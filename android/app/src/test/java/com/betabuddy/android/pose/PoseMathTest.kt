package com.betabuddy.android.pose

import com.betabuddy.android.data.ClimbStore
import com.betabuddy.android.data.Joint
import com.betabuddy.android.data.Moment
import com.betabuddy.android.data.PoseAnalysis
import com.betabuddy.android.data.PoseFrame
import com.betabuddy.android.data.clockText
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Ported case by case from ios/CoreTests/PoseMathTests.swift. Same inputs, same expectations. */
class PoseMathTest {
    private fun frame(time: Double = 0.0) = PoseFrame(time, List(33) { Joint(0.5, 0.5, 1.0) })

    private fun PoseFrame.with(index: Int, change: (Joint) -> Joint) =
        copy(joints = joints.mapIndexed { i, joint -> if (i == index) change(joint) else joint })

    private fun analysis(frames: List<PoseFrame>, sampled: Int, aspect: Double = 1.0, duration: Double = 1.0) =
        PoseAnalysis(frames, sampled, duration, aspect, 1.0, emptyList())

    @Test fun partialBackgroundDetectionIsNotDrawn() {
        val pose = frame().with(23) { it.copy(y = -0.02) }
        val result = analysis(listOf(pose), 1)
        assertFalse(pose.hasReliableTorso)
        assertNull(result.nearestFrame(0.0))
    }

    @Test fun anglesRespectVideoAspectRatio() {
        val pose = frame()
            .with(11) { Joint(0.25, 0.25, 1.0) }
            .with(13) { Joint(0.5, 0.5, 1.0) }
            .with(15) { Joint(0.75, 0.25, 1.0) }
        assertEquals(90.0, PoseMath.angle(pose, 11, 13, 15, 1.0)!!, 0.001)
        assertEquals(126.8698976, PoseMath.angle(pose, 11, 13, 15, 2.0)!!, 0.001)
    }

    @Test fun unknownAndDegenerateAnglesAreAbsent() {
        var pose = frame()
        assertNull(PoseMath.angle(pose, 11, 13, 15, 1.0)) // all three points coincide
        pose = pose.with(11) { it.copy(confidence = 0.1) }
        assertNull(PoseMath.angle(pose, 11, 13, 15, 1.0))
        assertNull(PoseMath.angle(pose, -1, 13, 15, 1.0))
        assertNull(PoseMath.angle(pose, 11, 13, 15, Double.NaN))
        // Not in the iOS file, same contract: out-of-range index, zero / negative / infinite aspect.
        assertNull(PoseMath.angle(frame(), 11, 13, 33, 1.0))
        assertNull(PoseMath.angle(frame(), 11, 13, 15, 0.0))
        assertNull(PoseMath.angle(frame(), 11, 13, 15, -1.0))
        assertNull(PoseMath.angle(frame(), 11, 13, 15, Double.POSITIVE_INFINITY))
    }

    @Test fun coverageCountsMissingDetections() {
        val result = analysis(listOf(frame(), frame(1.0)), 10)
        assertEquals(0.2, result.coverage, 0.0)
        assertNull(result.nearestFrame(0.5))
        assertEquals(1.0, result.nearestFrame(0.91)!!.time, 0.0)
        assertEquals(0.0, result.nearestFrame(0.05)!!.time, 0.0)
    }

    @Test fun noVelocityAcrossTrackingGaps() {
        val a = frame().with(23) { it.copy(y = 0.9) }.with(24) { it.copy(y = 0.9) }
            .with(29) { it.copy(confidence = 0.0) }.with(30) { it.copy(confidence = 0.0) }
        val b = frame(1.0).with(23) { it.copy(y = 0.1) }.with(24) { it.copy(y = 0.1) }
        assertTrue(PoseMath.moments(listOf(a, b), 1.0).isEmpty())
    }

    @Test fun raisedHeelCandidatesAreDebounced() {
        val frames = (0 until 20).map { index -> frame(index / 10.0).with(29) { it.copy(y = 0.2) } }
        val moments = PoseMath.moments(frames, 1.0)
        assertEquals(1, moments.size)
        assertTrue(moments[0].title.contains("候选"))
        assertEquals(Moment.RAISED_HEEL, moments[0].kind)
    }

    @Test fun invalidCoordinatesCannotProduceHip() {
        assertNull(frame().with(23) { it.copy(x = Double.NaN) }.hip)
        assertNull(frame().with(23) { it.copy(x = 1.1) }.hip)
    }

    @Test fun archiveRoundTripPreservesAnalysis() {
        val original = analysis(listOf(frame()), 3, aspect = 0.5625, duration = 0.7)
        val decoded = ClimbStore.json.decodeFromString(
            PoseAnalysis.serializer(), ClimbStore.json.encodeToString(PoseAnalysis.serializer(), original))
        assertEquals(33, decoded.frames.first().joints.size)
        assertEquals(1.0 / 3, decoded.coverage, 0.0001)
        assertEquals("01:01", clockText(61.0))
        assertEquals("00:00", clockText(Double.NaN))
    }

    // The cases below are not in the iOS test file; they pin the remaining rules of PoseMath.swift.

    @Test fun fastMoveIsACandidateOnlyBetweenCloseSamples() {
        val a = frame(0.0)
        val b = frame(0.1).with(23) { it.copy(y = 0.4) }.with(24) { it.copy(y = 0.4) }
        val moments = PoseMath.moments(listOf(a, b), 1.0) // 0.1 of the frame height in 0.1 s
        assertEquals(1, moments.size)
        assertEquals(Moment.FAST_MOVE, moments[0].kind)
        assertEquals(0.1, moments[0].time, 0.0)
        val slow = frame(0.1).with(23) { it.copy(y = 0.48) }.with(24) { it.copy(y = 0.48) }
        assertTrue(PoseMath.moments(listOf(a, slow), 1.0).isEmpty())
    }

    @Test fun horizontalSpeedIsScaledByAspect() {
        val a = frame(0.0)
        val b = frame(0.1).with(23) { it.copy(x = 0.53) }.with(24) { it.copy(x = 0.53) }
        assertTrue(PoseMath.moments(listOf(a, b), 1.0).isEmpty())   // 0.3 / s
        assertEquals(1, PoseMath.moments(listOf(a, b), 2.0).size)   // 0.6 / s once width counts double
    }

    @Test fun lowConfidenceHeelIsIgnoredAndCandidatesAreCapped() {
        val hidden = frame().with(29) { Joint(0.5, 0.2, 0.59) }
        assertTrue(PoseMath.moments(listOf(hidden), 1.0).isEmpty())
        val many = (0 until 40).map { index -> frame(index * 3.0).with(30) { it.copy(y = 0.2) } }
        val moments = PoseMath.moments(many, 1.0)
        assertEquals(24, moments.size)
        assertEquals(moments.map { it.time }.sorted(), moments.map { it.time })
    }

    @Test fun confidenceThresholdAndFrameBounds() {
        assertTrue(Joint(0.0, 1.0, 0.6).valid)
        assertFalse(Joint(0.5, 0.5, 0.599).valid)
        assertFalse(Joint(0.5, 1.0001, 1.0).valid)
        assertFalse(Joint(0.5, 0.5, Double.NaN).valid)
        assertFalse(PoseFrame(0.0, List(32) { Joint(0.5, 0.5, 1.0) }).hasReliableTorso)
        assertNull(PoseFrame(0.0, emptyList()).hip)
    }

    @Test fun averageElbowAngleUsesOnlyValidArms() {
        assertNull(analysis(listOf(frame()), 1).averageElbowAngle)
        val pose = frame()
            .with(11) { Joint(0.25, 0.25, 1.0) }.with(15) { Joint(0.75, 0.25, 1.0) }
            .with(14) { it.copy(confidence = 0.0) }
        assertEquals(90.0, analysis(listOf(pose), 1).averageElbowAngle!!, 0.001)
        assertNotNull(analysis(listOf(pose), 1, aspect = 2.0).averageElbowAngle)
    }

    @Test fun nearestFrameStopsAtTheGapLimit() {
        val result = analysis(listOf(frame(1.0)), 1)
        assertNotNull(result.nearestFrame(1.15))
        assertNull(result.nearestFrame(1.17))
        assertNotNull(result.nearestFrame(0.85))
        assertNull(result.nearestFrame(0.83))
        assertNull(analysis(emptyList(), 0).nearestFrame(0.0))
        assertEquals(0.0, analysis(emptyList(), 0).coverage, 0.0)
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
