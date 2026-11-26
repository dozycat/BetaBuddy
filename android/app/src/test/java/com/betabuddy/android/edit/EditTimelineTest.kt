package com.betabuddy.android.edit

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

// From a plan to an output: which source moment each output moment shows, and through which window.
class EditTimelineTest {
    private val replayCamera = listOf(CameraKey(7.6, 0.5, 0.4, 1.8), CameraKey(9.2, 0.52, 0.3, 2.2))
    private val plan = EditPlan(
        duration = 23.4, trim = 1.2 to 21.0,
        segments = listOf(EditSegment(1.2, 8.0, 1.0), EditSegment(8.0, 8.9, 0.5), EditSegment(8.9, 14.0, 2.0), EditSegment(7.6, 9.2, 0.25, replayCamera)),
        camera = listOf(CameraKey(1.2, 0.5, 0.5, 1.0), CameraKey(6.0, 0.48, 0.42, 1.5)),
        keep = null,
    )

    @Test fun theSwitchesChooseTheClips() {
        val all = EditTimeline.clips(plan, EditOptions(), 23.4)
        assertEquals(4, all.size)
        assertEquals(plan.camera, all[0].camera)
        assertEquals(replayCamera, all[3].camera)   // a segment's own camera replaces the plan's
        // 6.8 + 0.9 / 0.5 + 5.1 / 2 + 1.6 / 0.25
        assertEquals(6.8 + 1.8 + 2.55 + 6.4, EditTimeline.outputDuration(all), 1e-9)

        val plain = EditTimeline.clips(plan, EditOptions(slowMotion = false), 23.4)
        assertEquals(listOf(EditClip(1.2, 21.0, 1.0, plan.camera)), plain)

        val still = EditTimeline.clips(plan, EditOptions(cameraMoves = false), 23.4)
        assertTrue(still.all { it.camera.isEmpty() })
        assertEquals(listOf(EditClip(1.2, 21.0, 1.0, emptyList())), EditTimeline.clips(plan, EditOptions(slowMotion = false, cameraMoves = false), 23.4))
    }

    @Test fun rangesAreCutToTheClipThatIsReallyThere() {
        val short = EditTimeline.clips(plan, EditOptions(), 9.0)
        assertEquals(listOf(1.2 to 8.0, 8.0 to 8.9, 8.9 to 9.0, 7.6 to 9.0), short.map { it.t0 to it.t1 })
        // Nothing of the plan is inside the clip: the clip is shown as it is.
        assertEquals(listOf(EditClip(0.0, 1.0, 1.0, plan.camera)), EditTimeline.clips(plan, EditOptions(), 1.0))
        assertEquals(1.2 to 20.0, EditTimeline.clips(plan, EditOptions(slowMotion = false), 20.0).single().let { it.t0 to it.t1 })
    }

    @Test fun outputTimeMapsToSourceTimeThroughSpeedAndReplays() {
        val clips = EditTimeline.clips(plan, EditOptions(), 23.4)
        fun at(output: Double) = EditTimeline.sourceTime(clips, output)
        assertEquals(0 to 1.2, at(0.0))
        assertEquals(0, at(3.0).first); assertEquals(4.2, at(3.0).second, 1e-9)
        // Where the slow range begins, one second of output is half a second of clip.
        val slow = EditTimeline.outputStart(clips, 1)
        assertEquals(6.8, slow, 1e-9)
        assertEquals(1, at(slow).first); assertEquals(8.0, at(slow).second, 1e-9)
        assertEquals(8.5, at(slow + 1.0).second, 1e-9)
        // The fast range: one second of output is two of clip.
        val fast = EditTimeline.outputStart(clips, 2)
        assertEquals(2, at(fast).first); assertEquals(8.9, at(fast).second, 1e-9)
        assertEquals(10.9, at(fast + 1.0).second, 1e-9)
        // The replay goes back in the clip, at a quarter of the speed.
        val replay = EditTimeline.outputStart(clips, 3)
        assertEquals(6.8 + 1.8 + 2.55, replay, 1e-9)
        assertEquals(3, at(replay).first); assertEquals(7.6, at(replay).second, 1e-9)
        assertEquals(8.1, at(replay + 2.0).second, 1e-9)
        // Past either end the first and last frames hold.
        assertEquals(3 to 9.2, at(1000.0)); assertEquals(1.2, at(-5.0).second, 1e-9)

        assertEquals(replay + 2.0, EditTimeline.outputTime(clips[3], replay, 8.1), 1e-9)
        // What the renderer does: time since the clip's first frame, times its speed.
        assertEquals(8.25, EditTimeline.sourceTime(clips[1], 0.5), 1e-9)
        assertEquals(8.9, EditTimeline.sourceTime(clips[1], 99.0), 1e-9)
        assertEquals(8.0, EditTimeline.sourceTime(clips[1], -1.0), 1e-9)
    }

    @Test fun theCameraEasesBetweenKeyframesAndHoldsOutside() {
        val keys = listOf(CameraKey(2.0, 0.4, 0.5, 1.0), CameraKey(4.0, 0.6, 0.3, 2.0))
        assertEquals(CameraWindow.FULL, CameraPath.window(emptyList(), 3.0))
        // Zoom 1 shows the whole frame, wherever the keyframe says the centre is.
        assertEquals(CameraWindow(0.5, 0.5, 1.0), CameraPath.window(keys, 0.0))
        assertEquals(CameraWindow(0.5, 0.5, 1.0), CameraPath.window(keys, 2.0))
        assertEquals(CameraWindow(0.6, 0.3, 2.0), CameraPath.window(keys, 4.0))
        assertEquals(CameraWindow(0.6, 0.3, 2.0), CameraPath.window(keys, 99.0))
        // Half way the eased value is the mean; a quarter of the way it is behind a straight line.
        val middle = CameraPath.window(keys, 3.0)
        assertEquals(1.5, middle.s, 1e-9); assertEquals(0.5, middle.cx, 1e-9); assertEquals(0.4, middle.cy, 1e-9)
        val early = CameraPath.window(keys, 2.5)
        assertEquals(1.0 + 0.15625, early.s, 1e-9)   // smoothstep(0.25) = 0.15625
        assertTrue(early.s < 1.25)
    }

    @Test fun aCameraThatFollowsTheSubjectMovesInStraightLinesBetweenItsKeys() {
        val keys = listOf(CameraKey(2.0, 0.40, 0.50, 2.0), CameraKey(4.0, 0.60, 0.30, 3.0))
        // A quarter of the way in time is a quarter of the move…
        val quarter = CameraPath.window(keys, 2.5, linear = true)
        assertEquals(0.45, quarter.cx, 1e-9); assertEquals(0.45, quarter.cy, 1e-9); assertEquals(2.25, quarter.s, 1e-9)
        // …where the eased camera has barely set off.
        assertEquals(0.40 + 0.20 * 0.15625, CameraPath.window(keys, 2.5).cx, 1e-9)
        // The ends hold and the window stays in the frame, as always.
        assertEquals(CameraWindow(0.40, 0.50, 2.0), CameraPath.window(keys, 0.0, linear = true))
        assertEquals(CameraWindow(0.60, 0.30, 3.0), CameraPath.window(keys, 9.0, linear = true))
        val edge = CameraPath.window(listOf(CameraKey(0.0, 0.0, 0.0, 2.0), CameraKey(1.0, 0.1, 0.1, 2.0)), 0.5, linear = true)
        assertEquals(0.25, edge.cx, 1e-9); assertEquals(0.25, edge.cy, 1e-9)

        // The plan's mode reaches every clip: the plan's own camera, a segment's own, and with slow motion off.
        val tracked = plan.copy(cameraTrack = true)
        assertTrue(EditTimeline.clips(tracked, EditOptions(), 23.4).all { it.cameraLinear })
        assertTrue(EditTimeline.clips(tracked, EditOptions(slowMotion = false), 23.4).single().cameraLinear)
        assertTrue(EditTimeline.clips(tracked, EditOptions(), 1.0).single().cameraLinear)
        assertTrue(EditTimeline.clips(plan, EditOptions(), 23.4).none { it.cameraLinear })
    }

    @Test fun theWindowNeverLeavesTheFrame() {
        // At zoom 2 the window is half the frame, so its centre stays between 0.25 and 0.75.
        assertEquals(CameraWindow(0.25, 0.75, 2.0), CameraPath.clamp(0.0, 1.0, 2.0))
        assertEquals(CameraWindow(0.5, 0.5, 1.0), CameraPath.clamp(0.9, 0.1, 0.5))
        assertEquals(8.0, CameraPath.clamp(0.5, 0.5, 20.0).s, 0.0)
        assertEquals(6.0, CameraPath.clamp(0.5, 0.5, 6.0).s, 0.0)
        // At zoom 8 the window is an eighth of the frame, and still inside it.
        assertEquals(CameraWindow(0.0625, 0.9375, 8.0), CameraPath.clamp(0.0, 1.0, 8.0))
        val window = CameraPath.window(listOf(CameraKey(0.0, 0.95, 0.02, 2.5)), 1.0)
        assertEquals(0.8, window.cx, 1e-9); assertEquals(0.2, window.cy, 1e-9)
    }

    @Test fun keepBoxesMoveInStraightLinesOrJumpToTheNearerSample() {
        val samples = listOf(
            KeepSample(1.0, listOf(Box(0.1, 0.1, 0.2, 0.2))), KeepSample(2.0, listOf(Box(0.3, 0.5, 0.4, 0.2))),
            KeepSample(3.0, listOf(Box(0.3, 0.5, 0.4, 0.2), Box(0.7, 0.7, 0.1, 0.1))),
        )
        assertTrue(KeepPath.boxes(emptyList(), 1.0).isEmpty())
        assertEquals(samples[0].boxes, KeepPath.boxes(samples, 0.0))
        assertEquals(samples[2].boxes, KeepPath.boxes(samples, 9.0))
        val half = KeepPath.boxes(samples, 1.5).single()
        assertEquals(0.2, half.x, 1e-9); assertEquals(0.3, half.y, 1e-9); assertEquals(0.3, half.w, 1e-9); assertEquals(0.2, half.h, 1e-9)
        // One box, then two: no pairing is guessed.
        assertEquals(samples[1].boxes, KeepPath.boxes(samples, 2.4))
        assertEquals(samples[2].boxes, KeepPath.boxes(samples, 2.6))
    }

    @Test fun theRenderSizeIsTheUprightPictureAtMostFullHd() {
        assertEquals(1080 to 1920, RenderSize.of(1080, 1920))
        assertEquals(1080 to 1920, RenderSize.of(2160, 3840))
        assertEquals(1920 to 1080, RenderSize.of(3840, 2160))
        assertEquals(540 to 960, RenderSize.of(540, 960))
        assertEquals(360 to 640, RenderSize.of(1080, 1920, 640))
        // Both sides even, as encoders want.
        val (width, height) = RenderSize.of(1081, 1437)
        assertEquals(0, width % 2); assertEquals(0, height % 2)
    }
}
