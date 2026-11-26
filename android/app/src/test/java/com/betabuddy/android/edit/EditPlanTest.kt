package com.betabuddy.android.edit

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

// Reading the analysis page's plan for an automatic edit. Every plan here is made up.
class EditPlanTest {
    private val full = """
        {"v":1,"duration":23.4,"trim":[1.2,21.0],"later":"ignored",
         "segments":[
           {"t0":1.2,"t1":8.0,"speed":1,"kind":"run"},
           {"t0":8.0,"t1":8.9,"speed":0.35,"kind":"slow"},
           {"t0":8.9,"t1":14.0,"speed":2,"kind":"fast"},
           {"t0":7.6,"t1":9.2,"speed":0.25,"kind":"replay","camera":[{"t":9.2,"cx":0.52,"cy":0.3,"s":2.2},{"t":7.6,"cx":0.5,"cy":0.4,"s":1.8}]}],
         "camera":[{"t":1.2,"cx":0.5,"cy":0.5,"s":1},{"t":6.0,"cx":0.48,"cy":0.42,"s":1.5}],
         "keep":{"feather":0.03,"boxes":[{"t":1.2,"b":[[0.31,0.40,0.22,0.38]]},{"t":1.3,"b":[[0.32,0.40,0.22,0.38]]}]},
         "plate":{"times":[0.4,3.1,5.8,8.0,9.9,12.5,15.0,18.2,22.0]},
         "marks":[{"t0":8.0,"t1":8.9,"kind":"dyno"}]}
    """

    @Test fun aFullPlanIsReadAsWritten() {
        val plan = EditPlan.parse(full)!!
        assertEquals(23.4, plan.duration, 0.0)
        assertEquals(1.2 to 21.0, plan.trim)
        assertEquals(4, plan.segments.size)
        assertEquals(EditSegment(8.0, 8.9, 0.35), plan.segments[1])
        assertNull(plan.segments[0].camera)
        // Keyframes are put in order of time.
        assertEquals(listOf(7.6, 9.2), plan.segments[3].camera!!.map { it.t })
        assertEquals(CameraKey(6.0, 0.48, 0.42, 1.5), plan.camera[1])
        val keep = plan.keep!!
        assertEquals(0.03, keep.feather, 0.0)
        assertEquals(listOf(Box(0.31, 0.40, 0.22, 0.38)), keep.samples[0].boxes)
        assertEquals(9, keep.plateTimes.size)
        assertTrue(plan.hasSlowMotion); assertTrue(plan.hasCamera); assertTrue(plan.hasKeep)
    }

    @Test fun onlyAVersionOnePlanIsUsed() {
        assertNull(EditPlan.parse(full.replace("\"v\":1", "\"v\":2")))
        assertNull(EditPlan.parse(full.replace("\"v\":1,", "")))
        assertNull(EditPlan.parse(full.replace("\"v\":1", "\"v\":\"1\"")))
        assertNull(EditPlan.parse(null)); assertNull(EditPlan.parse("")); assertNull(EditPlan.parse("[]")); assertNull(EditPlan.parse("{"))
        assertNull(EditPlan.parse("""{"v":1,"duration":10}"""))
        assertNull(EditPlan.parse("""{"v":1,"duration":0,"segments":[{"t0":0,"t1":1,"speed":1}]}"""))
    }

    @Test fun aBarePlanIsTheAttemptAtNormalSpeedWithNoSwitches() {
        val plan = EditPlan.parse("""{"v":1,"duration":10,"trim":[2,8],"segments":[{"t0":2,"t1":8,"speed":1}]}""")!!
        assertFalse(plan.hasSlowMotion); assertFalse(plan.hasCamera); assertFalse(plan.hasKeep)
        assertTrue(plan.camera.isEmpty())
        // Without a trim the attempt is everything the segments cover.
        assertEquals(1.0 to 9.0, EditPlan.parse("""{"v":1,"duration":10,"segments":[{"t0":3,"t1":9,"speed":2},{"t0":1,"t1":3,"speed":1}]}""")!!.trim)
    }

    @Test fun numbersOutsideTheirRangeAreBroughtBackAndBrokenPartsAreLeftOut() {
        val plan = EditPlan.parse(
            """{"v":1,"duration":10,"segments":[
                 {"t0":-1,"t1":4,"speed":9},{"t0":4,"t1":40,"speed":0.01},{"t0":5,"t1":5.01,"speed":1},{"t0":6,"t1":5,"speed":1},
                 {"t0":1,"t1":2,"speed":0},{"t0":1,"t1":2},{"t0":"1","t1":2,"speed":1},"nonsense"],
               "camera":[{"t":1,"cx":-0.2,"cy":1.4,"s":11},{"t":3,"cx":0.5,"cy":0.5,"s":7},{"t":2,"cx":0.5,"cy":0.5},{"t":0,"cx":0.5,"cy":0.5,"s":0.2}]}""",
        )!!
        assertEquals(listOf(EditSegment(0.0, 4.0, 4.0), EditSegment(4.0, 10.0, 0.2)), plan.segments)
        // Zoom is kept between 1 and 8: a camera following a distant climber comes that close.
        assertEquals(listOf(CameraKey(0.0, 0.5, 0.5, 1.0), CameraKey(1.0, 0.0, 1.0, 8.0), CameraKey(3.0, 0.5, 0.5, 7.0)), plan.camera)
        val many = (0 until 60).joinToString(",") { """{"t0":$it,"t1":${it + 1},"speed":1}""" }
        assertEquals(EditPlan.MAX_SEGMENTS, EditPlan.parse("""{"v":1,"duration":100,"segments":[$many]}""")!!.segments.size)
    }

    @Test fun removalIsOfferedOnlyWithBoxesAndEnoughPlatePictures() {
        fun plan(keep: String, plate: String) = EditPlan.parse("""{"v":1,"duration":10,"segments":[{"t0":0,"t1":10,"speed":1}]$keep$plate}""")!!
        val boxes = ""","keep":{"feather":0.9,"boxes":[{"t":2,"b":[[0.9,0.9,0.5,0.5],[0.1,0.1,0,0.2],[1,2,3]]},{"t":1,"b":[]}]}"""
        assertNull(plan("", "").keep)
        assertNull(plan(boxes, "").keep)
        assertNull(plan(boxes, ""","plate":{"times":[1,2]}""").keep)
        assertNull(plan(""","keep":{"boxes":[]}""", ""","plate":{"times":[1,2,3]}""").keep)
        val keep = plan(boxes, ""","plate":{"times":[3,1,2,2,99,-1,"4"]}""").keep!!
        assertEquals(0.2, keep.feather, 0.0)
        assertEquals(listOf(1.0, 2.0), keep.samples.map { it.t })
        // A box reaching out of the picture is cut to it; an empty or malformed one is left out.
        assertEquals(1, keep.samples[1].boxes.size)
        assertEquals(0.1, keep.samples[1].boxes[0].w, 1e-9)
        assertEquals(listOf(1.0, 2.0, 3.0), keep.plateTimes)
        val many = (1..40).joinToString(",") { (it * 0.2).toString() }
        assertEquals(EditPlan.MAX_PLATE_TIMES, plan(boxes, ""","plate":{"times":[$many]}""").keep!!.plateTimes.size)
        assertNotNull(plan(boxes, ""","plate":{"times":[1,2,3]}""").keep)
    }

    @Test fun plateTimesAndKeepSamplesMayLieOutsideTheAttempt() {
        // A plan of the size a long clip gives: keep samples over the whole tracked stretch (sparser outside
        // the attempt), one box each, and plate pictures from before and after the attempt.
        val samples = ((0 until 15).map { it / 3.0 } + (0 until 200).map { 5.0 + it / 10.0 } + (0 until 15).map { 25.0 + it / 3.0 })
            .joinToString(",") { t -> """{"t":$t,"b":[[${0.2 + t / 100},0.3,0.2,0.4]]}""" }
        val times = (0 until 25).joinToString(",") { (0.5 + it * 1.2).toString() }
        val text = """{"v":1,"duration":30,"trim":[5,25],"segments":[{"t0":5,"t1":25,"speed":1,"kind":"run"}],
            "keep":{"feather":0.03,"boxes":[$samples]},"plate":{"times":[$times]}}"""
        assertTrue(text.length > 10_000)
        val plan = EditPlan.parse(text)!!
        val keep = plan.keep!!
        assertEquals(230, keep.samples.size)
        assertEquals(25, keep.plateTimes.size)
        assertTrue(keep.plateTimes.first() < plan.trim.first && keep.plateTimes.last() > plan.trim.second)
        // The keep box of a plate picture taken before the attempt is the one tracked at that moment…
        assertEquals(0.2 + 0.5 / 100, KeepPath.boxes(keep.samples, 0.5).single().x, 0.002)
        assertEquals(0.2 + 29.3 / 100, KeepPath.boxes(keep.samples, 29.3).single().x, 0.002)
        // …and beyond the tracked stretch, the nearest sample's.
        assertEquals(keep.samples.last().boxes, KeepPath.boxes(keep.samples, 29.99))
        assertEquals(keep.samples[40].boxes, KeepPath.boxes(keep.samples.drop(40), 0.5))
    }

    @Test fun onlyTheTrackModeMakesTheCameraLinear() {
        fun plan(mode: String) = EditPlan.parse("""{"v":1,"duration":10,"segments":[{"t0":0,"t1":10,"speed":1}],"camera":[{"t":0,"cx":0.5,"cy":0.5,"s":1.5}]$mode}""")!!
        assertTrue(plan(""","cameraMode":"track"""").cameraTrack)
        // Absent, or anything this build does not know: the eased camera.
        for (other in listOf("", ""","cameraMode":"ease"""", ""","cameraMode":"orbit"""", ""","cameraMode":"Track"""", ""","cameraMode":1""", ""","cameraMode":null""", ""","cameraMode":["track"]""")) {
            assertFalse(other, plan(other).cameraTrack)
        }
        assertFalse(EditPlan.parse(full)!!.cameraTrack)
    }
}
