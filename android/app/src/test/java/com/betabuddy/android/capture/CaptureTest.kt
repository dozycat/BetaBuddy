package com.betabuddy.android.capture

import com.betabuddy.android.R
import com.betabuddy.android.pose.ImportRules
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** The shape of the capture page's messages and calls. Synthetic data only. Counterpart of ios/CoreTests/CaptureTests.swift. */
class CaptureTest {
    private fun assessment(json: String) = (CaptureMessage.parse(json) as CaptureMessage.Assessment).assessment

    @Test fun aFullAssessmentIsRead() {
        val a = assessment(
            """{"type":"captureAssessment","t":12.4,"ok":false,"score":0.62,"steady":true,
               "issues":[{"code":"cut_top","severity":"block","move":"back"},{"code":"start_unclear","severity":"warn"}],
               "hint":{"code":"cut_top","move":"back"},"person":"none","wall":{"top":0.06,"bottom":0.91,"left":0,"right":1},
               "palette":[{"key":"yellow","rgb":[226,196,40],"count":9},{"key":"blue","rgb":[40.4,110,300],"n":4}],
               "color":"yellow",
               "holds":[{"id":"h3","box":[0.41,0.33,0.06,0.04],"poly":[[0.41,0.34],[0.44,0.33],[0.47,0.36]],"role":"start_hand","conf":0.8,"user":false},
                        {"id":"h7","box":[0.5,0.8,0.05,0.03],"role":"start_foot","user":true},
                        {"id":"h9","poly":[[0.1,0.1],[0.2,0.1],[0.2,0.3]],"role":"dyno_pad_2031"},
                        {"id":"h10","role":"hold"}],
               "start":{"hands":["h3"],"feet":["h7"],"conf":0.7},"finish":{"id":"h12","conf":0.6},"somethingNew":{"a":1}}""",
        )
        assertEquals(12.4, a.time, 1e-9)
        assertFalse(a.ok); assertTrue(a.steady); assertFalse(a.ready)
        assertEquals(0.62, a.score!!, 1e-9)
        assertEquals(listOf("cut_top", "start_unclear"), a.issues.map { it.code })
        assertEquals(listOf(true, false), a.issues.map { it.blocking })
        assertEquals(CaptureAssessment.Issue("cut_top", true, "back"), a.hint)
        assertEquals(CaptureBox(0.0, 0.06, 1.0, 0.85), a.wall!!.copy(height = Math.round(a.wall.height * 100) / 100.0))
        assertEquals(listOf("yellow", "blue"), a.palette.map { it.key })
        assertEquals(listOf(40, 110, 255), a.palette[1].rgb)   // rounded and kept within a byte
        assertEquals(listOf(9, 4), a.palette.map { it.count })
        assertEquals("yellow", a.color)
        // A hold with nothing to draw is left out; one with only an outline gets its box from it.
        assertEquals(listOf("h3", "h7", "h9"), a.holds.map { it.id })
        assertEquals(CaptureRole.START_HAND, a.holds[0].role)
        assertEquals(3, a.holds[0].poly.size)
        assertTrue(a.holds[1].user); assertTrue(a.holds[1].poly.isEmpty())
        assertEquals(CaptureRole.HOLD, a.holds[2].role)   // a role this build has never heard of
        assertEquals(0.1, a.holds[2].box.x, 1e-9); assertEquals(0.2, a.holds[2].box.height, 1e-9)
        assertEquals(listOf("h3"), a.startHands); assertEquals(listOf("h7"), a.startFeet); assertEquals("h12", a.finish)
        assertNull(a.route)
        assertFalse(a.startConfirmed); assertFalse(a.finishConfirmed)
    }

    @Test fun anEmptyAssessmentFallsBackToTheCautiousReading() {
        val a = assessment("""{"type":"captureAssessment"}""")
        assertTrue(a.ok)            // no issues were reported
        assertFalse(a.steady)       // the page has not said the picture is still
        assertFalse(a.ready)
        assertNull(a.score); assertNull(a.hint); assertNull(a.wall); assertNull(a.color); assertNull(a.finish); assertNull(a.route)
        assertEquals("unknown", a.person)
        assertTrue(a.issues.isEmpty() && a.palette.isEmpty() && a.holds.isEmpty() && a.startHands.isEmpty() && a.startFeet.isEmpty())
    }

    @Test fun wrongTypesAreIgnoredFieldByField() {
        val a = assessment(
            """{"type":"captureAssessment","t":"soon","ok":"yes","score":7,"steady":1,"issues":[3,{"severity":"warn"},{"code":"","move":"back"},{"code":"x"}],
               "hint":"cut_top","person":5,"wall":{"top":0.9,"bottom":0.1,"left":0,"right":1},"palette":[{"rgb":[1,2,3]},{"key":"red","rgb":[1,2]},"blue"],
               "color":"","holds":{"id":"h1"},"start":["h1"],"finish":"h2","route":"some-route"}""",
        )
        assertEquals(0.0, a.time, 0.0)
        assertEquals(1.0, a.score!!, 0.0)           // kept within 0–1
        assertFalse(a.steady)                       // a number is not a yes
        assertEquals(listOf("x"), a.issues.map { it.code })
        assertTrue(a.issues[0].blocking)            // no severity: taken as blocking, so the user is asked
        assertFalse(a.ok)
        assertEquals("x", a.hint?.code)             // the first issue stands in for an unreadable hint
        assertNull(a.wall); assertNull(a.color); assertNull(a.finish); assertNull(a.route)
        assertEquals(listOf("red"), a.palette.map { it.key })
        assertEquals(listOf(128, 128, 128), a.palette[0].rgb)
        assertTrue(a.holds.isEmpty() && a.startHands.isEmpty())
    }

    @Test fun aHintWithoutSeverityTakesItFromItsIssueOrFromOk() {
        val warned = assessment("""{"type":"captureAssessment","ok":true,"issues":[{"code":"oblique","severity":"warn","move":"left"}],"hint":{"code":"oblique","move":"left"}}""")
        assertFalse(warned.hint!!.blocking)
        val orphan = assessment("""{"type":"captureAssessment","ok":false,"issues":[],"hint":{"code":"from_the_future"}}""")
        assertTrue(orphan.hint!!.blocking)
    }

    @Test fun confirmationFollowsWhatTheUserMarked() {
        val a = assessment(
            """{"type":"captureAssessment","holds":[{"id":"a","box":[0,0,0.1,0.1],"role":"start_hand","user":true},{"id":"b","box":[0,0,0.1,0.1],"role":"finish","user":false}]}""",
        )
        assertTrue(a.startConfirmed); assertFalse(a.finishConfirmed)
    }

    @Test fun aRecognisedProblemIsReadLeniently() {
        val full = assessment("""{"type":"captureAssessment","route":{"id":"synthetic-east/yellow-01","conf":0.9,"name":"Synthetic Yellow 1","nameZh":"合成线路 黄 1","grade":"V3","colour":"yellow","wall":"East","others":[{"id":"x"}]}}""").route!!
        assertEquals("synthetic-east/yellow-01", full.id)
        assertEquals(0.9, full.confidence!!, 1e-9)
        // On screen: the name in the page's language, else the problem's own name, else its id.
        assertEquals("Synthetic Yellow 1", full.title); assertEquals("合成线路 黄 1", full.nameZh)
        assertEquals("V3", full.grade); assertEquals("yellow", full.colour); assertEquals("East", full.wall)
        val bare = assessment("""{"type":"captureAssessment","route":{"id":"r1","conf":"high","name":7}}""").route!!
        assertNull(bare.confidence); assertNull(bare.name)
        assertEquals("r1", bare.title)            // nothing to show but the id
        assertNull(assessment("""{"type":"captureAssessment","route":null}""").route)
        assertNull(assessment("""{"type":"captureAssessment","route":{"conf":0.9}}""").route)
        assertEquals("只有中文名", assessment("""{"type":"captureAssessment","route":{"id":"r","nameZh":"只有中文名","name":""}}""").route!!.title)
    }

    @Test fun onlyAPlainVGradeIsTakenForTheLogbook() {
        fun grade(text: String?) = com.betabuddy.android.data.CaptureRoute("r", grade = text).albumGrade
        assertEquals("V0", grade("V0")); assertEquals("V3", grade("V3")); assertEquals("V12", grade("V12"))
        // Gyms write grades in their own ways; anything else is shown as it is and never filled in.
        for (other in listOf(null, "", "V13", "V-1", "V03", "v3", "V3+", "V3-4", " V3", "6A", "紫条 11–12", "V", "VB")) assertNull(other, grade(other))
    }

    @Test fun otherMessages() {
        assertEquals(CaptureMessage.Ready("0.0.0-standin"), CaptureMessage.parse("""{"type":"captureReady","packVersion":"0.0.0-standin"}"""))
        assertEquals(CaptureMessage.Ready(""), CaptureMessage.parse("""{"type":"captureReady"}"""))
        assertEquals(CaptureMessage.Failed("frame"), CaptureMessage.parse("""{"type":"captureError","code":"frame","message":"x"}"""))
        assertEquals(CaptureMessage.PageError("boom"), CaptureMessage.parse("""{"type":"error","message":"boom"}"""))
        for (text in listOf(null, "", "not json", "[]", "{}", """{"type":"captureProgress2031"}""", """{"type":7}""", """{"type":"ready"}""")) {
            assertNull(text, CaptureMessage.parse(text))
        }
    }

    @Test fun lockedKeepsThePlanAsItIs() {
        val locked = CaptureMessage.parse(
            """{"type":"captureLocked","id":"C1","plan":"{\"v\":1,\"x\":\"é\"}","summary":{"color":"yellow","rgb":[226,196,40],"holds":6,"start":{"hands":2,"feet":1},"finish":true,"ok":false,"issues":["cut_top",3],"route":{"id":"r1","conf":0.9}}}""",
        ) as CaptureMessage.Locked
        assertEquals("C1", locked.id)
        assertEquals("""{"v":1,"x":"é"}""", locked.plan)
        val summary = locked.summary
        assertEquals("yellow", summary.color); assertEquals(listOf(226, 196, 40), summary.rgb)
        assertEquals(6, summary.holds); assertEquals(2, summary.startHands); assertEquals(1, summary.startFeet)
        assertTrue(summary.finish); assertFalse(summary.ok)
        assertEquals(listOf("cut_top"), summary.issues)
        assertEquals("r1", summary.route?.id)

        // A page that hands over an object instead of a string is still kept; nothing and an empty string are no plan.
        val asObject = CaptureMessage.parse("""{"type":"captureLocked","plan":{"v":1}}""") as CaptureMessage.Locked
        assertEquals("""{"v":1}""", asObject.plan)
        assertEquals("", asObject.id)
        assertEquals(0, asObject.summary.holds); assertNull(asObject.summary.color)
        assertNull((CaptureMessage.parse("""{"type":"captureLocked","plan":""}""") as CaptureMessage.Locked).plan)
        assertNull((CaptureMessage.parse("""{"type":"captureLocked","plan":null}""") as CaptureMessage.Locked).plan)
        assertNull((CaptureMessage.parse("""{"type":"captureLocked","plan":7}""") as CaptureMessage.Locked).plan)
    }

    @Test fun callsAreBuiltFromJsonLiterals() {
        val begin = CaptureScript.begin("A\"B", 360, 640, 51.234, null)
        assertTrue(begin, begin.contains("""window.BetaBuddyCapture.begin({"id":"A\"B","w":360,"h":640,"fovH":51.23})"""))
        assertTrue(begin.startsWith("window.BetaBuddyCapture && typeof window.BetaBuddyCapture.begin === 'function' ? ("))
        assertTrue(begin.endsWith(", true) : false"))
        assertTrue(CaptureScript.begin("A", 2, 2, Double.NaN, "synthetic-east").contains("""{"id":"A","w":2,"h":2,"venue":"synthetic-east"}"""))
        assertFalse(CaptureScript.begin("A", 2, 2, null, "").contains("venue"))

        // Nothing known about the frame: the third argument is left out entirely.
        assertTrue(CaptureScript.addFrame(1.23456, "QUJD/w==", CaptureFrameInfo()).contains("""addFrame(1.235,"QUJD/w==")"""))
        val withInfo = CaptureScript.addFrame(2.0, "QQ==", CaptureFrameInfo(pitch = 12.345, roll = -0.5, shake = 0.012345, pose = listOf(listOf(0.123456, 0.5, Double.NaN))))
        assertTrue(withInfo, withInfo.contains("""addFrame(2.0,"QQ==",{"pitch":12.35,"roll":-0.5,"shake":0.0123,"pose":[[0.1235,0.5,0.0]]})"""))
        assertTrue(CaptureScript.addFrame(0.0, "QQ==", CaptureFrameInfo(shake = 0.5)).contains("""{"shake":0.5}"""))

        assertTrue(CaptureScript.setColor("yellow").contains("""setColor("yellow")"""))
        assertTrue(CaptureScript.setColor(null).contains("setColor(null)"))
        // A line separator would end the line in an older JavaScript engine: it goes over escaped.
        assertTrue(CaptureScript.setColor("x\u2028\"y").contains("""setColor("x\u2028\"y")"""))
        val tap = CaptureScript.tap(1.5, -0.2, CaptureRole.NONE)
        assertTrue(tap, tap.contains("""(window.BetaBuddyCapture.tap(1.0,0.0,"none") || '')"""))
        assertTrue(CaptureScript.tap(0.123456, 0.5, CaptureRole.START_HAND).contains("""tap(0.1235,0.5,"start_hand")"""))
        assertTrue(CaptureScript.resetTaps().contains("resetTaps()"))
        assertTrue(CaptureScript.lock().contains("lock()"))
        // Only called when the page has it: the same guard as every other call.
        assertTrue(CaptureScript.rejectRoute("a/b").startsWith("window.BetaBuddyCapture && typeof window.BetaBuddyCapture.rejectRoute === 'function' ? (window.BetaBuddyCapture.rejectRoute(\"a/b\"), true)"))
    }

    @Test fun whatTheCallsAnswer() {
        assertEquals("y1", CaptureScript.tapAnswer("\"y1\""))
        assertEquals("", CaptureScript.tapAnswer("\"\""))     // no hold there
        assertNull(CaptureScript.tapAnswer("false"))            // the page has no tap
        assertNull(CaptureScript.tapAnswer("null")); assertNull(CaptureScript.tapAnswer(null)); assertNull(CaptureScript.tapAnswer("oops"))
        assertTrue(CaptureScript.made("true")); assertFalse(CaptureScript.made("false")); assertFalse(CaptureScript.made(null))
    }

    @Test fun rolesOnTheWire() {
        assertEquals(listOf("start_hand", "start_foot", "finish", "hold", "none"), CaptureRole.entries.map { it.wire })
        assertEquals(CaptureRole.FINISH, CaptureRole.of("finish")); assertNull(CaptureRole.of("campus"))
    }

    @Test fun attitudeFromGravity() {
        // Android reports the vector opposing gravity, in m/s²: upright portrait is (0, 9.81, 0).
        fun angles(x: Double, y: Double, z: Double): Pair<Double, Double> {
            val (dx, dy, dz) = CaptureMotion.down(x, y, z)!!
            return CaptureMotion.pitch(dx, dy, dz)!! to CaptureMotion.roll(dx, dy, dz)!!
        }
        val upright = angles(0.0, 9.81, 0.0)
        assertEquals(0.0, upright.first, 1e-6); assertEquals(0.0, upright.second, 1e-6)
        // Tilted back so the camera looks up by 30°: the screen faces partly down.
        val up = angles(0.0, 9.81 * Math.cos(Math.toRadians(30.0)), -9.81 * Math.sin(Math.toRadians(30.0)))
        assertEquals(30.0, up.first, 1e-6); assertEquals(0.0, up.second, 1e-6)
        // Lying on a table, screen up: the camera looks straight down, and there is no upright to measure roll from.
        val flat = angles(0.0, 0.0, 9.81)
        assertEquals(-90.0, flat.first, 1e-6); assertEquals(0.0, flat.second, 0.0)
        // Turned 10° clockwise as seen on the screen.
        val turned = angles(-9.81 * Math.sin(Math.toRadians(10.0)), 9.81 * Math.cos(Math.toRadians(10.0)), 0.0)
        assertEquals(10.0, turned.second, 1e-6)
        assertNull(CaptureMotion.down(0.0, 0.0, 0.0)); assertNull(CaptureMotion.down(Double.NaN, 1.0, 1.0))
        assertNull(CaptureMotion.pitch(Double.NaN, 0.0, 0.0)); assertNull(CaptureMotion.roll(0.0, Double.POSITIVE_INFINITY, 0.0))
    }

    @Test fun fieldOfView() {
        // A sensor 6.4 mm wide behind a 4.8 mm lens.
        assertEquals(67.38, CaptureMotion.fieldOfView(6.4, 4.8)!!, 0.01)
        assertNull(CaptureMotion.fieldOfView(0.0, 4.8)); assertNull(CaptureMotion.fieldOfView(6.4, Double.NaN))
    }

    @Test fun theFieldOfViewIsThatOfTheStreamDelivered() {
        // A 4:3 sensor area 6.4 × 4.8 mm behind a 4.8 mm lens. A 16:9 stream keeps its length and loses height…
        assertEquals(41.11, CaptureMotion.streamFieldOfView(6.4, 4.8, 4.8, 1280, 720)!!, 0.01)
        assertEquals(41.11, CaptureMotion.streamFieldOfView(4.8, 6.4, 4.8, 720, 1280)!!, 0.01)   // whichever way round the sides are given
        // …a 4:3 stream uses all of it, and the upright picture is then wider than a 16:9 one…
        assertEquals(53.13, CaptureMotion.streamFieldOfView(6.4, 4.8, 4.8, 640, 480)!!, 0.01)
        // …and a square one cannot be wider than the sensor is high.
        assertEquals(53.13, CaptureMotion.streamFieldOfView(6.4, 4.8, 4.8, 480, 480)!!, 0.01)
        // A 16:9 sensor area gives a 16:9 stream all of itself.
        assertEquals(CaptureMotion.fieldOfView(3.6, 4.8)!!, CaptureMotion.streamFieldOfView(6.4, 3.6, 4.8, 1920, 1080)!!, 1e-9)
        assertNull(CaptureMotion.streamFieldOfView(0.0, 4.8, 4.8, 1280, 720)); assertNull(CaptureMotion.streamFieldOfView(6.4, 4.8, 4.8, 0, 720))
        assertNull(CaptureMotion.streamFieldOfView(6.4, 4.8, Double.NaN, 1280, 720))
    }

    @Test fun nothingInAnInterfaceLanguageIsKeptWithARecord() {
        val route = com.betabuddy.android.data.CaptureRoute("r1", 0.9, name = "Synthetic Yellow 1", nameZh = "合成线路 黄 1", grade = "Purple tag 11–12", colour = "yellow", wall = "East")
        assertEquals(com.betabuddy.android.data.CaptureRoute("r1", 0.9, nameZh = "合成线路 黄 1"), route.stored())
        val summary = com.betabuddy.android.data.CaptureSummary(color = "yellow", holds = 6, route = route)
        assertEquals(summary.copy(route = route.stored()), summary.stored())
        assertNull(com.betabuddy.android.data.CaptureSummary().stored().route)
    }

    @Test fun frameSizeForThePage() {
        assertEquals(360 to 640, CaptureLimits.frameSize(1080, 1920))
        assertEquals(360 to 640, CaptureLimits.frameSize(720, 1280))
        assertEquals(640 to 360, CaptureLimits.frameSize(1920, 1080))
        assertEquals(360 to 480, CaptureLimits.frameSize(1080, 1440))
        assertEquals(240 to 320, CaptureLimits.frameSize(240, 320))     // never scaled up
        assertEquals(360 to 426, CaptureLimits.frameSize(1080, 1279))   // both sides even
        assertEquals(2 to 2, CaptureLimits.frameSize(1, 1))
    }

    @Test fun aRecordingStaysUnderWhatTheAnalyzerAccepts() {
        // As long as on iOS, and never cut at what an import may be.
        assertTrue(CaptureLimits.MAX_SECONDS < ImportRules.CAMERA_MAX_SECONDS && CaptureLimits.MAX_SECONDS > ImportRules.MAX_SECONDS)
        assertEquals(20, CaptureLimits.MAX_MINUTES)
        val big = 3L * 1024 * 1024 * 1024
        assertNull(ImportRules.checkCameraClip(big, 19.9 * 60, true))
        assertEquals(com.betabuddy.android.pose.AnalysisError.TOO_LONG, ImportRules.checkCameraClip(big, 20.1 * 60, true))
        assertEquals(com.betabuddy.android.pose.AnalysisError.INVALID_VIDEO, ImportRules.checkCameraClip(0, 60.0, true))
        // An import is held to what it was before.
        assertEquals(com.betabuddy.android.pose.AnalysisError.TOO_LONG, ImportRules.check(1_000, 181.0, true))
        assertEquals(com.betabuddy.android.pose.AnalysisError.TOO_LARGE, ImportRules.check(big, 60.0, true))
        assertTrue(CaptureLimits.STOPPING_FREE_BYTES < CaptureLimits.STARTING_FREE_BYTES)
    }

    // Guidance text

    @Test fun everyKnownCodeAndMoveHasItsOwnSentence() {
        val reasons = CaptureGuidance.Code.entries.map { CaptureText.reason(it.wire) }
        assertEquals(reasons.size, reasons.toSet().size)
        assertFalse(R.string.capture_reason_other in reasons)
        val actions = CaptureGuidance.Move.entries.map { CaptureText.action(it.wire) }
        assertEquals(actions.size, actions.filterNotNull().toSet().size)
        assertEquals(R.string.capture_reason_cut_top, CaptureText.reason("cut_top"))
        assertEquals(R.string.capture_move_hold_still, CaptureText.action("hold_still"))
    }

    @Test fun anUnknownCodeGetsTheGenericLineAndAnUnknownMoveNone() {
        assertEquals(R.string.capture_reason_other, CaptureText.reason("lens_flare_2031"))
        assertEquals(R.string.capture_reason_other, CaptureText.reason(""))
        assertEquals(R.string.capture_reason_other, CaptureText.reason(null))
        assertEquals(R.string.capture_reason_other, CaptureText.reason("CUT_TOP"))   // codes are matched as written
        assertNull(CaptureText.action("spin_around")); assertNull(CaptureText.action(null))
    }

    @Test fun thePromptFollowsWhatIsMarked() {
        fun holds(start: Boolean, finish: Boolean) = assessment(
            """{"type":"captureAssessment","holds":[{"id":"a","box":[0,0,0.1,0.1],"role":"start_hand","user":$start},{"id":"b","box":[0,0,0.1,0.1],"role":"finish","user":$finish}]}""",
        )
        assertEquals(CaptureText.Stage.COLOUR, CaptureText.stage(false, false, null))
        assertEquals(CaptureText.Stage.START, CaptureText.stage(false, true, null))
        assertEquals(CaptureText.Stage.START, CaptureText.stage(false, true, holds(false, true)))
        // With a recognised problem the marks come from the pack's data: the prompt says so until the user marks the start.
        assertEquals(CaptureText.Stage.ROUTE, CaptureText.stage(false, true, holds(false, false), recognised = true))
        assertEquals(CaptureText.Stage.FINISH, CaptureText.stage(false, true, holds(true, false), recognised = true))
        assertEquals(CaptureText.Stage.COLOUR, CaptureText.stage(false, false, null, recognised = true))
        assertEquals(CaptureText.Stage.FINISH, CaptureText.stage(false, true, holds(true, false)))
        assertEquals(CaptureText.Stage.MARKED, CaptureText.stage(false, true, holds(true, true)))
        assertEquals(CaptureText.Stage.MISSED, CaptureText.stage(true, true, holds(true, true)))
        assertEquals(R.string.capture_prompt_start, CaptureText.prompt(CaptureText.Stage.START))
        assertEquals(6, CaptureText.Stage.entries.map(CaptureText::prompt).toSet().size)
    }

    @Test fun stopsNoticesRolesAndTheIndicator() {
        assertNull(CaptureText.stop(CaptureStop.USER)); assertNull(CaptureText.stop(null))
        assertEquals(4, CaptureStop.entries.mapNotNull(CaptureText::stop).toSet().size)
        assertEquals(3, CaptureSession.Notice.entries.map(CaptureText::notice).toSet().size)
        assertEquals(5, CaptureRole.entries.map(CaptureText::role).toSet().size)
        assertEquals(listOf(1, 1, 1, 2, 3, 5, 5, 5), listOf(-1.0, 0.0, 0.15, 0.4, 0.45, 0.95, 1.0, 3.0).map(CaptureText::level))
        assertNotNull(CaptureText.reason("no_wall"))
    }

    // Which recorder writes a clip

    @Test fun theLibrarysRecorderKeepsAClipItIsWriting() {
        val switch = RecorderSwitch(graceMillis = 3_000)
        assertEquals(RecorderSwitch.Path.LIBRARY, switch.path)
        assertFalse(switch.shouldSwitch(1_000))
        switch.libraryWrote()
        // It has written video: whatever the clock says afterwards, the clip stays its own.
        assertFalse(switch.shouldSwitch(3_000)); assertFalse(switch.shouldSwitch(60_000))
        assertFalse(switch.libraryEndedEmpty(stopRequested = false))
        assertEquals(RecorderSwitch.Path.LIBRARY, switch.path); assertFalse(switch.switched)
    }

    @Test fun nothingWrittenInTimeHandsTheClipToTheAppsOwnRecorderOnce() {
        val switch = RecorderSwitch(graceMillis = 3_000)
        assertFalse(switch.shouldSwitch(0)); assertFalse(switch.shouldSwitch(2_999))
        assertTrue(switch.shouldSwitch(3_000))
        assertEquals(RecorderSwitch.Path.OWN, switch.path); assertTrue(switch.switched)
        // Only once, and what the library's recorder says afterwards changes nothing.
        assertFalse(switch.shouldSwitch(10_000))
        switch.libraryWrote()
        assertFalse(switch.libraryWrote)
        assertEquals(RecorderSwitch.Path.OWN, switch.path)
    }

    @Test fun aLibraryRecorderThatGivesUpEmptyIsReplacedAtOnceUnlessTheUserStoppedIt() {
        val gaveUp = RecorderSwitch()
        assertTrue(gaveUp.libraryEndedEmpty(stopRequested = false))
        assertEquals(RecorderSwitch.Path.OWN, gaveUp.path)
        assertFalse(gaveUp.libraryEndedEmpty(stopRequested = false))
        // The user pressed stop before anything was written: that is a clip too short to keep, not a broken recorder.
        val stopped = RecorderSwitch()
        assertFalse(stopped.libraryEndedEmpty(stopRequested = true))
        assertEquals(RecorderSwitch.Path.LIBRARY, stopped.path); assertFalse(stopped.switched)
    }

    @Test fun aDeviceThatNeededItStartsOnTheAppsOwnRecorder() {
        val switch = RecorderSwitch(remembered = true)
        assertEquals(RecorderSwitch.Path.OWN, switch.path)
        assertFalse(switch.switched)                       // nothing changed over during this clip
        assertFalse(switch.shouldSwitch(60_000)); assertFalse(switch.libraryEndedEmpty(false))
        assertEquals(3_000L, RecorderSwitch.GRACE_MILLIS)
    }

    @Test fun planesAreCopiedAcrossWhateverTheirStrides() {
        // A 4 × 2 block out of a 6-wide plane with padded rows, starting one in and one down.
        val source = java.nio.ByteBuffer.wrap(ByteArray(8 * 3) { it.toByte() })   // row stride 8
        val tight = java.nio.ByteBuffer.allocate(4 * 2)
        YuvPlanes.copy(source, 8, 1, 1, 1, tight, 4, 1, 4, 2, ByteArray(0))
        assertEquals(listOf(9, 10, 11, 12, 17, 18, 19, 20), tight.array().map { it.toInt() })
        assertEquals(0, source.position()); assertEquals(0, tight.position())
        // Interleaved source (every second byte), into a padded target.
        val interleaved = java.nio.ByteBuffer.wrap(ByteArray(8 * 2) { (100 + it).toByte() })
        val padded = java.nio.ByteBuffer.allocate(6 * 2)
        YuvPlanes.copy(interleaved, 8, 2, 0, 0, padded, 6, 1, 4, 2, ByteArray(1))
        assertEquals(listOf(100, 102, 104, 106, 0, 0, 108, 110, 112, 114, 0, 0), padded.array().map { it.toInt() })
        // Tight source into an interleaved target, the other bytes left alone; the last row may stop short of the stride.
        val plain = java.nio.ByteBuffer.wrap(byteArrayOf(1, 2, 3, 4, 5, 6))
        val spaced = java.nio.ByteBuffer.wrap(ByteArray(11) { 9 })
        YuvPlanes.copy(plain, 3, 1, 0, 0, spaced, 6, 2, 3, 2, ByteArray(8))
        assertEquals(listOf(1, 9, 2, 9, 3, 9, 4, 9, 5, 9, 6), spaced.array().map { it.toInt() })
    }
}
