package com.betabuddy.android.capture

import android.Manifest
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.semantics.getOrNull
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.click
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.compose.ui.test.onAllNodesWithTag
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performTouchInput
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.betabuddy.android.MainActivity
import com.betabuddy.android.R
import com.betabuddy.android.data.Climb
import com.betabuddy.android.data.ClimbStore
import com.betabuddy.android.i18n.AppLanguage
import com.betabuddy.android.journal.CameraGate
import com.betabuddy.android.journal.JournalGraph
import com.betabuddy.android.pose.VideoAnalyzer
import java.io.File
import java.io.FileOutputStream
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The capture screen on a device, from the logbook's button to a saved and analysed record, with a
 * drawn wall played from a file in place of the camera and stand-in pages in place of a pack's.
 * What is exercised is the app's side of the interface; the stand-ins recognise nothing.
 * Counterpart of ios/BetaBuddyUITests/CaptureUITests.swift and ios/BetaBuddyTests/CaptureFlowTests.swift.
 *
 * Pictures of each state are written to the app's external files directory, under `capture-shots`.
 */
@RunWith(AndroidJUnit4::class)
class CaptureScreenTest {
    @get:Rule val rule = createEmptyComposeRule()
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext
    private val store: ClimbStore by lazy { JournalGraph.store(context) }
    private var scenario: ActivityScenario<MainActivity>? = null
    private val sessions = mutableListOf<CaptureSession>()
    private val made = mutableListOf<String>()
    private lateinit var before: Set<String>

    @Before fun setUp() {
        CaptureDebug.reset()
        before = store.climbs.map { it.id }.toSet()
        CaptureDebug.directory = StandInPack.directory
        CaptureDebug.attitude = CaptureFrameInfo(pitch = 8.0, roll = 0.5, shake = 0.01)
        CaptureDebug.configure = { session -> sessions += session; session.framingInterval = 0.2 }
    }

    @After fun tearDown() {
        scenario?.close()
        setLanguage(null)
        CaptureDebug.reset()
        for (climb in store.climbs) if (climb.id !in before) runCatching { store.delete(climb) }
        clips.values.forEach { it.delete() }
        // A clip a failed test left undecided would be taken into the logbook at the next start.
        JournalGraph.stagingDirectory(context).listFiles().orEmpty().filter { it.name.startsWith("betabuddy-capture-") }.forEach { it.delete() }
        // A clip still being closed when the screen went away is finished first; then the gate is let go.
        val deadline = System.currentTimeMillis() + 15_000
        while (CameraGate.open.value && System.currentTimeMillis() < deadline) Thread.sleep(100)
        assertFalse("the camera gate is left open", CameraGate.open.value)
    }

    private val clips = mutableMapOf<Boolean, File>()
    /** The drawn wall, with or without somebody standing in front of it. */
    private fun wall(person: Boolean): File = clips.getOrPut(person) {
        val photo: Bitmap? = if (person) instrumentation.context.assets.open("pose.jpg").use { BitmapFactory.decodeStream(it) } else null
        SyntheticWallClip.make(File(context.cacheDir, "synthetic-wall-$person-${System.nanoTime()}.mp4"), photo = photo)
    }

    private fun standIn(capture: Boolean = true, analysis: Boolean = true, step: Int = 5, directoryIds: Boolean = true, silent: Boolean = false) {
        val pack = StandInPack(instrumentation.context, capture, analysis, step, directoryIds, silent)
        CaptureDebug.pack = pack.pack
        CaptureDebug.packFiles = pack
    }

    /**
     * The app's own language setting, made before any activity exists so that none has to be
     * rebuilt mid-test: through the system's per-app setting where there is one (Android 13+),
     * through AppCompat's own store below that. Null follows the device.
     */
    private fun setLanguage(language: String?) {
        if (android.os.Build.VERSION.SDK_INT >= 33) {
            instrumentation.uiAutomation.executeShellCommand("cmd locale set-app-locales ${context.packageName} --locales ${language ?: "\"\""}").close()
            Thread.sleep(500)
        } else {
            instrumentation.runOnMainSync { AppLanguage.select(language) }
        }
    }

    private fun launch(language: String? = null) {
        if (language != null) setLanguage(language)
        scenario = ActivityScenario.launch(
            Intent(context, MainActivity::class.java).putExtra("initialTab", 1).putExtra("gymNoticeSeen", true).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
        )
        await("captureClimb")
        rule.onNodeWithTag("captureClimb").performClick()
        await("captureScreen")
    }

    // While the activity is being rebuilt (a language change) there is no hierarchy to look in: that is "not there yet".
    private fun exists(tag: String) = runCatching { rule.onAllNodesWithTag(tag, useUnmergedTree = true).fetchSemanticsNodes().isNotEmpty() }.getOrDefault(false)
    private fun await(tag: String, seconds: Int = 20) = rule.waitUntil(seconds * 1000L) { exists(tag) }
    private fun awaitGone(tag: String, seconds: Int = 20) = rule.waitUntil(seconds * 1000L) { !exists(tag) }
    private fun state(tag: String): String? =
        rule.onAllNodesWithTag(tag, useUnmergedTree = true).fetchSemanticsNodes().firstOrNull()?.config?.getOrNull(SemanticsProperties.StateDescription)
    private fun awaitState(tag: String, value: String, seconds: Int = 20) = rule.waitUntil(seconds * 1000L) { state(tag) == value }
    /** Everything written inside the node with this tag, in order. */
    private fun text(tag: String): String {
        fun collect(node: androidx.compose.ui.semantics.SemanticsNode): List<String> =
            node.config.getOrNull(SemanticsProperties.Text).orEmpty().map { it.text } + node.children.flatMap(::collect)
        return collect(rule.onAllNodesWithTag(tag, useUnmergedTree = true).fetchSemanticsNodes().first()).joinToString(" ")
    }
    private fun tapPicture(x: Double, y: Double) =
        rule.onNodeWithTag("capturePicture").performTouchInput { click(Offset((width * x).toFloat(), (height * y).toFloat())) }
    private fun newClimbs(): List<Climb> = store.climbs.filter { it.id !in before }
    private fun session() = sessions.last()

    private fun shot(name: String) {
        rule.waitForIdle()
        val directory = File(context.getExternalFilesDir(null), "capture-shots").apply { mkdirs() }
        val picture = instrumentation.uiAutomation.takeScreenshot() ?: return
        FileOutputStream(File(directory, "$name.png")).use { picture.compress(Bitmap.CompressFormat.PNG, 100, it) }
        picture.recycle()
    }

    @Test fun fromFramingToASavedAndAnalysedRecord() {
        standIn(step = 12)   // each scripted verdict stays long enough to be looked at
        CaptureDebug.video = wall(person = true)
        CaptureDebug.fieldOfView = 52.0
        CaptureDebug.location = GymLocator.Fix(StandInPack.LATITUDE, StandInPack.LONGITUDE, 30.0)
        launch("zh-Hans")

        // The page's verdicts arrive as codes; each is shown as this build's own sentence.
        awaitState("captureHint", "too_dark")
        assertEquals("画面太暗，换个亮一点的位置", text("captureHint"))
        // The guidance sits above the picture, not on it: the top of the wall stays visible and tappable.
        val pill = rule.onNodeWithTag("captureHint").fetchSemanticsNode().boundsInRoot
        val picture = rule.onNodeWithTag("capturePicture").fetchSemanticsNode().boundsInRoot
        assertTrue("$pill / $picture", pill.bottom <= picture.top + 1f)
        // A gym 200 m away is offered, not chosen.
        await("captureOffer")
        assertTrue(text("captureOffer"), text("captureOffer").contains(StandInPack.DIRECTORY_GYM))
        assertEquals("what is selected is still the last record's gym, if any", store.lastGym?.second.orEmpty(), session().gymName)
        shot("11-framing-blocked")
        awaitState("captureHint", "cut_top")
        assertEquals("上面出画了，往后退一点", text("captureHint"))
        shot("12-framing-move")
        // A code and a move no build has heard of: the generic line, and no move.
        awaitState("captureHint", "lens_flare_2031")
        assertEquals("换个机位试试", text("captureHint"))
        rule.onNodeWithTag("useNearestGym").performClick()
        awaitGone("captureOffer")
        assertEquals(StandInPack.DIRECTORY_GYM, session().gymName)
        await("captureReady", 30)
        assertTrue(exists("captureQuality"))
        assertEquals("colour", state("capturePrompt"))
        pictureShare("ready, colours to pick")
        shot("13-framing-ready-colours")

        // Swatches carry no colour name, only a count.
        await("captureColor-yellow")
        val swatch = rule.onAllNodesWithTag("captureColor-yellow", useUnmergedTree = true).fetchSemanticsNodes().first().config
        assertEquals(listOf("颜色 1, 6 个岩点"), swatch.getOrNull(SemanticsProperties.ContentDescription))
        rule.onNodeWithTag("captureColor-yellow").performClick()
        // The gym is one of the pack's, so the stand-in "recognises" a problem; it is shown as a guess,
        // and the prompt says where the start and finish marks come from.
        await("captureRoute")
        assertEquals("认出来像是：合成线路 黄 1 · V3", text("captureRoute"))
        awaitState("capturePrompt", "route")
        assertEquals("起步和结束是按认出的这条线标的。不对的话，点一下岩点改。", text("capturePrompt"))
        // Its grade is filled in, and marked as not the user's own.
        await("captureGradePrefilled")
        assertEquals("V3", session().grade); assertTrue(session().gradePrefilled)
        rule.waitUntil(10_000) { session().assessment?.holds?.size == 6 }
        pictureShare("problem recognised, grade filled in")
        shot("14-colour-chosen-guesses")

        // One tap marks the start hand; then the finish.
        tapPicture(0.30, 0.80)
        awaitState("capturePrompt", "finish")
        rule.onNodeWithTag("captureTool-finish").performScrollTo().performClick()
        tapPicture(0.55, 0.22)
        awaitState("capturePrompt", "marked")
        shot("15-start-and-finish-marked")
        tapPicture(0.95, 0.05)   // nothing there
        awaitState("capturePrompt", "missed")
        shot("16-no-hold-there")
        val marked = session().assessment!!
        assertEquals(setOf("y1"), marked.holds.filter { it.role == CaptureRole.START_HAND && it.user }.map { it.id }.toSet())
        assertEquals("y6", marked.holds.single { it.role == CaptureRole.FINISH }.id)
        assertTrue(exists("captureResetMarks"))

        // "Not this one": the page moves on to its next candidate. That one's grade is written the
        // way some gyms do, which is shown but never filled in; the filled-in grade goes away with the problem it came from.
        rule.onNodeWithTag("captureRejectRoute").performClick()
        rule.waitUntil(20_000) { exists("captureRoute") && text("captureRoute").contains("合成线路 黄 2") }
        assertEquals("认出来像是：合成线路 黄 2 · 紫条 11–12", text("captureRoute"))
        awaitGone("captureGradePrefilled")
        assertEquals(com.betabuddy.android.data.UNGRADED_VALUE, session().grade)

        // The user's own choice of grade is kept whatever is recognised afterwards.
        rule.onNodeWithTag("captureGrade").performClick()
        await("captureGrade-V4")
        rule.onNodeWithTag("captureGrade-V4").performClick()
        rule.waitUntil(5_000) { session().grade == "V4" && !session().gradePrefilled }

        // Nothing is analysed while the camera is in use.
        assertTrue(CameraGate.open.value)
        rule.onNodeWithTag("captureRecord").performClick()
        await("captureStop")
        rule.waitUntil(20_000) { session().recordedSeconds >= 2.0 }
        assertTrue(exists("captureElapsed"))
        shot("17-recording")
        rule.onNodeWithTag("captureStop").performClick()

        // Saved first…
        await("captureSaved", 30)
        shot("18-saved-analysis-queued")
        val saved = newClimbs().single()
        assertEquals("V4", saved.grade)
        assertEquals(StandInPack.DIRECTORY_GYM, saved.gymName); assertEquals("syn:east", saved.gymId)
        val capture = saved.capture!!
        assertEquals("yellow", capture.colorKey); assertEquals(listOf(226, 196, 40), capture.colorRGB)
        assertEquals(StandInPack.VERSION, capture.packVersion)
        assertTrue(capture.hasPlan)
        // The problem that was not rejected is kept, by id and its own name; the rejected one is not.
        assertEquals("${StandInPack.VENUE_ID}/yellow-02", capture.routeId); assertEquals("合成线路 黄 2", capture.routeName)
        assertEquals("${StandInPack.VENUE_ID}/yellow-02", capture.summary?.route?.id)
        assertTrue(store.file(saved.videoName).length() > 10_000)
        assertTrue(saved.analysis.duration >= 1.0)
        val plan = Json.parseToJsonElement(store.capturePlan(saved.id)!!).jsonObject
        assertEquals(StandInPack.VENUE_ID, plan.getValue("meta").jsonObject.getValue("venue").jsonPrimitive.content)
        assertEquals(52.0, plan.getValue("meta").jsonObject.getValue("fovH").jsonPrimitive.content.toDouble(), 0.01)
        val roles = plan.getValue("holds").jsonArray.associate { it.jsonObject.getValue("id").jsonPrimitive.content to it.jsonObject.getValue("role").jsonPrimitive.content }
        assertEquals("start_hand", roles["y1"]); assertEquals("finish", roles["y6"])
        assertEquals(2, capture.summary?.startHands?.let { 2 }); assertTrue(capture.summary?.finish == true)

        // …then pose recognition, then the pack's analysis with the plan, none of it waited for.
        try {
            await("captureViewAnalysis", 120)
        } catch (error: androidx.compose.ui.test.ComposeTimeoutException) {
            val now = store.climbs.first { it.id == saved.id }
            throw AssertionError("not analysed: pending=${now.pendingAnalysis} queued=${now.capture?.analysisQueued} poses=${now.analysis.frames.size} world=${now.analysis.hasWorldLandmarks} " +
                "service=${com.betabuddy.android.journal.AnalysisService.state.value} results=${store.analysisResultIds()} gate=${CameraGate.open.value}")
        }
        shot("19-saved-analysed")
        val done = store.climbs.first { it.id == saved.id }
        assertNull(done.pendingAnalysis); assertNull(done.capture?.analysisQueued)
        assertTrue(done.analysis.hasWorldLandmarks)
        val result = Json.parseToJsonElement(store.analysisResult(saved.id)!!).jsonObject
        assertEquals("the plan reaches the analysis page as it was stored", store.capturePlan(saved.id), result.getValue("capture").jsonPrimitive.content)
        assertTrue(result.getValue("poses").jsonPrimitive.content.toInt() > 0)

        rule.onNodeWithTag("captureAgain").performClick()
        await("capturePrompt")
        // Another attempt keeps the gym, the grade and the colour.
        assertEquals("yellow", session().color); assertEquals("V4", session().grade); assertEquals(StandInPack.DIRECTORY_GYM, session().gymName)
        assertTrue(CameraGate.open.value)
        rule.onNodeWithTag("captureClose").performClick()
        awaitGone("captureScreen")
        rule.waitUntil(5_000) { !CameraGate.open.value }
        // The page's frames were upright JPEGs of the announced size, handed over a few times a second.
        val metrics = sessions.first().metrics
        assertTrue(metrics.summary, metrics.frames > 20 && metrics.assessments > 20)
        android.util.Log.i("capture-test", "stand-in page, file source: ${metrics.summary}")
    }

    @Test fun aBlockedPositionAsksBeforeRecordingAndAnInterruptionSavesWhatThereIs() {
        standIn(analysis = false, step = 100_000)
        CaptureDebug.video = wall(person = false)
        launch()
        awaitState("captureHint", "too_dark")
        assertEquals(context.getString(R.string.capture_reason_too_dark), text("captureHint"))
        rule.onNodeWithTag("captureRecord").performClick()
        await("recordAnyway")
        shot("21-confirm-blocked")
        rule.onNodeWithTag("recordAnyway").performClick()
        await("captureStop")
        rule.waitUntil(20_000) { session().recordedSeconds >= 1.5 }
        // The pictures end the way another app taking the camera would end them.
        (CaptureDebug.lastSource as FileCaptureSource).interrupt()
        await("captureSaved", 30)
        await("captureStopReason")
        assertEquals(context.getString(R.string.capture_stop_interrupted), text("captureStopReason"))
        shot("22-saved-after-interruption")
        val saved = newClimbs().single()
        assertFalse(saved.capture!!.summary!!.ok)
        assertEquals(listOf("too_dark"), saved.capture!!.summary!!.issues)
        assertNull("no analysis page in this pack, so none is asked for", saved.capture!!.analysisQueued)
        assertNull(saved.capture!!.colorKey)
        // Nobody in the picture: pose recognition says so, and the record stays.
        await("captureAnalysisMessage", 90)
        assertNull(store.climbs.first { it.id == saved.id }.pendingAnalysis)
        rule.onNodeWithTag("captureDone").performClick()
        awaitGone("captureScreen")
    }

    @Test fun withoutACapturePageTheCameraStillRecordsAndTheLimitStopsIt() {
        CaptureDebug.noPack = true
        CaptureDebug.video = wall(person = false)
        val configure = CaptureDebug.configure
        CaptureDebug.configure = { session -> configure?.invoke(session); session.maxSeconds = 2.0 }
        launch()
        await("captureNoChecks")
        assertEquals(context.getString(R.string.capture_no_checks), text("captureNoChecks"))
        assertFalse(exists("capturePrompt"))
        shot("31-no-capture-page")
        rule.onNodeWithTag("captureRecord").performClick()
        await("captureSaved", 40)
        assertEquals(context.getString(R.string.capture_stop_limit, CaptureLimits.MAX_MINUTES), text("captureStopReason"))
        val saved = newClimbs().single()
        assertNull(saved.capture!!.packVersion); assertFalse(saved.capture!!.hasPlan); assertNull(saved.capture!!.analysisQueued)
        assertEquals("balanced", saved.pendingAnalysis ?: "balanced")
        assertTrue(saved.analysis.duration in 1.5..4.0)
        rule.onNodeWithTag("captureDone").performClick()
        awaitGone("captureScreen")
    }

    @Test fun tooLittleSpaceAndTooShortARecordingAreSaidAndNothingIsSaved() {
        standIn(analysis = false)
        CaptureDebug.video = wall(person = false)
        var free = 10L * 1024 * 1024
        val configure = CaptureDebug.configure
        CaptureDebug.configure = { session -> configure?.invoke(session); session.freeBytes = { free } }
        launch()
        await("captureHint")
        rule.onNodeWithTag("captureRecord").performClick()
        await("recordAnyway"); rule.onNodeWithTag("recordAnyway").performClick()
        await("captureNotice")
        assertEquals(context.getString(R.string.capture_notice_low_storage), text("captureNotice"))
        shot("41-low-storage")
        instrumentation.runOnMainSync { session().notice = null }
        awaitGone("captureNotice")
        free = 10L * 1024 * 1024 * 1024
        // Stopped at once: there is nothing to review.
        instrumentation.runOnMainSync { session().startRecording(); session().stop(CaptureStop.USER) }
        await("captureNotice")
        assertEquals(context.getString(R.string.capture_notice_too_short), text("captureNotice"))
        assertTrue(newClimbs().isEmpty())
        assertTrue(File(JournalGraph.stagingDirectory(context).path).listFiles().orEmpty().none { it.name.startsWith("betabuddy-capture-") })
        // Space runs out while recording: closed and saved with what there is.
        instrumentation.runOnMainSync { session().notice = null; session().watchIntervalMillis = 300 }
        awaitGone("captureNotice")
        rule.onNodeWithTag("captureRecord").performClick()
        await("recordAnyway"); rule.onNodeWithTag("recordAnyway").performClick()
        await("captureStop")
        rule.waitUntil(20_000) { session().recordedSeconds >= 1.5 }
        free = 100L * 1024 * 1024
        await("captureSaved", 30)
        assertEquals(context.getString(R.string.capture_stop_storage), text("captureStopReason"))
        assertEquals(1, newClimbs().size)
    }

    /** How much of the screen's width the picture has; it must stay the main thing on the screen. */
    private fun pictureShare(what: String): Float {
        val picture = rule.onNodeWithTag("capturePicture").fetchSemanticsNode().boundsInRoot
        val screen = rule.onNodeWithTag("captureScreen").fetchSemanticsNode().boundsInRoot
        val share = picture.width / screen.width
        android.util.Log.i("capture-test", "picture is ${"%.0f".format(share * 100)}% of the width ($what)")
        assertTrue("picture only ${share * 100}% of the width ($what)", share >= 0.70f)
        return share
    }

    private fun stagedClips() = JournalGraph.stagingDirectory(context).listFiles().orEmpty().filter { it.name.startsWith("betabuddy-capture-") && it.extension == "mp4" }
    private fun closeScreen() { rule.onNodeWithTag("captureClose").performClick(); awaitGone("captureScreen") }
    private fun reopen() { rule.onNodeWithTag("captureClimb").performClick(); await("captureScreen") }

    @Test fun aTapRightBeforeRecordingIsInThePlanAndAChangedGymIsNotLockedUnderTheOldOne() {
        assumeTrue("needs a logbook without a gym in it", store.lastGym == null)
        standIn(analysis = false, step = 1)
        CaptureDebug.video = wall(person = false)
        CaptureDebug.location = GymLocator.Fix(StandInPack.LATITUDE, StandInPack.LONGITUDE, 30.0)
        launch()
        // No gym selected yet and one right here: offered with its distance and where that comes from, to be chosen, not chosen.
        awaitState("captureOffer", "here")
        assertTrue(text("captureOffer"), text("captureOffer").contains(StandInPack.DIRECTORY_GYM) && text("captureOffer").contains("200 m"))
        assertEquals("© OpenStreetMap contributors", text("captureOfferCredit"))
        assertEquals(context.getString(R.string.capture_offer_choose), text("useNearestGym"))
        assertEquals("", session().gymName)
        rule.onNodeWithTag("useNearestGym").performClick()
        awaitGone("captureOffer")
        await("captureColor-yellow")
        rule.onNodeWithTag("captureColor-yellow").performClick()
        rule.waitUntil(20_000) { session().assessment?.holds?.size == 6 && session().assessment?.route != null }
        val before = session().assessment!!
        assertFalse(before.startConfirmed)
        assertEquals(2, before.holds.count { it.role == CaptureRole.START_HAND })

        // A tap, and Record in the same breath: no reply of the page's has shown the tap yet.
        val handed = session().metrics.frames
        instrumentation.runOnMainSync { session().correct(0.30, 0.80, CaptureRole.START_HAND); session().startRecording() }
        await("captureStop")
        assertTrue(exists("captureRecordingNote"))
        rule.waitUntil(20_000) { session().recordedSeconds >= 1.5 }
        rule.onNodeWithTag("captureStop").performClick()
        await("captureSaved", 30)
        // Nothing was prepared for the page after `lock`: it would only have been dropped.
        assertTrue("${session().metrics.frames} after $handed", session().metrics.frames <= handed + 1)
        val first = newClimbs().single()
        // What was stored is the page's answer to `lock`, which had the tap in it.
        val plan = Json.parseToJsonElement(store.capturePlan(first.id)!!).jsonObject
        val y1 = plan.getValue("holds").jsonArray.map { it.jsonObject }.single { it.getValue("id").jsonPrimitive.content == "y1" }
        assertEquals("start_hand", y1.getValue("role").jsonPrimitive.content); assertEquals("true", y1.getValue("user").jsonPrimitive.content)
        assertEquals(1, first.capture!!.summary!!.startHands)     // the page's own second guess at a start hand was withdrawn
        assertEquals(StandInPack.VENUE_ID, plan.getValue("meta").jsonObject.getValue("venue").jsonPrimitive.content)
        // The problem is kept by id and its own name; what the page called it in this language, and its grade text, are not.
        assertEquals("${StandInPack.VENUE_ID}/yellow-01", first.capture!!.routeId); assertEquals("合成线路 黄 1", first.capture!!.routeName)
        val kept = first.capture!!.summary!!.route!!
        assertNull(kept.name); assertNull(kept.grade); assertEquals("合成线路 黄 1", kept.nameZh)

        // Another take. The gym is changed and Record pressed before the page has seen a frame for it:
        // what the page holds belongs to the other gym, so this clip is recorded without a plan.
        rule.onNodeWithTag("captureAgain").performClick()
        rule.waitUntil(20_000) { session().assessment?.holds?.size == 6 }
        instrumentation.runOnMainSync { session().selectGym("syn:west", "合成岩馆（西门）"); session().startRecording() }
        await("captureStop")
        rule.waitUntil(20_000) { session().recordedSeconds >= 1.5 }
        rule.onNodeWithTag("captureStop").performClick()
        rule.waitUntil(30_000) { newClimbs().size == 2 && exists("captureSaved") }
        val second = newClimbs().first { it.id != first.id }
        assertEquals("合成岩馆（西门）", second.gymName)
        assertFalse(second.capture!!.hasPlan); assertNull(second.capture!!.summary); assertNull(store.capturePlan(second.id)); assertNull(second.capture!!.routeId)
        closeScreen()
    }

    @Test fun aPageThatNeverComesUpAndACameraThatStopsAreSaidAndRecordWaits() {
        standIn(analysis = false, silent = true)
        CaptureDebug.video = wall(person = false)
        val configure = CaptureDebug.configure
        CaptureDebug.configure = { session -> configure?.invoke(session); session.readyLimitMillis = 2_000 }
        launch()
        // The page loaded but never said it was ready: after a while the screen stops waiting, and nothing was prepared for it.
        await("captureNoChecks", 30)
        assertEquals(context.getString(R.string.capture_checks_failed), text("captureNoChecks"))
        assertEquals(0, session().metrics.frames)
        assertEquals(CaptureSession.Camera.RUNNING, session().camera)
        // The pictures stop while framing: the preview is a frozen picture now, and recording cannot be started from it.
        (CaptureDebug.lastSource as FileCaptureSource).interrupt()
        rule.waitUntil(10_000) { session().camera != CaptureSession.Camera.RUNNING }
        awaitGone("captureNoChecks")
        rule.onNodeWithTag("captureRecord").performClick()
        rule.waitForIdle()
        assertEquals(CaptureSession.Phase.Framing, session().phase)
        assertFalse(exists("captureStop"))
        closeScreen()
    }

    @Test fun aClipThatCouldNotBeSavedStaysUntilTheUserDecides() {
        CaptureDebug.noPack = true
        CaptureDebug.video = wall(person = false)
        val configure = CaptureDebug.configure
        CaptureDebug.configure = { session -> configure?.invoke(session); session.saveFault = { throw java.io.IOException("synthetic failure") } }
        launch()
        await("captureNoChecks")
        rule.onNodeWithTag("captureRecord").performClick()
        await("captureStop")
        rule.waitUntil(20_000) { session().recordedSeconds >= 1.5 }
        rule.onNodeWithTag("captureStop").performClick()
        await("captureSaveFailed", 30)
        shot("42-save-failed")
        assertTrue(newClimbs().isEmpty())
        assertEquals(1, stagedClips().size)
        // The screen cannot be left with the clip undecided: closing does nothing, the clip stays.
        rule.onNodeWithTag("captureClose").performClick()
        rule.waitForIdle()
        assertTrue(exists("captureScreen") && exists("captureSaveRetry") && exists("captureSaveDiscard"))
        assertEquals(1, stagedClips().size)
        instrumentation.runOnMainSync { session().saveFault = null }
        rule.onNodeWithTag("captureSaveRetry").performClick()
        await("captureSaved", 30)
        assertEquals(1, newClimbs().size)
        assertTrue(stagedClips().isEmpty())
        assertTrue(JournalGraph.stagingDirectory(context).listFiles().orEmpty().none { it.name.endsWith(".json") })
        closeScreen()
    }

    @Test fun thePositionIsSaidAsItIsAndANearestGymIsOnlyEverPutForward() {
        assumeTrue("needs a logbook without a gym in it", store.lastGym == null)
        CaptureDebug.noPack = true
        CaptureDebug.video = wall(person = false)
        fun openSheet() { rule.onNodeWithTag("captureGym").performClick(); await("captureGymSheet") }
        fun closeSheet() { instrumentation.uiAutomation.performGlobalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_BACK); awaitGone("captureGymSheet") }

        // Refused: said so, with where to change it. Nothing about gyms nearby, which nobody looked for.
        CaptureDebug.locationState = GymLocator.StandIn(GymLocator.Access.REFUSED)
        launch()
        await("captureNoChecks")
        assertFalse(exists("captureOffer") || exists("captureAskLocation"))
        openSheet()
        assertEquals(context.getString(R.string.capture_location_refused), text("locationRefused"))
        assertFalse(exists("noGymNear") || exists("askLocation") || exists("coordinateCredit"))
        assertTrue(exists("allGyms"))
        closeSheet(); closeScreen()

        // Waiting for a position.
        CaptureDebug.locationState = GymLocator.StandIn(GymLocator.Access.ALLOWED, fix = null, locating = true)
        reopen(); await("captureNoChecks")
        assertFalse(exists("captureOffer") || exists("captureAskLocation"))
        openSheet()
        assertTrue(text("locationWaiting").contains(context.getString(R.string.capture_location_waiting)))
        assertFalse(exists("noGymNear") || exists("locationUnavailable"))
        closeSheet(); closeScreen()

        // Allowed, but no position could be had.
        CaptureDebug.locationState = GymLocator.StandIn(GymLocator.Access.ALLOWED, fix = null, locating = false)
        reopen(); await("captureNoChecks")
        openSheet()
        assertEquals(context.getString(R.string.capture_location_unavailable), text("locationUnavailable"))
        assertTrue(exists("locationRetry"))
        assertFalse(exists("noGymNear") || exists("locationWaiting"))
        shot("52-gym-sheet-no-position")
        closeSheet(); closeScreen()

        // A position, and no gym of the directory anywhere near it: only now is that what is said.
        CaptureDebug.locationState = GymLocator.StandIn(fix = GymLocator.Fix(10.0, 10.0, 30.0))
        reopen(); await("captureNoChecks")
        assertFalse(exists("captureOffer"))
        openSheet()
        assertEquals(context.getString(R.string.capture_gym_none_near), text("noGymNear"))
        closeSheet(); closeScreen()

        // Never asked: one tap away, on the screen itself and in the sheet. (The system's own question is not pressed here.)
        CaptureDebug.locationState = GymLocator.StandIn(GymLocator.Access.NOT_ASKED)
        reopen(); await("captureNoChecks")
        assertEquals(context.getString(R.string.capture_gym_ask_location), text("captureAskLocation"))
        shot("53-ask-location")
        openSheet()
        assertTrue(exists("askLocation"))
        closeSheet()
        rule.onNodeWithTag("declineAskLocation").performClick()
        awaitGone("captureAskLocation")
        closeScreen()

        // A coarse position (what approximate location gives): the nearest gym is put forward as a guess, without a distance.
        CaptureDebug.locationState = GymLocator.StandIn(fix = GymLocator.Fix(StandInPack.LATITUDE, StandInPack.LONGITUDE, 2_000.0))
        reopen(); await("captureNoChecks")
        awaitState("captureOffer", "guess")
        assertEquals(context.getString(R.string.capture_offer_guess, StandInPack.DIRECTORY_GYM), text("captureOffer"))
        assertFalse(exists("captureOfferCredit"))
        assertEquals("", session().gymName)
        shot("54-gym-guess")
        rule.onNodeWithTag("useNearestGym").performClick()
        awaitGone("captureOffer")
        assertEquals(StandInPack.DIRECTORY_GYM, session().gymName)
        closeScreen()
    }

    // The two tests below take a real fix from the system's location service. They are run one at a
    // time with `-e geoFix 1`, after the permission was granted from outside (granting one here and
    // revoking the other would end the process) and the emulator was given a position next to the
    // synthetic directory's first gym:  adb emu geo fix 120.0 30.0
    private fun granted(permission: String) = context.checkSelfPermission(permission) == android.content.pm.PackageManager.PERMISSION_GRANTED
    private fun realFix() = InstrumentationRegistry.getArguments().getString("geoFix") == "1"

    @Test fun withPreciseLocationTheNearestGymIsOfferedWithItsDistance() {
        assumeTrue("run with -e geoFix 1 and precise location granted", realFix() && granted(Manifest.permission.ACCESS_FINE_LOCATION))
        assumeTrue("needs a logbook without a gym in it", store.lastGym == null)
        CaptureDebug.noPack = true
        CaptureDebug.video = wall(person = false)
        launch()
        await("captureNoChecks")
        // One fix from the system: a gym whose own position is known, 200 m away. Offered with the number, not chosen.
        rule.waitUntil(60_000) { state("captureOffer") == "here" }
        assertTrue(text("captureOffer"), text("captureOffer").contains(StandInPack.DIRECTORY_GYM) && Regex("[12]\\d0 m").containsMatchIn(text("captureOffer")))
        assertEquals("© OpenStreetMap contributors", text("captureOfferCredit"))
        assertEquals("", session().gymName)
        shot("55-precise-location-offer")
        rule.onNodeWithTag("captureGym").performClick(); await("captureGymSheet")
        assertTrue(text("nearbyGym-syn:east"), Regex("[12]\\d0 m").containsMatchIn(text("nearbyGym-syn:east")))
        assertTrue(text("nearbyGym-syn:station").contains("about 1.5 km"))
        shot("56-precise-location-sheet")
        instrumentation.uiAutomation.performGlobalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_BACK); awaitGone("captureGymSheet")
        closeScreen()
    }

    @Test fun withApproximateLocationTheListIsOrderedWithoutNumbersAndTheNearestIsAGuess() {
        assumeTrue("run with -e geoFix 1 and only approximate location granted",
            realFix() && granted(Manifest.permission.ACCESS_COARSE_LOCATION) && !granted(Manifest.permission.ACCESS_FINE_LOCATION))
        assumeTrue("needs a logbook without a gym in it", store.lastGym == null)
        CaptureDebug.noPack = true
        CaptureDebug.video = wall(person = false)
        launch()
        await("captureNoChecks")
        // The system blurs the position by a kilometre or two: the nearest gym the directory can place is a guess, without a distance.
        rule.waitUntil(60_000) { state("captureOffer") == "guess" }
        assertFalse(exists("captureOfferCredit"))
        assertFalse(text("captureOffer"), Regex("\\d+ ?(m|km)\\b").containsMatchIn(text("captureOffer")))
        assertEquals("", session().gymName)
        shot("57-approximate-location-guess")
        rule.onNodeWithTag("captureGym").performClick(); await("captureGymSheet")
        // The gyms that can be placed are listed, nearest first as far as a blurred position can tell, and none carries a number.
        for (id in listOf("syn:east", "syn:station", "syn:west")) {
            assertTrue(id, exists("nearbyGym-$id"))
            assertFalse(text("nearbyGym-$id"), Regex("\\d+(\\.\\d+)? ?(m|km)\\b").containsMatchIn(text("nearbyGym-$id")))
        }
        shot("58-approximate-location-sheet")
        instrumentation.uiAutomation.performGlobalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_BACK); awaitGone("captureGymSheet")
        closeScreen()
    }

    @Test fun gymsAreSuggestedByDistanceAsFarAsTheDirectoryKnows() {
        standIn(analysis = false)
        CaptureDebug.video = wall(person = false)
        // Somewhere between the gyms: none is close enough to be offered.
        CaptureDebug.location = GymLocator.Fix(StandInPack.LATITUDE + 0.006, StandInPack.LONGITUDE, 30.0)
        // The most recent record was at the west gym: that is what the screen starts with.
        val earlier = wall(person = false).copyTo(File(context.cacheDir, "earlier-${System.nanoTime()}.mp4"))
        store.save("", "Earlier", "V2", false, "", earlier, byteArrayOf(1), com.betabuddy.android.data.PoseAnalysis(emptyList(), 0, 3.0, 0.5625),
            gymId = "syn:west", gymName = "合成岩馆（西门）", climbedAtMillis = System.currentTimeMillis() + 86_400_000)
        launch()
        await("captureHint")
        rule.waitUntil(10_000) { session().gymName == "合成岩馆（西门）" }
        assertEquals("syn:west", session().gymId)
        assertFalse(exists("captureOffer"))
        rule.onNodeWithTag("captureGym").performClick()
        await("captureGymSheet")
        // Gyms whose own position is known, nearest first: a number for a place; for one known only by
        // its nearest station (610 m away here), "about", and never less than a kilometre.
        assertTrue(text("nearbyGym-syn:east"), text("nearbyGym-syn:east").contains("470 m"))
        assertTrue(text("nearbyGym-syn:station"), text("nearbyGym-syn:station").contains("about 1 km"))
        assertTrue(exists("nearbyGym-syn:west"))
        // The ones only placed by district or city are listed without a distance.
        assertTrue(exists("cityGym-syn:district") && exists("cityGym-syn:city") && exists("cityGym-syn:nowhere"))
        assertFalse(exists("nearbyGym-syn:district") || exists("nearbyGym-syn:far") || exists("cityGym-syn:far"))
        assertFalse(text("cityGym-syn:district").contains(" km") || text("cityGym-syn:district").contains(" m"))
        rule.onNodeWithTag("coordinateCredit").assertIsDisplayed()
        assertEquals("© OpenStreetMap contributors", text("coordinateCredit"))
        assertFalse(exists("askLocation"))
        shot("51-gym-sheet")
        assertEquals("合成岩馆（西门）", session().gymName)   // looking at the list, nearer gyms and all, chooses nothing
        rule.onNodeWithTag("nearbyGym-syn:east").performClick()
        awaitGone("captureGymSheet")
        assertEquals(StandInPack.DIRECTORY_GYM, session().gymName); assertEquals("syn:east", session().gymId)
        rule.onNodeWithTag("captureClose").performClick()
        awaitGone("captureScreen")
    }

    @Test fun theDevicesOwnCameraStartsRecordsAndSaves() {
        // A smoke test of the camera library on whatever camera the device (or emulator scene) has; it says nothing about framing.
        assumeTrue(context.packageManager.hasSystemFeature(android.content.pm.PackageManager.FEATURE_CAMERA_ANY))
        instrumentation.uiAutomation.grantRuntimePermission(context.packageName, Manifest.permission.CAMERA)
        CaptureDebug.noPack = true
        launch()
        await("captureNoChecks", 40)
        assertEquals(CaptureSession.Camera.RUNNING, session().camera)
        val (width, height) = session().frameSize
        assertTrue("$width x $height", height > width && Math.abs(height * 9 - width * 16) <= 16)   // upright, 16:9
        assertNotNull(session().source.horizontalFieldOfView)
        shot("61-device-camera")
        rule.onNodeWithTag("captureRecord").performClick()
        await("captureStop")
        rule.waitUntil(30_000) { session().recordedSeconds >= 2.0 || session().notice != null }
        // A camera that hands the recorder nothing is noticed and said; here that is a failure of this device's camera.
        assertNull("the device's camera gave the recorder no video", session().notice)
        rule.onNodeWithTag("captureStop").performClick()
        await("captureSaved", 40)
        val saved = newClimbs().single()
        val info = VideoAnalyzer.probe(store.file(saved.videoName))
        assertTrue(info.hasVideo && (info.durationSeconds ?: 0.0) >= 1.5)
        // The clip is upright and has a picture track only: no sound was recorded.
        val extractor = android.media.MediaExtractor().apply { setDataSource(store.file(saved.videoName).path) }
        val kinds = (0 until extractor.trackCount).map { extractor.getTrackFormat(it).getString(android.media.MediaFormat.KEY_MIME).orEmpty().substringBefore('/') }
        extractor.release()
        assertEquals(listOf("video"), kinds)
        assertTrue(saved.analysis.aspectRatio < 1.0)
        rule.onNodeWithTag("captureDone").performClick()
        awaitGone("captureScreen")
    }
}
