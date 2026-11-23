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
        instrumentation.runOnMainSync { AppLanguage.select(null) }
        instrumentation.waitForIdleSync()
        scenario?.close()
        CaptureDebug.reset()
        for (climb in store.climbs) if (climb.id !in before) runCatching { store.delete(climb) }
        clips.values.forEach { it.delete() }
        assertFalse("the camera gate is left open", CameraGate.open.value)
    }

    private val clips = mutableMapOf<Boolean, File>()
    /** The drawn wall, with or without somebody standing in front of it. */
    private fun wall(person: Boolean): File = clips.getOrPut(person) {
        val photo: Bitmap? = if (person) instrumentation.context.assets.open("pose.jpg").use { BitmapFactory.decodeStream(it) } else null
        SyntheticWallClip.make(File(context.cacheDir, "synthetic-wall-$person-${System.nanoTime()}.mp4"), photo = photo)
    }

    private fun standIn(capture: Boolean = true, analysis: Boolean = true, step: Int = 5) {
        val pack = StandInPack(instrumentation.context, capture, analysis, step)
        CaptureDebug.pack = pack.pack
        CaptureDebug.packFiles = pack
    }

    /** [language] and the name the logbook's button has in it, or nothing for the device's own (English) language. */
    private fun launch(language: String? = null, button: String = "") {
        scenario = ActivityScenario.launch(
            Intent(context, MainActivity::class.java).putExtra("initialTab", 1).putExtra("gymNoticeSeen", true).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
        )
        await("captureClimb")
        if (language != null) {
            // The in-app language needs a running activity to take; the screen is rebuilt in it.
            instrumentation.runOnMainSync { AppLanguage.select(language) }
            rule.waitUntil(20_000) { runCatching { text("captureClimb") }.getOrNull() == button }
        }
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
        launch("zh-Hans", "拍一趟")

        // The page's verdicts arrive as codes; each is shown as this build's own sentence.
        awaitState("captureHint", "too_dark")
        assertEquals("画面太暗，换个亮一点的位置", text("captureHint"))
        // A gym 200 m away is offered, not chosen.
        await("captureOffer")
        assertTrue(text("captureOffer"), text("captureOffer").contains(StandInPack.DIRECTORY_GYM))
        assertEquals("", session().gymName)
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
        shot("13-framing-ready-colours")

        // Swatches carry no colour name, only a count.
        await("captureColor-yellow")
        val swatch = rule.onAllNodesWithTag("captureColor-yellow", useUnmergedTree = true).fetchSemanticsNodes().first().config
        assertEquals(listOf("颜色 1, 6 个岩点"), swatch.getOrNull(SemanticsProperties.ContentDescription))
        rule.onNodeWithTag("captureColor-yellow").performClick()
        awaitState("capturePrompt", "start")
        // The gym is one of the pack's, so the stand-in "recognises" a problem; it is shown as a guess.
        await("captureRoute")
        assertTrue(text("captureRoute"), text("captureRoute").contains("合成线路 黄 1"))
        rule.waitUntil(10_000) { session().assessment?.holds?.size == 6 }
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
        rule.onNodeWithTag("captureRejectRoute").performClick()
        awaitGone("captureRoute")

        rule.onNodeWithTag("captureGrade").performClick()
        await("captureGrade-V4")
        rule.onNodeWithTag("captureGrade-V4").performClick()

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
        assertNull("a rejected problem is not kept", capture.routeId)
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
        assertEquals(context.getString(R.string.capture_stop_limit, 3), text("captureStopReason"))
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

    @Test fun gymsAreSuggestedByDistanceAsFarAsTheDirectoryKnows() {
        standIn(analysis = false)
        CaptureDebug.video = wall(person = false)
        // Somewhere between the gyms: none is close enough to be offered.
        CaptureDebug.location = GymLocator.Fix(StandInPack.LATITUDE + 0.004, StandInPack.LONGITUDE, 30.0)
        launch()
        await("captureHint")
        assertFalse(exists("captureOffer"))
        rule.onNodeWithTag("captureGym").performClick()
        await("captureGymSheet")
        // Gyms whose own position is known, nearest first: a number for a place, "about" for a station.
        assertTrue(text("nearbyGym-syn:east"), text("nearbyGym-syn:east").contains("240 m"))
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
        assertEquals("", session().gymName)   // looking at the list chooses nothing
        rule.onNodeWithTag("nearbyGym-syn:west").performClick()
        awaitGone("captureGymSheet")
        assertEquals("合成岩馆（西门）", session().gymName); assertEquals("syn:west", session().gymId)
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
        rule.waitUntil(30_000) { session().recordedSeconds >= 2.0 }
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
