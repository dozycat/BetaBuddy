package com.betabuddy.android.edit

import android.content.Intent
import android.graphics.Bitmap
import androidx.activity.compose.setContent
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.SemanticsMatcher
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.compose.ui.test.onAllNodesWithTag
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.core.content.FileProvider
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.betabuddy.android.MainActivity
import com.betabuddy.android.capture.SyntheticWallClip
import com.betabuddy.android.data.ClimbStore
import com.betabuddy.android.data.PoseAnalysis
import com.betabuddy.android.gym.GymPack
import com.betabuddy.android.gym.GymPackStatus
import com.betabuddy.android.journal.JournalGraph
import com.betabuddy.android.review.rememberMainScope
import com.betabuddy.android.ui.theme.BetaBuddyTheme
import java.io.File
import java.io.FileOutputStream
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The edit page on a device, with a drawn wall for the clip, a made-up plan and the bundled people
 * detector: it renders, plays, renders again when a switch changes, offers the file for sharing and
 * leaves nothing behind when closed. Pictures go to the app's external files directory, under `edit-shots`.
 */
@RunWith(AndroidJUnit4::class)
class EditScreenTest {
    @get:Rule val rule = createEmptyComposeRule()
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext
    private val store: ClimbStore by lazy { JournalGraph.store(context) }
    private var scenario: ActivityScenario<MainActivity>? = null
    private val created = mutableListOf<File>()
    private val made = mutableListOf<String>()

    @After fun tearDown() {
        scenario?.close()
        for (climb in store.climbs) if (climb.id in made) runCatching { store.delete(climb) }
        created.forEach { it.delete() }
    }

    private val plan = """{"v":1,"duration":3,"trim":[0.2,2.6],
        "segments":[{"t0":0.2,"t1":1.4,"speed":1,"kind":"run"},{"t0":1.4,"t1":2.0,"speed":0.5,"kind":"slow"},{"t0":2.0,"t1":2.6,"speed":1,"kind":"run"}],
        "camera":[{"t":0.2,"cx":0.5,"cy":0.5,"s":1},{"t":2.0,"cx":0.5,"cy":0.45,"s":1.6}],
        "keep":{"feather":0.03,"boxes":[{"t":0,"b":[[0.3,0.2,0.4,0.72]]},{"t":3,"b":[[0.3,0.2,0.4,0.72]]}]},
        "plate":{"times":[0.2,0.5,0.8,1.1,1.4,1.7,2.0,2.3,2.6]}}"""

    private fun wall(): File = SyntheticWallClip.make(File(context.cacheDir, "edit-screen-${System.nanoTime()}.mp4").also { created += it })

    private fun exists(tag: String) = runCatching { rule.onAllNodesWithTag(tag, useUnmergedTree = true).fetchSemanticsNodes().isNotEmpty() }.getOrDefault(false)
    private fun await(tag: String, seconds: Int = 20) = rule.waitUntil(seconds * 1000L) { exists(tag) }

    /** The picture the preview is showing right now: the texture view of whichever window has one. */
    private fun previewPicture(): Bitmap? {
        var picture: Bitmap? = null
        instrumentation.runOnMainSync {
            fun find(view: android.view.View): android.view.TextureView? = when (view) {
                is android.view.TextureView -> view
                is android.view.ViewGroup -> (0 until view.childCount).firstNotNullOfOrNull { find(view.getChildAt(it)) }
                else -> null
            }
            picture = android.view.inspector.WindowInspector.getGlobalWindowViews().firstNotNullOfOrNull(::find)?.bitmap
        }
        return picture
    }

    /** True once the preview shows the drawn wall: mostly pale, not an empty black stage. */
    private fun previewShowsTheWall(): Boolean {
        val picture = previewPicture() ?: return false
        var pale = 0
        var total = 0
        for (y in 0 until picture.height step 16) for (x in 0 until picture.width step 16) {
            total += 1
            val colour = picture.getPixel(x, y)
            if (android.graphics.Color.red(colour) > 150 && android.graphics.Color.green(colour) > 150 && android.graphics.Color.blue(colour) > 140) pale += 1
        }
        return total > 0 && pale > total / 2
    }

    private fun shot(name: String) {
        rule.waitForIdle()
        val directory = File(context.getExternalFilesDir(null), "edit-shots").apply { mkdirs() }
        val picture = instrumentation.uiAutomation.takeScreenshot() ?: return
        FileOutputStream(File(directory, "$name.png")).use { picture.compress(Bitmap.CompressFormat.PNG, 100, it) }
        picture.recycle()
    }

    @Test fun thePageRendersPlaysRendersAgainAndCleansUp() {
        val clip = wall()
        val id = "EDIT-SCREEN-TEST"
        var closed = false
        scenario = ActivityScenario.launch(Intent(context, MainActivity::class.java).putExtra("initialTab", 1).putExtra("gymNoticeSeen", true).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        scenario!!.onActivity { activity ->
            activity.setContent {
                BetaBuddyTheme {
                    val scope = rememberMainScope()
                    val session = androidx.compose.runtime.remember { EditSession(activity, id, clip, EditPlan.parse(plan)!!, scope) }
                    if (!closed) EditScreen(session, onDone = { closed = true; activity.setContent { } })
                }
            }
        }
        await("editScreen")
        // The switches on offer that the plan has something for; passer-by removal is not offered yet.
        for (tag in listOf("editSlow", "editCamera", "editShare")) rule.onNodeWithTag(tag).assertIsDisplayed()
        rule.onNodeWithTag("editRemovePassers").assertDoesNotExist()
        shot("01-working")
        await("editPreview", 180)
        rule.waitUntil(15_000) { previewShowsTheWall() }
        shot("02-ready")
        val directory = EditFiles.directory(context, id)
        val first = directory.listFiles()!!.single()
        assertTrue(first.length() > 10_000)
        // What the share sheet would be handed: a content URI of the app's own provider that reads as the file.
        val uri = FileProvider.getUriForFile(context, context.packageName + ".edits", first)
        assertEquals("content", uri.scheme)
        assertEquals(first.length(), context.contentResolver.openInputStream(uri)!!.use { it.readBytes().size.toLong() })
        assertEquals("video/mp4", context.contentResolver.getType(uri))

        // A switch changed: rendered again, and only the new file is kept.
        rule.onNodeWithTag("editSlow").performClick()
        rule.waitUntil(20_000) { exists("editWorking") || directory.listFiles()!!.none { it == first } }
        await("editPreview", 180)
        rule.waitUntil(10_000) { directory.listFiles()!!.size == 1 }
        val second = directory.listFiles()!!.single()
        assertFalse(second == first)
        shot("03-slow-motion-off")

        rule.onNodeWithTag("editDone").performClick()
        rule.waitUntil(10_000) { !exists("editScreen") }
        rule.waitUntil(10_000) { !second.exists() }
        assertTrue(directory.listFiles().isNullOrEmpty())
    }

    @Test fun writeFailureRetryActualProgressAndCancellation() {
        val clip = SyntheticWallClip.make(File(context.cacheDir,"edit-recovery-${System.nanoTime()}.mp4").also { created += it },width=720,height=1280,seconds=12.0)
        val renderPlan = EditPlan.parse("""{"v":1,"duration":12,"trim":[0,12],"segments":[{"t0":0,"t1":12,"speed":0.5}],"camera":[{"t":0,"cx":0.5,"cy":0.5,"s":1},{"t":12,"cx":0.5,"cy":0.5,"s":1.2}]}""")!!
        val id = "EDIT-RECOVERY-TEST"
        lateinit var session: EditSession
        lateinit var directory: File
        var closed = false
        scenario = ActivityScenario.launch(Intent(context,MainActivity::class.java).putExtra("initialTab",1).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        scenario!!.onActivity { activity ->
            directory = EditFiles.directory(context,id).apply { deleteRecursively(); writeText("blocked output") }
            activity.setContent { BetaBuddyTheme {
                val scope = rememberMainScope()
                session = androidx.compose.runtime.remember { EditSession(activity,id,clip,renderPlan,scope) }
                EditScreen(session) { closed=true; activity.setContent {} }
            } }
        }
        await("editFailed",60); rule.onNodeWithTag("editShare").assertIsNotEnabled(); shot("recovery-failed")
        assertTrue(directory.delete())
        rule.onNodeWithTag("editRetry").performClick()
        rule.waitUntil(60_000) { (session.state as? EditSession.State.Working)?.progress?.let { it > 0 && it < 1 } == true }
        val progressNodes = rule.onAllNodes(SemanticsMatcher.keyIsDefined(SemanticsProperties.ProgressBarRangeInfo)).fetchSemanticsNodes()
        assertTrue("Rendered progress must have accessible range information",progressNodes.isNotEmpty())
        val fraction = progressNodes.first().config[SemanticsProperties.ProgressBarRangeInfo].current
        assertTrue(fraction in 0f..1f); shot("recovery-working")
        await("editPreview",180); shot("recovery-ready")
        assertTrue(directory.listFiles()!!.any { it.length()>1000 })
        rule.onNodeWithTag("editSlow").performClick()
        rule.waitUntil(60_000) { session.state is EditSession.State.Working }
        rule.onNodeWithTag("editDone").performClick()
        rule.waitUntil(10_000) { closed && !directory.exists() }
        assertTrue(clip.exists())
        assertFalse(session.state is EditSession.State.Ready)
    }

    /** A clip the page could not analyse but still planned an edit for: no result, and the edit is offered all the same. */
    @Test fun theReplayOffersTheEditForAPlanThatCameWithoutAResult() {
        val pack = (GymPack.load(com.betabuddy.android.gym.AssetGymPackFiles(context.assets)) as? GymPackStatus.Ready)?.pack
        assumeTrue("no pack with an analysis page in this build", pack?.analysisUrl("en") != null)
        val title = "没有结果的一趟 ${System.nanoTime() % 100000}"
        val climb = store.save(title, "x", "V3", false, "", wall(), byteArrayOf(1), PoseAnalysis(emptyList(), 0, 3.0, 0.5625))
        made += climb.id
        store.saveEditPlan(climb.id, plan)
        scenario = ActivityScenario.launch(Intent(context, MainActivity::class.java).putExtra("initialTab", 1).putExtra("gymNoticeSeen", true).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        rule.waitUntil(20_000) { runCatching { rule.onNodeWithText(title).assertExists() }.isSuccess }
        rule.onNodeWithText(title).performScrollTo().performClick()
        // "Analyse this attempt" is still the way to a result; the edit sits under it.
        await("analyzeAttempt")
        await("openEdit")
        assertFalse(exists("viewAnalysis"))
        rule.onNodeWithTag("openEdit").performScrollTo().performClick()
        await("editScreen")
        rule.onNodeWithTag("editDone").performClick()
        rule.waitUntil(10_000) { !exists("editScreen") }
    }

    /** From the replay: needs a build whose pack has an analysis page, since the button lives with the analysis buttons. */
    @Test fun theReplayOffersTheEditWhenAPlanIsStored() {
        val pack = (GymPack.load(com.betabuddy.android.gym.AssetGymPackFiles(context.assets)) as? GymPackStatus.Ready)?.pack
        assumeTrue("no pack with an analysis page in this build", pack?.analysisUrl("en") != null)
        val clip = wall()
        val title = "合成的一趟 ${System.nanoTime() % 100000}"
        val climb = store.save(title, "x", "V3", false, "", clip, byteArrayOf(1), PoseAnalysis(emptyList(), 0, 3.0, 0.5625))
        made += climb.id
        store.saveAnalysisResult(climb.id, "{}")
        scenario = ActivityScenario.launch(Intent(context, MainActivity::class.java).putExtra("initialTab", 1).putExtra("gymNoticeSeen", true).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        rule.waitUntil(20_000) { runCatching { rule.onNodeWithText(title).assertExists() }.isSuccess }
        rule.onNodeWithText(title).performScrollTo().performClick()
        await("viewAnalysis")
        // A result without a plan (an older pack's): no edit button.
        assertFalse(exists("openEdit"))
        rule.onNodeWithTag("reviewBack").performClick()
        store.saveEditPlan(climb.id, plan)
        rule.onNodeWithText(title).performScrollTo().performClick()
        await("openEdit")
        rule.onNodeWithTag("openEdit").performScrollTo()
        shot("10-replay-with-edit")
        rule.onNodeWithTag("openEdit").performClick()
        await("editScreen")
        await("editPreview", 180)
        rule.waitUntil(15_000) { previewShowsTheWall() }
        shot("11-edit-from-replay")
        rule.onNodeWithTag("editDone").performClick()
        rule.waitUntil(10_000) { !exists("editScreen") }
        assertTrue(EditFiles.directory(context, climb.id).listFiles().isNullOrEmpty())
    }
}
