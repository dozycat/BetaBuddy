package com.betabuddy.android.review

import android.content.Intent
import android.graphics.Bitmap
import androidx.activity.compose.setContent
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.semantics.SemanticsActions
import androidx.lifecycle.ViewModelProvider
import androidx.test.core.app.ActivityScenario
import androidx.test.platform.app.InstrumentationRegistry
import com.betabuddy.android.MainActivity
import com.betabuddy.android.R
import com.betabuddy.android.capture.SyntheticWallClip
import com.betabuddy.android.data.*
import com.betabuddy.android.journal.JournalGraph
import com.betabuddy.android.journal.JournalViewModel
import com.betabuddy.android.pose.SubjectResolver
import com.betabuddy.android.pose.VideoAnalyzer
import com.betabuddy.android.pose.FrameGeometry
import com.betabuddy.android.ui.theme.BetaBuddyTheme
import java.io.File
import java.io.FileOutputStream
import org.junit.After
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test

class SubjectSelectionScreenTest {
    @get:Rule val rule = createEmptyComposeRule()
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext
    private val store by lazy { JournalGraph.store(context) }
    private var scenario: ActivityScenario<MainActivity>? = null
    private var record: Climb? = null
    @After fun cleanup() { scenario?.close(); record?.let(store::delete) }

    @Test fun cachedUnresolvedPeopleRecoverAutomaticallyAndSaveWithoutManualAnchor() {
        val args=InstrumentationRegistry.getArguments()
        val clip=args.getString("subjectClip");val baseline=args.getString("subjectUnresolved")
        assumeTrue("Optional unresolved real-video candidates",clip!=null && baseline!=null)
        val input=ClimbStore.json.decodeFromString<PoseAnalysis>(File(context.filesDir,baseline!!).readText())
        assertTrue(input.hasSubjectCandidates);assertFalse(input.hasSelectedSubject)
        val video=File(context.cacheDir,"auto-ui-${System.nanoTime()}.mp4")
        File(context.filesDir,clip!!).copyTo(video)
        val climb=store.save("Automatic UI","Automatic UI",UNGRADED_VALUE,false,"",video,byteArrayOf(),input)
        record=climb
        scenario=ActivityScenario.launch(Intent(context,MainActivity::class.java).putExtra("initialTab",1))
        fun open(snapshot:Climb) { scenario!!.onActivity { activity->
            val model=ViewModelProvider(activity)[JournalViewModel::class.java]
            activity.setContent { BetaBuddyTheme { SubjectSelectionScreen(snapshot,model) { activity.setContent {} } } }
        } }
        open(climb)
        rule.waitUntil(180_000) { runCatching { rule.onNodeWithTag("saveSubject").assertIsEnabled();true }.getOrDefault(false) }
        rule.onNodeWithText(context.getString(R.string.subject_automatic)).assertExists()
        rule.onNodeWithTag("subjectStage").assertIsDisplayed()
        rule.waitForIdle();instrumentation.waitForIdleSync();Thread.sleep(300)
        val out=File(context.getExternalFilesDir(null),"subject-auto-shots").apply { mkdirs() }
        instrumentation.uiAutomation.takeScreenshot()?.let { image->
            FileOutputStream(File(out,"automatic.png")).use { image.compress(Bitmap.CompressFormat.PNG,100,it) };image.recycle()
        }
        rule.onNodeWithTag("saveSubject").performClick()
        rule.waitUntil(10_000) { store.climbs.first { it.id==climb.id }.subjectRevision==1 }
        val saved=store.climbs.first { it.id==climb.id }
        assertTrue(saved.subjectAnchors.isNullOrEmpty());assertFalse(saved.analysis.subject!!.manual)
        assertTrue(saved.analysis.subjectPositions!!.size>=input.sampledFrames*.9)
        open(saved)
        rule.waitUntil(15_000) { rule.onAllNodesWithText(context.getString(R.string.subject_automatic)).fetchSemanticsNodes().isNotEmpty() }
        rule.onNodeWithText(context.getString(R.string.journal_cancel)).performClick()
        assertEquals(1,store.climbs.first { it.id==climb.id }.subjectRevision)
    }

    @Test fun drawPreviewTrackSaveAndReopenOnReference() {
        val args = InstrumentationRegistry.getArguments()
        val name = args.getString("subjectClip"); val raw = args.getString("regionBounds")
        assumeTrue("Optional real reference and observed region",name != null && raw != null && SubjectResolver(context).available())
        val bounds = raw!!.split(",").map(String::toFloat)
        val video = File(context.cacheDir,"region-ui-${System.nanoTime()}.mp4")
        File(context.filesDir,name!!).copyTo(video)
        val info = VideoAnalyzer.probe(video)
        val size = FrameGeometry.displaySize(info.width,info.height,info.rotation)
        val input = PoseAnalysis(emptyList(),0,info.durationSeconds!!,size.first.toDouble()/size.second)
        val climb = store.save("Region UI","Region UI",UNGRADED_VALUE,false,"",video,byteArrayOf(),input)
        record = climb
        scenario = ActivityScenario.launch(Intent(context,MainActivity::class.java).putExtra("initialTab",1).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        fun open(snapshot: Climb) { scenario!!.onActivity { activity ->
            val model = ViewModelProvider(activity)[JournalViewModel::class.java]
            activity.setContent { BetaBuddyTheme { SubjectSelectionScreen(snapshot,model) { activity.setContent {} } } }
        } }
        open(climb)
        rule.onNodeWithTag("drawSubjectRegion").performScrollTo().performClick()
        rule.onNodeWithTag("subjectRegionCanvas").performTouchInput {
            swipe(Offset(width*bounds[0],height*bounds[1]),Offset(width*(bounds[0]+bounds[2]),height*(bounds[1]+bounds[3])),500)
        }
        val out = File(context.getExternalFilesDir(null),"subject-region-shots").apply { mkdirs() }
        fun screenshot(name: String) {
            rule.waitForIdle(); Thread.sleep(250)
            instrumentation.uiAutomation.takeScreenshot()?.let { bitmap ->
                FileOutputStream(File(out,name)).use { bitmap.compress(Bitmap.CompressFormat.PNG,100,it) }; bitmap.recycle()
            }
        }
        screenshot("preview.png")
        rule.onNodeWithTag("saveSubject").assertIsNotEnabled()
        if (args.getString("regionAdjust") == "1") {
            rule.onNodeWithTag("adjustSubjectRegion").performScrollTo().performClick()
            for ((label,value) in listOf(R.string.subject_draw_width to bounds[2],R.string.subject_draw_height to bounds[3],
                R.string.subject_draw_horizontal to bounds[0],R.string.subject_draw_vertical to bounds[1])) {
                rule.onNodeWithContentDescription(context.getString(label)).performScrollTo().performSemanticsAction(SemanticsActions.SetProgress) { it(value) }
                rule.onNodeWithTag("subjectStage").assertIsDisplayed()
            }
            screenshot("adjusting.png")
        }
        rule.onNodeWithTag("trackSubjectRegion").performScrollTo().performClick()
        if (args.getString("regionInterrupt") == "1") {
            scenario!!.moveToState(androidx.lifecycle.Lifecycle.State.CREATED)
            scenario!!.moveToState(androidx.lifecycle.Lifecycle.State.RESUMED)
        }
        rule.waitUntil(150_000) { runCatching { rule.onNodeWithTag("saveSubject").assertIsEnabled(); true }.getOrDefault(false) }
        rule.onNodeWithTag("subjectStage").assertIsDisplayed(); screenshot("tracked.png")
        rule.onNodeWithTag("saveSubject").performClick()
        rule.waitUntil(10_000) { store.climbs.first { it.id == climb.id }.subjectRevision == 1 }
        val saved = store.climbs.first { it.id == climb.id }
        assertTrue(saved.subjectAnchors!!.single().region)
        assertTrue(saved.analysis.subjectPositions!!.size >= input.duration*9)
        assertEquals(bounds[0].toDouble(),saved.subjectAnchors!!.single().bounds.x,.025)
        open(saved)
        rule.waitUntil(30_000) { runCatching { rule.onNodeWithText(context.getString(R.string.subject_manual)).assertExists(); true }.getOrDefault(false) }
        rule.onNodeWithTag("subjectStage").assertIsDisplayed(); screenshot("reopened.png")
        rule.onNodeWithText(context.getString(R.string.journal_cancel)).performClick()
        assertEquals(1,store.climbs.first { it.id == climb.id }.subjectRevision)
    }

    @Test fun outlineTapSaveReopenAndCancelPreserveIdentity() {
        assumeTrue("Optional private policy",SubjectResolver(context).available())
        val args = InstrumentationRegistry.getArguments()
        val input = args.getString("subjectAnalysis")?.let { ClimbStore.json.decodeFromString<PoseAnalysis>(File(context.filesDir,it).readText()) }
            ?: PoseAnalysis(emptyList(),30,3.0,.6,people = (0..29).map { i -> SubjectSample(i/10.0,listOf(.2,.7).mapIndexed { person,x ->
                SubjectCandidate("$i-$person",SubjectBounds(x,.2,.1,.5),List(12) { x },PoseFrame(i/10.0,List(33) { Joint(x,.4,1.0) }))
            }) })
        val video = File(context.cacheDir,"subject-test-${System.nanoTime()}.mp4")
        args.getString("subjectClip")?.let { File(context.filesDir,it).copyTo(video) } ?: SyntheticWallClip.make(video)
        val target = input.people!!.first().candidates.firstOrNull { it.tracking != null } ?: input.people!!.first().candidates.last()
        val climb = store.save("Selection UI","Selection UI",UNGRADED_VALUE,false,"",video,byteArrayOf(),input)
        record = climb
        scenario = ActivityScenario.launch(Intent(context,MainActivity::class.java).putExtra("initialTab",1).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        fun open(snapshot: Climb) { scenario!!.onActivity { activity ->
            val model = ViewModelProvider(activity)[JournalViewModel::class.java]
            activity.setContent { BetaBuddyTheme { SubjectSelectionScreen(snapshot,model) { activity.setContent {} } } }
        } }
        open(climb)
        val personTag = "subject-${target.id}"
        rule.waitUntil(30_000) { runCatching { rule.onAllNodesWithTag(personTag).fetchSemanticsNodes().isNotEmpty() }.getOrDefault(false) }
        rule.onNodeWithTag(personTag).performClick()
        rule.waitUntil(30_000) { runCatching { rule.onNodeWithTag("saveSubject").assertIsEnabled(); true }.getOrDefault(false) }
        rule.onNodeWithTag(personTag).assertIsSelected()
        rule.onNodeWithTag("subjectStage").assertIsDisplayed()
        val out = File(context.getExternalFilesDir(null),"subject-shots").apply { mkdirs() }
        instrumentation.uiAutomation.takeScreenshot()?.let { bitmap ->
            FileOutputStream(File(out,"selected.png")).use { bitmap.compress(Bitmap.CompressFormat.PNG,100,it) }; bitmap.recycle()
        }
        rule.onNodeWithTag("saveSubject").performClick()
        rule.waitUntil(10_000) { store.climbs.first { it.id == climb.id }.subjectRevision == 1 }
        val saved = store.climbs.first { it.id == climb.id }
        assertEquals(target.id,saved.analysis.subject!!.picks.first().personID)
        assertEquals(1,saved.subjectAnchors!!.size)
        open(saved)
        rule.waitUntil(30_000) { runCatching { rule.onNodeWithText(context.getString(R.string.subject_manual)).assertExists(); true }.getOrDefault(false) }
        rule.onNodeWithTag(personTag).assertIsSelected()
        if (args.getString("subjectAnalysis") == null) {
            // Once chosen, only that outline is drawn; another person remains tappable.
            val other = input.people!!.first().candidates.first { it.id != target.id }
            rule.onNodeWithTag("subject-${other.id}").performClick()
            rule.waitUntil(10_000) { runCatching { rule.onNodeWithTag("subject-${other.id}").assertIsSelected(); true }.getOrDefault(false) }
        }
        rule.onNodeWithText(context.getString(R.string.journal_cancel)).performClick()
        assertEquals(1,store.climbs.first { it.id == climb.id }.subjectRevision)
        assertEquals(target.id,store.climbs.first { it.id == climb.id }.analysis.subject!!.picks.first().personID)
    }
}
