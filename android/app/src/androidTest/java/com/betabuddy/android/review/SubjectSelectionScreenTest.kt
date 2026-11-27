package com.betabuddy.android.review

import android.content.Intent
import android.graphics.Bitmap
import androidx.activity.compose.setContent
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createEmptyComposeRule
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
        rule.onNodeWithTag(personTag).performScrollTo().performClick()
        rule.waitUntil(30_000) { runCatching { rule.onNodeWithTag("saveSubject").assertIsEnabled(); true }.getOrDefault(false) }
        rule.onNodeWithTag(personTag).assertIsSelected()
        rule.onNodeWithTag("subjectStage").performScrollTo()
        val out = File(context.getExternalFilesDir(null),"subject-shots").apply { mkdirs() }
        instrumentation.uiAutomation.takeScreenshot()?.let { bitmap ->
            FileOutputStream(File(out,"selected.png")).use { bitmap.compress(Bitmap.CompressFormat.PNG,100,it) }; bitmap.recycle()
        }
        rule.onNodeWithTag("saveSubject").performScrollTo().performClick()
        rule.waitUntil(10_000) { store.climbs.first { it.id == climb.id }.subjectRevision == 1 }
        val saved = store.climbs.first { it.id == climb.id }
        assertEquals(target.id,saved.analysis.subject!!.picks.first().personID)
        assertEquals(1,saved.subjectAnchors!!.size)
        open(saved)
        rule.waitUntil(30_000) { runCatching { rule.onNodeWithText(context.getString(R.string.subject_manual)).assertExists(); true }.getOrDefault(false) }
        rule.onNodeWithTag(personTag).assertIsSelected()
        rule.onNodeWithText(context.getString(R.string.journal_cancel)).performScrollTo().performClick()
        assertEquals(1,store.climbs.first { it.id == climb.id }.subjectRevision)
        assertEquals(target.id,store.climbs.first { it.id == climb.id }.analysis.subject!!.picks.first().personID)
    }
}
