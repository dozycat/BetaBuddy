package com.betabuddy.android.review

import android.content.Intent
import androidx.lifecycle.ViewModelProvider
import androidx.test.core.app.ActivityScenario
import androidx.test.platform.app.InstrumentationRegistry
import com.betabuddy.android.MainActivity
import com.betabuddy.android.data.*
import com.betabuddy.android.edit.*
import com.betabuddy.android.gym.AssetGymPackFiles
import com.betabuddy.android.gym.GymPack
import com.betabuddy.android.gym.GymPackStatus
import com.betabuddy.android.journal.JournalGraph
import com.betabuddy.android.journal.JournalViewModel
import com.betabuddy.android.pose.VideoAnalyzer
import java.io.File
import kotlinx.coroutines.*
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test

class SelectedSubjectEditTest {
    @Test fun selectedPositionsReachActualPageAndRenderedCamera() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val args = InstrumentationRegistry.getArguments()
        val inputName = args.getString("subjectAnalysis"); val clipName = args.getString("subjectClip")
        assumeTrue("Optional local clip and selected observations",inputName != null && clipName != null)
        val pack = (GymPack.load(AssetGymPackFiles(context.assets)) as? GymPackStatus.Ready)?.pack
        assumeTrue(pack?.analysisUrl("en") != null)
        val input = ClimbStore.json.decodeFromString<PoseAnalysis>(File(context.filesDir,inputName!!).readText())
        val positions = requireNotNull(input.subjectPositions)
        assertTrue(positions.size >= input.frames.size)
        val staged = File(context.cacheDir,"selected-edit-${System.nanoTime()}.mp4")
        File(context.filesDir,clipName!!).copyTo(staged)
        val store = JournalGraph.store(context)
        val saved = store.save("Selected edit","Selected edit",UNGRADED_VALUE,false,"",staged,byteArrayOf(),input)
        val pick = input.subject!!.picks.first { it.personID != null }
        val person = input.people!!.first { kotlin.math.abs(it.time-pick.time) < .0001 }.candidates.first { it.id == pick.personID }
        val anchors = args.getString("subjectAnchors")?.let {
            ClimbStore.json.decodeFromString<List<SubjectAnchor>>(File(context.filesDir,it).readText())
        } ?: if(input.subject?.manual == false) emptyList() else listOf(SubjectAnchor(pick.time,person))
        store.selectSubject(saved,input,anchors)
        val scope = CoroutineScope(SupervisorJob()+Dispatchers.Main.immediate)
        var session: AnalysisSession? = null
        var scenario: ActivityScenario<MainActivity>? = null
        try {
            scenario = ActivityScenario.launch(Intent(context,MainActivity::class.java).putExtra("initialTab",1).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            scenario.onActivity { activity ->
                val model = ViewModelProvider(activity)[JournalViewModel::class.java]
                session = AnalysisSession(context,pack!!,pack.analysisUrl("en")!!,saved.id,store.file(saved.videoName),model,scope,poses = { input.frames },onSeek = {})
                session!!.open(); activity.setContentView(session!!.webView)
            }
            val plan = withTimeout(180_000) {
                while (store.editPlan(saved.id) == null) delay(100)
                requireNotNull(EditPlan.parse(store.editPlan(saved.id)))
            }
            assertTrue(plan.cameraTrack); assertTrue(plan.camera.size > 2)
            assertTrue(plan.trim.second-plan.trim.first > input.duration*.9)
            if (args.getString("expectCamera") == "wide") {
                assertTrue("This close reference should retain the full scene across the occlusion", plan.camera.all {
                    kotlin.math.abs(it.s-1) < .0001 && kotlin.math.abs(it.cx-.5) < .0001 && kotlin.math.abs(it.cy-.5) < .0001
                })
            } else {
                assertTrue("A full-frame fallback alone cannot prove following", plan.camera.any { it.s > 1.05 })
            }
            for (position in positions) {
                val window = CameraPath.window(plan.camera,position.time,linear = true)
                val half = .5/window.s; val b = position.bounds
                assertTrue(window.cx-half <= b.x+.005); assertTrue(window.cy-half <= b.y+.005)
                assertTrue(window.cx+half >= b.x+b.width-.005); assertTrue(window.cy+half >= b.y+b.height-.005)
            }
            val output = File(context.getExternalFilesDir(null),"subject-edit").apply { mkdirs() }
            val movie = File(output,"selected-camera.mp4").apply { delete() }
            val options = EditOptions(removePassersBy = false)
            withContext(Dispatchers.Main) { EditRenderer(context,store.file(saved.videoName),plan) { ModelPeopleFinder(context) }.render(options,movie) {} }
            val duration = VideoAnalyzer.probe(movie).durationSeconds!!
            val expected = EditTimeline.clips(plan,options,input.duration).sumOf { (it.t1-it.t0)/it.speed }
            assertEquals(expected,duration,.3)
            File(output,"plan.json").writeText(store.editPlan(saved.id)!!)
            println("Selected Android follow: ${positions.size} positions, ${input.frames.size} real poses, ${plan.camera.size} camera keys, $duration seconds")
        } finally {
            withContext(Dispatchers.Main) { session?.close(); scope.cancel() }
            scenario?.close(); store.delete(saved)
        }
    }
}
