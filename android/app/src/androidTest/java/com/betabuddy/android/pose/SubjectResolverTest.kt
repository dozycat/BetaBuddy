package com.betabuddy.android.pose

import androidx.test.platform.app.InstrumentationRegistry
import com.betabuddy.android.data.*
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.coroutines.withTimeout

class SubjectResolverTest {
    @Test fun automaticSelectionMatchesReferencePositions() = runBlocking {
        val args=InstrumentationRegistry.getArguments()
        val name=args.getString("subjectClip");val expectedName=args.getString("subjectExpected")
        assumeTrue("Optional video and previously reviewed manual-selection positions",name!=null && expectedName!=null)
        val context=InstrumentationRegistry.getInstrumentation().targetContext
        val expected=ClimbStore.json.decodeFromString<List<SubjectPosition>>(File(context.filesDir,expectedName!!).readText())
        assertTrue(expected.isNotEmpty())
        val start=System.nanoTime()
        val result=withTimeout(300_000) { SubjectResolver(context).analyze(File(context.filesDir,name!!),AnalysisQuality.BALANCED,anchors=emptyList()) {} }
        val output=File(context.getExternalFilesDir(null),"subject-automatic").apply { mkdirs() }
        File(output,"analysis.json").writeText(ClimbStore.json.encodeToString(result))
        val positions=result.subjectPositions.orEmpty()
        val distances=expected.mapNotNull { reference->
            positions.minByOrNull { kotlin.math.abs(it.time-reference.time) }?.takeIf { kotlin.math.abs(it.time-reference.time)<=.16 }?.let { selected->
                kotlin.math.hypot(selected.bounds.x+selected.bounds.width/2-reference.bounds.x-reference.bounds.width/2,selected.bounds.y+selected.bounds.height/2-reference.bounds.y-reference.bounds.height/2)
            }
        }
        File(output,"metrics.txt").writeText("positions=${positions.size} poses=${result.frames.size} coverage=${distances.size}/${expected.size} meanDistance=${distances.average()} maxDistance=${distances.maxOrNull()} seconds=${(System.nanoTime()-start)/1e9}\n")
        assertFalse(result.subject!!.manual)
        assertTrue("Target coverage remains incomplete",distances.size>=expected.size*.9)
        assertTrue("Automatic selection is on a different person",distances.average()<.05 && distances.all { it<.15 })
        assertTrue(result.frames.isNotEmpty())
    }
    @Test fun installedPolicySelectsTheAnchoredPersonWithoutFallback() = runBlocking {
        val resolver = SubjectResolver(InstrumentationRegistry.getInstrumentation().targetContext)
        assumeTrue("Optional private policy is not installed", resolver.available())
        val samples = (0..19).map { i ->
            SubjectSample(i / 10.0, listOf(.2,.7).mapIndexed { person, x ->
                SubjectCandidate("$i-$person",SubjectBounds(x,.2,.1,.5),List(12) { x },
                    PoseFrame(i/10.0,List(33) { Joint(x,.4,1.0) }))
            })
        }
        val input = PoseAnalysis(emptyList(),samples.size,2.0,.6,people = samples)
        val chosen = resolver.resolve(input,listOf(SubjectAnchor(.5,samples[5].candidates[1])))
        assertEquals(20,chosen.frames.size)
        assertTrue(chosen.frames.all { it.joints.first().x == .7 })
        assertTrue(chosen.subject!!.manual)
        assertEquals(0,chosen.subject.unresolved)
    }
    @Test fun realCandidateDetectionAndManualSelection() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val name = InstrumentationRegistry.getArguments().getString("subjectClip")
        assumeTrue("Optional local video not supplied", name != null)
        val file = File(context.filesDir, requireNotNull(name))
        val resolver = SubjectResolver(context)
        assumeTrue(resolver.available())
        val input = VideoAnalyzer(context).analyze(file, AnalysisQuality.BALANCED, collectSubjects = true) { }
        val sample = requireNotNull(input.people).first { it.candidates.isNotEmpty() }
        val selected = resolver.resolve(input,listOf(SubjectAnchor(sample.time,sample.candidates.first())))
        assertEquals(sample.candidates.first().id,selected.subject!!.picks.first { it.time == sample.time }.personID)
        assertTrue(selected.frames.isNotEmpty())
        File(context.filesDir,"subject-reference.json").writeText(Json.encodeToString(selected))
    }
}
