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

class SubjectResolverTest {
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
