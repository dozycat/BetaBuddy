package com.betabuddy.android.pose

import androidx.test.platform.app.InstrumentationRegistry
import com.betabuddy.android.data.*
import java.io.File
import kotlinx.coroutines.*
import kotlinx.serialization.encodeToString
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test

class SubjectRegionTest {
    @Test fun obscuredTargetStaysMissingUntilAnExplicitCorrection() = runBlocking {
        val args = InstrumentationRegistry.getArguments()
        val name = args.getString("subjectGapClip")
        val firstRaw = args.getString("regionBounds"); val nextRaw = args.getString("correctionBounds")
        assumeTrue("Optional reference copy with a 3–5 second black interval",name != null && firstRaw != null && nextRaw != null)
        fun bounds(raw: String): SubjectBounds { val v=raw.split(",").map(String::toDouble); return SubjectBounds(v[0],v[1],v[2],v[3]) }
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val clip = File(context.filesDir,name!!); val info = VideoAnalyzer.probe(clip)
        val size = FrameGeometry.displaySize(info.width,info.height,info.rotation)
        val input = PoseAnalysis(emptyList(),0,info.durationSeconds!!,size.first.toDouble()/size.second)
        val first = SubjectRegionAnalyzer(context).refine(clip,input,bounds(firstRaw!!),0.0) {}
        val selected = SubjectResolver(context).resolve(first.analysis,listOf(first.anchor))
        assertTrue(selected.subjectPositions!!.size >= 20)
        assertTrue("Tracker must stop on the obscured target",selected.subjectPositions!!.all { it.time < 3.1 })
        val next = SubjectRegionAnalyzer(context).refine(clip,first.analysis,bounds(nextRaw!!),6.0) {}
        val corrected = SubjectResolver(context).resolve(next.analysis,listOf(first.anchor,next.anchor))
        assertEquals(0,corrected.subject!!.unresolved)
        assertTrue(corrected.subjectPositions!!.any { it.time < 2.5 })
        assertTrue(corrected.subjectPositions!!.any { it.time > 19 })
        assertTrue("No positions may be invented through the black interval",corrected.subjectPositions!!.none { it.time in 3.1..4.9 })
        assertTrue(corrected.frames.none { it.time in 3.1..4.9 })
        val out = File(context.getExternalFilesDir(null),"subject-gap").apply { mkdirs() }
        File(out,"analysis.json").writeText(ClimbStore.json.encodeToString(corrected))
        File(out,"metrics.txt").writeText("before=${selected.subjectPositions!!.size} corrected=${corrected.subjectPositions!!.size} poses=${corrected.frames.size}\n")
    }

    @Test fun drawnRegionTracksWithoutFullFramePoseCandidates() = runBlocking {
        val args = InstrumentationRegistry.getArguments()
        val name = args.getString("subjectClip"); val raw = args.getString("regionBounds")
        assumeTrue("Optional real reference and manually observed region",name != null && raw != null)
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val clip = File(context.filesDir,name!!)
        val values = raw!!.split(",").map(String::toDouble)
        val bounds = SubjectBounds(values[0],values[1],values[2],values[3])
        val info = VideoAnalyzer.probe(clip);val size = FrameGeometry.displaySize(info.width,info.height,info.rotation)
        val input = PoseAnalysis(emptyList(),0,info.durationSeconds!!,size.first.toDouble()/size.second)
        val start = System.nanoTime()
        val output = File(context.getExternalFilesDir(null),"subject-region").apply { mkdirs() }
        File(output,"rejected.json").delete()
        val analyzer = SubjectRegionAnalyzer(context).apply { observationObserver = { previous,current,seed,accepted ->
            if (!accepted) File(output,"rejected.json").writeText(ClimbStore.json.encodeToString(listOf(previous,current,seed)))
        } }
        val refined = withTimeout(180_000) { analyzer.refine(clip,input,bounds,args.getString("regionTime")?.toDouble() ?: 0.0) {} }
        val selected = SubjectResolver(context).resolve(refined.analysis,listOf(refined.anchor))
        File(output,"analysis.json").writeText(ClimbStore.json.encodeToString(selected))
        File(output,"metrics.txt").writeText("samples=${selected.people!!.size} positions=${selected.subjectPositions!!.size} poses=${selected.frames.size} seconds=${(System.nanoTime()-start)/1e9}\n")
        assertTrue(selected.subjectPositions!!.size >= selected.people!!.size*.9)
        assertTrue(selected.frames.isNotEmpty())
        assertTrue(selected.frames.all { it.hasReliableTorso })
        assertTrue(selected.subjectPositions!!.all { it.bounds.valid })
        assertEquals(0,selected.subject!!.unresolved)
        assertTrue(refined.anchor.region)
        if (args.getString("reanalyzeRegion") == "1") {
            // Same fresh-video path as the background queue, after encoding the saved choice.
            // No candidate observations are passed into this call.
            val anchor = ClimbStore.json.decodeFromString<SubjectAnchor>(ClimbStore.json.encodeToString(refined.anchor))
            val fresh = withTimeout(180_000) { SubjectResolver(context).analyze(clip,AnalysisQuality.BALANCED,anchors=listOf(anchor)) {} }
            File(output,"reanalyzed.json").writeText(ClimbStore.json.encodeToString(fresh))
            assertEquals(0,fresh.subject!!.unresolved)
            assertTrue(fresh.subjectPositions!!.size >= selected.people!!.size*.9)
            val distance = fresh.subjectPositions!!.map { position ->
                val old = selected.subjectPositions!!.minBy { kotlin.math.abs(it.time-position.time) }.bounds
                kotlin.math.hypot(position.bounds.x-old.x,position.bounds.y-old.y)
            }.average()
            assertTrue("Fresh analysis switched away from the saved region: $distance",distance < .05)
        }
    }
}
