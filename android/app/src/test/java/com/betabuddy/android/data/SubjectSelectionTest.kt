package com.betabuddy.android.data

import org.junit.Assert.*
import org.junit.Test
import kotlinx.serialization.encodeToString

class SubjectSelectionTest {
    @Test fun missingIntervalsRemainVisibleAndNavigationWraps() {
        val picks = (0 until 60).map { i -> SubjectPick(i/10.0, if (i in 10 until 20 || i in 40 until 50) null else "person") }
        val timeline = SubjectTimeline(picks,6.0)
        assertEquals(3,timeline.selected.size); assertEquals(2,timeline.gaps.size)
        assertEquals(.95,timeline.gaps[0].start,.000001); assertEquals(1.95,timeline.gaps[0].end,.000001)
        assertEquals(1.45,timeline.nextGap(0.0)!!,.000001)
        assertEquals(4.45,timeline.nextGap(1.45)!!,.000001)
        assertEquals(1.45,timeline.nextGap(4.45)!!,.000001)
        assertNull(picks[10].personID)
    }
    @Test fun sparseObservationsDoNotBecomeContinuousCoverage() {
        val sparse = SubjectTimeline(listOf(SubjectPick(0.0,"a"),SubjectPick(5.0,"b")),6.0)
        assertEquals(2,sparse.selected.size); assertTrue(sparse.gaps[0].end-sparse.gaps[0].start>4)
        val complete = SubjectTimeline((0 until 30).map { SubjectPick(it/10.0,"a") },3.0)
        assertEquals(listOf(SubjectTimeline.Span(0.0,3.0)),complete.selected)
        assertNull(complete.nextGap(0.0))
        assertEquals(listOf(SubjectTimeline.Span(0.0,3.0)),SubjectTimeline(emptyList(),3.0).gaps)
        assertTrue(SubjectTimeline(emptyList(),Double.NaN).gaps.isEmpty())
    }
    @Test fun legacyAnchorWithoutRegionStillLoads() {
        val anchor = ClimbStore.json.decodeFromString<SubjectAnchor>("""{"time":0,"bounds":{"x":0.1,"y":0.1,"width":0.2,"height":0.6},"appearance":[]}""")
        assertFalse(anchor.region)
    }
    @Test fun selectedCandidateControlsPosesAndMissingSamplesStayMissing() {
        val a = SubjectCandidate("a", SubjectBounds(.1,.1,.2,.6), listOf(.2), PoseFrame(0.0, List(33) { Joint(.2,.3,1.0) }))
        val b = a.copy(id = "b", pose = a.pose.copy(joints = List(33) { Joint(.7,.4,1.0) }))
        val original = PoseAnalysis(emptyList(),2,1.0,.6, people = listOf(SubjectSample(0.0,listOf(a,b)),SubjectSample(.5,emptyList())))
        val resolution = SubjectResolution(1,listOf(SubjectPick(0.0,"b"),SubjectPick(.5)),true,0)
        val selected = original.selecting(resolution)
        assertEquals(listOf(b.pose),selected.frames)
        assertEquals("b",selected.subject!!.picks[0].personID)
        assertNull(original.subjectSample(.25))
        assertNotNull(original.subjectSample(.1))
        assertFalse(selected.needsSubjectRecovery)
        val requested = resolution.copy(recoveryRecommended = true)
        val restored = ClimbStore.json.decodeFromString<SubjectResolution>(ClimbStore.json.encodeToString(requested))
        val recoverable = original.selecting(restored)
        assertTrue(recoverable.hasSelectedSubject)
        assertTrue(recoverable.needsSubjectRecovery)
        assertEquals(selected.subject!!.picks, recoverable.subject!!.picks)
        assertTrue(original.needsSubjectRecovery)
        val legacy = ClimbStore.json.decodeFromString<SubjectResolution>("""{"schema":1,"picks":[],"manual":false,"unresolved":0}""")
        assertNull(legacy.recoveryRecommended)
        assertThrows(IllegalArgumentException::class.java) { original.selecting(resolution.copy(picks = listOf(SubjectPick(0.0,"unknown"),SubjectPick(.5)))) }
        assertThrows(IllegalArgumentException::class.java) { original.selecting(resolution.copy(picks = listOf(SubjectPick(.1,"b"),SubjectPick(.5)))) }
    }
}
