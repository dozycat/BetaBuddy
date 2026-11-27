package com.betabuddy.android.data

import org.junit.Assert.*
import org.junit.Test

class SubjectSelectionTest {
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
        assertThrows(IllegalArgumentException::class.java) { original.selecting(resolution.copy(picks = listOf(SubjectPick(0.0,"unknown"),SubjectPick(.5)))) }
        assertThrows(IllegalArgumentException::class.java) { original.selecting(resolution.copy(picks = listOf(SubjectPick(.1,"b"),SubjectPick(.5)))) }
    }
}
