package com.betabuddy.android.data

import java.io.File
import java.nio.file.Files
import org.junit.After
import org.junit.Assert.*
import org.junit.Test

class SubjectStorageTest {
    private val parent = Files.createTempDirectory("subject-store").toFile()
    private val store = ClimbStore(File(parent,"store"))
    @After fun cleanup() { parent.deleteRecursively() }
    private val samples = (0..599).map { i -> SubjectSample(i/10.0,listOf(
        SubjectCandidate("$i-a",SubjectBounds(.1,.2,.2,.4),List(12) { .2 },PoseFrame(i/10.0,List(33) { Joint(.2,.3,1.0) })),
        SubjectCandidate("$i-b",SubjectBounds(.6,.2,.2,.4),List(12) { .8 },PoseFrame(i/10.0,emptyList())),
    )) }
    private val analysis get() = PoseAnalysis(emptyList(),samples.size,60.0,.6,people = samples)
    private fun save(): Climb = store.save("Test","Test",UNGRADED_VALUE,false,"",File(parent,"video.mp4").apply { writeText("video") },byteArrayOf(),analysis)
    private fun selected() = analysis.selecting(SubjectResolution(1,samples.map { SubjectPick(it.time,it.candidates[1].id) },true,0))

    @Test fun selectedPositionSurvivesWithoutPoseAndOldWorkCannotReturn() {
        val old = save()
        store.saveAnalysisResult(old.id,"old",0); store.saveEditPlan(old.id,"old-plan",0)
        val choice = selected()
        assertTrue(choice.frames.isEmpty()); assertTrue(choice.hasSelectedSubject)
        store.selectSubject(old,choice,listOf(SubjectAnchor(0.0,samples.first().candidates[1])))
        assertNull(store.analysisResult(old.id)); assertNull(store.editPlan(old.id))
        assertFalse(store.saveAnalysisResult(old.id,"stale",0)); store.saveEditPlan(old.id,"stale",0)
        store.replaceAnalysis(old,analysis)
        val reopened = ClimbStore(store.root)
        val current = reopened.climbs.single()
        assertEquals(1,current.subjectRevision); assertTrue(current.analysis.hasSelectedSubject)
        assertEquals(600,current.analysis.subjectPositions!!.size)
        assertNull(reopened.analysisResult(old.id)); assertNull(reopened.editPlan(old.id))
        assertTrue(reopened.saveAnalysisResult(old.id,"current",1))
        assertEquals("current",reopened.analysisResult(old.id))
        assertThrows(IllegalArgumentException::class.java) { store.selectSubject(old,choice,emptyList()) }
    }

    @Test fun candidatesAreLazyAndReplacementFailureKeepsPreviousFile() {
        val climb = save()
        assertNull(climb.analysis.people); assertNotNull(climb.analysis.peopleReference)
        assertTrue(store.file(ClimbStore.INDEX_NAME).length() < 3000)
        assertEquals(600,ClimbStore(store.root).subjectAnalysis(climb.id)!!.people!!.size)
        val original = SubjectCandidateStorage.file(store.root,climb.id,climb.analysis.peopleReference!!)
        val temporary = store.file(ClimbStore.INDEX_NAME+".tmp").apply { mkdir() }
        File(temporary,"block").writeText("occupied")
        assertThrows(Exception::class.java) { store.selectSubject(climb,selected(),emptyList()) }
        assertTrue(original.isFile); assertNull(store.climbs.single().subjectRevision)
        assertEquals(1,store.root.listFiles()!!.count { it.name.endsWith(".json.gz") })
        temporary.deleteRecursively()
        store.selectSubject(climb,selected(),emptyList())
        assertFalse(original.exists())
        store.file(climb.id+".s99.edit.json").writeText("interrupted old result")
        store.delete(climb)
        assertTrue(store.root.listFiles()!!.none { it.name.startsWith(climb.id+".") })
    }

    @Test fun corruptCandidateCacheCanBeRegeneratedWithoutLosingRecord() {
        val climb = save()
        SubjectCandidateStorage.file(store.root,climb.id,climb.analysis.peopleReference!!).writeText("corrupt")
        assertNull(store.subjectAnalysis(climb.id)!!.people)
        assertFalse(store.subjectAnalysis(climb.id)!!.hasSubjectCandidates)
        assertEquals(climb.id,ClimbStore(store.root).climbs.single().id)
    }
}
