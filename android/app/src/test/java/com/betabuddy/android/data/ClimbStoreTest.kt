package com.betabuddy.android.data

import java.io.File
import java.io.IOException
import java.nio.file.Files
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test

/** Ported from ios/BetaBuddyTests/ClimbStoreTests.swift, plus cases for the atomic write. */
class ClimbStoreTest {
    private lateinit var root: File
    private val analysis = PoseAnalysis(emptyList(), 0, 1.0, 1.0)

    @Before fun setUp() { root = File(Files.createTempDirectory("betabuddy-store").toFile(), "BetaBuddy") }
    @After fun tearDown() { root.parentFile?.deleteRecursively() }

    private fun staged(): File = File(root.parentFile, "fixture.mov").apply { writeBytes(byteArrayOf(0, 1, 2)) }

    private fun ClimbStore.saveFixture(title: String = "测试", grade: String = "V3", video: File = staged()) =
        save(title, "今天的攀爬", grade, false, "", video, ByteArray(0), analysis)

    @Test fun pendingAnalysisSurvivesReloadAndCompletionPreservesEdits() {
        val store = ClimbStore(root)
        val climb = store.save("Queue", "Default", "V1", false, "", staged(), ByteArray(0), analysis,
            climbedAtMillis = 1_000_000, pendingAnalysis = "detailed")
        val reopened = ClimbStore(root)
        assertEquals("detailed", reopened.climbs.single().pendingAnalysis)
        assertEquals(0, reopened.climbs.single().analysis.sampledFrames)
        reopened.update(climb, "Edited during analysis", true)
        reopened.organize(climb, "V4", null, null, 2_000_000, null)
        reopened.replaceAnalysis(climb, analysis.copy(sampledFrames = 42))
        val saved = ClimbStore(root).climbs.single()
        assertNull(saved.pendingAnalysis)
        assertEquals("Edited during analysis", saved.note)
        assertTrue(saved.sent)
        assertEquals("V4", saved.grade)
        assertEquals(2_000_000L, saved.climbedAtMillis)
        assertEquals(42, saved.analysis.sampledFrames)
        reopened.requestAnalysis(climb.id, "balanced")
        reopened.requestAnalysis(climb.id, null)
        assertNull(ClimbStore(root).climbs.single().pendingAnalysis)
        assertTrue(reopened.file(climb.videoName).exists())
        reopened.delete(climb)
        reopened.replaceAnalysis(climb, analysis)
        assertTrue(reopened.climbs.isEmpty())
    }

    @Test fun saveReloadEditAndDelete() {
        val store = ClimbStore(root)
        val climb = store.saveFixture()
        assertEquals(1, ClimbStore(root).climbs.size)
        store.update(climb, "下一次看脚", true)
        val reopened = ClimbStore(root)
        assertEquals("下一次看脚", reopened.climbs.first().note)
        assertTrue(reopened.climbs.first().sent)
        reopened.replaceAnalysis(climb, analysis.copy(sampledFrames = 42))
        val refreshed = ClimbStore(root).climbs.first()
        assertEquals(42, refreshed.analysis.sampledFrames)
        assertEquals("下一次看脚", refreshed.note)
        assertTrue(refreshed.sent)
        assertEquals(climb.id, refreshed.id)
        assertArrayEquals(byteArrayOf(0, 1, 2), store.file(climb.videoName).readBytes())
        assertTrue(climb.videoName.endsWith(".mov"))
        reopened.delete(climb)
        assertTrue(ClimbStore(root).climbs.isEmpty())
        assertFalse(store.file(climb.videoName).exists())
        assertFalse(store.file(climb.thumbnailName).exists())
    }

    @Test fun foldersSurviveReloadAndDeletingFolderKeepsAllMedia() {
        val store = ClimbStore(root)
        val climb = store.saveFixture()
        store.saveFolder(" Projects ")
        val folder = store.archive.value.folders.single()
        assertEquals("Projects", folder.name)
        store.organize(climb, "V5", "synthetic-gym", "Synthetic gym", 1_000_000, folder.id)
        store.update(climb, "Keep this note", true) // stale snapshot must not reset metadata
        store.saveFolder("Weekend", folder.id)
        val reopened = ClimbStore(root)
        assertEquals("Weekend", reopened.archive.value.folders.single().name)
        val edited = reopened.climbs.single()
        assertEquals("V5", edited.grade)
        assertEquals(climb.date, edited.date, 0.0)
        assertEquals(1_000_000L, edited.climbedAtMillis)
        assertEquals(folder.id, edited.folderId)
        assertEquals("Keep this note", edited.note)
        reopened.deleteFolder(folder.id)
        val unfiled = ClimbStore(root).climbs.single()
        assertNull(unfiled.folderId)
        assertEquals("synthetic-gym", unfiled.gymId)
        assertArrayEquals(byteArrayOf(0, 1, 2), store.file(climb.videoName).readBytes())
        assertTrue(store.file(climb.thumbnailName).isFile)
    }

    @Test fun invalidFolderAndFailedWritesCannotPartiallyModifyMetadata() {
        val store = ClimbStore(root)
        val climb = store.saveFixture()
        store.saveFolder("Projects")
        try { store.saveFolder(" projects "); fail() } catch (expected: IllegalArgumentException) { }
        try { store.saveFolder("   "); fail() } catch (expected: IllegalArgumentException) { }
        try { store.organize(climb, "V5", null, "", 0, "missing"); fail() } catch (expected: IllegalArgumentException) { }
        val folder = store.archive.value.folders.single()
        store.organize(climb, "V5", null, "", 0, folder.id)
        val before = File(root, "archive.json").readBytes()
        File(root, "archive.json.tmp").mkdir()
        File(root, "archive.json.tmp/blocker").writeText("x")
        try { store.deleteFolder(folder.id); fail() } catch (expected: IOException) { }
        assertEquals(folder.id, store.climbs.single().folderId)
        assertEquals(1, store.archive.value.folders.size)
        assertArrayEquals(before, File(root, "archive.json").readBytes())
    }

    @Test fun corruptArchiveIsNeverOverwritten() {
        root.mkdirs()
        val file = File(root, "archive.json")
        file.writeText("broken")
        val store = ClimbStore(root)
        assertTrue(store.loadFailed)
        val video = staged()
        try {
            store.saveFixture(title = "", grade = UNGRADED_VALUE, video = video)
            fail("A store that could not read its archive must not write")
        } catch (expected: IOException) { }
        assertEquals("broken", file.readText())
        assertTrue(video.exists())
        assertEquals(listOf("archive.json"), root.list()!!.toList())
    }

    @Test fun newerArchiveVersionIsKeptAndNotWritten() {
        root.mkdirs()
        val file = File(root, "archive.json")
        file.writeText("""{"version":2,"climbs":[]}""")
        val store = ClimbStore(root)
        assertTrue(store.loadFailed)
        try { store.saveFixture(); fail() } catch (expected: IOException) { }
        assertEquals("""{"version":2,"climbs":[]}""", file.readText())
    }

    // An archive as earlier builds of either app wrote it: a retired `training` array, and an analysis
    // that still carries its processing time and candidate cues and has no world landmarks.
    // Unknown and retired fields are ignored, not an error, and nothing of the record is lost.
    @Test fun archiveFromEarlierBuildsStillLoads() {
        root.mkdirs()
        val legacy = """
        {"version":1,"training":[{"id":"6F9619FF-8B86-D011-B42D-00C04FC964FF","date":774000000,"drillID":"quiet-feet"}],
         "climbs":[{"id":"7F9619FF-8B86-D011-B42D-00C04FC964FF","date":774000000,"title":"旧记录","grade":"未标级","sent":true,"note":"n",
           "videoName":"v.mov","thumbnailName":"t.jpg","futureField":{"a":[1,2]},
           "analysis":{"frames":[{"time":0.1,"joints":[{"x":0.5,"y":0.25,"confidence":0.9,"z":-0.1}]}],"sampledFrames":4,"duration":2,"aspectRatio":0.5,"processingSeconds":1,
             "moments":[{"id":"8F9619FF-8B86-D011-B42D-00C04FC964FF","time":1,"title":"高抬脚 · 候选片段","detail":"d"},
                        {"id":"9F9619FF-8B86-D011-B42D-00C04FC964FF","time":2,"title":"t","detail":"d","kind":"fastMove"}]}}]}
        """.trimIndent()
        File(root, "archive.json").writeText(legacy)
        val store = ClimbStore(root)
        assertFalse(store.loadFailed)
        val climb = store.climbs.first()
        assertEquals("旧记录", climb.title)
        assertNull(climb.folderId)
        assertNull(climb.gymId)
        assertTrue(store.archive.value.folders.isEmpty())
        assertEquals(UNGRADED_VALUE, climb.grade)
        assertEquals(4, climb.analysis.sampledFrames)
        assertEquals(2.0, climb.analysis.duration, 0.0)
        assertEquals(0.25, climb.analysis.frames[0].joints[0].y, 0.0)
        // No world landmarks in old records: the record still opens, the analysis page needs pose recognition again.
        assertNull(climb.analysis.frames[0].world)
        assertFalse(climb.analysis.hasWorldLandmarks)
        // No climbing date of its own: the record date stands in. 774000000 s after 2001-01-01 = 2025-07-12T08:00:00Z
        assertEquals(1_752_307_200_000L, climb.climbedAtMillis)
        store.update(climb, "改过", false)
        val rewritten = File(root, "archive.json").readText()
        for (retired in listOf("training", "futureField", "moments", "processingSeconds", "\"world\"")) assertFalse(retired, rewritten.contains(retired))
        assertTrue(rewritten.contains("\"version\":1"))
        val reread = ClimbStore(root).climbs.first()
        assertEquals("改过", reread.note)
        assertEquals(climb.analysis, reread.analysis)
    }

    @Test fun writtenArchiveUsesTheIosFieldNames() {
        val store = ClimbStore(root)
        val frame = PoseFrame(0.5, List(33) { Joint(0.5, 0.5, 1.0) }, List(33) { listOf(0.1234, -0.5, 0.0, 0.9876) })
        val climb = store.save("a", "x", "V4", true, "n", staged(), byteArrayOf(1),
            PoseAnalysis(listOf(frame), 2, 1.0, 0.5625), nowMillis = 978_307_200_000L + 5000)
        val text = File(root, "archive.json").readText()
        for (key in listOf("version", "climbs", "id", "date", "title", "grade", "sent", "note", "videoName", "thumbnailName",
            "analysis", "frames", "time", "joints", "x", "y", "confidence", "world", "sampledFrames", "duration", "aspectRatio")) {
            assertTrue(key, text.contains("\"$key\":"))
        }
        assertEquals(5.0, climb.date, 0.0)
        assertTrue(text.contains("\"date\":5.0"))
        assertEquals(climb.id.uppercase(), climb.id)
        assertEquals(36, climb.id.length)
    }

    @Test fun worldLandmarksSurviveReloadAsPlainArrays() {
        val store = ClimbStore(root)
        val world = List(33) { index -> listOf(rounded4(index / 7.0), rounded4(-index / 13.0), 0.0, 1.0) }
        val frame = PoseFrame(0.5, List(33) { Joint(0.5, 0.5, 1.0) }, world)
        val bare = PoseFrame(0.6, List(33) { Joint(0.5, 0.5, 1.0) })
        store.save("a", "x", "V4", true, "n", staged(), byteArrayOf(1), PoseAnalysis(listOf(frame, bare), 2, 1.0, 0.5625))
        val text = File(root, "archive.json").readText()
        assertTrue(text, text.contains("\"world\":[[0.0,0.0,0.0,1.0],[0.1429,-0.0769,0.0,1.0],"))
        assertEquals(1, Regex("\"world\"").findAll(text).count()) // a sample without them writes no field at all
        val reread = ClimbStore(root).climbs.single().analysis
        assertEquals(world, reread.frames[0].world)
        assertTrue(reread.frames[0].hasWorldLandmarks)
        assertFalse(reread.frames[1].hasWorldLandmarks)
        assertTrue(reread.hasWorldLandmarks)
        assertFalse(PoseAnalysis(listOf(bare), 1, 1.0, 1.0).hasWorldLandmarks)
        assertFalse(bare.copy(world = world.take(32)).hasWorldLandmarks)
        assertFalse(bare.copy(world = List(33) { listOf(0.0, 0.0, 0.0) }).hasWorldLandmarks)
    }

    @Test fun storedLandmarksKeepFourDecimals() {
        assertEquals(0.1235, rounded4(0.123456789), 0.0)
        assertEquals(-0.1235, rounded4(-0.12345678), 0.0)
        assertEquals(1.0, rounded4(0.99996), 0.0)
        assertEquals(0.0, rounded4(0.00004), 0.0)
    }

    // The result of the pack's analysis page is opaque text here; the fixture is synthetic.
    @Test fun analysisResultIsKeptAsReceivedReplacedAndDeletedWithTheRecord() {
        val store = ClimbStore(root)
        val climb = store.saveFixture()
        val other = store.saveFixture(title = "other")
        assertNull(store.analysisResult(climb.id))
        assertTrue(store.analysisResultIds().isEmpty())
        val first = """{"v":1,"note":"引号 \" 反斜杠 \\ 换行 \n","plate":"data:image/jpeg;base64,AAAA"}"""
        assertTrue(store.saveAnalysisResult(climb.id, first))
        assertEquals(first, store.analysisResult(climb.id))
        assertEquals(first, File(root, climb.id + ".analysis.json").readText())
        assertFalse(File(root, climb.id + ".analysis.json.tmp").exists())
        assertTrue(store.saveAnalysisResult(climb.id, """{"v":1,"again":true}"""))
        assertEquals("""{"v":1,"again":true}""", ClimbStore(root).analysisResult(climb.id))
        assertEquals(setOf(climb.id), ClimbStore(root).analysisResultIds())
        // Pose recognition again, notes and filing do not touch the saved result.
        store.replaceAnalysis(climb, analysis.copy(sampledFrames = 3))
        store.update(climb, "note", true)
        assertEquals("""{"v":1,"again":true}""", store.analysisResult(climb.id))
        // "Analyse again" drops it; the record and its clip stay.
        store.discardAnalysisResult(climb.id)
        assertNull(store.analysisResult(climb.id))
        assertTrue(store.file(climb.videoName).exists())
        assertEquals(2, store.climbs.size)
        // Deleting the record removes the result with the clip.
        assertTrue(store.saveAnalysisResult(climb.id, first))
        assertTrue(store.saveAnalysisResult(other.id, first))
        store.delete(climb)
        assertFalse(File(root, climb.id + ".analysis.json").exists())
        assertEquals(setOf(other.id), store.analysisResultIds())
        assertEquals(first, store.analysisResult(other.id))
    }

    @Test fun analysisResultForAMissingRecordIsNotWritten() {
        val store = ClimbStore(root)
        val climb = store.saveFixture()
        store.delete(climb)
        assertFalse(store.saveAnalysisResult(climb.id, "{}")) // deleted while the page was open
        assertFalse(store.saveAnalysisResult("../outside", "{}"))
        assertEquals(listOf("archive.json"), root.list()!!.toList())
        assertFalse(File(root.parentFile, "outside.analysis.json").exists())
    }

    @Test fun analysisResultWriteIsAtomic() {
        val store = ClimbStore(root)
        val climb = store.saveFixture()
        assertTrue(store.saveAnalysisResult(climb.id, "first"))
        // If the temporary file cannot be created, the previous result survives byte for byte.
        val blocker = File(root, climb.id + ".analysis.json.tmp")
        blocker.mkdir()
        File(blocker, "blocker").writeText("x")
        try { store.saveAnalysisResult(climb.id, "second"); fail() } catch (expected: IOException) { }
        assertEquals("first", store.analysisResult(climb.id))
        blocker.deleteRecursively()
        // A write that never reached the rename is cleaned up on the next start and is not a result.
        blocker.writeText("half a res")
        val reopened = ClimbStore(root)
        assertFalse(blocker.exists())
        assertEquals("first", reopened.analysisResult(climb.id))
        assertEquals(setOf(climb.id), reopened.analysisResultIds())
    }

    @Test fun writeIsAtomicAndLeavesNoTemporaryFile() {
        val store = ClimbStore(root)
        val climb = store.saveFixture()
        assertFalse(File(root, "archive.json.tmp").exists())
        // If the temporary file cannot be created, the previous index survives byte for byte.
        val before = File(root, "archive.json").readBytes()
        File(root, "archive.json.tmp").mkdir()
        File(root, "archive.json.tmp/blocker").writeText("x")
        try { store.update(climb, "不会写入", true); fail() } catch (expected: IOException) { }
        assertArrayEquals(before, File(root, "archive.json").readBytes())
        assertEquals("", store.climbs.first().note) // memory matches disk
        assertEquals("", ClimbStore(root).climbs.first().note)
    }

    @Test fun interruptedWriteIsCleanedUpOnNextStart() {
        val store = ClimbStore(root)
        store.saveFixture()
        File(root, "archive.json.tmp").writeText("{\"version\":1,\"cli") // a write that never reached the rename
        val reopened = ClimbStore(root)
        assertFalse(reopened.loadFailed)
        assertEquals(1, reopened.climbs.size)
        assertFalse(File(root, "archive.json.tmp").exists())
    }

    @Test fun failedSavePutsTheStagedVideoBack() {
        val store = ClimbStore(root)
        File(root, "archive.json.tmp").mkdir()
        File(root, "archive.json.tmp/blocker").writeText("x")
        val video = staged()
        try { store.saveFixture(video = video); fail() } catch (expected: IOException) { }
        assertTrue(video.exists())
        assertEquals(listOf("archive.json.tmp"), root.list()!!.toList())
        assertTrue(store.climbs.isEmpty())
    }

    @Test fun titlesNotesAndOrdering() {
        val store = ClimbStore(root)
        val unnamed = store.save("   ", "今天的攀爬", UNGRADED_VALUE, false, "x".repeat(1200), staged(), ByteArray(0), analysis, nowMillis = 1_000_000)
        assertEquals("今天的攀爬", unnamed.title)
        assertEquals(1000, unnamed.note.length)
        val long = store.save("🧗".repeat(70), "d", "V1", false, "", staged(), ByteArray(0), analysis, nowMillis = 2_000_000)
        assertEquals(60, long.title.codePointCount(0, long.title.length)) // no emoji cut in half
        assertEquals(listOf(long.id, unnamed.id), store.climbs.map { it.id }) // newest first
        val noExtension = File(root.parentFile, "clip").apply { writeBytes(byteArrayOf(1)) }
        assertTrue(store.save("a", "d", "V1", false, "", noExtension, ByteArray(0), analysis).videoName.endsWith(".mp4"))
    }

    @Test fun deleteNeverTouchesFilesOutsideTheStore() {
        val store = ClimbStore(root)
        val outside = File(root.parentFile, "keep.txt").apply { writeText("keep") }
        val climb = store.saveFixture()
        val tampered = climb.copy(videoName = "../keep.txt")
        store.delete(tampered)
        assertTrue(outside.exists())
        assertTrue(store.climbs.isEmpty())
    }

    @Test fun updatingAMissingRecordChangesNothing() {
        val store = ClimbStore(root)
        val climb = store.saveFixture()
        store.delete(climb)
        val before = File(root, "archive.json").readText()
        store.update(climb, "x", true)
        store.replaceAnalysis(climb, analysis.copy(sampledFrames = 9))
        assertEquals(before, File(root, "archive.json").readText())
    }
}
