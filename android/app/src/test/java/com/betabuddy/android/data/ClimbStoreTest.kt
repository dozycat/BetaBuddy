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
    private val analysis = PoseAnalysis(emptyList(), 0, 1.0, 1.0, 0.0, emptyList())

    @Before fun setUp() { root = File(Files.createTempDirectory("betabuddy-store").toFile(), "BetaBuddy") }
    @After fun tearDown() { root.parentFile.deleteRecursively() }

    private fun staged(): File = File(root.parentFile, "fixture.mov").apply { writeBytes(byteArrayOf(0, 1, 2)) }

    private fun ClimbStore.saveFixture(title: String = "测试", grade: String = "V3", video: File = staged()) =
        save(title, "今天的攀爬", grade, false, "", video, ByteArray(0), analysis)

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

    // An archive exactly as the iOS app writes it, including a retired `training` array and a cue
    // saved before `kind` existed. Unknown fields are ignored, not an error.
    @Test fun iosArchiveWithRetiredFieldsStillLoads() {
        root.mkdirs()
        val legacy = """
        {"version":1,"training":[{"id":"6F9619FF-8B86-D011-B42D-00C04FC964FF","date":774000000,"drillID":"quiet-feet"}],
         "climbs":[{"id":"7F9619FF-8B86-D011-B42D-00C04FC964FF","date":774000000,"title":"旧记录","grade":"未标级","sent":true,"note":"n",
           "videoName":"v.mov","thumbnailName":"t.jpg","futureField":{"a":[1,2]},
           "analysis":{"frames":[{"time":0.1,"joints":[{"x":0.5,"y":0.25,"confidence":0.9,"z":-0.1}]}],"sampledFrames":4,"duration":2,"aspectRatio":0.5,"processingSeconds":1,
             "moments":[{"id":"8F9619FF-8B86-D011-B42D-00C04FC964FF","time":1,"title":"高抬脚 · 候选片段","detail":"d"}]}}]}
        """.trimIndent()
        File(root, "archive.json").writeText(legacy)
        val store = ClimbStore(root)
        assertFalse(store.loadFailed)
        val climb = store.climbs.first()
        assertEquals("旧记录", climb.title)
        assertEquals(UNGRADED_VALUE, climb.grade)
        assertNull(climb.analysis.moments[0].kind)
        assertEquals(0.25, climb.analysis.frames[0].joints[0].y, 0.0)
        assertEquals(1_752_307_200_000L, climb.dateMillis) // 774000000 s after 2001-01-01 = 2025-07-12T08:00:00Z
        store.update(climb, "改过", false)
        val rewritten = File(root, "archive.json").readText()
        assertFalse(rewritten.contains("training"))
        assertFalse(rewritten.contains("futureField"))
        assertFalse(rewritten.contains("\"kind\"")) // absent stays absent, as iOS writes it
        assertTrue(rewritten.contains("\"version\":1"))
        assertEquals("改过", ClimbStore(root).climbs.first().note)
    }

    @Test fun writtenArchiveUsesTheIosFieldNames() {
        val store = ClimbStore(root)
        val frame = PoseFrame(0.5, List(33) { Joint(0.5, 0.5, 1.0) })
        val moment = Moment(time = 0.5, title = "t", detail = "d", kind = Moment.FAST_MOVE)
        val climb = store.save("a", "x", "V4", true, "n", staged(), byteArrayOf(1),
            PoseAnalysis(listOf(frame), 2, 1.0, 0.5625, 3.0, listOf(moment)), nowMillis = 978_307_200_000L + 5000)
        val text = File(root, "archive.json").readText()
        for (key in listOf("version", "climbs", "id", "date", "title", "grade", "sent", "note", "videoName", "thumbnailName",
            "analysis", "frames", "time", "joints", "x", "y", "confidence", "sampledFrames", "duration", "aspectRatio",
            "processingSeconds", "moments", "detail", "kind")) {
            assertTrue(key, text.contains("\"$key\":"))
        }
        assertEquals(5.0, climb.date, 0.0)
        assertTrue(text.contains("\"date\":5.0"))
        assertEquals(climb.id.uppercase(), climb.id)
        assertEquals(36, climb.id.length)
        assertEquals(moment.id.uppercase(), moment.id)
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
