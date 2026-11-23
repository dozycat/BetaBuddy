package com.betabuddy.android.journal

import com.betabuddy.android.data.CaptureInfo
import com.betabuddy.android.data.CaptureRoute
import com.betabuddy.android.data.CaptureSummary
import com.betabuddy.android.data.Climb
import com.betabuddy.android.data.ClimbStore
import com.betabuddy.android.data.Joint
import com.betabuddy.android.data.PoseAnalysis
import com.betabuddy.android.data.PoseFrame
import com.betabuddy.android.gym.GymManifest
import com.betabuddy.android.review.AnalysisScript
import java.io.File
import java.nio.file.Files
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Records made with the in-app camera: saved in one write, analysed afterwards, in that order. */
class CaptureRecordsTest {
    private val root: File = Files.createTempDirectory("capture-records").toFile()
    private val store = ClimbStore(root)
    private val empty = PoseAnalysis(emptyList(), 0, 4.0, 9.0 / 16)
    private val plan = """{"v":1,"standIn":true,"note":"é\n\"quoted\""}"""

    @After fun cleanUp() { root.deleteRecursively() }

    private fun clip(name: String = "clip.mp4") = File(root, "staging").apply { mkdirs() }.resolve(name).apply { writeBytes(ByteArray(64) { 1 }) }
    private fun capture(queued: Boolean? = true) = CaptureInfo(
        colorKey = "yellow", colorRGB = listOf(226, 196, 40), summary = CaptureSummary(color = "yellow", holds = 6, startHands = 2, finish = true, route = CaptureRoute("r1", 0.9, name = "Synthetic 1")),
        packVersion = "0.0.0-standin", analysisQueued = queued, routeId = "r1", routeName = "Synthetic 1",
    )
    private fun save(plan: String? = this.plan, capture: CaptureInfo? = capture(), pending: String? = "balanced", gym: String? = null, at: Long? = null) =
        store.save("", "Today", "V3", false, "", clip(), byteArrayOf(1, 2), empty, gymId = gym?.let { "id:$it" }, gymName = gym, climbedAtMillis = at, pendingAnalysis = pending, capture = capture, capturePlan = plan)
    private fun posed(world: Boolean) = PoseAnalysis(
        listOf(PoseFrame(0.0, List(33) { Joint(0.5, 0.5, 0.9) }, if (world) List(33) { listOf(0.0, 0.0, 0.0, 0.9) } else null)), 1, 4.0, 9.0 / 16,
    )

    @Test fun theClipThePlanAndBothRequestsGoInWithOneWrite() {
        val climb = save()
        // Everything the analysis needs is on disk the moment the record exists…
        val reread = ClimbStore(root).climbs.single()
        assertEquals(climb.id, reread.id)
        assertEquals("balanced", reread.pendingAnalysis)
        assertEquals(true, reread.capture?.analysisQueued)
        assertEquals(true, reread.capture?.hasPlan)
        assertEquals("yellow", reread.capture?.colorKey); assertEquals(listOf(226, 196, 40), reread.capture?.colorRGB)
        assertEquals("r1", reread.capture?.routeId); assertEquals(6, reread.capture?.summary?.holds)
        assertTrue(store.file(climb.videoName).isFile)
        assertEquals(plan, store.capturePlan(climb.id))
        assertEquals(plan, ClimbStore(root).capturePlan(climb.id))
        // …and no analysis result exists yet: nothing was analysed before the save.
        assertTrue(store.analysisResultIds().isEmpty())
    }

    @Test fun withoutAPlanTheRecordSaysSo() {
        val climb = save(plan = null)
        assertEquals(false, climb.capture?.hasPlan)
        assertNull(store.capturePlan(climb.id))
        assertFalse(File(root, climb.id + ClimbStore.CAPTURE_PLAN_SUFFIX).exists())
        // A plan that arrives after the clip was saved is still kept, with the fuller summary.
        store.saveCapturePlan(climb.id, plan, CaptureSummary(holds = 9))
        assertEquals(plan, store.capturePlan(climb.id))
        assertEquals(9, store.climbs.single().capture?.summary?.holds)
        store.saveCapturePlan(climb.id, "", null)            // an empty one changes nothing
        assertEquals(plan, store.capturePlan(climb.id))
    }

    @Test fun anImportedRecordHasNoCaptureAndTakesNoPlan() {
        val climb = save(plan = plan, capture = null, pending = null)
        assertNull(climb.capture)
        assertNull(store.capturePlan(climb.id))
        store.saveCapturePlan(climb.id, plan, null)
        store.setAnalysisQueued(climb.id, true)
        assertNull(store.climbs.single().capture)
        assertNull(store.capturePlan(climb.id))
    }

    @Test fun aFailedSaveLeavesNothingBehind() {
        val staged = clip()
        try {
            store.save("", "Today", "V3", false, "", staged, byteArrayOf(1), empty, folderId = "no-such-folder", capture = capture(), capturePlan = plan)
        } catch (expected: IllegalArgumentException) { }
        assertTrue(store.climbs.isEmpty())
        assertTrue(staged.isFile)   // the clip is still held, to be saved again
        assertTrue(root.listFiles()!!.none { it.name.endsWith(ClimbStore.CAPTURE_PLAN_SUFFIX) })
    }

    @Test fun deletingARecordRemovesItsPlan() {
        val climb = save()
        val file = File(root, climb.id + ClimbStore.CAPTURE_PLAN_SUFFIX)
        assertTrue(file.isFile)
        store.delete(climb)
        assertFalse(file.exists())
    }

    @Test fun theQueuedRequestIsClearedOnce() {
        val climb = save()
        store.setAnalysisQueued(climb.id, false)
        assertNull(store.climbs.single().capture?.analysisQueued)
        assertNull(ClimbStore(root).climbs.single().capture?.analysisQueued)
        store.setAnalysisQueued(climb.id, true)
        assertEquals(true, store.climbs.single().capture?.analysisQueued)
    }

    @Test fun theLastGymIsTheDefaultForTheNextRecording() {
        assertNull(store.lastGym)
        save(gym = "合成岩馆（东门）", at = 1_000_000L)
        save(gym = null, at = 3_000_000L)                 // the most recent record has no gym: the one before it counts
        save(gym = "合成岩馆（西门）", at = 2_000_000L)
        assertEquals("id:合成岩馆（西门）" to "合成岩馆（西门）", store.lastGym)
    }

    @Test fun anArchiveWrittenBeforeTheCameraExistedStillReads() {
        File(root, ClimbStore.INDEX_NAME).writeText(
            """{"version":1,"climbs":[{"id":"A","date":1.0,"title":"t","grade":"V1","sent":false,"note":"","videoName":"A.mp4","thumbnailName":"A.jpg","analysis":{"frames":[],"sampledFrames":0,"duration":1.0,"aspectRatio":0.5}},
               {"id":"B","date":2.0,"title":"t","grade":"V1","sent":false,"note":"","videoName":"B.mov","thumbnailName":"B.jpg","analysis":{"frames":[],"sampledFrames":0,"duration":1.0,"aspectRatio":0.5},
                "capture":{"colorKey":"blue","colorRGB":[40,110,220],"hasPlan":false,"analysisQueued":true,"summary":{"holds":5,"route":{"id":"r","confidence":0.8,"name":"n","grade":"V2"},"later":1},"addedLater":true}}]}""",
        )
        val read = ClimbStore(root)
        assertFalse(read.loadFailed)
        assertNull(read.climbs.first { it.id == "A" }.capture)
        val camera = read.climbs.first { it.id == "B" }.capture!!   // as the iOS app writes it
        assertEquals("blue", camera.colorKey); assertEquals(true, camera.analysisQueued); assertEquals("V2", camera.summary?.route?.grade)
    }

    // The order of things afterwards

    private fun next(climbs: List<Climb>, results: Set<String> = emptySet(), settled: Set<String> = emptySet(), paused: Boolean = false, hasPage: Boolean = true) =
        AnalysisQueuePolicy.next(climbs, results, settled, paused, hasPage)

    @Test fun thePagesAnalysisWaitsForPoseRecognition() {
        val climb = save()
        // Saved, pose recognition still asked for: the page is not run.
        assertEquals(AnalysisQueuePolicy.Step.Wait, next(store.climbs))
        // Pose recognition done, with what the page can use: now it runs.
        store.replaceAnalysis(climb, posed(world = true))
        val ready = store.climbs.single()
        assertNull(ready.pendingAnalysis)
        assertEquals(AnalysisQueuePolicy.Step.Run(ready), next(store.climbs))
        // Never while the camera is open or the app is in the background.
        assertEquals(AnalysisQueuePolicy.Step.Wait, next(store.climbs, paused = true))
        // Done: the request is cleared, and nothing is left to do.
        assertEquals(AnalysisQueuePolicy.Step.Settle(climb.id), next(store.climbs, results = setOf(climb.id)))
        store.setAnalysisQueued(climb.id, false)
        assertEquals(AnalysisQueuePolicy.Step.Wait, next(store.climbs))
    }

    @Test fun requestsThatCannotBeMetAreSettledNotRetried() {
        val nobody = save()
        store.requestAnalysis(nobody.id, null)   // pose recognition found nobody, or was stopped
        assertEquals(AnalysisQueuePolicy.Step.Settle(nobody.id), next(store.climbs))
        assertEquals(AnalysisQueuePolicy.Step.Wait, next(store.climbs, settled = setOf(nobody.id)))
        val old = save()
        store.replaceAnalysis(old, posed(world = false))
        assertEquals(AnalysisQueuePolicy.Step.Settle(old.id), next(store.climbs, settled = setOf(nobody.id)))
        val fine = save()
        store.replaceAnalysis(fine, posed(world = true))
        // A pack without an analysis page: nothing to run.
        assertEquals(AnalysisQueuePolicy.Step.Settle(fine.id), next(store.climbs.filter { it.id == fine.id }, hasPage = false))
        // Records that never asked are left alone.
        save(capture = capture(queued = null))
        save(capture = null, pending = null)
        assertEquals(AnalysisQueuePolicy.Step.Wait, next(store.climbs.filter { it.capture?.analysisQueued != true }))
    }

    @Test fun theCameraGateIsClosedUnlessSet() {
        assertFalse(CameraGate.open.value)
        CameraGate.set(true); assertTrue(CameraGate.open.value)
        CameraGate.set(false); assertFalse(CameraGate.open.value)
    }

    @Test fun thePlanIsHandedToTheAnalysisPageAsAString() {
        val script = AnalysisScript.begin("A", 540, 960, null, plan)
        assertTrue(script, script.contains(""""capture":"{\"v\":1,\"standIn\":true,\"note\":\"é\\n\\\"quoted\\\"\"}""""))
        assertFalse(AnalysisScript.begin("A", 540, 960, null, null).contains("capture"))
        assertFalse(AnalysisScript.begin("A", 540, 960, null, "").contains("capture"))
    }

    // Which of the pack's gyms a directory entry stands for

    private fun manifest(venues: String) = GymManifest.parse(
        """{"schema":1,"packVersion":"0.0.0-standin","generatedAt":"2026-01-01T00:00:00Z","gymName":"合成岩馆","routeCount":0,"files":[],"venues":$venues}""",
    )

    @Test fun aVenueIsRecognisedByTheDirectoryIdsThePackLists() {
        val manifest = manifest("""[{"id":"synthetic-east","name":"合成岩馆 抱石 东门店","directoryIds":["syn:east","syn:east-old"]},{"id":"synthetic-west","name":"合成岩馆 抱石 西门店","directoryIds":["syn:west"]}]""")
        assertEquals("synthetic-east", manifest.venueId("syn:east-old", "名字完全不一样"))
        assertEquals("synthetic-west", manifest.venueId("syn:west", null))
        // Not listed: the name decides, as before.
        assertEquals("synthetic-east", manifest.venueId("syn:other", "合成岩馆（东门）"))
        assertNull(manifest.venueId("syn:other", "别的岩馆（东门）"))
        // Listed for two venues: nothing is guessed.
        assertNull(manifest("""[{"id":"a","directoryIds":["x"]},{"id":"b","directoryIds":["x"]}]""").venueId("x", "a"))
    }

    @Test fun withoutIdsAVenueIsRecognisedByBrandAndBranch() {
        val manifest = manifest("""[{"id":"synthetic-east","name":"合成岩馆 抱石 东门店","names":{"en":"Synthetic Gym East"}},{"id":"synthetic-west","name":"合成岩馆 抱石 西门店"},7,{"name":"no id"}]""")
        assertEquals(2, manifest.venues.size)
        assertEquals("synthetic-east", manifest.venueId(null, "合成岩馆（东门）"))
        assertEquals("synthetic-west", manifest.venueId(null, "合成岩馆(西门)"))
        assertEquals("synthetic-east", manifest.venueId(null, "synthetic gym east"))
        assertNull(manifest.venueId(null, "合成岩馆"))              // a brand alone says nothing about which branch
        assertNull(manifest.venueId(null, "合成岩馆（南门）"))
        assertNull(manifest.venueId(null, "")); assertNull(manifest.venueId(null, null)); assertNull(manifest.venueId(null, "（）"))
        assertTrue(manifest("[]").venues.isEmpty())
        assertNull(GymManifest.parse("""{"packVersion":"1","generatedAt":"x","gymName":"g","routeCount":0,"files":[]}""").venueId("a", "g（b）"))
    }
}
