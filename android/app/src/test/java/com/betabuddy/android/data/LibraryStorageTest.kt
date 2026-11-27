package com.betabuddy.android.data

import java.io.File
import java.nio.file.Files
import org.junit.After
import org.junit.Assert.*
import org.junit.Test

class LibraryStorageTest {
    private val root = Files.createTempDirectory("library-store").toFile()
    private val store = ClimbStore(File(root,"records"))
    @After fun cleanup() { root.deleteRecursively() }
    private fun save(id: String? = "asset") = store.save("","Video",UNGRADED_VALUE,false,"",
        File(root,"staged.mp4").apply { writeText("original") },byteArrayOf(),PoseAnalysis(emptyList(),0,5.0,1.0),
        sourceAssetID = id,environment = "indoor",gymId = "test",gymName = "Test gym",climbedAtMillis = 1_600_000_000_000,
        libraryReceipt = LibraryReceipt("source1","model1",true))

    @Test fun importedReceiptSurvivesCorrectionDeletionAndRestart() {
        val climb = save()
        assertTrue(climb.gymAuto == true && climb.environmentAuto == true)
        store.organize(climb,UNGRADED_VALUE,null,null,climb.climbedAtMillis,null,environment = "outdoor",gymChosen = true)
        val corrected = store.climbs.single()
        assertEquals("outdoor",corrected.environment); assertNull(corrected.gymAuto); assertNull(corrected.environmentAuto)
        store.delete(corrected)
        val reopened = ClimbStore(store.root)
        assertTrue(reopened.hasHandledAsset("asset","changed-source","changed-model"))
        assertTrue(reopened.climbs.isEmpty())
        store.recordLibraryNonmatch("asset","later","later")
        assertTrue(store.archive.value.libraryLedger.getValue("asset").imported)
        assertThrows(IllegalArgumentException::class.java) { save() }
        assertNotNull(save(null)) // manual import remains available
    }

    @Test fun negativesExpireAndAtomicWriteFailureDoesNotLoseTheSource() {
        store.recordLibraryNonmatch("negative","source1","model1")
        assertTrue(store.hasHandledAsset("negative","source1","model1"))
        assertFalse(store.hasHandledAsset("negative","source2","model1"))
        assertFalse(store.hasHandledAsset("negative","source1","model2"))
        val temporary = store.file(ClimbStore.INDEX_NAME+".tmp").apply { mkdir() }
        File(temporary,"occupied").writeText("block atomic replace")
        assertThrows(Exception::class.java) { save() }
        assertFalse(store.hasHandledAsset("asset","source1","model1"))
        assertEquals("original",File(root,"staged.mp4").readText())
        assertTrue(store.climbs.isEmpty())
        temporary.deleteRecursively()
        save()
        assertTrue(ClimbStore(store.root).hasImportedAsset("asset"))
    }
}
