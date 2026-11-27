package com.betabuddy.android.journal

import android.Manifest
import android.app.Application
import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Build
import android.provider.MediaStore
import androidx.test.platform.app.InstrumentationRegistry
import com.betabuddy.android.capture.SyntheticWallClip
import com.betabuddy.android.data.Climb
import java.io.File
import kotlinx.coroutines.*
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test

/** Optional real clip comes from the test runner; no private footage or labels enter the repository. */
class LibraryImportTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext

    @Test fun optionalClassifierVideoMatrix() = runBlocking {
        val name = InstrumentationRegistry.getArguments().getString("libraryMatrix")
        assumeTrue("Local evaluation matrix not supplied",name != null)
        val input = org.json.JSONObject(File(context.filesDir,name!!).readText()).getJSONArray("clips")
        assertTrue(input.length()>0)
        val rows = org.json.JSONArray(); val failures = mutableListOf<String>()
        withContext(Dispatchers.IO) {
            LibraryClassifier(context).use { classifier ->
                for (i in 0 until input.length()) {
                    val clip=input.getJSONObject(i); val id=clip.getString("id")
                    val expected=if (clip.isNull("expected")) null else clip.getString("expected")
                    val start=System.nanoTime()
                    val result=org.json.JSONObject().put("id",id).put("expected",expected ?: org.json.JSONObject.NULL)
                    try {
                        val actual=classifier.classify(context,android.net.Uri.fromFile(File(context.filesDir,clip.getString("path"))))
                        result.put("actual",actual ?: org.json.JSONObject.NULL)
                        if (actual!=expected) failures += "$id: expected $expected, got $actual"
                    } catch (failure: Exception) { result.put("error",failure.toString()); failures += "$id: $failure" }
                    result.put("seconds",(System.nanoTime()-start)/1e9); rows.put(result)
                }
                File(context.getExternalFilesDir(null),"library-classifier-matrix.json").writeText(org.json.JSONObject().put("revision",classifier.revision).put("clips",rows).toString(2))
            }
        }
        assertTrue(failures.joinToString("\n"),failures.isEmpty())
    }

    @Test fun encoderRejectsBlankVideoAndClosesResources() = runBlocking {
        val bitmap = Bitmap.createBitmap(320,480,Bitmap.Config.ARGB_8888).apply { eraseColor(Color.DKGRAY) }
        val clip = File(context.cacheDir,"library-negative-${System.nanoTime()}.mp4")
        try {
            SyntheticWallClip.make(clip,width=320,height=480,seconds=2.0,photo=bitmap)
            withContext(Dispatchers.IO) {
                LibraryClassifier(context).use { classifier ->
                    val vector = classifier.embedding(bitmap)
                    assertEquals(512,vector.size); assertEquals(1.0,vector.sumOf { it*it },.005)
                    assertNull(classifier.classify(context,android.net.Uri.fromFile(clip)))
                }
            }
        } finally { bitmap.recycle(); clip.delete() }
    }

    @Test fun realLibraryClipImportsOnceKeepsDateAndHonoursStopAndDeletion() = runBlocking {
        val name = InstrumentationRegistry.getArguments().getString("libraryClip")
        assumeTrue("Optional local climbing fixture",name != null)
        val fixture = File(context.filesDir,name!!); assertTrue(fixture.isFile)
        val permission = if (Build.VERSION.SDK_INT >= 33) Manifest.permission.READ_MEDIA_VIDEO else Manifest.permission.READ_EXTERNAL_STORAGE
        instrumentation.uiAutomation.grantRuntimePermission(context.packageName,permission)
        assertEquals(LibraryAccess.ALL,LibraryPermission.access(context))
        val date = 1_600_000_000_000L
        val values = ContentValues().apply {
            put(MediaStore.Video.Media.DISPLAY_NAME,"library-test-${System.nanoTime()}.mp4")
            put(MediaStore.Video.Media.MIME_TYPE,"video/mp4")
            put(MediaStore.Video.Media.DATE_TAKEN,date)
            if (Build.VERSION.SDK_INT >= 29) { put(MediaStore.Video.Media.RELATIVE_PATH,"Movies/BetaBuddyTests"); put(MediaStore.Video.Media.IS_PENDING,1) }
        }
        val uri = context.contentResolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI,values)!!
        val prior = context.getSharedPreferences("library-import",0).getBoolean("enabled",false)
        var model: LibraryImport? = null
        var saved: Climb? = null
        val store = withContext(Dispatchers.IO) { JournalGraph.store(context) }
        try {
            context.contentResolver.openOutputStream(uri)!!.use { output -> fixture.inputStream().use { it.copyTo(output) } }
            if (Build.VERSION.SDK_INT >= 29) context.contentResolver.update(uri,ContentValues().apply { put(MediaStore.Video.Media.IS_PENDING,0); put(MediaStore.Video.Media.DATE_TAKEN,date) },null,null)
            val source = withContext(Dispatchers.IO) { LibraryMedia(context).videos(uri).single() }
            assertEquals(date,source.taken)
            withContext(Dispatchers.Main) {
                model = LibraryImport(context.applicationContext as Application)
                model!!.setEnabled(true); model!!.setActive(true,false); model!!.stop()
            }
            withTimeout(30_000) { while (model!!.state.value.running) delay(50) }
            assertFalse(store.hasImportedAsset(source.id))
            withContext(Dispatchers.Main) { model!!.retry() }
            withTimeout(180_000) { while (model!!.state.value.running) delay(100) }
            assertNull(model!!.state.value.error)
            saved = store.climbs.single { it.sourceAssetID == source.id }
            assertEquals(date,saved!!.climbedAtMillis)
            assertEquals("indoor",saved!!.environment)
            assertTrue(saved!!.environmentAuto == true)
            assertTrue(store.file(saved!!.videoName).isFile)
            withContext(Dispatchers.Main) { model!!.retry() }
            withTimeout(30_000) { while (model!!.state.value.running) delay(100) }
            assertEquals(1,store.climbs.count { it.sourceAssetID == source.id })
            withContext(Dispatchers.IO) { store.delete(saved!!); saved = null }
            withContext(Dispatchers.Main) { model!!.retry() }
            withTimeout(30_000) { while (model!!.state.value.running) delay(100) }
            assertTrue(store.climbs.none { it.sourceAssetID == source.id })
            assertTrue(store.hasImportedAsset(source.id))
        } finally {
            withContext(Dispatchers.Main) { model?.setActive(false,false); model?.setEnabled(prior) }
            saved?.let { withContext(Dispatchers.IO) { store.delete(it) } }
            context.contentResolver.delete(uri,null,null)
        }
    }
}
