package com.betabuddy.android.journal

import android.Manifest
import android.app.Application
import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.BitmapFactory
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

    @Test fun optionalEncoderImageParity() = runBlocking {
        val manifest = InstrumentationRegistry.getArguments().getString("encoderImages")
        assumeTrue("Optional common decoded images not supplied", manifest != null)
        val inputs = org.json.JSONArray(File(context.filesDir, manifest!!).readText())
        assertTrue(inputs.length() > 0)
        val rows = org.json.JSONArray()
        withContext(Dispatchers.IO) {
            LibraryClassifier(context).use { classifier ->
                for (i in 0 until inputs.length()) {
                    val input = inputs.getJSONObject(i)
                    val image = requireNotNull(BitmapFactory.decodeFile(File(context.filesDir, input.getString("path")).path))
                    try {
                        val prepared = input.getBoolean("prepared")
                        for (mode in if (prepared) listOf("prepared") else listOf("center", "fit")) {
                            val expected = input.optJSONObject("views")?.getJSONArray(mode) ?: input.getJSONArray("expected")
                            val vector = classifier.embedding(image, if (prepared) "stretch" else mode)
                            assertEquals(expected.length(), vector.size)
                            assertEquals(1.0, vector.sumOf { it*it }, .005)
                            val cosine = vector.indices.sumOf { vector[it]*expected.getDouble(it) }
                            rows.put(org.json.JSONObject().put("id", input.getString("id")).put("mode", mode)
                                .put("embedding", org.json.JSONArray(vector)).put("cosine", cosine))
                            assertTrue("Native sampling and encoder must agree: ${input.getString("id")}, $mode", cosine > .999)
                        }
                    } finally { image.recycle() }
                }
            }
        }
        File(context.getExternalFilesDir(null), "encoder-parity.json").writeText(rows.toString())
    }

    @Test fun samplingAveragesFineDetailAndKeepsPaddingColourAndOrientation() = runBlocking {
        val width = 448; val height = 896
        val image = Bitmap.createBitmap(width,height,Bitmap.Config.ARGB_8888)
        try {
            val checker = IntArray(width*height) { if ((it%width+it/width)%2 == 0) Color.BLACK else Color.WHITE }
            image.setPixels(checker,0,width,0,0,width,height)
            val pixels = LibraryClassifier.sampledPixels(image,"fit")
            for (channel in 0..2) assertEquals(128/255.0,pixels[channel*224*224+112*224+112].toDouble(),1/255.0)
            assertEquals(0f,pixels[112*224],0f)
            image.setPixels(IntArray(width*height) { if (it/width < height/2) Color.RED else Color.BLUE },0,width,0,0,width,height)
            val colours = LibraryClassifier.sampledPixels(image,"fit")
            assertEquals(1f,colours[10*224+112],0f); assertEquals(0f,colours[2*224*224+10*224+112],0f)
            assertEquals(0f,colours[210*224+112],0f); assertEquals(1f,colours[2*224*224+210*224+112],0f)
        } finally { image.recycle() }
    }

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
            val imported = model!!.state.value.imported
            assertTrue(imported >= 1)
            withContext(Dispatchers.Main) { model!!.setActive(false,false); model!!.setActive(true,false) }
            withTimeout(30_000) { while (model!!.state.value.running) delay(100) }
            assertEquals(imported,model!!.state.value.imported)
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
