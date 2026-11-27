package com.betabuddy.android.journal

import android.Manifest
import android.app.Application
import android.content.ContentUris
import android.content.Context
import android.content.pm.PackageManager
import android.database.ContentObserver
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import androidx.core.content.ContextCompat
import androidx.core.content.edit
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.betabuddy.android.R
import com.betabuddy.android.data.LibraryReceipt
import com.betabuddy.android.data.PoseAnalysis
import com.betabuddy.android.data.UNGRADED_VALUE
import com.betabuddy.android.pose.FrameGeometry
import com.betabuddy.android.pose.ImportRules
import com.betabuddy.android.pose.VideoAnalyzer
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.coroutines.coroutineContext

internal enum class LibraryAccess { NONE, SELECTED, ALL }
internal object LibraryPermission {
    fun granted(context: Context, permission: String) = ContextCompat.checkSelfPermission(context,permission) == PackageManager.PERMISSION_GRANTED
    fun access(context: Context): LibraryAccess = when {
        Build.VERSION.SDK_INT >= 33 && granted(context,Manifest.permission.READ_MEDIA_VIDEO) -> LibraryAccess.ALL
        Build.VERSION.SDK_INT >= 34 && granted(context,Manifest.permission.READ_MEDIA_VISUAL_USER_SELECTED) -> LibraryAccess.SELECTED
        Build.VERSION.SDK_INT < 33 && granted(context,Manifest.permission.READ_EXTERNAL_STORAGE) -> LibraryAccess.ALL
        else -> LibraryAccess.NONE
    }
    fun request(): Array<String> = buildList {
        if (Build.VERSION.SDK_INT >= 33) add(Manifest.permission.READ_MEDIA_VIDEO) else add(Manifest.permission.READ_EXTERNAL_STORAGE)
        if (Build.VERSION.SDK_INT >= 34) add(Manifest.permission.READ_MEDIA_VISUAL_USER_SELECTED)
        if (Build.VERSION.SDK_INT >= 29) add(Manifest.permission.ACCESS_MEDIA_LOCATION)
    }.toTypedArray()
}

internal data class LibraryImportState(
    val enabled: Boolean = false, val access: LibraryAccess = LibraryAccess.NONE,
    val running: Boolean = false, val checked: Int = 0, val total: Int = 0, val imported: Int = 0,
    val unavailable: Int = 0, val error: Int? = null,
)

/** Local MediaStore only, never cloud picker providers. A source is re-read before every commit. */
internal class LibraryMedia(private val context: Context) {
    data class Video(val uri: Uri, val id: String, val revision: String, val size: Long, val duration: Long, val taken: Long?)
    private val version = if (Build.VERSION.SDK_INT >= 29) MediaStore.getVersion(context) else "legacy"
    suspend fun videos(uri: Uri = MediaStore.Video.Media.EXTERNAL_CONTENT_URI): List<Video> {
        if (LibraryPermission.access(context) == LibraryAccess.NONE) throw SecurityException("Library permission unavailable")
        val fields = mutableListOf("_id", "_size", "duration", "datetaken", "date_modified", "date_added", "width", "height")
        if (Build.VERSION.SDK_INT >= 30) fields += "generation_modified"
        val selection = if (Build.VERSION.SDK_INT >= 29) "is_pending = 0" + (if (Build.VERSION.SDK_INT >= 30) " AND is_trashed = 0" else "") else null
        val values = mutableListOf<Video>()
        context.contentResolver.query(uri,fields.toTypedArray(),selection,null,"datetaken DESC, _id DESC")?.use { cursor ->
            while (cursor.moveToNext()) {
                coroutineContext.ensureActive()
                fun number(key: String) = cursor.getLong(cursor.getColumnIndexOrThrow(key))
                val item = ContentUris.withAppendedId(MediaStore.Video.Media.EXTERNAL_CONTENT_URI,number("_id"))
                val id = "android:" + LibraryClassifier.digest("$version:$item:${number("date_added")}".toByteArray())
                val revision = LibraryClassifier.digest(fields.drop(1).joinToString(":") { number(it).toString() }.toByteArray())
                values += Video(item,id,revision,number("_size"),number("duration"),number("datetaken").takeIf { it > 0 })
            }
        } ?: error("Library unavailable")
        return values
    }
    suspend fun current(video: Video): Boolean = videos(video.uri).singleOrNull()?.let { it.id == video.id && it.revision == video.revision } == true
    suspend fun pickedIdentity(uri: Uri): String? {
        val media = if (Build.VERSION.SDK_INT >= 29) runCatching { MediaStore.getMediaUri(context,uri) }.getOrNull() ?: uri else uri
        if (media.authority != MediaStore.AUTHORITY || "video" !in media.pathSegments) return null
        return try { videos(media).singleOrNull()?.id } catch (error: CancellationException) { throw error } catch (_: Exception) { null }
    }
    /** No permission prompt here, and never a request for the current device location. */
    fun location(video: Video): Pair<Double,Double>? {
        if (Build.VERSION.SDK_INT >= 29 && !LibraryPermission.granted(context,Manifest.permission.ACCESS_MEDIA_LOCATION)) return null
        val reader = MediaMetadataRetriever()
        return try {
            val uri = if (Build.VERSION.SDK_INT >= 29) MediaStore.setRequireOriginal(video.uri) else video.uri
            reader.setDataSource(context,uri)
            LibraryMetadata.location(reader.extractMetadata(MediaMetadataRetriever.METADATA_KEY_LOCATION))
        } catch (_: Exception) { null }
        finally { runCatching { reader.release() } }
    }
}

/** Opt-in and foreground only. Cancelled or unreadable sources never become negative receipts. */
internal class LibraryImport(application: Application) : AndroidViewModel(application) {
    private val context = application.applicationContext
    private val preferences = context.getSharedPreferences("library-import", Context.MODE_PRIVATE)
    private val mutable = MutableStateFlow(LibraryImportState(enabled = preferences.getBoolean("enabled",false), access = LibraryPermission.access(context)))
    val state = mutable.asStateFlow()
    private var foreground = false
    private var paused = false
    private var stopped = false
    private var job: Job? = null
    private var debounce: Job? = null
    @Volatile private var changed = false
    private var observing = false
    private val observer = object : ContentObserver(Handler(Looper.getMainLooper())) {
        override fun onChange(selfChange: Boolean) {
            changed = true
            debounce?.cancel()
            debounce = viewModelScope.launch { delay(600); if (job == null && eligible) scan() }
        }
    }
    private val eligible get() = foreground && !paused && !stopped && mutable.value.enabled && LibraryPermission.access(context) != LibraryAccess.NONE

    fun setEnabled(value: Boolean) {
        preferences.edit { putBoolean("enabled",value) }
        mutable.update { it.copy(enabled = value,error = null,imported = if(value && !it.enabled) 0 else it.imported) }
        stopped = false
        refresh()
    }
    fun setActive(active: Boolean, busy: Boolean) {
        foreground = active; paused = busy
        refresh()
    }
    fun refresh() {
        mutable.update { it.copy(access = LibraryPermission.access(context)) }
        val shouldObserve = foreground && mutable.value.enabled && mutable.value.access != LibraryAccess.NONE
        if (shouldObserve != observing) {
            try {
                if (shouldObserve) context.contentResolver.registerContentObserver(MediaStore.Video.Media.EXTERNAL_CONTENT_URI,true,observer)
                else context.contentResolver.unregisterContentObserver(observer)
                observing = shouldObserve
            } catch (_: SecurityException) {
                job?.cancel()
                mutable.update { it.copy(access = LibraryPermission.access(context),error = R.string.library_access_error) }
                return
            }
        }
        if (!eligible) job?.cancel() else if (job == null) scan()
    }
    fun stop() { stopped = true; job?.cancel(); debounce?.cancel() }
    fun retry() { if(job == null) mutable.update { it.copy(imported = 0) }; stopped = false; refresh() }
    private fun scan() {
        if (job != null || !eligible) return
        changed = false
        mutable.update { it.copy(running = true,checked = 0,total = 0,unavailable = 0,error = null) }
        job = viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) { runScan() }
            } catch (error: CancellationException) { throw error }
            catch (_: SecurityException) { mutable.update { it.copy(access = LibraryPermission.access(context),error = R.string.library_access_error) } }
            catch (_: Exception) { mutable.update { it.copy(error = R.string.library_scan_error) } }
            finally {
                mutable.update { it.copy(running = false) }; job = null
                // Resume after a foreground/camera transition even if cancellation was still unwinding then.
                if (eligible && (changed || !isActive)) scan()
            }
        }
    }
    private suspend fun runScan() {
        val store = JournalGraph.store(context)
        check(!store.loadFailed)
        val classifier = try { LibraryClassifier(context) } catch (_: Exception) {
            mutable.update { it.copy(error = R.string.library_model_missing) }; return
        }
        classifier.use {
            val media = LibraryMedia(context)
            val pending = media.videos().filterNot { store.hasHandledAsset(it.id,it.revision,classifier.revision) }
            mutable.update { it.copy(total = pending.size) }
            for (video in pending) {
                coroutineContext.ensureActive()
                try {
                    if (video.size > ImportRules.MAX_BYTES || video.duration > 1_200_000 || video.size <= 0) {
                        // Oversized, pending or unindexed originals remain retryable, never a negative classification.
                        mutable.update { it.copy(unavailable = it.unavailable+1) }; continue
                    }
                    val setting = classifier.classify(context,video.uri)
                    coroutineContext.ensureActive()
                    if (!media.current(video)) { changed = true; continue }
                    if (setting == null) {
                        store.recordLibraryNonmatch(video.id,video.revision,classifier.revision)
                    } else {
                        val activeContext = coroutineContext
                        val staged = VideoStager.stage(context,video.uri,JournalGraph.stagingDirectory(context)) { activeContext.isActive }
                        try {
                            val info = VideoAnalyzer.probe(staged)
                            val (width,height) = FrameGeometry.displaySize(info.width,info.height,info.rotation)
                            val thumbnail = VideoAnalyzer(context).thumbnail(staged)
                            val date = video.taken ?: VideoDates.read(context,video.uri,staged)
                            val gym = LibraryMetadata.suggestedGym(setting,media.location(video),loadDirectory(context))
                            coroutineContext.ensureActive()
                            if (!media.current(video)) { changed = true; continue }
                            coroutineContext.ensureActive()
                            store.save("",context.getString(R.string.journal_default_title),UNGRADED_VALUE,false,"",staged,thumbnail,
                                PoseAnalysis(emptyList(),0,info.durationSeconds!!,width.toDouble()/height),
                                gymId = gym?.id,gymName = gym?.name,climbedAtMillis = date,environment = setting,
                                sourceAssetID = video.id,libraryReceipt = LibraryReceipt(video.revision,classifier.revision,true))
                            mutable.update { it.copy(imported = it.imported+1) }
                        } finally { staged.delete() }
                    }
                } catch (error: CancellationException) { throw error }
                catch (_: InterruptedException) { coroutineContext.ensureActive(); mutable.update { it.copy(unavailable = it.unavailable+1) } }
                catch (error: SecurityException) { throw error }
                catch (_: Exception) { mutable.update { it.copy(unavailable = it.unavailable+1) } }
                finally { mutable.update { it.copy(checked = it.checked+1) } }
            }
        }
    }
    override fun onCleared() {
        foreground = false
        job?.cancel(); debounce?.cancel()
        if (observing) context.contentResolver.unregisterContentObserver(observer)
    }
}
