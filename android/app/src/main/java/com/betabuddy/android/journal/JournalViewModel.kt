package com.betabuddy.android.journal

import android.app.Application
import android.content.Context
import android.net.Uri
import androidx.annotation.StringRes
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.betabuddy.android.R
import com.betabuddy.android.data.Climb
import com.betabuddy.android.data.ClimbFolder
import com.betabuddy.android.data.ClimbStore
import com.betabuddy.android.data.PoseAnalysis
import com.betabuddy.android.pose.FrameGeometry
import com.betabuddy.android.pose.ImportRules
import java.util.Locale
import com.betabuddy.android.pose.AnalysisError
import com.betabuddy.android.pose.AnalysisException
import com.betabuddy.android.pose.AnalysisQuality
import com.betabuddy.android.pose.VideoAnalyzer
import java.io.File
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** A message resolved to the current language only when shown. */
data class UiMessage(@StringRes val id: Int, val argument: String? = null)

data class ImportUiState(
    val staged: File? = null,
    val recordingDateMillis: Long? = null,
    /** Copying the picked video into private storage. */
    val importing: Boolean = false,
    /** Preparing the thumbnail and committing the saved clip. */
    val busy: Boolean = false,
    val error: UiMessage? = null,
    val savedId: String? = null,
)

data class ReanalysisUiState(
    val climbId: String? = null,
    val busy: Boolean = false,
    val progress: Double = 0.0,
    val message: UiMessage? = null,
)

/** One store per process, kept in app-private storage (`filesDir/BetaBuddy`), never on shared storage. */
internal object JournalGraph {
    @Volatile private var store: ClimbStore? = null

    fun root(context: Context): File = File(context.applicationContext.filesDir, "BetaBuddy")
    fun stagingDirectory(context: Context): File = File(root(context), "staging")

    /** Blocks on disk the first time; call from a background thread. */
    fun store(context: Context): ClimbStore = store ?: synchronized(this) {
        store ?: run {
            VideoStager.clear(stagingDirectory(context))
            ClimbStore(root(context)).also { store = it }
        }
    }
}

internal fun AnalysisError.message(): UiMessage = UiMessage(
    when (this) {
        AnalysisError.MISSING_MODEL -> R.string.journal_error_missing_model
        AnalysisError.INVALID_VIDEO -> R.string.journal_error_invalid_video
        AnalysisError.TOO_LONG -> R.string.journal_error_too_long
        AnalysisError.TOO_LARGE -> R.string.journal_error_too_large
        AnalysisError.NO_POSE -> R.string.journal_error_no_pose
        AnalysisError.OVERHEATED -> R.string.journal_error_overheated
    },
)

/**
 * State for the logbook and one import at a time. Analysis belongs to the service.
 * Disk and inference never run on the main thread; the UI only receives low-frequency progress.
 */
class JournalViewModel(application: Application) : AndroidViewModel(application) {
    private val analyzer = VideoAnalyzer(application)

    private val climbsState = MutableStateFlow<List<Climb>?>(null)
    /** Newest first; null until the index has been read. */
    val climbs: StateFlow<List<Climb>?> = climbsState.asStateFlow()

    private val foldersState = MutableStateFlow<List<ClimbFolder>>(emptyList())
    val folders: StateFlow<List<ClimbFolder>> = foldersState.asStateFlow()

    private val unreadableState = MutableStateFlow(false)
    /** The index exists but could not be read. It is kept as it is and nothing is written. */
    val storeUnreadable: StateFlow<Boolean> = unreadableState.asStateFlow()

    private val importState = MutableStateFlow(ImportUiState())
    val import: StateFlow<ImportUiState> = importState.asStateFlow()

    val reanalysis: StateFlow<ReanalysisUiState> = AnalysisService.state

    private var stagingJob: Job? = null
    private var saveJob: Job? = null

    init {
        viewModelScope.launch {
            val store = store()
            unreadableState.value = store.loadFailed
            store.archive.collect { climbsState.value = ClimbStore.sorted(it); foldersState.value = it.folders }
        }
    }

    private suspend fun store(): ClimbStore = withContext(Dispatchers.IO) { JournalGraph.store(getApplication()) }

    fun videoFile(climb: Climb): File = File(JournalGraph.root(getApplication()), climb.videoName)
    fun thumbnailFile(climb: Climb): File = File(JournalGraph.root(getApplication()), climb.thumbnailName)

    // Import

    /** Called with the single item the system photo picker returned, or null when it was dismissed. */
    fun onVideoPicked(uri: Uri?) {
        if (uri == null || importState.value.busy || importState.value.importing) return
        importState.update { it.copy(importing = true, error = null) }
        stagingJob = viewModelScope.launch {
            try {
                val staged = withContext(Dispatchers.IO) {
                    store() // also clears stale staging files before the first import
                    VideoStager.stage(getApplication(), uri, JournalGraph.stagingDirectory(getApplication())) { isActive }
                }
                val recordingDate = withContext(Dispatchers.IO) { VideoDates.read(getApplication(), uri, staged) }
                val previous = importState.value.staged
                importState.update { it.copy(staged = staged, recordingDateMillis = recordingDate, error = null) }
                previous?.delete()
            } catch (error: CancellationException) {
                throw error
            } catch (error: InterruptedException) {
                // closed while copying; the partial file is already removed
            } catch (error: AnalysisException) {
                importState.update { it.copy(error = error.error.message()) }
            } catch (error: VideoUnavailableException) {
                importState.update { it.copy(error = UiMessage(R.string.journal_import_unavailable)) }
            } catch (error: Exception) {
                importState.update { it.copy(error = UiMessage(R.string.journal_import_failed, error.localizedMessage ?: error.javaClass.simpleName)) }
            } finally {
                importState.update { it.copy(importing = false) }
            }
        }
    }

    fun saveVideo(title: String, grade: String, sent: Boolean, note: String, quality: AnalysisQuality, defaultTitle: String, gymId: String? = null, gymName: String? = null, climbedAtMillis: Long? = null, folderId: String? = null, analyze: Boolean = false) {
        val state = importState.value
        val video = state.staged ?: return
        if (state.busy || state.importing) return
        importState.update { it.copy(busy = true, error = null) }
        saveJob = viewModelScope.launch {
            try {
                val analysis = withContext(Dispatchers.IO) {
                    val info = VideoAnalyzer.probe(video)
                    ImportRules.check(video.length(), info.durationSeconds, info.hasVideo)?.let { throw AnalysisException(it) }
                    val (width, height) = FrameGeometry.displaySize(info.width, info.height, info.rotation)
                    PoseAnalysis(emptyList(), 0, info.durationSeconds!!, width.toDouble() / height, 0.0, emptyList())
                }
                val thumbnail = analyzer.thumbnail(video)
                ensureActive()
                // Commit the clip and durable request before starting optional inference.
                val climb = withContext(Dispatchers.IO) {
                    store().save(title, defaultTitle, grade, sent, note, video, thumbnail, analysis, gymId = gymId, gymName = gymName, climbedAtMillis = climbedAtMillis, folderId = folderId, pendingAnalysis = if (analyze) quality.name.lowercase(Locale.ROOT) else null)
                }
                if (analyze) AnalysisService.start(getApplication())
                importState.update { it.copy(staged = null, savedId = climb.id) }
            } catch (error: CancellationException) {
                importState.update { it.copy(error = UiMessage(R.string.journal_import_stopped)) }
            } catch (error: AnalysisException) {
                importState.update { it.copy(error = error.error.message()) }
            } catch (error: Exception) {
                importState.update { it.copy(error = UiMessage(R.string.journal_import_failed, error.localizedMessage ?: error.javaClass.simpleName)) }
            } finally {
                importState.update { it.copy(busy = false) }
            }
        }
    }

    /** Leaves the import flow: stops any work and removes a staged clip that was not saved. */
    fun closeImport() {
        stagingJob?.cancel()
        saveJob?.cancel()
        val staged = importState.value.staged
        importState.value = ImportUiState()
        if (staged != null) viewModelScope.launch(Dispatchers.IO) {
            saveJob?.join() // the decoder has let go of the file by now
            staged.delete()
        }
    }

    suspend fun saveFolder(name: String, id: String? = null): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching { store().saveFolder(name, id) }
    }
    suspend fun deleteFolder(id: String): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching { store().deleteFolder(id) }
    }
    suspend fun organize(climb: Climb, grade: String, gymId: String?, gymName: String?, climbedAtMillis: Long, folderId: String?): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching { store().organize(climb, grade, gymId, gymName, climbedAtMillis, folderId) }
    }

    // Review

    suspend fun saveNote(climb: Climb, note: String, sent: Boolean): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching { store().update(climb, note, sent) }
    }

    suspend fun delete(climb: Climb): Result<Unit> {
        AnalysisService.cancel(getApplication(), climb.id)
        return withContext(Dispatchers.IO) { runCatching { store().delete(climb) } }
    }

    /** The service owns inference, so leaving this screen or activity does not stop it. */
    fun reanalyze(climb: Climb) {
        if (climb.pendingAnalysis != null) return
        viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) { store().requestAnalysis(climb.id, "balanced") }
                AnalysisService.start(getApplication())
            } catch (error: Exception) {
                AnalysisService.reportFailure(climb.id, error)
            }
        }
    }

    fun cancelReanalysis(climb: Climb) { AnalysisService.cancel(getApplication(), climb.id) }

    override fun onCleared() {
        importState.value.staged?.delete()
    }
}
