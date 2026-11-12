package com.betabuddy.android.journal

import android.app.Application
import android.content.Context
import android.net.Uri
import androidx.annotation.StringRes
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.betabuddy.android.R
import com.betabuddy.android.data.Climb
import com.betabuddy.android.data.ClimbStore
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
    /** Copying the picked video into private storage. */
    val importing: Boolean = false,
    /** Analysing. */
    val busy: Boolean = false,
    val progress: Double = 0.0,
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
 * State for the logbook: the records, one import at a time and one re-analysis at a time.
 * Disk and inference never run on the main thread; the UI only receives low-frequency progress.
 */
class JournalViewModel(application: Application) : AndroidViewModel(application) {
    private val analyzer = VideoAnalyzer(application)

    private val climbsState = MutableStateFlow<List<Climb>?>(null)
    /** Newest first; null until the index has been read. */
    val climbs: StateFlow<List<Climb>?> = climbsState.asStateFlow()

    private val unreadableState = MutableStateFlow(false)
    /** The index exists but could not be read. It is kept as it is and nothing is written. */
    val storeUnreadable: StateFlow<Boolean> = unreadableState.asStateFlow()

    private val importState = MutableStateFlow(ImportUiState())
    val import: StateFlow<ImportUiState> = importState.asStateFlow()

    private val reanalysisState = MutableStateFlow(ReanalysisUiState())
    val reanalysis: StateFlow<ReanalysisUiState> = reanalysisState.asStateFlow()

    private var stagingJob: Job? = null
    private var analysisJob: Job? = null
    private var reanalysisJob: Job? = null

    init {
        viewModelScope.launch {
            val store = store()
            unreadableState.value = store.loadFailed
            store.archive.collect { climbsState.value = ClimbStore.sorted(it) }
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
                val previous = importState.value.staged
                importState.update { it.copy(staged = staged, error = null) }
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

    fun startAnalysis(title: String, grade: String, sent: Boolean, note: String, quality: AnalysisQuality, defaultTitle: String) {
        val state = importState.value
        val video = state.staged ?: return
        if (state.busy || state.importing) return
        importState.update { it.copy(busy = true, progress = 0.0, error = null) }
        analysisJob = viewModelScope.launch {
            try {
                val analysis = analyzer.analyze(video, quality) { value -> importState.update { it.copy(progress = value) } }
                val thumbnail = analyzer.thumbnail(video)
                ensureActive()
                // Only a finished result is written to the record.
                val climb = withContext(Dispatchers.IO) {
                    store().save(title, defaultTitle, grade, sent, note, video, thumbnail, analysis)
                }
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

    /** Stop button, and leaving the foreground. The video stays selected. */
    fun cancelAnalysis() { analysisJob?.cancel() }

    /** Leaves the import flow: stops any work and removes a staged clip that was not saved. */
    fun closeImport() {
        stagingJob?.cancel()
        analysisJob?.cancel()
        val staged = importState.value.staged
        importState.value = ImportUiState()
        if (staged != null) viewModelScope.launch(Dispatchers.IO) {
            analysisJob?.join() // the decoder has let go of the file by now
            staged.delete()
        }
    }

    // Review

    suspend fun saveNote(climb: Climb, note: String, sent: Boolean): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching { store().update(climb, note, sent) }
    }

    suspend fun delete(climb: Climb): Result<Unit> {
        if (reanalysisState.value.climbId == climb.id) reanalysisJob?.cancel()
        return withContext(Dispatchers.IO) { runCatching { store().delete(climb) } }
    }

    /** Recomputes the skeleton from the stored clip. Notes, send status, id and date are kept. */
    fun reanalyze(climb: Climb) {
        if (reanalysisState.value.busy || importState.value.busy) return
        reanalysisState.value = ReanalysisUiState(climbId = climb.id, busy = true)
        reanalysisJob = viewModelScope.launch {
            val message = try {
                val result = analyzer.analyze(videoFile(climb), AnalysisQuality.BALANCED) { value ->
                    reanalysisState.update { it.copy(progress = value) }
                }
                ensureActive()
                withContext(Dispatchers.IO) { store().replaceAnalysis(climb, result) }
                UiMessage(R.string.journal_reanalyze_done)
            } catch (error: CancellationException) {
                UiMessage(R.string.journal_reanalyze_stopped) // cancelled or failed: the old analysis stays
            } catch (error: AnalysisException) {
                error.error.message()
            } catch (error: Exception) {
                UiMessage(R.string.journal_import_failed, error.localizedMessage ?: error.javaClass.simpleName)
            }
            reanalysisState.update { it.copy(busy = false, message = message) }
        }
    }

    fun cancelReanalysis() { reanalysisJob?.cancel() }

    fun clearReanalysisMessage() {
        if (!reanalysisState.value.busy) reanalysisState.value = ReanalysisUiState()
    }

    override fun onCleared() {
        importState.value.staged?.delete()
    }
}
