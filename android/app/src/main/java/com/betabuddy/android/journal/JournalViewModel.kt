package com.betabuddy.android.journal

import android.app.Application
import android.content.Context
import android.net.Uri
import androidx.annotation.StringRes
import androidx.core.content.edit
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.betabuddy.android.R
import com.betabuddy.android.data.Archive
import com.betabuddy.android.data.CapturePlanUse
import com.betabuddy.android.data.CaptureSweep
import com.betabuddy.android.data.Climb
import com.betabuddy.android.data.ClimbFolder
import com.betabuddy.android.data.ClimbStore
import com.betabuddy.android.data.PoseAnalysis
import com.betabuddy.android.edit.EditFiles
import com.betabuddy.android.edit.EditPlan
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
    val sourceAssetID: String? = null,
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
            val created = ClimbStore(root(context))
            // Rendered edits are remade on demand; whatever the last run left in the cache goes.
            EditFiles.clearAll(context)
            // Left over from the last run: unfinished imports go; a clip the camera recorded that
            // never reached the logbook is taken in if it can be played.
            val app = context.applicationContext
            CaptureSweep.sweep(
                stagingDirectory(context), created, app.getString(R.string.journal_default_title),
                facts = { clip ->
                    val info = VideoAnalyzer.probe(clip)
                    val seconds = info.durationSeconds
                    if (!info.hasVideo || seconds == null) null else {
                        val (width, height) = FrameGeometry.displaySize(info.width, info.height, info.rotation)
                        CaptureSweep.Facts(seconds, width.toDouble() / height)
                    }
                },
                thumbnail = { clip -> kotlinx.coroutines.runBlocking { VideoAnalyzer(app).thumbnail(clip) } },
            )
            created.also { store = it }
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
 * State for the logbook and one import at a time. Pose recognition belongs to the service.
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

    private val resultsState = MutableStateFlow<Set<String>>(emptySet())
    /** Ids of the climbs that have a saved result from the pack's analysis page. */
    val analysisResults: StateFlow<Set<String>> = resultsState.asStateFlow()

    private val awaitingState = MutableStateFlow<String?>(null)
    /** The climb whose pose recognition was asked for by "Analyse this attempt" and has not finished yet. */
    val awaitingPose: StateFlow<String?> = awaitingState.asStateFlow()

    private val poseReadyState = MutableStateFlow<String?>(null)
    /** That climb, once its pose recognition has finished with data the analysis page can use. */
    val poseReady: StateFlow<String?> = poseReadyState.asStateFlow()

    private val editRevisionState = MutableStateFlow(0)
    /** Changes whenever a stored edit plan may have changed; screens read the plan again then. */
    val editRevision: StateFlow<Int> = editRevisionState.asStateFlow()

    private val queueMessageState = MutableStateFlow<Map<String, UiMessage>>(emptyMap())
    /** How the analysis a camera record asked for ended, per record, for as long as the app runs. */
    val queueMessages: StateFlow<Map<String, UiMessage>> = queueMessageState.asStateFlow()

    private val preferences by lazy { application.getSharedPreferences("journal", Context.MODE_PRIVATE) }

    private var stagingJob: Job? = null
    private var saveJob: Job? = null

    init {
        viewModelScope.launch {
            val store = store()
            unreadableState.value = store.loadFailed
            resultsState.value = withContext(Dispatchers.IO) { store.analysisResultIds() }
            store.archive.collect {
                climbsState.value = ClimbStore.sorted(it)
                foldersState.value = it.folders
                settleAwaitedPose(it)
            }
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
                val sourceID = withContext(Dispatchers.IO) { runCatching { LibraryMedia(getApplication()).pickedIdentity(uri) }.getOrNull() }
                val previous = importState.value.staged
                importState.update { it.copy(staged = staged, recordingDateMillis = recordingDate, sourceAssetID = sourceID, error = null) }
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

    fun saveVideo(title: String, grade: String, sent: Boolean, note: String, quality: AnalysisQuality, defaultTitle: String, gymId: String? = null, gymName: String? = null, climbedAtMillis: Long? = null, folderId: String? = null, analyze: Boolean = false, discipline: String? = null) {
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
                    PoseAnalysis(emptyList(), 0, info.durationSeconds!!, width.toDouble() / height)
                }
                val thumbnail = analyzer.thumbnail(video)
                ensureActive()
                // Commit the clip and durable request before starting optional inference.
                val climb = withContext(Dispatchers.IO) {
                    store().save(title, defaultTitle, grade, sent, note, video, thumbnail, analysis, gymId = gymId, gymName = gymName, climbedAtMillis = climbedAtMillis, folderId = folderId, pendingAnalysis = if (analyze) quality.name.lowercase(Locale.ROOT) else null, discipline = discipline, sourceAssetID = state.sourceAssetID)
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
    /** [discipline] is the user's choice, or null when they did not touch it. */
    suspend fun organize(climb: Climb, grade: String, gymId: String?, gymName: String?, climbedAtMillis: Long, folderId: String?, discipline: String? = null, environment: String? = null, gymChosen: Boolean = false): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching { store().organize(climb, grade, gymId, gymName, climbedAtMillis, folderId, discipline, environment, gymChosen) }
    }

    // Review

    suspend fun saveNote(climb: Climb, note: String, sent: Boolean): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching { store().update(climb, note, sent) }
    }

    suspend fun delete(climb: Climb): Result<Unit> {
        AnalysisService.cancel(getApplication(), climb.id)
        return withContext(Dispatchers.IO) { runCatching { store().delete(climb); EditFiles.clear(getApplication(), climb.id) } }
            .onSuccess { resultsState.update { ids -> ids - climb.id }; editRevisionState.update { it + 1 } }
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

    // The pack's analysis page

    /**
     * "Analyse this attempt" on a climb whose saved poses the page cannot use (never recognised, or
     * saved before world landmarks were kept): pose recognition goes through the usual queue, and
     * [poseReady] names the climb when it has finished with usable data.
     */
    fun analyzeWhenPoseReady(climb: Climb) {
        viewModelScope.launch {
            try {
                val store = store()
                val queued = store.archive.value.climbs.firstOrNull { it.id == climb.id }?.pendingAnalysis != null
                if (!queued) withContext(Dispatchers.IO) { store.requestAnalysis(climb.id, "balanced") }
                poseReadyState.value = null
                awaitingState.value = climb.id
                settleAwaitedPose(store.archive.value) // the request is in the archive by now, so this only ends the wait if it is already over
                AnalysisService.start(getApplication())
            } catch (error: Exception) {
                AnalysisService.reportFailure(climb.id, error)
            }
        }
    }

    /** The wait ends when the request leaves the archive: finished, stopped, failed, or the climb deleted. */
    private fun settleAwaitedPose(archive: Archive) {
        val id = awaitingState.value ?: return
        val climb = archive.climbs.firstOrNull { it.id == id }
        if (climb?.pendingAnalysis != null) return
        awaitingState.value = null
        if (climb != null && (climb.analysis.hasWorldLandmarks || climb.analysis.hasSelectedSubject)) poseReadyState.value = id
    }

    /** The screen has acted on [poseReady], or is going away and no longer wants to be told. */
    fun forgetAnalysisIntent(id: String) {
        if (awaitingState.value == id) awaitingState.value = null
        if (poseReadyState.value == id) poseReadyState.value = null
    }

    suspend fun analysisResult(id: String): String? = withContext(Dispatchers.IO) { store().analysisResult(id) }
    suspend fun currentClimb(id: String): Climb? = withContext(Dispatchers.IO) { store().climbs.firstOrNull { it.id == id } }
    suspend fun subjectAnalysis(id: String): com.betabuddy.android.data.PoseAnalysis? = withContext(Dispatchers.IO) { store().subjectAnalysis(id) }

    suspend fun selectSubject(climb: Climb, analysis: com.betabuddy.android.data.PoseAnalysis, anchors: List<com.betabuddy.android.data.SubjectAnchor>): Result<Unit> {
        AnalysisService.cancel(getApplication(),climb.id)
        return withContext(Dispatchers.IO) {
            runCatching { store().selectSubject(climb,analysis,anchors); EditFiles.clear(getApplication(),climb.id) }
        }.onSuccess {
            resultsState.update { it - climb.id }; editRevisionState.update { it + 1 }
            queueMessageState.update { it - climb.id }; forgetAnalysisIntent(climb.id)
        }
    }

    /**
     * Keeps what the page handed back: the result next to the clip, replacing an earlier one, and
     * the body measurements it used in this phone's preferences for the next analysis.
     */
    suspend fun saveAnalysisResult(
        id: String, json: String, climber: String?, planUse: CapturePlanUse? = null,
        /** The page's plan for an automatic edit, kept as it came; a result without one removes the last. */
        edit: String? = null,
        /** What the page judged the climb to be. Kept only while the user has not said so themselves. */
        discipline: String? = null,
        expectedSubjectRevision: Int? = null,
    ): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching {
            if (!store().saveAnalysisResult(id, json, expectedSubjectRevision)) return@runCatching
            resultsState.update { it + id }
            store().saveEditPlan(id, edit, expectedSubjectRevision)
            editRevisionState.update { it + 1 }
            if (discipline != null) store().setAutomaticDiscipline(id, discipline, expectedSubjectRevision)
            // Whether the page went by the capture plan is kept with the record, as the page said it.
            if (planUse != null) store().setPlanUse(id, planUse, expectedSubjectRevision)
            if (climber != null) preferences.edit { putString(KEY_ANALYSIS_CLIMBER, climber) }
        }
    }

    /** What the page made of the climbing style when it could not analyse the clip. Never replaces the user's choice. */
    /** A plan for an automatic edit that came without a result (the page could not analyse, but could still plan an edit). */
    suspend fun saveEditPlan(id: String, plan: String, expectedSubjectRevision: Int? = null) {
        withContext(Dispatchers.IO) { runCatching { store().saveEditPlan(id, plan, expectedSubjectRevision) } }
        editRevisionState.update { it + 1 }
    }

    suspend fun setAutomaticDiscipline(id: String, discipline: String, expectedSubjectRevision: Int? = null) {
        withContext(Dispatchers.IO) { runCatching { store().setAutomaticDiscipline(id, discipline, expectedSubjectRevision) } }
    }

    /** The capture page's plan for a record made with the in-app camera, or null. */
    suspend fun capturePlan(id: String): String? = withContext(Dispatchers.IO) { store().capturePlan(id) }

    /** The queue is done with this record's request: cleared in the archive, with what to tell the user, if anything. */
    suspend fun settleQueuedAnalysis(id: String, message: UiMessage?) {
        // Analysed without the marks made while framing is still analysed, but it is said.
        val unused = message?.id == R.string.capture_analysis_done && withContext(Dispatchers.IO) { store().archive.value.climbs.firstOrNull { it.id == id }?.capture?.planUse?.used == false }
        val message = if (unused) UiMessage(R.string.capture_plan_not_used) else message
        withContext(Dispatchers.IO) { runCatching { store().setAnalysisQueued(id, false) } }
        if (message != null) queueMessageState.update { it + (id to message) }
    }

    suspend fun discardAnalysisResult(id: String) {
        withContext(Dispatchers.IO) { runCatching { store().discardAnalysisResult(id) } }
        resultsState.update { it - id }
        editRevisionState.update { it + 1 }
    }

    /** The stored plan for an automatic edit, when there is one this build can use. */
    suspend fun editPlan(id: String): EditPlan? = withContext(Dispatchers.IO) { EditPlan.parse(store().editPlan(id)) }

    /** The discipline the user chose for this record themselves, or null. */
    suspend fun chosenDiscipline(id: String): String? = withContext(Dispatchers.IO) { store().archive.value.climbs.firstOrNull { it.id == id }?.chosenDiscipline }

    /** JSON text of the body measurements the page last used, or null before the first result. */
    suspend fun analysisClimber(): String? = withContext(Dispatchers.IO) { preferences.getString(KEY_ANALYSIS_CLIMBER, null) }

    override fun onCleared() {
        importState.value.staged?.delete()
    }

    private companion object {
        const val KEY_ANALYSIS_CLIMBER = "analysisClimber"
    }
}
