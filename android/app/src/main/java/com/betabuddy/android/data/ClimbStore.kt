package com.betabuddy.android.data

import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.serialization.json.Json

/**
 * Local records: one JSON index plus the original clip and a thumbnail per climb, for a climb
 * the pack's analysis page has worked on, that page's result (`<id>.analysis.json`) and, when it
 * gave one, its plan for an automatic edit (`<id>.edit.json`), and for a climb
 * made with the in-app camera, the capture page's plan (`<id>.capture.json`). All inside
 * [root] (an app-private directory). Counterpart of ios/BetaBuddy/Core/ClimbStore.swift.
 *
 * - The index and a result are written to a temporary file, synced, then renamed over the old one.
 * - An index that cannot be read is left untouched and the store refuses to write.
 * - Unknown fields are ignored on read.
 *
 * No Android classes: runs in plain JVM unit tests. Calls block on disk; use a background thread.
 */
class ClimbStore(val root: File) {
    private val state = MutableStateFlow(Archive())
    val archive: StateFlow<Archive> = state.asStateFlow()

    /** True when the existing index could not be read. Nothing is written in that case. */
    var loadFailed: Boolean = false
        private set
    private var canWrite = true

    init {
        try {
            if (!root.isDirectory && !root.mkdirs()) throw IOException("Cannot create ${root.name}")
            val index = file(INDEX_NAME)
            if (index.exists()) {
                val loaded = json.decodeFromString(Archive.serializer(), index.readText(Charsets.UTF_8))
                if (loaded.version != 1) throw IOException("Unsupported archive version ${loaded.version}")
                state.value = loaded
                if (loaded.climbs.any { it.analysis.people != null }) persist(loaded)
            }
            // Left behind only if a write was interrupted before the rename.
            file("$INDEX_NAME.tmp").delete()
            root.listFiles { _, name -> name.endsWith("$ANALYSIS_RESULT_SUFFIX.tmp") || name.endsWith("$CAPTURE_PLAN_SUFFIX.tmp") || name.endsWith("$EDIT_PLAN_SUFFIX.tmp") }?.forEach { it.delete() }
        } catch (error: Exception) {
            canWrite = false // Never overwrite an unreadable archive with an empty one.
            loadFailed = true
        }
    }

    /** Newest first. */
    val climbs: List<Climb> get() = sorted(state.value)

    fun file(name: String): File = File(root, name)

    @Synchronized
    private fun persist(next: Archive) {
        if (!canWrite) throw IOException("The archive could not be read and is kept as it is")
        val created = mutableListOf<File>()
        try {
            val prepared = next.copy(climbs = next.climbs.map { climb ->
                val samples = climb.analysis.people ?: return@map climb
                val reference = newId()
                val target = SubjectCandidateStorage.file(root,climb.id,reference)
                created += target
                writeAtomically(target,SubjectCandidateStorage.encode(samples))
                climb.copy(analysis = climb.analysis.copy(people = null,peopleReference = reference))
            })
            writeAtomically(file(INDEX_NAME), json.encodeToString(Archive.serializer(), prepared).toByteArray(Charsets.UTF_8))
            val previous = state.value
            state.value = prepared
            previous.climbs.forEach { old ->
                val reference = old.analysis.peopleReference ?: return@forEach
                if (prepared.climbs.firstOrNull { it.id == old.id }?.analysis?.peopleReference != reference)
                    runCatching { SubjectCandidateStorage.file(root,old.id,reference).delete() }
            }
        } catch (error: Exception) {
            created.forEach { it.delete() }
            throw error
        }
    }

    private fun writeAtomically(target: File, bytes: ByteArray) {
        val temporary = File(target.parentFile, target.name + ".tmp")
        try {
            FileOutputStream(temporary).use { stream ->
                stream.write(bytes)
                stream.flush()
                stream.fd.sync()
            }
            // rename(2) replaces the destination atomically on Android and other POSIX systems.
            if (!temporary.renameTo(target)) throw IOException("Cannot replace ${target.name}")
        } catch (error: Exception) {
            temporary.delete()
            throw error
        }
    }

    /**
     * Moves [stagedVideo] into the store and records the climb. On failure the staged file is put
     * back where it was and nothing is recorded.
     *
     * [capture] and [capturePlan] come from the in-app camera. The clip, its thumbnail, the plan and
     * the request for analysis go in together: the index is written last, and only once.
     */
    @Synchronized
    fun save(
        title: String, defaultTitle: String, grade: String, sent: Boolean, note: String,
        stagedVideo: File, thumbnail: ByteArray, analysis: PoseAnalysis,
        nowMillis: Long = System.currentTimeMillis(),
        gymId: String? = null, gymName: String? = null, climbedAtMillis: Long? = null, folderId: String? = null, pendingAnalysis: String? = null,
        capture: CaptureInfo? = null, capturePlan: String? = null,
        /** What the user chose when saving, if anything. */
        discipline: String? = null,
    ): Climb {
        if (!canWrite) throw IOException("The archive could not be read and is kept as it is")
        val id = newId()
        val extension = stagedVideo.extension.filter { it.isLetterOrDigit() }.ifEmpty { "mp4" }
        val videoName = "$id.$extension"
        val thumbnailName = "$id.jpg"
        val climb = Climb(
            id = id, date = Climb.dateFromMillis(nowMillis),
            title = if (title.isBlank()) defaultTitle else title.prefix(60),
            grade = grade, sent = sent, note = note.prefix(1000),
            videoName = videoName, thumbnailName = thumbnailName, analysis = analysis,
            gymId = gymId, gymName = gymName?.trim()?.prefix(120)?.takeIf { it.isNotEmpty() },
            climbedAt = climbedAtMillis?.let(Climb::dateFromMillis), folderId = validFolder(folderId), pendingAnalysis = pendingAnalysis,
            capture = capture?.copy(hasPlan = !capturePlan.isNullOrEmpty()),
            discipline = discipline?.takeIf(Discipline::valid),
        )
        var moved = false
        try {
            move(stagedVideo, file(videoName))
            moved = true
            file(thumbnailName).writeBytes(thumbnail)
            if (climb.capture?.hasPlan == true && capturePlan != null) writeAtomically(file(id + CAPTURE_PLAN_SUFFIX), capturePlan.toByteArray(Charsets.UTF_8))
            persist(state.value.copy(climbs = state.value.climbs + climb))
            return state.value.climbs.first { it.id == id }
        } catch (error: Exception) {
            if (moved) runCatching { move(file(videoName), stagedVideo) }
            file(thumbnailName).delete()
            file(id + CAPTURE_PLAN_SUFFIX).delete()
            throw error
        }
    }

    /** Removes the record, then its clip, thumbnail, saved analysis result and edit plan. */
    @Synchronized
    fun delete(climb: Climb) {
        persist(state.value.copy(climbs = state.value.climbs.filterNot { it.id == climb.id }))
        owned(climb.videoName)?.delete()
        owned(climb.thumbnailName)?.delete()
        root.listFiles { _, name -> name.startsWith(climb.id + ".") }?.forEach { it.delete() }
    }

    // Records made with the in-app camera

    private fun capturePlanFile(id: String): File? = owned(id + CAPTURE_PLAN_SUFFIX)

    /** The capture page's plan exactly as it handed it over, for the analysis page. Null for imported clips. */
    fun capturePlan(id: String): String? {
        if (state.value.climbs.firstOrNull { it.id == id }?.capture?.hasPlan != true) return null
        return try {
            capturePlanFile(id)?.takeIf { it.isFile }?.readText(Charsets.UTF_8)
        } catch (error: IOException) {
            null
        }
    }

    /** A plan that arrived (or arrived again, fuller) after the clip was saved. Only for a record made with the camera. */
    @Synchronized
    fun saveCapturePlan(id: String, plan: String, summary: CaptureSummary?) {
        val climb = state.value.climbs.firstOrNull { it.id == id }
        val capture = climb?.capture ?: return
        if (plan.isEmpty()) return
        if (!canWrite) throw IOException("The archive could not be read and is kept as it is")
        writeAtomically(capturePlanFile(id) ?: throw IOException("Unusable record id"), plan.toByteArray(Charsets.UTF_8))
        val next = capture.copy(hasPlan = true, summary = summary?.stored() ?: capture.summary)
        persist(state.value.copy(climbs = state.value.climbs.map { if (it.id == id) it.copy(capture = next) else it }))
    }

    /** What the analysis page said it did with this record's capture plan, kept with the record. */
    @Synchronized
    fun setPlanUse(id: String, use: CapturePlanUse, expectedSubjectRevision: Int? = null) {
        if (expectedSubjectRevision != null && !isSubjectCurrent(id,expectedSubjectRevision)) return
        val capture = state.value.climbs.firstOrNull { it.id == id }?.capture ?: return
        val next = capture.copy(planUse = use)
        if (next == capture) return
        persist(state.value.copy(climbs = state.value.climbs.map { if (it.id == id) it.copy(capture = next) else it }))
    }

    /** Sets or clears the request for the pack's analysis that a camera record was saved with. */
    @Synchronized
    fun setAnalysisQueued(id: String, queued: Boolean) {
        val capture = state.value.climbs.firstOrNull { it.id == id }?.capture ?: return
        if ((capture.analysisQueued ?: false) == queued) return
        val next = capture.copy(analysisQueued = if (queued) true else null)
        persist(state.value.copy(climbs = state.value.climbs.map { if (it.id == id) it.copy(capture = next) else it }))
    }

    /**
     * The gym of the record made most recently that has one: where the next recording most likely
     * is. By when the record was made, not by the climbing date the user may have set on it.
     */
    val lastGym: Pair<String?, String>?
        get() = state.value.climbs.filter { !it.gymName.isNullOrBlank() }.maxByOrNull { it.date }?.let { it.gymId to it.gymName.orEmpty() }

    // Result of the pack's analysis page. The text is the page's own format: kept and handed back as it is.

    private fun derivedName(id: String, suffix: String): String? {
        val climb = state.value.climbs.firstOrNull { it.id == id } ?: return null
        return id + (climb.subjectRevision?.let { ".s$it" } ?: "") + suffix
    }
    private fun analysisResultFile(id: String): File? = derivedName(id,ANALYSIS_RESULT_SUFFIX)?.let(::owned)
    fun isSubjectCurrent(id: String, revision: Int): Boolean = state.value.climbs.any { it.id == id && (it.subjectRevision ?: 0) == revision }

    /** Ids of the climbs that have a saved result. */
    fun analysisResultIds(): Set<String> {
        val names = root.list()?.toSet().orEmpty()
        return state.value.climbs.mapNotNullTo(mutableSetOf()) { climb -> climb.id.takeIf { derivedName(it,ANALYSIS_RESULT_SUFFIX) in names } }
    }

    /** The page's result exactly as it handed it over, or null when there is none. */
    fun analysisResult(id: String): String? = try {
        analysisResultFile(id)?.takeIf { it.isFile }?.readText(Charsets.UTF_8)
    } catch (error: IOException) {
        null
    }

    /**
     * Kept next to the clip and written like the index. The page sends a result again after the
     * user adjusts it; each one replaces the last. Returns false when the record is gone
     * (deleted while the page was open) and nothing was written.
     */
    @Synchronized
    fun saveAnalysisResult(id: String, json: String, expectedSubjectRevision: Int? = null): Boolean {
        if (state.value.climbs.none { it.id == id }) return false
        if (expectedSubjectRevision != null && !isSubjectCurrent(id,expectedSubjectRevision)) return false
        if (!canWrite) throw IOException("The archive could not be read and is kept as it is")
        val target = analysisResultFile(id) ?: throw IOException("Unusable record id")
        writeAtomically(target, json.toByteArray(Charsets.UTF_8))
        return true
    }

    @Synchronized
    fun discardAnalysisResult(id: String) {
        analysisResultFile(id)?.delete()
        editPlanFile(id)?.delete()
    }

    // The analysis page's plan for an automatic edit. The page's own format: kept as it arrived.

    private fun editPlanFile(id: String): File? = derivedName(id,EDIT_PLAN_SUFFIX)?.let(::owned)

    /** The plan exactly as the page handed it over, or null when there is none. */
    fun editPlan(id: String): String? = try {
        editPlanFile(id)?.takeIf { it.isFile }?.readText(Charsets.UTF_8)
    } catch (error: IOException) {
        null
    }

    /**
     * Every result of the page replaces the plan: [plan] is what came with it, null when it came
     * without one (the plan of an earlier result then goes). Nothing is written for a record that is gone.
     */
    @Synchronized
    fun saveEditPlan(id: String, plan: String?, expectedSubjectRevision: Int? = null) {
        if (state.value.climbs.none { it.id == id }) return
        if (expectedSubjectRevision != null && !isSubjectCurrent(id,expectedSubjectRevision)) return
        val target = editPlanFile(id) ?: throw IOException("Unusable record id")
        if (plan.isNullOrEmpty()) { target.delete(); return }
        if (!canWrite) throw IOException("The archive could not be read and is kept as it is")
        writeAtomically(target, plan.toByteArray(Charsets.UTF_8))
    }

    /**
     * What the analysis page judged the climb to be. Kept only while the user has not chosen one
     * themselves, and never at the cost of a grade the user marked: when the record's grade is of
     * the other scale the judgement is not stored at all, and everything stays as it was.
     */
    @Synchronized
    fun setAutomaticDiscipline(id: String, discipline: String, expectedSubjectRevision: Int? = null) {
        if (expectedSubjectRevision != null && !isSubjectCurrent(id,expectedSubjectRevision)) return
        if (!Discipline.valid(discipline)) return
        val climb = state.value.climbs.firstOrNull { it.id == id } ?: return
        if (climb.chosenDiscipline != null) return
        if (GradeScale.fitting(climb.grade, discipline) != climb.grade) return
        val next = climb.copy(discipline = discipline, disciplineAuto = true)
        if (next == climb) return
        persist(state.value.copy(climbs = state.value.climbs.map { if (it.id == id) next else it }))
    }

    @Synchronized
    fun update(climb: Climb, note: String, sent: Boolean) {
        val current = state.value
        if (current.climbs.none { it.id == climb.id }) return
        persist(current.copy(climbs = current.climbs.map {
            if (it.id == climb.id) it.copy(note = note.prefix(1000), sent = sent) else it
        }))
    }

    private fun validFolder(id: String?): String? {
        require(id == null || state.value.folders.any { it.id == id }) { "Unknown folder" }
        return id
    }

    @Synchronized
    fun saveFolder(name: String, id: String? = null) {
        val cleaned = name.trim().prefix(40)
        require(cleaned.isNotEmpty()) { "Empty folder name" }
        require(state.value.folders.none { it.id != id && it.name.equals(cleaned, ignoreCase = true) }) { "Duplicate folder" }
        val next = if (id == null) state.value.folders + ClimbFolder(name = cleaned) else {
            require(state.value.folders.any { it.id == id }) { "Unknown folder" }
            state.value.folders.map { if (it.id == id) it.copy(name = cleaned) else it }
        }
        persist(state.value.copy(folders = next))
    }

    /** Removing a folder only unfiles its climbs; clips and analysis stay on disk. */
    @Synchronized
    fun deleteFolder(id: String) {
        persist(state.value.copy(
            folders = state.value.folders.filterNot { it.id == id },
            climbs = state.value.climbs.map { if (it.folderId == id) it.copy(folderId = null) else it },
        ))
    }

    /**
     * [discipline] is what the user chose in the sheet, or null when they left it alone (an
     * automatic one then stays automatic). A grade that is not of the chosen discipline's scale is
     * stored as "no grade".
     */
    @Synchronized
    fun organize(climb: Climb, grade: String, gymId: String?, gymName: String?, climbedAtMillis: Long, folderId: String?, discipline: String? = null) {
        validFolder(folderId)
        require(discipline == null || Discipline.valid(discipline)) { "Unknown discipline" }
        val current = state.value
        require(current.climbs.any { it.id == climb.id }) { "Unknown climb" }
        persist(current.copy(climbs = current.climbs.map {
            if (it.id == climb.id) it.copy(grade = if (discipline == null) grade else GradeScale.fitting(grade, discipline), gymId = gymId,
                gymName = gymName?.trim()?.prefix(120)?.takeIf(String::isNotEmpty),
                climbedAt = Climb.dateFromMillis(climbedAtMillis), folderId = folderId,
                discipline = discipline ?: it.discipline, disciplineAuto = if (discipline == null) it.disciplineAuto else null) else it
        }))
    }

    /** Queues pose recognition for a climb ([quality] names the level), or clears the request with null. */
    @Synchronized
    fun requestAnalysis(id: String, quality: String?) {
        persist(state.value.copy(climbs = state.value.climbs.map {
            if (it.id == id) it.copy(pendingAnalysis = quality) else it
        }))
    }

    /** Replaces only the pose data; id, date, notes, send status, the clip and a saved analysis result stay as they are. */
    @Synchronized
    fun replaceAnalysis(climb: Climb, analysis: PoseAnalysis) {
        val current = state.value
        if (!isSubjectCurrent(climb.id,climb.subjectRevision ?: 0)) return
        persist(current.copy(climbs = current.climbs.map {
            if (it.id == climb.id) it.copy(analysis = analysis, pendingAnalysis = null) else it
        }))
    }

    /** Missing/corrupt candidate caches can be regenerated from the original video. */
    @Synchronized
    fun subjectAnalysis(id: String): PoseAnalysis? {
        val analysis = state.value.climbs.firstOrNull { it.id == id }?.analysis ?: return null
        if (analysis.people != null) return analysis
        val reference = analysis.peopleReference ?: return analysis
        return try { analysis.copy(people = SubjectCandidateStorage.read(root,id,reference)) }
        catch (_: Exception) { analysis.copy(people = null,peopleReference = null) }
    }

    /** Commit the identity and invalidate all results together; late work keeps its old revision. */
    @Synchronized
    fun selectSubject(climb: Climb, analysis: PoseAnalysis, anchors: List<SubjectAnchor>) {
        require(isSubjectCurrent(climb.id,climb.subjectRevision ?: 0)) { "Selection changed while this screen was open" }
        require(analysis.subject != null && analysis.hasSubjectCandidates)
        val oldResult = analysisResultFile(climb.id); val oldEdit = editPlanFile(climb.id)
        persist(state.value.copy(climbs = state.value.climbs.map {
            if (it.id == climb.id) it.copy(analysis = analysis, subjectAnchors = anchors.takeIf { it.isNotEmpty() },
                subjectRevision = (it.subjectRevision ?: 0)+1, pendingAnalysis = null,
                capture = it.capture?.copy(analysisQueued = null,planUse = null),
                discipline = if (it.disciplineAuto == true) null else it.discipline, disciplineAuto = null) else it
        }))
        oldResult?.delete(); oldEdit?.delete()
    }

    /** A stored name is only ever resolved to a file directly inside [root]. */
    private fun owned(name: String): File? {
        val target = file(name)
        return if (name.isNotEmpty() && target.canonicalFile.parentFile == root.canonicalFile) target else null
    }

    private fun move(from: File, to: File) {
        if (from.renameTo(to)) return
        from.copyTo(to, overwrite = false) // different volume: copy, then remove the source
        if (!from.delete()) { to.delete(); throw IOException("Cannot move ${from.name}") }
    }

    companion object {
        const val INDEX_NAME = "archive.json"
        const val ANALYSIS_RESULT_SUFFIX = ".analysis.json"
        const val CAPTURE_PLAN_SUFFIX = ".capture.json"
        const val EDIT_PLAN_SUFFIX = ".edit.json"

        /** Missing optional fields read as null, unknown fields are ignored, defaults are written. */
        val json = Json { ignoreUnknownKeys = true; explicitNulls = false; encodeDefaults = true }

        fun sorted(archive: Archive): List<Climb> = archive.climbs.sortedByDescending { it.climbedAt ?: it.date }

        /** At most [count] characters, never cutting a surrogate pair in half. */
        internal fun String.prefix(count: Int): String {
            if (codePointCount(0, length) <= count) return this
            return substring(0, offsetByCodePoints(0, count))
        }
    }
}
