package com.betabuddy.android.data

import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.serialization.json.Json

/**
 * Local records: one JSON index plus the original clip and a thumbnail per climb, all inside
 * [root] (an app-private directory). Counterpart of ios/BetaBuddy/Core/ClimbStore.swift.
 *
 * - The index is written to a temporary file, synced, then renamed over the old one.
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
            }
            file("$INDEX_NAME.tmp").delete() // left behind only if a write was interrupted before the rename
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
        val temporary = file("$INDEX_NAME.tmp")
        val bytes = json.encodeToString(Archive.serializer(), next).toByteArray(Charsets.UTF_8)
        try {
            FileOutputStream(temporary).use { stream ->
                stream.write(bytes)
                stream.flush()
                stream.fd.sync()
            }
            // rename(2) replaces the destination atomically on Android and other POSIX systems.
            if (!temporary.renameTo(file(INDEX_NAME))) throw IOException("Cannot replace $INDEX_NAME")
        } catch (error: Exception) {
            temporary.delete()
            throw error
        }
        state.value = next
    }

    /**
     * Moves [stagedVideo] into the store and records the climb. On failure the staged file is put
     * back where it was and nothing is recorded.
     */
    @Synchronized
    fun save(
        title: String, defaultTitle: String, grade: String, sent: Boolean, note: String,
        stagedVideo: File, thumbnail: ByteArray, analysis: PoseAnalysis,
        nowMillis: Long = System.currentTimeMillis(),
        gymId: String? = null, gymName: String? = null, climbedAtMillis: Long? = null, folderId: String? = null,
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
            climbedAt = climbedAtMillis?.let(Climb::dateFromMillis), folderId = validFolder(folderId),
        )
        var moved = false
        try {
            move(stagedVideo, file(videoName))
            moved = true
            file(thumbnailName).writeBytes(thumbnail)
            persist(state.value.copy(climbs = state.value.climbs + climb))
            return climb
        } catch (error: Exception) {
            if (moved) runCatching { move(file(videoName), stagedVideo) }
            file(thumbnailName).delete()
            throw error
        }
    }

    /** Removes the record, then its clip and thumbnail. */
    @Synchronized
    fun delete(climb: Climb) {
        persist(state.value.copy(climbs = state.value.climbs.filterNot { it.id == climb.id }))
        owned(climb.videoName)?.delete()
        owned(climb.thumbnailName)?.delete()
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

    @Synchronized
    fun organize(climb: Climb, grade: String, gymId: String?, gymName: String?, climbedAtMillis: Long, folderId: String?) {
        validFolder(folderId)
        val current = state.value
        require(current.climbs.any { it.id == climb.id }) { "Unknown climb" }
        persist(current.copy(climbs = current.climbs.map {
            if (it.id == climb.id) it.copy(grade = grade, gymId = gymId,
                gymName = gymName?.trim()?.prefix(120)?.takeIf(String::isNotEmpty),
                climbedAt = Climb.dateFromMillis(climbedAtMillis), folderId = folderId) else it
        }))
    }

    /** Replaces only the analysis; id, date, notes, send status and the clip stay as they are. */
    @Synchronized
    fun replaceAnalysis(climb: Climb, analysis: PoseAnalysis) {
        val current = state.value
        if (current.climbs.none { it.id == climb.id }) return
        persist(current.copy(climbs = current.climbs.map {
            if (it.id == climb.id) it.copy(analysis = analysis) else it
        }))
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
