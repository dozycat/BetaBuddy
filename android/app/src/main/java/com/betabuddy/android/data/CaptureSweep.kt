package com.betabuddy.android.data

import java.io.File
import kotlinx.serialization.Serializable

/**
 * What was chosen when a recording started, written next to the clip while it is being recorded,
 * so that a clip found again after the app was closed still knows its gym and grade.
 */
@Serializable
data class ClipNote(
    val gymId: String? = null,
    val gymName: String? = null,
    val grade: String? = null,
    val startedMillis: Long? = null,
)

/**
 * Looks through the staging directory when the app starts. A clip the in-app camera recorded but
 * that never reached the logbook (the app was closed, or the save failed and the screen went
 * away) is taken into the logbook as it is when it can be played; one that cannot (the recorder
 * was cut off before it closed the file) is removed. Everything else in there is left over from
 * an import that did not finish, and is removed as before. Pure: the tests use synthetic files.
 */
object CaptureSweep {
    const val CLIP_PREFIX = "betabuddy-capture-"
    private const val NOTE_SUFFIX = ".json"
    /** Shorter than this there is nothing to review. */
    const val MIN_SECONDS = 1.0

    /** What can be told about a clip: [seconds] long, [aspectRatio] as displayed. */
    class Facts(val seconds: Double, val aspectRatio: Double)

    class Outcome(val adopted: List<Climb>, val removed: List<String>, val kept: List<String>)

    fun noteFile(clip: File): File = File(clip.path + NOTE_SUFFIX)

    fun writeNote(clip: File, note: ClipNote) {
        runCatching { noteFile(clip).writeText(ClimbStore.json.encodeToString(ClipNote.serializer(), note), Charsets.UTF_8) }
    }

    private fun readNote(clip: File): ClipNote? = runCatching {
        ClimbStore.json.decodeFromString(ClipNote.serializer(), noteFile(clip).readText(Charsets.UTF_8))
    }.getOrNull()

    /**
     * @param facts what a clip is, or null when it cannot be played
     * @param thumbnail a picture for the logbook, or null when none could be made
     */
    fun sweep(
        staging: File, store: ClimbStore, defaultTitle: String,
        facts: (File) -> Facts?, thumbnail: (File) -> ByteArray?,
    ): Outcome {
        val adopted = ArrayList<Climb>()
        val removed = ArrayList<String>()
        val kept = ArrayList<String>()
        val files = staging.listFiles()?.sortedBy { it.name }.orEmpty()
        for (file in files) {
            if (!file.isFile || (file.name.startsWith(CLIP_PREFIX) && file.name.endsWith(NOTE_SUFFIX))) continue
            if (!file.name.startsWith(CLIP_PREFIX)) {
                file.delete(); removed += file.name
                continue
            }
            val known = runCatching { facts(file) }.getOrNull()
            if (known == null || !(known.seconds >= MIN_SECONDS) || !(known.aspectRatio > 0)) {
                file.delete(); noteFile(file).delete(); removed += file.name
                continue
            }
            val note = readNote(file)
            try {
                adopted += store.save(
                    title = "", defaultTitle = defaultTitle, grade = note?.grade ?: UNGRADED_VALUE, sent = false, note = "",
                    stagedVideo = file, thumbnail = runCatching { thumbnail(file) }.getOrNull() ?: ByteArray(0),
                    analysis = PoseAnalysis(emptyList(), 0, known.seconds, known.aspectRatio),
                    gymId = note?.gymId, gymName = note?.gymName,
                    climbedAtMillis = note?.startedMillis ?: (file.lastModified() - (known.seconds * 1000).toLong()),
                    // A record of the camera's like any other: pose recognition is asked for, there is no plan.
                    pendingAnalysis = "balanced", capture = CaptureInfo(),
                )
                noteFile(file).delete()
            } catch (error: Exception) {
                kept += file.name   // The logbook cannot be written now: the clip waits for the next start.
            }
        }
        // Notes whose clip is gone.
        for (file in staging.listFiles().orEmpty()) {
            if (file.name.startsWith(CLIP_PREFIX) && file.name.endsWith(NOTE_SUFFIX) && !File(file.path.removeSuffix(NOTE_SUFFIX)).exists()) file.delete()
        }
        return Outcome(adopted, removed, kept)
    }
}
