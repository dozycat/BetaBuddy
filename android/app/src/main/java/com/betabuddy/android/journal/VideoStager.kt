package com.betabuddy.android.journal

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import android.webkit.MimeTypeMap
import com.betabuddy.android.data.newId
import com.betabuddy.android.pose.AnalysisError
import com.betabuddy.android.pose.AnalysisException
import com.betabuddy.android.pose.ImportRules
import com.betabuddy.android.pose.VideoAnalyzer
import java.io.File
import java.io.FileNotFoundException

/** The picked item could not be opened (for example a cloud original that was not downloaded). */
internal class VideoUnavailableException : Exception()

/**
 * Copies the one video the user picked into app-private storage and checks it against the import
 * limits. Reads only the picked URI through the content resolver; needs no storage permission.
 */
internal object VideoStager {
    fun stage(context: Context, uri: Uri, directory: File, keepGoing: () -> Boolean): File {
        val resolver = context.contentResolver
        val declaredSize = runCatching {
            resolver.query(uri, arrayOf(OpenableColumns.SIZE), null, null, null)?.use { cursor ->
                if (cursor.moveToFirst() && !cursor.isNull(0)) cursor.getLong(0) else null
            }
        }.getOrNull()
        if (declaredSize != null && declaredSize > ImportRules.MAX_BYTES) throw AnalysisException(AnalysisError.TOO_LARGE)

        val extension = resolver.getType(uri)?.let { MimeTypeMap.getSingleton().getExtensionFromMimeType(it) }
            ?.filter { it.isLetterOrDigit() }?.ifEmpty { null } ?: "mp4"
        if (!directory.isDirectory && !directory.mkdirs()) throw FileNotFoundException(directory.name)
        val target = File(directory, "betabuddy-${newId()}.$extension")
        try {
            val input = (try {
                resolver.openInputStream(uri)
            } catch (error: FileNotFoundException) {
                null
            }) ?: throw VideoUnavailableException()
            input.use { source ->
                target.outputStream().use { sink ->
                    try {
                        copyLimited(source, sink, ImportRules.MAX_BYTES, keepGoing)
                    } catch (error: TooLargeException) {
                        throw AnalysisException(AnalysisError.TOO_LARGE)
                    }
                }
            }
            val info = VideoAnalyzer.probe(target)
            ImportRules.check(target.length(), info.durationSeconds, info.hasVideo)?.let { throw AnalysisException(it) }
            return target
        } catch (error: Throwable) {
            target.delete()
            throw error
        }
    }

    /** Removes clips left in the staging directory by a process that died mid-import. */
    fun clear(directory: File) {
        directory.listFiles()?.forEach { it.delete() }
    }
}
