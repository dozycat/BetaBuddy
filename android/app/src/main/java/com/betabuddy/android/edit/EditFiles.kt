package com.betabuddy.android.edit

import android.content.Context
import android.content.Intent
import androidx.core.content.FileProvider
import java.io.File

/**
 * Where rendered edits live: the app's cache, one directory per record. They are not part of the
 * record: made when the edit page asks, gone when it closes, when the record is deleted, and
 * whatever a stopped process left behind when the logbook is next opened. Nothing is written to
 * shared storage or the photo library.
 */
internal object EditFiles {
    private const val DIRECTORY = "edits"
    private const val WORK_DIRECTORY = "edit-work"
    private const val AUTHORITY_SUFFIX = ".edits"

    private fun root(context: Context) = File(context.applicationContext.cacheDir, DIRECTORY)

    /** A record's directory, created. [id] is used only when it is a plain name. */
    fun directory(context: Context, id: String): File {
        val name = id.filter { it.isLetterOrDigit() || it == '-' }.ifEmpty { "edit" }
        return File(root(context), name).apply { mkdirs() }
    }

    fun clear(context: Context, id: String) { directory(context, id).deleteRecursively() }

    /** Where a render keeps what it needs only while it runs (the frames a plate is made of). */
    fun work(context: Context): File = File(context.applicationContext.cacheDir, WORK_DIRECTORY).apply { mkdirs() }

    fun clearAll(context: Context) {
        root(context).deleteRecursively()
        File(context.applicationContext.cacheDir, WORK_DIRECTORY).deleteRecursively()
    }

    /**
     * Hands [file] to the system share sheet: the receiving app is given read access to this one
     * file through the app's own content provider, for as long as it needs it. No storage permission.
     */
    fun share(context: Context, file: File, title: String) {
        val uri = FileProvider.getUriForFile(context, context.packageName + AUTHORITY_SUFFIX, file)
        val send = Intent(Intent.ACTION_SEND).setType("video/mp4").putExtra(Intent.EXTRA_STREAM, uri)
            .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        send.clipData = android.content.ClipData.newRawUri(title, uri)
        context.startActivity(Intent.createChooser(send, title).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION))
    }
}
