package com.betabuddy.android.journal

import android.content.Context
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.provider.MediaStore
import java.io.File

internal object VideoDates {
    /** Never use the staged file's creation/modified time: that is the import time. */
    fun read(context: Context, uri: Uri, video: File): Long? {
        val libraryDate = runCatching {
            context.contentResolver.query(uri, arrayOf(MediaStore.Video.Media.DATE_TAKEN), null, null, null)?.use {
                val column = it.getColumnIndex(MediaStore.Video.Media.DATE_TAKEN)
                if (column >= 0 && it.moveToFirst() && !it.isNull(column)) it.getLong(column).takeIf { value -> value > 0 } else null
            }
        }.getOrNull()
        if (libraryDate != null) return libraryDate
        val retriever = MediaMetadataRetriever()
        return try {
            retriever.setDataSource(video.path)
            RecordingDate.parse(retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DATE))
        } catch (_: Exception) { null }
        finally { runCatching { retriever.release() } }
    }
}
