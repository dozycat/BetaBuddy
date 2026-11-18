package com.betabuddy.android.review

import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.os.Build
import android.util.Base64
import androidx.core.graphics.scale
import java.io.ByteArrayOutputStream
import java.io.File
import kotlin.coroutines.coroutineContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext

/**
 * Still pictures of a clip for the pack's analysis page: upright (rotation applied, the same
 * orientation the pose landmarks use), scaled down, JPEG, base64. Made when the page opens and
 * handed over one at a time; nothing is written to disk.
 */
internal object AnalysisFrames {
    private const val JPEG_QUALITY = 70

    /**
     * Takes the picture nearest each of [times] (seconds, ascending) and passes it to [onFrame]
     * with its time. Times the decoder has no picture for are skipped. Returns how many were
     * delivered; zero when the file cannot be read. Runs off the main thread and stops when cancelled.
     */
    suspend fun extract(video: File, times: List<Double>, width: Int, height: Int, onFrame: suspend (time: Double, jpegBase64: String) -> Unit): Int =
        withContext(Dispatchers.IO) {
            val retriever = MediaMetadataRetriever()
            var delivered = 0
            try {
                // A file that is not a readable video: nothing is delivered.
                if (runCatching { retriever.setDataSource(video.path) }.isFailure) return@withContext 0
                val output = ByteArrayOutputStream()
                for (time in times) {
                    coroutineContext.ensureActive()
                    val frame = runCatching { frameAt(retriever, time, width, height) }.getOrNull() ?: continue
                    output.reset()
                    val encoded = frame.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, output)
                    frame.recycle()
                    if (!encoded) continue
                    onFrame(time, Base64.encodeToString(output.toByteArray(), Base64.NO_WRAP))
                    delivered += 1
                }
            } finally {
                runCatching { retriever.release() }
            }
            delivered
        }

    /** The retriever applies the clip's rotation itself, so its pictures are already upright. */
    private fun frameAt(retriever: MediaMetadataRetriever, time: Double, width: Int, height: Int): Bitmap? {
        val timeUs = (time * 1_000_000).toLong()
        val frame = if (Build.VERSION.SDK_INT >= 27) {
            // Decoded close to the wanted size; a square box, so it fits whichever way the clip is turned.
            val box = maxOf(width, height)
            retriever.getScaledFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST, box, box)
        } else {
            retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST)   // API 26: full size, scaled below
        } ?: return null
        if (frame.width == width && frame.height == height) return frame
        val scaled = frame.scale(width, height)
        if (scaled !== frame) frame.recycle()
        return scaled
    }
}
