package com.betabuddy.android.journal

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.Color
import android.media.MediaMetadataRetriever
import android.net.Uri
import com.betabuddy.android.data.ClimbStore
import com.betabuddy.android.gym.GymManifest
import java.io.Closeable
import java.nio.FloatBuffer
import java.security.MessageDigest
import kotlinx.coroutines.ensureActive
import kotlin.coroutines.coroutineContext

/** CPU-only, offline image encoder. One instance is used serially for one scan and then closed. */
internal class LibraryClassifier(context: Context) : Closeable {
    val config: LibraryClassifierConfig
    val revision: String
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        environment.setTelemetry(false)
        val bytes = context.assets.open("gympack/library-classifier.json").use { it.readBytes() }
        val manifest = context.assets.open("gympack/manifest.json").bufferedReader().use { GymManifest.parse(it.readText()) }
        require(manifest.files.any { it.path == "library-classifier.json" && it.sha256 == digest(bytes) })
        revision = digest(bytes)
        config = ClimbStore.json.decodeFromString<LibraryClassifierConfig>(bytes.toString(Charsets.UTF_8)).also { it.validate() }
        val model = context.assets.open(config.androidModel.file).use { it.readBytes() }
        require(digest(model) == config.androidModel.sha256) { "Encoder does not match the installed policy" }
        session = OrtSession.SessionOptions().use { options ->
            options.setIntraOpNumThreads(2)
            options.setInterOpNumThreads(1)
            environment.createSession(model, options)
        }
    }

    suspend fun classify(context: Context, uri: Uri): String? {
        val reader = MediaMetadataRetriever()
        try {
            reader.setDataSource(context, uri)
            val duration = reader.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toDoubleOrNull()
            require(duration != null && duration.isFinite() && duration > 0 && duration <= 1_200_000)
            val width = reader.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 0
            val height = reader.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 0
            require(width > 0 && height > 0)
            val scale = minOf(1.0, 960.0 / maxOf(width,height))
            val votes = mutableListOf<String?>()
            for (i in 0 until config.samples) {
                coroutineContext.ensureActive()
                val time = (duration * 1000 * (i + .5) / config.samples).toLong()
                val frame = reader.getScaledFrameAtTime(time, MediaMetadataRetriever.OPTION_CLOSEST, maxOf(1,(width*scale).toInt()), maxOf(1,(height*scale).toInt()))
                    ?: error("Video frame unavailable")
                try {
                    val views = config.imageModes.map { mode ->
                        coroutineContext.ensureActive()
                        config.tag(embedding(frame,mode))
                    }
                    votes += config.frameVote(views)
                } finally { frame.recycle() }
            }
            coroutineContext.ensureActive()
            return config.decision(votes)
        } finally { reader.release() }
    }

    /** Model includes normalization; policy chooses image framing, RGB 0…1 in CHW order. */
    internal fun embedding(frame: Bitmap, mode: String = "center"): List<Double> {
        val side = minOf(frame.width,frame.height)
        val x = (frame.width-side)/2; val y = (frame.height-side)/2
        val image = Bitmap.createBitmap(224,224,Bitmap.Config.ARGB_8888)
        val pixels = IntArray(224*224)
        try {
            val canvas = Canvas(image)
            canvas.drawColor(Color.BLACK)
            val source = if (mode == "center") Rect(x,y,x+side,y+side) else Rect(0,0,frame.width,frame.height)
            val scale = 224f/maxOf(frame.width,frame.height)
            val width = frame.width*scale; val height = frame.height*scale
            val target = if (mode == "fit") RectF((224-width)/2,(224-height)/2,(224+width)/2,(224+height)/2) else RectF(0f,0f,224f,224f)
            canvas.drawBitmap(frame,source,target,Paint(Paint.FILTER_BITMAP_FLAG))
            image.getPixels(pixels,0,224,0,0,224,224)
        } finally { image.recycle() }
        val input = FloatArray(pixels.size*3)
        pixels.forEachIndexed { i, pixel ->
            input[i] = ((pixel shr 16) and 255)/255f
            input[i+pixels.size] = ((pixel shr 8) and 255)/255f
            input[i+pixels.size*2] = (pixel and 255)/255f
        }
        return OnnxTensor.createTensor(environment,FloatBuffer.wrap(input),longArrayOf(1,3,224,224)).use { tensor ->
            session.run(mapOf("image" to tensor)).use { result ->
                val output = result.get("embedding").orElseThrow() as OnnxTensor
                val buffer = output.floatBuffer
                List(buffer.remaining()) { buffer.get().toDouble() }
            }
        }
    }
    override fun close() { session.close() }
    companion object {
        fun digest(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
    }
}
