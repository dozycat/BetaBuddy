package com.betabuddy.android.journal

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
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
        revision = digest(bytes + "|image-sampling-2".toByteArray())
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
            val largeFrame = maxOf(width,height) > 3840
            require(!largeFrame || Build.VERSION.SDK_INT >= 27)
            val votes = mutableListOf<String?>()
            for (i in 0 until config.samples) {
                coroutineContext.ensureActive()
                val time = (duration * 1000 * (i + .5) / config.samples).toLong()
                val frame = (if (largeFrame && Build.VERSION.SDK_INT >= 27) reader.getScaledFrameAtTime(time, MediaMetadataRetriever.OPTION_CLOSEST,3840,3840)
                    else reader.getFrameAtTime(time, MediaMetadataRetriever.OPTION_CLOSEST))
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
    internal suspend fun embedding(frame: Bitmap, mode: String = "center"): List<Double> {
        val input = sampledPixels(frame, mode)
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
        /** Separable, scale-aware triangle filter: reducing an image averages its full footprint.
         * The same RGB sampling is used by both native encoders, including integer black padding.
         */
        internal suspend fun sampledPixels(frame: Bitmap, mode: String): FloatArray {
            coroutineContext.ensureActive()
            val side = minOf(frame.width, frame.height)
            val sw = if (mode == "center") side else frame.width
            val sh = if (mode == "center") side else frame.height
            val left = (frame.width-sw)/2; val top = (frame.height-sh)/2
            val scale = 224.0/maxOf(sw,sh)
            val dw = if (mode == "fit") maxOf(1,Math.rint(sw*scale).toInt()) else 224
            val dh = if (mode == "fit") maxOf(1,Math.rint(sh*scale).toInt()) else 224
            val padX = Math.rint((224-dw)/2.0).toInt(); val padY = Math.rint((224-dh)/2.0).toInt()
            val pixels = IntArray(frame.width*frame.height)
            frame.getPixels(pixels,0,frame.width,0,0,frame.width,frame.height)
            val wx = weights(sw,dw); val wy = weights(sh,dh)
            val horizontal = IntArray(sh*dw*3)
            for (y in 0 until sh) {
                coroutineContext.ensureActive()
                for (x in 0 until dw) for (channel in 0..2) {
                    var value = 0.0
                    for ((index,weight) in wx[x]) value += ((pixels[(top+y)*frame.width+left+index] ushr ((2-channel)*8)) and 255)*weight
                    horizontal[(y*dw+x)*3+channel] = (value+.5).toInt().coerceIn(0,255)
                }
            }
            val result = FloatArray(224*224*3)
            for (y in 0 until dh) {
                coroutineContext.ensureActive()
                for (x in 0 until dw) for (channel in 0..2) {
                    var value = 0.0
                    for ((index,weight) in wy[y]) value += horizontal[(index*dw+x)*3+channel]*weight
                    result[channel*224*224+(y+padY)*224+x+padX] = (value+.5).toInt().coerceIn(0,255)/255f
                }
            }
            return result
        }

        private fun weights(source: Int, target: Int): List<List<Pair<Int,Double>>> {
            val scale = source.toDouble()/target; val radius = maxOf(1.0,scale)
            return List(target) { output ->
                val center = (output+.5)*scale
                val lo = maxOf(0,kotlin.math.floor(center-radius).toInt())
                val hi = minOf(source-1,kotlin.math.ceil(center+radius).toInt())
                val values = (lo..hi).mapNotNull { input ->
                    val weight = maxOf(0.0,1-kotlin.math.abs(input+.5-center)/radius)
                    if (weight > 0) input to weight else null
                }
                val sum = values.sumOf { it.second }
                values.map { it.first to it.second/sum }
            }
        }
        fun digest(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
    }
}
