package com.betabuddy.android.edit

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.opengl.GLES20
import android.os.Handler
import android.os.Looper
import android.os.storage.StorageManager
import androidx.annotation.OptIn
import androidx.core.graphics.createBitmap
import androidx.core.graphics.get
import androidx.media3.common.C
import androidx.media3.common.Effect
import androidx.media3.common.MediaItem
import androidx.media3.common.MimeTypes
import androidx.media3.common.audio.SpeedProvider
import androidx.media3.common.util.ExperimentalApi
import androidx.media3.common.util.UnstableApi
import androidx.media3.effect.Presentation
import androidx.media3.transformer.Composition
import androidx.media3.transformer.EditedMediaItem
import androidx.media3.transformer.EditedMediaItemSequence
import androidx.media3.transformer.Effects
import androidx.media3.transformer.ExperimentalAnalyzerModeFactory
import androidx.media3.transformer.ExportException
import androidx.media3.transformer.ExportResult
import androidx.media3.transformer.ProgressHolder
import androidx.media3.transformer.Transformer
import com.betabuddy.android.pose.FrameGeometry
import com.betabuddy.android.pose.VideoAnalyzer
import java.io.BufferedInputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Collections
import java.util.UUID
import kotlin.coroutines.coroutineContext
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.math.ceil
import kotlin.math.floor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext

/** The clip cannot be edited: it has no readable video. */
internal class EditSourceException : IOException("no readable video")

/**
 * Renders the automatic edit of one clip from the analysis page's plan, on the device, with
 * Media3 Transformer: the plan's source ranges in order, each at its speed, each frame through
 * the plan's camera window, and passers-by replaced by a clean plate. It carries out the plan and
 * decides nothing itself.
 *
 * Passer-by removal takes two passes over the clip before the export, both through the same
 * decoder and OpenGL pipeline as the export itself, so their pixels and frame times agree with it:
 * 1. the people in the picture are found every [DETECT_INTERVAL] seconds on a small copy of the
 *    frame ([PeopleFinder]) and followed from one look to the next; somebody who never touches the
 *    plan's keep boxes, and is seen from where they come in to where they leave, may be taken out;
 * 2. if there is anybody like that, the frames at the plan's plate times are read back where they
 *    walk and reduced to their per-channel median, the plate. Each frame sits out the pixels inside
 *    its own moment's keep boxes; a pixel fewer than three frames speak for has no plate.
 * A person is then taken out only if the plate knows every pixel they ever cover and still shows
 * the scene around them as the frames do ([Passers.plateFits]): all of them for all of their time,
 * or not at all. With nobody to take out the picture is exported untouched.
 * Both results are kept for the life of this object, so changing a switch only exports again.
 *
 * Nothing is uploaded and no audio is written. Cancelling the calling coroutine stops the work.
 */
@OptIn(UnstableApi::class, ExperimentalApi::class)
internal class EditRenderer(
    context: Context,
    private val video: File,
    private val plan: EditPlan,
    private val newFinder: () -> PeopleFinder,
) {
    private val context = context.applicationContext
    private val handler = Handler(Looper.getMainLooper())

    private class Source(val width: Int, val height: Int, val duration: Double, val frameRate: Int)

    private var source: Source? = null
    /** Everybody the first pass found, followed through the clip; null until the clip has been looked through. */
    private var people: List<PersonTrack>? = null
    /** Those of them who may be taken out, if the plate turns out to be good for them. */
    private var candidates: List<PersonTrack> = emptyList()
    /** Whom the plate turned out to be good for: the people really taken out. Null until the plate has been tried. */
    private var removed: List<PersonTrack>? = null
    private var plate: Plate? = null

    /** How many people the last render took out. For tests. */
    val removedCount: Int get() = removed?.size ?: 0

    private var plateView: PlateView? = null
    private var scanned: Pair<Double, Double>? = null
    private val timings = LinkedHashMap<String, Long>()

    /**
     * What the passes before the export found, as plain text for a report on a render of a local
     * clip: everybody followed through the clip, whether they are taken out and why not, how much
     * of the plate is known, and how long each pass took. Nothing here is shown in the app.
     */
    fun report(): String = buildString {
        val keep = plan.keep
        val everybody = people
        if (keep == null || everybody == null) { appendLine("people: not looked for"); return@buildString }
        val (from, until) = scanned ?: (0.0 to 0.0)
        appendLine("people looked for from %.2f s to %.2f s; tracks: %d, candidates: %d, removed: %d".format(from, until, everybody.size, candidates.size, removedCount))
        for (track in everybody.sortedBy { it.start }) {
            val first = track.sightings.first().box
            val last = track.sightings.last().box
            appendLine(
                "  %.2f–%.2f s, %d sightings, from (%.2f, %.2f) to (%.2f, %.2f), about %.2f × %.2f of the frame: %s".format(
                    track.start, track.end, track.sightings.size, first.x + first.w / 2, first.y + first.h / 2, last.x + last.w / 2, last.y + last.h / 2,
                    track.sightings.map { it.box.w }.average(), track.sightings.map { it.box.h }.average(),
                    Passers.verdict(track, keep.samples, from, until, candidates, plateView, removed.orEmpty()),
                ),
            )
        }
        val made = plateStats
        appendLine(if (made == null) "plate: none made" else "plate: %d × %d px at (%d, %d) from the bottom-left, %d pictures, %.1f%% of its pixels known".format(made.width, made.height, made.left, made.bottom, made.pictures, made.known * 100))
        for ((pass, millis) in timings) appendLine("time, $pass: %.1f s".format(millis / 1000.0))
    }

    private class PlateStats(val left: Int, val bottom: Int, val width: Int, val height: Int, val pictures: Int, val known: Double)
    private var plateStats: PlateStats? = null

    private inline fun <T> timed(pass: String, work: () -> T): T {
        val started = System.nanoTime()
        try { return work() } finally { timings[pass] = (timings[pass] ?: 0L) + (System.nanoTime() - started) / 1_000_000 }
    }

    /** What the last export wrote: output size and how many frames. For tests. */
    var lastResult: ExportResult? = null
        private set

    /**
     * Writes the edit for these switches to [output] (an mp4, H.264, no audio). [progress] is
     * called on the main thread with 0…1 over everything this call has to do.
     */
    suspend fun render(options: EditOptions, output: File, progress: (Double) -> Unit) {
        val source = source ?: probe().also { source = it }
        val keep = plan.keep
        val remove = options.removePassersBy && keep != null
        val scanShare = if (remove && people == null) 0.45 else 0.0
        val plateShare = if (remove && removed == null) 0.15 else 0.0
        var look = EditLook()
        if (remove && keep != null) {
            val everybody = people ?: timed("looking for people") { scan(source, keep) { progress(it * scanShare) } }.also { people = it }
            val found = candidates
            if (removed == null) {
                var made: Plate? = null
                var fit = emptyList<PersonTrack>()
                if (found.isNotEmpty()) {
                    made = timed("plate") { plate(source, keep, found) { progress(scanShare + it * plateShare) } }
                    val (width, height) = RenderSize.of(source.width, source.height)
                    val scanWidth = RenderSize.of(source.width, source.height, SCAN_LONG_SIDE).first
                    val view = made?.let { PlateProbe(it, width, height, radius = maxOf(1, Math.round(1.5f * width / scanWidth))) }
                    plateView = view
                    if (view != null) fit = withContext(Dispatchers.Default) { Passers.removable(everybody, found, view, keep.samples) }
                }
                // Nobody the plate is good for: no plate is used and the picture is left as it is.
                plate = made.takeIf { fit.isNotEmpty() }
                removed = fit
            }
            look = EditLook(keep.samples, removed.orEmpty(), keep.feather, plate)
        }
        coroutineContext.ensureActive()
        val (width, height) = RenderSize.of(source.width, source.height)
        val scaled = width != source.width || height != source.height
        val clips = EditTimeline.clips(plan, options, source.duration)
        val items = clips.map { clip -> item(source, clip, listOfNotNull(presentation(width, height).takeIf { scaled }, EditFrameEffect(clip, look))) }
        output.parentFile?.mkdirs()
        output.delete()
        val done = scanShare + plateShare
        lastResult = timed("export") { run(analyze = false, items, output) { progress(done + it * (1 - done)) } }
        progress(1.0)
    }

    private suspend fun probe(): Source = withContext(Dispatchers.IO) {
        val info = VideoAnalyzer.probe(video)
        val duration = info.durationSeconds
        if (!info.hasVideo || duration == null || !(duration > 0)) throw EditSourceException()
        val (width, height) = FrameGeometry.displaySize(info.width, info.height, info.rotation)
        Source(width, height, duration, frameRate())
    }

    /** The clip's own frame rate, at most [MAX_FRAME_RATE]; fast segments are thinned to it. */
    private fun frameRate(): Int {
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(video.path)
            for (index in 0 until extractor.trackCount) {
                val format = extractor.getTrackFormat(index)
                if (format.getString(MediaFormat.KEY_MIME)?.startsWith("video/") != true || !format.containsKey(MediaFormat.KEY_FRAME_RATE)) continue
                val rate = runCatching { format.getInteger(MediaFormat.KEY_FRAME_RATE) }.getOrElse { format.getFloat(MediaFormat.KEY_FRAME_RATE).toInt() }
                if (rate > 0) return minOf(rate, MAX_FRAME_RATE)
            }
        } catch (error: Exception) {
            // not said: the common rate
        } finally {
            runCatching { extractor.release() }
        }
        return DEFAULT_FRAME_RATE
    }

    // Pass 1: who is in the picture, and who of them is not the climber.

    /** Everybody found, followed through the clip. Who of them may be taken out is left in [candidates]. */
    private suspend fun scan(source: Source, keep: EditKeep, progress: (Double) -> Unit): List<PersonTrack> {
        val start = minOf(plan.trim.first, plan.segments.minOf { it.t0 })
        val end = minOf(maxOf(plan.trim.second, plan.segments.maxOf { it.t1 }), source.duration)
        if (end - start < EditPlan.MIN_RANGE) return emptyList()
        scanned = start to end
        val clip = EditClip(start, end, 1.0, emptyList())
        val (width, height) = RenderSize.of(source.width, source.height, SCAN_LONG_SIDE)
        val found = Collections.synchronizedList(ArrayList<List<Sighting>>())
        val reader = FrameReader()
        // Made and used on the pipeline's GL thread, closed when the pass is over. The lock keeps a pass
        // that was cancelled from closing the detector under a frame that is still being looked at.
        val lock = Any()
        var finder: PeopleFinder? = null
        var over = false
        var next = Double.NEGATIVE_INFINITY
        val effect = EditFrameEffect(clip) { time, frameWidth, frameHeight ->
            if (time >= next) synchronized(lock) {
                if (over) return@synchronized
                next = time + DETECT_INTERVAL
                val detector = finder ?: newFinder().also { finder = it }
                val frame = reader.upright(frameWidth, frameHeight)
                // With each person, what the frame shows just around them: compared with the plate later.
                found += detector.find(frame).map { box -> Sighting(time, box, ring(frame, box)) }
            }
        }
        try {
            run(analyze = true, listOf(item(source, clip, listOf(presentation(width, height), effect))), scratch(), progress)
        } finally {
            synchronized(lock) {
                over = true
                runCatching { finder?.close() }
                reader.release()
            }
        }
        val everybody = Passers.link(found.toList())
        candidates = Passers.candidates(everybody, keep.samples, start, end)
        return everybody
    }

    /** The frame's colour at each of the ring points around a person, a small patch each; -1 outside the picture. */
    private fun ring(frame: Bitmap, box: Box): IntArray {
        val points = Passers.ringPoints(Passers.region(box))
        return IntArray(points.size) { index ->
            val x = (points[index].first * frame.width).toInt()
            val y = (points[index].second * frame.height).toInt()
            if (points[index].first < 0 || points[index].second < 0 || x >= frame.width || y >= frame.height) return@IntArray -1
            var red = 0; var green = 0; var blue = 0; var count = 0
            for (dy in -1..1) for (dx in -1..1) {
                val px = x + dx; val py = y + dy
                if (px < 0 || py < 0 || px >= frame.width || py >= frame.height) continue
                val colour = frame[px, py]
                red += colour shr 16 and 0xFF; green += colour shr 8 and 0xFF; blue += colour and 0xFF; count += 1
            }
            (red / count shl 16) or (green / count shl 8) or (blue / count)
        }
    }

    // Pass 2: the plate, for the part of the picture passers-by cross.

    private suspend fun plate(source: Source, keep: EditKeep, tracks: List<PersonTrack>, progress: (Double) -> Unit): Plate? {
        val extent = Passers.extent(tracks) ?: return null
        val (width, height) = RenderSize.of(source.width, source.height)
        val margin = keep.feather * minOf(width, height) + 8
        val left = floor(extent.x * width - margin).toInt().coerceIn(0, width - 1)
        val right = ceil(extent.right * width + margin).toInt().coerceIn(left + 1, width)
        val top = floor(extent.y * height - margin).toInt().coerceIn(0, height - 1)
        val low = ceil(extent.bottom * height + margin).toInt().coerceIn(top + 1, height)
        val regionWidth = right - left
        val regionHeight = low - top
        val bottom = height - low
        val frameBytes = regionWidth.toLong() * regionHeight * 3

        val work = File(EditFiles.work(context), UUID.randomUUID().toString()).apply { mkdirs() }
        try {
            // The frames wait on disk, not in memory. With little room, fewer of the plate's pictures are used.
            val storage = context.getSystemService(StorageManager::class.java)
            val free = runCatching { storage.getAllocatableBytes(storage.getUuidForPath(work)) }.getOrElse { work.usableSpace }
            val room = ((free - SPACE_MARGIN) / frameBytes).coerceIn(0, Int.MAX_VALUE.toLong()).toInt()
            val times = thinned(keep.plateTimes.map { it.coerceIn(0.0, maxOf(0.0, source.duration - PLATE_CLIP)) }.distinct(), room)
            // Too little room even for the fewest pictures a plate can be made of: nobody is taken out.
            if (times.size < EditPlan.MIN_PLATE_TIMES) return null
            val scaled = width != source.width || height != source.height
            val reader = FrameReader()
            val items = times.mapIndexed { index, time ->
                val clip = EditClip(time, minOf(time + PLATE_CLIP, source.duration), 1.0, emptyList())
                var taken = false
                // Only the first frame of each short clip is wanted: the one at the plate time.
                val effect = EditFrameEffect(clip) { _, _, _ ->
                    if (!taken) {
                        taken = true
                        reader.region(left, bottom, regionWidth, regionHeight, File(work, "$index.rgb"))
                    }
                }
                item(source, clip, listOfNotNull(presentation(width, height).takeIf { scaled }, effect))
            }
            try {
                run(analyze = true, items, scratch()) { progress(it * 0.7) }
            } finally {
                reader.release()
            }
            // Each picture with the keep boxes of its own moment: where the climber was, that picture says nothing.
            val frames = times.indices.map { File(work, "$it.rgb") to KeepPath.boxes(keep.samples, times[it]).map { box -> box.grown(Passers.PLATE_MARGIN, Passers.PLATE_MARGIN) } }
                .filter { it.first.length() == frameBytes }
            if (frames.size < EditPlan.MIN_PLATE_TIMES) return null
            val pixels = median(frames, left, bottom, regionWidth, regionHeight, width, height) { progress(0.7 + it * 0.3) }
            var known = 0
            for (pixel in 0 until regionWidth * regionHeight) if (pixels.get(pixel * 4 + 3).toInt() != 0) known += 1
            plateStats = PlateStats(left, bottom, regionWidth, regionHeight, frames.size, known.toDouble() / (regionWidth * regionHeight))
            return Plate(left, bottom, regionWidth, regionHeight, pixels)
        } finally {
            work.deleteRecursively()
        }
    }

    /**
     * Per-channel median of the stored frames, a strip of rows at a time, into one RGBA buffer with
     * alpha premultiplied: a pixel the pictures cannot vouch for is transparent and stays live.
     */
    private suspend fun median(
        frames: List<Pair<File, List<Box>>>, left: Int, bottom: Int, width: Int, height: Int, pictureWidth: Int, pictureHeight: Int, progress: (Double) -> Unit,
    ): ByteBuffer = withContext(Dispatchers.Default) {
        val rowBytes = width * 3
        val rows = maxOf(1, STRIP_BYTES / rowBytes)
        val strips = List(frames.size) { ByteArray(rows * rowBytes) }
        val masks = List(frames.size) { ByteArray(rows * width) }
        val masked = frames.any { it.second.isNotEmpty() }
        val merged = ByteArray(rows * rowBytes)
        val alpha = ByteArray(rows * width)
        val result = ByteBuffer.allocateDirect(width * height * 4).order(ByteOrder.nativeOrder())
        val rgba = ByteArray(rows * width * 4)
        val streams = frames.map { BufferedInputStream(FileInputStream(it.first), 1 shl 16) }
        try {
            var row = 0
            while (row < height) {
                ensureActive()
                val count = minOf(rows, height - row)
                for ((index, stream) in streams.withIndex()) {
                    readFully(stream, strips[index], count * rowBytes)
                    if (masked) PlateMedian.mark(masks[index], width, count, left, bottom + row, pictureWidth, pictureHeight, frames[index].second)
                }
                // Split across the cores: each part of the strip is independent of the others.
                val parts = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
                val pixels = count * width
                coroutineScope {
                    (0 until parts).map { part ->
                        async { PlateMedian.median(strips, masks.takeIf { masked }, merged, alpha, pixels * part / parts, pixels * (part + 1) / parts) }
                    }.awaitAll()
                }
                var out = 0
                for (pixel in 0 until pixels) {
                    rgba[out] = merged[pixel * 3]; rgba[out + 1] = merged[pixel * 3 + 1]; rgba[out + 2] = merged[pixel * 3 + 2]; rgba[out + 3] = alpha[pixel]
                    out += 4
                }
                result.put(rgba, 0, out)
                row += rows
                withContext(Dispatchers.Main) { progress(minOf(1.0, row.toDouble() / height)) }
            }
        } finally {
            streams.forEach { runCatching { it.close() } }
        }
        result.rewind()
        result
    }

    private fun readFully(stream: BufferedInputStream, target: ByteArray, count: Int) {
        var done = 0
        while (done < count) {
            val read = stream.read(target, done, count - done)
            if (read < 0) throw IOException("plate frame ended early")
            done += read
        }
    }

    // The pieces a pass is made of.

    private fun presentation(width: Int, height: Int): Effect = Presentation.createForWidthAndHeight(width, height, Presentation.LAYOUT_SCALE_TO_FIT)

    private fun item(source: Source, clip: EditClip, effects: List<Effect>): EditedMediaItem {
        val clipping = MediaItem.ClippingConfiguration.Builder()
            .setStartPositionUs((clip.t0 * 1_000_000).toLong())
            .setEndPositionUs(if (clip.t1 >= source.duration - 0.001) C.TIME_END_OF_SOURCE else (clip.t1 * 1_000_000).toLong())
            .build()
        val builder = EditedMediaItem.Builder(MediaItem.Builder().setUri(Uri.fromFile(video)).setClippingConfiguration(clipping).build())
            .setRemoveAudio(true)
            .setEffects(Effects(emptyList(), effects))
        if (clip.speed != 1.0) builder.setSpeed(ConstantSpeed(clip.speed.toFloat()))
        // A frame-rate limit is only for a clip played faster than it was shot, which would otherwise
        // come out at several times the clip's rate. At normal speed or slower every frame is kept:
        // the limiter drops a frame that arrives a fraction of a millisecond early, which ordinary
        // timestamps do all the time, and that would be a visible step after every cut.
        if (clip.speed > 1.0) builder.setFrameRate(source.frameRate)
        return builder.build()
    }

    /** Where a pass that writes nothing is told to write. */
    private fun scratch() = File(EditFiles.work(context), "none.mp4")

    /**
     * Runs [items] one after the other through Transformer and returns when it is done. With
     * [analyze] the frames are decoded and drawn but nothing is encoded or written. Transformer
     * lives on the main thread; its work is on threads of its own.
     */
    private suspend fun run(analyze: Boolean, items: List<EditedMediaItem>, output: File, progress: (Double) -> Unit): ExportResult =
        withContext(Dispatchers.Main.immediate) {
            val composition = Composition.Builder(EditedMediaItemSequence.Builder(items).build())
                // An HDR clip comes out as ordinary video, like everything else the app shows.
                .setHdrMode(Composition.HDR_MODE_TONE_MAP_HDR_TO_SDR_USING_OPEN_GL)
                .build()
            suspendCancellableCoroutine { continuation ->
                var transformer: Transformer? = null
                val holder = ProgressHolder()
                val poll = object : Runnable {
                    override fun run() {
                        val running = transformer ?: return
                        if (running.getProgress(holder) == Transformer.PROGRESS_STATE_AVAILABLE) progress(holder.progress / 100.0)
                        handler.postDelayed(this, PROGRESS_INTERVAL_MS)
                    }
                }
                val listener = object : Transformer.Listener {
                    override fun onCompleted(composition: Composition, exportResult: ExportResult) {
                        handler.removeCallbacks(poll)
                        if (continuation.isActive) continuation.resume(exportResult)
                    }

                    override fun onError(composition: Composition, exportResult: ExportResult, exportException: ExportException) {
                        handler.removeCallbacks(poll)
                        if (continuation.isActive) continuation.resumeWithException(exportException)
                    }
                }
                val base = Transformer.Builder(context)
                    .setVideoMimeType(MimeTypes.VIDEO_H264)
                    // No session is opened with the system's media metrics service for an export.
                    .setUsePlatformDiagnostics(false)
                    .addListener(listener)
                    .build()
                val created = if (analyze) ExperimentalAnalyzerModeFactory.buildAnalyzer(context, base) else base
                transformer = created
                continuation.invokeOnCancellation {
                    // Stopped before anything else of this renderer starts: at once when already on the main thread.
                    val stop = Runnable {
                        handler.removeCallbacks(poll)
                        runCatching { created.cancel() }
                    }
                    if (Looper.myLooper() == Looper.getMainLooper()) stop.run() else handler.post(stop)
                }
                created.start(composition, output.path)
                handler.postDelayed(poll, PROGRESS_INTERVAL_MS)
            }
        }

    private class ConstantSpeed(private val speed: Float) : SpeedProvider {
        override fun getSpeed(timeUs: Long): Float = speed
        override fun getNextSpeedChangeTimeUs(timeUs: Long): Long = C.TIME_UNSET
        override fun equals(other: Any?): Boolean = other is ConstantSpeed && other.speed == speed
        override fun hashCode(): Int = speed.hashCode()
    }

    internal companion object {
        const val MAX_FRAME_RATE = 60
        const val DEFAULT_FRAME_RATE = 30
        /** People are looked for on a copy of the frame this long on its long side. */
        const val SCAN_LONG_SIDE = 640
        /** Seconds of clip between two looks for people: every third or fourth frame of a 30 fps clip. */
        const val DETECT_INTERVAL = 0.12
        /** Each plate picture is taken from a clip this short; only its first frame is used. */
        const val PLATE_CLIP = 0.25
        private const val STRIP_BYTES = 512 * 1024
        private const val SPACE_MARGIN = 64L * 1024 * 1024
        private const val PROGRESS_INTERVAL_MS = 200L

        /** What went wrong, as the system named it. It has no translation and is shown as it is. */
        fun reason(error: Throwable): String = (error as? ExportException)?.errorCodeName ?: error.localizedMessage ?: error.javaClass.simpleName

        /** At most [count] of [times], spread evenly over them. */
        fun thinned(times: List<Double>, count: Int): List<Double> {
            if (count >= times.size) return times
            if (count <= 0) return emptyList()
            if (count == 1) return listOf(times[times.size / 2])
            return List(count) { index -> times[index * (times.size - 1) / (count - 1)] }
        }
    }
}

/** The plate as the removal rule sees it: whether it knows a region, and what colour it has at a point. */
private class PlateProbe(private val plate: Plate, private val width: Int, private val height: Int, private val radius: Int) : PlateView {
    /** Alpha of the plate at a picture pixel (x from the left, y from the top); -1 outside the plate. */
    private fun offset(x: Int, y: Int): Int {
        val column = x - plate.left
        val row = (height - 1 - y) - plate.bottom
        return if (column < 0 || row < 0 || column >= plate.width || row >= plate.height) -1 else (row * plate.width + column) * 4
    }

    override fun covers(region: Box): Boolean {
        val x0 = floor(region.x * width).toInt().coerceIn(0, width - 1)
        val x1 = ceil(region.right * width).toInt().coerceIn(x0 + 1, width)
        val y0 = floor(region.y * height).toInt().coerceIn(0, height - 1)
        val y1 = ceil(region.bottom * height).toInt().coerceIn(y0 + 1, height)
        if (offset(x0, y0) < 0 || offset(x1 - 1, y1 - 1) < 0) return false
        for (y in y0 until y1) for (x in x0 until x1) if (plate.pixels.get(offset(x, y) + 3).toInt() == 0) return false
        return true
    }

    override fun colour(x: Double, y: Double): Int? {
        if (x < 0 || y < 0 || x >= 1 || y >= 1) return null
        val cx = (x * width).toInt()
        val cy = (y * height).toInt()
        var red = 0; var green = 0; var blue = 0; var count = 0; var missing = 0
        for (py in cy - radius..cy + radius) for (px in cx - radius..cx + radius) {
            if (px < 0 || py < 0 || px >= width || py >= height) continue
            val at = offset(px, py)
            if (at < 0 || plate.pixels.get(at + 3).toInt() == 0) { missing += 1; continue }
            red += plate.pixels.get(at).toInt() and 0xFF; green += plate.pixels.get(at + 1).toInt() and 0xFF; blue += plate.pixels.get(at + 2).toInt() and 0xFF
            count += 1
        }
        // A point the plate mostly does not know cannot vouch for anything.
        if (count == 0 || missing > count) return null
        return (red / count shl 16) or (green / count shl 8) or (blue / count)
    }
}

/**
 * Reads back what was just drawn into the bound framebuffer. GL thread only. Rows come out of
 * OpenGL bottom first; [upright] turns them over for a picture, [region] writes them as they are.
 */
private class FrameReader {
    private var buffer: ByteBuffer? = null
    private var bitmap: Bitmap? = null
    private var flipped: ByteBuffer? = null
    private var row = ByteArray(0)
    private var packed = ByteArray(0)

    private fun read(left: Int, bottom: Int, width: Int, height: Int): ByteBuffer {
        val size = width * height * 4
        val target = buffer?.takeIf { it.capacity() >= size } ?: ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder()).also { buffer = it }
        target.clear()
        GLES20.glPixelStorei(GLES20.GL_PACK_ALIGNMENT, 1)
        GLES20.glReadPixels(left, bottom, width, height, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, target)
        target.position(0).limit(size)
        return target
    }

    /** The whole frame as an upright bitmap, reused from call to call. */
    fun upright(width: Int, height: Int): Bitmap {
        val pixels = read(0, 0, width, height)
        val stride = width * 4
        if (row.size != stride) row = ByteArray(stride)
        val turned = flipped?.takeIf { it.capacity() == stride * height } ?: ByteBuffer.allocateDirect(stride * height).also { flipped = it }
        for (line in 0 until height) {
            pixels.position(line * stride)
            pixels.get(row, 0, stride)
            turned.position((height - 1 - line) * stride)
            turned.put(row, 0, stride)
        }
        turned.rewind()
        val target = bitmap?.takeIf { it.width == width && it.height == height }
            ?: createBitmap(width, height).also { bitmap?.recycle(); bitmap = it }
        target.copyPixelsFromBuffer(turned)
        return target
    }

    /** A part of the frame, three bytes a pixel, rows bottom first, into [file]. */
    fun region(left: Int, bottom: Int, width: Int, height: Int, file: File) {
        val pixels = read(left, bottom, width, height)
        val stride = width * 4
        if (row.size != stride) row = ByteArray(stride)
        if (packed.size != width * 3) packed = ByteArray(width * 3)
        FileOutputStream(file).buffered(1 shl 16).use { output ->
            for (line in 0 until height) {
                pixels.position(line * stride)
                pixels.get(row, 0, stride)
                var out = 0
                for (position in 0 until stride step 4) {
                    packed[out] = row[position]; packed[out + 1] = row[position + 1]; packed[out + 2] = row[position + 2]
                    out += 3
                }
                output.write(packed, 0, out)
            }
        }
    }

    fun release() {
        bitmap?.recycle()
        bitmap = null
        buffer = null
        flipped = null
    }
}
