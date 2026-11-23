package com.betabuddy.android.capture

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import com.betabuddy.android.gym.GymManifest
import com.betabuddy.android.gym.GymPack
import com.betabuddy.android.gym.PackWebView
import com.betabuddy.android.journal.DirectoryGym
import com.betabuddy.android.journal.GymDirectory
import java.io.ByteArrayInputStream
import java.io.File
import java.io.InputStream

/**
 * A synthetic gym pack for device tests: an entry page that only says it is ready, and stand-ins
 * for the capture and analysis pages that answer from a script (the same two pages the iOS tests
 * use, see ios/TestSupport). No venue, route or algorithm is in it.
 */
internal class StandInPack(
    testContext: Context, capture: Boolean = true, analysis: Boolean = true, step: Int = 5, directoryIds: Boolean = true,
    /** The capture page loads but never says it is ready, like one whose script did not come up. */
    silent: Boolean = false,
) : PackWebView.Files {
    private val pages = buildMap {
        put("index.html", "<!doctype html><meta charset=\"utf-8\"><title>synthetic pack</title><p>synthetic pack</p><script>window.BetaBuddyGym={openWall(){},openRoute(){},openVenue(){},state(){return {ready:true}}};try{window.webkit.messageHandlers.gym.postMessage({type:'ready',packVersion:'$VERSION',routeCount:0})}catch(e){}</script>")
        // A real pack's pages load their code from a file of its own, and so do these: the stand-in's
        // script is split off into one. (An older WebView can only inject the message shim once the
        // page has started loading; an inline script would speak before it is there.)
        fun page(name: String, script: String, change: (String) -> String = { it }) {
            val text = testContext.assets.open(name).use { it.readBytes().toString(Charsets.UTF_8) }
            val code = text.substringAfter("<script>").substringBeforeLast("</script>")
            check(code.isNotBlank() && code != text)
            put(script, change(code))
            put(name.replace("-standin", ""), "<!doctype html><meta charset=\"utf-8\"><title>stand-in</title><script src=\"$script\"></script>")
        }
        // `step` is how many frames each scripted state of the capture stand-in lasts.
        if (capture) page("capture-standin.html", "capture.js") { code ->
            if (silent) "/* says nothing */" else code.replace("const STEP = 5;", "const STEP = $step;").also { check("const STEP = $step;" in it) }
        }
        if (analysis) page("analysis-standin.html", "analysis.js")
    }

    val pack = GymPack(
        GymManifest.parse(
            """{"schema":1,"packVersion":"$VERSION","generatedAt":"2026-01-01T00:00:00Z","gymName":"合成岩馆","routeCount":0,"entry":"index.html",
               "files":[${pages.keys.sorted().joinToString(",") { """{"path":"$it","sha256":"00"}""" }}],
               "venues":[{"id":"$VENUE_ID","files":[],"name":"合成岩馆 抱石 东门店","names":{"en":"Synthetic Gym East"}${if (directoryIds) ""","directoryIds":["syn:east"]""" else ""}}]}""",
        ),
    )

    override fun open(path: String): InputStream? = pages[path]?.let { ByteArrayInputStream(it.toByteArray(Charsets.UTF_8)) }

    companion object {
        const val VERSION = "0.0.0-standin"
        const val VENUE_ID = "synthetic-east"
        const val DIRECTORY_GYM = "合成岩馆（东门）"

        /** The made-up wall the capture stand-in reports: colour, hold id, x, y, normalised to the picture. */
        val holds = listOf(
            Hold("yellow", "y1", 0.30, 0.80), Hold("yellow", "y2", 0.42, 0.78), Hold("yellow", "y3", 0.36, 0.90), Hold("yellow", "y4", 0.50, 0.60), Hold("yellow", "y5", 0.40, 0.42), Hold("yellow", "y6", 0.55, 0.22),
            Hold("blue", "b1", 0.70, 0.82), Hold("blue", "b2", 0.78, 0.66), Hold("blue", "b3", 0.66, 0.50), Hold("blue", "b4", 0.80, 0.36), Hold("blue", "b5", 0.72, 0.18),
            Hold("red", "r1", 0.15, 0.70), Hold("red", "r2", 0.20, 0.50), Hold("red", "r3", 0.12, 0.30),
        )
        val colors = mapOf("yellow" to Color.rgb(226, 196, 40), "blue" to Color.rgb(40, 110, 220), "red" to Color.rgb(214, 52, 44))

        /** A synthetic gym directory with the coordinate fields, around a made-up position. */
        const val LATITUDE = 30.0
        const val LONGITUDE = 120.0
        val directory: List<DirectoryGym> = GymDirectory.parse(
            """{"version":1,"updatedAt":"2026-01-01","gyms":[
             {"id":"syn:east","name":"$DIRECTORY_GYM","province":"合成省","city":"合成市","district":"东区","address":"东门路 1 号","sourceURL":"https://example.invalid/","lat":30.0018,"lon":120.0,"coordPrecision":"poi"},
             {"id":"syn:station","name":"只知道最近车站的岩馆","province":"合成省","city":"合成市","district":"北区","address":"","sourceURL":"https://example.invalid/","lat":30.0115,"lon":120.0,"coordPrecision":"station"},
             {"id":"syn:west","name":"合成岩馆（西门）","province":"合成省","city":"合成市","district":"西区","address":"西门路 2 号","sourceURL":"https://example.invalid/","lat":30.0,"lon":119.95,"coordPrecision":"address"},
             {"id":"syn:district","name":"只知道在哪个区的岩馆","province":"合成省","city":"合成市","district":"南区","address":"","sourceURL":"https://example.invalid/","lat":29.99,"lon":120.0,"coordPrecision":"district"},
             {"id":"syn:city","name":"只知道在哪个市的岩馆","province":"合成省","city":"合成市","address":"","sourceURL":"https://example.invalid/","lat":30.02,"lon":120.02,"coordPrecision":"city"},
             {"id":"syn:far","name":"很远的岩馆","province":"远方省","city":"远方市","address":"","sourceURL":"https://example.invalid/","lat":40.0,"lon":116.0,"coordPrecision":"poi"},
             {"id":"syn:nowhere","name":"没有坐标的岩馆","province":"合成省","city":"合成市","address":"","sourceURL":"https://example.invalid/"}
            ]}""",
        )
    }

    data class Hold(val color: String, val id: String, val x: Double, val y: Double)
}

/**
 * A drawn climbing wall as a short video, standing in for the camera: a pale wall with panel
 * seams, coloured blobs where [StandInPack.holds] says, a mat along the bottom and a dot that
 * moves so that frames differ. Nothing in it comes from a real gym. With [photo], a picture of a
 * person is put in front of the wall so that pose recognition has somebody to find.
 */
internal object SyntheticWallClip {
    fun draw(canvas: Canvas, width: Int, height: Int, frame: Int, photo: Bitmap? = null) {
        val w = width.toFloat(); val h = height.toFloat()
        val paint = Paint(Paint.ANTI_ALIAS_FLAG)
        canvas.drawColor(Color.rgb(219, 214, 204))
        paint.color = Color.rgb(178, 173, 163); paint.strokeWidth = maxOf(1f, w / 270)
        for (column in 1 until 4) canvas.drawLine(w * column / 4, 0f, w * column / 4, h, paint)
        for (row in 1 until 6) canvas.drawLine(0f, h * row / 6, w, h * row / 6, paint)
        paint.color = Color.rgb(51, 61, 82)
        canvas.drawRect(0f, h * 0.94f, w, h, paint)
        if (photo != null) {
            val scale = minOf(w / photo.width, h * 0.8f / photo.height)
            val pw = photo.width * scale; val ph = photo.height * scale
            canvas.drawBitmap(photo, null, RectF((w - pw) / 2, h * 0.92f - ph, (w + pw) / 2, h * 0.92f), paint)
        }
        for (hold in StandInPack.holds) {
            paint.color = StandInPack.colors.getValue(hold.color)
            canvas.drawOval(RectF(((hold.x - 0.035) * w).toFloat(), ((hold.y - 0.02) * h).toFloat(), ((hold.x + 0.035) * w).toFloat(), ((hold.y + 0.02) * h).toFloat()), paint)
        }
        paint.color = Color.WHITE
        val swing = (frame % 60) / 60f
        canvas.drawCircle((0.05f + 0.9f * swing) * w, h * 0.975f, w * 0.01f, paint)
    }

    /** Writes the clip and returns it. Frames are stamped by the clock, so this takes [seconds] to run. */
    fun make(target: File, width: Int = 540, height: Int = 960, seconds: Double = 3.0, photo: Bitmap? = null): File {
        val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
            setInteger(MediaFormat.KEY_BIT_RATE, 4_000_000)
            setInteger(MediaFormat.KEY_FRAME_RATE, 30)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        }
        val encoder = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
        encoder.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        val surface = encoder.createInputSurface()
        encoder.start()
        val muxer = MediaMuxer(target.path, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        val info = MediaCodec.BufferInfo()
        var track = -1
        var firstTime = -1L
        fun drain(untilEnd: Boolean) {
            while (true) {
                val index = encoder.dequeueOutputBuffer(info, if (untilEnd) 10_000 else 0)
                when {
                    index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> { track = muxer.addTrack(encoder.outputFormat); muxer.start() }
                    index >= 0 -> {
                        val data = encoder.getOutputBuffer(index)!!
                        if (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0 && info.size > 0 && track >= 0) {
                            if (firstTime < 0) firstTime = info.presentationTimeUs
                            info.presentationTimeUs -= firstTime
                            muxer.writeSampleData(track, data, info)
                        }
                        val end = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                        encoder.releaseOutputBuffer(index, false)
                        if (end) return
                    }
                    !untilEnd -> return
                }
            }
        }
        repeat((seconds * 30).toInt()) { frame ->
            val canvas = surface.lockCanvas(Rect(0, 0, width, height))
            draw(canvas, width, height, frame, photo)
            surface.unlockCanvasAndPost(canvas)
            Thread.sleep(33)
            drain(untilEnd = false)
        }
        encoder.signalEndOfInputStream()
        drain(untilEnd = true)
        encoder.stop(); encoder.release(); surface.release()
        muxer.stop(); muxer.release()
        return target
    }
}
