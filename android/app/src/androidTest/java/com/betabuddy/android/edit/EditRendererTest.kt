package com.betabuddy.android.edit

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.media.MediaMetadataRetriever
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.betabuddy.android.TestClips
import com.betabuddy.android.pose.VideoAnalyzer
import java.io.File
import kotlin.math.abs
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The automatic edit rendered on a device, from a made-up plan and a drawn clip: a grey wall with
 * a blue bar along its top, a green square going up it (the "climber") and a magenta square
 * walking along the bottom (the "passer-by"). Nothing in it comes from a gym or from the pack.
 *
 * What is checked is the app's side: that the output shows, at every moment, the source moment
 * the plan asks for (through speed changes and a replay), the right way up, through the plan's
 * camera window, and with the passer-by gone while the climber stays. The people finder here is a
 * stand-in that finds the two squares by their colour; the bundled model has a test of its own.
 */
@RunWith(AndroidJUnit4::class)
class EditRendererTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext
    private val created = mutableListOf<File>()

    @After fun cleanUp() { created.forEach { it.delete() } }

    private fun file(name: String) = File(context.cacheDir, "edit-test-$name-${System.nanoTime()}.mp4").also { created += it }

    /** Three seconds of the drawn clip. With [rotation] the frames are stored turned and carry the hint to turn them back. */
    private fun clip(rotation: Int = 0, lingering: Boolean = false, standing: Boolean = false, bystander: Boolean = false): File {
        val turned = rotation == 90
        val width = if (turned) HEIGHT else WIDTH
        val height = if (turned) WIDTH else HEIGHT
        return TestClips.encode(file("source"), width, height, frames = 90, rotation = rotation) { canvas, frame ->
            if (turned) {
                // Drawn so that a player that honours the hint shows the same upright picture.
                canvas.save(); canvas.rotate(-90f); canvas.translate(-WIDTH.toFloat(), 0f)
                draw(canvas, frame / 90.0, lingering, standing, bystander)
                canvas.restore()
            } else {
                draw(canvas, frame / 90.0, lingering, standing, bystander)
            }
        }
    }

    /**
     * With [lingering] the climber stands at the bottom for most of the clip, then goes up fast, and
     * the passer-by walks in only afterwards, across the spot the climber stood on. With [standing]
     * the climber stays at the bottom throughout, in the passer-by's way. With [bystander] a small
     * yellow square paces up and down on the passer-by's path: somebody else, further away, who is
     * not the climber and never stands still long enough to be part of the plate.
     */
    private fun draw(canvas: Canvas, progress: Double, lingering: Boolean = false, standing: Boolean = false, bystander: Boolean = false) {
        val w = WIDTH.toFloat(); val h = HEIGHT.toFloat()
        val paint = Paint()
        canvas.drawColor(Color.rgb(190, 190, 190))
        paint.color = Color.rgb(150, 150, 150)
        for (row in 1 until 6) canvas.drawRect(0f, h * row / 6, w, h * row / 6 + 2, paint)
        paint.color = BLUE
        canvas.drawRect(0f, 0f, w, h * 0.04f, paint)
        val climberY = if (standing) 0.87 else if (lingering) 0.87 - 0.57 * ((progress - 0.60) / 0.08).coerceIn(0.0, 1.0) else 0.78 - 0.56 * progress
        paint.color = GREEN
        canvas.drawRect(RectF(w * 0.43f, (h * (climberY - 0.06)).toFloat(), w * 0.57f, (h * (climberY + 0.06)).toFloat()), paint)
        if (bystander) {
            val turn = (progress * 2) % 1.0
            val y = 0.81 + 0.12 * (if (turn < 0.5) turn * 2 else 2 - turn * 2)
            paint.color = YELLOW
            canvas.drawRect(RectF(w * 0.73f, (h * (y - 0.025)).toFloat(), w * 0.77f, (h * (y + 0.025)).toFloat()), paint)
        }
        val passerX = if (lingering) -0.1 + 1.2 * (progress - 0.55) / 0.45 else 0.10 + 0.80 * progress
        paint.color = MAGENTA
        canvas.drawRect(RectF((w * (passerX - 0.05)).toFloat(), h * 0.80f, (w * (passerX + 0.05)).toFloat(), h * 0.94f), paint)
    }

    /** Finds the squares by colour, as boxes in fractions of the picture. Upside-down or mistimed frames give wrong boxes. */
    private class ColourFinder(
        /** Whether the yellow square is noticed on a given call: a detector does not see everybody all the time. */
        private val seesBystander: (call: Int) -> Boolean = { true },
    ) : PeopleFinder {
        var calls = 0
        var closed = false
        override fun find(frame: Bitmap): List<Box> {
            calls += 1
            return listOfNotNull(blob(frame, ::isGreen), blob(frame, ::isMagenta), blob(frame, ::isYellow).takeIf { seesBystander(calls) })
        }
        override fun close() { closed = true }
    }

    private fun frameAt(video: File, time: Double): Bitmap {
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(video.path)
            return retriever.getFrameAtTime((time * 1_000_000).toLong(), MediaMetadataRetriever.OPTION_CLOSEST) ?: throw AssertionError("no frame at $time in ${video.name}")
        } finally {
            retriever.release()
        }
    }

    private fun duration(video: File): Double = VideoAnalyzer.probe(video).durationSeconds!!

    /** A plan around where the green square really is in [source]: slow, fast and replayed ranges, a camera, keep boxes, plate times. */
    private fun plan(source: File, withCamera: Boolean = false): EditPlan {
        val duration = duration(source)
        val keep = (0..((duration - 0.05) * 10).toInt()).mapNotNull { index ->
            val time = index / 10.0
            blob(frameAt(source, time), ::isGreen)?.let { KeepSample(time, listOf(it.grown(0.03, 0.03))) }
        }
        assertTrue("the climber is found in the source: ${keep.size}", keep.size > 20)
        val replayCamera = listOf(CameraKey(1.0, 0.5, 0.5, 1.0))
        return EditPlan(
            duration = duration, trim = 0.2 to 2.8,
            segments = listOf(
                EditSegment(0.2, 1.2, 1.0), EditSegment(1.2, 1.8, 0.5), EditSegment(1.8, 2.8, 2.0),
                EditSegment(1.0, 1.6, 0.5, replayCamera.takeIf { withCamera }),
            ),
            camera = if (withCamera) listOf(CameraKey(0.2, 0.5, 0.5, 1.0), CameraKey(1.5, 0.5, 0.4, 2.0)) else emptyList(),
            keep = EditKeep(0.03, keep, (1..9).map { it * duration / 10 }),
        )
    }

    private fun render(source: File, plan: EditPlan, options: EditOptions, finder: PeopleFinder = ColourFinder()): Pair<File, EditRenderer> = runBlocking {
        val output = file("output")
        val renderer = EditRenderer(context, source, plan) { finder }
        var last = 0.0
        renderer.render(options, output) { assertTrue("progress goes forward: $it after $last", it >= last - 1e-6); last = it }
        assertEquals(1.0, last, 1e-6)
        output to renderer
    }

    private fun assertNear(message: String, expected: Double, actual: Double, tolerance: Double) =
        assertTrue("$message: expected $expected, got $actual", abs(expected - actual) <= tolerance)

    /** At sampled output moments the output shows the climber where the source has it at the plan's source moment. */
    private fun assertFollowsThePlan(source: File, output: File, clips: List<EditClip>, passerGone: Boolean) {
        val length = EditTimeline.outputDuration(clips)
        assertNear("output duration", length, duration(output), 0.25)
        val marker = blob(frameAt(source, 1.0), ::isBlue)!!
        var checked = 0
        var time = 0.15
        while (time < length - 0.15) {
            val (index, sourceTime) = EditTimeline.sourceTime(clips, time)
            // Close to a cut the nearest frame may belong to either side.
            val start = EditTimeline.outputStart(clips, index)
            if (time - start > 0.12 && start + clips[index].outputDuration - time > 0.12) {
                val shown = frameAt(output, time)
                val expected = blob(frameAt(source, sourceTime), ::isGreen)!!
                val climber = blob(shown, ::isGreen) ?: throw AssertionError("no climber in the output at $time (clip $index, source $sourceTime)")
                assertNear("climber at output $time (clip $index, source $sourceTime)", expected.y + expected.h / 2, climber.y + climber.h / 2, 0.035)
                assertNear("climber across at $time", 0.5, climber.x + climber.w / 2, 0.03)
                // The right way up: the bar is where the source has it.
                val bar = blob(shown, ::isBlue) ?: throw AssertionError("no bar at $time")
                assertNear("bar at $time", marker.y + marker.h / 2, bar.y + bar.h / 2, 0.02)
                assertEquals("passer-by at output $time (source $sourceTime)", passerGone, blob(shown, ::isMagenta) == null)
                checked += 1
            }
            time += 0.23
        }
        assertTrue("enough moments checked: $checked over $length s", checked >= (length / 0.23 * 0.6).toInt())
    }

    @Test fun theOutputFollowsThePlanThroughSpeedChangesAndAReplayWithThePasserByGone() {
        val source = clip()
        val plan = plan(source)
        val finder = ColourFinder()
        val (output, renderer) = render(source, plan, EditOptions(), finder)
        val result = renderer.lastResult!!
        assertEquals(WIDTH, minOf(result.width, result.height)); assertEquals(HEIGHT, maxOf(result.width, result.height))
        assertTrue("people were looked for every few frames, not on every one: ${finder.calls}", finder.calls in 12..40)
        assertTrue(finder.closed)
        assertEquals("the passer-by, and not the climber", 1, renderer.removedCount)
        assertFollowsThePlan(source, output, EditTimeline.clips(plan, EditOptions(), duration(source)), passerGone = true)
        // No audio track is written.
        val retriever = MediaMetadataRetriever()
        retriever.setDataSource(output.path)
        assertNull(retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_HAS_AUDIO))
        retriever.release()

        // The same renderer with a switch changed: nothing is looked for again, only exported.
        val before = finder.calls
        val plain = file("plain")
        runBlocking { renderer.render(EditOptions(slowMotion = false), plain) { } }
        assertEquals(before, finder.calls)
        assertFollowsThePlan(source, plain, EditTimeline.clips(plan, EditOptions(slowMotion = false), duration(source)), passerGone = true)
    }

    @Test fun whereTheClimberLingeredThePlateIsTheWallNotTheClimber() {
        val source = clip(lingering = true)
        val length = duration(source)
        val keep = plan(source).keep!!
        val plan = EditPlan(length, 0.1 to length - 0.1, listOf(EditSegment(0.1, length - 0.1, 1.0)), emptyList(), keep)
        val (output, renderer) = render(source, plan, EditOptions())
        assertEquals(1, renderer.removedCount)
        // While the passer-by is over the spot the climber stood on for most of the clip: in the source
        // they are there, in the output there is wall: no passer-by, and no climber left behind either.
        var checked = 0
        for (step in 0 until 12) {
            val time = length * (0.70 + 0.02 * step)
            val original = frameAt(source, time)
            val passer = blob(original, ::isMagenta) ?: continue
            if (passer.right < 0.40 || passer.x > 0.60) continue
            val shown = frameAt(output, time - 0.1)
            assertNull("passer-by at $time", blob(shown, ::isMagenta))
            val climber = blob(shown, ::isGreen) ?: throw AssertionError("no climber at $time")
            assertTrue("no climber left standing at the bottom at $time: $climber", climber.bottom < 0.5)
            checked += 1
        }
        assertTrue("moments checked: $checked", checked >= 3)
        // Before the passer-by comes the climber is simply where they are.
        assertTrue(blob(frameAt(output, length * 0.3), ::isGreen)!!.y > 0.7)
    }

    @Test fun aPasserByWhoWalksThroughTheClimberIsLeftAloneFromStartToFinish() {
        val source = clip(standing = true)
        val length = duration(source)
        val keep = plan(source).keep!!
        val plan = EditPlan(length, 0.1 to length - 0.1, listOf(EditSegment(0.1, length - 0.1, 1.0)), emptyList(), keep)
        val (output, renderer) = render(source, plan, EditOptions())
        assertEquals("nobody can be taken out cleanly, so nobody is", 0, renderer.removedCount)
        // All or nothing: the passer-by is in every frame, not only while they overlap the climber, and whole.
        var checked = 0
        var time = 0.3
        while (time < length - 0.4) {
            val original = blob(frameAt(source, time + 0.1), ::isMagenta)!!
            val shown = blob(frameAt(output, time), ::isMagenta) ?: throw AssertionError("the passer-by is missing at $time")
            assertNear("passer-by's width at $time", original.w, shown.w, 0.03)
            assertNear("passer-by's height at $time", original.h, shown.h, 0.03)
            checked += 1
            time += 0.2
        }
        assertTrue("moments checked: $checked", checked >= 8)
    }

    @Test fun aPasserByWhoCrossesSomebodyWhoStaysIsLeftAloneAndSoIsThatSomebody() {
        val source = clip(bystander = true)
        val length = duration(source)
        val keep = plan(source).keep!!
        val plan = EditPlan(length, 0.1 to length - 0.1, listOf(EditSegment(0.1, length - 0.1, 1.0)), emptyList(), keep)
        // The bystander is noticed only for a stretch in the middle, so they are nobody the edit may take
        // out. The passer-by, clean in every other respect, walks right over them towards the end: taking
        // the passer-by out would take the bystander out with them for those frames.
        val (output, renderer) = render(source, plan, EditOptions(), ColourFinder(seesBystander = { call -> call in 6..14 }))
        assertEquals("nobody is taken out", 0, renderer.removedCount)
        var checked = 0
        var time = 0.3
        while (time < length - 0.4) {
            val original = frameAt(source, time + 0.1)
            val shown = frameAt(output, time)
            for ((name, colour) in listOf<Pair<String, (Int) -> Boolean>>("bystander" to ::isYellow, "passer-by" to ::isMagenta)) {
                // For a moment the passer-by is in front of the bystander and hides them in the clip itself.
                val before = blob(original, colour) ?: continue
                // Right where one walks in front of the other, the two files' frames can be one frame apart:
                // gone in the clip a frame either side means hidden, not taken out.
                val after = blob(shown, colour)
                    ?: if (listOf(-0.04, 0.04).any { blob(frameAt(source, time + 0.1 + it), colour) == null }) continue
                    else throw AssertionError("the $name is missing at $time")
                assertNear("$name's width at $time", before.w, after.w, 0.012)
                assertNear("$name's height at $time", before.h, after.h, 0.012)
            }
            checked += 1
            time += 0.1
        }
        assertTrue("moments checked: $checked", checked >= 16)
    }

    @Test fun atNormalSpeedEveryFrameIsKeptAndOnlyFasterClipsAreThinned() {
        val source = clip()
        val length = duration(source)
        fun sourceFrames(from: Double, until: Double): Int {
            val extractor = android.media.MediaExtractor()
            extractor.setDataSource(source.path)
            extractor.selectTrack(0)
            var count = 0
            while (true) {
                val time = extractor.sampleTime
                if (time < 0) break
                if (time / 1_000_000.0 >= from && time / 1_000_000.0 < until) count += 1
                extractor.advance()
            }
            extractor.release()
            return count
        }
        fun rendered(segments: List<EditSegment>): Int {
            val plan = EditPlan(length, 0.2 to 2.6, segments, emptyList(), null)
            return render(source, plan, EditOptions()).second.lastResult!!.videoFrameCount
        }
        // Three cuts at normal speed: no frame is lost at any of them.
        val whole = sourceFrames(0.2, 2.6)
        val cut = rendered(listOf(EditSegment(0.2, 1.0, 1.0), EditSegment(1.0, 1.8, 1.0), EditSegment(1.8, 2.6, 1.0)))
        android.util.Log.i("edit-test", "frames: source $whole, three cuts at normal speed $cut")
        assertTrue("frames at normal speed: $cut of $whole", cut >= whole - 1 && cut <= whole + 1)
        // Slowed down: still every frame.
        val slow = rendered(listOf(EditSegment(0.2, 2.6, 0.5)))
        assertTrue("frames in slow motion: $slow of $whole", slow >= whole - 1 && slow <= whole + 1)
        // Twice as fast: about half of them, not a clip at twice the frame rate.
        val fast = rendered(listOf(EditSegment(0.2, 2.6, 2.0)))
        assertTrue("frames at double speed: $fast of $whole", fast <= whole * 0.7 && fast >= whole * 0.3)
    }

    @Test fun withRemovalOffEverybodyStaysAndNobodyIsLookedFor() {
        val source = clip()
        val plan = plan(source)
        val finder = ColourFinder()
        val options = EditOptions(removePassersBy = false)
        val (output, _) = render(source, plan, options, finder)
        assertEquals(0, finder.calls)
        assertFollowsThePlan(source, output, EditTimeline.clips(plan, options, duration(source)), passerGone = false)
    }

    @Test fun aClipStoredTurnedComesOutUprightAndInStep() {
        val source = clip(rotation = 90)
        val upright = frameAt(source, 1.0)
        assertTrue("the fixture is shown upright: ${upright.width}x${upright.height}", upright.height > upright.width)
        val plan = plan(source)
        val (output, _) = render(source, plan, EditOptions())
        val shown = frameAt(output, 1.0)
        assertTrue("${shown.width}x${shown.height}", shown.height > shown.width)
        assertFollowsThePlan(source, output, EditTimeline.clips(plan, EditOptions(), duration(source)), passerGone = true)
    }

    @Test fun theCameraWindowZoomsAndMovesAsThePlanSays() {
        val source = clip()
        val plan = plan(source, withCamera = true)
        val options = EditOptions(removePassersBy = false, slowMotion = false)
        val (output, _) = render(source, plan, options)
        val clips = EditTimeline.clips(plan, options, duration(source))
        for (time in listOf(0.3, 1.0, 1.6, 2.1)) {
            val (index, sourceTime) = EditTimeline.sourceTime(clips, time)
            val window = CameraPath.window(clips[index].camera, sourceTime)
            val original = blob(frameAt(source, sourceTime), ::isGreen)!!
            val shown = blob(frameAt(output, time), ::isGreen) ?: throw AssertionError("no climber at $time")
            assertNear("climber down at $time (zoom ${window.s})", (original.y + original.h / 2 - window.cy) * window.s + 0.5, shown.y + shown.h / 2, 0.05)
            assertNear("climber across at $time", (original.x + original.w / 2 - window.cx) * window.s + 0.5, shown.x + shown.w / 2, 0.04)
            assertNear("climber's size at $time (zoom ${window.s})", original.w * window.s, shown.w, 0.05)
        }
        // By two seconds in the window has closed to half the frame: the bar along the top is out of it.
        assertNull(blob(frameAt(output, 2.1), ::isBlue))
        assertNotNull(blob(frameAt(output, 0.1), ::isBlue))

        // Camera moves off: the full frame again.
        val (still, _) = render(source, plan, options.copy(cameraMoves = false))
        assertNotNull(blob(frameAt(still, 2.1), ::isBlue))
    }

    @Test fun cancellingStopsTheWorkAndTheRendererCanBeUsedAgain() = runBlocking {
        val source = clip()
        val plan = plan(source)
        val finder = ColourFinder()
        val renderer = EditRenderer(context, source, plan) { finder }
        val output = file("cancelled")
        val job = async {
            try { renderer.render(EditOptions(), output) { }; false } catch (error: CancellationException) { true }
        }
        delay(400)
        job.cancel()
        try { job.await(); fail("not cancelled") } catch (expected: CancellationException) { }
        // Nothing is left running: a new render with the same renderer goes through.
        val again = file("again")
        renderer.render(EditOptions(), again) { }
        assertFollowsThePlan(source, again, EditTimeline.clips(plan, EditOptions(), duration(source)), passerGone = true)
    }

    @Test fun aFileThatIsNotAVideoIsRefused() = runBlocking {
        val broken = file("broken").apply { writeText("not a video") }
        val plan = EditPlan(3.0, 0.0 to 3.0, listOf(EditSegment(0.0, 3.0, 1.0)), emptyList(), null)
        try {
            EditRenderer(context, broken, plan) { ColourFinder() }.render(EditOptions(), file("never")) { }
            fail("rendered a text file")
        } catch (expected: EditSourceException) { }
    }

    /**
     * The bundled detector on a drawn frame: Google's MediaPipe sample picture of a person, small, in
     * front of a plain wall. Not a climb and not a measure of how well passers-by are found.
     */
    @Test fun theBundledModelFindsAPersonInADrawnFrame() {
        val photo = instrumentation.context.assets.open("pose.jpg").use { BitmapFactory.decodeStream(it) }
        val frame = Bitmap.createBitmap(540, 960, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(frame)
        canvas.drawColor(Color.rgb(190, 190, 190))
        val empty = frame.copy(Bitmap.Config.ARGB_8888, false)
        // A third of the frame's height, off to the left.
        val height = 320f
        val width = height * photo.width / photo.height
        val placed = RectF(60f, 520f, 60f + width, 520f + height)
        canvas.drawBitmap(photo, null, placed, Paint(Paint.FILTER_BITMAP_FLAG))
        val expected = Box(placed.left / 540.0, placed.top / 960.0, placed.width() / 540.0, placed.height() / 960.0)
        ModelPeopleFinder(context).use { finder ->
            val started = System.nanoTime()
            val people = finder.find(frame)
            android.util.Log.i("edit-test", "people model: ${(System.nanoTime() - started) / 1_000_000} ms, found $people, expected within $expected")
            assertTrue("a person is found where the picture is: $people", people.any { it.intersects(expected) && it.overlap(expected) > 0.2 })
            assertTrue("nobody on an empty wall: ${finder.find(empty)}", finder.find(empty).isEmpty())
        }
    }

    private companion object {
        const val WIDTH = 360
        const val HEIGHT = 640
        val GREEN = Color.rgb(20, 200, 40)
        val MAGENTA = Color.rgb(220, 30, 220)
        val BLUE = Color.rgb(30, 40, 220)
        val YELLOW = Color.rgb(235, 200, 30)

        fun isGreen(color: Int) = Color.green(color) > 140 && Color.red(color) < 110 && Color.blue(color) < 110
        fun isMagenta(color: Int) = Color.red(color) > 150 && Color.blue(color) > 150 && Color.green(color) < 110
        fun isYellow(color: Int) = Color.red(color) > 170 && Color.green(color) > 140 && Color.blue(color) < 110
        fun isBlue(color: Int) = Color.blue(color) > 150 && Color.red(color) < 110 && Color.green(color) < 110

        /** The box around every pixel of one colour, in fractions of the picture; null when there are too few to be a square. */
        fun blob(frame: Bitmap, matches: (Int) -> Boolean): Box? {
            val pixels = IntArray(frame.width * frame.height)
            frame.getPixels(pixels, 0, frame.width, 0, 0, frame.width, frame.height)
            var left = Int.MAX_VALUE; var top = Int.MAX_VALUE; var right = -1; var bottom = -1; var count = 0
            for (y in 0 until frame.height) for (x in 0 until frame.width) {
                if (matches(pixels[y * frame.width + x])) {
                    count += 1
                    if (x < left) left = x; if (x > right) right = x; if (y < top) top = y; if (y > bottom) bottom = y
                }
            }
            if (count < frame.width * frame.height / 1500) return null
            return Box(left.toDouble() / frame.width, top.toDouble() / frame.height, (right - left + 1.0) / frame.width, (bottom - top + 1.0) / frame.height)
        }
    }
}
