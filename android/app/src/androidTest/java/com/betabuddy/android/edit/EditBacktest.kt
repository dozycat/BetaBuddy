package com.betabuddy.android.edit

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.betabuddy.android.pose.VideoAnalyzer
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * An optional run of the automatic edit on a clip of your own, with the bundled people detector:
 * skipped unless the instrumentation is told which clip and which plan. Nothing about the clip is in
 * the repository; `android/scripts/edit-backtest.sh <video> <plan.json> <out>` puts both into the
 * app's cache (`cache/edit-backtest/`), runs this, and brings back what it wrote to `out/` there:
 *
 * - `full.mp4`: every switch on;
 * - `plain.mp4`: every switch off (the attempt, as it is);
 * - `removal-only.mp4`: passer-by removal alone, so each frame can be laid next to the source's;
 * - `report.txt`: who was found and followed, who was taken out or left and why, how much of the
 *   plate is known, and how long each pass took.
 *
 * It checks only that the renders come out. Whether the result is any good is for a person to look at.
 */
@RunWith(AndroidJUnit4::class)
class EditBacktest {
    @Test fun rendersALocalClipWithTheBundledDetector() = runBlocking {
        val arguments = InstrumentationRegistry.getArguments()
        val clipName = arguments.getString("editClip")
        val planName = arguments.getString("editPlan")
        assumeTrue("no clip given: -e editClip <file> -e editPlan <file>, both in the app's cache/edit-backtest/", clipName != null && planName != null)
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val directory = File(context.cacheDir, "edit-backtest")
        val clip = File(directory, File(clipName!!).name)
        val planFile = File(directory, File(planName!!).name)
        assertTrue("missing ${clip.path}", clip.isFile)
        assertTrue("missing ${planFile.path}", planFile.isFile)
        val plan = EditPlan.parse(planFile.readText())
        assertNotNull("the plan is not a usable version-1 plan", plan)
        plan!!
        val out = File(directory, "out").apply { deleteRecursively(); mkdirs() }

        val info = VideoAnalyzer.probe(clip)
        val report = StringBuilder()
        report.appendLine("clip: ${info.width} × ${info.height}, turned ${info.rotation}°, ${"%.2f".format(info.durationSeconds ?: 0.0)} s, ${clip.length() / 1024} KB")
        report.appendLine(
            "plan: made for ${"%.2f".format(plan.duration)} s, attempt ${"%.2f".format(plan.trim.first)}–${"%.2f".format(plan.trim.second)} s, ${plan.segments.size} segments " +
                "(speeds ${plan.segments.map { it.speed }.distinct().sorted().joinToString { "%.2f".format(it) }}), camera keys ${plan.camera.size}" +
                (plan.keep?.let { ", keep samples ${it.samples.size}, plate pictures ${it.plateTimes.size}, feather ${it.feather}" } ?: ", no keep boxes (removal not offered)"),
        )
        report.appendLine("device: ${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}, Android ${android.os.Build.VERSION.RELEASE}, ${Runtime.getRuntime().availableProcessors()} cores")

        val renderer = EditRenderer(context, clip, plan) { ModelPeopleFinder(context) }
        suspend fun render(name: String, options: EditOptions) {
            val target = File(out, name)
            val started = System.nanoTime()
            renderer.render(options, target) { }
            val seconds = (System.nanoTime() - started) / 1e9
            val result = renderer.lastResult!!
            assertTrue("$name is empty", target.length() > 1000)
            report.appendLine(
                "$name (${listOfNotNull("removal".takeIf { options.removePassersBy }, "slow motion".takeIf { options.slowMotion }, "camera".takeIf { options.cameraMoves }).ifEmpty { listOf("nothing") }.joinToString(" + ")}): " +
                    "${"%.1f".format(seconds)} s to make, ${result.width} × ${result.height}, ${result.videoFrameCount} frames, ${"%.2f".format(result.durationMs / 1000.0)} s long, ${target.length() / 1024} KB, " +
                    "${renderer.removedCount.takeIf { options.removePassersBy } ?: 0} people taken out",
            )
        }
        // The first render that removes anybody also looks for people and makes the plate; its time includes both.
        render("full.mp4", EditOptions())
        render("plain.mp4", EditOptions(removePassersBy = false, slowMotion = false, cameraMoves = false))
        render("removal-only.mp4", EditOptions(removePassersBy = true, slowMotion = false, cameraMoves = false))
        report.appendLine()
        report.append(renderer.report())
        File(out, "report.txt").writeText(report.toString())
    }
}
