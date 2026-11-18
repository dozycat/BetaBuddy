package com.betabuddy.android.review

import com.betabuddy.android.data.Joint
import com.betabuddy.android.data.PoseFrame
import com.betabuddy.android.gym.GymMessage
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

// What crosses between the app and the pack's analysis page. All fixtures are synthetic; the
// result text is opaque to the app and is only ever stored and handed back.
class AnalysisBridgeTest {
    private val prefix = "window.BetaBuddyAnalysis && typeof window.BetaBuddyAnalysis."

    /** The argument list of a generated call, e.g. `("a",1)` → `"a",1`. */
    private fun arguments(script: String, function: String): String {
        val open = "(window.BetaBuddyAnalysis.$function("
        assertTrue(script, script.startsWith("$prefix$function === 'function' ? $open"))
        assertTrue(script, script.endsWith("), true) : false"))
        return script.substring(script.indexOf(open) + open.length, script.length - "), true) : false".length)
    }

    private fun message(type: String, build: kotlinx.serialization.json.JsonObjectBuilder.() -> Unit = {}) =
        buildJsonObject { put("type", type); build() }.toString()

    @Test fun pageMessagesParse() {
        assertEquals(AnalysisMessage.Ready, AnalysisMessage.parse("""{"type":"analysisReady","packVersion":"0.0.1"}"""))
        assertEquals(AnalysisMessage.Seek(12.5), AnalysisMessage.parse("""{"type":"analysisSeek","t":12.5}"""))
        assertEquals(AnalysisMessage.Seek(0.0), AnalysisMessage.parse("""{"type":"analysisSeek","t":-3}"""))
        assertEquals(AnalysisMessage.Error("some_code"), AnalysisMessage.parse("""{"type":"analysisError","code":"some_code","message":"shown by the page"}"""))
        assertEquals(AnalysisMessage.Error(""), AnalysisMessage.parse("""{"type":"analysisError"}"""))
    }

    @Test fun resultIsKeptExactlyAsThePageSentIt() {
        // The page's own format: opaque here. Quotes, backslashes, line ends and non-ASCII must come through untouched.
        val result = """{"v":1,"text":"引号 \" 反斜杠 \\ 换行 \n  ","plate":"data:image/jpeg;base64,AAAA"}"""
        val parsed = AnalysisMessage.parse(message("analysisResult") {
            put("id", "ABC")
            put("json", result)
            put("climber", buildJsonObject { put("height", 172); put("weight", 60.5) })
            put("somethingNew", true)
        }) as AnalysisMessage.Result
        assertEquals("ABC", parsed.id)
        assertEquals(result, parsed.json)
        assertEquals("""{"height":172,"weight":60.5}""", parsed.climber)
        // After showing a saved result the page no longer names the record, and it may send no measurements.
        val bare = AnalysisMessage.parse(message("analysisResult") { put("json", "{}") }) as AnalysisMessage.Result
        assertEquals("", bare.id)
        assertNull(bare.climber)
        // A result as large as a real one (a couple of hundred kilobytes) is accepted.
        val large = "x".repeat(400_000)
        assertEquals(large, (AnalysisMessage.parse(message("analysisResult") { put("json", large) }) as AnalysisMessage.Result).json)
    }

    @Test fun progressUnknownAndMalformedMessagesAreIgnored() {
        for (ignored in listOf(
            null, "", "analysisReady", "[]", """{"type":7}""", """{"type":"somethingNew","x":1}""",
            """{"type":"analysisProgress","stage":"s","fraction":0.4}""",     // the page draws its own progress
            """{"type":"analysisResult"}""", """{"type":"analysisResult","json":""}""", """{"type":"analysisResult","json":{"v":1}}""",
            """{"type":"analysisSeek"}""", """{"type":"analysisSeek","t":"3"}""", """{"type":"analysisSeek","t":1e999}""",
            // The gym page's own messages are not this page's.
            """{"type":"ready","packVersion":"1","routeCount":3}""", """{"type":"routeSelected","id":"r1"}""", """{"type":"openSettings"}""",
        )) assertNull(ignored, AnalysisMessage.parse(ignored))
    }

    @Test fun gymMessagesAreUntouchedByTheAnalysisTypes() {
        for (type in listOf("analysisReady", "analysisProgress", "analysisResult", "analysisSeek", "analysisError")) {
            assertNull(type, GymMessage.parse(message(type) { put("json", "{}"); put("t", 1) }))
        }
        assertEquals(GymMessage.Ready, GymMessage.parse("""{"type":"ready","packVersion":"1","routeCount":3}"""))
    }

    @Test fun beginCarriesTheRecordTheClipSizeAndEarlierMeasurements() {
        val meta = Json.parseToJsonElement(arguments(AnalysisScript.begin("ID-1", 1080, 1920, """{"height":172,"weight":60}"""), "begin")).jsonObject
        assertEquals(setOf("id", "w", "h", "climber"), meta.keys)
        assertEquals("ID-1", meta.getValue("id").jsonPrimitive.content)
        assertEquals(1080, meta.getValue("w").jsonPrimitive.int)
        assertEquals(1920, meta.getValue("h").jsonPrimitive.int)
        assertEquals(172, meta.getValue("climber").jsonObject.getValue("height").jsonPrimitive.int)
        // No earlier result, or something unusable in the preferences: the field is left out.
        for (climber in listOf(null, "", "not json", "[1,2]", "\"text\"")) {
            val bare = Json.parseToJsonElement(arguments(AnalysisScript.begin("ID-1", 640, 480, climber), "begin")).jsonObject
            assertEquals(climber, setOf("id", "w", "h"), bare.keys)
        }
    }

    @Test fun stringsAreEncodedNotQuotedByHand() {
        val hostile = "\"); alert(1); (\"\\ \n     </script> 引号"
        // An id that tries to break out of the call stays inside one JSON string.
        val begin = arguments(AnalysisScript.begin(hostile, 1, 1, null), "begin")
        assertEquals(hostile, Json.parseToJsonElement(begin).jsonObject.getValue("id").jsonPrimitive.content)
        // A saved result is handed back as one string literal holding exactly the stored text.
        val show = arguments(AnalysisScript.show(hostile), "show")
        assertEquals(hostile, (Json.parseToJsonElement(show) as JsonPrimitive).content)
        for (script in listOf(begin, show, AnalysisScript.addPoses(hostile))) {
            assertFalse(script.contains('\n'))
            assertFalse(script.contains(' '))
            assertFalse(script.contains(' '))
        }
        assertTrue(show.contains("\\u2028") && show.contains("\\u2029"))
    }

    @Test fun framesAndRunAreCalledAsTheContractNamesThem() {
        assertEquals("3,1.25,\"QUJD\"", arguments(AnalysisScript.addFrame(3, 1.25, "QUJD"), "addFrame"))
        assertEquals("0,0.0,\"\"", arguments(AnalysisScript.addFrame(0, Double.NaN, ""), "addFrame"))
        assertEquals("", arguments(AnalysisScript.run(), "run"))
    }

    @Test fun pictureTimesAreSpreadEvenlyOverTheClip() {
        val times = AnalysisFeed.frameTimes(120.0)
        assertEquals(60, times.size)
        assertEquals(1.0, times.first(), 1e-9)
        assertEquals(119.0, times.last(), 1e-9)
        assertTrue(times.zipWithNext().all { (a, b) -> Math.abs(b - a - 2.0) < 1e-9 })
        // A short clip gives fewer pictures rather than the same one many times.
        val short = AnalysisFeed.frameTimes(1.0)
        assertEquals(10, short.size)
        assertEquals(0.05, short.first(), 1e-9)
        assertEquals(0.95, short.last(), 1e-9)
        assertEquals(5, AnalysisFeed.frameTimes(0.5).size)
        assertEquals(1, AnalysisFeed.frameTimes(0.05).size)
        assertTrue(AnalysisFeed.frameTimes(6.0).all { it > 0 && it < 6.0 })
        for (unusable in listOf(0.0, -1.0, Double.NaN, Double.POSITIVE_INFINITY)) assertTrue(AnalysisFeed.frameTimes(unusable).isEmpty())
    }

    @Test fun picturesAreScaledToTheWidthThePageWorksAt() {
        assertEquals(540 to 960, AnalysisFeed.frameSize(1080, 1920))
        assertEquals(960 to 540, AnalysisFeed.frameSize(1920, 1080)) // the shorter side is bounded
        assertEquals(360 to 640, AnalysisFeed.frameSize(360, 640)) // never upscaled
        assertEquals(1 to 1, AnalysisFeed.frameSize(1, 1))
    }

    @Test fun posesAreHandedOverInBatchesWithBothLandmarkSets() {
        fun frame(time: Double, world: Boolean = true) = PoseFrame(
            time, List(33) { Joint(0.25 + it / 100.0, 0.5, 0.9) },
            if (world) List(33) { listOf(it / 10.0, -0.5, 0.125, 0.75) } else null,
        )
        val frames = (0 until 95).map { frame(it / 10.0) } + frame(9.5, world = false) + frame(9.6)
        val batches = AnalysisFeed.poseBatches(frames)
        assertEquals(listOf(40, 40, 16), batches.map { Json.parseToJsonElement(it).jsonArray.size })
        val first = Json.parseToJsonElement(batches[0]).jsonArray[1].jsonObject
        assertEquals(setOf("t", "img", "lm"), first.keys)
        assertEquals(0.1, first.getValue("t").jsonPrimitive.double, 0.0)
        val img = first.getValue("img").jsonArray
        val lm = first.getValue("lm").jsonArray
        assertEquals(33, img.size)
        assertEquals(33, lm.size)
        // The third number is the landmarker's own visibility (kept with the world landmark), not the stored confidence.
        assertEquals(listOf(0.27, 0.5, 0.75), (img[2] as JsonArray).map { it.jsonPrimitive.double })
        assertEquals(listOf(0.2, -0.5, 0.125, 0.75), (lm[2] as JsonArray).map { it.jsonPrimitive.double })
        // The sample saved without world landmarks is left out; the one after it is still there.
        val last = Json.parseToJsonElement(batches[2]).jsonArray
        assertEquals(listOf(9.4, 9.6), last.takeLast(2).map { (it as JsonObject).getValue("t").jsonPrimitive.double })
        // Records from before world landmarks were kept give the page nothing.
        assertTrue(AnalysisFeed.poseBatches(listOf(frame(0.0, world = false))).isEmpty())
        assertTrue(AnalysisFeed.poseBatches(emptyList()).isEmpty())
        // Every batch survives the trip into the page as one string argument.
        val call = AnalysisScript.addPoses(batches[2])
        assertEquals(batches[2], (Json.parseToJsonElement(arguments(call, "addPoses")) as JsonPrimitive).content)
    }
}
