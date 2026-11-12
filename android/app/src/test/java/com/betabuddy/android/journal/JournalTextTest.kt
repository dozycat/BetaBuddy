package com.betabuddy.android.journal

import com.betabuddy.android.data.Climb
import com.betabuddy.android.data.Moment
import com.betabuddy.android.data.PoseAnalysis
import com.betabuddy.android.data.UNGRADED_VALUE
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.util.Locale
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class JournalTextTest {
    private fun climb(title: String, grade: String, sent: Boolean) = Climb(
        id = title, date = 0.0, title = title, grade = grade, sent = sent, note = "", videoName = "v", thumbnailName = "t",
        analysis = PoseAnalysis(emptyList(), 0, 1.0, 1.0, 0.0, emptyList()),
    )
    private val climbs = listOf(climb("Purple Slab", "V3", true), climb("周日的紫色线路", "V10", false), climb("warm-up", UNGRADED_VALUE, false))
    private val english: (String) -> String = { if (it == UNGRADED_VALUE) "Ungraded" else it }

    @Test fun filtersBySendStatus() {
        assertEquals(3, filterClimbs(climbs, JournalFilter.ALL, "", english).size)
        assertEquals(listOf("Purple Slab"), filterClimbs(climbs, JournalFilter.SENT, "", english).map { it.title })
        assertEquals(2, filterClimbs(climbs, JournalFilter.PROJECTING, "", english).size)
    }

    @Test fun searchesNameOrDisplayedGrade() {
        assertEquals(listOf("Purple Slab"), filterClimbs(climbs, JournalFilter.ALL, "purple", english).map { it.title })
        assertEquals(listOf("周日的紫色线路"), filterClimbs(climbs, JournalFilter.ALL, "紫色", english).map { it.title })
        assertEquals(listOf("Purple Slab"), filterClimbs(climbs, JournalFilter.ALL, "v3", english).map { it.title })
        assertEquals(listOf("周日的紫色线路"), filterClimbs(climbs, JournalFilter.ALL, "V1", english).map { it.title })
        // The grade is matched as it is displayed in the current language, not as stored.
        assertEquals(listOf("warm-up"), filterClimbs(climbs, JournalFilter.ALL, " ungraded ", english).map { it.title })
        assertTrue(filterClimbs(climbs, JournalFilter.ALL, "未标级", english).isEmpty())
        assertTrue(filterClimbs(climbs, JournalFilter.SENT, "warm", english).isEmpty())
    }

    @Test fun percentFollowsTheLocale() {
        assertEquals("56%", percentText(0.556, Locale.US))
        assertEquals("56%", percentText(0.556, Locale.SIMPLIFIED_CHINESE))
        assertEquals("0%", percentText(Double.NaN, Locale.US))
        assertEquals("100%", percentText(1.0, Locale.KOREA))
        assertTrue(percentText(0.5, Locale.GERMANY).startsWith("50")) // "50 %": the locale decides the spacing
    }

    @Test fun cueKindsResolveForOldAndNewRecords() {
        assertEquals(Moment.FAST_MOVE, Moment(time = 1.0, title = "stored", detail = "d", kind = Moment.FAST_MOVE).resolvedKind())
        assertEquals("futureKind", Moment(time = 1.0, title = "stored", detail = "d", kind = "futureKind").resolvedKind())
        assertEquals(Moment.RAISED_HEEL, Moment(time = 1.0, title = "高抬脚 · 候选片段", detail = "d").resolvedKind())
        assertEquals(Moment.FAST_MOVE, Moment(time = 1.0, title = "快速移动 · 回看一下", detail = "d").resolvedKind())
        assertNull(Moment(time = 1.0, title = "something else", detail = "d").resolvedKind())
    }

    @Test fun reportLayoutMatchesIos() {
        assertEquals(
            "BetaBuddy · Purple Slab\nV3 · Sent · 00:42\nHips found in 56% of sampled frames\nkeep hips in\n\ncaveat",
            reportText("Purple Slab", "V3", "Sent", "00:42", "Hips found in 56% of sampled frames", "keep hips in", "caveat"),
        )
    }

    @Test fun copyStopsAsSoonAsTheLimitIsExceeded() {
        val data = ByteArray(1000) { it.toByte() }
        val exact = ByteArrayOutputStream()
        assertEquals(1000L, copyLimited(ByteArrayInputStream(data), exact, 1000))
        assertArrayEquals(data, exact.toByteArray())
        try { copyLimited(ByteArrayInputStream(data), ByteArrayOutputStream(), 999); fail() } catch (expected: TooLargeException) { }
        try { copyLimited(ByteArrayInputStream(data), ByteArrayOutputStream(), 5000) { false }; fail() } catch (expected: InterruptedException) { }
        assertEquals(0L, copyLimited(ByteArrayInputStream(ByteArray(0)), ByteArrayOutputStream(), 10))
    }
}
