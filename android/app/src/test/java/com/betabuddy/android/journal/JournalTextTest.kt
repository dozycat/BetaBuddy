package com.betabuddy.android.journal

import com.betabuddy.android.data.Climb
import com.betabuddy.android.data.PoseAnalysis
import com.betabuddy.android.data.UNGRADED_VALUE
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class JournalTextTest {
    private fun climb(title: String, grade: String, sent: Boolean) = Climb(
        id = title, date = 0.0, title = title, grade = grade, sent = sent, note = "", videoName = "v", thumbnailName = "t",
        analysis = PoseAnalysis(emptyList(), 0, 1.0, 1.0),
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

    @Test fun reportIsWhatTheUserRecorded() {
        assertEquals(
            "BetaBuddy · Purple Slab\nV3 · Sent · 00:42\nkeep hips in",
            reportText("Purple Slab", "V3", "Sent", "00:42", "keep hips in"),
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
