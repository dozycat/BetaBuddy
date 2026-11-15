package com.betabuddy.android.journal

import java.time.Instant
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class RecordingDateTest {
    @Test fun readsOffsetsFractionalSecondsAndCompactDates() {
        val expected = Instant.parse("2024-03-02T06:35:00Z").toEpochMilli()
        assertEquals(expected, RecordingDate.parse("2024-03-02T14:35:00+08:00"))
        assertEquals(expected, RecordingDate.parse("20240302T063500.000Z"))
        assertEquals(expected, RecordingDate.parse("20240302T143500+0800"))
        assertEquals(expected + 123, RecordingDate.parse("2024-03-02T06:35:00.123Z"))
    }

    @Test fun missingInvalidAndAmbiguousTimesStayUnset() {
        listOf(null, "", "not a date", "2024-03-02T14:35:00", "2024-13-40T00:00:00Z", "20240230T000000Z", "1904-01-01T00:00:00Z").forEach {
            assertNull(RecordingDate.parse(it))
        }
    }
}
