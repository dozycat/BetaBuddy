package com.betabuddy.android.journal

import com.betabuddy.android.data.Climb
import com.betabuddy.android.data.PoseAnalysis
import java.time.LocalDate
import java.time.ZoneId
import org.junit.Assert.*
import org.junit.Test

class JournalDayGroupTest {
    private fun climb(id: String, millis: Long) = Climb(id, Climb.dateFromMillis(millis), "", "V3", false, "", "v", "t", PoseAnalysis(emptyList(), 0, 1.0, 1.0))

    @Test fun captureDateAndGymIdDetermineGroupsAndCorrectionsMoveRecords() {
        val zone = ZoneId.of("Asia/Shanghai")
        val day = LocalDate.of(2026, 11, 28)
        val time = day.atStartOfDay(zone).toInstant().toEpochMilli()
        val first = climb("a", time + 864_000_000).copy(climbedAt = Climb.dateFromMillis(time + 3_600_000), gymId = "fixture-a", gymName = "Fixture")
        val second = climb("b", time + 7_200_000).copy(gymId = "fixture-a", environment = "indoor")
        val branch = second.copy(id = "c", gymId = "fixture-b")
        val outdoor = second.copy(id = "d", environment = "outdoor")
        val groups = JournalDayGroup.make(listOf(first, second, branch, outdoor), zone)
        assertEquals(3, groups.size)
        assertTrue(groups.all { it.day == day })
        assertEquals(listOf("b", "a"), groups.first { it.gym == "id:fixture-a" }.climbs.map { it.id })
        assertEquals("", groups.first { it.environment == "outdoor" }.gym)
        assertEquals(2, JournalDayGroup.make(listOf(first, second.copy(climbedAt = Climb.dateFromMillis(time - 1))), zone).size)
    }

    @Test fun twentyFiveHourDayIsOneGroupAndUnknownIsNotOutdoor() {
        val zone = ZoneId.of("America/Los_Angeles")
        val day = LocalDate.of(2026, 11, 1)
        val start = day.atStartOfDay(zone).toInstant().toEpochMilli()
        val next = day.plusDays(1).atStartOfDay(zone).toInstant().toEpochMilli()
        assertEquals(25 * 3_600_000L, next - start)
        val early = climb("a", start)
        val late = climb("b", next - 1)
        assertEquals(1, JournalDayGroup.make(listOf(early, late), zone).size)
        assertEquals(2, JournalDayGroup.make(listOf(early, late.copy(environment = "outdoor")), zone).size)
        assertEquals(2, JournalDayGroup.make(listOf(late, climb("c", next)), zone).size)
    }
}
