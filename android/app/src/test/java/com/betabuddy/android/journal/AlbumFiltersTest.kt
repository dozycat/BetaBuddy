package com.betabuddy.android.journal

import com.betabuddy.android.data.Climb
import com.betabuddy.android.data.PoseAnalysis
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import org.junit.Assert.*
import org.junit.Test

class AlbumFiltersTest {
    private val zone = ZoneId.of("Asia/Shanghai")
    private val climb = Climb("id", Climb.dateFromMillis(Instant.parse("2026-11-13T16:00:00Z").toEpochMilli()), "project", "V3", false, "", "v", "t", PoseAnalysis(emptyList(), 0, 1.0, 1.0, 0.0, emptyList()))

    @Test fun datesAreInclusiveInLocalTimeAndOldRecordsFallBackToImportDate() {
        val day = LocalDate.of(2026, 11, 14)
        assertTrue(AlbumFilters(from = day, through = day).matches(climb, zone))
        assertFalse(AlbumFilters(through = day.minusDays(1)).matches(climb, zone))
        assertFalse(AlbumFilters(from = day.plusDays(1)).matches(climb, zone))
        val edited = climb.copy(climbedAt = Climb.dateFromMillis(Instant.parse("2026-11-10T12:00:00Z").toEpochMilli()))
        assertFalse(AlbumFilters(from = day, through = day).matches(edited, zone))
    }

    @Test fun gymBranchesUseIdsAndFiltersCombine() {
        val one = climb.copy(gymId = "branch-1", gymName = "Synthetic gym", folderId = "folder")
        val two = one.copy(gymId = "branch-2")
        val filters = AlbumFilters(folderId = "folder", gymKey = one.gymKey, grade = "V3")
        assertTrue(filters.matches(one, zone))
        assertFalse(filters.matches(two, zone))
        assertFalse(filters.copy(grade = "V4").matches(one, zone))
        assertFalse(filters.copy(folderId = "").matches(one, zone))
        assertTrue(AlbumFilters(folderId = "", gymKey = "").matches(climb, zone))
        assertEquals("name:Custom", climb.copy(gymName = " Custom ").gymKey)
        assertEquals(listOf(one), filterClimbs(listOf(one), JournalFilter.PROJECTING, "synthetic") { it })
    }
}
