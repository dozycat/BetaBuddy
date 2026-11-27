package com.betabuddy.android.journal

import com.betabuddy.android.data.Climb
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId

/** Derived grouping: changing a record's date or gym moves it without touching user folders. */
internal data class JournalDayGroup(val day: LocalDate, val gym: String, val environment: String, val climbs: List<Climb>) {
    val gymName: String? get() = climbs.firstNotNullOfOrNull { it.gymName?.takeIf(String::isNotBlank) }

    companion object {
        fun make(climbs: List<Climb>, zone: ZoneId = ZoneId.systemDefault()): List<JournalDayGroup> = climbs.groupBy {
            val setting = when {
                it.environment == "outdoor" -> "outdoor"
                it.gymKey.isNotEmpty() -> "indoor"
                it.environment == "indoor" -> "indoor"
                else -> "unknown"
            }
            Triple(Instant.ofEpochMilli(it.climbedAtMillis).atZone(zone).toLocalDate(), if (setting == "outdoor") "" else it.gymKey, setting)
        }.map { (key, values) ->
            JournalDayGroup(key.first, key.second, key.third, values.sortedWith(compareByDescending<Climb> { it.climbedAtMillis }.thenBy { it.id }))
        }.sortedWith(compareByDescending<JournalDayGroup> { it.day }.thenByDescending { it.climbs.first().climbedAtMillis }.thenBy { it.climbs.first().id })
    }
}
