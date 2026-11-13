package com.betabuddy.android.journal

import com.betabuddy.android.data.Climb
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId

/** Empty folder key means unfiled; null means all. Dates are inclusive calendar dates. */
data class AlbumFilters(
    val folderId: String? = null,
    val gymKey: String? = null,
    val grade: String? = null,
    val from: LocalDate? = null,
    val through: LocalDate? = null,
) {
    val active: Boolean get() = folderId != null || gymKey != null || grade != null || from != null || through != null
    fun matches(climb: Climb, zone: ZoneId = ZoneId.systemDefault()): Boolean {
        val day = Instant.ofEpochMilli(climb.climbedAtMillis).atZone(zone).toLocalDate()
        return (folderId == null || (climb.folderId ?: "") == folderId) &&
            (gymKey == null || climb.gymKey == gymKey) &&
            (grade == null || climb.grade == grade) &&
            (from == null || day >= from) && (through == null || day <= through)
    }
}

/** Stable directory IDs distinguish branches with the same display name. */
val Climb.gymKey: String get() = gymId?.takeIf { it.isNotBlank() }?.let { "id:$it" }
    ?: gymName?.trim()?.takeIf { it.isNotEmpty() }?.let { "name:$it" } ?: ""
