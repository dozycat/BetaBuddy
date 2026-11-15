package com.betabuddy.android.journal

import java.time.Instant
import java.time.OffsetDateTime
import java.time.format.DateTimeFormatter

/** ISO metadata and Android's compact UTC representation; timezone-less values are ambiguous. */
internal object RecordingDate {
    fun parse(value: String?): Long? {
        val text = value?.trim()?.takeIf { it.isNotEmpty() } ?: return null
        val parsed = runCatching { Instant.parse(text).toEpochMilli() }.getOrNull()
            ?: runCatching { OffsetDateTime.parse(text).toInstant().toEpochMilli() }.getOrNull()
            ?: listOf("uuuuMMdd'T'HHmmss.SSSXX", "uuuuMMdd'T'HHmmssXX").firstNotNullOfOrNull { pattern ->
                runCatching { OffsetDateTime.parse(text, DateTimeFormatter.ofPattern(pattern).withResolverStyle(java.time.format.ResolverStyle.STRICT)).toInstant().toEpochMilli() }.getOrNull()
            }
        return parsed?.takeIf { it > 0 }
    }
}
