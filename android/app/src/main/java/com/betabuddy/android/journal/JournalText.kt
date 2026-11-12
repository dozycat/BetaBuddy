package com.betabuddy.android.journal

import com.betabuddy.android.data.Climb
import com.betabuddy.android.data.Moment
import java.io.InputStream
import java.io.OutputStream
import java.text.NumberFormat
import java.util.Locale

// Text and list logic with no Android dependency, so it runs in JVM unit tests.

enum class JournalFilter { ALL, SENT, PROJECTING }

/** Filter by send status, then by name or displayed grade (case-insensitive substring). */
fun filterClimbs(climbs: List<Climb>, filter: JournalFilter, search: String, gradeText: (String) -> String): List<Climb> {
    val query = search.trim()
    return climbs.filter { climb ->
        val statusMatches = when (filter) {
            JournalFilter.ALL -> true
            JournalFilter.SENT -> climb.sent
            JournalFilter.PROJECTING -> !climb.sent
        }
        statusMatches && (query.isEmpty() || climb.title.contains(query, ignoreCase = true) ||
            gradeText(climb.grade).contains(query, ignoreCase = true))
    }
}

/** Whole percent in the given locale's format, e.g. "56%" or "56 %". */
fun percentText(fraction: Double, locale: Locale): String {
    val format = NumberFormat.getPercentInstance(locale)
    format.maximumFractionDigits = 0
    return format.format(if (fraction.isFinite()) fraction else 0.0)
}

/** Cues saved before `kind` existed (by early iOS builds) are recognised by their stored title. */
fun Moment.resolvedKind(): String? = kind ?: when {
    title.startsWith("高抬脚") -> Moment.RAISED_HEEL
    title.startsWith("快速移动") -> Moment.FAST_MOVE
    else -> null
}

/** The plain-text recap handed to the system share sheet. Same layout as iOS. */
fun reportText(title: String, grade: String, status: String, clock: String, coverageLine: String, note: String, caveat: String): String =
    "BetaBuddy · $title\n$grade · $status · $clock\n$coverageLine\n$note\n\n$caveat"

class TooLargeException : Exception()

/**
 * Copies at most [maxBytes]; throws [TooLargeException] as soon as the source proves larger, so an
 * oversized pick never fills the disk. Returns the number of bytes copied.
 */
fun copyLimited(input: InputStream, output: OutputStream, maxBytes: Long, keepGoing: () -> Boolean = { true }): Long {
    val buffer = ByteArray(256 * 1024)
    var total = 0L
    while (keepGoing()) {
        val read = input.read(buffer)
        if (read < 0) return total
        total += read
        if (total > maxBytes) throw TooLargeException()
        output.write(buffer, 0, read)
    }
    throw InterruptedException()
}
