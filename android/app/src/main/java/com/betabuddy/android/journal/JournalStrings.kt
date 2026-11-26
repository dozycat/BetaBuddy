package com.betabuddy.android.journal

import android.text.format.DateFormat
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.res.stringResource
import com.betabuddy.android.R
import com.betabuddy.android.data.Discipline
import com.betabuddy.android.data.UNGRADED_VALUE
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

// Display text for values that are stored language-independently. Nothing here is written to the
// archive: the record keeps the type or the raw value, and the text is looked up when shown.

/** The language the app is showing, for number and date formats. */
@Composable
internal fun appLocale(): Locale = LocalConfiguration.current.locales[0]

/** Grades (V on a boulder, YDS on a rope) are not translated; only the "no grade" sentinel has display text. */
@Composable
internal fun gradeText(grade: String): String = if (grade == UNGRADED_VALUE) stringResource(R.string.journal_ungraded) else grade

/** Display text of a stored discipline value. */
@Composable
internal fun disciplineText(discipline: String): String = stringResource(
    when (discipline) {
        Discipline.TOPROPE -> R.string.journal_discipline_toprope
        Discipline.LEAD -> R.string.journal_discipline_lead
        else -> R.string.journal_discipline_boulder
    },
)

@Composable
internal fun UiMessage.text(): String = if (argument == null) stringResource(id) else stringResource(id, argument)

/** Localized day and month, with the year when [withYear]; pattern chosen by the platform for [locale]. */
@Composable
internal fun dateText(millis: Long, withYear: Boolean): String {
    val locale = appLocale()
    return remember(millis, withYear, locale) {
        val pattern = DateFormat.getBestDateTimePattern(locale, if (withYear) "yMMMd" else "MMMd")
        SimpleDateFormat(pattern, locale).format(Date(millis))
    }
}
