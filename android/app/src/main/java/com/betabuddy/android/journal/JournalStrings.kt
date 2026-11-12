package com.betabuddy.android.journal

import android.text.format.DateFormat
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.res.stringResource
import androidx.core.os.ConfigurationCompat
import com.betabuddy.android.R
import com.betabuddy.android.data.Moment
import com.betabuddy.android.data.UNGRADED_VALUE
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

// Display text for values that are stored language-independently. Nothing here is written to the
// archive: the record keeps the type or the raw value, and the text is looked up when shown.

/** The language the app is showing, for number and date formats. */
@Composable
internal fun appLocale(): Locale = ConfigurationCompat.getLocales(LocalConfiguration.current)[0] ?: Locale.getDefault()

/** V grades are not translated; only the "no grade" sentinel has display text. */
@Composable
internal fun gradeText(grade: String): String = if (grade == UNGRADED_VALUE) stringResource(R.string.journal_ungraded) else grade

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

/** Known cue kinds are shown in the current language; unknown ones fall back to the stored text. */
@Composable
internal fun Moment.displayTitle(): String = when (resolvedKind()) {
    Moment.RAISED_HEEL -> stringResource(R.string.journal_cue_raised_heel_title)
    Moment.FAST_MOVE -> stringResource(R.string.journal_cue_fast_move_title)
    else -> title
}

@Composable
internal fun Moment.displayDetail(): String = when (resolvedKind()) {
    Moment.RAISED_HEEL -> stringResource(R.string.journal_cue_raised_heel_detail)
    Moment.FAST_MOVE -> stringResource(R.string.journal_cue_fast_move_detail)
    else -> detail
}
