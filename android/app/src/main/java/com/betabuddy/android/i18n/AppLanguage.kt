package com.betabuddy.android.i18n

import android.content.Context
import androidx.appcompat.app.AppCompatDelegate
import androidx.compose.runtime.Composable
import androidx.compose.runtime.ReadOnlyComposable
import androidx.compose.ui.res.stringResource
import androidx.core.os.LocaleListCompat
import com.betabuddy.android.R

/**
 * The interface language: which of the five the app is showing, and the user's in-app choice.
 *
 * The choice is stored by AppCompat's per-app language support: on Android 13+ it is the same
 * setting the system shows under App languages; below that AppCompat keeps it itself
 * (`autoStoreLocales` in the manifest). "Follow the system" is the empty choice.
 */
object AppLanguage {
    val supported: List<String> get() = LanguageTags.supported

    /**
     * One of `zh-Hans`, `zh-Hant`, `en`, `ja`, `ko`: the language of the resources [context] resolves.
     * Each strings.xml carries its own code, so this is exactly what is on screen, English fallback included.
     */
    fun current(context: Context): String =
        context.getString(R.string.app_language_code).takeIf { it in LanguageTags.supported } ?: LanguageTags.FALLBACK

    /** [current] for composables; recomposes when the language changes. */
    @Composable
    @ReadOnlyComposable
    fun current(): String =
        stringResource(R.string.app_language_code).takeIf { it in LanguageTags.supported } ?: LanguageTags.FALLBACK

    /** The language picked in the app, or null when the app follows the system. */
    fun selected(): String? {
        val locales = AppCompatDelegate.getApplicationLocales()
        if (locales.isEmpty) return null
        return LanguageTags.match(locales.toLanguageTags().substringBefore(','))
    }

    /** Pick one of the five, or null to follow the system. Takes effect at once and is remembered. */
    fun select(code: String?) {
        val locales = if (code == null) LocaleListCompat.getEmptyLocaleList() else LocaleListCompat.forLanguageTags(code)
        AppCompatDelegate.setApplicationLocales(locales)
    }

    fun nativeName(code: String): String = LanguageTags.nativeName(code)
}
