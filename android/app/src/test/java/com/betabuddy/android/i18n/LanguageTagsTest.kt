package com.betabuddy.android.i18n

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class LanguageTagsTest {
    @Test fun tagsMapOntoTheFiveLanguages() {
        val expected = mapOf(
            "zh-Hans" to "zh-Hans", "zh-CN" to "zh-Hans", "zh-Hans-CN" to "zh-Hans", "zh-SG" to "zh-Hans", "zh" to "zh-Hans", "zh_CN" to "zh-Hans",
            "zh-Hant" to "zh-Hant", "zh-TW" to "zh-Hant", "zh-HK" to "zh-Hant", "zh-MO" to "zh-Hant", "zh-Hant-TW" to "zh-Hant", "zh-Hans-HK" to "zh-Hans",
            "en" to "en", "en-GB" to "en", "EN-us" to "en", "ja" to "ja", "ja-JP" to "ja", "ko" to "ko", "ko-KR" to "ko",
        )
        for ((tag, code) in expected) assertEquals(tag, code, LanguageTags.match(tag))
        for (tag in listOf("fr", "de-DE", "es-419", "", "  ", "yue-HK")) assertNull(tag, LanguageTags.match(tag))
    }

    @Test fun firstSupportedPreferenceWinsAndEnglishIsTheFallback() {
        assertEquals("ja", LanguageTags.resolve(listOf("fr-FR", "ja-JP", "en-US")))
        assertEquals("zh-Hant", LanguageTags.resolve(listOf("zh-TW")))
        assertEquals("en", LanguageTags.resolve(listOf("fr-FR", "de-DE")))
        assertEquals("en", LanguageTags.resolve(emptyList()))
    }

    @Test fun everyLanguageNamesItself() {
        assertEquals(listOf("简体中文", "繁體中文", "English", "日本語", "한국어"), LanguageTags.supported.map(LanguageTags::nativeName))
    }
}
