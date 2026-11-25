package com.betabuddy.android.i18n

import com.betabuddy.android.StringResources
import com.betabuddy.android.StringResources.folders
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The shell, gym and settings strings (strings.xml) and the folder and filter strings
 * (strings_album.xml) exist in all five languages with matching placeholders and line breaks,
 * and each language's strings.xml names its own language. The logbook and camera files have the
 * same checks in JournalStringsTest and CaptureStringsTest.
 */
class ShellStringsTest {
    private fun load(folder: String, file: String = "strings.xml") = StringResources.load(folder, file)

    // The app name is the one string that is not translated.
    @Test fun everyStringIsInAllFiveLanguagesWithMatchingPlaceholders() = StringResources.assertComplete("strings.xml", atLeast = 30, untranslated = setOf("app_name"))

    @Test fun everyAlbumStringIsInAllFiveLanguagesWithMatchingPlaceholders() = StringResources.assertComplete("strings_album.xml", atLeast = 20)

    @Test fun eachFileNamesItsOwnLanguage() {
        // AppLanguage reads this to know which resource set is on screen.
        for ((language, folder) in folders) assertEquals(language, load(folder).getValue("app_language_code"))
        assertEquals(folders.keys.toSet(), LanguageTags.supported.toSet())
    }

    @Test fun noNameIsSharedWithTheLogbookFile() {
        for (folder in folders.values) {
            val shared = load(folder).keys intersect load(folder, "strings_journal.xml").keys
            assertTrue("$folder: $shared", shared.isEmpty())
        }
    }

    @Test fun englishAndKoreanContainNoChineseLeftovers() {
        StringResources.assertNoChineseLeftovers("strings.xml")
        StringResources.assertNoChineseLeftovers("strings_album.xml")
    }

    @Test fun theSettingsListsEachLanguageInItself() {
        assertEquals(listOf("简体中文", "繁體中文", "English", "日本語", "한국어"), AppLanguage.supported.map(AppLanguage::nativeName))
    }
}
