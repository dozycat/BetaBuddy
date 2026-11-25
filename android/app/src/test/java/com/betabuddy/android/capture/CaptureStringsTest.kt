package com.betabuddy.android.capture

import com.betabuddy.android.StringResources
import com.betabuddy.android.StringResources.folders
import org.junit.Assert.assertFalse
import org.junit.Test

/**
 * The in-app camera's strings (strings_capture.xml) exist in all five languages with matching
 * placeholders and line breaks, keep the analysis honest, and none of them names the other
 * platform's phone.
 */
class CaptureStringsTest {
    private val file = "strings_capture.xml"
    private fun load(folder: String) = StringResources.load(folder, file)

    @Test fun everyStringIsInAllFiveLanguagesWithMatchingPlaceholders() = StringResources.assertComplete(file, atLeast = 90)

    @Test fun englishAndKoreanContainNoChineseLeftovers() = StringResources.assertNoChineseLeftovers(file)

    @Test fun wordingKeepsTheAnalysisHonest() = StringResources.assertHonestWording(file)

    @Test fun noneOfThemNamesAnotherPlatformsPhone() {
        for ((language, folder) in folders) {
            for ((name, text) in load(folder)) assertFalse("$name / $language", text.contains("iPhone", ignoreCase = true) || text.contains("iOS"))
        }
    }

    @Test fun coloursAreNeverNamed() {
        // The page's keys for colours are not shown; no sentence here spells one out either.
        val keys = listOf("red", "orange", "yellow", "green", "teal", "blue", "purple", "pink", "black", "white", "grey")
        for ((name, text) in load("values")) for (key in keys) assertFalse("$name names $key", Regex("\\b$key\\b", RegexOption.IGNORE_CASE).containsMatchIn(text))
    }
}
