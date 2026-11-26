package com.betabuddy.android.journal

import com.betabuddy.android.StringResources
import org.junit.Test

/**
 * The logbook's strings (strings_journal.xml) exist in all five languages with matching
 * placeholders and line breaks, and keep the analysis honest. Counterpart of
 * ios/BetaBuddyTests/LocalizationTests.swift.
 */
class JournalStringsTest {
    private val file = "strings_journal.xml"

    @Test fun everyStringIsInAllFiveLanguagesWithMatchingPlaceholders() = StringResources.assertComplete(file, atLeast = 65)

    @Test fun englishAndKoreanContainNoChineseLeftovers() = StringResources.assertNoChineseLeftovers(file)

    @Test fun wordingKeepsTheAnalysisHonest() = StringResources.assertHonestWording(file)
}
