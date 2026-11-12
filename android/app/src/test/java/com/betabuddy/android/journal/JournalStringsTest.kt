package com.betabuddy.android.journal

import java.io.File
import javax.xml.parsers.DocumentBuilderFactory
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.w3c.dom.Element

/**
 * The logbook's strings exist in all five languages with matching placeholders and line breaks.
 * Counterpart of ios/BetaBuddyTests/LocalizationTests.swift. Reads the resource XML directly.
 */
class JournalStringsTest {
    private val folders = mapOf(
        "en" to "values", "zh-Hans" to "values-b+zh+Hans", "zh-Hant" to "values-b+zh+Hant", "ja" to "values-ja", "ko" to "values-ko",
    )

    /** name (or name#quantity for plurals) → raw resource text */
    private fun load(folder: String): Map<String, String> {
        val file = File("src/main/res/$folder/strings_journal.xml")
        assertTrue("Missing ${file.path}", file.isFile)
        val root = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(file).documentElement
        val result = LinkedHashMap<String, String>()
        val nodes = root.childNodes
        for (index in 0 until nodes.length) {
            val element = nodes.item(index) as? Element ?: continue
            val name = element.getAttribute("name")
            when (element.tagName) {
                "string" -> result[name] = element.textContent
                "plurals" -> {
                    val items = element.getElementsByTagName("item")
                    for (i in 0 until items.length) {
                        val item = items.item(i) as Element
                        result["$name#${item.getAttribute("quantity")}"] = item.textContent
                    }
                }
            }
        }
        return result
    }

    private fun placeholders(text: String) = Regex("%\\d\\$[sd]").findAll(text).map { it.value }.sorted().toList()
    private fun lines(text: String) = text.split("\\n").size

    @Test fun everyStringIsInAllFiveLanguagesWithMatchingPlaceholders() {
        val english = load("values")
        assertTrue(english.size > 80)
        for ((language, folder) in folders) {
            val strings = load(folder)
            // English alone has a singular plural form; every `other` form and every string must be everywhere.
            assertEquals(language, english.keys.filterNot { it.endsWith("#one") }.toSet(), strings.keys.filterNot { it.endsWith("#one") }.toSet())
            for ((name, text) in strings) {
                val reference = english[name] ?: english.getValue(name.substringBefore('#') + "#other")
                assertTrue("$name / $language", text.isNotBlank())
                assertEquals("$name / $language", placeholders(reference), placeholders(text))
                assertEquals("$name / $language", lines(reference), lines(text))
                assertFalse("$name / $language: unescaped apostrophe", Regex("(?<!\\\\)'").containsMatchIn(text))
                assertFalse("$name / $language: stray percent", text.replace(Regex("%\\d\\$[sd]"), "").contains('%'))
            }
        }
    }

    @Test fun englishAndKoreanContainNoChineseLeftovers() {
        for (language in listOf("en", "ko")) {
            for ((name, text) in load(folders.getValue(language))) {
                assertFalse("$name / $language: $text", text.any { it.code in 0x4E00..0x9FFF })
            }
        }
    }

    @Test fun wordingKeepsTheAnalysisHonest() {
        // The hip trail is never called a centre of mass, and nothing promises a score or accuracy.
        val banned = listOf("重心", "center of mass", "centre of mass", "精准", "最佳", "教练级", "accurate", "accuracy", "AI coach")
        for ((language, folder) in folders) {
            for ((name, text) in load(folder)) {
                for (word in banned) assertFalse("$name / $language contains “$word”", text.contains(word, ignoreCase = true))
            }
        }
        val english = load("values")
        assertTrue(english.getValue("journal_cue_raised_heel_title").contains("cue"))
        assertTrue(load("values-b+zh+Hans").getValue("journal_cue_raised_heel_title").contains("候选"))
        assertTrue(english.getValue("journal_stat_elbow").contains("2D"))
        assertTrue(english.getValue("journal_stat_caveat").contains("not a score"))
    }
}
