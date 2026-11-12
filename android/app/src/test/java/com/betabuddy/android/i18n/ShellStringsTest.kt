package com.betabuddy.android.i18n

import java.io.File
import javax.xml.parsers.DocumentBuilderFactory
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.w3c.dom.Element

/**
 * The shell, gym and settings strings (strings.xml) exist in all five languages with matching
 * placeholders and line breaks, and each file names its own language. The logbook's file has
 * the same check in JournalStringsTest. Reads the resource XML directly.
 */
class ShellStringsTest {
    private val folders = mapOf(
        "en" to "values", "zh-Hans" to "values-b+zh+Hans", "zh-Hant" to "values-b+zh+Hant", "ja" to "values-ja", "ko" to "values-ko",
    )

    private fun load(folder: String, file: String = "strings.xml"): Map<String, String> {
        val source = File("src/main/res/$folder/$file")
        assertTrue("Missing ${source.path}", source.isFile)
        val nodes = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(source).documentElement.childNodes
        val result = LinkedHashMap<String, String>()
        for (index in 0 until nodes.length) {
            val element = nodes.item(index) as? Element ?: continue
            if (element.tagName == "string") result[element.getAttribute("name")] = element.textContent
        }
        return result
    }

    private fun placeholders(text: String) = Regex("%\\d\\$[sd]").findAll(text).map { it.value }.sorted().toList()
    private fun lines(text: String) = text.split("\\n").size

    @Test fun everyStringIsInAllFiveLanguagesWithMatchingPlaceholders() {
        val english = load("values")
        assertTrue(english.size > 30)
        for ((language, folder) in folders) {
            val strings = load(folder)
            // The app name is the one string that is not translated.
            assertEquals(language, english.keys - "app_name", strings.keys - "app_name")
            for ((name, text) in strings) {
                assertTrue("$name / $language", text.isNotBlank())
                assertEquals("$name / $language", placeholders(english.getValue(name)), placeholders(text))
                assertEquals("$name / $language", lines(english.getValue(name)), lines(text))
                assertFalse("$name / $language: unescaped apostrophe", Regex("(?<!\\\\)'").containsMatchIn(text))
                assertFalse("$name / $language: stray percent", text.replace(Regex("%\\d\\$[sd]"), "").contains('%'))
            }
        }
    }

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
        for (language in listOf("en", "ko")) {
            for ((name, text) in load(folders.getValue(language))) {
                assertFalse("$name / $language: $text", text.any { it.code in 0x4E00..0x9FFF })
            }
        }
    }

    @Test fun theSettingsListsEachLanguageInItself() {
        assertEquals(listOf("简体中文", "繁體中文", "English", "日本語", "한국어"), AppLanguage.supported.map(AppLanguage::nativeName))
    }
}
