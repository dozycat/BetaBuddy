package com.betabuddy.android

import java.io.File
import javax.xml.parsers.DocumentBuilderFactory
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.w3c.dom.Element

/**
 * Reads the string resources straight from the XML, for the per-file string tests. Paths are
 * relative to the module directory, which is where Gradle runs unit tests.
 */
object StringResources {
    /** Language code → resource folder; `values/` is English. */
    val folders = mapOf(
        "en" to "values", "zh-Hans" to "values-b+zh+Hans", "zh-Hant" to "values-b+zh+Hant", "ja" to "values-ja", "ko" to "values-ko",
    )

    /** name (or name#quantity for plurals) → raw resource text */
    fun load(folder: String, file: String): Map<String, String> {
        val source = File("src/main/res/$folder/$file")
        assertTrue("Missing ${source.path}", source.isFile)
        val nodes = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(source).documentElement.childNodes
        val result = LinkedHashMap<String, String>()
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

    private val placeholder = Regex("%\\d\\$[sd]")
    private fun placeholders(text: String) = placeholder.findAll(text).map { it.value }.sorted().toList()
    private fun lines(text: String) = text.split("\\n").size

    /**
     * Every string of [file] is in all five languages, not blank, with the English placeholders and
     * line breaks, no unescaped apostrophe and no stray percent sign. English alone has a singular
     * plural form; [untranslated] names strings that only English defines.
     */
    fun assertComplete(file: String, atLeast: Int, untranslated: Set<String> = emptySet()) {
        val english = load("values", file)
        assertTrue("$file has ${english.size} strings", english.size > atLeast)
        fun names(strings: Map<String, String>) = strings.keys.filterNot { it.endsWith("#one") || it in untranslated }.toSet()
        for ((language, folder) in folders) {
            val strings = load(folder, file)
            assertEquals("$file / $language", names(english), names(strings))
            for ((name, text) in strings) {
                if (name in untranslated) continue
                val reference = english[name] ?: english.getValue(name.substringBefore('#') + "#other")
                assertTrue("$name / $language", text.isNotBlank())
                assertEquals("$name / $language", placeholders(reference), placeholders(text))
                assertEquals("$name / $language", lines(reference), lines(text))
                assertFalse("$name / $language: unescaped apostrophe", Regex("(?<!\\\\)'").containsMatchIn(text))
                assertFalse("$name / $language: stray percent", text.replace(placeholder, "").contains('%'))
            }
        }
    }

    /** English and Korean text has no CJK ideographs left over from a copied Chinese string. */
    fun assertNoChineseLeftovers(file: String) {
        for (language in listOf("en", "ko")) {
            for ((name, text) in load(folders.getValue(language), file)) {
                assertFalse("$name / $language: $text", text.any { it.code in 0x4E00..0x9FFF })
            }
        }
    }

    /** Nothing is called a centre of mass, and nothing promises a score, accuracy or coaching. */
    fun assertHonestWording(file: String) {
        val banned = listOf("重心", "center of mass", "centre of mass", "精准", "最佳", "教练级", "accurate", "accuracy", "AI coach")
        for ((language, folder) in folders) {
            for ((name, text) in load(folder, file)) {
                for (word in banned) assertFalse("$name / $language contains “$word”", text.contains(word, ignoreCase = true))
            }
        }
    }
}
