package com.betabuddy.android.i18n

/** Maps BCP 47 language tags onto the app's five interface languages. No Android types, so it runs in plain unit tests. */
object LanguageTags {
    const val ZH_HANS = "zh-Hans"
    const val ZH_HANT = "zh-Hant"
    const val EN = "en"
    const val JA = "ja"
    const val KO = "ko"

    /** The codes the gym page's `lang=` parameter and `gymNameLocalized` use. Same list as the iOS app. */
    val supported: List<String> = listOf(ZH_HANS, ZH_HANT, EN, JA, KO)

    /** What an unsupported language reads. */
    const val FALLBACK = EN

    private val traditionalRegions = setOf("TW", "HK", "MO")

    /** The interface language a tag such as `zh-CN`, `zh-Hant-HK`, `ja-JP` or `en_GB` stands for, or null. */
    fun match(tag: String): String? {
        val parts = tag.trim().replace('_', '-').split('-').filter { it.isNotEmpty() }
        val language = parts.firstOrNull()?.lowercase() ?: return null
        val rest = parts.drop(1)
        return when (language) {
            "en" -> EN
            "ja" -> JA
            "ko" -> KO
            "zh" -> {
                val script = rest.firstOrNull { it.length == 4 }?.lowercase()
                val region = rest.firstOrNull { it.length == 2 }?.uppercase()
                when {
                    script == "hant" -> ZH_HANT
                    script == "hans" -> ZH_HANS
                    region in traditionalRegions -> ZH_HANT
                    else -> ZH_HANS
                }
            }
            else -> null
        }
    }

    /** The first preferred language the app has; English when none of them is supported. */
    fun resolve(preferred: List<String>): String = preferred.firstNotNullOfOrNull(::match) ?: FALLBACK

    /** Each language's own name for itself, shown as is in the language setting. */
    fun nativeName(code: String): String = when (code) {
        ZH_HANS -> "简体中文"
        ZH_HANT -> "繁體中文"
        JA -> "日本語"
        KO -> "한국어"
        else -> "English"
    }
}
