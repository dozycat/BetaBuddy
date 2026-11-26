import SwiftUI

// Display text for values that are stored language-independently or in the source language.

func gradeText(_ grade: String) -> String {
    grade == ungradedValue ? String(localized: "未标级", bundle: AppLanguage.bundle) : grade
}

extension Climb {
    /// A climb saved without a name is stored with an empty title and shown as “today's climb” in the
    /// current language. Titles that are exactly that default in any language (written by earlier
    /// builds at save time) are treated the same way, so they follow a language change too.
    var displayTitle: String {
        let name = String(localized: "今天的攀爬", bundle: AppLanguage.bundle)
        if title.isEmpty { return name }
        let defaults = AppLanguage.supported.map { String(localized: "今天的攀爬", bundle: AppLanguage.bundle(for: $0)) }
        return defaults.contains(title) ? name : title
    }
}
