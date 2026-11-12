import Foundation

/// The interface language the app shows: the user's in-app choice if there is one, otherwise the
/// system's (which already includes iOS's per-app language setting). One of the catalog's five codes.
enum AppLanguage {
    static let supported = ["zh-Hans", "zh-Hant", "en", "ja", "ko"]
    /// Each language written in itself, so a wrong choice can always be undone.
    static let nativeNames = ["zh-Hans": "简体中文", "zh-Hant": "繁體中文", "en": "English", "ja": "日本語", "ko": "한국어"]
    static let selectionKey = "appLanguage"

    /// The stored in-app choice; `nil` means follow the system.
    static func selection(in defaults: UserDefaults = .standard) -> String? {
        defaults.string(forKey: selectionKey).flatMap { supported.contains($0) ? $0 : nil }
    }
    static func select(_ language: String?, in defaults: UserDefaults = .standard) {
        if let language, supported.contains(language) { defaults.set(language, forKey: selectionKey) }
        else { defaults.removeObject(forKey: selectionKey) }
    }

    /// First system preference that maps to a supported language. English when none does.
    static func match(_ preferences: [String]) -> String {
        for identifier in preferences {
            let language = Locale.Language(identifier: identifier)
            switch language.languageCode?.identifier {
            case "zh": return language.maximalIdentifier.contains("Hant") ? "zh-Hant" : "zh-Hans"
            case "en": return "en"
            case "ja": return "ja"
            case "ko": return "ko"
            default: continue
            }
        }
        return "en"
    }
    static func resolve(selection: String?, system: [String]) -> String {
        selection.flatMap { supported.contains($0) ? $0 : nil } ?? match(system)
    }
    static var current: String { resolve(selection: selection(), system: Locale.preferredLanguages) }

    /// Language from the app, region from the device, for dates and numbers.
    static func locale(for language: String) -> Locale {
        Locale(identifier: Locale.current.region.map { "\(language)_\($0.identifier)" } ?? language)
    }
    static var locale: Locale { locale(for: current) }

    /// The string table of the current language. `String(localized:bundle:)` and SwiftUI's `\.locale`
    /// both read from it, so text follows the in-app choice without restarting.
    static func bundle(for language: String) -> Bundle {
        Bundle.main.path(forResource: language, ofType: "lproj").flatMap(Bundle.init(path:)) ?? .main
    }
    static var bundle: Bundle { bundle(for: current) }
}
