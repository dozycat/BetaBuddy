import Foundation

/// The interface language the app is actually showing, as one of the catalog's five codes.
enum AppLanguage {
    static let supported = ["zh-Hans", "zh-Hant", "en", "ja", "ko"]
    static func resolve(_ preferred: [String]) -> String {
        preferred.first { supported.contains($0) } ?? "en"
    }
    static var current: String { resolve(Bundle.main.preferredLocalizations) }
}
