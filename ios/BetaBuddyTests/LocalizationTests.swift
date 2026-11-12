import XCTest
@testable import BetaBuddy

final class LocalizationTests: XCTestCase {
    private let languages = ["zh-Hans", "zh-Hant", "en", "ja", "ko"]

    // The catalogs are read from the source tree: the built app only has the compiled tables.
    private func catalog(_ name: String) throws -> [String: [String: Any]] {
        let resources = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("BetaBuddy/Resources/\(name).xcstrings")
        let root = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: resources)) as? [String: Any])
        XCTAssertEqual(root["sourceLanguage"] as? String, "zh-Hans")
        return try XCTUnwrap(root["strings"] as? [String: [String: Any]])
    }
    private func value(_ entry: [String: Any], _ language: String) -> String? {
        let unit = ((entry["localizations"] as? [String: Any])?[language] as? [String: Any])?["stringUnit"] as? [String: Any]
        return unit?["state"] as? String == "translated" ? unit?["value"] as? String : nil
    }
    private func placeholders(_ text: String) -> [String] {
        let pattern = try! NSRegularExpression(pattern: "%(?:\\d+\\$)?(?:lld|ld|d|@|f)")
        return pattern.matches(in: text, range: NSRange(text.startIndex..., in: text)).map { String(text[Range($0.range, in: text)!]) }.sorted()
    }

    func testEveryKeyIsTranslatedInAllFiveLanguagesWithMatchingPlaceholders() throws {
        for name in ["Localizable", "InfoPlist"] {
            let strings = try catalog(name)
            XCTAssertFalse(strings.isEmpty, name)
            for (key, entry) in strings {
                let source = value(entry, "zh-Hans") ?? key
                for language in languages {
                    guard let text = value(entry, language) else { XCTFail("\(name): “\(key)” has no \(language) translation"); continue }
                    XCTAssertFalse(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, "\(key) / \(language)")
                    XCTAssertEqual(placeholders(text), placeholders(source), "\(key) / \(language)")
                    XCTAssertEqual(text.components(separatedBy: "\n").count, source.components(separatedBy: "\n").count, "\(key) / \(language)")
                }
            }
        }
    }
    func testTranslationsOutsideChineseContainNoHanOnlyLeftovers() throws {
        // English and Korean text should not contain Chinese characters at all.
        for (key, entry) in try catalog("Localizable") {
            for language in ["en", "ko"] {
                let text = try XCTUnwrap(value(entry, language))
                XCTAssertNil(text.unicodeScalars.first { (0x4E00...0x9FFF).contains($0.value) }, "\(key) / \(language): \(text)")
            }
        }
    }
    func testBuiltAppCarriesAllFiveLocalizations() {
        XCTAssertEqual(Set(Bundle.main.localizations).intersection(languages), Set(languages))
        for language in languages {
            let table = Bundle.main.path(forResource: language, ofType: "lproj").flatMap(Bundle.init(path:))
            XCTAssertNotNil(table, language)
            XCTAssertNotEqual(table?.localizedString(forKey: "攀爬册", value: "?", table: nil), "?", language)
        }
        let english = Bundle.main.path(forResource: "en", ofType: "lproj").flatMap(Bundle.init(path:))
        XCTAssertEqual(english?.localizedString(forKey: "岩馆", value: nil, table: nil), "Gym")
        XCTAssertEqual(english?.localizedString(forKey: "导入失败：%@", value: nil, table: nil), "Import failed: %@")
    }
    func testLanguageResolutionAndPolicyFallback() {
        XCTAssertEqual(AppLanguage.resolve(["ja", "en"]), "ja")
        XCTAssertEqual(AppLanguage.resolve(["zh-Hant"]), "zh-Hant")
        XCTAssertEqual(AppLanguage.resolve(["fr"]), "en")
        XCTAssertEqual(AppLanguage.resolve([]), "en")
        XCTAssertTrue(AppLanguage.supported.contains(AppLanguage.current))
        XCTAssertEqual(PrivacyPolicyView.resourceName(for: "zh-Hans"), "PrivacyPolicy.zh-Hans")
        XCTAssertEqual(PrivacyPolicyView.resourceName(for: "zh-Hant"), "PrivacyPolicy.zh-Hant")
        for language in ["en", "ja", "ko"] { XCTAssertEqual(PrivacyPolicyView.resourceName(for: language), "PrivacyPolicy.en") }
        for language in languages {
            XCTAssertNotNil(Bundle.main.url(forResource: PrivacyPolicyView.resourceName(for: language), withExtension: "txt"), language)
        }
    }
    func testStoredValuesHaveDisplayText() {
        XCTAssertFalse(gradeText(ungradedValue).isEmpty)
        XCTAssertEqual(gradeText("V4"), "V4")
        let cue = Moment(time: 1, title: "stored", detail: "stored detail", kind: Moment.fastMove)
        XCTAssertNotEqual(cue.displayTitle, "stored")
        let unknown = Moment(time: 1, title: "stored", detail: "stored detail", kind: "futureKind")
        XCTAssertEqual(unknown.displayTitle, "stored")
        XCTAssertEqual(unknown.displayDetail, "stored detail")
    }
}
