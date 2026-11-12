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
    func testSystemPreferencesMapToSupportedLanguagesAndFallBackToEnglish() {
        XCTAssertEqual(AppLanguage.match(["ja-JP", "en-US"]), "ja")
        XCTAssertEqual(AppLanguage.match(["zh-Hans-CN"]), "zh-Hans")
        XCTAssertEqual(AppLanguage.match(["zh-CN"]), "zh-Hans")
        XCTAssertEqual(AppLanguage.match(["zh-TW"]), "zh-Hant")
        XCTAssertEqual(AppLanguage.match(["zh-Hant-HK"]), "zh-Hant")
        XCTAssertEqual(AppLanguage.match(["en-GB"]), "en")
        XCTAssertEqual(AppLanguage.match(["ko-KR"]), "ko")
        XCTAssertEqual(AppLanguage.match(["fr-FR", "ko-KR"]), "ko")
        XCTAssertEqual(AppLanguage.match(["fr-FR", "de-DE"]), "en")   // not Simplified Chinese
        XCTAssertEqual(AppLanguage.match([]), "en")
    }
    func testInAppChoiceWinsOverSystemAndPersists() throws {
        XCTAssertEqual(AppLanguage.resolve(selection: "ko", system: ["ja-JP"]), "ko")
        XCTAssertEqual(AppLanguage.resolve(selection: nil, system: ["ja-JP"]), "ja")
        XCTAssertEqual(AppLanguage.resolve(selection: "tlh", system: ["ja-JP"]), "ja")
        let suite = "language-tests-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        XCTAssertNil(AppLanguage.selection(in: defaults))
        AppLanguage.select("ja", in: defaults)
        XCTAssertEqual(AppLanguage.selection(in: defaults), "ja")
        XCTAssertEqual(UserDefaults(suiteName: suite)?.string(forKey: AppLanguage.selectionKey), "ja")
        AppLanguage.select("tlh", in: defaults)       // unknown codes clear the choice
        XCTAssertNil(AppLanguage.selection(in: defaults))
        AppLanguage.select("en", in: defaults)
        AppLanguage.select(nil, in: defaults)
        XCTAssertNil(AppLanguage.selection(in: defaults))
    }
    @MainActor func testLanguageSettingsPublishesTheChoiceAndRestoresIt() throws {
        let suite = "language-tests-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let settings = LanguageSettings(defaults: defaults)
        XCTAssertNil(settings.selection)
        XCTAssertEqual(settings.language, AppLanguage.match(Locale.preferredLanguages))
        settings.select("ko")
        XCTAssertEqual(settings.selection, "ko")
        XCTAssertEqual(settings.language, "ko")
        XCTAssertEqual(settings.locale.language.languageCode?.identifier, "ko")
        let relaunched = LanguageSettings(defaults: defaults)
        XCTAssertEqual(relaunched.language, "ko")
        relaunched.select(nil)
        XCTAssertNil(LanguageSettings(defaults: defaults).selection)
    }
    func testTextAndGymLanguageFollowTheChosenLanguage() throws {
        XCTAssertEqual(String(localized: "岩馆", bundle: AppLanguage.bundle(for: "en")), "Gym")
        XCTAssertEqual(String(localized: "岩馆", bundle: AppLanguage.bundle(for: "ja")), "ジム")
        let reason = "x"
        XCTAssertEqual(String(localized: "导入失败：\(reason)", bundle: AppLanguage.bundle(for: "ko")), "가져오기 실패: x")
        XCTAssertEqual(AppLanguage.locale(for: "zh-Hant").language.script?.identifier, "Hant")
        let chosen = AppLanguage.resolve(selection: "ja", system: ["en-US"])
        let entry = "gympack://local/index.html?lang=\(chosen)"
        XCTAssertEqual(entry, "gympack://local/index.html?lang=ja")
        XCTAssertTrue(AppLanguage.supported.contains(AppLanguage.current))
        XCTAssertEqual(Set(AppLanguage.nativeNames.keys), Set(AppLanguage.supported))
    }
    func testPolicyFallback() {
        XCTAssertEqual(PrivacyPolicyView.resourceName(for: "zh-Hans"), "PrivacyPolicy.zh-Hans")
        XCTAssertEqual(PrivacyPolicyView.resourceName(for: "zh-Hant"), "PrivacyPolicy.zh-Hant")
        for language in ["en", "ja", "ko"] { XCTAssertEqual(PrivacyPolicyView.resourceName(for: language), "PrivacyPolicy.en") }
        for language in languages {
            XCTAssertNotNil(Bundle.main.url(forResource: PrivacyPolicyView.resourceName(for: language), withExtension: "txt"), language)
        }
    }
    @MainActor func testUnnamedClimbsFollowTheInterfaceLanguage() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ClimbStore(root: root)
        let staged = root.appendingPathComponent("fixture.mov")
        try Data([0]).write(to: staged)
        let analysis = PoseAnalysis(frames: [], sampledFrames: 0, duration: 1, aspectRatio: 1, processingSeconds: 0, moments: [])
        var climb = try store.save(title: "  ", grade: ungradedValue, sent: false, note: "", stagedVideo: staged, thumbnail: Data(), analysis: analysis)
        XCTAssertEqual(climb.title, "")                       // no language is written into the archive
        XCTAssertFalse(climb.displayTitle.isEmpty)
        let shown = climb.displayTitle
        for stored in ["今天的攀爬", "Today’s climb", "今日のクライミング", "오늘의 등반"] {   // defaults saved by earlier builds
            climb.title = stored
            XCTAssertEqual(climb.displayTitle, shown, stored)
        }
        climb.title = "周日的紫色线路"
        XCTAssertEqual(climb.displayTitle, "周日的紫色线路")
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
