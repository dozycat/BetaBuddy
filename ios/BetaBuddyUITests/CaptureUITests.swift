import XCTest

/// The capture screen driven through the interface, with a drawn wall played from a file in place
/// of the camera and a scripted stand-in for the pack's capture page. Screenshots of each state are
/// attached; `TEST_RUNNER_BB_SCREENSHOT_DIR=<directory>` also writes them there as PNG files.
final class CaptureUITests: XCTestCase {
    private var work: URL!
    private var clip: URL?

    override func setUp() {
        continueAfterFailure = false
        work = FileManager.default.temporaryDirectory.appendingPathComponent("capture-ui-\(UUID())")
        try? FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
    }
    override func tearDown() {
        try? FileManager.default.removeItem(at: work)
        if let clip { try? FileManager.default.removeItem(at: clip) }
    }

    @MainActor private func keep(_ app: XCUIApplication, _ name: String) {
        let shot = app.screenshot()
        let attachment = XCTAttachment(screenshot: shot)
        attachment.name = name; attachment.lifetime = .keepAlways
        add(attachment)
        if let directory = ProcessInfo.processInfo.environment["BB_SCREENSHOT_DIR"], !directory.isEmpty {
            try? FileManager.default.createDirectory(atPath: directory, withIntermediateDirectories: true)
            try? shot.pngRepresentation.write(to: URL(fileURLWithPath: directory).appendingPathComponent(name + ".png"))
        }
    }
    @MainActor private func launch(language: String = "zh-Hans", locale: String = "zh_CN", capture: Bool = true, source: Bool = true, step: Int = 14) async throws -> XCUIApplication {
        try StandInPack.write(to: work, capture: capture, step: step)
        let directory = work.appendingPathComponent("gyms.json")
        try Data(StandInPack.directory.utf8).write(to: directory)
        let app = XCUIApplication()
        var arguments = ["-AppleLanguages", "(\(language))", "-AppleLocale", locale, "-resetLanguage", "YES", "-initialTab", "1",
                         "-debugGymPackRoot", work.path, "-debugGymDirectory", directory.path,
                         "-debugLocation", "\(StandInPack.origin.latitude),\(StandInPack.origin.longitude)"]
        if source {
            let clip = try await SyntheticWallClip.make(width: 540, height: 960, seconds: 4)
            self.clip = clip
            arguments += ["-debugCaptureVideo", clip.path]
        }
        app.launchArguments = arguments
        app.launch()
        return app
    }
    /// The guidance line while it reports `code`.
    @MainActor private func hint(_ app: XCUIApplication, _ code: String, timeout: TimeInterval = 40) -> XCUIElement {
        let element = app.descendants(matching: .any).matching(NSPredicate(format: "identifier == 'captureHint' AND value == %@", code)).firstMatch
        XCTAssertTrue(element.waitForExistence(timeout: timeout), "guidance never showed \(code)")
        return element
    }
    @MainActor private func removeNewestRecord(_ app: XCUIApplication, more: String, confirm: String) {
        app.buttons[more].tap()
        app.buttons["deleteClimb"].tap()
        let button = app.sheets.buttons[confirm]
        XCTAssertTrue(button.waitForExistence(timeout: 5))
        button.tap()
    }

    @MainActor
    func testFramingGuidanceRecordingAndSavingInChinese() async throws {
        let app = try await launch()
        let record = app.buttons["recordClimb"]
        XCTAssertTrue(record.waitForExistence(timeout: 15))
        XCTAssertTrue(app.buttons["importClimb"].exists)                 // Importing stays, as the second way in.
        keep(app, "00-album-entry")
        record.tap()

        // The stand-in's script: a blocking problem, one with a move, a code this app does not know, then ready.
        XCTAssertEqual(hint(app, "too_dark").label, "画面太暗，换个亮一点的位置")
        XCTAssertTrue(app.buttons["captureColor-yellow"].waitForExistence(timeout: 5))
        keep(app, "01-blocked")
        XCTAssertEqual(hint(app, "cut_top").label, "上面出画了，往后退一点")
        keep(app, "02-blocked-with-move")
        XCTAssertEqual(hint(app, "lens_flare_2031").label, "换个机位试试")
        keep(app, "03-unknown-code")
        let ready = app.descendants(matching: .any)["captureReady"]
        XCTAssertTrue(ready.waitForExistence(timeout: 40))
        keep(app, "04-ready")

        // Gym: nearest first, a distance only where the directory knows the gym itself.
        app.buttons["captureGym"].tap()
        let east = app.buttons["nearbyGym-syn:east"]
        XCTAssertTrue(east.waitForExistence(timeout: 5))
        let rows = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH 'nearbyGym-'")).allElementsBoundByIndex.map(\.identifier)
        XCTAssertEqual(rows, ["nearbyGym-syn:east", "nearbyGym-syn:district", "nearbyGym-syn:city", "nearbyGym-syn:west"])
        XCTAssertTrue(east.label.contains("300"), east.label)
        XCTAssertFalse(app.buttons["nearbyGym-syn:district"].label.contains("km"), app.buttons["nearbyGym-syn:district"].label)
        XCTAssertTrue(app.buttons["allGyms"].exists)
        keep(app, "05-gym-by-distance")
        east.tap()
        XCTAssertTrue(app.buttons["captureGym"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.buttons["captureGym"].label.contains(StandInPack.directoryGymName))

        // A gym that is one of the pack's venues starts the page's judgement over; this time, record against its advice.
        _ = hint(app, "too_dark")
        app.buttons["captureRecord"].tap()
        let anyway = app.buttons["recordAnyway"]
        XCTAssertTrue(anyway.waitForExistence(timeout: 5))
        keep(app, "06-ask-before-recording")
        app.buttons["再调一下机位"].tap()
        XCTAssertFalse(app.descendants(matching: .any)["captureElapsed"].exists)
        XCTAssertTrue(ready.waitForExistence(timeout: 60))

        app.buttons["captureGrade"].tap()
        XCTAssertTrue(app.buttons["V4"].waitForExistence(timeout: 5))
        app.buttons["V4"].tap()

        app.buttons["captureColor-yellow"].tap()
        let note = app.staticTexts["captureColourNote"]
        XCTAssertTrue(note.waitForExistence(timeout: 5))
        XCTAssertEqual(note.label, "虚线是自动认的，可能认错。点一下岩点可以改。")
        try await Task.sleep(for: .seconds(1))                       // The outlines come with the next assessment.
        keep(app, "07-colour-holds-start-finish")

        // Tap a hold and say what it is.
        app.descendants(matching: .any)["capturePicture"].coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.6)).tap()
        XCTAssertTrue(app.buttons["结束点"].waitForExistence(timeout: 5))
        keep(app, "08-correct-a-hold")
        app.buttons["结束点"].tap()
        try await Task.sleep(for: .seconds(1.5))
        keep(app, "09-corrected")

        app.buttons["captureRecord"].tap()
        XCTAssertTrue(app.descendants(matching: .any)["captureElapsed"].waitForExistence(timeout: 5))
        XCTAssertFalse(app.buttons["captureClose"].isEnabled)
        try await Task.sleep(for: .seconds(3))
        keep(app, "10-recording")
        app.buttons["captureStop"].tap()

        XCTAssertTrue(app.descendants(matching: .any)["captureSaved"].waitForExistence(timeout: 20))
        keep(app, "11-saved")
        app.buttons["captureDone"].tap()

        // In the album, with what was chosen on the capture screen; nothing was asked again.
        XCTAssertTrue(record.waitForExistence(timeout: 10))
        XCTAssertTrue(app.staticTexts[StandInPack.directoryGymName].waitForExistence(timeout: 5))
        XCTAssertTrue(app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH 'V4'")).firstMatch.exists)
        try await Task.sleep(for: .seconds(2))
        keep(app, "12-album-after")
        app.terminate(); app.launch()
        XCTAssertTrue(app.staticTexts[StandInPack.directoryGymName].waitForExistence(timeout: 10))
        // The next visit starts from the gym of the most recent record.
        app.buttons["recordClimb"].tap()
        XCTAssertTrue(app.buttons["captureGym"].waitForExistence(timeout: 10))
        XCTAssertTrue(app.buttons["captureGym"].label.contains(StandInPack.directoryGymName))
        app.buttons["captureClose"].tap()
        app.staticTexts[StandInPack.directoryGymName].firstMatch.tap()
        XCTAssertTrue(app.staticTexts["playbackClock"].waitForExistence(timeout: 10))
        keep(app, "13-review")
        removeNewestRecord(app, more: "更多操作", confirm: "删除这一趟")
    }

    @MainActor
    func testWithoutACapturePageInEnglish() async throws {
        let app = try await launch(language: "en", locale: "en_US", capture: false)
        XCTAssertTrue(app.buttons["recordClimb"].waitForExistence(timeout: 15))
        app.buttons["recordClimb"].tap()
        let note = app.descendants(matching: .any)["captureNoChecks"]
        XCTAssertTrue(note.waitForExistence(timeout: 15))
        XCTAssertEqual(note.label, "This build has no camera-position check or hold recognition. Recording works as usual.")
        XCTAssertFalse(app.buttons["captureColor-yellow"].exists)
        keep(app, "20-no-capture-page-en")
        app.buttons["captureRecord"].tap()                          // Nothing to ask: there is no judgement to go against.
        XCTAssertTrue(app.descendants(matching: .any)["captureElapsed"].waitForExistence(timeout: 5))
        try await Task.sleep(for: .seconds(2.5))
        keep(app, "21-recording-en")
        app.buttons["captureStop"].tap()
        XCTAssertTrue(app.descendants(matching: .any)["captureSaved"].waitForExistence(timeout: 20))
        keep(app, "22-saved-en")
        app.buttons["captureAgain"].tap()
        XCTAssertTrue(note.waitForExistence(timeout: 10))
        app.buttons["captureClose"].tap()
        XCTAssertTrue(app.buttons["recordClimb"].waitForExistence(timeout: 10))
        app.staticTexts["Today’s climb"].firstMatch.tap()
        XCTAssertTrue(app.staticTexts["playbackClock"].waitForExistence(timeout: 10))
        removeNewestRecord(app, more: "More actions", confirm: "Delete this climb")
    }

    @MainActor
    func testReadyStateInEnglish() async throws {
        let app = try await launch(language: "en", locale: "en_US", step: 3)
        XCTAssertTrue(app.buttons["recordClimb"].waitForExistence(timeout: 15))
        app.buttons["recordClimb"].tap()
        XCTAssertTrue(app.descendants(matching: .any)["captureReady"].waitForExistence(timeout: 40))
        app.buttons["captureColor-red"].tap()
        // Nothing blocks, but the stand-in is not sure about this colour's start: a warning under “ready”.
        XCTAssertEqual(hint(app, "start_unclear").label, "Not sure about the start; tap a hold to mark it yourself")
        XCTAssertTrue(app.descendants(matching: .any)["captureReady"].exists)
        keep(app, "30-ready-with-warning-en")
        app.buttons["captureClose"].tap()
    }

    /// The simulator has no camera: without the file source the screen says so and offers the import.
    @MainActor
    func testNoCameraFallsBackToImport() async throws {
        let app = try await launch(source: false)
        XCTAssertTrue(app.buttons["recordClimb"].waitForExistence(timeout: 15))
        app.buttons["recordClimb"].tap()
        XCTAssertTrue(app.staticTexts["captureUnavailable"].waitForExistence(timeout: 15))
        XCTAssertFalse(app.buttons["captureRecord"].isEnabled)
        keep(app, "40-no-camera")
        app.buttons["改从相册导入"].tap()
        XCTAssertTrue(app.buttons["saveClimb"].waitForExistence(timeout: 10))
        app.buttons["importCancel"].tap()
    }
}
