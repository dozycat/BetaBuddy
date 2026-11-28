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
    @MainActor private func launch(language: String = "zh-Hans", locale: String = "zh_CN", capture: Bool = true, source: Bool = true, step: Int = 14,
                                   location: String = "\(StandInPack.origin.latitude),\(StandInPack.origin.longitude)", extra: [String] = []) async throws -> XCUIApplication {
        try StandInPack.write(to: work, capture: capture, step: step)
        let directory = work.appendingPathComponent("gyms.json")
        try Data(StandInPack.directory.utf8).write(to: directory)
        let app = XCUIApplication()
        var arguments = ["-AppleLanguages", "(\(language))", "-AppleLocale", locale, "-resetLanguage", "YES", "-initialTab", "1",
                         "-debugGymPackRoot", work.path, "-debugGymDirectory", directory.path,
                         "-debugLocation", location]
        if source {
            let clip = try await SyntheticWallClip.make(width: 540, height: 960, seconds: 4)
            self.clip = clip
            arguments += ["-debugCaptureVideo", clip.path]
        }
        app.launchArguments = arguments + extra
        app.launch()
        return app
    }
    /// How much of the screen's width the picture takes, noted with the screenshots.
    @MainActor @discardableResult private func pictureShare(_ app: XCUIApplication, _ state: String) -> Double {
        let picture = app.descendants(matching: .any)["capturePicture"]
        guard picture.waitForExistence(timeout: 5), app.frame.width > 0 else { XCTFail("no picture"); return 0 }
        let share = Double(picture.frame.width / app.frame.width)
        let line = String(format: "%@: picture %.0f of %.0f pt = %.0f%%", state, picture.frame.width, app.frame.width, share * 100)
        let attachment = XCTAttachment(string: line); attachment.name = "Picture width, " + state; attachment.lifetime = .keepAlways; add(attachment)
        if let directory = ProcessInfo.processInfo.environment["BB_SCREENSHOT_DIR"], !directory.isEmpty {
            let file = URL(fileURLWithPath: directory).appendingPathComponent("picture-width.txt")
            let old = (try? String(contentsOf: file, encoding: .utf8)) ?? ""
            try? (old + line + "\n").write(to: file, atomically: true, encoding: .utf8)
        }
        return share
    }
    /// The camera-position check is no longer a line on the screen: the record button's ring shows it,
    /// and the button's value reads it out. Waits until it says `text`.
    @MainActor private func position(_ app: XCUIApplication, _ text: String, timeout: TimeInterval = 40) {
        let record = app.buttons["captureRecord"]
        let said = XCTWaiter.wait(for: [expectation(for: NSPredicate(format: "value == %@", text), evaluatedWith: record)], timeout: timeout)
        XCTAssertEqual(said, .completed, "the record button never said \(text)")
    }
    @MainActor private func pick(_ app: XCUIApplication, discipline id: String) {
        app.buttons["captureDiscipline"].tap()
        let item = app.buttons["discipline-" + id]
        XCTAssertTrue(item.waitForExistence(timeout: 10))
        item.tap()
    }
    /// One grade further on the wheel: a tap on the grade right of the middle turns to it.
    @MainActor private func stepUp(_ wheel: XCUIElement, to value: String) {
        wheel.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).withOffset(CGVector(dx: 78, dy: 0)).tap()
        let reached = NSPredicate(format: "value == %@", value)
        XCTAssertEqual(XCTWaiter.wait(for: [expectation(for: reached, evaluatedWith: wheel)], timeout: 3), .completed, "the wheel did not reach \(value)")
    }
    @MainActor private func removeNewestRecord(_ app: XCUIApplication, more: String, confirm: String) {
        app.buttons[more].tap()
        app.buttons["deleteClimb"].tap()
        let button = app.sheets.buttons[confirm]
        XCTAssertTrue(button.waitForExistence(timeout: 5))
        button.tap()
    }

    @MainActor private func cancelConfirmation(_ app: XCUIApplication, label: String) {
        if app.buttons[label].exists {
            app.buttons[label].tap()
        } else {
            // Compact popovers omit the cancel row and dismiss by tapping outside.
            let outside = app.otherElements["PopoverDismissRegion"]
            XCTAssertTrue(outside.exists, app.debugDescription)
            outside.coordinate(withNormalizedOffset: CGVector(dx: 0.1, dy: 0.2)).tap()
        }
    }

    @MainActor
    func testFramingGuidanceRecordingAndSavingInChinese() async throws {
        let app = try await launch()
        let record = app.buttons["recordClimb"]
        XCTAssertTrue(record.waitForExistence(timeout: 15))
        XCTAssertTrue(app.buttons["importClimb"].exists)                 // Importing stays, as the second way in.
        keep(app, "00-album-entry")
        record.tap()

        // The nearest gym whose own position the directory knows is selected by itself, with how far it is.
        let gym = app.buttons["captureGym"]
        XCTAssertTrue(gym.waitForExistence(timeout: 10))
        XCTAssertTrue(XCTWaiter.wait(for: [expectation(for: NSPredicate(format: "label CONTAINS %@", StandInPack.directoryGymName), evaluatedWith: gym)], timeout: 15) == .completed, gym.label)
        XCTAssertTrue(gym.label.contains("最近"), gym.label)
        // Nothing else on the screen: no guidance line, no colours, no marks, no recognised problem.
        XCTAssertFalse(app.buttons["captureColor-yellow"].exists)
        XCTAssertFalse(app.staticTexts["captureRoute"].exists)
        XCTAssertFalse(app.buttons["useNearestGym"].exists)

        // The stand-in's script, read from the record button: a blocking problem, one with a move, a code this app does not know, then ready.
        position(app, "画面太暗，换个亮一点的位置")
        keep(app, "01-blocked")
        position(app, "上面出画了，往后退一点")
        position(app, "换个机位试试")
        position(app, "机位可以，开始吧")
        keep(app, "04-ready")
        XCTAssertGreaterThanOrEqual(pictureShare(app, "framing"), 0.95)

        // Gym: only gyms whose own position is known are ranked, nearest first; a station-level position
        // is shown as approximate; the rest of the city follows without distances.
        gym.tap()
        let east = app.buttons["nearbyGym-syn:east"]
        XCTAssertTrue(east.waitForExistence(timeout: 5))
        let rows = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH 'nearbyGym-'")).allElementsBoundByIndex.map(\.identifier)
        XCTAssertEqual(rows, ["nearbyGym-syn:east", "nearbyGym-syn:station", "nearbyGym-syn:west"])
        XCTAssertTrue(east.label.contains("200"), east.label)
        XCTAssertTrue(app.buttons["nearbyGym-syn:station"].label.contains("约 1.5"), app.buttons["nearbyGym-syn:station"].label)
        XCTAssertTrue(app.staticTexts["© OpenStreetMap contributors"].exists)
        keep(app, "05-gym-by-distance")
        // Another gym: the page's judgement starts over; this time, record against its advice.
        app.buttons["nearbyGym-syn:station"].tap()
        XCTAssertTrue(gym.label.contains("只知道最近车站的岩馆"), gym.label)
        position(app, "画面太暗，换个亮一点的位置")
        app.buttons["captureRecord"].tap()
        let anyway = app.buttons["recordAnyway"]
        XCTAssertTrue(anyway.waitForExistence(timeout: 5))
        keep(app, "06-ask-before-recording")
        cancelConfirmation(app, label: "再调一下机位")
        XCTAssertFalse(app.descendants(matching: .any)["captureElapsed"].exists)
        // Back to the nearest one.
        gym.tap()
        XCTAssertTrue(east.waitForExistence(timeout: 5))
        east.tap()
        position(app, "机位可以，开始吧", timeout: 60)

        // Discipline: automatic until picked; a rope discipline turns the wheel to YDS.
        let discipline = app.buttons["captureDiscipline"]
        let grade = app.descendants(matching: .any)["captureGrade"]
        XCTAssertEqual(discipline.value as? String, "自动判断")
        pick(app, discipline: "toprope")
        XCTAssertEqual(discipline.value as? String, "顶绳")
        XCTAssertEqual(XCTWaiter.wait(for: [expectation(for: NSPredicate(format: "value == %@", "未标级"), evaluatedWith: grade)], timeout: 5), .completed)
        stepUp(grade, to: "5.5")
        keep(app, "07-top-rope-yds")
        // Back to automatic: the YDS grade does not fit the bouldering scale and goes back to ungraded.
        pick(app, discipline: "auto")
        XCTAssertEqual(XCTWaiter.wait(for: [expectation(for: NSPredicate(format: "value == %@", "未标级"), evaluatedWith: grade)], timeout: 5), .completed)
        // The user's own grade, one step at a time on the wheel.
        for value in ["V0", "V1", "V2", "V3", "V4"] { stepUp(grade, to: value) }
        keep(app, "09-grade-picked")

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
        // The gym name is the day group's heading; open the recorded attempt itself.
        app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH 'V4'")).firstMatch.tap()
        XCTAssertTrue(app.staticTexts["playbackClock"].waitForExistence(timeout: 10))
        keep(app, "13-review")
        removeNewestRecord(app, more: "更多操作", confirm: "删除这一趟")
    }

    @MainActor
    func testWithoutACapturePageInEnglish() async throws {
        let app = try await launch(language: "en", locale: "en_US", capture: false)
        XCTAssertTrue(app.buttons["recordClimb"].waitForExistence(timeout: 15))
        app.buttons["recordClimb"].tap()
        position(app, "This build has no camera-position check or hold recognition. Recording works as usual.", timeout: 15)
        keep(app, "20-no-capture-page-en")
        app.buttons["captureRecord"].tap()                          // Nothing to ask: there is no judgement to go against.
        XCTAssertTrue(app.descendants(matching: .any)["captureElapsed"].waitForExistence(timeout: 5))
        try await Task.sleep(for: .seconds(2.5))
        keep(app, "21-recording-en")
        app.buttons["captureStop"].tap()
        XCTAssertTrue(app.descendants(matching: .any)["captureSaved"].waitForExistence(timeout: 20))
        keep(app, "22-saved-en")
        app.buttons["captureAgain"].tap()
        position(app, "This build has no camera-position check or hold recognition. Recording works as usual.", timeout: 10)
        app.buttons["captureClose"].tap()
        XCTAssertTrue(app.buttons["recordClimb"].waitForExistence(timeout: 10))
        app.staticTexts["Today’s climb"].firstMatch.tap()
        XCTAssertTrue(app.staticTexts["playbackClock"].waitForExistence(timeout: 10))
        removeNewestRecord(app, more: "More actions", confirm: "Delete this climb")
    }

    @MainActor
    func testReadyStateInEnglish() async throws {
        // With the numbers a debug build can show over the picture for a first run on a phone.
        let app = try await launch(language: "en", locale: "en_US", step: 3, extra: ["-debugCaptureOverlay", "YES"])
        XCTAssertTrue(app.buttons["recordClimb"].waitForExistence(timeout: 15))
        app.buttons["recordClimb"].tap()
        position(app, "Position works. Go ahead")
        let numbers = app.staticTexts.matching(NSPredicate(format: "label CONTAINS 'fovH' AND label CONTAINS 'replies' AND label CONTAINS 'hand-over' AND label CONTAINS 'issues'")).firstMatch
        XCTAssertTrue(numbers.waitForExistence(timeout: 5))
        keep(app, "31-debug-overlay")
        XCTAssertTrue(app.buttons["captureRecord"].isEnabled)
        XCTAssertEqual(app.buttons["captureDiscipline"].value as? String, "Auto")
        keep(app, "30-ready-en")
        // A debug build leaves a picture of this screen on disk; here the preview itself comes out in it.
        XCTAssertTrue(app.staticTexts.matching(NSPredicate(format: "label CONTAINS 'snapshot hierarchy'")).firstMatch.waitForExistence(timeout: 8))
        app.buttons["captureClose"].tap()
    }

    /// A debug build can go through the screen by itself when nobody can tap the phone: this test
    /// touches nothing. With the gym given, the colour given, and no start known for that colour, it
    /// marks start and finish, records, stops, and waits for what comes after saving.
    @MainActor
    func testAutopilotGoesFromColourToSavedWithoutATap() async throws {
        let app = try await launch(step: 3, extra: ["-debugOpenCapture", "YES", "-debugCaptureAutopilot", "YES", "-debugCaptureGym", "syn:east", "-debugCaptureColour", "red",
                                                    "-debugCaptureSeconds", "4", "-debugCaptureOverlay", "YES"])
        func note(_ text: String) -> XCUIElement { app.staticTexts.matching(NSPredicate(format: "label CONTAINS %@", "autopilot " + text)).firstMatch }
        XCTAssertTrue(note("recording").waitForExistence(timeout: 60), "it got as far as recording")
        XCTAssertTrue(app.buttons["captureStop"].exists)
        XCTAssertTrue(note("done").waitForExistence(timeout: 90), "and to the end")
        // The saved sheet stays up.
        XCTAssertTrue(app.descendants(matching: .any)["captureSaved"].exists)
        keep(app, "33-autopilot-done")
        // The record carries the gym it selected by itself. It is taken out again: the other tests start from an album without one.
        app.buttons["captureDone"].tap()
        XCTAssertTrue(app.staticTexts[StandInPack.directoryGymName].waitForExistence(timeout: 10))
        app.staticTexts[StandInPack.directoryGymName].firstMatch.tap()
        XCTAssertTrue(app.staticTexts["playbackClock"].waitForExistence(timeout: 10))
        removeNewestRecord(app, more: "更多操作", confirm: "删除这一趟")
    }

    /// A camera preview is left black in a snapshot of the view hierarchy. The picture a debug build
    /// leaves on disk then has the frame last handed to the capture page put under the outlines instead.
    @MainActor
    func testDiagnosticsPictureWhenThePreviewDoesNotComeOutInASnapshot() async throws {
        let app = try await launch(language: "en", locale: "en_US", step: 3, extra: ["-debugCaptureOverlay", "YES", "-debugCaptureHidePreview", "YES"])
        XCTAssertTrue(app.buttons["recordClimb"].waitForExistence(timeout: 15))
        app.buttons["recordClimb"].tap()
        position(app, "Position works. Go ahead")
        XCTAssertTrue(app.staticTexts.matching(NSPredicate(format: "label CONTAINS 'snapshot composite'")).firstMatch.waitForExistence(timeout: 10))
        keep(app, "32-preview-hidden")
        // Long enough for one more picture with the outlines in it.
        try await Task.sleep(for: .seconds(4))
        XCTAssertTrue(app.staticTexts.matching(NSPredicate(format: "label CONTAINS 'snapshot composite'")).firstMatch.exists)
        app.buttons["captureClose"].tap()
    }

    /// The whole chain against the installed pack: recorded, saved, then pose recognition and the pack's
    /// own analysis page, out of sight. It needs a real climbing clip in place of the camera, which never
    /// enters this repository, so it is skipped unless one is supplied:
    ///
    ///     TEST_RUNNER_BB_CAPTURE_CLIP=<clip> xcodebuild … test -only-testing:BetaBuddyUITests/CaptureUITests/testRecordingIsSavedThenAnalysedByTheInstalledPack
    ///
    /// `TEST_RUNNER_BB_CAPTURE_SECONDS` is how long to record (default 13).
    @MainActor
    func testRecordingIsSavedThenAnalysedByTheInstalledPack() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let clip = environment["BB_CAPTURE_CLIP"], FileManager.default.fileExists(atPath: clip) else {
            throw XCTSkip("Optional private climbing clip is not supplied")
        }
        let app = XCUIApplication()
        // The simulator has no motion sensor; a phone standing nearly upright and still is given instead.
        app.launchArguments = ["-AppleLanguages", "(zh-Hans)", "-AppleLocale", "zh_CN", "-resetLanguage", "YES", "-initialTab", "1", "-debugCaptureVideo", clip,
                               "-debugCaptureAttitude", "6,1,0.01"]
        app.launch()
        XCTAssertTrue(app.buttons["recordClimb"].waitForExistence(timeout: 15))
        app.buttons["recordClimb"].tap()
        let record = app.buttons["captureRecord"]
        XCTAssertTrue(record.waitForExistence(timeout: 10))
        XCTAssertTrue(record.wait(for: \.isEnabled, toEqual: true, timeout: 15))
        try await Task.sleep(for: .seconds(6))                       // Let a capture page, if the pack has one, say something first.
        keep(app, "50-installed-pack-framing")
        record.tap()
        if app.buttons["recordAnyway"].waitForExistence(timeout: 2) { keep(app, "51-installed-pack-asked"); app.buttons["recordAnyway"].tap() }
        XCTAssertTrue(app.descendants(matching: .any)["captureElapsed"].waitForExistence(timeout: 5))
        try await Task.sleep(for: .seconds(Double(environment["BB_CAPTURE_SECONDS"] ?? "") ?? 13))
        app.buttons["captureStop"].tap()
        XCTAssertTrue(app.descendants(matching: .any)["captureSaved"].waitForExistence(timeout: 30))
        keep(app, "52-installed-pack-saved")

        let view = app.buttons["captureViewAnalysis"]
        guard GymPackInstalled.analysis(app, view) else {
            keep(app, "53-installed-pack-no-analysis")
            let message = app.staticTexts["captureAnalysisMessage"]
            XCTFail("No analysis result appeared. " + (message.exists ? message.label : ""))
            return
        }
        keep(app, "53-installed-pack-analysed")
        view.tap()
        XCTAssertTrue(app.buttons["analysisDone"].waitForExistence(timeout: 20))
        // Any web view's button: the capture page is a web view too, kept out of sight under this screen, and has none.
        XCTAssertTrue(app.webViews.buttons.firstMatch.waitForExistence(timeout: 60), "The saved result did not open")
        try await Task.sleep(for: .seconds(1))
        keep(app, "54-installed-pack-result")
        app.buttons["analysisDone"].tap()
        app.buttons["captureDone"].tap()
        XCTAssertTrue(app.buttons["recordClimb"].waitForExistence(timeout: 10))
        app.staticTexts["今天的攀爬"].firstMatch.tap()
        XCTAssertTrue(app.buttons["viewAnalysis"].waitForExistence(timeout: 10))
        removeNewestRecord(app, more: "更多操作", confirm: "删除这一趟")
    }

    /// A gym is only selected by itself when the directory knows where that gym is; otherwise the
    /// pill stays empty. When there is no distance to go by, the list says which of three reasons it is.
    @MainActor
    func testGymOffersOnFirstUseAndWhyThereIsNoDistance() async throws {
        // Nearest gym known only by its station: not chosen, and the list shows it as approximate.
        var app = try await launch(step: 2, location: "30.0115,120.0005")
        XCTAssertTrue(app.buttons["recordClimb"].waitForExistence(timeout: 15))
        app.buttons["recordClimb"].tap()
        XCTAssertTrue(app.buttons["captureGym"].waitForExistence(timeout: 15))
        try await Task.sleep(for: .seconds(3))                       // One fix from the system.
        XCTAssertTrue(app.buttons["captureGym"].label.contains("选择岩馆"), app.buttons["captureGym"].label)
        keep(app, "70-gym-guess-not-chosen")
        app.terminate()

        // Never asked: one tap away, in the list.
        app = try await launch(step: 2, location: "ask")
        app.buttons["recordClimb"].tap()
        XCTAssertTrue(app.buttons["captureGym"].waitForExistence(timeout: 15))
        app.buttons["captureGym"].tap()
        XCTAssertTrue(app.buttons["askLocation"].waitForExistence(timeout: 5))
        keep(app, "71-location-not-asked")
        app.terminate()

        // Refused: nothing on the screen; the list says so and how to change it.
        app = try await launch(step: 2, location: "refused")
        app.buttons["recordClimb"].tap()
        XCTAssertTrue(app.buttons["captureGym"].waitForExistence(timeout: 15))
        app.buttons["captureGym"].tap()
        XCTAssertTrue(app.staticTexts["locationRefused"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.buttons["打开系统设置"].exists)
        XCTAssertFalse(app.staticTexts["附近没有找到目录里的岩馆。"].exists)
        keep(app, "72-location-refused")
        app.terminate()

        // Allowed, but no position to be had: said as that, with another try.
        app = try await launch(step: 2, location: "unavailable")
        app.buttons["recordClimb"].tap()
        XCTAssertTrue(app.buttons["captureGym"].waitForExistence(timeout: 15))
        app.buttons["captureGym"].tap()
        XCTAssertTrue(app.staticTexts["locationUnavailable"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.buttons["再试一次"].exists)
        XCTAssertFalse(app.staticTexts["附近没有找到目录里的岩馆。"].exists)
        keep(app, "73-location-unavailable")
    }

    /// A clip that cannot be written into the album: the card says so, and closing asks before anything is dropped.
    /// A debug build is told to fail the next two saves.
    @MainActor
    func testAClipThatCannotBeSavedIsNotDroppedWithoutAsking() async throws {
        let app = try await launch(step: 2, extra: ["-debugFailSaves", "2"])
        XCTAssertTrue(app.buttons["recordClimb"].waitForExistence(timeout: 15))
        let before = app.staticTexts.matching(identifier: "今天的攀爬").count        // Whatever earlier runs left in the album.
        app.buttons["recordClimb"].tap()
        position(app, "机位可以，开始吧")
        app.buttons["captureRecord"].tap()
        XCTAssertTrue(app.descendants(matching: .any)["captureElapsed"].waitForExistence(timeout: 5))
        try await Task.sleep(for: .seconds(2))
        app.buttons["captureStop"].tap()
        // Not saved: the clip is held, with what can be done about it.
        XCTAssertTrue(app.staticTexts["这一趟还没存进攀爬册"].waitForExistence(timeout: 20))
        XCTAssertFalse(app.descendants(matching: .any)["captureSaved"].exists)
        keep(app, "80-save-failed")
        // Closing asks first.
        app.buttons["captureClose"].tap()
        let sheet = app.sheets.firstMatch
        XCTAssertTrue(sheet.buttons["discardAndClose"].waitForExistence(timeout: 5))
        keep(app, "81-save-failed-close-asks")
        XCTAssertTrue(sheet.buttons["再存一次"].exists, app.debugDescription)
        // Cancel: nothing happens, the clip is still held.
        cancelConfirmation(app, label: "取消")
        XCTAssertTrue(app.staticTexts["这一趟还没存进攀爬册"].waitForExistence(timeout: 5))
        // Retry from the question: it fails once more (the second arranged failure) and the clip is still held.
        app.buttons["captureClose"].tap()
        XCTAssertTrue(sheet.buttons["再存一次"].waitForExistence(timeout: 5))
        sheet.buttons["再存一次"].tap()
        XCTAssertTrue(app.staticTexts["这一趟还没存进攀爬册"].waitForExistence(timeout: 10))
        XCTAssertFalse(app.descendants(matching: .any)["captureSaved"].exists)
        // Discard: only now is the clip dropped, and the album has no record of it.
        app.buttons["captureClose"].tap()
        XCTAssertTrue(sheet.buttons["discardAndClose"].waitForExistence(timeout: 5))
        // Newer accessibility trees expose both the enclosing and inner button.
        sheet.buttons["discardAndClose"].firstMatch.tap()
        XCTAssertTrue(app.buttons["recordClimb"].waitForExistence(timeout: 10))
        XCTAssertEqual(app.staticTexts.matching(identifier: "今天的攀爬").count, before)
        // Nothing is left for the next launch to adopt either.
        app.terminate(); app.launch()
        XCTAssertTrue(app.buttons["recordClimb"].waitForExistence(timeout: 15))
        XCTAssertEqual(app.staticTexts.matching(identifier: "今天的攀爬").count, before)
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

private enum GymPackInstalled {
    /// Waits for pose recognition and the analysis after it; `false` when they ended without a result.
    @MainActor static func analysis(_ app: XCUIApplication, _ view: XCUIElement) -> Bool {
        let deadline = Date().addingTimeInterval(420)
        while Date() < deadline {
            if view.exists { return true }
            if app.staticTexts["captureAnalysisMessage"].exists, !app.descendants(matching: .any)["captureAnalysing"].exists,
               !app.descendants(matching: .any)["captureQueued"].exists { return view.waitForExistence(timeout: 3) }
            Thread.sleep(forTimeInterval: 1)
        }
        return false
    }
}
