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
                                   location: String = "\(StandInPack.origin.latitude),\(StandInPack.origin.longitude)") async throws -> XCUIApplication {
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
    /// The line under the picture while it is at `stage`.
    @MainActor private func prompt(_ app: XCUIApplication, _ stage: String, timeout: TimeInterval = 10) -> XCUIElement {
        let element = app.descendants(matching: .any).matching(NSPredicate(format: "identifier == 'capturePrompt' AND value == %@", stage)).firstMatch
        XCTAssertTrue(element.waitForExistence(timeout: timeout), "the prompt never reached \(stage)")
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
        // Colours are swatches with a count; the page's names for them are not shown.
        XCTAssertTrue(app.buttons["captureColor-yellow"].waitForExistence(timeout: 5))
        XCTAssertEqual(app.buttons["captureColor-yellow"].label, "颜色 1"); XCTAssertEqual(app.buttons["captureColor-yellow"].value as? String, "6 个岩点")
        XCTAssertEqual(prompt(app, "colour").label, "选你要爬的颜色")
        keep(app, "01-blocked")
        XCTAssertEqual(hint(app, "cut_top").label, "上面出画了，往后退一点")
        keep(app, "02-blocked-with-move")
        XCTAssertEqual(hint(app, "lens_flare_2031").label, "换个机位试试")
        keep(app, "03-unknown-code")
        let ready = app.descendants(matching: .any)["captureReady"]
        XCTAssertTrue(ready.waitForExistence(timeout: 40))
        keep(app, "04-ready")

        // Gym: only gyms whose own position is known are ranked, nearest first; a station-level position
        // is shown as approximate; the rest of the city follows without distances.
        app.buttons["captureGym"].tap()
        let east = app.buttons["nearbyGym-syn:east"]
        XCTAssertTrue(east.waitForExistence(timeout: 5))
        let rows = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH 'nearbyGym-'")).allElementsBoundByIndex.map(\.identifier)
        XCTAssertEqual(rows, ["nearbyGym-syn:east", "nearbyGym-syn:station", "nearbyGym-syn:west"])
        XCTAssertTrue(east.label.contains("200"), east.label)
        XCTAssertTrue(app.buttons["nearbyGym-syn:station"].label.contains("约 1.5"), app.buttons["nearbyGym-syn:station"].label)
        XCTAssertTrue(app.staticTexts["© OpenStreetMap contributors"].exists)
        keep(app, "05-gym-by-distance")
        app.swipeUp()
        let district = app.buttons["cityGym-syn:district"]
        XCTAssertTrue(district.waitForExistence(timeout: 5))
        XCTAssertNil(district.label.rangeOfCharacter(from: .decimalDigits), district.label)
        XCTAssertTrue(app.buttons["allGyms"].waitForExistence(timeout: 5))
        keep(app, "05b-gym-same-city")
        app.buttons["取消"].tap()
        // The gym right here is offered, not chosen: the selection only changes when the user says so.
        XCTAssertTrue(app.buttons["captureGym"].label.contains("选择岩馆"), app.buttons["captureGym"].label)
        XCTAssertTrue(app.buttons["useNearestGym"].waitForExistence(timeout: 5))
        keep(app, "05c-nearest-gym-offered")
        app.buttons["useNearestGym"].tap()
        XCTAssertTrue(app.buttons["captureGym"].label.contains(StandInPack.directoryGymName))
        XCTAssertFalse(app.buttons["useNearestGym"].exists)

        // A gym that is one of the pack's venues starts the page's judgement over; this time, record against its advice.
        _ = hint(app, "too_dark")
        app.buttons["captureRecord"].tap()
        let anyway = app.buttons["recordAnyway"]
        XCTAssertTrue(anyway.waitForExistence(timeout: 5))
        keep(app, "06-ask-before-recording")
        app.buttons["再调一下机位"].tap()
        XCTAssertFalse(app.descendants(matching: .any)["captureElapsed"].exists)
        XCTAssertTrue(ready.waitForExistence(timeout: 60))

        // A colour: its holds are outlined, the guessed start and finish carry a question mark, and the
        // screen asks for the start first. Marking is one tap on the picture, with what it marks chosen below.
        app.buttons["captureColor-yellow"].tap()
        XCTAssertTrue(app.buttons["captureTool-start_hand"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.buttons["captureTool-start_hand"].isSelected)
        // The gym is one of the synthetic pack's venues, so the stand-in names a problem; it is shown as a guess that can be turned down.
        let route = app.staticTexts["captureRoute"]
        XCTAssertTrue(route.waitForExistence(timeout: 5))
        XCTAssertEqual(route.label, "认出来像是：合成线路 黄 1 · V3")
        XCTAssertTrue(app.buttons["captureRejectRoute"].exists)
        // With a recognised problem the start is not a guess at the picture, and the screen does not call it one.
        XCTAssertEqual(prompt(app, "route").label, "起步和结束是按认出的这条线标的。不对的话，点一下岩点改。")
        // No grade picked yet: the problem's grade is filled in, and the screen says that it was.
        XCTAssertTrue(app.staticTexts["captureGradePrefilled"].waitForExistence(timeout: 5))
        XCTAssertEqual(app.buttons["captureGrade"].value as? String, "V3")
        try await Task.sleep(for: .seconds(1))                       // The outlines come with the next assessment.
        keep(app, "07-colour-guessed-start-finish")
        // “Not this one” moves to the page's next candidate; its grade is not a V grade and is only shown, not filled in.
        app.buttons["captureRejectRoute"].tap()
        let next = app.staticTexts.matching(NSPredicate(format: "identifier == 'captureRoute' AND label CONTAINS '黄 2'")).firstMatch
        XCTAssertTrue(next.waitForExistence(timeout: 5))
        XCTAssertEqual(next.label, "认出来像是：合成线路 黄 2 · 紫条 11–12")
        XCTAssertFalse(app.staticTexts["captureGradePrefilled"].exists)
        XCTAssertEqual(app.buttons["captureGrade"].value as? String, "未标级")
        keep(app, "07b-next-candidate")
        // The user's own grade.
        app.buttons["captureGrade"].tap()
        XCTAssertTrue(app.buttons["V4"].waitForExistence(timeout: 5))
        app.buttons["V4"].tap()
        // Turned down as well: nothing is recognised any more, and the start is a guess again.
        app.buttons["captureRejectRoute"].tap()
        XCTAssertEqual(prompt(app, "start").label, "虚线是自动猜的，起步点常常猜错。点一下起步的手点。")
        XCTAssertFalse(app.staticTexts["captureRoute"].exists)
        XCTAssertEqual(app.buttons["captureGrade"].value as? String, "V4")

        let picture = app.descendants(matching: .any)["capturePicture"]
        picture.coordinate(withNormalizedOffset: CGVector(dx: 0.30, dy: 0.80)).tap()
        XCTAssertEqual(prompt(app, "finish").label, "起步标好了。选“结束”，再点一下结束点。")
        keep(app, "08-start-marked")
        app.buttons["captureTool-finish"].tap()
        picture.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.6)).tap()
        XCTAssertEqual(prompt(app, "marked").label, "起步和结束都标好了。其余虚线仍是自动猜的。")
        XCTAssertTrue(app.buttons["captureResetMarks"].exists)
        // A tap on bare wall marks nothing and says so.
        picture.coordinate(withNormalizedOffset: CGVector(dx: 0.93, dy: 0.5)).tap()
        XCTAssertEqual(prompt(app, "missed", timeout: 3).label, "这里没有认出岩点。")
        _ = prompt(app, "marked")
        keep(app, "09-start-and-finish-marked")

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
        XCTAssertEqual(app.buttons["captureColor-red"].label, "Colour 3"); XCTAssertEqual(app.buttons["captureColor-red"].value as? String, "3 holds")
        app.buttons["captureColor-red"].tap()
        // Nothing blocks. The start is not known for this colour, and the screen asks for it; recording is allowed all the same.
        XCTAssertEqual(prompt(app, "start").label, "Dashed outlines are automatic guesses, and the start is often guessed wrong. Tap the start hand holds.")
        XCTAssertTrue(app.descendants(matching: .any)["captureReady"].exists)
        XCTAssertTrue(app.buttons["captureRecord"].isEnabled)
        XCTAssertFalse(app.staticTexts["captureRoute"].exists)          // No gym chosen: nothing to recognise a problem in.
        keep(app, "30-ready-start-not-marked-en")
        // Back to the swatches to choose again.
        app.buttons["captureChangeColour"].tap()
        XCTAssertTrue(app.buttons["captureColor-blue"].waitForExistence(timeout: 5))
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
        // With a capture page: choose the first colour it offers. No hold is marked here: a blind tap marks
        // whatever hold lies there as the start, the plan carries that, and the analysis then rightly finds no climb from it.
        let swatch = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH 'captureColor-'")).firstMatch
        if swatch.waitForExistence(timeout: 5) {
            swatch.tap()
            XCTAssertTrue(app.buttons["captureTool-start_hand"].waitForExistence(timeout: 5))
            try await Task.sleep(for: .seconds(2))
            keep(app, "50b-installed-pack-colour")
        }
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

    /// The real directory with a real (simulated) position, and the installed pack recognising a problem
    /// in one of its own gyms. Needs a clip shot in such a gym and that gym's id in the directory, and the
    /// simulator told where it is beforehand:
    ///
    ///     xcrun simctl privacy <device> grant location com.betabuddy.ios
    ///     xcrun simctl location <device> set <lat>,<lon>
    ///     TEST_RUNNER_BB_CAPTURE_CLIP=<clip> TEST_RUNNER_BB_CAPTURE_GYM_ID=<directory id> xcodebuild … test \
    ///       -only-testing:BetaBuddyUITests/CaptureUITests/testInstalledPackRecognisesAProblemInOneOfItsGyms
    @MainActor
    func testInstalledPackRecognisesAProblemInOneOfItsGyms() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let clip = environment["BB_CAPTURE_CLIP"], FileManager.default.fileExists(atPath: clip),
              let gym = environment["BB_CAPTURE_GYM_ID"], !gym.isEmpty else {
            throw XCTSkip("Optional private clip from one of the pack's gyms, and that gym's directory id, are not supplied")
        }
        let app = XCUIApplication()
        app.launchArguments = ["-AppleLanguages", "(zh-Hans)", "-AppleLocale", "zh_CN", "-resetLanguage", "YES", "-initialTab", "1", "-debugCaptureVideo", clip,
                               "-debugCaptureAttitude", "6,1,0.01", "-debugCaptureFieldOfView", "50"]
        app.launch()
        XCTAssertTrue(app.buttons["recordClimb"].waitForExistence(timeout: 15))
        app.buttons["recordClimb"].tap()
        XCTAssertTrue(app.buttons["captureGym"].waitForExistence(timeout: 10))
        try await Task.sleep(for: .seconds(3))                       // One fix from the system.
        keep(app, "60-real-directory-framing")
        app.buttons["captureGym"].tap()
        XCTAssertTrue(app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH 'nearbyGym-'")).firstMatch.waitForExistence(timeout: 15),
                      "No gym was ranked by distance: is location granted and set for this simulator?")
        keep(app, "61-real-directory-nearby")
        app.swipeUp()
        keep(app, "62-real-directory-same-city")
        app.swipeDown()
        let row = app.buttons.matching(NSPredicate(format: "identifier == %@ OR identifier == %@", "nearbyGym-" + gym, "cityGym-" + gym)).firstMatch
        if !row.waitForExistence(timeout: 3) { app.swipeUp() }
        XCTAssertTrue(row.waitForExistence(timeout: 5), "The gym is not among those suggested here")
        row.tap()
        let swatch = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH 'captureColor-'")).firstMatch
        XCTAssertTrue(swatch.waitForExistence(timeout: 20))
        (environment["BB_CAPTURE_COLOUR"].map { app.buttons["captureColor-" + $0] } ?? swatch).tap()
        let route = app.staticTexts["captureRoute"]
        let found = route.waitForExistence(timeout: 30)
        try await Task.sleep(for: .seconds(1))
        keep(app, "63-real-route-recognised")
        XCTAssertTrue(found, "The page recognised no problem")
        let label = XCTAttachment(string: route.label); label.name = "Recognised problem as shown"; label.lifetime = .keepAlways; add(label)
        app.buttons["captureClose"].tap()
    }

    /// With no gym chosen yet, the distance recommendation is on the screen itself; and when there is
    /// no distance to go by, the list says which of three reasons it is.
    @MainActor
    func testGymOffersOnFirstUseAndWhyThereIsNoDistance() async throws {
        // Nearest gym known only by its station: offered as a guess, not chosen.
        var app = try await launch(step: 2, location: "30.0115,120.0005")
        XCTAssertTrue(app.buttons["recordClimb"].waitForExistence(timeout: 15))
        app.buttons["recordClimb"].tap()
        let text = app.staticTexts["gymOfferText"]
        XCTAssertTrue(text.waitForExistence(timeout: 15))
        XCTAssertEqual(text.label, "可能在：只知道最近车站的岩馆")
        XCTAssertTrue(app.buttons["captureGym"].label.contains("选择岩馆"))
        XCTAssertEqual(app.buttons["useNearestGym"].label, "选这家")
        keep(app, "70-gym-guess-offered")
        app.buttons["declineNearestGym"].tap()
        XCTAssertFalse(text.waitForExistence(timeout: 1))
        XCTAssertTrue(app.buttons["captureGym"].label.contains("选择岩馆"))
        app.terminate()

        // Never asked: one tap away, on the screen and in the list.
        app = try await launch(step: 2, location: "ask")
        app.buttons["recordClimb"].tap()
        XCTAssertTrue(app.buttons["askLocationHere"].waitForExistence(timeout: 15))
        keep(app, "71-location-not-asked")
        app.buttons["captureGym"].tap()
        XCTAssertTrue(app.buttons["askLocation"].waitForExistence(timeout: 5))
        app.terminate()

        // Refused: nothing on the screen; the list says so and how to change it.
        app = try await launch(step: 2, location: "refused")
        app.buttons["recordClimb"].tap()
        XCTAssertTrue(app.buttons["captureGym"].waitForExistence(timeout: 15))
        XCTAssertFalse(app.buttons["askLocationHere"].exists); XCTAssertFalse(app.staticTexts["gymOfferText"].exists)
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
