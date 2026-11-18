import XCTest

/// End-to-end run of “分析这一趟” against the installed pack's analysis page. It needs a real climbing
/// clip, which never enters this repository, so it is skipped unless one is supplied:
///
///     TEST_RUNNER_BB_ANALYSIS_CLIP=<clip> xcodebuild … test -only-testing:BetaBuddyUITests/AnalysisUITests
///
/// `TEST_RUNNER_BB_ANALYSIS_SEEK=<label>` may give the label of the page's “show this moment in the
/// video” button in Simplified Chinese; the text belongs to the pack and is not written here.
final class AnalysisUITests: XCTestCase {
    func testAnalyseAttemptWithLocalClip() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let clip = environment["BB_ANALYSIS_CLIP"], FileManager.default.fileExists(atPath: clip) else {
            throw XCTSkip("Optional private climbing clip is not supplied")
        }
        let app = XCUIApplication()
        app.launchArguments = ["-AppleLanguages", "(zh-Hans)", "-AppleLocale", "zh_CN", "-resetLanguage", "YES", "-initialTab", "1", "-debugImportVideo", clip]
        app.launch()
        XCTAssertTrue(app.buttons["importClimb"].waitForExistence(timeout: 10))
        app.buttons["importClimb"].tap()
        let save = app.buttons["startAnalysis"]
        XCTAssertTrue(save.waitForExistence(timeout: 5))
        XCTAssertTrue(save.wait(for: \.isEnabled, toEqual: true, timeout: 20))
        if !save.isHittable { app.swipeUp() }
        save.tap()

        let analyse = app.buttons["analyseClimb"]
        guard analyse.waitForExistence(timeout: 20) else {
            delete(app)
            throw XCTSkip("The installed build has no pack with an analysis page; the entry is not shown")
        }
        if !analyse.isHittable { app.swipeUp() }
        keep(app, "Review with the analysis entry")
        analyse.tap()

        // Pose recognition runs first, then the page opens by itself.
        let done = app.buttons["analysisDone"]
        XCTAssertTrue(done.waitForExistence(timeout: 300), "The analysis page did not open after pose recognition")
        // The page's controls can only be reached once it shows a result.
        let page = app.webViews.firstMatch
        XCTAssertTrue(page.waitForExistence(timeout: 30))
        let seek = environment["BB_ANALYSIS_SEEK"].flatMap { $0.isEmpty ? nil : page.buttons[$0] }
        let control = seek ?? page.buttons.firstMatch
        XCTAssertTrue(control.waitForExistence(timeout: 300), "The analysis page did not show a result")
        Thread.sleep(forTimeInterval: 1)   // Let the page paint what it has just laid out before the screenshot.
        XCTAssertFalse(app.staticTexts["analysisError"].exists)
        keep(app, "Analysis page with a result")
        (seek ?? done).tap()
        // Back on the clip; the result was kept, so the entry now reads “看分析”.
        let view = app.buttons["viewAnalysis"]
        XCTAssertTrue(view.waitForExistence(timeout: 10))
        XCTAssertTrue(app.buttons["analyseAgain"].exists)
        XCTAssertFalse(done.exists)
        let clock = app.staticTexts["playbackClock"]
        if !clock.isHittable { app.swipeDown() }
        XCTAssertTrue(clock.waitForExistence(timeout: 5))
        let shown = XCTAttachment(string: clock.label); shown.name = "Playback clock after leaving the page"; shown.lifetime = .keepAlways; add(shown)
        keep(app, "Back on the clip")

        // The saved result opens without analysing again.
        if !view.isHittable { app.swipeUp() }
        view.tap()
        XCTAssertTrue(done.waitForExistence(timeout: 10))
        XCTAssertTrue(app.webViews.firstMatch.buttons.firstMatch.waitForExistence(timeout: 30))
        keep(app, "Saved result")
        done.tap()
        XCTAssertTrue(view.waitForExistence(timeout: 10))

        delete(app)
    }

    /// Leave the simulator as it was found.
    private func delete(_ app: XCUIApplication) {
        app.buttons["更多操作"].tap()
        app.buttons["deleteClimb"].tap()
        app.buttons["删除这一趟"].firstMatch.tap()
    }

    private func keep(_ app: XCUIApplication, _ name: String) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
