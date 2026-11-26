import XCTest

/// The way to the edit page and the page itself, with the synthetic pack: its analysis stand-in hands
/// back a made-up render plan, the clip is a drawn wall with a photo of a person on it. What the edit
/// looks like is not judged here; `EditRendererTests` checks the rendering.
final class EditUITests: XCTestCase {
    private var work: URL!
    private var clip: URL?

    override func setUp() {
        continueAfterFailure = false
        work = FileManager.default.temporaryDirectory.appendingPathComponent("edit-ui-\(UUID())")
        try? FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
    }
    override func tearDown() {
        try? FileManager.default.removeItem(at: work)
        if let clip { try? FileManager.default.removeItem(at: clip) }
    }
    @MainActor private func keep(_ app: XCUIApplication, _ name: String) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name; attachment.lifetime = .keepAlways
        add(attachment)
    }

    @MainActor
    func testDisciplineGradeScaleAndTheEditPage() async throws {
        try StandInPack.write(to: work)
        let photo = try XCTUnwrap(Bundle(for: Self.self).url(forResource: "pose", withExtension: "jpg").flatMap { UIImage(contentsOfFile: $0.path)?.cgImage })
        let clip = try await SyntheticWallClip.make(width: 360, height: 640, seconds: 3, photo: photo)
        self.clip = clip
        let app = XCUIApplication()
        app.launchArguments = ["-AppleLanguages", "(zh-Hans)", "-AppleLocale", "zh_CN", "-resetLanguage", "YES", "-initialTab", "1",
                               "-debugGymPackRoot", work.path, "-debugImportVideo", clip.path]
        app.launch()
        XCTAssertTrue(app.buttons["importClimb"].waitForExistence(timeout: 10))
        app.buttons["importClimb"].tap()
        let title = "Edit-" + UUID().uuidString.prefix(6)
        XCTAssertTrue(app.textFields["climbTitle"].waitForExistence(timeout: 5))
        app.textFields["climbTitle"].tap()
        app.textFields["climbTitle"].typeText(String(title) + "\n")

        // Three ways to climb; the grade picker follows the one chosen.
        let style = app.segmentedControls["climbStyle"]
        XCTAssertTrue(style.waitForExistence(timeout: 5))
        XCTAssertTrue(style.buttons["抱石"].isSelected)
        style.buttons["先锋"].tap()
        XCTAssertTrue(style.buttons["先锋"].wait(for: \.isSelected, toEqual: true, timeout: 5))
        let grade = app.buttons.matching(NSPredicate(format: "label BEGINSWITH '线路难度'")).firstMatch
        XCTAssertTrue(grade.waitForExistence(timeout: 5))
        grade.tap()
        XCTAssertTrue(app.buttons["5.10a"].waitForExistence(timeout: 5))
        XCTAssertFalse(app.buttons["V4"].exists)
        keep(app, "01-import-lead-grades")
        app.buttons["5.10a"].tap()

        let save = app.buttons["saveClimb"]
        XCTAssertTrue(save.wait(for: \.isEnabled, toEqual: true, timeout: 20))
        save.tap()
        XCTAssertTrue(app.staticTexts[String(title)].waitForExistence(timeout: 20))
        XCTAssertTrue(app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH '5.10a'")).firstMatch.exists)
        app.staticTexts[String(title)].tap()
        XCTAssertTrue(app.staticTexts["playbackClock"].waitForExistence(timeout: 10))
        XCTAssertFalse(app.buttons["autoEdit"].exists)                    // No analysis yet, so no plan to edit from.

        // Analyse: pose recognition first, then the stand-in page, which hands back a result and a plan.
        let analyse = app.buttons["analyseClimb"]
        if !analyse.isHittable { app.swipeUp() }
        analyse.tap()
        XCTAssertTrue(app.buttons["analysisDone"].waitForExistence(timeout: 120))
        // The button appears under the analysis page as soon as the plan is stored; wait for the plan,
        // then close the page before using what is behind it.
        let edit = app.buttons["autoEdit"], done = app.buttons["analysisDone"]
        XCTAssertTrue(edit.waitForExistence(timeout: 60))
        for _ in 0..<10 where done.exists { done.tap(); _ = done.waitForNonExistence(timeout: 3) }
        XCTAssertFalse(done.exists)
        XCTAssertTrue(app.buttons["viewAnalysis"].exists)
        if !edit.isHittable { app.swipeUp() }
        keep(app, "02-review-with-edit")
        edit.tap()

        // The page: the switches on offer that the plan has data for (passer-by removal is not offered yet),
        // and either the result or why there is none.
        let share = app.buttons["editShare"], error = app.staticTexts["editError"]
        XCTAssertTrue(app.buttons["editSlow"].waitForExistence(timeout: 10))
        XCTAssertTrue(app.buttons["editCamera"].exists)
        XCTAssertFalse(app.buttons["editRemove"].exists)
        XCTAssertTrue(app.staticTexts["自动剪辑，可能剪错。"].exists)
        let deadline = Date().addingTimeInterval(120)
        while !share.exists && !error.exists && Date() < deadline { try await Task.sleep(for: .milliseconds(200)) }
        keep(app, "03-edit-first-render")
        XCTAssertFalse(error.exists)
        XCTAssertTrue(share.waitForExistence(timeout: 120))
        XCTAssertTrue(app.descendants(matching: .any)["editResult"].exists)
        keep(app, "04-edit-result")
        // Changing a switch renders again.
        app.buttons["editSlow"].tap()
        XCTAssertTrue(app.buttons["editSlow"].wait(for: \.isSelected, toEqual: false, timeout: 5))
        XCTAssertTrue(share.waitForExistence(timeout: 120))
        keep(app, "05-edit-without-slow-motion")
        app.buttons["editDone"].tap()
        XCTAssertTrue(app.staticTexts["playbackClock"].waitForExistence(timeout: 10))

        // The discipline chosen at import is shown when organising, with its own scale.
        app.buttons["更多操作"].tap()
        app.buttons["organizeClimb"].tap()
        XCTAssertTrue(app.segmentedControls["climbStyle"].buttons["先锋"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.segmentedControls["climbStyle"].buttons["先锋"].wait(for: \.isSelected, toEqual: true, timeout: 5))
        keep(app, "06-organise")
        app.buttons["取消"].tap()
        app.buttons["更多操作"].tap()
        app.buttons["deleteClimb"].tap()
        let confirm = app.sheets.buttons["删除这一趟"]
        XCTAssertTrue(confirm.waitForExistence(timeout: 5))
        confirm.tap()
        XCTAssertTrue(app.buttons["importClimb"].waitForExistence(timeout: 10))
    }
}
