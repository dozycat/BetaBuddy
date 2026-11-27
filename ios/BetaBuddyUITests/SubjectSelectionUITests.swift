import XCTest

final class SubjectSelectionUITests: XCTestCase {
    @MainActor
    func testOutlineSelectionSaveAndReopen() throws {
        guard let clip = ProcessInfo.processInfo.environment["BB_SUBJECT_CLIP"] else { throw XCTSkip("Optional private video not supplied") }
        continueAfterFailure = false
        let app = XCUIApplication(), title = "Subject-" + UUID().uuidString.prefix(6)
        app.launchArguments = ["-AppleLanguages", "(zh-Hans)", "-AppleLocale", "zh_CN", "-resetLanguage", "YES", "-initialTab", "1", "-automaticPhotoImport", "NO", "-debugImportVideo", clip]
        app.launch()
        XCTAssertTrue(app.buttons["importClimb"].waitForExistence(timeout: 10))
        app.buttons["importClimb"].tap()
        let field = app.textFields["climbTitle"]
        XCTAssertTrue(field.waitForExistence(timeout: 10)); field.tap(); field.typeText(String(title))
        let save = app.buttons["saveClimb"]
        XCTAssertTrue(save.wait(for: \.isEnabled, toEqual: true, timeout: 20)); save.tap()
        XCTAssertTrue(app.staticTexts[String(title)].waitForExistence(timeout: 20)); app.staticTexts[String(title)].tap()
        let choose = app.buttons["chooseSubject"]
        XCTAssertTrue(choose.waitForExistence(timeout: 10))
        if !choose.isHittable { app.swipeUp() }; choose.tap()
        let people = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH 'subject-'"))
        if let region = ProcessInfo.processInfo.environment["BB_SUBJECT_REGION"],
           let bounds = try JSONSerialization.jsonObject(with: Data(region.utf8)) as? [String: Double] {
            let draw = app.buttons["drawSubjectRegion"]
            XCTAssertTrue(draw.wait(for: \.isEnabled, toEqual: true, timeout: 60))
            if !draw.isHittable { app.swipeUp() }; draw.tap()
            let canvas = app.otherElements["subjectRegionCanvas"]
            XCTAssertTrue(canvas.waitForExistence(timeout: 10))
            let aspect = Double(ProcessInfo.processInfo.environment["BB_SUBJECT_ASPECT"] ?? "0.56666667")!
            let width = min(canvas.frame.width, canvas.frame.height*aspect), height = width/aspect
            let x = (canvas.frame.width-width)/2, y = (canvas.frame.height-height)/2
            let start = canvas.coordinate(withNormalizedOffset: CGVector(dx: (x+width*bounds["x"]!)/canvas.frame.width, dy: (y+height*bounds["y"]!)/canvas.frame.height))
            let end = canvas.coordinate(withNormalizedOffset: CGVector(dx: (x+width*(bounds["x"]!+bounds["width"]!))/canvas.frame.width, dy: (y+height*(bounds["y"]!+bounds["height"]!))/canvas.frame.height))
            start.press(forDuration: 0.1, thenDragTo: end)
        } else {
            XCTAssertTrue(people.firstMatch.waitForExistence(timeout: 60))
            people.firstMatch.tap()
        }
        XCTAssertTrue(app.buttons["saveSubject"].wait(for: \.isEnabled, toEqual: true, timeout: 120))
        let selected = XCTAttachment(screenshot: app.screenshot())
        selected.name = "Subject outline on the original video"; selected.lifetime = .keepAlways; add(selected)
        app.buttons["saveSubject"].tap()
        XCTAssertTrue(choose.waitForExistence(timeout: 10)); if !choose.isHittable { app.swipeUp() }; choose.tap()
        XCTAssertTrue(app.staticTexts["已按你的选择跟踪。"].waitForExistence(timeout: 15))
        app.buttons["取消"].tap()
        app.buttons["更多操作"].tap(); app.buttons["deleteClimb"].tap(); app.buttons["删除这一趟"].tap()
        XCTAssertFalse(app.staticTexts[String(title)].waitForExistence(timeout: 2))
    }
}
