import XCTest

final class NavigationTests: XCTestCase {
    func testBothEntriesInSimplifiedChinese() { walk(language:"zh-Hans", locale:"zh_CN", gym:"岩馆", logbook:"攀爬册", about:"关于 BetaBuddy", done:"完成") }
    func testBothEntriesInEnglish() { walk(language:"en", locale:"en_US", gym:"Gym", logbook:"Logbook", about:"About BetaBuddy", done:"Done") }

    // Works with or without a gym pack: without one the first entry explains itself; with one it opens the page.
    private func walk(language: String, locale: String, gym: String, logbook: String, about: String, done: String) {
        let app = XCUIApplication()
        app.launchArguments = ["-AppleLanguages","(\(language))","-AppleLocale",locale]
        app.launch()
        // The app opens on the gym.
        XCTAssertTrue(app.buttons["tabGym"].waitForExistence(timeout:10))
        XCTAssertEqual(app.buttons["tabGym"].label,gym)
        XCTAssertEqual(app.buttons["tabJournal"].label,logbook)
        XCTAssertTrue(app.buttons["tabGym"].isSelected)
        if !app.staticTexts["gymUnavailable"].waitForExistence(timeout:3) {
            // The notice is shown only until it has been accepted once on this device.
            if app.buttons["enterGym"].exists {
                XCTAssertTrue(app.staticTexts["gymName"].exists)
                app.buttons["enterGym"].tap()
            }
            XCTAssertTrue(app.webViews.firstMatch.waitForExistence(timeout:20))
            XCTAssertFalse(app.staticTexts["gymError"].exists)
        }
        // Logbook: import lives here.
        app.buttons["tabJournal"].tap()
        XCTAssertTrue(app.buttons["importClimb"].waitForExistence(timeout:5))
        app.buttons["importClimb"].tap()
        XCTAssertTrue(app.textFields["climbTitle"].waitForExistence(timeout:5))
        XCTAssertFalse(app.buttons["startAnalysis"].isEnabled)
        XCTAssertEqual(app.buttons["importDone"].label,done)
        app.buttons["importDone"].tap()
        XCTAssertTrue(app.buttons["about"].waitForExistence(timeout:5))
        app.buttons["about"].tap()
        XCTAssertTrue(app.staticTexts[about].waitForExistence(timeout:5))
        app.navigationBars.buttons.firstMatch.tap()
        app.buttons["tabGym"].tap()
        XCTAssertTrue(app.buttons["tabGym"].isSelected)
    }
}
