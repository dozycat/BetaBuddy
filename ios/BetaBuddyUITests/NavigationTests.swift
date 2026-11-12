import XCTest

final class NavigationTests: XCTestCase {
    func testImportAndTrainingNavigation() {
        let app = XCUIApplication()
        app.launch()
        XCTAssertTrue(app.buttons["importClimb"].waitForExistence(timeout:10))
        app.buttons["importClimb"].tap()
        XCTAssertTrue(app.textFields["climbTitle"].waitForExistence(timeout:5))
        XCTAssertFalse(app.buttons["startAnalysis"].isEnabled)
        app.buttons["完成"].tap()
        app.buttons["攀爬册"].tap()
        XCTAssertTrue(app.textFields["找线路或难度"].exists)
        app.buttons["练习"].tap()
        app.staticTexts["让脚步安静下来"].tap()
        XCTAssertTrue(app.staticTexts["一个小练习"].waitForExistence(timeout:5))
    }
    // Works for both kinds of build: without a pack the entry explains itself; with one it opens the page.
    func testGymEntry() {
        let app = XCUIApplication()
        app.launch()
        XCTAssertTrue(app.buttons["岩馆"].waitForExistence(timeout:10))
        app.buttons["岩馆"].tap()
        if app.buttons["enterGym"].waitForExistence(timeout:3) {
            XCTAssertTrue(app.staticTexts["gymName"].exists)
            app.buttons["enterGym"].tap()
            XCTAssertTrue(app.webViews.firstMatch.waitForExistence(timeout:20))
            XCTAssertFalse(app.staticTexts["gymError"].exists)
        } else {
            XCTAssertTrue(app.staticTexts["这个构建没有\n内置岩馆数据"].exists)
            XCTAssertTrue(app.staticTexts["gymUnavailable"].exists)
        }
        app.buttons["上墙"].tap()
        XCTAssertTrue(app.buttons["importClimb"].waitForExistence(timeout:5))
    }
}
