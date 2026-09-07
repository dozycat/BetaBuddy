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
}
