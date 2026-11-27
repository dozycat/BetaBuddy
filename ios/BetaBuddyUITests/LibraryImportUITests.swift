import XCTest

final class LibraryImportUITests: XCTestCase {
    @MainActor
    func testOptInExplainsScopeAndDoesNotEnableItself() {
        let app = XCUIApplication()
        app.launchArguments = ["-AppleLanguages", "(zh-Hans)", "-AppleLocale", "zh_CN", "-resetLanguage", "YES",
                               "-initialTab", "1", "-automaticPhotoImport", "NO"]
        app.launch()
        XCTAssertTrue(app.buttons["libraryImport"].waitForExistence(timeout: 10))
        app.buttons["libraryImport"].tap()
        let toggle = app.switches["libraryImportToggle"]
        XCTAssertTrue(toggle.waitForExistence(timeout: 5))
        XCTAssertEqual(toggle.value as? String, "0")
        XCTAssertTrue(app.staticTexts["把相册里的攀爬收好"].exists)
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = "Library import opt-in"; attachment.lifetime = .keepAlways; add(attachment)
        app.buttons["完成"].tap()
        XCTAssertTrue(app.buttons["libraryImport"].exists)
    }
}
