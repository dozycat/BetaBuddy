import XCTest

final class AlbumUITests: XCTestCase {
    func testFolderCreateRenameDeleteSurvivesRelaunch() {
        let app = XCUIApplication()
        app.launchArguments = ["-AppleLanguages", "(zh-Hans)", "-AppleLocale", "zh_CN", "-resetLanguage", "YES", "-initialTab", "1"]
        app.launch()
        XCTAssertTrue(app.buttons["manageFolders"].waitForExistence(timeout: 10))
        app.buttons["manageFolders"].tap()
        app.buttons["createFolder"].tap()
        let name = "UI-" + UUID().uuidString.prefix(6)
        app.textFields["folderName"].tap()
        app.textFields["folderName"].typeText(String(name))
        app.buttons["saveFolder"].tap()
        XCTAssertTrue(app.staticTexts[String(name)].waitForExistence(timeout: 3))
        app.buttons["folderMenu-\(name)"].tap()
        app.buttons["重命名"].tap()
        let field = app.textFields["folderName"]
        field.tap(); field.typeText(String(repeating: XCUIKeyboardKey.delete.rawValue, count: name.count) + "Session-1")
        app.buttons["saveFolder"].tap()
        XCTAssertTrue(app.staticTexts["Session-1"].waitForExistence(timeout: 3))
        app.buttons["完成"].tap()
        let attachment = XCTAttachment(screenshot: app.screenshot()); attachment.name = "Album folders"; attachment.lifetime = .keepAlways; add(attachment)
        app.terminate(); app.launch()
        XCTAssertTrue(app.buttons["manageFolders"].waitForExistence(timeout: 10))
        app.buttons["manageFolders"].tap()
        XCTAssertTrue(app.staticTexts["Session-1"].exists)
        app.buttons["folderMenu-Session-1"].tap(); app.buttons["删除文件夹"].tap()
        app.buttons["删除文件夹"].tap()
        XCTAssertFalse(app.staticTexts["Session-1"].exists)
    }
}
