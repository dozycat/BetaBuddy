import XCTest

final class LibraryImportUITests: XCTestCase {
    @MainActor
    func testOptionalLimitedPhotoPermissionFlow() throws {
        guard let mode = ProcessInfo.processInfo.environment["BB_PHOTOS_PERMISSION"],
              ["limited", "expand", "revoked"].contains(mode) else { throw XCTSkip("Dedicated simulator permission fixture not supplied") }
        continueAfterFailure = false
        let app = XCUIApplication()
        app.launchArguments = ["-AppleLanguages", "(en)", "-AppleLocale", "en_US", "-resetLanguage", "YES",
                               "-initialTab", "1", "-automaticPhotoImport", mode == "limited" ? "NO" : "YES"]
        if ProcessInfo.processInfo.environment["BB_LARGE_TEXT"] == "1" { app.launchArguments += ["-UIPreferredContentSizeCategoryName", "UICTContentSizeCategoryAccessibilityXXXL"] }
        app.launch()
        XCTAssertTrue(app.buttons["libraryImport"].waitForExistence(timeout: 10))
        app.buttons["libraryImport"].tap()
        if mode == "revoked" {
            XCTAssertTrue(app.staticTexts["Photo access required"].waitForExistence(timeout: 5))
            XCTAssertTrue(app.buttons["Open Settings"].exists)
            XCTAssertFalse(app.otherElements["libraryImportStatus"].exists)
        } else {
            if mode == "limited" {
                XCTAssertEqual(app.switches["libraryImportToggle"].value as? String, "0")
                app.switches["libraryImportToggle"].tap()
                let system = XCUIApplication(bundleIdentifier: "com.apple.springboard")
                XCTAssertTrue(system.buttons["kTCCServicePhotos-limited-button"].firstMatch.waitForExistence(timeout: 5))
                system.buttons["kTCCServicePhotos-limited-button"].firstMatch.tap()
            } else {
                XCTAssertTrue(app.buttons["libraryAllowedVideos"].waitForExistence(timeout: 10))
                app.buttons["libraryAllowedVideos"].tap()
            }
            let seconds = ProcessInfo.processInfo.environment["BB_PHOTOS_PICK_SECONDS"] ?? "five"
            let label = mode == "limited" ? "Video, thirty seconds," : "Video, \(seconds) seconds,"
            let video = app.images.matching(NSPredicate(format: "label BEGINSWITH %@", label)).firstMatch
            XCTAssertTrue(video.waitForExistence(timeout: 10))
            video.tap()
            app.buttons["Update"].tap()
            XCTAssertTrue(app.buttons["libraryAllowedVideos"].waitForExistence(timeout: 10))
            XCTAssertTrue(app.buttons["libraryScan"].waitForExistence(timeout: 60))
        }
        XCUIDevice.shared.press(.home)
        app.activate()
        XCTAssertTrue((mode == "revoked" ? app.staticTexts["Photo access required"] : app.buttons["libraryAllowedVideos"]).waitForExistence(timeout: 10))
        if mode == "revoked" { XCTAssertFalse(app.otherElements["libraryImportStatus"].exists) }
        let attachment = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
        attachment.name = "Photo permission \(mode)"; attachment.lifetime = .keepAlways; add(attachment)
        app.terminate()
    }
    @MainActor
    func testOptInStaysOffUntilRequested() {
        let app = XCUIApplication()
        app.launchArguments = ["-AppleLanguages", "(zh-Hans)", "-AppleLocale", "zh_CN", "-resetLanguage", "YES",
                               "-initialTab", "1", "-automaticPhotoImport", "NO"]
        if ProcessInfo.processInfo.environment["BB_LARGE_TEXT"] == "1" { app.launchArguments += ["-UIPreferredContentSizeCategoryName", "UICTContentSizeCategoryAccessibilityXXXL"] }
        app.launch()
        XCTAssertTrue(app.buttons["libraryImport"].waitForExistence(timeout: 10))
        app.buttons["libraryImport"].tap()
        let toggle = app.switches["libraryImportToggle"]
        XCTAssertTrue(toggle.waitForExistence(timeout: 5))
        XCTAssertEqual(toggle.value as? String, "0")
        XCTAssertTrue(app.navigationBars["扫相册"].exists)
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = "Library import opt-in"; attachment.lifetime = .keepAlways; add(attachment)
        app.buttons["完成"].tap()
        XCTAssertTrue(app.buttons["libraryImport"].exists)
    }
}
