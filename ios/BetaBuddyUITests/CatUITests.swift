import XCTest
import StoreKitTest

final class CatUITests: XCTestCase {
    @MainActor
    func testMemberCatReplayOnBothPresets() async throws {
        let ios = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        guard FileManager.default.fileExists(atPath: ios.appendingPathComponent("GymPack/cat.js").path) else {
            throw XCTSkip("Install the private cat pack for replay QA")
        }
        // The first problem the installed pack has cat plans for, so no route id is written into this repository.
        let plans = try String(contentsOf: ios.appendingPathComponent("GymPack/catplans.0.js"), encoding: .utf8)
        let route = try XCTUnwrap(plans.firstMatch(of: #/"routes":\{"([^"]+)"/#).map { String($0.1) }, "The installed pack has no cat plans")
        let session = try SKTestSession(contentsOf: ios.appendingPathComponent("StoreKit/Lifetime.storekit"))
        session.disableDialogs = true
        session.clearTransactions()
        defer { session.clearTransactions() }
        try await session.buyProduct(identifier: "com.betabuddy.ios.lifetime")
        let app = XCUIApplication()
        for preset in 0...1 {
            app.launchArguments = ["-AppleLanguages", "(en)", "-AppleLocale", "en_US", "-resetLanguage", "YES", "-gymNoticeSeen", "YES",
                                   "-gymQuery", "route=\(route)&species=cat&catp=\(preset)&catstrict=0&sheet=mini&hint=0"]
            app.launch()
            let page = app.webViews.firstMatch
            XCTAssertTrue(page.waitForExistence(timeout: 30))
            let play = page.buttons["Play"]
            XCTAssertTrue(play.waitForExistence(timeout: 30))
            XCTAssertTrue(play.wait(for: \.isEnabled, toEqual: true, timeout: 30))
            XCTAssertFalse(app.staticTexts["gymError"].exists)
            XCTAssertFalse(page.staticTexts["The cat data failed to load"].exists)
            play.tap()
            XCTAssertTrue(page.buttons["Pause"].waitForExistence(timeout: 5))
            let screenshot = XCTAttachment(screenshot: app.screenshot())
            screenshot.name = "Cat replay preset \(preset)"; screenshot.lifetime = .keepAlways; add(screenshot)
            app.terminate()
        }
    }
}
