import XCTest

final class NavigationTests: XCTestCase {
    func testBothEntriesInSimplifiedChinese() { walk(language:"zh-Hans", locale:"zh_CN", gym:"岩馆", logbook:"攀爬册", settings:"设置与关于", done:"完成") }
    func testBothEntriesInEnglish() { walk(language:"en", locale:"en_US", gym:"Gym", logbook:"Logbook", settings:"Settings and About", done:"Done") }

    func testMembershipOfferKeepsCoreFeaturesFree() {
        let app = launch("en", "en_US", reset:true)
        app.buttons["tabJournal"].tap()
        app.buttons["settings"].tap()
        let details = app.buttons["membershipDetails"]
        XCTAssertTrue(details.waitForExistence(timeout: 10))
        if !details.isHittable { app.swipeUp() }
        details.tap()
        XCTAssertTrue(app.staticTexts["Lifetime member badge in Logbook"].waitForExistence(timeout:5))
        XCTAssertTrue(app.buttons["buyLifetime"].exists || app.buttons["retryMembershipProduct"].exists)
        XCTAssertTrue(app.staticTexts["Gym and Logbook remain free to use; membership is optional."].exists)
        XCTAssertTrue(app.buttons["restoreLifetime"].exists)
        let offer = XCTAttachment(screenshot: app.screenshot())
        offer.name = "Membership offer"
        offer.lifetime = .keepAlways
        add(offer)
        app.buttons["membershipDone"].tap()
        app.buttons["settingsDone"].tap()
        XCTAssertTrue(app.buttons["importClimb"].exists)
    }

    private func launch(_ language: String, _ locale: String, reset: Bool) -> XCUIApplication {
        let app = XCUIApplication()
        // `-resetLanguage` clears an in-app choice left by an earlier run, so the system language applies.
        app.launchArguments = ["-AppleLanguages","(\(language))","-AppleLocale",locale] + (reset ? ["-resetLanguage","YES"] : [])
        app.launch()
        return app
    }

    // Works with or without a gym pack: without one the first entry explains itself; with one it opens the page.
    private func walk(language: String, locale: String, gym: String, logbook: String, settings: String, done: String) {
        let app = launch(language, locale, reset:true)
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
        // A single settings entry stays in the bottom bar.
        XCTAssertTrue(app.buttons["settings"].waitForExistence(timeout:5))
        app.buttons["settings"].tap()
        XCTAssertTrue(app.staticTexts[settings].waitForExistence(timeout:5))
        XCTAssertTrue(app.buttons["language-system"].isSelected)
        app.buttons["settingsDone"].tap()
        app.buttons["tabGym"].tap()
        XCTAssertTrue(app.buttons["tabGym"].isSelected)
    }

    func testChangingLanguageInTheAppAppliesAtOnceAndSurvivesRelaunch() {
        var app = launch("zh-Hans", "zh_CN", reset:true)
        XCTAssertTrue(app.buttons["tabGym"].waitForExistence(timeout:10))
        XCTAssertEqual(app.buttons["tabGym"].label,"岩馆")
        XCTAssertEqual(app.buttons["tabJournal"].label,"攀爬册")
        // Settings from the bar, reachable from the gym as well.
        app.buttons["settings"].tap()
        XCTAssertTrue(app.buttons["language-en"].waitForExistence(timeout:5))
        app.buttons["language-en"].tap()
        XCTAssertTrue(app.staticTexts["Settings and About"].waitForExistence(timeout:5))
        XCTAssertTrue(app.buttons["language-en"].isSelected)
        app.buttons["settingsDone"].tap()
        XCTAssertEqual(app.buttons["tabGym"].label,"Gym")
        XCTAssertEqual(app.buttons["tabJournal"].label,"Logbook")
        app.buttons["tabJournal"].tap()
        XCTAssertTrue(app.staticTexts["Logbook"].waitForExistence(timeout:5))
        // Japanese, using the same settings entry from Logbook.
        app.buttons["settings"].tap()
        app.buttons["language-ja"].tap()
        XCTAssertTrue(app.staticTexts["設定とこのアプリについて"].waitForExistence(timeout:5))
        app.buttons["settingsDone"].tap()
        XCTAssertEqual(app.buttons["tabGym"].label,"ジム")
        XCTAssertEqual(app.buttons["tabJournal"].label,"ログブック")
        XCTAssertTrue(app.staticTexts["ログブック"].exists)
        // The choice outlives the process, whatever the system language is.
        app.terminate()
        app = launch("zh-Hans", "zh_CN", reset:false)
        XCTAssertTrue(app.buttons["tabGym"].waitForExistence(timeout:10))
        XCTAssertEqual(app.buttons["tabGym"].label,"ジム")
        XCTAssertEqual(app.buttons["tabJournal"].label,"ログブック")
        // Back to following the system.
        app.buttons["settings"].tap()
        XCTAssertTrue(app.buttons["language-ja"].isSelected)
        app.buttons["language-system"].tap()
        app.buttons["settingsDone"].tap()
        XCTAssertEqual(app.buttons["tabGym"].label,"岩馆")
    }
}
