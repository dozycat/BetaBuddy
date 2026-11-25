import XCTest
import StoreKitTest
import Combine
@testable import BetaBuddy

@MainActor
final class MembershipTests: XCTestCase {
    private func waitUntil(_ predicate: () async -> Bool) async throws {
        let deadline = Date().addingTimeInterval(20)
        while Date() < deadline {
            if await predicate() { return }
            try await Task.sleep(for: .milliseconds(100))
        }
        XCTFail("Timed out waiting for StoreKit or the gym page")
    }

    func testVerifiedPurchaseRelaunchRestoreAndRefund() async throws {
        let file = try XCTUnwrap(Bundle(for: Self.self).url(forResource: "Lifetime", withExtension: "storekit"))
        let storeKit = try SKTestSession(contentsOf: file)
        storeKit.disableDialogs = true
        storeKit.clearTransactions()
        defer { storeKit.clearTransactions() }
        let membership = MembershipStore()
        await membership.load()
        XCTAssertFalse(membership.isMember)
        XCTAssertNotNil(membership.product)
        await membership.purchase()
        XCTAssertTrue(membership.isMember)
        let reopened = MembershipStore()
        await reopened.refreshEntitlement()
        XCTAssertTrue(reopened.isMember)
        await reopened.restore()
        XCTAssertTrue(reopened.isMember)
        let transaction = try XCTUnwrap(storeKit.allTransactions().first)
        try storeKit.refundTransaction(identifier: transaction.identifier)
        // Revocations arrive through Transaction.updates, asynchronously.
        try await waitUntil { !reopened.isMember }
        XCTAssertFalse(reopened.isMember)
    }

    func testCatPurchaseRestoreLanguageAndRefundWithBundledPack() async throws {
        guard let pack = GymPack.current, pack.manifest.catCode != nil else { throw XCTSkip("Install the cat pack") }
        let storeKit = try SKTestSession(contentsOf: XCTUnwrap(Bundle(for: Self.self).url(forResource: "Lifetime", withExtension: "storekit")))
        storeKit.disableDialogs = true
        storeKit.clearTransactions()
        defer { storeKit.clearTransactions() }
        let membership = MembershipStore()
        await membership.load()
        let gym = GymSession(status: .ready(pack))
        let observer = membership.$isMember.sink { gym.updateMembership($0) }
        defer { observer.cancel(); gym.webView?.configuration.userContentController.removeScriptMessageHandler(forName: GymSession.handlerName) }
        gym.start()
        func state() async -> [String: Any] {
            (try? await gym.webView?.evaluateJavaScript("window.BetaBuddyGym?.state()") as? [String: Any]) ?? [:]
        }
        try await waitUntil { await state()["ready"] as? Bool == true }
        let web = try XCTUnwrap(gym.webView)
        _ = try await web.evaluateJavaScript("document.querySelector('#species [data-sp=cat]').click()")
        try await waitUntil { gym.showingMembership }
        var value = await state()
        XCTAssertEqual(value["species"] as? String, "human")
        let unloaded = try await web.evaluateJavaScript("!document.querySelector('script[src*=catplans]')") as? Bool
        XCTAssertEqual(unloaded, true)
        gym.cancelMembershipRequest(); gym.showingMembership = false
        _ = try await web.evaluateJavaScript("BetaBuddyGym.debug({route:window.GYM_VENUES[0].routeIds[0],species:'cat',catp:1,catstrict:'1'})")
        try await waitUntil { gym.showingMembership }
        value = await state()
        XCTAssertEqual(value["species"] as? String, "human")
        gym.cancelMembershipRequest(); gym.showingMembership = false
        _ = try await web.evaluateJavaScript("BetaBuddyGym.openRoute(window.GYM_VENUES[0].routeIds[0]); BetaBuddyGym.setSpecies('cat',{p:1,strict:true})")
        try await waitUntil { gym.pendingCat?.preset == 1 }
        XCTAssertEqual(gym.pendingCat?.strict, true)
        await membership.purchase()
        try await waitUntil { (await state()["cat"] as? [String: Any])?["loaded"] as? Bool == true }
        value = await state()
        XCTAssertEqual(value["species"] as? String, "cat")
        XCTAssertEqual((value["cat"] as? [String: Any])?["p"] as? Int, 1)
        XCTAssertEqual((value["cat"] as? [String: Any])?["strict"] as? Bool, true)
        XCTAssertFalse(gym.showingMembership)
        gym.languageChanged()
        try await waitUntil {
            let current = await state()
            return gym.webView?.url?.query?.contains("species=cat") == true && (current["cat"] as? [String: Any])?["loaded"] as? Bool == true
        }
        value = await state()
        XCTAssertEqual((value["cat"] as? [String: Any])?["p"] as? Int, 1)
        XCTAssertEqual((value["cat"] as? [String: Any])?["strict"] as? Bool, true)
        _ = try await web.evaluateJavaScript("BetaBuddyGym.setSpecies('human'); BetaBuddyGym.setSpecies('cat',{p:0,strict:false})")
        try await waitUntil { (await state()["cat"] as? [String: Any])?["p"] as? Int == 0 }
        let restored = MembershipStore()
        await restored.restore()
        XCTAssertTrue(restored.isMember)
        let transaction = try XCTUnwrap(storeKit.allTransactions().first)
        try storeKit.refundTransaction(identifier: transaction.identifier)
        try await waitUntil { !membership.isMember }
        try await waitUntil { await state()["species"] as? String == "human" }
        _ = try await web.evaluateJavaScript("BetaBuddyGym.setSpecies('cat')")
        try await waitUntil { gym.showingMembership }
        value = await state()
        XCTAssertEqual(value["species"] as? String, "human")
    }

    func testBundledVenueSwitchPurchaseAndRevocation() async throws {
        guard let pack = GymPack.current, pack.manifest.venues.count > 1 else {
            throw XCTSkip("Install a multi-venue pack for this integration test")
        }
        let file = try XCTUnwrap(Bundle(for: Self.self).url(forResource: "Lifetime", withExtension: "storekit"))
        let storeKit = try SKTestSession(contentsOf: file)
        storeKit.disableDialogs = true
        storeKit.clearTransactions()
        defer { storeKit.clearTransactions() }
        let membership = MembershipStore()
        await membership.load()
        let free = try XCTUnwrap(pack.manifest.freeVenueID)
        let premium = try XCTUnwrap(pack.manifest.venues.first { $0.id != free }).id
        let gym = GymSession(status: .ready(pack))
        let observer = membership.$isMember.sink { gym.updateMembership($0) }
        defer { observer.cancel(); gym.webView?.configuration.userContentController.removeScriptMessageHandler(forName: GymSession.handlerName) }
        gym.start()
        func venueIs(_ id: String) async -> Bool {
            guard let web = gym.webView,
                  let state = try? await web.evaluateJavaScript("window.BetaBuddyGym?.state()") as? [String: Any] else { return false }
            return state["venue"] as? String == id && state["ready"] as? Bool == true
        }
        try await waitUntil { await venueIs(free) }
        let web = try XCTUnwrap(gym.webView)
        let lockedCount = try await web.evaluateJavaScript("document.querySelectorAll('#venuelist [data-membership]').length") as? Int
        XCTAssertEqual(lockedCount, pack.manifest.venues.count - 1)
        _ = try await web.evaluateJavaScript("document.querySelector('#venuelist [data-venue=\"\(premium)\"]').click()")
        try await waitUntil { gym.showingMembership }
        let stillFree = await venueIs(free)
        XCTAssertTrue(stillFree)
        gym.cancelMembershipRequest()
        gym.showingMembership = false
        // A direct venue URL must also pass the native membership check.
        XCTAssertFalse(gym.authorize(pack.entryURL(hintSeen: true, extraQuery: "venue=\(premium)")))
        await membership.purchase()
        XCTAssertTrue(membership.isMember)
        try await waitUntil { await venueIs(premium) }
        XCTAssertFalse(gym.showingMembership)
        gym.languageChanged()
        try await waitUntil { await venueIs(premium) }
        let transaction = try XCTUnwrap(storeKit.allTransactions().first)
        try storeKit.refundTransaction(identifier: transaction.identifier)
        try await waitUntil { !membership.isMember }
        try await waitUntil { await venueIs(free) }
        XCTAssertFalse(gym.authorize(pack.entryURL(hintSeen: true, extraQuery: "venue=\(premium)")))
    }
}
