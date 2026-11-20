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
        let gym = GymSession(status: .ready(pack))
        let observer = membership.$isMember.sink { gym.updateMembership($0) }
        defer { observer.cancel(); gym.webView?.configuration.userContentController.removeScriptMessageHandler(forName: GymSession.handlerName) }
        gym.start()
        func venueIs(_ id: String) async -> Bool {
            guard let web = gym.webView,
                  let state = try? await web.evaluateJavaScript("window.BetaBuddyGym?.state()") as? [String: Any] else { return false }
            return state["venue"] as? String == id && state["ready"] as? Bool == true
        }
        try await waitUntil { await venueIs("banana-kerry") }
        let web = try XCTUnwrap(gym.webView)
        let lockedCount = try await web.evaluateJavaScript("document.querySelectorAll('#venuelist [data-membership]').length") as? Int
        XCTAssertEqual(lockedCount, pack.manifest.venues.count - 1)
        _ = try await web.evaluateJavaScript("document.querySelector('#venuelist [data-venue=\"dome-xuhui\"]').click()")
        try await waitUntil { gym.showingMembership }
        let stillFree = await venueIs("banana-kerry")
        XCTAssertTrue(stillFree)
        gym.cancelMembershipRequest()
        gym.showingMembership = false
        // A direct venue URL must also pass the native membership check.
        let premium = pack.manifest.venues.first { $0.id == "dome-xuhui" }!
        XCTAssertFalse(gym.authorize(pack.entryURL(hintSeen: true, extraQuery: "venue=\(premium.id)")))
        await membership.purchase()
        XCTAssertTrue(membership.isMember)
        try await waitUntil { await venueIs("dome-xuhui") }
        XCTAssertFalse(gym.showingMembership)
        gym.languageChanged()
        try await waitUntil { await venueIs("dome-xuhui") }
        let transaction = try XCTUnwrap(storeKit.allTransactions().first)
        try storeKit.refundTransaction(identifier: transaction.identifier)
        try await waitUntil { !membership.isMember }
        try await waitUntil { await venueIs("banana-kerry") }
        XCTAssertFalse(gym.authorize(pack.entryURL(hintSeen: true, extraQuery: "venue=dome-xuhui")))
    }
}
