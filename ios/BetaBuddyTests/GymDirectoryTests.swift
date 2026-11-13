import XCTest
@testable import BetaBuddy

final class GymDirectoryTests: XCTestCase {
    func testOfflineDirectoryIsBundledAndDecodesWithStableUniqueIDs() throws {
        let directory = try XCTUnwrap(GymDirectory.bundled)
        XCTAssertFalse(directory.gyms.isEmpty)
        XCTAssertFalse(directory.updatedAt.isEmpty)
        XCTAssertEqual(Set(directory.gyms.map(\.id)).count, directory.gyms.count)
        for gym in directory.gyms {
            XCTAssertFalse(gym.id.isEmpty)
            XCTAssertFalse(gym.name.isEmpty)
            XCTAssertNotNil(URL(string: gym.sourceURL)?.host)
        }
    }
}
