import XCTest
@testable import BetaBuddyCore

final class SubjectTimelineTests: XCTestCase {
    func testMissingIntervalsRemainVisibleAndNavigationWraps() throws {
        let picks = (0..<60).map { i in SubjectPick(time: Double(i)/10, personID: (10..<20).contains(i) || (40..<50).contains(i) ? nil : "person") }
        let timeline = SubjectTimeline(picks: picks, duration: 6)
        XCTAssertEqual(timeline.selected.count, 3)
        XCTAssertEqual(timeline.gaps.count, 2)
        XCTAssertEqual(timeline.gaps[0].start, 0.95, accuracy: 0.000001)
        XCTAssertEqual(timeline.gaps[0].end, 1.95, accuracy: 0.000001)
        XCTAssertEqual(try XCTUnwrap(timeline.nextGap(after: 0)), 1.45, accuracy: 0.000001)
        XCTAssertEqual(try XCTUnwrap(timeline.nextGap(after: 1.45)), 4.45, accuracy: 0.000001)
        XCTAssertEqual(try XCTUnwrap(timeline.nextGap(after: 4.45)), 1.45, accuracy: 0.000001)
        XCTAssertNil(picks[10].personID, "Navigation never changes the selection")
    }
    func testSparseObservationsDoNotBecomeContinuousCoverage() {
        let sparse = SubjectTimeline(picks: [.init(time: 0, personID: "a"), .init(time: 5, personID: "b")], duration: 6)
        XCTAssertEqual(sparse.selected.count, 2)
        XCTAssertGreaterThan(sparse.gaps[0].end-sparse.gaps[0].start, 4)
        let complete = SubjectTimeline(picks: (0..<30).map { .init(time: Double($0)/10, personID: "a") }, duration: 3)
        XCTAssertEqual(complete.selected, [.init(start: 0, end: 3)])
        XCTAssertNil(complete.nextGap(after: 0))
        XCTAssertEqual(SubjectTimeline(picks: [], duration: 3).gaps, [.init(start: 0, end: 3)])
        XCTAssertTrue(SubjectTimeline(picks: [], duration: .nan).gaps.isEmpty)
    }
}
