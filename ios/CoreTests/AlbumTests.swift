import XCTest
@testable import BetaBuddyCore

final class AlbumTests: XCTestCase {
    private func climb(date: Date) -> Climb {
        Climb(id: UUID(), date: date, title: "", grade: "V3", sent: false, note: "keep", videoName: "v.mov", thumbnailName: "t.jpg", analysis: PoseAnalysis(frames: [], sampledFrames: 0, duration: 1, aspectRatio: 1, processingSeconds: 0, moments: []))
    }
    func testLegacyArchiveDecodesWithoutOrganizationFields() throws {
        let original = climb(date: Date(timeIntervalSince1970: 1000))
        let bytes = try JSONEncoder().encode(Archive(climbs: [original]))
        var root = try XCTUnwrap(JSONSerialization.jsonObject(with: bytes) as? [String: Any])
        root.removeValue(forKey: "folders")
        let legacy = try JSONDecoder().decode(Archive.self, from: JSONSerialization.data(withJSONObject: root))
        XCTAssertTrue(legacy.folders.isEmpty)
        XCTAssertNil(legacy.climbs[0].climbedAt)
        XCTAssertEqual(legacy.climbs[0].climbingDate, original.date)
        XCTAssertEqual(legacy.climbs[0].note, "keep")
        XCTAssertNil(legacy.climbs[0].folderId)
        XCTAssertEqual(legacy.climbs[0].gymKey, "")
    }
    func testFiltersIntersectAndUseClimbedDateInsteadOfImportDate() {
        let folder = UUID(), day = Date(timeIntervalSince1970: 1700000000)
        var record = climb(date: day.addingTimeInterval(86400 * 100))
        record.folderId = folder; record.gymId = "fixture-gym"; record.gymName = "fixture"; record.climbedAt = day
        var filter = JournalFilter(folderId: folder, gymKey: "id:fixture-gym", grade: "V3", startDate: day, endDate: day)
        XCTAssertTrue(filter.matches(record))
        filter.grade = "V4"; XCTAssertFalse(filter.matches(record))
        filter.grade = nil; filter.unfiled = true; XCTAssertFalse(filter.matches(record))
        record.folderId = nil; filter.folderId = nil; XCTAssertTrue(filter.matches(record))
    }
    func testInclusiveCalendarDayAcrossDST() throws {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = try XCTUnwrap(TimeZone(identifier: "America/Los_Angeles"))
        let day = try XCTUnwrap(calendar.date(from: DateComponents(year: 2026, month: 11, day: 1)))
        let nextDay = try XCTUnwrap(calendar.date(byAdding: .day, value: 1, to: day))
        XCTAssertEqual(nextDay.timeIntervalSince(day), 25 * 3600)
        let filter = JournalFilter(startDate: day, endDate: day)
        XCTAssertTrue(filter.matches(climb(date: nextDay.addingTimeInterval(-1)), calendar: calendar))
        XCTAssertFalse(filter.matches(climb(date: nextDay), calendar: calendar))
        XCTAssertFalse(filter.matches(climb(date: day.addingTimeInterval(-1)), calendar: calendar))
    }
}
