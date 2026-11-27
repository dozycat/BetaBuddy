import XCTest
@testable import BetaBuddyCore

final class AlbumTests: XCTestCase {
    private func climb(date: Date) -> Climb {
        Climb(id: UUID(), date: date, title: "", grade: "V3", sent: false, note: "keep", videoName: "v.mov", thumbnailName: "t.jpg", analysis: PoseAnalysis(frames: [], sampledFrames: 0, duration: 1, aspectRatio: 1))
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

    func testDayGroupsUseCaptureDateAndKeepBranchesAndOutdoorApart() throws {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = try XCTUnwrap(TimeZone(identifier: "Asia/Shanghai"))
        let day = try XCTUnwrap(calendar.date(from: DateComponents(year: 2026, month: 11, day: 28)))
        var first = climb(date: day.addingTimeInterval(86_400 * 10))
        first.climbedAt = day.addingTimeInterval(3_600); first.gymId = "fixture-a"; first.gymName = "Fixture"
        var second = climb(date: day)
        second.climbedAt = day.addingTimeInterval(7_200); second.gymId = "fixture-a"; second.environment = "indoor"
        var otherBranch = second; otherBranch.id = UUID(); otherBranch.gymId = "fixture-b"
        var outdoor = second; outdoor.id = UUID(); outdoor.environment = "outdoor"
        let groups = JournalDayGroup.make([first, second, otherBranch, outdoor], calendar: calendar)
        XCTAssertEqual(groups.count, 3)
        XCTAssertTrue(groups.allSatisfy { $0.id.day == day })
        XCTAssertEqual(groups.first { $0.id.gymKey == "id:fixture-a" }?.climbs.map(\.id), [second.id, first.id])
        XCTAssertEqual(groups.first { $0.id.environment == .outdoor }?.id.gymKey, "")
        second.climbedAt = day.addingTimeInterval(-1)
        XCTAssertEqual(JournalDayGroup.make([first, second], calendar: calendar).count, 2)
    }

    func testDayGroupingUsesCalendarDaysAcrossDSTAndKeepsUnknownSeparate() throws {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = try XCTUnwrap(TimeZone(identifier: "America/Los_Angeles"))
        let day = try XCTUnwrap(calendar.date(from: DateComponents(year: 2026, month: 11, day: 1)))
        let next = try XCTUnwrap(calendar.date(byAdding: .day, value: 1, to: day))
        let early = climb(date: day), late = climb(date: next.addingTimeInterval(-1))
        var outdoor = early; outdoor.id = UUID(); outdoor.environment = "outdoor"
        XCTAssertEqual(JournalDayGroup.make([early, late], calendar: calendar).count, 1)
        XCTAssertEqual(JournalDayGroup.make([early, late, outdoor], calendar: calendar).count, 2)
        XCTAssertEqual(JournalDayGroup.make([late, climb(date: next)], calendar: calendar).count, 2)
    }
}
