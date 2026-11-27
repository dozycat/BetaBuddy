import AVFoundation
import XCTest
@testable import BetaBuddy

@MainActor
final class SubjectSelectionTests: XCTestCase {
    private func analysis() -> PoseAnalysis {
        let pose = PoseFrame(time: 0, joints: Array(repeating: Joint(x: 0.3, y: 0.4, confidence: 1), count: 33))
        let a = SubjectCandidate(id: "a", bounds: .init(x: 0.2, y: 0.1, width: 0.2, height: 0.5), appearance: [0.1], pose: pose)
        var b = a; b.id = "b"; b.bounds.x = 0.6
        var p = pose; p.time = 1
        return PoseAnalysis(frames: [], sampledFrames: 2, duration: 2, aspectRatio: 0.5,
            people: [.init(time: 0, candidates: [a,b]), .init(time: 1, candidates: [])])
    }
    func testMissingSelectionNeverUsesAnotherPersonAndUnknownIDsAreRejected() throws {
        let original = analysis()
        let result = SubjectResolution(schema: 1, picks: [.init(time: 0, personID: "b"), .init(time: 1, personID: nil)], manual: true, unresolved: 0)
        let selected = try original.selecting(result)
        XCTAssertEqual(selected.frames.count, 1)
        XCTAssertEqual(selected.subject?.picks[0].personID, "b")
        XCTAssertNil(selected.subjectSample(at: 0.5))
        XCTAssertNotNil(selected.subjectSample(at: 0.1))
        var wrong = result; wrong.picks[0].personID = "missing"
        XCTAssertThrowsError(try original.selecting(wrong))
    }
    func testChangingSelectionInvalidatesCachesAndRejectsStaleResultsAcrossRelaunch() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ClimbStore(root: root)
        let staged = root.appendingPathComponent("fixture.mov"); try Data([1,2]).write(to: staged)
        let original = analysis()
        let climb = try store.save(title: "", grade: ungradedValue, sent: false, note: "keep", stagedVideo: staged, thumbnail: Data(), analysis: original)
        try store.saveAnalysisResult("old result", for: climb.id)
        let selected = try original.selecting(.init(schema: 1, picks: [.init(time: 0, personID: "a"), .init(time: 1, personID: nil)], manual: true, unresolved: 0))
        let anchor = SubjectAnchor(time: 0, candidate: original.people![0].candidates[0])
        try store.selectSubject(climb, analysis: selected, anchors: [anchor])
        XCTAssertEqual(store.climbs[0].subjectRevision, 1)
        XCTAssertFalse(store.analysisResults.contains(climb.id))
        XCTAssertNil(store.analysisResult(for: climb))
        // Both stale web messages and an old pose worker must be harmless.
        try store.saveAnalysisResult("stale", for: climb.id, expectedSubjectRevision: 0)
        try store.saveEditPlan(#"{"v":1,"duration":2,"trim":[0,2],"segments":[{"t0":0,"t1":2,"speed":1}]}"#, for: climb.id, expectedSubjectRevision: 0)
        XCTAssertFalse(store.editPlans.contains(climb.id))
        try store.replaceAnalysis(climb, analysis: original)
        XCTAssertNil(store.analysisResult(for: climb))
        XCTAssertEqual(store.climbs[0].analysis.frames.count, 1)
        // Simulate interruption before old cache cleanup: it must not become the current result.
        try Data("orphan".utf8).write(to: root.appendingPathComponent("\(climb.id).analysis.json"))
        let reopened = ClimbStore(root: root)
        XCTAssertFalse(reopened.analysisResults.contains(climb.id))
        XCTAssertEqual(reopened.climbs[0].subjectAnchors, [anchor])
        try reopened.saveAnalysisResult("current", for: climb.id, expectedSubjectRevision: 1)
        XCTAssertEqual(ClimbStore(root: root).analysisResult(for: reopened.climbs[0]), "current")
        let currentFile = root.appendingPathComponent("\(climb.id).s1.analysis.json")
        XCTAssertTrue(FileManager.default.fileExists(atPath: currentFile.path))
        try reopened.delete(reopened.climbs[0])
        XCTAssertFalse(FileManager.default.fileExists(atPath: currentFile.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("\(climb.id).analysis.json").path))
    }
    func testInstalledPolicyAndRealPoseDetection() async throws {
        guard let path = ProcessInfo.processInfo.environment["BB_SUBJECT_CLIP"], SubjectResolver.available else { throw XCTSkip("Private policy and positive clip are optional") }
        let start = Date()
        let result = try await VideoAnalyzer().analyze(url: URL(fileURLWithPath: path), quality: .balanced) { _ in }
        XCTAssertTrue(result.hasSubjectCandidates)
        XCTAssertNotNil(result.subject)
        XCTAssertFalse(result.frames.isEmpty, "Inspect automatic selection on the positive reference clip")
        let sample = try XCTUnwrap(result.people?.first { !$0.candidates.isEmpty })
        let anchor = SubjectAnchor(time: sample.time, candidate: sample.candidates[0])
        let manual = try await SubjectResolver().resolve(result, anchors: [anchor])
        XCTAssertEqual(manual.subject?.manual, true)
        XCTAssertEqual(manual.subject?.picks.first { $0.time == sample.time }?.personID, sample.candidates[0].id)
        print("Subject integration: samples=\(result.people?.count ?? 0), selected=\(result.frames.count), seconds=\(Date().timeIntervalSince(start))")
        if let output = ProcessInfo.processInfo.environment["BB_SUBJECT_OUTPUT"] {
            try JSONEncoder().encode(result).write(to: URL(fileURLWithPath: output))
        }
    }
    func testCandidateArchiveLoadsOnDemandAndFailedIndexWriteKeepsPreviousSelection() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ClimbStore(root: root)
        let staged = root.appendingPathComponent("fixture.mov"); try Data([1]).write(to: staged)
        var full = analysis()
        let candidate = full.people![0].candidates[0]
        full.people = (0..<600).map { index in
            var c = candidate; c.id = "sample-\(index)"; c.pose.time = Double(index) / 10
            return SubjectSample(time: c.pose.time, candidates: [c])
        }
        let saved = try store.save(title: "", grade: ungradedValue, sent: false, note: "", stagedVideo: staged, thumbnail: Data(), analysis: full)
        XCTAssertNil(saved.analysis.people)
        XCTAssertTrue(saved.analysis.hasSubjectCandidates)
        let reference = try XCTUnwrap(saved.analysis.peopleReference)
        let index = store.file("archive.json")
        XCTAssertLessThan(try Data(contentsOf: index).count, 3000, "Journal metadata must not contain every candidate")
        let reopened = ClimbStore(root: root)
        XCTAssertNil(reopened.climbs[0].analysis.people)
        let restored = try await reopened.subjectAnalysis(for: reopened.climbs[0])
        XCTAssertEqual(restored.people?.count, 600)
        XCTAssertEqual(restored.people?.last?.candidates[0].id, "sample-599")
        // Make only the index unwritable after a new immutable sidecar can be written.
        let backup = store.file("saved-index.json")
        try FileManager.default.moveItem(at: index, to: backup)
        try FileManager.default.createDirectory(at: index, withIntermediateDirectories: false)
        XCTAssertThrowsError(try reopened.replaceAnalysis(saved, analysis: full))
        XCTAssertEqual(reopened.climbs[0].analysis.peopleReference, reference)
        let sidecars = try FileManager.default.contentsOfDirectory(atPath: root.path).filter { $0.contains(".subjects.") }
        XCTAssertEqual(sidecars.count, 1, "A failed index write must remove only its new uncommitted file")
        try FileManager.default.removeItem(at: index)
        try FileManager.default.moveItem(at: backup, to: index)
        try reopened.replaceAnalysis(saved, analysis: full)
        XCTAssertNotEqual(reopened.climbs[0].analysis.peopleReference, reference)
        XCTAssertFalse(FileManager.default.fileExists(atPath: SubjectCandidateStorage.file(root: root, record: saved.id, reference: reference).path))
        try reopened.delete(reopened.climbs[0])
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: root.path).filter { $0.contains(".subjects.") }.isEmpty)
    }
    func testManualRegionOnRealReference() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["BB_SUBJECT_CLIP"], let region = env["BB_SUBJECT_REGION"], SubjectResolver.available else { throw XCTSkip("Optional private reference and region not supplied") }
        let bounds = try JSONDecoder().decode(SubjectBounds.self, from: Data(region.utf8))
        let url = URL(fileURLWithPath: path)
        let base = try await VideoAnalyzer().analyze(url: url, quality: .balanced) { _ in }
        let start = Date()
        let result = try await SubjectRegionAnalyzer().refine(url: url, analysis: base, bounds: bounds, at: Double(env["BB_SUBJECT_TIME"] ?? "0") ?? 0) { _ in }
        let selected = try await SubjectResolver().resolve(result.analysis, anchors: [result.anchor])
        XCTAssertGreaterThan(selected.frames.count, 2)
        XCTAssertEqual(selected.subject?.unresolved, 0)
        if result.anchor.time > 0.5 && result.anchor.time < base.duration-0.5 {
            XCTAssertTrue(selected.frames.contains { $0.time < result.anchor.time }, "Selection must also cover detected frames before the chosen moment")
            XCTAssertTrue(selected.frames.contains { $0.time > result.anchor.time }, "Selection must also cover detected frames after the chosen moment")
        }
        print("Manual region: selected=\(selected.frames.count)/\(selected.sampledFrames), seconds=\(Date().timeIntervalSince(start))")
        if let output = env["BB_SUBJECT_OUTPUT"] { try JSONEncoder().encode(selected).write(to: URL(fileURLWithPath: output)) }
    }
}
