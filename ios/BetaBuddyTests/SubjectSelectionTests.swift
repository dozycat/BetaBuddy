import AVFoundation
import XCTest
@testable import BetaBuddy

@MainActor
final class SubjectSelectionTests: XCTestCase {
    func testOptionalRealOcclusionCorrectionsSurviveSavingAndReanalysis() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["BB_SUBJECT_CLIP"], let fixture = env["BB_SUBJECT_CORRECTIONS"],
              let output = env["BB_SUBJECT_OUTPUT"], SubjectResolver.available else { throw XCTSkip("Optional observed correction fixture") }
        let seeds = try JSONDecoder().decode([SubjectAnchor].self, from: Data(contentsOf: URL(fileURLWithPath: fixture)))
        XCTAssertGreaterThanOrEqual(seeds.count, 2)
        let url = URL(fileURLWithPath: path)
        var selected = try await VideoAnalyzer().videoSummary(url: url)
        var anchors: [SubjectAnchor] = []
        for (index, seed) in seeds.enumerated() {
            let refined = try await SubjectRegionAnalyzer().refine(url: url, analysis: selected, bounds: seed.bounds, at: seed.time) { _ in }
            anchors.append(refined.anchor)
            selected = try await SubjectResolver().resolve(refined.analysis, anchors: anchors)
            for anchor in anchors {
                let seed = try XCTUnwrap(selected.people?.first { abs($0.time-anchor.time) < 0.0001 })
                XCTAssertTrue(seed.candidates.contains { $0.id == anchor.candidateID }, "A correction must preserve earlier tracked seeds")
            }
            try JSONEncoder().encode(selected).write(to: URL(fileURLWithPath: output + ".step\(index).json"))
            XCTAssertEqual(selected.subject?.unresolved, 0)
        }
        func checkSeeds(_ analysis: PoseAnalysis) throws {
            for seed in seeds {
                let position = try XCTUnwrap(analysis.subjectPositions?.min { abs($0.time-seed.time) < abs($1.time-seed.time) })
                XCTAssertEqual(position.time, seed.time, accuracy: 0.15)
                XCTAssertGreaterThan(position.bounds.overlap(seed.bounds), 0.4, "Correction must select the observed target")
            }
        }
        try checkSeeds(selected)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ClimbStore(root: root)
        let staged = root.appendingPathComponent("reference.mp4")
        try FileManager.default.copyItem(at: url, to: staged)
        let record = try store.save(title: "", grade: ungradedValue, sent: false, note: "", stagedVideo: staged,
            thumbnail: Data(), analysis: selected)
        try store.selectSubject(record, analysis: selected, anchors: anchors)
        let reopened = ClimbStore(root: root)
        let saved = try XCTUnwrap(reopened.climbs.first)
        XCTAssertEqual(saved.subjectAnchors, anchors)
        try checkSeeds(try await reopened.subjectAnalysis(for: saved))
        try JSONEncoder().encode(selected).write(to: URL(fileURLWithPath: output))
        try JSONEncoder().encode(anchors).write(to: URL(fileURLWithPath: output + ".anchors.json"))
        let fresh = try await VideoAnalyzer().analyze(url: reopened.file(saved.videoName), quality: .balanced, subjectAnchors: saved.subjectAnchors!) { _ in }
        XCTAssertEqual(fresh.subject?.unresolved, 0)
        try checkSeeds(fresh)
        try JSONEncoder().encode(fresh).write(to: URL(fileURLWithPath: output + ".reanalyzed.json"))
        print("Real corrections: \(anchors.count) anchors, \(selected.subjectPositions?.count ?? 0) positions, \(selected.frames.count) poses; reanalysis \(fresh.subjectPositions?.count ?? 0) positions")
    }
    func testOptionalTiledCandidateProbe() async throws {
        let env=ProcessInfo.processInfo.environment
        guard let path=env["BB_SUBJECT_CLIP"], let output=env["BB_SUBJECT_PROBE"] else { throw XCTSkip("Local candidate probe not supplied") }
        let url=URL(fileURLWithPath:path)
        let summary=try await VideoAnalyzer().videoSummary(url:url)
        let samples=try await SubjectRegionAnalyzer().discover(url:url,analysis:summary) { _ in }
        try JSONEncoder().encode(samples).write(to:URL(fileURLWithPath:output))
        print("Tiled candidate probe: \(samples.reduce(0) { $0+$1.candidates.count }) observations; manual inspection required")
    }
    func testLegacyAnchorWithoutRegionStillLoads() throws {
        let raw = #"{"time":0,"bounds":{"x":0.1,"y":0.1,"width":0.2,"height":0.6},"appearance":[]}"#
        XCTAssertNil(try JSONDecoder().decode(SubjectAnchor.self, from: Data(raw.utf8)).region)
    }
    func testCachedUnresolvedSelectionCanRecoverWithoutManualAnchor() async throws {
        let env=ProcessInfo.processInfo.environment
        guard let path=env["BB_SUBJECT_CLIP"],let baseline=env["BB_SUBJECT_BASE"] else { throw XCTSkip("Optional unresolved video observations") }
        let original=try JSONDecoder().decode(PoseAnalysis.self,from:Data(contentsOf:URL(fileURLWithPath:baseline)))
        XCTAssertTrue(original.hasSubjectCandidates);XCTAssertFalse(original.hasSelectedSubject)
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:root) }
        let store=ClimbStore(root:root)
        let staged=root.appendingPathComponent("fixture.mp4")
        try FileManager.default.copyItem(at:URL(fileURLWithPath:path),to:staged)
        let climb=try store.save(title:"",grade:ungradedValue,sent:false,note:"",stagedVideo:staged,thumbnail:Data(),analysis:original)
        let session=SubjectSelectionSession(climb:climb,store:store)
        defer { session.stop() }
        session.prepare()
        let deadline=Date().addingTimeInterval(180)
        while session.busy && Date()<deadline { try await Task.sleep(for:.milliseconds(100)) }
        XCTAssertFalse(session.busy);XCTAssertNil(session.error)
        let result=try XCTUnwrap(session.analysis)
        XCTAssertEqual(result.subject?.manual,false);XCTAssertTrue(session.anchors.isEmpty)
        XCTAssertGreaterThanOrEqual(result.subjectPositions?.count ?? 0,Int(ceil(Double(original.sampledFrames)*0.9)))
    }
    private func analysis() -> PoseAnalysis {
        let pose = PoseFrame(time: 0, joints: Array(repeating: Joint(x: 0.3, y: 0.4, confidence: 1), count: 33))
        let a = SubjectCandidate(id: "a", bounds: .init(x: 0.2, y: 0.1, width: 0.2, height: 0.5), appearance: [0.1], pose: pose)
        var b = a; b.id = "b"; b.bounds.x = 0.6
        var p = pose; p.time = 1
        return PoseAnalysis(frames: [], sampledFrames: 2, duration: 2, aspectRatio: 0.5,
            people: [.init(time: 0, candidates: [a,b]), .init(time: 1, candidates: [])])
    }
    func testReopeningSavedPartialAutomaticSelectionDoesNotAnalyzeAgain() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ClimbStore(root: root)
        // Deliberately not a playable video: reopening must only read saved observations.
        let staged = root.appendingPathComponent("saved.mp4"); try Data([0]).write(to: staged)
        let resolution = SubjectResolution(schema: 1, picks: [.init(time: 0, personID: "b"), .init(time: 1, personID: nil)], manual: false, unresolved: 0, recoveryRecommended: true)
        let selected = try analysis().selecting(resolution)
        let climb = try store.save(title: "", grade: ungradedValue, sent: false, note: "", stagedVideo: staged, thumbnail: Data(), analysis: selected)
        try store.selectSubject(climb, analysis: selected, anchors: [])
        let saved = try XCTUnwrap(store.climbs.first)
        let session = SubjectSelectionSession(climb: saved, store: store)
        defer { session.stop() }
        session.prepare()
        let deadline = Date().addingTimeInterval(3)
        while session.busy && Date() < deadline { try await Task.sleep(for: .milliseconds(20)) }
        XCTAssertFalse(session.busy); XCTAssertNil(session.error)
        XCTAssertEqual(session.analysis?.subject?.picks, resolution.picks)
        XCTAssertEqual(session.analysis?.subject?.recoveryRecommended, true)
        XCTAssertEqual(store.climbs.first?.subjectRevision, saved.subjectRevision)
        XCTAssertFalse(try XCTUnwrap(session.timeline).gaps.isEmpty)
    }
    func testRecoveryRequestPreservesSelectionAndLegacyRecords() throws {
        let legacy = #"{"schema":1,"picks":[{"time":0,"personID":"a"},{"time":1}],"manual":false,"unresolved":0}"#
        let resolution = try JSONDecoder().decode(SubjectResolution.self, from: Data(legacy.utf8))
        let selected = try analysis().selecting(resolution)
        XCTAssertTrue(selected.hasSelectedSubject)
        XCTAssertFalse(selected.needsSubjectRecovery)
        var requested = resolution; requested.recoveryRecommended = true
        let restored = try JSONDecoder().decode(SubjectResolution.self, from: JSONEncoder().encode(requested))
        let recoverable = try analysis().selecting(restored)
        XCTAssertTrue(recoverable.hasSelectedSubject)
        XCTAssertTrue(recoverable.needsSubjectRecovery)
        XCTAssertEqual(recoverable.subject?.picks, selected.subject?.picks)
        XCTAssertTrue(analysis().needsSubjectRecovery)
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
    func testSelectedRegionKeepsPositionWithoutInventingPose() throws {
        var original = analysis()
        original.people![0].candidates[1].pose.joints = []
        let selected = try original.selecting(.init(schema: 1, picks: [.init(time: 0, personID: "b"), .init(time: 1, personID: nil)], manual: true, unresolved: 0))
        XCTAssertTrue(selected.frames.isEmpty)
        XCTAssertTrue(selected.hasSelectedSubject)
        XCTAssertEqual(selected.subjectPositions?.count, 1)
        XCTAssertEqual(selected.subjectPositions?.first?.bounds.x, 0.6)
        original.people![0].candidates[1].pose.time = 1
        XCTAssertThrowsError(try original.selecting(selected.subject!))
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
        if let path=ProcessInfo.processInfo.environment["BB_SUBJECT_EXPECTED"] {
            let expected=try JSONDecoder().decode([SubjectPosition].self,from:Data(contentsOf:URL(fileURLWithPath:path)))
            XCTAssertFalse(expected.isEmpty)
            let distances=expected.compactMap { reference -> Double? in
                guard let selected=result.subjectPositions?.min(by:{abs($0.time-reference.time)<abs($1.time-reference.time)}),abs(selected.time-reference.time)<=0.16 else { return nil }
                return hypot(selected.bounds.x+selected.bounds.width/2-reference.bounds.x-reference.bounds.width/2,selected.bounds.y+selected.bounds.height/2-reference.bounds.y-reference.bounds.height/2)
            }
            XCTAssertGreaterThanOrEqual(distances.count,Int(ceil(Double(expected.count)*0.9)),"Target coverage remains incomplete")
            XCTAssertLessThan(distances.reduce(0,+)/Double(max(1,distances.count)),0.05,"Automatic selection is on a different person")
            XCTAssertTrue(distances.allSatisfy { $0<0.15 })
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
        print("Manual region: positions=\(selected.subjectPositions?.count ?? 0), poses=\(selected.frames.count)/\(selected.sampledFrames), seconds=\(Date().timeIntervalSince(start))")
        if let output = env["BB_SUBJECT_OUTPUT"] { try JSONEncoder().encode(selected).write(to: URL(fileURLWithPath: output)) }
        XCTAssertEqual(result.anchor.region, true)
        if env["BB_SUBJECT_REANALYZE"] == "1" {
            let anchor = try JSONDecoder().decode(SubjectAnchor.self, from: JSONEncoder().encode(result.anchor))
            let fresh = try await VideoAnalyzer().analyze(url: url, quality: .balanced, subjectAnchors: [anchor]) { _ in }
            XCTAssertEqual(fresh.subject?.unresolved, 0)
            let positions = try XCTUnwrap(fresh.subjectPositions)
            XCTAssertGreaterThanOrEqual(positions.count, Int(Double(selected.people!.count) * 0.9))
            let distance = positions.map { position in
                let old = selected.subjectPositions!.min { abs($0.time-position.time) < abs($1.time-position.time) }!.bounds
                return hypot(position.bounds.x-old.x, position.bounds.y-old.y)
            }.reduce(0,+) / Double(max(1, positions.count))
            XCTAssertLessThan(distance, 0.05, "Fresh analysis must keep the saved region")
        }
    }
}
