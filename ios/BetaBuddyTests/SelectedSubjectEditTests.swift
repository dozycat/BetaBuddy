import AVFoundation
import XCTest
@testable import BetaBuddy

@MainActor
final class SelectedSubjectEditTests: XCTestCase {
    func testSelectedPositionsReachThePackAndRenderedCamera() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let source = env["BB_SUBJECT_CLIP"], let input = env["BB_SUBJECT_ANALYSIS"],
              let pack = GymPack.current else { throw XCTSkip("Optional private clip and selected observations are not supplied") }
        let selected = try JSONDecoder().decode(PoseAnalysis.self, from: Data(contentsOf: URL(fileURLWithPath:input)))
        let positions = try XCTUnwrap(selected.subjectPositions)
        XCTAssertGreaterThan(positions.count,selected.frames.count,"This reference must include frames with a position but no pose")
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:root) }
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let staged = root.appendingPathComponent("clip.mp4")
        try FileManager.default.copyItem(at:URL(fileURLWithPath:source),to:staged)
        let store = ClimbStore(root:root.appendingPathComponent("store"))
        let summary = try await VideoAnalyzer().videoSummary(url:staged)
        let record = try store.save(title:"",grade:ungradedValue,sent:false,note:"",stagedVideo:staged,thumbnail:Data(),analysis:summary,capture:CaptureInfo(analysisQueued:false))
        let first = try XCTUnwrap(selected.people?.first)
        let id = try XCTUnwrap(selected.subject?.picks.first?.personID)
        let candidate = try XCTUnwrap(first.candidates.first { $0.id == id })
        let anchors = selected.subject?.manual == false ? [] : [SubjectAnchor(time:first.time,candidate:candidate)]
        try store.selectSubject(record,analysis:selected,anchors:anchors)
        try store.setAnalysisQueued(record.id,true)
        let queue = AnalysisQueue(store:store,pack:pack)
        queue.setSuspended(false)
        let deadline = Date().addingTimeInterval(180)
        while !store.editPlans.contains(record.id) && Date() < deadline {
            if store.climbs.first?.capture?.analysisQueued == nil { break }
            try await Task.sleep(for:.milliseconds(100))
        }
        let climb = try XCTUnwrap(store.climbs.first)
        let plan = try XCTUnwrap(store.editPlan(for:climb),store.analysisMessages[record.id] ?? "No selected-subject edit plan")
        XCTAssertTrue(plan.tracksSubject)
        XCTAssertGreaterThan(plan.camera.count,2)
        XCTAssertGreaterThan(plan.trim.upperBound-plan.trim.lowerBound,selected.duration * 0.9)
        XCTAssertTrue(plan.camera.contains { $0.s > 1.05 },"A full-frame fallback alone cannot prove following")
        for position in positions where plan.trim.contains(position.time) {
            let window = EditWindow.at(position.time,keys:plan.camera,linear:plan.tracksSubject)
            let box = position.bounds
            XCTAssertLessThanOrEqual(window.x,box.x+0.005)
            XCTAssertLessThanOrEqual(window.y,box.y+0.005)
            XCTAssertGreaterThanOrEqual(window.x+window.side,box.x+box.width-0.005)
            XCTAssertGreaterThanOrEqual(window.y+window.side,box.y+box.height-0.005)
        }
        let rendered = try await EditRenderer(source:store.file(climb.videoName),plan:plan,directory:root.appendingPathComponent("render")).render(options:EditOptions(removePassersBy:false)) { _ in }
        let duration = try await AVURLAsset(url:rendered).load(.duration).seconds
        let timeline = EditTimeline(plan: plan, options: EditOptions(removePassersBy: false), sourceDuration: selected.duration)
        XCTAssertEqual(duration,timeline.duration,accuracy:0.3)
        if let output = env["BB_EDIT_OUT"] {
            let directory = URL(fileURLWithPath:output)
            try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
            try FileManager.default.copyItem(at:rendered,to:directory.appendingPathComponent("selected-camera.mp4"))
            try FileManager.default.copyItem(at:store.file("\(record.id).s\(climb.subjectRevision ?? 0)"+ClimbStore.editPlanSuffix),to:directory.appendingPathComponent("plan.json"))
        }
        print("Selected follow: \(positions.count) positions, \(selected.frames.count) real poses, \(plan.camera.count) camera keys, \(duration) seconds")
    }
}
