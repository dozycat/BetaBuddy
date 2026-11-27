import XCTest
import AVFoundation
import CoreImage
import SwiftUI
@testable import BetaBuddy

/// What comes with an analysis result besides the result: the render plan and the discipline.
/// The pack is the synthetic one; its analysis page makes both up.
@MainActor
final class EditFlowTests: XCTestCase {
    private var root: URL!
    private var clips: [URL] = []
    override func setUp() async throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("edit-flow-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }
    override func tearDown() async throws { for url in clips + [root!] { try? FileManager.default.removeItem(at: url) } }

    private func store() -> ClimbStore { ClimbStore(root: root.appendingPathComponent("store"), defaults: UserDefaults(suiteName: "edit-\(UUID())")!) }
    private func pack() throws -> GymPack {
        let directory = root.appendingPathComponent("pack-\(UUID())")
        try StandInPack.write(to: directory)
        return try GymPack(directory: directory.appendingPathComponent(GymPack.directoryName))
    }
    /// A record with made-up poses, as if pose recognition had already run, asking to be analysed.
    private func record(in store: ClimbStore, grade: String = ungradedValue, discipline: Discipline? = nil) async throws -> Climb {
        let joints = [Joint](repeating: Joint(x: 0.5, y: 0.5, confidence: 0.9), count: 33)
        let world = [[Double]](repeating: [0, 0, 0, 0.9], count: 33)
        let poses = PoseAnalysis(frames: (0..<10).map { PoseFrame(time: Double($0) * 0.2, joints: joints, world: world) }, sampledFrames: 10, duration: 2, aspectRatio: 0.5625)
        let clip = try await SyntheticWallClip.make(width: 360, height: 640, seconds: 2)
        clips.append(clip)
        return try store.save(title: "", grade: grade, sent: false, note: "", stagedVideo: clip, thumbnail: Data(), analysis: poses,
                              capture: CaptureInfo(analysisQueued: true), discipline: discipline)
    }
    private func until(_ what: String, within seconds: Double = 60, file: StaticString = #filePath, line: UInt = #line, _ condition: () -> Bool) async {
        let deadline = Date().addingTimeInterval(seconds)
        while !condition() {
            if Date() > deadline { XCTFail("Timed out: \(what)", file: file, line: line); return }
            try? await Task.sleep(for: .milliseconds(20))
        }
    }
    private let plan = #"{"v":1,"duration":4,"trim":[0,4],"segments":[{"t0":0,"t1":4,"speed":0.5}]}"#

    func testBelayProgressFitsCompactAndLargeTypeLayouts() async throws {
        for (name,size,category) in [
            ("normal",CGSize(width:360,height:420),ContentSizeCategory.large),
            ("large-type",CGSize(width:320,height:260),ContentSizeCategory.accessibilityExtraExtraExtraLarge),
            ("compact",CGSize(width:320,height:140),ContentSizeCategory.large)
        ] {
            let host = UIHostingController(rootView: BelayProgressView(progress: 0.42)
                .environment(\.sizeCategory,category)
                .frame(width:size.width,height:size.height).background(.black).ignoresSafeArea())
            let window = UIWindow(frame:CGRect(origin:.zero,size:size))
            window.rootViewController = host; window.makeKeyAndVisible()
            host.view.frame = window.bounds; host.view.setNeedsLayout(); host.view.layoutIfNeeded()
            try await Task.sleep(for:.milliseconds(100))
            let image = UIGraphicsImageRenderer(size:size).image { _ in host.view.drawHierarchy(in:window.bounds,afterScreenUpdates:true) }
            let attachment = XCTAttachment(image:image); attachment.name = "Belay progress \(name)"; attachment.lifetime = .keepAlways; add(attachment)
            window.isHidden = true; window.rootViewController = nil
        }
    }

    func testEditSessionRecoversFromWriteFailureAndCancellationCleansUp() async throws {
        let store = store()
        let source = try await SyntheticWallClip.make(width: 720, height: 1280, seconds: 12)
        clips.append(source)
        let record = try store.save(title: "", grade: ungradedValue, sent: false, note: "", stagedVideo: source, thumbnail: Data(),
                                    analysis: PoseAnalysis(frames: [], sampledFrames: 0, duration: 12, aspectRatio: 0.5625))
        let renderPlan = try XCTUnwrap(EditPlan(json: #"{"v":1,"duration":12,"trim":[0,12],"segments":[{"t0":0,"t1":12,"speed":0.5}]}"#))
        let directory = EditFiles.directory(for: record.id)
        try FileManager.default.createDirectory(at: directory.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("blocked output".utf8).write(to: directory)
        let session = EditSession(climb: record, plan: renderPlan, source: store.file(record.videoName))
        defer { session.close(); try? store.delete(record) }
        session.start()
        await until("write failure") { if case .failed = session.phase { return true }; return false }
        try FileManager.default.removeItem(at: directory)
        session.render()
        await until("real intermediate progress") { if case .rendering(let value) = session.phase { return value > 0 && value < 1 }; return false }
        await until("recovered export") { if case .ready = session.phase { return true }; return false }
        guard case .ready(let file) = session.phase else { return XCTFail("No recovered movie") }
        XCTAssertGreaterThan(try Data(contentsOf: file).count, 1000)
        session.render()
        await until("second render started") { if case .rendering(let value) = session.phase { return value > 0 && value < 1 }; return false }
        session.close()
        await until("cancelled files removed") { !FileManager.default.fileExists(atPath: directory.path) }
        try await Task.sleep(for: .milliseconds(300))
        if case .ready = session.phase { XCTFail("Cancelled render published a movie") }
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: store.file(record.videoName).path))
    }

    func testAResultBringsItsRenderPlanAndAGuessedDiscipline() async throws {
        let store = store()
        let record = try await record(in: store)
        XCTAssertNil(store.editPlan(for: record)); XCTAssertNil(record.style)
        let queue = AnalysisQueue(store: store, pack: try pack())
        _ = queue
        await until("the analysis") { store.analysisResults.contains(record.id) && store.climbs.first?.capture?.analysisQueued == nil }

        // The plan is kept next to the result, as the page gave it, and is one this build can render.
        XCTAssertTrue(store.editPlans.contains(record.id))
        let plan = try XCTUnwrap(store.editPlan(for: record))
        XCTAssertEqual(plan.segments.map(\.speed), [1, 0.5])
        XCTAssertTrue(plan.offersRemoval); XCTAssertTrue(plan.offersSlowMotion); XCTAssertTrue(plan.offersCamera)
        XCTAssertEqual(plan.plateTimes.count, 9)
        let kept = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: store.file("\(record.id)" + ClimbStore.editPlanSuffix))) as? [String: Any])
        XCTAssertEqual(kept["v"] as? Int, 1); XCTAssertNotNil(kept["keep"]); XCTAssertEqual((kept["segments"] as? [[String: Any]])?.first?["kind"] as? String, "run")

        // The page was not told a discipline, guessed one, and that is stored as a guess.
        let result = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try XCTUnwrap(store.analysisResult(for: record)).utf8)) as? [String: Any])
        XCTAssertTrue(result["discipline"] is NSNull)
        XCTAssertEqual(result["duration"] as? Double, 2)                   // The clip's real length went in with `begin`.
        XCTAssertEqual(plan.duration, 2)
        var climb = try XCTUnwrap(store.climbs.first)
        XCTAssertEqual(climb.style, .toprope); XCTAssertEqual(climb.disciplineAuto, true); XCTAssertNil(climb.chosenStyle)
        let reopened = ClimbStore(root: store.root)
        XCTAssertEqual(reopened.climbs.first?.discipline, "toprope")
        XCTAssertEqual(reopened.editPlans, [record.id]); XCTAssertNotNil(reopened.editPlan(for: record))

        // A later guess replaces a guess. The user's own choice replaces either and is never replaced.
        try store.setAutomaticDiscipline(.lead, for: record.id)
        XCTAssertEqual(store.climbs.first?.style, .lead)
        try store.organize(climb, title: "", grade: "5.10a", gymId: nil, gymName: nil, climbedAt: climb.climbingDate, folderId: nil, discipline: .toprope)
        climb = try XCTUnwrap(store.climbs.first)
        XCTAssertEqual(climb.chosenStyle, .toprope); XCTAssertNil(climb.disciplineAuto); XCTAssertEqual(climb.grade, "5.10a")
        try store.setAutomaticDiscipline(.boulder, for: record.id)
        XCTAssertEqual(store.climbs.first?.chosenStyle, .toprope)

        // Throwing the result away takes its plan with it; so does deleting the record.
        store.discardAnalysisResult(climb)
        XCTAssertNil(store.editPlan(for: climb)); XCTAssertTrue(store.editPlans.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: store.file("\(record.id)" + ClimbStore.editPlanSuffix).path))
        try store.saveEditPlan(self.plan, for: record.id)
        try FileManager.default.createDirectory(at: EditFiles.directory(for: record.id), withIntermediateDirectories: true)
        try store.delete(climb)
        XCTAssertFalse(FileManager.default.fileExists(atPath: store.file("\(record.id)" + ClimbStore.editPlanSuffix).path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: EditFiles.directory(for: record.id).path))
        XCTAssertTrue(ClimbStore(root: store.root).editPlans.isEmpty)
    }

    func testTheUsersOwnDisciplineIsHandedToThePageAndStaysTheirs() async throws {
        let store = store()
        let record = try await record(in: store, grade: "5.9", discipline: .lead)
        XCTAssertEqual(record.chosenStyle, .lead)
        let queue = AnalysisQueue(store: store, pack: try pack())
        _ = queue
        await until("the analysis") { store.analysisResults.contains(record.id) && store.climbs.first?.capture?.analysisQueued == nil }
        let result = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try XCTUnwrap(store.analysisResult(for: record)).utf8)) as? [String: Any])
        XCTAssertEqual(result["discipline"] as? String, "lead")           // In `begin`, as the user chose it.
        let climb = try XCTUnwrap(store.climbs.first)
        XCTAssertEqual(climb.chosenStyle, .lead); XCTAssertNil(climb.disciplineAuto); XCTAssertEqual(climb.grade, "5.9")
    }

    func testDisciplineGradeAndPlanRulesInTheStore() async throws {
        let store = store()
        let record = try await record(in: store, grade: "V4")
        // A grade the user marked on the bouldering scale is not cleared by a guess, and the guess is not taken.
        try store.setAutomaticDiscipline(.lead, for: record.id)
        XCTAssertNil(store.climbs.first?.discipline); XCTAssertEqual(store.climbs.first?.grade, "V4")
        try store.setAutomaticDiscipline(.boulder, for: record.id)
        XCTAssertEqual(store.climbs.first?.style, .boulder); XCTAssertEqual(store.climbs.first?.disciplineAuto, true)
        // Organising without touching the discipline leaves the guess a guess.
        var climb = try XCTUnwrap(store.climbs.first)
        try store.organize(climb, title: "a", grade: "V5", gymId: nil, gymName: nil, climbedAt: climb.climbingDate, folderId: nil)
        XCTAssertEqual(store.climbs.first?.disciplineAuto, true); XCTAssertEqual(store.climbs.first?.grade, "V5")
        // The user picks lead while the form still shows the V grade: the grade of the other scale is cleared.
        try store.organize(climb, title: "a", grade: "V5", gymId: nil, gymName: nil, climbedAt: climb.climbingDate, folderId: nil, discipline: .lead)
        climb = try XCTUnwrap(store.climbs.first)
        XCTAssertEqual(climb.chosenStyle, .lead); XCTAssertEqual(climb.grade, ungradedValue)
        try store.organize(climb, title: "a", grade: "5.11c", gymId: nil, gymName: nil, climbedAt: climb.climbingDate, folderId: nil)
        XCTAssertEqual(store.climbs.first?.grade, "5.11c")
        try store.organize(climb, title: "a", grade: "V2", gymId: nil, gymName: nil, climbedAt: climb.climbingDate, folderId: nil)
        XCTAssertEqual(store.climbs.first?.grade, ungradedValue)           // Still lead: a V grade does not fit.

        // A plan of a newer version is kept as it came but gives no edit; no plan removes the old one.
        try store.saveEditPlan(plan, for: record.id)
        XCTAssertEqual(store.editPlan(for: climb)?.segments.first?.speed, 0.5)
        try store.saveEditPlan(plan.replacingOccurrences(of: "\"v\":1", with: "\"v\":2"), for: record.id)
        XCTAssertTrue(store.editPlans.contains(record.id)); XCTAssertNil(store.editPlan(for: climb))
        try store.saveEditPlan(nil, for: record.id)
        XCTAssertFalse(store.editPlans.contains(record.id))
        try store.saveEditPlan(plan, for: UUID())                          // A record that is gone: nothing is written.
        XCTAssertTrue(store.editPlans.isEmpty)
    }

    func testMessageAndBeginCarryThePlanAndTheDiscipline() throws {
        let body: [String: Any] = ["type": "analysisResult", "id": "abc", "json": "{}",
                                   "edit": ["v": 1, "duration": 4, "trim": [0, 4], "segments": [["t0": 0, "t1": 4, "speed": 0.5]], "later": ["x": 1]],
                                   "discipline": ["kind": "lead", "sure": false]]
        guard case .result(_, _, _, _, let edit, let discipline)? = AnalysisMessage(body: body) else { return XCTFail("not a result") }
        XCTAssertEqual(discipline, DisciplineGuess(kind: .lead, sure: false))
        let text = try XCTUnwrap(edit)
        XCTAssertTrue(text.contains("\"later\""))                          // Kept as it came, unknown parts and all.
        XCTAssertEqual(EditPlan(json: text)?.segments, [EditPlan.Segment(t0: 0, t1: 4, speed: 0.5)])
        // Without them, or with something else in their place, the result is still a result.
        for odd: [String: Any] in [[:], ["edit": "plan", "discipline": "lead"], ["edit": [1, 2], "discipline": ["kind": "speed"]]] {
            XCTAssertEqual(AnalysisMessage(body: ["type": "analysisResult", "json": "{}"].merging(odd) { $1 }), .result(id: "", json: "{}", climber: [:]))
        }
        XCTAssertTrue(AnalysisCommand.begin(id: "a", width: 1, height: 1, climber: nil, discipline: .toprope).script.contains(#""discipline":"toprope""#))
        XCTAssertFalse(AnalysisCommand.begin(id: "a", width: 1, height: 1, climber: nil).script.contains("discipline"))
        XCTAssertTrue(AnalysisCommand.begin(id: "a", width: 1, height: 1, climber: nil, duration: 12.34567).script.contains(#""duration":12.3457"#))
        for none in [nil, 0, -1, Double.nan, .infinity] as [Double?] {
            XCTAssertFalse(AnalysisCommand.begin(id: "a", width: 1, height: 1, climber: nil, duration: none).script.contains("duration"))
        }
    }
}

/// The installed pack's own plan for a real climbing clip, rendered. Neither is in this repository,
/// so this is skipped unless both are there:
///
///     TEST_RUNNER_BB_ANALYSIS_CLIP=<clip> xcodebuild … test -only-testing:BetaBuddyTests/EditInstalledPackTests
///
/// `TEST_RUNNER_BB_EDIT_OUT=<directory>` also writes the rendered files and a few of their frames there.
@MainActor
final class EditInstalledPackTests: XCTestCase {
    /// Nobody is looked for; one made-up "passer-by" in the bottom-left corner of every frame, so that part of
    /// the plate is seen. The first picture it is shown is the plate itself, where it reports nobody.
    final class CornerFinder: PeopleFinder {
        private var asked = 0
        func people(in image: CIImage, context: CIContext) throws -> [FoundPerson] {
            asked += 1
            return asked == 1 ? [] : [FoundPerson(box: EditPlan.Box(x: 0.03, y: 0.72, w: 0.2, h: 0.22), mask: nil)]
        }
    }

    func testInstalledPackPlansAndRendersALocalClip() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["BB_ANALYSIS_CLIP"], FileManager.default.fileExists(atPath: path) else { throw XCTSkip("Optional private climbing clip is not supplied") }
        guard let pack = GymPack.current, pack.supportsAnalysis else { throw XCTSkip("The installed build has no pack with an analysis page") }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("edit-real-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let staged = root.appendingPathComponent("clip." + URL(fileURLWithPath: path).pathExtension)
        try FileManager.default.copyItem(at: URL(fileURLWithPath: path), to: staged)
        let store = ClimbStore(root: root.appendingPathComponent("store"), defaults: UserDefaults(suiteName: "edit-real-\(UUID())")!)
        let summary = try await VideoAnalyzer().videoSummary(url: staged)
        let record = try store.save(title: "", grade: ungradedValue, sent: false, note: "", stagedVideo: staged, thumbnail: Data(), analysis: summary,
                                    pendingAnalysis: AnalysisQuality.balanced.rawValue, capture: CaptureInfo(analysisQueued: true))
        let queue = AnalysisQueue(store: store, pack: pack)
        _ = queue
        store.resumeAnalysis()
        let deadline = Date().addingTimeInterval(900)
        while !(store.analysisResults.contains(record.id) && store.climbs.first?.capture?.analysisQueued == nil) {
            if store.climbs.first?.capture?.analysisQueued == nil, store.climbs.first?.pendingAnalysis == nil { break }   // Given up on.
            if Date() > deadline { return XCTFail("no analysis after fifteen minutes") }
            try await Task.sleep(for: .milliseconds(100))
        }
        XCTAssertTrue(store.analysisResults.contains(record.id), store.analysisMessages[record.id] ?? "no result")
        let climb = try XCTUnwrap(store.climbs.first)
        var notes = ["pack \(pack.manifest.packVersion)", "clip \(String(format: "%.2f", climb.analysis.duration)) s, \(climb.analysis.frames.count) pose frames",
                     "discipline \(climb.discipline ?? "none") auto \(climb.disciplineAuto ?? false)", "plan file \(store.editPlans.contains(record.id))"]
        func keepNotes() { let note = XCTAttachment(string: notes.joined(separator: "\n")); note.name = "Installed pack plan and render"; note.lifetime = .keepAlways; add(note) }
        defer { keepNotes(); if let out = environment["BB_EDIT_OUT"] { try? notes.joined(separator: "\n").write(toFile: out + "/notes.txt", atomically: true, encoding: .utf8) } }
        // What the page was handed, for comparing with other ways of feeding it: the poses as `addPoses` gives them, and the stills' times and size.
        if let out = environment["BB_EDIT_OUT"] {
            try? FileManager.default.createDirectory(atPath: out, withIntermediateDirectories: true)
            let script = AnalysisCommand.addPoses(climb.analysis.frames).script
            if let from = script.range(of: ".addPoses(", options: .backwards), let to = script.range(of: "), true) : false", options: .backwards) {
                try? String(script[from.upperBound..<to.lowerBound]).write(toFile: out + "/fed-poses.json", atomically: true, encoding: .utf8)
            }
            if let stills = try? await VideoAnalyzer().stills(url: store.file(climb.videoName)) {
                try? "w \(stills.width) h \(stills.height) frames \(stills.frames.count) times \(stills.frames.map { String(format: "%.3f", $0.time) }.joined(separator: " "))"
                    .write(toFile: out + "/fed-stills.txt", atomically: true, encoding: .utf8)
            }
        }
        // The pack may decline to plan a clip (one with a cut in the attempt, for instance): that is its call, not a failure here.
        guard let plan = store.editPlan(for: climb) else {
            notes.append("the result brought no plan")
            throw XCTSkip("The installed pack gave no edit plan for this clip")
        }
        // The plan as the page gave it, for looking at and for the Mac backtest of the renderer.
        if let out = environment["BB_EDIT_OUT"] {
            try? FileManager.default.createDirectory(atPath: out, withIntermediateDirectories: true)
            try? FileManager.default.copyItem(at: store.file("\(record.id)" + ClimbStore.editPlanSuffix), to: URL(fileURLWithPath: out).appendingPathComponent("plan.json"))
        }
        notes.append("plan: duration \(plan.duration), trim \(plan.trim), \(plan.segments.count) segments, speeds \(plan.segments.map { String(format: "%.2f", $0.speed) }.joined(separator: " "))")
        notes.append("camera keys \(plan.camera.count), segment cameras \(plan.segments.filter { $0.camera != nil }.count), keep samples \(plan.keep?.samples.count ?? 0), boxes per sample \(Set(plan.keep?.samples.map(\.boxes.count) ?? [])), plate times \(plan.plateTimes.count)")
        XCTAssertEqual(plan.duration, climb.analysis.duration, accuracy: 0.5)

        let out = environment["BB_EDIT_OUT"].map { URL(fileURLWithPath: $0, isDirectory: true) }
        if let out { try? FileManager.default.createDirectory(at: out, withIntermediateDirectories: true) }
        func check(_ name: String, _ options: EditOptions, finder: PeopleFinder) async throws {
            let began = Date()
            let url = try await EditRenderer(source: store.file(climb.videoName), plan: plan, directory: root.appendingPathComponent("edit-" + name), finder: finder).render(options: options) { _ in }
            let asset = AVURLAsset(url: url)
            let seconds = try await asset.load(.duration).seconds
            let expected = EditTimeline(plan: plan, options: options, sourceDuration: climb.analysis.duration).duration
            notes.append("\(name): \(String(format: "%.2f", seconds)) s (timeline \(String(format: "%.2f", expected)) s), rendered in \(String(format: "%.1f", Date().timeIntervalSince(began))) s")
            XCTAssertEqual(seconds, expected, accuracy: 0.3)
            guard let out else { return }
            try? FileManager.default.removeItem(at: out.appendingPathComponent(name + ".mp4"))
            try FileManager.default.copyItem(at: url, to: out.appendingPathComponent(name + ".mp4"))
        }
        // Vision cannot run on every simulator, so here nobody is looked for: first no removal at all, then a made-up passer-by.
        try await check("slow-camera", EditOptions(removePassersBy: false), finder: CornerFinder())
        try await check("plain", EditOptions(removePassersBy: false, slowMotion: false, cameraMoves: false), finder: CornerFinder())
        if plan.offersRemoval { try await check("corner-replaced", EditOptions(slowMotion: false, cameraMoves: false), finder: CornerFinder()) }
    }
}
