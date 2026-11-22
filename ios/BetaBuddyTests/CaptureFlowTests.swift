import XCTest
import AVFoundation
import Combine
import UIKit
@testable import BetaBuddy

/// The capture screen's logic end to end, without a camera: a drawn wall played from a file is the
/// source, and a stand-in page with a fixed script is the pack's capture page. Nothing here says
/// anything about how a real page judges a camera position.
@MainActor
final class CaptureFlowTests: XCTestCase {
    private var root: URL!
    private var clips: [URL] = []
    private var watching: Set<AnyCancellable> = []

    override func setUp() async throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("capture-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }
    override func tearDown() async throws {
        watching = []
        for url in clips + [root!] { try? FileManager.default.removeItem(at: url) }
        clips = []
    }

    private func pack(capture: Bool = true, analysis: Bool = true, step: Int = 3) throws -> GymPack {
        let directory = root.appendingPathComponent("pack-\(UUID())")
        try StandInPack.write(to: directory, capture: capture, analysis: analysis, step: step)
        return try GymPack(directory: directory.appendingPathComponent(GymPack.directoryName))
    }
    private func wall(seconds: Double = 2, width: Int = 360, height: Int = 640, photo: CGImage? = nil) async throws -> URL {
        let url = try await SyntheticWallClip.make(width: width, height: height, seconds: seconds, photo: photo)
        clips.append(url)
        return url
    }
    private func store() -> ClimbStore { ClimbStore(root: root.appendingPathComponent("store"), defaults: UserDefaults(suiteName: "capture-\(UUID())")!) }
    private func session(_ store: ClimbStore, _ pack: GymPack?, _ clip: URL) -> CaptureSession {
        let session = CaptureSession(store: store, pack: pack, source: FileCaptureSource(url: clip, horizontalFieldOfView: 41.5))
        session.framingInterval = 0.04
        session.recordingInterval = 0.2
        session.watchInterval = .milliseconds(100)
        return session
    }
    /// Polls on the main actor until `condition` holds.
    private func until(_ what: String, within seconds: Double = 20, file: StaticString = #filePath, line: UInt = #line, _ condition: () -> Bool) async {
        let deadline = Date().addingTimeInterval(seconds)
        while !condition() {
            if Date() > deadline { XCTFail("Timed out: \(what)", file: file, line: line); return }
            try? await Task.sleep(for: .milliseconds(20))
        }
    }
    private func pageState(_ session: CaptureSession) async throws -> [String: Any] {
        let view = try XCTUnwrap(session.webView)
        let value = try await view.evaluateJavaScript("window.BetaBuddyCapture.state()")
        return try XCTUnwrap(value as? [String: Any])
    }
    private func person() throws -> CGImage {
        let url = try XCTUnwrap(Bundle(for: Self.self).url(forResource: "pose", withExtension: "jpg"))
        return try XCTUnwrap(UIImage(contentsOfFile: url.path)?.cgImage)
    }

    func testFramingGuidanceColourCorrectionsRecordingAndSaveThenAnalyse() async throws {
        let store = store()
        let pack = try pack()
        let session = session(store, pack, try await wall(seconds: 2))
        session.gymName = StandInPack.directoryGymName
        session.gymId = "syn:east"
        session.grade = "V4"
        XCTAssertEqual(session.page, .loading)
        var events: [String] = []
        session.$phase.sink { events.append("phase:\($0)") }.store(in: &watching)
        store.$analysisProgress.sink { if !$0.isEmpty { events.append("analysis") } }.store(in: &watching)
        var hints: [String] = []
        var seen: [CaptureAssessment] = []
        session.$assessment.sink { value in
            guard let value else { return }
            seen.append(value)
            let code = value.ready ? "ready" : value.hint?.code ?? "settling"
            if hints.last != code { hints.append(code) }
        }.store(in: &watching)

        session.start()
        XCTAssertTrue(store.cameraOpen)                      // Nothing else is analysed while the camera is in use.
        await until("camera and page") { session.camera == .running && session.page == .ready }
        await until("first assessment") { !seen.isEmpty }
        let first = try XCTUnwrap(seen.first)
        XCTAssertEqual(first.hint, .init(code: "too_dark", blocking: true, move: nil))
        XCTAssertFalse(first.ok)
        XCTAssertEqual(first.palette.map(\.key), ["yellow", "blue", "red"])
        XCTAssertTrue(first.holds.isEmpty)                   // No colour chosen yet.
        // The page was told the size of the frames it gets, the field of view, and the pack's own venue.
        var state = try await pageState(session)
        let meta = try XCTUnwrap(state["meta"] as? [String: Any])
        XCTAssertEqual(meta["w"] as? Int, 360); XCTAssertEqual(meta["h"] as? Int, 640)
        XCTAssertEqual(meta["fovH"] as? Double, 41.5); XCTAssertEqual(meta["venue"] as? String, StandInPack.venueID)
        XCTAssertEqual((state["decoded"] as? [String: Any])?["w"] as? Int, 360)     // What arrived really was a JPEG of that size.
        XCTAssertEqual(session.frameSize, CGSize(width: 360, height: 640))

        await until("ready") { session.assessment?.ready == true }
        XCTAssertEqual(hints, ["too_dark", "cut_top", "lens_flare_2031", "settling", "ready"])
        XCTAssertFalse(session.needsConfirmation)

        session.selectColor("yellow")
        await until("holds of the chosen colour") { session.assessment?.holds.count == 6 }
        var holds = try XCTUnwrap(session.assessment).holds
        XCTAssertEqual(holds.filter { $0.role == .startHand }.map(\.id), ["y1", "y2"])
        XCTAssertEqual(holds.first { $0.role == .finish }?.id, "y6")
        XCTAssertTrue(try XCTUnwrap(holds.first { $0.id == "y5" }).poly.isEmpty)    // No outline: its box is drawn.
        XCTAssertFalse(holds.contains(where: \.user))

        // The user says another hold is the finish, and that a blue one belongs to the problem.
        session.correct(at: CGPoint(x: 0.50, y: 0.60), role: .finish)
        session.correct(at: CGPoint(x: 0.66, y: 0.50), role: .hold)
        session.correct(at: CGPoint(x: 0.36, y: 0.90), role: .none)
        await until("corrections") { session.assessment?.holds.first { $0.id == "y4" }?.role == .finish && session.assessment?.holds.contains { $0.id == "b3" } == true }
        holds = try XCTUnwrap(session.assessment).holds
        XCTAssertTrue(try XCTUnwrap(holds.first { $0.id == "y4" }).user)
        XCTAssertFalse(holds.contains { $0.id == "y3" })
        session.resetCorrections()
        await until("corrections cleared") { session.assessment?.holds.count == 6 && session.assessment?.holds.contains(where: \.user) == false }
        session.correct(at: CGPoint(x: 0.50, y: 0.60), role: .startHand)
        await until("correction kept for the plan") { session.assessment?.holds.filter { $0.role == .startHand }.count == 3 }

        session.startRecording()
        XCTAssertEqual(session.phase, .recording)
        try await Task.sleep(for: .seconds(1.6))
        XCTAssertEqual(session.phase, .recording)
        XCTAssertNotNil(session.summary, "The page answered lock() with its plan")
        XCTAssertTrue(store.climbs.isEmpty)
        await session.stop(.user)

        guard case .saved(let id) = session.phase else { return XCTFail("not saved: \(session.phase)") }
        // Saved first: what is on disk at this point is complete, whatever happens to the analysis.
        let reopened = ClimbStore(root: store.root)
        let record = try XCTUnwrap(reopened.climbs.first { $0.id == id })
        XCTAssertEqual(record.grade, "V4"); XCTAssertEqual(record.gymId, "syn:east"); XCTAssertEqual(record.gymName, StandInPack.directoryGymName)
        XCTAssertEqual(record.pendingAnalysis, AnalysisQuality.balanced.rawValue)        // The request is part of the saved record.
        XCTAssertEqual(record.climbingDate.timeIntervalSinceNow, -1.6, accuracy: 5)
        XCTAssertGreaterThan(record.analysis.duration, 1); XCTAssertLessThan(record.analysis.duration, 4)
        XCTAssertEqual(record.analysis.aspectRatio, 360.0 / 640, accuracy: 0.01)
        let capture = try XCTUnwrap(record.capture)
        XCTAssertEqual(capture.colorKey, "yellow"); XCTAssertEqual(capture.colorRGB, [226, 196, 40])
        XCTAssertEqual(capture.packVersion, StandInPack.version)
        XCTAssertEqual(capture.analysisQueued, true); XCTAssertTrue(capture.hasPlan)
        XCTAssertEqual(capture.summary, CaptureSummary(color: "yellow", rgb: [226, 196, 40], holds: 6, startHands: 3, startFeet: 1, finish: true, ok: true, issues: []))
        let plan = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try XCTUnwrap(reopened.capturePlan(for: record)).utf8)) as? [String: Any])
        XCTAssertEqual(plan["color"] as? String, "yellow")
        XCTAssertEqual((plan["meta"] as? [String: Any])?["venue"] as? String, StandInPack.venueID)
        XCTAssertTrue(try XCTUnwrap(plan["holds"] as? [[String: Any]]).contains { $0["id"] as? String == "y4" && $0["role"] as? String == "start_hand" && $0["user"] as? Bool == true })
        XCTAssertGreaterThan(try Data(contentsOf: store.file(record.videoName)).count, 10_000)
        XCTAssertNotNil(UIImage(contentsOfFile: store.file(record.thumbnailName).path))
        let clip = AVURLAsset(url: store.file(record.videoName))
        let track = try await clip.loadTracks(withMediaType: .video).first
        let size = try await XCTUnwrap(track).load(.naturalSize)
        XCTAssertEqual(size, CGSize(width: 360, height: 640))
        let written = await VideoAnalyzer().recordingDate(url: store.file(record.videoName))
        XCTAssertEqual(try XCTUnwrap(written).timeIntervalSince(record.climbingDate), 0, accuracy: 2)

        // …and only then was analysis let go.
        await until("pose recognition to start and end") { events.contains("analysis") && store.climbs.first?.pendingAnalysis == nil }
        let saved = try XCTUnwrap(events.firstIndex(of: "phase:saved(\(id))"))
        XCTAssertEqual(Array(events.prefix(saved + 1)), ["phase:framing", "phase:recording", "phase:saving", "phase:saved(\(id))"])
        XCTAssertGreaterThan(try XCTUnwrap(events.firstIndex(of: "analysis")), saved)
        XCTAssertFalse(store.cameraOpen)
        // A drawn wall has nobody on it, so pose recognition says so; the clip stays.
        XCTAssertEqual(store.analysisMessages[id], AnalysisError.noPose.errorDescription)
        XCTAssertEqual(store.climbs.count, 1)

        // After lock the page kept getting frames, more slowly, and judged nothing new.
        state = try await pageState(session)
        XCTAssertEqual(state["locked"] as? Bool, true)
        let print = XCTAttachment(string: session.metrics.summary); print.name = "Frame hand-over, 360x640 drawn wall, no person"; print.lifetime = .keepAlways; add(print)
        XCTAssertGreaterThan(session.metrics.frames, 15)

        // Another attempt starts a fresh judgement and holds analysis back again.
        session.frameAgain()
        XCTAssertEqual(session.phase, .framing); XCTAssertTrue(store.cameraOpen); XCTAssertNil(session.assessment)
        XCTAssertEqual(session.color, "yellow")
        await until("fresh judgement with the colour kept") { session.assessment?.color == "yellow" && session.assessment?.holds.count == 6 }
        await session.close()
        XCTAssertFalse(store.cameraOpen); XCTAssertNil(session.webView)
    }

    func testWithoutACapturePageTheCameraStillRecordsAndSaves() async throws {
        let store = store()
        for pack in [try pack(capture: false), nil] as [GymPack?] {
            let session = session(store, pack, try await wall())
            XCTAssertEqual(session.page, .unavailable)
            session.start()
            await until("camera") { session.camera == .running }
            try await Task.sleep(for: .milliseconds(400))
            XCTAssertNil(session.assessment); XCTAssertNil(session.webView); XCTAssertTrue(session.palette.isEmpty)
            XCTAssertFalse(session.needsConfirmation)
            XCTAssertEqual(session.metrics.frames, 0)            // Nothing is prepared for a page that is not there.
            session.startRecording()
            try await Task.sleep(for: .seconds(1.3))
            await session.stop(.user)
            guard case .saved(let id) = session.phase else { return XCTFail("not saved: \(session.phase)") }
            let record = try XCTUnwrap(store.climbs.first { $0.id == id })
            XCTAssertEqual(record.capture, CaptureInfo(analysisQueued: pack == nil ? nil : true))
            XCTAssertNil(store.capturePlan(for: record))
            XCTAssertEqual(record.pendingAnalysis, AnalysisQuality.balanced.rawValue)
            await session.close()
        }
        XCTAssertEqual(store.climbs.count, 2)
    }

    func testInterruptionLimitStorageAndTooShort() async throws {
        let store = store()
        let pack = try pack(capture: false)

        // A call or another app takes the camera away: what was recorded is closed and saved.
        var source = FileCaptureSource(url: try await wall())
        var session = CaptureSession(store: store, pack: pack, source: source)
        session.start()
        await until("camera") { session.camera == .running }
        session.startRecording()
        try await Task.sleep(for: .seconds(1.3))
        source.interrupt()
        await until("saved after the interruption") { if case .saved = session.phase { true } else { false } }
        XCTAssertEqual(session.stopReason, .interrupted)
        XCTAssertEqual(store.climbs.count, 1)
        await session.close()

        // So does the app going to the background.
        source = FileCaptureSource(url: try await wall())
        session = CaptureSession(store: store, pack: pack, source: source)
        session.start()
        await until("camera") { session.camera == .running }
        session.startRecording()
        try await Task.sleep(for: .seconds(1.3))
        NotificationCenter.default.post(name: UIApplication.didEnterBackgroundNotification, object: nil)
        await until("saved after backgrounding") { if case .saved = session.phase { true } else { false } }
        XCTAssertEqual(session.stopReason, .interrupted)
        XCTAssertEqual(store.climbs.count, 2)
        await session.close()

        // The length limit closes the clip by itself.
        session = CaptureSession(store: store, pack: pack, source: FileCaptureSource(url: try await wall()))
        session.maximumDuration = 1.2
        session.start()
        await until("camera") { session.camera == .running }
        session.startRecording()
        await until("saved at the limit") { if case .saved = session.phase { true } else { false } }
        XCTAssertEqual(session.stopReason, .limit)
        let limited = try XCTUnwrap(store.climbs.first { session.phase == .saved($0.id) })
        XCTAssertEqual(limited.analysis.duration, 1.2, accuracy: 0.3)
        await session.close()

        // Too little space: recording does not start; space running out closes a recording and keeps it.
        session = CaptureSession(store: store, pack: pack, source: FileCaptureSource(url: try await wall()))
        session.watchInterval = .milliseconds(100)
        var free: Int64 = 10 * 1024 * 1024
        session.freeBytes = { free }
        session.start()
        await until("camera") { session.camera == .running }
        session.startRecording()
        XCTAssertEqual(session.phase, .framing); XCTAssertEqual(session.notice, .lowStorage)
        session.notice = nil
        free = CaptureLimits.startingFreeBytes + 1
        session.startRecording()
        XCTAssertEqual(session.phase, .recording)
        try await Task.sleep(for: .seconds(1.3))
        free = CaptureLimits.stoppingFreeBytes - 1
        await until("saved when space ran out") { if case .saved = session.phase { true } else { false } }
        XCTAssertEqual(session.stopReason, .storage)
        XCTAssertEqual(store.climbs.count, 4)

        // Stopped at once: nothing worth keeping, and the screen is framing again.
        session.frameAgain()
        session.freeBytes = { nil }
        session.startRecording()
        await session.stop(.user)
        XCTAssertEqual(session.phase, .framing); XCTAssertEqual(session.notice, .tooShort)
        XCTAssertEqual(store.climbs.count, 4)
        await session.close()
        XCTAssertTrue((try FileManager.default.contentsOfDirectory(atPath: NSTemporaryDirectory())).filter { $0.hasPrefix("betabuddy-capture-") }.isEmpty)
    }

    func testAFailedSaveKeepsTheClipForAnotherTry() async throws {
        let store = store()
        let session = session(store, nil, try await wall())
        session.start()
        await until("camera") { session.camera == .running }
        session.startRecording()
        try await Task.sleep(for: .seconds(1.3))
        // The album cannot be written.
        try FileManager.default.removeItem(at: store.root)
        try Data().write(to: store.root)
        await session.stop(.user)
        guard case .saveFailed = session.phase else { return XCTFail("\(session.phase)") }
        XCTAssertTrue(store.climbs.isEmpty)
        try FileManager.default.removeItem(at: store.root)
        try FileManager.default.createDirectory(at: store.root, withIntermediateDirectories: true)
        await session.save()
        guard case .saved = session.phase else { return XCTFail("\(session.phase)") }
        XCTAssertEqual(ClimbStore(root: store.root).climbs.count, 1)
        await session.close()
    }

    /// A photo of a person in front of the drawn wall, so that pose recognition has someone to find:
    /// recording → saved → pose recognition → the pack's analysis, which is handed the capture plan.
    func testRecordedClipIsAnalysedAfterSavingAndTheAnalysisPageGetsThePlan() async throws {
        let store = store()
        let pack = try pack()
        let queue = AnalysisQueue(store: store, pack: pack)
        let session = session(store, pack, try await wall(seconds: 2, photo: try person()))
        session.start()
        await until("ready") { session.assessment?.ready == true }
        session.selectColor("blue")
        await until("holds") { session.assessment?.holds.count == 5 }
        session.startRecording()
        try await Task.sleep(for: .seconds(2.2))
        XCTAssertNil(queue.running)
        await session.stop(.user)
        guard case .saved(let id) = session.phase else { return XCTFail("not saved: \(session.phase)") }
        XCTAssertFalse(store.analysisResults.contains(id))
        let plan = try XCTUnwrap(store.capturePlan(for: try XCTUnwrap(store.climbs.first)))

        await until("pose recognition", within: 120) { store.climbs.first?.pendingAnalysis == nil }
        XCTAssertTrue(try XCTUnwrap(store.climbs.first).analysis.hasWorldLandmarks, store.analysisMessages[id] ?? "")
        await until("analysis after pose recognition", within: 60) { store.analysisResults.contains(id) && store.climbs.first?.capture?.analysisQueued == nil }
        let result = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try XCTUnwrap(store.analysisResult(for: try XCTUnwrap(store.climbs.first))).utf8)) as? [String: Any])
        XCTAssertEqual(result["capture"] as? String, plan)               // Handed over exactly as stored.
        XCTAssertGreaterThan(try XCTUnwrap(result["poses"] as? Int), 3)
        XCTAssertGreaterThan(try XCTUnwrap(result["frames"] as? Int), 3)
        XCTAssertNil(queue.running); XCTAssertNil(queue.webView)
        XCTAssertNotNil(store.analysisMessages[id])
        // After a restart nothing is left to do, and the plan is still there.
        let reopened = ClimbStore(root: store.root)
        XCTAssertNil(reopened.climbs.first?.capture?.analysisQueued)
        XCTAssertEqual(reopened.capturePlan(for: try XCTUnwrap(reopened.climbs.first)), plan)

        // Live pose found the person in the preview frames, and the page was told.
        let print = XCTAttachment(string: session.metrics.summary); print.name = "Frame hand-over with live pose, 360x640 with a person"; print.lifetime = .keepAlways; add(print)
        XCTAssertGreaterThan(session.metrics.posesFound, 0, session.metrics.summary)
        await session.close()
    }

    /// Not an assertion of speed: the simulator runs on the Mac's processor. It keeps the numbers
    /// with the test results so they can be compared from run to run.
    func testCostOfHandingOverFullSizeFrames() async throws {
        let store = store()
        let session = CaptureSession(store: store, pack: try pack(), source: FileCaptureSource(url: try await wall(seconds: 3, width: 1080, height: 1920, photo: try person())))
        session.start()
        await until("thirty frames at three a second", within: 30) { session.metrics.frames >= 30 }
        let metrics = session.metrics
        XCTAssertEqual(session.frameSize, CGSize(width: 360, height: 640))          // 1080 × 1920 handed over with its short side at 360.
        XCTAssertLessThan(metrics.jpegBytes / metrics.frames, 120_000)
        let print = XCTAttachment(string: metrics.summary); print.name = "Frame hand-over with live pose, 1080x1920 source at 3 fps"; print.lifetime = .keepAlways; add(print)
        NSLog("CAPTURE-COST %@", metrics.summary)
        await session.close()
    }
}
