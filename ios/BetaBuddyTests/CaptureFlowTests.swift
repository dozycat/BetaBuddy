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

    private func pack(capture: Bool = true, analysis: Bool = true, step: Int = 3, directoryIds: Bool = true, captureComesUp: Bool = true) throws -> GymPack {
        let directory = root.appendingPathComponent("pack-\(UUID())")
        try StandInPack.write(to: directory, capture: capture, analysis: analysis, step: step, directoryIds: directoryIds, captureComesUp: captureComesUp)
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
        session.pickGrade("V4")
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
        XCTAssertNil(session.recognised)

        session.selectColor("yellow")
        await until("holds of the chosen colour") { session.assessment?.holds.count == 6 }
        var holds = try XCTUnwrap(session.assessment).holds
        XCTAssertEqual(holds.filter { $0.role == .startHand }.map(\.id), ["y1", "y2"])
        XCTAssertEqual(holds.first { $0.role == .finish }?.id, "y6")
        XCTAssertTrue(try XCTUnwrap(holds.first { $0.id == "y5" }).poly.isEmpty)    // No outline: its box is drawn.
        XCTAssertFalse(holds.contains(where: \.user))
        // Start and finish are the page's guesses until the user marks them; the page keeps saying so, without blocking.
        var judged = try XCTUnwrap(session.assessment)
        XCTAssertFalse(judged.startConfirmed); XCTAssertFalse(judged.finishConfirmed)
        XCTAssertEqual(judged.hint, .init(code: "start_unclear", blocking: false, move: nil)); XCTAssertTrue(judged.ready)
        // The gym is one of the pack's venues, so the page may say which of its problems this is.
        let recognised = try XCTUnwrap(session.recognised)
        XCTAssertEqual(recognised.id, StandInPack.venueID + "/yellow-01"); XCTAssertEqual(recognised.confidence, 0.9)
        XCTAssertEqual(recognised.nameZh, "合成线路 黄 1"); XCTAssertEqual(recognised.grade, "V3")
        XCTAssertTrue(["合成线路 黄 1", "Synthetic Yellow 1"].contains(recognised.title))       // In the language the page was loaded with.
        XCTAssertEqual(session.grade, "V4"); XCTAssertFalse(session.gradePrefilled)            // The user picked first: nothing is filled in.

        // The user says another hold is the finish, and that a blue one belongs to the problem.
        session.correct(at: CGPoint(x: 0.50, y: 0.60), role: .finish)
        session.correct(at: CGPoint(x: 0.66, y: 0.50), role: .hold)
        session.correct(at: CGPoint(x: 0.36, y: 0.90), role: .none)
        await until("corrections") { session.assessment?.holds.first { $0.id == "y4" }?.role == .finish && session.assessment?.holds.contains { $0.id == "b3" } == true }
        holds = try XCTUnwrap(session.assessment).holds
        XCTAssertTrue(try XCTUnwrap(holds.first { $0.id == "y4" }).user)
        XCTAssertEqual(holds.first { $0.id == "y6" }?.role, .hold)              // There is one finish.
        XCTAssertFalse(holds.contains { $0.id == "y3" })
        XCTAssertTrue(try XCTUnwrap(session.assessment).finishConfirmed)
        // A tap where there is no hold changes nothing and is said to have missed.
        session.correct(at: CGPoint(x: 0.95, y: 0.04), role: .finish)
        await until("the missed tap is noticed") { session.tapMissed }
        XCTAssertEqual(session.assessment?.holds.first { $0.role == .finish }?.id, "y4")
        session.resetCorrections()
        await until("corrections cleared") { session.assessment?.holds.count == 6 && session.assessment?.holds.contains(where: \.user) == false }
        session.correct(at: CGPoint(x: 0.50, y: 0.60), role: .startHand)
        await until("the marked start replaces the guessed one") { session.assessment?.holds.filter { $0.role == .startHand }.map(\.id) == ["y4"] }
        judged = try XCTUnwrap(session.assessment)
        XCTAssertTrue(judged.startConfirmed); XCTAssertFalse(judged.finishConfirmed)
        await until("nothing left to say") { session.assessment?.hint == nil }

        let handedOver = session.metrics.frames
        session.startRecording()
        XCTAssertEqual(session.phase, .recording)
        try await Task.sleep(for: .seconds(1.6))
        XCTAssertEqual(session.phase, .recording)
        XCTAssertNotNil(session.summary, "The page answered lock() with its plan")
        XCTAssertTrue(store.climbs.isEmpty)
        // The page uses nothing after `lock`: no more frames are prepared for it while recording.
        XCTAssertLessThanOrEqual(session.metrics.frames, handedOver + 1)
        // While it is being recorded the clip lies in the store's own directory with what is known about it, in case the app does not live to save it.
        let recording = CaptureRecovery.directory(in: store.root)
        let lying = try FileManager.default.contentsOfDirectory(atPath: recording.path).sorted()
        XCTAssertEqual(lying.map { ($0 as NSString).pathExtension }, ["json", "mov"])
        let draft = try JSONDecoder().decode(CaptureDraft.self, from: Data(contentsOf: recording.appendingPathComponent(lying[0])))
        XCTAssertEqual(draft.grade, "V4"); XCTAssertEqual(draft.gymId, "syn:east"); XCTAssertEqual(draft.info.colorKey, "yellow"); XCTAssertNotNil(draft.plan)
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
        // In the album the problem keeps its id and its own name; the name and grade in the interface language are not written down.
        XCTAssertEqual(capture.summary, CaptureSummary(color: "yellow", rgb: [226, 196, 40], holds: 6, startHands: 1, startFeet: 1, finish: true, ok: true, issues: [],
                                                       route: .init(id: recognised.id, confidence: 0.9, nameZh: "合成线路 黄 1")))
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: recording.path).isEmpty)       // Nothing is left lying once it is in the album.
        XCTAssertEqual(capture.routeId, recognised.id); XCTAssertEqual(capture.routeName, "合成线路 黄 1")   // Recognised and not rejected: kept with the record, under its own name.
        XCTAssertEqual(record.grade, "V4")                                                             // The grade is the one the user picked.
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
            XCTAssertNil(record.capture?.routeId)
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
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: CaptureRecovery.directory(in: store.root).path).isEmpty)
    }

    func testRejectingAProblemMovesToTheNextCandidateAndItsGradeIsOnlyFilledInUntilTheUserPicks() async throws {
        let store = store()
        let session = session(store, try pack(), try await wall())
        session.gymId = "syn:east"                         // Known to the pack by its directory id; the name plays no part.
        session.gymName = "随便写的名字"
        session.start()
        await until("ready") { session.assessment?.ready == true }
        XCTAssertEqual(session.grade, ungradedValue); XCTAssertFalse(session.gradePrefilled)
        session.selectColor("yellow")
        await until("recognised") { session.recognised != nil }
        let first = try XCTUnwrap(session.recognised)
        XCTAssertTrue(first.id.hasSuffix("/yellow-01")); XCTAssertEqual(first.nameZh, "合成线路 黄 1"); XCTAssertFalse(first.title.isEmpty)
        // The user has not picked a grade: the problem's is filled in, and marked as filled in.
        await until("grade filled in") { session.gradePrefilled }
        XCTAssertEqual(session.grade, "V3")

        // “Not this one”: the page moves to its next candidate. Its grade is not one the album uses, so nothing is filled in.
        session.rejectRoute()
        XCTAssertNil(session.recognised)                    // At once, before the page has answered.
        XCTAssertEqual(session.grade, ungradedValue); XCTAssertFalse(session.gradePrefilled)
        await until("the next candidate") { session.recognised?.id.hasSuffix("/yellow-02") == true }
        XCTAssertNotNil(session.recognised?.grade); XCTAssertNil(session.recognised?.albumGrade)
        XCTAssertEqual(session.grade, ungradedValue); XCTAssertFalse(session.gradePrefilled)
        // The user's own pick is never replaced, whatever is recognised afterwards.
        session.pickGrade("V6")
        session.rejectRoute()
        await until("no candidate left") { session.assessment?.route == nil && session.assessment?.hint?.code == "start_unclear" }
        XCTAssertNil(session.recognised); XCTAssertEqual(session.grade, "V6"); XCTAssertFalse(session.gradePrefilled)
        var state = try await pageState(session)
        XCTAssertEqual((state["rejected"] as? [String])?.count, 2)

        session.startRecording()
        try await Task.sleep(for: .seconds(1.3))
        await session.stop(.user)
        let record = try XCTUnwrap(store.climbs.first)
        XCTAssertNil(record.capture?.routeId); XCTAssertNil(record.capture?.summary?.route); XCTAssertEqual(record.grade, "V6")

        // A new take starts clean on both sides, and the first problem is offered again; the picked grade stays.
        session.frameAgain()
        XCTAssertTrue(session.rejectedRoutes.isEmpty)
        await until("recognised again") { session.recognised?.id.hasSuffix("/yellow-01") == true }
        state = try await pageState(session)
        XCTAssertEqual((state["rejected"] as? [String])?.count, 0)
        XCTAssertEqual(session.grade, "V6"); XCTAssertFalse(session.gradePrefilled)
        await session.close()
    }

    /// Packs before 1.9.1 have no `rejectRoute`: the problem keeps coming, and is simply not shown or stored.
    func testRejectingAProblemOnAPageWithoutRejectRoute() async throws {
        let store = store()
        let session = session(store, try pack(directoryIds: false), try await wall())
        session.gymId = "syn:east"
        session.gymName = StandInPack.directoryGymName          // Such a pack does not list directory ids either: the name decides.
        session.start()
        await until("ready") { session.assessment?.ready == true }
        let view = try XCTUnwrap(session.webView)
        _ = try await view.evaluateJavaScript("delete window.BetaBuddyCapture.rejectRoute; true")
        session.selectColor("yellow")
        await until("grade filled in") { session.gradePrefilled && session.grade == "V3" }
        session.rejectRoute()
        XCTAssertNil(session.recognised); XCTAssertFalse(session.gradePrefilled); XCTAssertEqual(session.grade, ungradedValue)
        try await Task.sleep(for: .milliseconds(400))
        XCTAssertNotNil(session.assessment?.route); XCTAssertNil(session.recognised)       // Still reported, still not shown.
        XCTAssertEqual(session.grade, ungradedValue)
        session.startRecording()
        try await Task.sleep(for: .seconds(1.3))
        await session.stop(.user)
        let record = try XCTUnwrap(store.climbs.first)
        XCTAssertNil(record.capture?.routeId); XCTAssertNil(record.capture?.routeName)
        XCTAssertNotNil(record.capture?.summary?.route)                 // What the page said stays in its summary.
        await session.close()
    }

    func testAPageThatGoesQuietDoesNotLeaveItsLastJudgementOnTheScreen() async throws {
        let store = store()
        let session = session(store, try pack(), try await wall())
        session.quietLimit = 0.6
        session.start()
        await until("ready") { session.assessment?.ready == true }
        // The page stops answering, as after a script error; frames keep going in.
        let view = try XCTUnwrap(session.webView)
        _ = try await view.evaluateJavaScript("window.BetaBuddyCapture.addFrame = function () {}; true")
        await until("the screen to stop showing a judgement", within: 5) { session.page == .failed && session.assessment == nil }
        XCTAssertFalse(session.needsConfirmation)
        // Recording works all the same, and the record simply has no plan… unless the page still answers `lock`.
        session.startRecording()
        XCTAssertEqual(session.phase, .recording)
        try await Task.sleep(for: .seconds(1.3))
        await session.stop(.user)
        guard case .saved = session.phase else { return XCTFail("\(session.phase)") }
        await session.close()
    }

    func testAFailedSaveKeepsTheClipForAnotherTryAndLeavingDoesNotThrowItAway() async throws {
        let store = store()
        var session = session(store, nil, try await wall())
        session.start()
        await until("camera") { session.camera == .running }
        session.pickGrade("V2")
        session.gymName = "手填的岩馆"
        session.startRecording()
        try await Task.sleep(for: .seconds(1.3))
        // The album cannot be written.
        let index = store.root.appendingPathComponent("archive.json")
        try FileManager.default.createDirectory(at: index, withIntermediateDirectories: true)
        try Data().write(to: index.appendingPathComponent("in-the-way"))
        await session.stop(.user)
        guard case .saveFailed = session.phase else { return XCTFail("\(session.phase)") }
        XCTAssertTrue(store.climbs.isEmpty)
        // Leaving the screen now keeps the clip where recordings lie…
        await session.close()
        let recording = CaptureRecovery.directory(in: store.root)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: recording.path).filter { $0.hasSuffix(".mov") }.count, 1)
        // …and the next launch takes it into the album with what was chosen.
        try FileManager.default.removeItem(at: index)
        let swept = await CaptureRecovery.sweep(store: store, analysisAvailable: false)
        XCTAssertEqual(swept.adopted.count, 1)
        let adopted = try XCTUnwrap(ClimbStore(root: store.root).climbs.first)
        XCTAssertEqual(adopted.grade, "V2"); XCTAssertEqual(adopted.gymName, "手填的岩馆"); XCTAssertNotNil(adopted.capture)
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: recording.path).isEmpty)

        // The same failure, saved again from the screen; and given up for good only when the user says so.
        session = self.session(store, nil, try await wall())
        session.start()
        await until("camera") { session.camera == .running }
        for retry in [true, false] {
            session.startRecording()
            try await Task.sleep(for: .seconds(1.3))
            try FileManager.default.moveItem(at: index, to: store.root.appendingPathComponent("archive.kept"))
            try FileManager.default.createDirectory(at: index, withIntermediateDirectories: true)
            try Data().write(to: index.appendingPathComponent("in-the-way"))
            await session.stop(.user)
            guard case .saveFailed = session.phase else { return XCTFail("\(session.phase)") }
            try FileManager.default.removeItem(at: index)
            try FileManager.default.moveItem(at: store.root.appendingPathComponent("archive.kept"), to: index)
            if retry {
                await session.save()
                guard case .saved = session.phase else { return XCTFail("\(session.phase)") }
                session.frameAgain()
            } else {
                session.discardHeldClip()
                XCTAssertEqual(session.phase, .framing)
            }
            XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: recording.path).isEmpty)
        }
        XCTAssertEqual(ClimbStore(root: store.root).climbs.count, 2)
        await session.close()
    }

    func testWhatAnEarlierRunLeftBehindIsAdoptedOrRemovedAtLaunch() async throws {
        let store = store()
        let recording = CaptureRecovery.directory(in: store.root)
        // A clip that was closed properly, with its draft.
        var clip = try CaptureRecovery.newClip(in: store.root)
        try FileManager.default.moveItem(at: try await wall(seconds: 2), to: clip.movie)
        let began = Date(timeIntervalSince1970: 1_700_000_000)
        let info = CaptureInfo(colorKey: "blue", colorRGB: [1, 2, 3], summary: CaptureSummary(color: "blue", holds: 5, route: .init(id: "r1", name: "Localized", nameZh: "线", grade: "V2")), routeId: "r1", routeName: "线")
        CaptureRecovery.write(CaptureDraft(started: began, gymId: "syn:east", gymName: "合成岩馆（东门）", grade: "V7", info: info, plan: "{\"v\":1}"), for: clip.movie)
        // A clip whose writer never got to finish: the app was killed while recording.
        clip = try CaptureRecovery.newClip(in: store.root)
        let source = FileCaptureSource(url: try await wall(seconds: 2))
        var recorder: ClipRecorder? = try ClipRecorder(url: clip.movie, width: 360, height: 640, settings: ClipRecorder.defaultSettings(width: 360, height: 640), fragmentSeconds: 0.5)
        let unfinished = clip.movie
        source.onFrame = { [recorder] buffer, time in recorder?.append(buffer, at: time) }
        try await source.start()
        try await Task.sleep(for: .seconds(3))
        source.stop()
        source.onFrame = nil
        recorder = nil                                         // Never finished, never cancelled.
        try await Task.sleep(for: .milliseconds(300))
        XCTAssertTrue(FileManager.default.fileExists(atPath: unfinished.path))
        // Things that are not clips: a file that is not a movie, a clip too short to keep, a draft without a clip, and a leftover of older builds.
        try Data("not a movie".utf8).write(to: recording.appendingPathComponent("garbage.mov"))
        try FileManager.default.moveItem(at: try await wall(seconds: 0.4), to: recording.appendingPathComponent("short.mov"))
        try Data("{}".utf8).write(to: recording.appendingPathComponent("orphan.json"))
        let legacy = FileManager.default.temporaryDirectory.appendingPathComponent("betabuddy-capture-\(UUID()).mov")
        try Data([0]).write(to: legacy)
        let leftovers = try FileManager.default.contentsOfDirectory(atPath: NSTemporaryDirectory()).filter { $0.hasPrefix("betabuddy-capture-") }.count

        let swept = await CaptureRecovery.sweep(store: store, analysisAvailable: true)
        XCTAssertEqual(swept.adopted.count, 2, "The finished clip and the unfinished one")
        XCTAssertEqual(swept.removed, 3 + leftovers)
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: recording.path).isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: legacy.path))
        let reopened = ClimbStore(root: store.root)
        XCTAssertEqual(reopened.climbs.count, 2)
        let withDraft = try XCTUnwrap(reopened.climbs.first { $0.grade == "V7" })
        XCTAssertEqual(withDraft.gymId, "syn:east"); XCTAssertEqual(withDraft.climbedAt, began)
        XCTAssertEqual(withDraft.capture?.colorKey, "blue"); XCTAssertEqual(withDraft.capture?.routeId, "r1"); XCTAssertEqual(withDraft.capture?.analysisQueued, true)
        XCTAssertEqual(withDraft.capture?.summary?.route, .init(id: "r1", nameZh: "线"))            // Stored without the localized name and grade.
        XCTAssertEqual(reopened.capturePlan(for: withDraft), "{\"v\":1}")
        XCTAssertEqual(withDraft.pendingAnalysis, AnalysisQuality.balanced.rawValue)
        XCTAssertGreaterThan(withDraft.analysis.duration, 1.5)
        // The unfinished clip plays up to its last complete fragment; there was no draft, so it is a bare record.
        let bare = try XCTUnwrap(reopened.climbs.first { $0.id != withDraft.id })
        XCTAssertGreaterThanOrEqual(bare.analysis.duration, 1); XCTAssertLessThan(bare.analysis.duration, 3.5)
        XCTAssertEqual(bare.grade, ungradedValue); XCTAssertEqual(bare.capture, CaptureInfo(analysisQueued: true))
        XCTAssertNotNil(UIImage(contentsOfFile: store.file(bare.thumbnailName).path))
        XCTAssertEqual(store.analysisMessages[bare.id], CaptureText.stop(.interrupted))
        // A second sweep finds nothing; and none is made while the camera is in use.
        let again = await CaptureRecovery.sweep(store: store, analysisAvailable: true)
        XCTAssertTrue(again.adopted.isEmpty); XCTAssertEqual(again.removed, 0)
        store.pauseAnalysisForCamera()
        try Data("x".utf8).write(to: recording.appendingPathComponent("busy.mov"))
        let busy = await CaptureRecovery.sweep(store: store, analysisAvailable: true)
        XCTAssertEqual(busy.removed, 0); XCTAssertTrue(FileManager.default.fileExists(atPath: recording.appendingPathComponent("busy.mov").path))
    }

    func testAChangedGymIsNotLockedUnderTheOldGymsJudgement() async throws {
        let store = store()
        let session = session(store, try pack(), try await wall())
        session.start()
        await until("ready") { session.assessment?.ready == true }
        session.selectColor("yellow")
        await until("holds") { session.assessment?.holds.count == 6 }
        // The gym changes and Record is pressed before the page has been told: what it has is a judgement for no gym at all.
        session.gymId = "syn:east"; session.gymName = StandInPack.directoryGymName
        session.startRecording()
        try await Task.sleep(for: .seconds(1.3))
        let state = try await pageState(session)
        XCTAssertEqual(state["locked"] as? Bool, false)
        XCTAssertNil(session.summary)
        await session.stop(.user)
        let record = try XCTUnwrap(store.climbs.first)
        XCTAssertEqual(record.gymId, "syn:east")
        XCTAssertEqual(record.capture?.hasPlan, false); XCTAssertNil(store.capturePlan(for: record)); XCTAssertNil(record.capture?.summary)
        XCTAssertEqual(record.capture?.colorKey, "yellow")                    // What the user chose is still kept.
        await session.close()
    }

    func testAStalledCameraIsNotRecordedFromAndAPauseDoesNotLookLikeABrokenPage() async throws {
        let store = store()
        let source = FileCaptureSource(url: try await wall())
        let session = CaptureSession(store: store, pack: try pack(), source: source)
        session.framingInterval = 0.04
        session.quietLimit = 0.6
        session.stallLimit = 0.5
        var cameras: [CaptureSession.Camera] = [], pages: [CaptureSession.Page] = []
        session.$camera.sink { if cameras.last != $0 { cameras.append($0) } }.store(in: &watching)
        session.$page.sink { if pages.last != $0 { pages.append($0) } }.store(in: &watching)
        session.start()
        await until("ready") { session.assessment?.ready == true }
        // The pictures stop without a word, as after a camera error. The preview is frozen: nothing to record from…
        XCTAssertEqual(cameras, [.starting, .running])
        source.stop()
        // …the camera is no longer taken to be running (Record needs it to be), and the source is started again by itself.
        await until("the stall to be noticed and frames to come back", within: 6) { cameras.count >= 4 && session.camera == .running }
        XCTAssertEqual(cameras, [.starting, .running, .starting, .running])
        // The pause was longer than the page is allowed to stay quiet, yet the page is not taken to be broken.
        try await Task.sleep(for: .seconds(1))
        XCTAssertFalse(pages.contains(.failed), "\(pages)")
        XCTAssertNotNil(session.assessment)
        await session.close()
    }

    func testAPageThatNeverComesUpIsGivenUpOn() async throws {
        let store = store()
        let session = session(store, try pack(captureComesUp: false), try await wall())
        session.loadingLimit = .milliseconds(600)
        session.start()
        await until("camera") { session.camera == .running }
        await until("the wait to end", within: 5) { session.page == .failed }
        let handed = session.metrics.frames
        XCTAssertEqual(handed, 0)                                  // Nothing was handed to a page that never said it was ready…
        session.startRecording()                                   // …and recording works as without one.
        try await Task.sleep(for: .seconds(1.3))
        await session.stop(.user)
        guard case .saved = session.phase else { return XCTFail("\(session.phase)") }
        XCTAssertEqual(store.climbs.first?.capture?.hasPlan, false)
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

    /// The installed pack's own capture page, fed a real climbing clip in place of the camera. Neither is
    /// in this repository, so this is skipped unless both are there:
    ///
    ///     TEST_RUNNER_BB_CAPTURE_CLIP=<clip> xcodebuild … test -only-testing:BetaBuddyTests/CaptureFlowTests/testInstalledCapturePageAnswersLocksAndItsPlanIsKept
    ///
    /// Here the web view is in no window at all, the least favourable case for a page that must keep answering.
    func testInstalledCapturePageAnswersLocksAndItsPlanIsKept() async throws {
        guard let clip = ProcessInfo.processInfo.environment["BB_CAPTURE_CLIP"], FileManager.default.fileExists(atPath: clip) else {
            throw XCTSkip("Optional private climbing clip is not supplied")
        }
        guard let pack = GymPack.current, pack.supportsCapture else { throw XCTSkip("The installed build has no pack with a capture page") }
        let store = store()
        let session = CaptureSession(store: store, pack: pack, source: FileCaptureSource(url: URL(fileURLWithPath: clip), horizontalFieldOfView: 50))
        // `TEST_RUNNER_BB_CAPTURE_GYM=<a gym's name as the directory writes it>` selects a gym, for a clip shot in one of the pack's own.
        if let gym = ProcessInfo.processInfo.environment["BB_CAPTURE_GYM"], !gym.isEmpty { session.gymName = gym }
        // …or `TEST_RUNNER_BB_CAPTURE_GYM_ID=<its id in the directory>`, which a pack from 1.9.1 on maps to its venue.
        if let id = ProcessInfo.processInfo.environment["BB_CAPTURE_GYM_ID"], !id.isEmpty { session.gymId = id }
        var codes: [String] = []
        session.$assessment.sink { value in
            guard let value else { return }
            let code = (value.ready ? "ready" : value.ok ? "settling" : "blocked") + (value.hint.map { ":" + $0.code + ($0.move.map { "/" + $0 } ?? "") } ?? "")
            if codes.last != code { codes.append(code) }
        }.store(in: &watching)
        session.start()
        await until("the page to come up", within: 30) { session.page == .ready }
        await until("twenty replies", within: 60) { session.metrics.assessments >= 20 }
        let framing = session.metrics
        XCTAssertGreaterThan(framing.assessmentsPerSecond, 1.5, framing.summary)          // Two frames a second go in; it keeps up.
        let seen = try XCTUnwrap(session.assessment)
        var report = ["framing: " + framing.summary, "states: " + codes.joined(separator: " → "),
                      "palette: " + seen.palette.map { "\($0.key)×\($0.count)" }.joined(separator: " "), "person: " + seen.person]
        if let colour = seen.palette.first?.key {
            session.selectColor(colour)
            await until("holds of the first colour", within: 15) { session.assessment?.color == colour && session.assessment?.holds.isEmpty == false }
            let chosen = try XCTUnwrap(session.assessment)
            // A page that recognises problems of the pack's own venues needs a moment for it.
            if pack.manifest.venueID(gymID: session.gymId, name: session.gymName) != nil {
                let deadline = Date().addingTimeInterval(12)
                while session.recognised == nil, Date() < deadline { try await Task.sleep(for: .milliseconds(100)) }
                report.append("venue given; recognised: \(session.recognised.map { "\($0.id) conf \($0.confidence ?? -1) name \($0.name ?? "none") nameZh \($0.nameZh ?? "none") grade \($0.grade ?? "none")" } ?? "nothing"); grade shown \(session.grade), filled in \(session.gradePrefilled)")
            }
            report.append("first colour: \(chosen.holds.count) holds, \(chosen.holds.filter { !$0.poly.isEmpty }.count) with outlines, start hands \(chosen.startHands.count), feet \(chosen.startFeet.count), finish \(chosen.finish != nil), hint \(chosen.hint?.code ?? "none")")
            // Mark the first hold as the start, the way a tap on the picture does.
            let hold = chosen.holds[0]
            session.correct(at: CGPoint(x: hold.box.midX, y: hold.box.midY), role: .startHand)
            await until("the tap to show", within: 10) { session.assessment?.startConfirmed == true }
            XCTAssertFalse(session.tapMissed)
        }
        let pressed = Date()
        session.startRecording()
        await until("the plan", within: 10) { session.summary != nil }
        report.append(String(format: "lock answered after %.0f ms", Date().timeIntervalSince(pressed) * 1000))
        try await Task.sleep(for: .seconds(2))
        await session.stop(.user)
        guard case .saved(let id) = session.phase else { return XCTFail("not saved: \(session.phase)") }
        let record = try XCTUnwrap(store.climbs.first { $0.id == id })
        let plan = try XCTUnwrap(store.capturePlan(for: record))
        report.append("plan: \(plan.utf8.count) bytes; summary: \(String(describing: record.capture?.summary))")
        XCTAssertGreaterThan(plan.utf8.count, 1_000)
        XCTAssertEqual(record.capture?.packVersion, pack.manifest.packVersion)
        let text = XCTAttachment(string: report.joined(separator: "\n")); text.name = "Installed capture page with a local clip"; text.lifetime = .keepAlways; add(text)
        NSLog("CAPTURE-REAL %@", report.joined(separator: " | "))
        await session.close()
    }

    /// Not an assertion of speed: the simulator runs on the Mac's processor. It keeps the numbers
    /// with the test results so they can be compared from run to run.
    func testCostOfHandingOverFullSizeFrames() async throws {
        let store = store()
        let session = CaptureSession(store: store, pack: try pack(), source: FileCaptureSource(url: try await wall(seconds: 3, width: 1080, height: 1920, photo: try person())))
        session.start()
        await until("thirty frames at two a second", within: 40) { session.metrics.frames >= 30 }
        let metrics = session.metrics
        XCTAssertEqual(session.frameSize, CGSize(width: 360, height: 640))          // 1080 × 1920 handed over with its short side at 360.
        XCTAssertLessThan(metrics.jpegBytes / metrics.frames, 120_000)
        let print = XCTAttachment(string: metrics.summary); print.name = "Frame hand-over with live pose, 1080x1920 source at 3 fps"; print.lifetime = .keepAlways; add(print)
        NSLog("CAPTURE-COST %@", metrics.summary)
        await session.close()
    }
}
