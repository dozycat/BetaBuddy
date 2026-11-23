#if DEBUG
import Foundation
import UIKit

/// Debug builds only, with `-debugCaptureAutopilot YES` on the launch command line: goes through the
/// capture screen the way a person would — gym, colour, start and finish, record, stop — when the
/// phone is somewhere nobody can tap it. It presses nothing of its own: every step calls what the
/// screen's own control calls. Each step is an `autopilot` line in the diagnostics, and when the
/// record has been analysed (or something stopped it) `autopilot-done.json` says how it went.
///
/// - `-debugCaptureGym <directory id>`: select that gym first.
/// - `-debugCaptureColour <palette key>`: the colour to choose; without it, the one with the most holds.
/// - `-debugCaptureTaps YES`: mark start and finish even when the page already guesses a start.
/// - `-debugCaptureSeconds <n>`: how long to record (20).
@MainActor
final class CaptureAutopilot {
    /// What the screen's controls do, handed over by the screen itself.
    struct Screen {
        var chooseGym: (DirectoryGym) -> Void
        var pickColour: (String) -> Void
        var chooseTool: (CaptureRole) -> Void
        /// A tap on the picture, in its own 0–1 coordinates.
        var tapPicture: (CGPoint) -> Void
        var pressRecord: () -> Void
        /// The “record anyway?” question is up.
        var asksToConfirm: () -> Bool
        var recordAnyway: () -> Void
        var pressStop: () -> Void
        /// Keeps a picture of the screen under this name.
        var shot: (String) -> Void
    }
    struct Halt: Error {
        var stage: String
        var reason: String
    }

    static let doneFile = "autopilot-done.json"
    static let shots = ["colour", "marked", "recording", "saved", "analysed"].map { "shot-\($0).png" }
    static var wanted: Bool { UserDefaults.standard.bool(forKey: "debugCaptureAutopilot") }
    private static var started = false

    private let session: CaptureSession
    private let store: ClimbStore
    private let screen: Screen
    private var seen: [(name: String, fields: [String: Any])] = []
    private var report: [String: Any] = [:]
    private var stages: [String: Double] = [:]
    private var stage = "start"
    private var stageBegan = Date()
    private var peak = ProcessInfo.ThermalState.nominal
    private var kept: [String] = []

    init(session: CaptureSession, store: ClimbStore, screen: Screen) {
        self.session = session; self.store = store; self.screen = screen
    }

    // MARK: Choices, kept apart so that they can be tried with made-up data

    /// The colour asked for, or the one with the most holds; `nil` when the one asked for is not there.
    nonisolated static func colour(in palette: [CaptureAssessment.Swatch], wanted: String?) -> String? {
        if let wanted, !wanted.isEmpty { return palette.contains { $0.key == wanted } ? wanted : nil }
        // The first of equals, which is the order the page gave them in.
        return palette.enumerated().max { ($0.element.count, -$0.offset) < ($1.element.count, -$1.offset) }?.element.key
    }

    /// The lowest hold with an outline as the start and the highest as the finish. One hold alone is only a start.
    nonisolated static func marks(in holds: [CaptureAssessment.Hold]) -> (start: CaptureAssessment.Hold, finish: CaptureAssessment.Hold?)? {
        let outlined = holds.filter { $0.box.width > 0 && $0.box.height > 0 }
        guard let start = outlined.max(by: { $0.box.midY < $1.box.midY }) else { return nil }
        let finish = outlined.min { $0.box.midY < $1.box.midY }
        return (start, finish?.id == start.id ? nil : finish)
    }

    // MARK: The run

    func run() async {
        guard Self.wanted, !Self.started else { return }
        Self.started = true
        CaptureDiagnostics.observer = { [weak self] name, fields in self?.seen.append((name, fields)) }
        session.diagnostics?.remove([Self.doneFile] + Self.shots)
        let defaults = UserDefaults.standard
        step("start", ["gym": defaults.string(forKey: "debugCaptureGym") as Any, "colour": defaults.string(forKey: "debugCaptureColour") as Any,
                       "taps": defaults.bool(forKey: "debugCaptureTaps"), "seconds": seconds])
        do {
            try await ready()
            try await colour()
            await marks()
            try await record()
            try await analysed()
            finish(nil)
        } catch let halt as Halt {
            finish(halt)
        } catch {
            finish(Halt(stage: stage, reason: "\(error)"))
        }
    }

    private var seconds: Double {
        let asked = UserDefaults.standard.double(forKey: "debugCaptureSeconds")
        return asked > 0 ? asked : 20
    }

    private func step(_ name: String, _ fields: [String: Any] = [:]) {
        session.autopilotNote = name
        CaptureDiagnostics.event("autopilot", fields.merging(["step": name]) { _, new in new })
    }

    /// The stage that was running is over and took this long; `name` begins.
    private func enter(_ name: String) {
        stages[stage] = (stages[stage] ?? 0) + Date().timeIntervalSince(stageBegan)
        stage = name
        stageBegan = Date()
    }

    private func latest(_ name: String, where matches: ([String: Any]) -> Bool = { _ in true }) -> [String: Any]? {
        seen.last { $0.name == name && matches($0.fields) }?.fields
    }

    /// `true` as soon as the condition holds, `false` when it has not within `seconds`.
    private func wait(_ seconds: Double, for condition: () -> Bool) async -> Bool {
        let end = Date().addingTimeInterval(seconds)
        while true {
            let now = ProcessInfo.processInfo.thermalState
            if now.rawValue > peak.rawValue { peak = now }
            if condition() { return true }
            if Date() >= end || Task.isCancelled { return false }
            try? await Task.sleep(for: .milliseconds(200))
        }
    }
    private func pause(_ seconds: Double) async { _ = await wait(seconds) { false } }

    private func shot(_ name: String) async {
        await pause(0.6)        // What was just done has to be on the screen first.
        screen.shot("shot-\(name).png")
        kept.append("shot-\(name).png")
    }

    /// Step 1 and the wait of step 2: the gym, then a page that answers with colours to choose from.
    private func ready() async throws {
        enter("ready")
        var afterGym: CaptureAssessment?
        var gymWanted: String?
        if let id = UserDefaults.standard.string(forKey: "debugCaptureGym"), !id.isEmpty {
            if let gym = GymDirectory.bundled?.gyms.first(where: { $0.id == id }) {
                // What the page said before the gym changed is about another place; it is begun again for this one.
                if session.gymId != id { afterGym = session.assessment; gymWanted = id }
                screen.chooseGym(gym)
                step("gym", ["id": id, "alreadySelected": gymWanted == nil])
            } else {
                step("gym", ["id": id, "error": "not in the directory"])
            }
        }
        let came = await wait(30) {
            guard session.camera == .running, session.page == .ready, let assessment = session.assessment, !assessment.palette.isEmpty else { return false }
            guard let gymWanted else { return true }
            return latest("begin")?["gym"] as? String == gymWanted && assessment != afterGym
        }
        guard came else {
            throw Halt(stage: "ready", reason: "no colours to choose from within 30 s: camera \(session.camera), page \(session.page), "
                       + (session.assessment == nil ? "no assessment" : "palette of \(session.assessment?.palette.count ?? 0)"))
        }
        let state = session.sampleFields()
        for key in ["gym", "venue", "pack", "fovH", "source", "frame"] { report[key] = state[key] ?? NSNull() }
        report["camera"] = "\(type(of: session.source))"
        step("ready", ["palette": session.palette.count])
    }

    /// The rest of step 2, and the wait of step 3.
    private func colour() async throws {
        enter("colour")
        let palette = session.palette
        let wanted = UserDefaults.standard.string(forKey: "debugCaptureColour")
        let described = palette.map { ["key": $0.key, "rgb": $0.rgb, "n": $0.count] as [String: Any] }
        guard let key = Self.colour(in: palette, wanted: wanted) else {
            throw Halt(stage: "colour", reason: "no colour \(wanted ?? "") among \(palette.map(\.key).joined(separator: ", "))")
        }
        screen.pickColour(key)
        let from = wanted?.isEmpty == false ? "argument" : "largest"
        step("colour", ["key": key, "from": from, "palette": described])
        report["colour"] = key
        report["colourFrom"] = from
        let came = await wait(15) {
            guard session.color == key, let assessment = session.assessment, !assessment.holds.isEmpty else { return false }
            return assessment.color == nil || assessment.color == key
        }
        let assessment = session.assessment
        step("holds", ["came": came, "holds": assessment?.holds.count ?? 0, "startHands": assessment?.startHands ?? [], "startFeet": assessment?.startFeet ?? [],
                       "finish": assessment?.finish as Any])
        report["holds"] = assessment?.holds.count ?? 0
        await shot("colour")
    }

    /// Step 3: the route if the page names one, and the start and finish when the page has none.
    private func marks() async {
        enter("marks")
        var taps: [[String: Any]] = []
        defer { report["taps"] = taps }
        guard let assessment = session.assessment else { return }
        if let route = assessment.route {
            step("route", ["id": route.id, "conf": route.confidence as Any])
            report["route"] = ["id": route.id, "conf": route.confidence as Any]
        }
        let force = UserDefaults.standard.bool(forKey: "debugCaptureTaps")
        if assessment.startHands.isEmpty || force, let marks = Self.marks(in: assessment.holds) {
            taps.append(await tap(marks.start, as: .startHand))
            _ = await wait(3) { session.assessment?.startConfirmed == true }
            if let finish = marks.finish {
                taps.append(await tap(finish, as: .finish))
                _ = await wait(3) { session.assessment?.finishConfirmed == true }
            }
        } else {
            step("marks", ["tapped": false, "why": assessment.startHands.isEmpty ? "no hold with an outline" : "the page already has a start"])
        }
        let now = session.assessment
        report["marked"] = ["startHands": now?.startHands ?? [], "startFeet": now?.startFeet ?? [], "finish": now?.finish as Any,
                            "startTapped": now?.startConfirmed ?? false, "finishTapped": now?.finishConfirmed ?? false] as [String: Any]
        await shot("marked")
    }

    private func tap(_ hold: CaptureAssessment.Hold, as role: CaptureRole) async -> [String: Any] {
        screen.chooseTool(role)
        await pause(0.2)
        let before = session.lastTap.serial
        let point = CGPoint(x: hold.box.midX, y: hold.box.midY)
        screen.tapPicture(point)
        let answered = await wait(5) { session.lastTap.serial > before }
        let fields: [String: Any] = ["role": role.rawValue, "x": Double(point.x), "y": Double(point.y), "wanted": hold.id,
                                     "answered": answered, "hold": (answered ? session.lastTap.hold : nil) as Any,
                                     "pageHasTap": answered ? session.lastTap.pageHasTap : false]
        step("tap", fields)
        return fields
    }

    /// Step 4.
    private func record() async throws {
        enter("record")
        screen.pressRecord()
        await pause(0.4)
        var anyway = false
        if screen.asksToConfirm() {
            anyway = true
            step("recordAnyway", ["issues": session.assessment?.issues.filter(\.blocking).map(\.code) ?? []])
            await pause(0.8)
            screen.recordAnyway()
        }
        report["recordAnyway"] = anyway
        guard await wait(10, for: { session.phase == .recording }) else {
            throw Halt(stage: "record", reason: "recording did not start: phase \(session.phase), camera \(session.camera), notice \(session.notice.map { "\($0)" } ?? "none")")
        }
        step("recording", ["seconds": seconds])
        await pause(min(2, seconds / 2))
        screen.shot("shot-recording.png"); kept.append("shot-recording.png")
        // Something else may end it first: an interruption, the length limit, the disk.
        if await wait(max(0, seconds - min(2, seconds / 2)), for: { session.phase != .recording }) {
            step("stopped", ["by": session.stopReason?.rawValue as Any])
        } else {
            screen.pressStop()
            step("stop")
        }
        enter("save")
        let settled = await wait(120) {
            switch session.phase {
            case .saved, .saveFailed: true
            default: false
            }
        }
        if let stop = latest("recordStop") {
            report["clip"] = ["seconds": stop["seconds"] ?? NSNull(), "frames": stop["frames"] ?? NSNull(), "dropped": stop["dropped"] ?? NSNull(),
                              "bytes": stop["bytes"] ?? NSNull(), "reason": stop["reason"] ?? NSNull()]
        }
        if let locked = latest("captureLocked") {
            report["locked"] = ["ms": locked["ms"] ?? NSNull(), "summary": locked["summary"] ?? NSNull()]
            report["planBytes"] = locked["planBytes"] ?? NSNull()
            if report["route"] == nil, let route = (locked["summary"] as? [String: Any])?["route"] { report["route"] = route }
        }
        guard settled else { throw Halt(stage: "save", reason: "not saved within 120 s: phase \(session.phase)") }
        if case .saveFailed(let message) = session.phase { throw Halt(stage: "save", reason: message) }
        guard case .saved(let id) = session.phase else { return }
        report["record"] = id.uuidString
        step("saved", ["record": id.uuidString])
        await shot("saved")
    }

    /// Step 5: the saved sheet stays up while pose recognition and the analysis page run.
    private func analysed() async throws {
        guard let record = report["record"] as? String else { return }
        func mine(_ fields: [String: Any]) -> Bool { fields["record"] as? String == record }
        enter("pose")
        let queued = latest("save", where: mine)?["analysisQueued"] as? Bool == true
        guard await wait(900, for: { latest("pose", where: mine) != nil || latest("analysisQueue", where: mine) != nil }) else {
            throw Halt(stage: "pose", reason: "pose recognition not over within 900 s")
        }
        if var pose = latest("pose", where: mine) {
            pose["record"] = nil
            report["pose"] = pose
            step("pose", pose)
        }
        enter("analysis")
        if queued {
            guard await wait(600, for: { latest("analysisQueue", where: mine) != nil }) else {
                throw Halt(stage: "analysis", reason: "the analysis page gave nothing within 600 s")
            }
            let queue = latest("analysisQueue", where: mine) ?? [:]
            report["queue"] = ["outcome": queue["outcome"] ?? NSNull(), "why": queue["why"] ?? NSNull()]
            // The page's own last word on this record: a result, or why not.
            if let last = latest("analysis", where: { mine($0) && $0["status"] as? String != "begin" }) {
                report["analysis"] = ["status": last["status"] ?? NSNull(), "code": last["code"] ?? NSNull(), "message": last["message"] ?? NSNull(),
                                      "capture": last["capture"] ?? NSNull(), "seconds": last["seconds"] ?? NSNull(), "resultBytes": last["resultBytes"] ?? NSNull()]
                if let begun = latest("analysis", where: { mine($0) && $0["status"] as? String == "begin" }) {
                    report["analysisInput"] = ["stills": begun["stills"] ?? NSNull(), "w": begun["w"] ?? NSNull(), "h": begun["h"] ?? NSNull(),
                                               "poseFrames": begun["poseFrames"] ?? NSNull(), "planBytes": begun["planBytes"] ?? NSNull()]
                }
            }
            step("analysed", ["outcome": queue["outcome"] ?? NSNull(), "status": (report["analysis"] as? [String: Any])?["status"] ?? NSNull()])
        } else {
            report["queue"] = ["outcome": "not queued", "why": "this build has no analysis page"]
            step("analysed", ["outcome": "not queued"])
        }
        await shot("analysed")
    }

    private func finish(_ halt: Halt?) {
        enter("done")
        report["ok"] = halt == nil
        report["stage"] = halt?.stage ?? "done"
        report["reason"] = halt?.reason ?? NSNull()
        report["stages"] = stages.filter { $0.key != "start" }
        report["thermalPeak"] = CaptureDiagnostics.thermal(peak)
        report["shots"] = kept
        report["environment"] = CaptureDiagnostics.environment
        report["session"] = session.diagnostics?.file.lastPathComponent ?? NSNull()
        report["t"] = ISO8601DateFormatter().string(from: Date())
        step(halt == nil ? "done" : "failed", halt.map { ["stage": $0.stage, "reason": $0.reason] } ?? [:])
        if let data = try? JSONSerialization.data(withJSONObject: CaptureDiagnostics.clean(report), options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]) {
            (session.diagnostics ?? CaptureDiagnostics()).write(data, as: Self.doneFile)
        }
        session.diagnostics?.flush()
        CaptureDiagnostics.observer = nil
    }
}
#endif
