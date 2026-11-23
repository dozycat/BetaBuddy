import CoreGraphics
import Foundation

// The capture page of a gym pack: the app shows the camera and draws everything, a page without
// any interface of its own looks at preview frames and answers with what it thinks of the camera
// position and what it recognises. What it computes is the pack's business; this file is only the
// shape of the messages, which the app treats leniently: unknown types, codes and extra fields are
// ignored, missing ones fall back to the cautious reading.

/// What a hold is to the problem being recorded.
enum CaptureRole: String, CaseIterable, Sendable {
    case startHand = "start_hand"
    case startFoot = "start_foot"
    case finish
    case hold
    /// Only ever sent by the app: “this one is not part of the problem”.
    case none
}

/// One reply of the capture page to a preview frame. Positions are normalised to that frame:
/// `x` to the right, `y` down, both 0–1.
struct CaptureAssessment: Equatable, Sendable {
    struct Issue: Equatable, Sendable {
        var code: String
        /// `true` for `severity: "block"`: analysis will most likely not work from here.
        var blocking: Bool
        var move: String?
    }
    struct Swatch: Equatable, Sendable, Identifiable {
        var key: String
        var rgb: [Int]
        var count: Int
        var id: String { key }
    }
    struct Hold: Equatable, Sendable, Identifiable {
        var id: String
        var box: CGRect
        /// Outline to draw. Empty when the page gave none; the box is drawn instead.
        var poly: [CGPoint]
        var role: CaptureRole
        var confidence: Double?
        /// The role was set by the user with a tap.
        var user: Bool
    }
    /// A problem of the pack's own that the page thinks this is. Only a newer page sends one, and
    /// only for a gym the pack carries. A guess like everything else here.
    struct Route: Codable, Equatable, Sendable {
        var id: String
        var confidence: Double?
        /// Display name and grade text in the interface language, when the page gives them; the
        /// app has no other way to learn them. `nameZh` is the problem's own name.
        var name: String?
        var nameZh: String?
        var grade: String?

        init(id: String, confidence: Double? = nil, name: String? = nil, nameZh: String? = nil, grade: String? = nil) {
            self.id = id; self.confidence = confidence; self.name = name; self.nameZh = nameZh; self.grade = grade
        }
        /// What to call it on screen: its name, else its own name, else its id.
        var title: String { name ?? nameZh ?? id }
        /// The grade as one of the values the album uses (V0–V12), or `nil` when the page's text is
        /// anything else. Gyms write grades in their own ways; only this form is the same in every language.
        var albumGrade: String? {
            guard let grade, grade.hasPrefix("V"), let number = Int(grade.dropFirst()), (0...12).contains(number), grade == "V\(number)" else { return nil }
            return grade
        }
        init?(body: Any?) {
            guard let body = body as? [String: Any], let id = body["id"] as? String, !id.isEmpty else { return nil }
            func text(_ key: String) -> String? { (body[key] as? String).flatMap { $0.isEmpty ? nil : String($0.prefix(120)) } }
            self.id = String(id.prefix(120))
            confidence = CaptureAssessment.number(body["conf"])
            name = text("name")
            nameZh = text("nameZh")
            grade = text("grade")
        }
    }
    static let maximumHolds = 300
    static let maximumPolygonPoints = 96

    var time: Double
    var ok: Bool
    /// 0–1, only for an indicator. `nil` when the page gave none.
    var score: Double?
    var steady: Bool
    var issues: [Issue]
    /// The one thing to tell the user now.
    var hint: Issue?
    /// `"none"`, `"partial"`, `"full"` or `"unknown"`.
    var person: String
    var wall: CGRect?
    var palette: [Swatch]
    var color: String?
    var holds: [Hold]
    var startHands: [String]
    var startFeet: [String]
    var finish: String?
    var route: Route?

    /// The user has said where the problem starts, and where it ends. Until then both are the
    /// page's guesses, and the start in particular is wrong more often than right.
    var startConfirmed: Bool { holds.contains { $0.role == .startHand && $0.user } }
    var finishConfirmed: Bool { holds.contains { $0.role == .finish && $0.user } }

    /// The camera position can be used and the picture has stopped moving.
    var ready: Bool { ok && steady }

    init?(body: [String: Any]) {
        time = Self.number(body["t"]) ?? 0
        let issues = (body["issues"] as? [Any] ?? []).prefix(32).compactMap(Self.issue)
        // A page that says nothing about `ok` is read from its issues; one that says nothing about
        // `steady` has not said the picture is still.
        let ok = body["ok"] as? Bool ?? !issues.contains(where: \.blocking)
        self.issues = issues
        self.ok = ok
        steady = body["steady"] as? Bool ?? false
        score = Self.number(body["score"]).map { min(1, max(0, $0)) }
        if var hint = Self.issue(body["hint"]) {
            // The hint carries no severity of its own; it is the first issue.
            hint.blocking = issues.first(where: { $0.code == hint.code })?.blocking ?? !ok
            self.hint = hint
        } else { hint = issues.first }
        person = String((body["person"] as? String ?? "unknown").prefix(16))
        wall = (body["wall"] as? [String: Any]).flatMap { wall in
            guard let top = Self.number(wall["top"]), let bottom = Self.number(wall["bottom"]),
                  let left = Self.number(wall["left"]), let right = Self.number(wall["right"]),
                  bottom > top, right > left else { return nil }
            return CGRect(x: left, y: top, width: right - left, height: bottom - top)
        }
        palette = (body["palette"] as? [Any] ?? []).prefix(16).compactMap { item in
            guard let item = item as? [String: Any], let key = item["key"] as? String, !key.isEmpty, key.count <= 40 else { return nil }
            let rgb = (item["rgb"] as? [Any] ?? []).compactMap { Self.number($0).map { Int(min(255, max(0, $0.rounded()))) } }
            return Swatch(key: key, rgb: rgb.count == 3 ? rgb : [128, 128, 128], count: Int(Self.number(item["count"]) ?? 0))
        }
        color = (body["color"] as? String).flatMap { $0.isEmpty ? nil : String($0.prefix(40)) }
        holds = (body["holds"] as? [Any] ?? []).prefix(Self.maximumHolds).compactMap { item in
            guard let item = item as? [String: Any], let id = item["id"] as? String, !id.isEmpty, id.count <= 40 else { return nil }
            let poly = (item["poly"] as? [Any] ?? []).prefix(Self.maximumPolygonPoints).compactMap { point -> CGPoint? in
                guard let pair = point as? [Any], pair.count >= 2, let x = Self.number(pair[0]), let y = Self.number(pair[1]) else { return nil }
                return CGPoint(x: x, y: y)
            }
            let numbers = (item["box"] as? [Any] ?? []).compactMap(Self.number)
            let box: CGRect
            if numbers.count == 4, numbers[2] >= 0, numbers[3] >= 0 {
                box = CGRect(x: numbers[0], y: numbers[1], width: numbers[2], height: numbers[3])
            } else if poly.count >= 3 {
                let xs = poly.map(\.x), ys = poly.map(\.y)
                box = CGRect(x: xs.min()!, y: ys.min()!, width: xs.max()! - xs.min()!, height: ys.max()! - ys.min()!)
            } else { return nil }   // Nothing to draw.
            // A role this build has never heard of is still a hold of the problem.
            let role = (item["role"] as? String).flatMap(CaptureRole.init(rawValue:)) ?? .hold
            return Hold(id: id, box: box, poly: poly.count >= 3 ? poly : [], role: role == .none ? .hold : role,
                        confidence: Self.number(item["conf"]), user: item["user"] as? Bool ?? false)
        }
        let start = body["start"] as? [String: Any]
        startHands = Self.ids(start?["hands"])
        startFeet = Self.ids(start?["feet"])
        finish = ((body["finish"] as? [String: Any])?["id"] as? String).flatMap { $0.isEmpty ? nil : String($0.prefix(40)) }
        route = Route(body: body["route"])
    }

    private static func ids(_ value: Any?) -> [String] {
        (value as? [Any] ?? []).prefix(8).compactMap { ($0 as? String).flatMap { $0.isEmpty ? nil : String($0.prefix(40)) } }
    }
    /// A finite JSON number. A JSON boolean is not one.
    fileprivate static func number(_ value: Any?) -> Double? {
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(), number.doubleValue.isFinite else { return nil }
        return number.doubleValue
    }
    fileprivate static func issue(_ value: Any?) -> Issue? {
        guard let item = value as? [String: Any], let code = item["code"] as? String, !code.isEmpty else { return nil }
        let move = (item["move"] as? String).flatMap { $0.isEmpty ? nil : String($0.prefix(40)) }
        // Anything that is not plainly a warning is taken as blocking: the user is then asked before recording.
        return Issue(code: String(code.prefix(40)), blocking: (item["severity"] as? String ?? "block") != "warn", move: move)
    }
}

/// What the page had decided when recording started. Kept with the record for display; the plan
/// itself is the page's own format and is stored next to the clip untouched.
struct CaptureSummary: Codable, Equatable, Sendable {
    var color: String?
    var rgb: [Int]?
    var holds: Int = 0
    var startHands: Int = 0
    var startFeet: Int = 0
    var finish: Bool = false
    var ok: Bool = false
    var issues: [String] = []
    /// The pack's own problem the page took this for, if it said so.
    var route: CaptureAssessment.Route?

    init(color: String? = nil, rgb: [Int]? = nil, holds: Int = 0, startHands: Int = 0, startFeet: Int = 0, finish: Bool = false, ok: Bool = false, issues: [String] = [],
         route: CaptureAssessment.Route? = nil) {
        self.color = color; self.rgb = rgb; self.holds = holds; self.startHands = startHands; self.startFeet = startFeet
        self.finish = finish; self.ok = ok; self.issues = issues; self.route = route
    }
    /// As it is written into the album: the recognised problem keeps its id and its own name. The
    /// name and grade the page gave in the interface language are not stored.
    var stored: CaptureSummary {
        var copy = self
        copy.route = route.map { CaptureAssessment.Route(id: $0.id, confidence: $0.confidence, nameZh: $0.nameZh) }
        return copy
    }
    init(body: [String: Any]) {
        color = (body["color"] as? String).flatMap { $0.isEmpty ? nil : String($0.prefix(40)) }
        let values = (body["rgb"] as? [Any] ?? []).compactMap { CaptureAssessment.number($0).map { Int(min(255, max(0, $0.rounded()))) } }
        rgb = values.count == 3 ? values : nil
        holds = max(0, Int(CaptureAssessment.number(body["holds"]) ?? 0))
        let start = body["start"] as? [String: Any]
        startHands = max(0, Int(CaptureAssessment.number(start?["hands"]) ?? 0))
        startFeet = max(0, Int(CaptureAssessment.number(start?["feet"]) ?? 0))
        finish = body["finish"] as? Bool ?? false
        ok = body["ok"] as? Bool ?? false
        issues = (body["issues"] as? [Any] ?? []).prefix(32).compactMap { ($0 as? String).map { String($0.prefix(40)) } }
        route = CaptureAssessment.Route(body: body["route"])
    }
}

/// Whether the analysis page could use the capture plan, as it says with its result. A plan is not
/// used when the camera was moved between framing and recording; the analysis is then as without one.
struct CapturePlanUse: Codable, Equatable, Sendable {
    var used: Bool
    /// Why not: `mismatch`, `unreadable`, or something a newer page says.
    var reason: String?
    /// Whose start and finish the result went by: `user`, `venue`, `plan` or `video`.
    var start: String?
    var finish: String?
    /// How far the picture had moved between framing and the clip when the plan was still accepted, as the page measured it.
    struct Shift: Codable, Equatable, Sendable {
        var dx: Double?
        var dy: Double?
        var rot: Double?
        var scale: Double?
    }
    var shift: Shift?

    init(used: Bool, reason: String? = nil, start: String? = nil, finish: String? = nil, shift: Shift? = nil) {
        self.used = used; self.reason = reason; self.start = start; self.finish = finish; self.shift = shift
    }
    init?(body: Any?) {
        guard let body = body as? [String: Any], let used = body["used"] as? Bool else { return nil }
        func text(_ key: String) -> String? { (body[key] as? String).flatMap { $0.isEmpty ? nil : String($0.prefix(40)) } }
        self.used = used
        reason = text("reason"); start = text("start"); finish = text("finish")
        if let moved = body["shift"] as? [String: Any] {
            func number(_ key: String) -> Double? { CaptureAssessment.number(moved[key]) }
            let shift = Shift(dx: number("dx"), dy: number("dy"), rot: number("rot"), scale: number("scale"))
            if shift != Shift() { self.shift = shift }
        }
    }
}

/// What a record made with the in-app camera keeps about how it was framed. Records imported
/// from the photo library, and every record written before the camera existed, have none.
struct CaptureInfo: Codable, Equatable, Sendable {
    /// The colour the user chose on the capture screen (the pack's key for it), and how it looked.
    var colorKey: String?
    var colorRGB: [Int]?
    var summary: CaptureSummary?
    /// The page's capture plan is stored next to the clip and is handed to the analysis page.
    var hasPlan = false
    /// Version of the pack whose capture page was used; `nil` when the build had no capture page.
    var packVersion: String?
    /// The analysis asked for when the clip was saved has not produced a result yet.
    var analysisQueued: Bool?
    /// The pack's own problem the capture page recognised and the user did not reject: its id, and its own name (not a translation) when the page gave one.
    var routeId: String?
    var routeName: String?
    /// What the analysis page said about the plan when it last gave a result; `nil` before that, and from pages that do not say.
    var planUse: CapturePlanUse?
}

/// Capture page → app messages. They arrive on the same `gym` handler, from the capture page's own web view.
enum CaptureMessage: Equatable, Sendable {
    case ready(packVersion: String)
    case assessment(CaptureAssessment)
    /// `plan` is `nil` when the page gave none that can be kept.
    case locked(id: String, plan: String?, summary: CaptureSummary)
    /// The page cannot assess. Recording goes on without it.
    case failed(code: String)
    /// A script error. Before `ready` it means the page never came up.
    case pageError(String)

    static let maximumPlanLength = 4_000_000

    init?(body: Any) {
        guard let body = body as? [String: Any], let type = body["type"] as? String else { return nil }
        switch type {
        case "captureReady": self = .ready(packVersion: String((body["packVersion"] as? String ?? "").prefix(40)))
        case "captureAssessment":
            guard let assessment = CaptureAssessment(body: body) else { return nil }
            self = .assessment(assessment)
        case "captureLocked":
            var plan = body["plan"] as? String
            // The plan is a string; a page that hands over an object instead is still kept.
            if plan == nil, let object = body["plan"], !(object is NSNull), JSONSerialization.isValidJSONObject(object),
               let data = try? JSONSerialization.data(withJSONObject: object) { plan = String(data: data, encoding: .utf8) }
            if let text = plan, text.isEmpty || text.utf8.count > Self.maximumPlanLength { plan = nil }
            self = .locked(id: String((body["id"] as? String ?? "").prefix(80)), plan: plan,
                           summary: CaptureSummary(body: body["summary"] as? [String: Any] ?? [:]))
        case "captureError": self = .failed(code: String((body["code"] as? String ?? "").prefix(80)))
        case "error": self = .pageError(String((body["message"] as? String ?? "").prefix(300)))
        default: return nil
        }
    }
}

/// What the app knows about a preview frame besides the picture. Every part is optional.
struct CaptureFrameInfo: Equatable, Sendable {
    /// Elevation of the camera's optical axis above the horizon, degrees, up positive.
    var pitch: Double?
    /// Rotation of the phone about the optical axis, degrees, clockwise positive as the user
    /// looks at the screen; upright portrait is 0.
    var roll: Double?
    /// Root mean square of the angular velocity since the previous frame, rad/s.
    var shake: Double?
    /// 33 × `[x, y, visibility]` when a person was detected in this frame.
    var pose: [[Double]]?
    var isEmpty: Bool { pitch == nil && roll == nil && shake == nil && pose == nil }
}

/// App → capture page calls on `window.BetaBuddyCapture`.
enum CaptureCommand: Sendable {
    case begin(id: String, width: Int, height: Int, fieldOfView: Double?, venue: String?)
    case addFrame(time: Double, jpegBase64: String, info: CaptureFrameInfo)
    case setColor(String?)
    case tap(x: Double, y: Double, role: CaptureRole)
    case resetTaps
    /// “Not this problem”. Only newer pages have it; the call says whether this one does.
    case rejectRoute(String)
    case lock

    private struct Meta: Encodable {
        var id: String
        var w: Int
        var h: Int
        var fovH: Double?
        var venue: String?
    }
    private struct Info: Encodable {
        var pitch: Double?
        var roll: Double?
        var shake: Double?
        var pose: [[Double]]?
    }

    /// Every argument becomes a JavaScript literal through JSON encoding, never by joining quotes.
    private static func literal<T: Encodable>(_ value: T) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .withoutEscapingSlashes   // Base64 is full of them.
        return (try? encoder.encode(value)).flatMap { String(data: $0, encoding: .utf8) } ?? "null"
    }
    private static func finite(_ value: Double?, places: Double = 100) -> Double? {
        value.flatMap { $0.isFinite ? ($0 * places).rounded() / places : nil }
    }

    var script: String {
        let (function, arguments): (String, [String]) = switch self {
        case .begin(let id, let width, let height, let fieldOfView, let venue):
            ("begin", [Self.literal(Meta(id: id, w: width, h: height, fovH: Self.finite(fieldOfView), venue: venue?.isEmpty == false ? venue : nil))])
        case .addFrame(let time, let jpegBase64, let info):
            // The third argument is left out entirely when there is nothing to say.
            ("addFrame", [Self.literal(Self.finite(time, places: 1000) ?? 0), Self.literal(jpegBase64)] + (info.isEmpty ? [] : [Self.literal(Info(
                pitch: Self.finite(info.pitch), roll: Self.finite(info.roll), shake: Self.finite(info.shake, places: 10_000),
                pose: info.pose?.map { $0.map { $0.isFinite ? rounded4($0) : 0 } }))]))
        case .setColor(let key): ("setColor", [key.map(Self.literal) ?? "null"])
        case .tap(let x, let y, let role):
            ("tap", [Self.literal(rounded4(min(1, max(0, x)))), Self.literal(rounded4(min(1, max(0, y)))), Self.literal(role.rawValue)])
        case .resetTaps: ("resetTaps", [])
        case .rejectRoute(let id): ("rejectRoute", [Self.literal(id)])
        case .lock: ("lock", [])
        }
        let call = "window.BetaBuddyCapture.\(function)(\(arguments.joined(separator: ",")))"
        // `tap` answers with the id of the hold it found, or nothing; the other calls only say that they were made.
        let made = function == "tap" ? "(\(call) || '')" : "(\(call), true)"
        return "window.BetaBuddyCapture && typeof window.BetaBuddyCapture.\(function) === 'function' ? \(made) : false"
    }
}

/// The codes and moves this build has a sentence for. A newer pack may send others: an unknown
/// code is shown as the generic “try another position”, an unknown move is not shown at all.
enum CaptureGuidance {
    enum Code: String, CaseIterable, Sendable {
        case noWall = "no_wall", tooDark = "too_dark", backlit, blurry, shaking, tilted
        case tooClose = "too_close", tooFar = "too_far"
        case cutTop = "cut_top", cutBottom = "cut_bottom", cutLeft = "cut_left", cutRight = "cut_right"
        case oblique, pitchHigh = "pitch_high", occluded, noHolds = "no_holds"
        case startUnclear = "start_unclear", finishUnclear = "finish_unclear"
    }
    enum Move: String, CaseIterable, Sendable {
        case back, closer, left, right, up, down
        case tiltUp = "tilt_up", tiltDown = "tilt_down", level, holdStill = "hold_still"
    }
}

/// Angles of the phone from the gravity vector Core Motion reports, in the device's own axes
/// (x to the right of the screen, y to its top, z out of the screen towards the user; the vector
/// points down and has length 1). The back camera looks along −z.
enum CaptureMotion {
    /// Elevation of the optical axis above the horizon, degrees: 0 with the phone upright, positive
    /// when the camera looks up.
    static func pitch(gravity: (x: Double, y: Double, z: Double)) -> Double? {
        guard gravity.x.isFinite, gravity.y.isFinite, gravity.z.isFinite else { return nil }
        return asin(min(1, max(-1, gravity.z))) * 180 / .pi
    }
    /// Rotation about the optical axis, degrees, clockwise positive as seen on the screen. With the
    /// phone lying nearly flat there is no “upright” to measure from: the answer is 0, and the pitch
    /// says what is wrong.
    static func roll(gravity: (x: Double, y: Double, z: Double)) -> Double? {
        guard gravity.x.isFinite, gravity.y.isFinite, gravity.z.isFinite else { return nil }
        guard (gravity.x * gravity.x + gravity.y * gravity.y).squareRoot() > 0.3 else { return 0 }
        return atan2(gravity.x, -gravity.y) * 180 / .pi
    }
    /// The camera reports the field of view across the long side of its sensor. The frames are
    /// upright, so their horizontal field of view is the one across the short side.
    static func uprightFieldOfView(sensorDegrees: Double, longSide: Double, shortSide: Double) -> Double? {
        guard sensorDegrees.isFinite, sensorDegrees > 1, sensorDegrees < 179, longSide > 0, shortSide > 0 else { return nil }
        return 2 * atan(tan(sensorDegrees * .pi / 360) * shortSide / longSide) * 180 / .pi
    }
}

enum CaptureLimits {
    /// Same ceiling as an imported video, less a little so the finished file stays under it.
    static let maximumDuration: Double = 20 * 60 - 2
    /// Shorter than this there is nothing to review.
    static let minimumDuration: Double = 1
    /// Free space needed to start recording, and the level at which a running recording is closed.
    static let startingFreeBytes: Int64 = 600 * 1024 * 1024
    static let stoppingFreeBytes: Int64 = 250 * 1024 * 1024
}

/// Why a recording ended.
enum CaptureStop: String, Equatable, Sendable {
    case user, limit, storage, interrupted, overheated
}
