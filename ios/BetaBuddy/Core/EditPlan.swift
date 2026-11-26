import Foundation

/// The render plan the pack's analysis page hands over with a result (`analysisResult.edit`, v 1).
/// What to slow down, where to look and what must stay live is the page's decision; this type only
/// reads the plan, refuses one that breaks its own rules, and answers the renderer's questions.
/// Times are seconds in the source clip; coordinates are 0–1 in the upright picture, origin top-left.
struct EditPlan: Equatable, Sendable {
    struct CameraKey: Equatable, Sendable {
        var t: Double, cx: Double, cy: Double, s: Double
    }
    struct Segment: Equatable, Sendable {
        var t0: Double, t1: Double, speed: Double
        /// Used instead of the plan's camera while this segment plays.
        var camera: [CameraKey]? = nil
    }
    struct Box: Equatable, Sendable {
        var x: Double, y: Double, w: Double, h: Double
        /// The columns (or rows) `from..<to` this box covers of `count`, whatever its numbers are.
        static func cells(_ start: Double, _ length: Double, of count: Int) -> Range<Int> {
            func cell(_ value: Double) -> Int { Int(min(Double(count), max(0, value.isFinite ? value : 0))) }
            let from = cell((start * Double(count)).rounded(.down)), to = cell(((start + length) * Double(count)).rounded(.up))
            return from..<max(from, to)
        }
        func intersects(_ other: Box) -> Bool {
            x < other.x + other.w && other.x < x + w && y < other.y + other.h && other.y < y + h
        }
        func grown(by margin: Double) -> Box { Box(x: x - margin, y: y - margin, w: w + 2 * margin, h: h + 2 * margin) }
    }
    struct Keep: Equatable, Sendable {
        /// Soft edge of what is replaced, as a fraction of the picture's short side.
        var feather: Double
        var samples: [(t: Double, boxes: [Box])]
        static func == (a: Keep, b: Keep) -> Bool {
            a.feather == b.feather && a.samples.count == b.samples.count && zip(a.samples, b.samples).allSatisfy { $0.t == $1.t && $0.boxes == $1.boxes }
        }

        /// The boxes that stay live at `time`: box by box between two samples with the same number
        /// of boxes, otherwise those of the nearer sample.
        func boxes(at time: Double) -> [Box] {
            guard let first = samples.first, let last = samples.last else { return [] }
            if time <= first.t { return first.boxes }
            if time >= last.t { return last.boxes }
            var low = 0, high = samples.count - 1   // samples[low].t <= time < samples[high].t
            while high - low > 1 {
                let mid = (low + high) / 2
                if samples[mid].t <= time { low = mid } else { high = mid }
            }
            let a = samples[low], b = samples[high]
            let span = b.t - a.t, k = span > 0 ? (time - a.t) / span : 0
            guard a.boxes.count == b.boxes.count else { return k < 0.5 ? a.boxes : b.boxes }
            return zip(a.boxes, b.boxes).map { Box(x: $0.x + ($1.x - $0.x) * k, y: $0.y + ($1.y - $0.y) * k, w: $0.w + ($1.w - $0.w) * k, h: $0.h + ($1.h - $0.h) * k) }
        }
    }

    var duration: Double
    /// The attempt: what plays, at its own speed, when slow motion is switched off.
    var trim: ClosedRange<Double>
    var segments: [Segment]
    var camera: [CameraKey]
    /// The camera keys are a dense track of the subject (`"cameraMode": "track"`): the window moves in
    /// a straight line from one key to the next. Otherwise each move between two keys is eased.
    var tracksSubject = false
    /// `nil` when the plan does not offer passer-by removal.
    var keep: Keep?
    /// Source times whose per-pixel median is the empty wall. Empty when `keep` is `nil`.
    var plateTimes: [Double]

    static let maximumSegments = 40
    static let maximumPlateTimes = 25
    static let maximumLength = 2_000_000

    var offersRemoval: Bool { keep != nil }
    var offersSlowMotion: Bool { segments != [Segment(t0: trim.lowerBound, t1: trim.upperBound, speed: 1)] }
    var offersCamera: Bool { !camera.isEmpty || segments.contains { $0.camera?.isEmpty == false } }

    init?(json: String) {
        guard json.utf8.count <= Self.maximumLength, let object = try? JSONSerialization.jsonObject(with: Data(json.utf8)) else { return nil }
        self.init(object: object)
    }

    /// `nil` for anything that is not a version 1 plan with valid segments. Optional parts that
    /// are malformed are left out rather than failing the whole plan.
    init?(object: Any?) {
        func number(_ value: Any?) -> Double? {
            guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(), number.doubleValue.isFinite else { return nil }
            return number.doubleValue
        }
        guard let plan = object as? [String: Any], number(plan["v"]) == 1,
              let duration = number(plan["duration"]), duration > 0 else { return nil }
        let slack = 0.05   // A plan's own rounding is not a reason to refuse it.
        func keys(_ value: Any?) -> [CameraKey] {
            ((value as? [Any]) ?? []).compactMap { item -> CameraKey? in
                guard let item = item as? [String: Any], let t = number(item["t"]), let cx = number(item["cx"]), let cy = number(item["cy"]), let s = number(item["s"]) else { return nil }
                return CameraKey(t: t, cx: min(1, max(0, cx)), cy: min(1, max(0, cy)), s: min(EditWindow.maximumZoom, max(1, s)))
            }.sorted { $0.t < $1.t }
        }
        guard let list = plan["segments"] as? [Any], (1...Self.maximumSegments).contains(list.count) else { return nil }
        var segments: [Segment] = []
        for item in list {
            guard let item = item as? [String: Any], let t0 = number(item["t0"]), let t1 = number(item["t1"]), let speed = number(item["speed"]),
                  t0 >= -slack, t1 <= duration + slack, t1 - t0 >= 0.2 - slack, (0.2...4).contains(speed) else { return nil }
            // A segment's own camera is a list; anything else there (absent, null) leaves the plan's camera in charge.
            segments.append(Segment(t0: max(0, t0), t1: min(duration, t1), speed: speed, camera: (item["camera"] as? [Any]).map { keys($0) }))
        }
        self.duration = duration
        self.segments = segments
        if let trim = plan["trim"] as? [Any], trim.count == 2, let a = number(trim[0]), let b = number(trim[1]), a >= -slack, b <= duration + slack,
           min(duration, b) - max(0, a) >= 0.05 {      // Checked as it will be used: cut to the clip, something is left.
            self.trim = max(0, a)...min(duration, b)
        } else {
            self.trim = (segments.map(\.t0).min() ?? 0)...(segments.map(\.t1).max() ?? duration)
        }
        camera = keys(plan["camera"])
        let times = (((plan["plate"] as? [String: Any])?["times"] as? [Any]) ?? []).compactMap(number).filter { $0 >= -slack && $0 <= duration + slack }.map { min(duration, max(0, $0)) }
        var keep: Keep?
        if let object = plan["keep"] as? [String: Any], let list = object["boxes"] as? [Any], !times.isEmpty {
            let samples = list.compactMap { item -> (t: Double, boxes: [Box])? in
                guard let item = item as? [String: Any], let t = number(item["t"]), let boxes = item["b"] as? [Any] else { return nil }
                return (t, boxes.compactMap { box -> Box? in
                    guard let box = box as? [Any], box.count == 4, let x = number(box[0]), let y = number(box[1]), let w = number(box[2]), let h = number(box[3]), w > 0, h > 0 else { return nil }
                    // Cut to the picture; a box that lies outside it altogether is no box.
                    if x >= 0, y >= 0, x + w <= 1, y + h <= 1 { return Box(x: x, y: y, w: w, h: h) }
                    let x0 = min(1, max(0, x)), x1 = min(1, max(0, x + w)), y0 = min(1, max(0, y)), y1 = min(1, max(0, y + h))
                    guard x1 > x0, y1 > y0 else { return nil }
                    return Box(x: x0, y: y0, w: x1 - x0, h: y1 - y0)
                })
            }.sorted { $0.t < $1.t }
            if !samples.isEmpty { keep = Keep(feather: min(0.2, max(0, number(object["feather"]) ?? 0.03)), samples: samples) }
        }
        self.keep = keep
        tracksSubject = plan["cameraMode"] as? String == "track"
        plateTimes = keep == nil ? [] : Array(times.prefix(Self.maximumPlateTimes))
    }
}

/// The three switches on the edit page. Each one only does something when the plan has the data.
struct EditOptions: Equatable, Sendable {
    /// Passer-by removal is built and tested but not offered yet: it has not been tried on real footage
    /// with passers-by. The edit page shows no switch for it and starts with it off.
    static let removalOffered = false
    /// What the edit page starts from.
    static var offered: EditOptions { EditOptions(removePassersBy: removalOffered) }
    var removePassersBy = true
    var slowMotion = true
    var cameraMoves = true
}

/// The part of the picture that is shown, in the upright picture's 0–1 coordinates.
struct EditWindow: Equatable, Sendable {
    var cx = 0.5, cy = 0.5, s = 1.0
    static let full = EditWindow()
    var x: Double { cx - 0.5 / s }
    var y: Double { cy - 0.5 / s }
    var side: Double { 1 / s }

    /// The plan format allows this much: a climber who is a twentieth of a wide shot still fills a fair share of the window.
    static let maximumZoom = 8.0

    /// Kept inside the picture: zoom between 1 and `maximumZoom`, the centre moved in from an edge as needed.
    var clamped: EditWindow {
        let s = min(Self.maximumZoom, max(1, self.s.isFinite ? self.s : 1)), half = 0.5 / s
        return EditWindow(cx: min(1 - half, max(half, cx.isFinite ? cx : 0.5)), cy: min(1 - half, max(half, cy.isFinite ? cy : 0.5)), s: s)
    }

    /// Between the two keyframes around `time`: eased (smoothstep), or in a straight line when the keys
    /// are a dense track of the subject (`linear`). The first and last hold outside them.
    static func at(_ time: Double, keys: [EditPlan.CameraKey], linear: Bool = false) -> EditWindow {
        guard let first = keys.first, let last = keys.last else { return .full }
        func window(_ key: EditPlan.CameraKey) -> EditWindow { EditWindow(cx: key.cx, cy: key.cy, s: key.s).clamped }
        if time <= first.t { return window(first) }
        if time >= last.t { return window(last) }
        guard let next = keys.firstIndex(where: { $0.t > time }), next > 0 else { return window(last) }
        let a = keys[next - 1], b = keys[next]
        let u = b.t > a.t ? (time - a.t) / (b.t - a.t) : 1, k = linear ? u : u * u * (3 - 2 * u)
        return EditWindow(cx: a.cx + (b.cx - a.cx) * k, cy: a.cy + (b.cy - a.cy) * k, s: a.s + (b.s - a.s) * k).clamped
    }
}

/// The output as a run of source ranges, each at its own speed: where in the clip a moment of the
/// result comes from, and which camera keyframes apply there.
struct EditTimeline: Equatable, Sendable {
    struct Piece: Equatable, Sendable {
        var t0: Double, t1: Double, speed: Double
        /// When this piece begins in the result, in seconds.
        var start: Double
        var camera: [EditPlan.CameraKey]
        /// The keys are a dense track: straight lines between them, no easing.
        var linear = false
        var length: Double { (t1 - t0) / speed }
        var end: Double { start + length }
        func sourceTime(atOutput time: Double) -> Double { min(t1, max(t0, t0 + (time - start) * speed)) }
    }
    var pieces: [Piece]
    var duration: Double { pieces.last?.end ?? 0 }

    /// `sourceDuration` is the clip as it really is: ranges are cut to it, and what is left of a
    /// range under a twentieth of a second is dropped.
    init(plan: EditPlan, options: EditOptions, sourceDuration: Double) {
        let segments = options.slowMotion ? plan.segments : [EditPlan.Segment(t0: plan.trim.lowerBound, t1: plan.trim.upperBound, speed: 1)]
        var pieces: [Piece] = [], cursor = 0.0
        for segment in segments {
            let t0 = max(0, segment.t0), t1 = min(sourceDuration, segment.t1)
            guard t1 - t0 >= 0.05 else { continue }
            let piece = Piece(t0: t0, t1: t1, speed: segment.speed, start: cursor, camera: options.cameraMoves ? segment.camera ?? plan.camera : [], linear: plan.tracksSubject)
            pieces.append(piece)
            cursor = piece.end
        }
        self.pieces = pieces
    }

    /// The piece playing at `time` of the result and the moment of the clip it shows.
    func source(atOutput time: Double) -> (piece: Int, time: Double)? {
        guard !pieces.isEmpty else { return nil }
        let index = pieces.lastIndex { $0.start <= time } ?? 0
        return (index, pieces[index].sourceTime(atOutput: time))
    }
}

/// People found frame after frame, linked into tracks: one person followed through the clip. A
/// passer-by is removed for their whole track or not at all, so a track is marked as failed the
/// first time one of its moments cannot be replaced cleanly, and stays failed.
/// Boxes are in the upright picture's 0–1 coordinates.
struct EditTracks: Sendable {
    struct Sample: Equatable, Sendable {
        var time: Double
        var box: EditPlan.Box
        /// Where the caller keeps this moment's outline, if it keeps one.
        var shape: Int? = nil
    }
    /// Why a person is left as filmed.
    enum Failure: String, Sendable {
        case keepBox = "touches keep box"
        case noPlate = "no plate"
        case plateMismatch = "plate mismatch"
        case midPicture = "mid-picture"
        case overKept = "overlaps kept person"
    }
    struct Track: Equatable, Sendable {
        var samples: [Sample]
        var removable = true
        /// The first reason found, when one was given. Only for reports; nothing is decided by it.
        var reason: Failure? = nil
    }
    /// What stands for a removable person at one moment: the outline found then, or, where they went
    /// unseen for a moment inside their track, a box between the ones before and after.
    struct Shape: Equatable, Sendable {
        var track: Int
        var box: EditPlan.Box
        var shape: Int?
        var bridged: Bool
    }
    /// How much a keep box is grown before anything is tested against it, as a fraction of the picture.
    static let margin = 0.02
    /// Seconds a person may go unseen and still be the same track.
    let gap: Double
    private(set) var tracks: [Track] = []

    init(gap: Double = 0.5) { self.gap = gap }

    /// The people found at `time` (later than every time given before). Each joins the track it
    /// continues, best overlap first, or starts one. Returns the track of each box.
    mutating func add(time: Double, boxes: [EditPlan.Box]) -> [Int] {
        var pairs: [(score: Double, track: Int, box: Int)] = []
        for (index, track) in tracks.enumerated() {
            guard let last = track.samples.last, time > last.time, time - last.time <= gap + 1e-9 else { continue }
            for (number, box) in boxes.enumerated() {
                let score = Self.likeness(last.box, box)
                if score > 0 { pairs.append((score, index, number)) }
            }
        }
        var chosen = [Int?](repeating: nil, count: boxes.count)
        var taken: Set<Int> = []
        for pair in pairs.sorted(by: { $0.score > $1.score }) where chosen[pair.box] == nil && !taken.contains(pair.track) {
            chosen[pair.box] = pair.track
            taken.insert(pair.track)
        }
        return boxes.indices.map { number in
            let sample = Sample(time: time, box: boxes[number])
            if let track = chosen[number] { tracks[track].samples.append(sample); return track }
            tracks.append(Track(samples: [sample]))
            return tracks.count - 1
        }
    }

    /// Overlap of the two boxes (intersection over union); for boxes that no longer overlap but are
    /// still close, a small value that falls with the distance between their centres.
    static func likeness(_ a: EditPlan.Box, _ b: EditPlan.Box) -> Double {
        let w = min(a.x + a.w, b.x + b.w) - max(a.x, b.x), h = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
        let shared = max(0, w) * max(0, h), union = a.w * a.h + b.w * b.h - shared
        let overlap = union > 0 ? shared / union : 0
        if overlap >= 0.1 { return overlap }
        let dx = a.x + a.w / 2 - b.x - b.w / 2, dy = a.y + a.h / 2 - b.y - b.h / 2
        let reach = 0.75 * max(a.w, a.h, b.w, b.h), distance = (dx * dx + dy * dy).squareRoot()
        return reach > 0 && distance < reach ? 0.09 * (1 - distance / reach) : 0
    }

    /// This person cannot be replaced cleanly at some moment: they are left as filmed throughout.
    mutating func fail(_ track: Int, _ reason: Failure? = nil) {
        guard tracks.indices.contains(track) else { return }
        if tracks[track].removable || tracks[track].reason == nil { tracks[track].reason = reason }
        tracks[track].removable = false
    }

    /// Where the caller keeps the outline of the moment just added to `track`.
    mutating func setShape(_ shape: Int, track: Int) {
        guard tracks.indices.contains(track), !tracks[track].samples.isEmpty else { return }
        tracks[track].samples[tracks[track].samples.count - 1].shape = shape
    }

    private static func between(_ a: Sample, _ b: Sample, at time: Double) -> EditPlan.Box {
        let k = b.time > a.time ? (time - a.time) / (b.time - a.time) : 0
        return EditPlan.Box(x: a.box.x + (b.box.x - a.box.x) * k, y: a.box.y + (b.box.y - a.box.y) * k, w: a.box.w + (b.box.w - a.box.w) * k, h: a.box.h + (b.box.h - a.box.h) * k)
    }

    /// A person who is suddenly there, or suddenly gone, in the middle of the picture was lost
    /// sight of rather than seen coming or going: taking them out only while they were found would
    /// have them appear and vanish. Such tracks are failed. A track may begin and end at an edge of
    /// the picture (`edge` = how close, as a fraction of it) or with the stretch that was looked at
    /// (`ranges`, in seconds; within two and a half steps of its start or end).
    mutating func failUnseenComingOrGoing(ranges: [(Double, Double)], step: Double, edge: Double = 0.02) {
        func atEdge(_ box: EditPlan.Box) -> Bool { box.x <= edge || box.y <= edge || box.x + box.w >= 1 - edge || box.y + box.h >= 1 - edge }
        for index in tracks.indices where tracks[index].removable {
            guard let first = tracks[index].samples.first, let last = tracks[index].samples.last else { continue }
            let begins = atEdge(first.box) || ranges.contains { first.time >= $0.0 - 1e-9 && first.time <= $0.0 + 2.5 * step }
            let ends = atEdge(last.box) || ranges.contains { last.time <= $0.1 + 1e-9 && last.time >= $0.1 - 2.5 * step }
            if !begins || !ends { fail(index, .midPicture) }
        }
    }

    /// Where the person of `track` is at `time`, removable or not: the sighting within 0.6 of a
    /// step, or a box between the sightings before and after. `nil` outside the time they are seen.
    func shape(of index: Int, at time: Double, step: Double) -> Shape? {
        guard tracks.indices.contains(index) else { return nil }
        let track = tracks[index]
        guard let first = track.samples.first, let last = track.samples.last, time >= first.time - 0.6 * step, time <= last.time + 0.6 * step else { return nil }
        let next = track.samples.firstIndex { $0.time > time } ?? track.samples.count
        let before = next > 0 ? track.samples[next - 1] : nil, after = next < track.samples.count ? track.samples[next] : nil
        let nearest = [before, after].compactMap { $0 }.min { abs($0.time - time) < abs($1.time - time) }
        if let nearest, abs(nearest.time - time) <= 0.6 * step { return Shape(track: index, box: nearest.box, shape: nearest.shape, bridged: false) }
        guard let before, let after else { return nil }
        return Shape(track: index, box: Self.between(before, after, at: time), shape: nil, bridged: true)
    }

    /// The removable people at `time`. `step` is how far apart the moments looked at were: the
    /// outline found within 0.6 of a step is used as it is, anything further inside a track is bridged.
    func shapes(at time: Double, step: Double) -> [Shape] {
        tracks.indices.compactMap { tracks[$0].removable ? shape(of: $0, at: time, step: step) : nil }
    }

    /// A person who is taken out must never be painted over one who stays. For every removable
    /// track and every moment it is replaced at (each sighting and each bridged moment), `overlaps`
    /// is asked about each kept person there at that moment; a track that overlaps one is kept as
    /// well. Repeated until nothing changes, since a track that is now kept can be overlapped in turn.
    mutating func failThoseOverKeptPeople(step: Double, overlaps: (_ moment: Shape, _ kept: EditPlan.Box) -> Bool = { $0.box.intersects($1) }) {
        var changed = true
        while changed {
            changed = false
            let bridges = bridges(step: step)
            for index in tracks.indices where tracks[index].removable {
                let moments = tracks[index].samples.map { (time: $0.time, shape: Shape(track: index, box: $0.box, shape: $0.shape, bridged: false)) }
                    + bridges.filter { $0.track == index }.map { (time: $0.time, shape: Shape(track: index, box: $0.box, shape: nil, bridged: true)) }
                let covered = moments.contains { moment in
                    tracks.indices.contains { other in
                        guard other != index, !tracks[other].removable, let kept = shape(of: other, at: moment.time, step: step) else { return false }
                        return overlaps(moment.shape, kept.box)
                    }
                }
                if covered { fail(index, .overKept); changed = true }
            }
        }
    }

    /// The moments inside removable tracks at which nobody was found and a box between the
    /// neighbours stands in, one per step: they have to be replaceable too.
    func bridges(step: Double) -> [(track: Int, time: Double, box: EditPlan.Box)] {
        guard step > 0 else { return [] }
        var result: [(track: Int, time: Double, box: EditPlan.Box)] = []
        for (index, track) in tracks.enumerated() where track.removable {
            for (a, b) in zip(track.samples, track.samples.dropFirst()) where b.time - a.time > 1.5 * step {
                var time = a.time + step
                while time < b.time - 0.5 * step {
                    result.append((index, time, Self.between(a, b, at: time)))
                    time += step
                }
            }
        }
        return result
    }
}

/// Where rendered edits live: the temporary directory, one folder per record. They are made again
/// on demand and are never part of a record.
enum EditFiles {
    static var root: URL { FileManager.default.temporaryDirectory.appendingPathComponent("BetaBuddyEdit", isDirectory: true) }
    static func directory(for id: UUID) -> URL { root.appendingPathComponent(id.uuidString, isDirectory: true) }
    static func remove(for id: UUID) { try? FileManager.default.removeItem(at: directory(for: id)) }
}
