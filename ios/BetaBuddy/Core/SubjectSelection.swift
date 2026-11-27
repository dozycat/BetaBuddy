import Foundation

/// Geometry in the upright video, normalized to its width and height. No screen coordinates.
struct SubjectBounds: Codable, Equatable, Sendable {
    var x: Double
    var y: Double
    var width: Double
    var height: Double
    var valid: Bool { [x, y, width, height].allSatisfy(\.isFinite) && x >= 0 && y >= 0 && width > 0 && height > 0 && x + width <= 1.0001 && y + height <= 1.0001 }
    func expanded(by fraction: Double) -> SubjectBounds {
        let left = max(0, x - width*fraction), top = max(0, y - height*fraction)
        let right = min(1, x + width*(1+fraction)), bottom = min(1, y + height*(1+fraction))
        return .init(x: left, y: top, width: right-left, height: bottom-top)
    }
    func overlap(_ other: SubjectBounds) -> Double {
        let area = max(0, min(x+width, other.x+other.width)-max(x, other.x)) * max(0, min(y+height, other.y+other.height)-max(y, other.y))
        return area / max(1e-9, width*height + other.width*other.height - area)
    }
}
struct SubjectCandidate: Codable, Sendable, Identifiable {
    var id: String
    var bounds: SubjectBounds
    /// Generic, low-resolution clothing colour samples, kept on this device with the poses.
    var appearance: [Double]
    var pose: PoseFrame
    var tracking: SubjectTracking? = nil
}
struct SubjectTracking: Codable, Sendable {
    var token: String
    var confidence: Double
}
struct SubjectPosition: Codable, Sendable {
    var time: Double
    var bounds: SubjectBounds
}
struct SubjectSample: Codable, Sendable {
    var time: Double
    var candidates: [SubjectCandidate]
}
struct SubjectAnchor: Codable, Equatable, Sendable {
    var time: Double
    var bounds: SubjectBounds
    var appearance: [Double]
    init(time: Double, candidate: SubjectCandidate) {
        self.time = time; bounds = candidate.bounds; appearance = candidate.appearance
    }
}
struct SubjectPick: Codable, Equatable, Sendable {
    var time: Double
    var personID: String?
}
struct SubjectResolution: Codable, Sendable {
    var schema: Int
    var picks: [SubjectPick]
    var manual: Bool
    var unresolved: Int
}

extension PoseAnalysis {
    var hasSubjectCandidates: Bool { people?.contains { !$0.candidates.isEmpty } ?? (peopleReference != nil) }
    var hasSelectedSubject: Bool { subjectPositions?.isEmpty == false || (subject != nil && !frames.isEmpty) }
    /// Only exact candidate ids returned by the policy may become the selected poses.
    /// Missing samples stay missing; never fall back to the first person in a frame.
    func selecting(_ result: SubjectResolution) throws -> PoseAnalysis {
        guard result.schema == 1, let people, result.picks.count == people.count else { throw CocoaError(.fileReadCorruptFile) }
        var copy = self
        let selected: [SubjectCandidate] = try zip(people, result.picks).compactMap { sample, pick in
            guard abs(sample.time - pick.time) < 0.0001 else { throw CocoaError(.fileReadCorruptFile) }
            guard let id = pick.personID else { return nil }
            guard let candidate = sample.candidates.first(where: { $0.id == id }) else { throw CocoaError(.fileReadCorruptFile) }
            guard candidate.bounds.valid, candidate.pose.time.isFinite,
                  abs(candidate.pose.time - sample.time) < 0.0001 else { throw CocoaError(.fileReadCorruptFile) }
            return candidate
        }
        copy.frames = selected.map(\.pose).filter(\.hasReliableTorso)
        copy.subjectPositions = selected.map { SubjectPosition(time: $0.pose.time, bounds: $0.bounds) }
        copy.subject = result
        return copy
    }
    func subjectSample(at time: Double) -> SubjectSample? {
        guard let people, !people.isEmpty else { return nil }
        var low = 0, high = people.count
        while low < high { let mid = (low + high) / 2; if people[mid].time < time { low = mid + 1 } else { high = mid } }
        let indices = [low - 1, low].filter { people.indices.contains($0) }
        return indices.map { people[$0] }.min { abs($0.time-time) < abs($1.time-time) }.flatMap { abs($0.time-time) <= 0.16 ? $0 : nil }
    }
}

/// Candidate poses can be much larger than the journal index. Files are immutable and the index
/// points at one revision, so a failed save cannot replace the currently readable candidate set.
enum SubjectCandidateStorage {
    static func file(root: URL, record: UUID, reference: UUID) -> URL {
        root.appendingPathComponent("\(record).subjects.\(reference).json.lzfse")
    }
    static func encode(_ people: [SubjectSample]) throws -> Data {
        try (JSONEncoder().encode(people) as NSData).compressed(using: .lzfse) as Data
    }
}

actor SubjectCandidateReader {
    func load(_ analysis: PoseAnalysis, root: URL, record: UUID) throws -> PoseAnalysis {
        try Task.checkCancellation()
        guard analysis.people == nil, let reference = analysis.peopleReference else { return analysis }
        let data = try Data(contentsOf: SubjectCandidateStorage.file(root: root, record: record, reference: reference))
        let decoded = try (data as NSData).decompressed(using: .lzfse) as Data
        var result = analysis
        result.people = try JSONDecoder().decode([SubjectSample].self, from: decoded)
        try Task.checkCancellation()
        return result
    }
}
