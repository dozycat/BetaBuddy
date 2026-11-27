import Foundation

/// Geometry in the upright video, normalized to its width and height. No screen coordinates.
struct SubjectBounds: Codable, Equatable, Sendable {
    var x: Double
    var y: Double
    var width: Double
    var height: Double
    var valid: Bool { [x, y, width, height].allSatisfy(\.isFinite) && x >= 0 && y >= 0 && width > 0 && height > 0 && x + width <= 1.0001 && y + height <= 1.0001 }
}
struct SubjectCandidate: Codable, Sendable, Identifiable {
    var id: String
    var bounds: SubjectBounds
    /// Generic, low-resolution clothing colour samples, kept on this device with the poses.
    var appearance: [Double]
    var pose: PoseFrame
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
    var hasSubjectCandidates: Bool { people?.contains { !$0.candidates.isEmpty } == true }
    /// Only exact candidate ids returned by the policy may become the selected poses.
    /// Missing samples stay missing; never fall back to the first person in a frame.
    func selecting(_ result: SubjectResolution) throws -> PoseAnalysis {
        guard result.schema == 1, let people, result.picks.count == people.count else { throw CocoaError(.fileReadCorruptFile) }
        var copy = self
        copy.frames = try zip(people, result.picks).compactMap { sample, pick in
            guard abs(sample.time - pick.time) < 0.0001 else { throw CocoaError(.fileReadCorruptFile) }
            guard let id = pick.personID else { return nil }
            guard let candidate = sample.candidates.first(where: { $0.id == id }) else { throw CocoaError(.fileReadCorruptFile) }
            return candidate.pose
        }
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
