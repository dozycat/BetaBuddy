import Foundation

struct Joint: Codable, Equatable, Sendable {
    var x: Double
    var y: Double
    var confidence: Double
    var valid: Bool { x.isFinite && y.isFinite && confidence >= 0.6 && (0...1).contains(x) && (0...1).contains(y) }
}

struct PoseFrame: Codable, Sendable {
    var time: Double
    var joints: [Joint]
    var hasReliableTorso: Bool {
        joints.count == 33 && [11, 12, 23, 24].allSatisfy { joints[$0].valid }
    }
    // Image-space hip midpoint; never presented as physical center of mass.
    var hip: Joint? {
        guard joints.count == 33, joints[23].valid, joints[24].valid else { return nil }
        return Joint(x: (joints[23].x + joints[24].x) / 2,
                     y: (joints[23].y + joints[24].y) / 2,
                     confidence: min(joints[23].confidence, joints[24].confidence))
    }
}

struct Moment: Codable, Identifiable, Sendable {
    var id = UUID()
    var time: Double
    var title: String
    var detail: String
    /// What triggered the cue. The app shows localized text for known kinds and falls back to
    /// the stored `title` / `detail`, which stay in the source language.
    var kind: String? = nil
    static let raisedHeel = "raisedHeel"
    static let fastMove = "fastMove"
}

struct PoseAnalysis: Codable, Sendable {
    var frames: [PoseFrame]
    var sampledFrames: Int
    var duration: Double
    var aspectRatio: Double
    var processingSeconds: Double
    var moments: [Moment]
    var coverage: Double { sampledFrames > 0 ? Double(frames.filter { $0.hip != nil }.count) / Double(sampledFrames) : 0 }
    var averageElbowAngle: Double? {
        let values = frames.flatMap { frame in
            [PoseMath.angle(frame, 11, 13, 15, aspect: aspectRatio), PoseMath.angle(frame, 12, 14, 16, aspect: aspectRatio)].compactMap { $0 }
        }
        return values.isEmpty ? nil : values.reduce(0, +) / Double(values.count)
    }
    func nearestFrame(to time: Double) -> PoseFrame? {
        // Binary search. Do not draw a stale skeleton across missing detections.
        var low = 0, high = frames.count
        while low < high {
            let mid = (low + high) / 2
            if frames[mid].time < time { low = mid + 1 } else { high = mid }
        }
        let choices = [low - 1, low].filter { frames.indices.contains($0) }.map { frames[$0] }
        return choices.min { abs($0.time - time) < abs($1.time - time) }.flatMap { abs($0.time - time) <= 0.16 && $0.hasReliableTorso ? $0 : nil }
    }
}

struct Climb: Codable, Identifiable, Sendable {
    var id: UUID
    var date: Date
    var title: String
    var grade: String
    var sent: Bool
    var note: String
    var videoName: String
    var thumbnailName: String
    var analysis: PoseAnalysis
}

struct Archive: Codable {
    var version = 1
    var climbs: [Climb] = []
    // Archives written by earlier builds may carry other keys; decoding ignores them.
}

func clockText(_ time: Double) -> String {
    let seconds = max(0, Int(time.isFinite ? time : 0))
    return String(format: "%02d:%02d", seconds / 60, seconds % 60)
}
