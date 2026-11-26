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
    /// The landmarker's world landmarks for the same sample: 33 × `[x, y, z, visibility]`.
    /// Absent in records written before they were kept; those need pose recognition again
    /// before the pack's analysis page can use them.
    var world: [[Double]]? = nil
    var hasReliableTorso: Bool {
        joints.count == 33 && [11, 12, 23, 24].allSatisfy { joints[$0].valid }
    }
    var hasWorldLandmarks: Bool { world?.count == 33 }
}

/// Pose recognition output for one clip. Fields older builds wrote and this one no longer
/// uses are ignored when an archive is read.
struct PoseAnalysis: Codable, Sendable {
    var frames: [PoseFrame]
    var sampledFrames: Int
    var duration: Double
    var aspectRatio: Double
    /// False for records analysed before world landmarks were kept, and for clips not analysed yet.
    var hasWorldLandmarks: Bool { frames.contains(where: \.hasWorldLandmarks) }
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
    var gymId: String? = nil
    var gymName: String? = nil
    var climbedAt: Date? = nil
    var folderId: UUID? = nil
    /// Persisted before starting work, so interrupted analysis can restart on the next launch.
    var pendingAnalysis: String? = nil
    /// Set for records made with the in-app camera: what was chosen and recognised while framing.
    var capture: CaptureInfo? = nil
    /// `Discipline` raw value, or absent. Kept as text so a value a newer build wrote does not make the archive unreadable.
    var discipline: String? = nil
    /// The discipline came from the analysis page, not from the user; a later result may replace it.
    var disciplineAuto: Bool? = nil
    var style: Discipline? { discipline.flatMap(Discipline.init(rawValue:)) }
    /// What the user chose themselves, for handing to the analysis page.
    var chosenStyle: Discipline? { disciplineAuto == true ? nil : style }
    var climbingDate: Date { climbedAt ?? date }
    var gymKey: String {
        if let id = gymId, !id.isEmpty { return "id:" + id }
        let name = gymName?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return name.isEmpty ? "" : "name:" + name
    }
}

struct ClimbFolder: Codable, Identifiable, Equatable, Sendable {
    var id: UUID
    var name: String
}

/// Calendar-day comparisons include the entire end date, including daylight-saving changes.
struct JournalFilter {
    var folderId: UUID?
    var unfiled = false
    var gymKey: String?
    var grade: String?
    var startDate: Date?
    var endDate: Date?

    func matches(_ climb: Climb, calendar: Calendar = .current) -> Bool {
        if unfiled && climb.folderId != nil { return false }
        if let folderId, climb.folderId != folderId { return false }
        if let gymKey, climb.gymKey != gymKey { return false }
        if let grade, climb.grade != grade { return false }
        if let startDate, climb.climbingDate < calendar.startOfDay(for: startDate) { return false }
        if let endDate, let exclusiveEnd = calendar.date(byAdding: .day, value: 1, to: calendar.startOfDay(for: endDate)), climb.climbingDate >= exclusiveEnd { return false }
        return true
    }
}

struct Archive: Codable {
    var version = 1
    var climbs: [Climb] = []
    var folders: [ClimbFolder] = []
    // Keep version 1: these additive fields preserve existing video and analysis records.
    private enum CodingKeys: String, CodingKey { case version, climbs, folders }
    init(version: Int = 1, climbs: [Climb] = [], folders: [ClimbFolder] = []) {
        self.version = version; self.climbs = climbs; self.folders = folders
    }
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        version = try values.decode(Int.self, forKey: .version)
        climbs = try values.decode([Climb].self, forKey: .climbs)
        folders = try values.decodeIfPresent([ClimbFolder].self, forKey: .folders) ?? []
    }
}

/// Four decimal places are enough for stored landmarks and keep the archive small.
func rounded4(_ value: Double) -> Double { (value * 10_000).rounded() / 10_000 }

func clockText(_ time: Double) -> String {
    let seconds = max(0, Int(time.isFinite ? time : 0))
    return String(format: "%02d:%02d", seconds / 60, seconds % 60)
}
