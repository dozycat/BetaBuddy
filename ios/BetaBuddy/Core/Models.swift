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
    var people: [SubjectSample]? = nil
    /// The full candidate set lives beside this record, loaded only while choosing a person.
    var peopleReference: UUID? = nil
    var subject: SubjectResolution? = nil
    /// Positions are independent of pose visibility; they can drive the camera without invented joints.
    var subjectPositions: [SubjectPosition]? = nil
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
    /// Raw value keeps archives readable if a later build adds another environment.
    var environment: String? = nil
    var environmentAuto: Bool? = nil
    /// Photo-library identity is local only; the import ledger survives deletion of this record.
    var sourceAssetID: String? = nil
    var gymAuto: Bool? = nil
    var subjectAnchors: [SubjectAnchor]? = nil
    /// Changes whenever the user saves another subject. Old asynchronous results are rejected.
    var subjectRevision: Int? = nil
    var setting: ClimbEnvironment { environment.flatMap(ClimbEnvironment.init(rawValue:)) ?? .unknown }
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

enum ClimbEnvironment: String, Codable, Sendable {
    case indoor, outdoor, unknown
}

/// A day at one place. Recomputed from records, so changing the date or gym moves the clip
/// immediately without creating, renaming or deleting the user's folders.
struct JournalDayGroup: Identifiable {
    struct ID: Hashable {
        var day: Date
        var gymKey: String
        var environment: ClimbEnvironment
    }
    var id: ID
    var climbs: [Climb]
    var gymName: String? { climbs.lazy.compactMap(\.gymName).first { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty } }

    static func make(_ climbs: [Climb], calendar: Calendar = .current) -> [JournalDayGroup] {
        let buckets = Dictionary(grouping: climbs) { climb in
            let environment: ClimbEnvironment = climb.setting == .outdoor ? .outdoor : (climb.gymKey.isEmpty ? climb.setting : .indoor)
            return ID(day: calendar.startOfDay(for: climb.climbingDate),
                      gymKey: environment == .outdoor ? "" : climb.gymKey, environment: environment)
        }
        return buckets.map { key, values in
            JournalDayGroup(id: key, climbs: values.sorted {
                if $0.climbingDate != $1.climbingDate { return $0.climbingDate > $1.climbingDate }
                return $0.id.uuidString < $1.id.uuidString
            })
        }.sorted {
            if $0.id.day != $1.id.day { return $0.id.day > $1.id.day }
            let first = $0.climbs[0], second = $1.climbs[0]
            if first.climbingDate != second.climbingDate { return first.climbingDate > second.climbingDate }
            return first.id.uuidString < second.id.uuidString
        }
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
    var libraryLedger: [String: LibraryReceipt] = [:]
    // Keep version 1: these additive fields preserve existing video and analysis records.
    private enum CodingKeys: String, CodingKey { case version, climbs, folders, libraryLedger }
    init(version: Int = 1, climbs: [Climb] = [], folders: [ClimbFolder] = []) {
        self.version = version; self.climbs = climbs; self.folders = folders
    }
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        version = try values.decode(Int.self, forKey: .version)
        climbs = try values.decode([Climb].self, forKey: .climbs)
        folders = try values.decodeIfPresent([ClimbFolder].self, forKey: .folders) ?? []
        libraryLedger = try values.decodeIfPresent([String: LibraryReceipt].self, forKey: .libraryLedger) ?? [:]
    }
}

/// Successful imports remain handled even after editing the source or deleting the local record.
/// A negative decision is reused only for the exact source revision and classifier configuration.
struct LibraryReceipt: Codable, Equatable, Sendable {
    var sourceRevision: String
    var classifierRevision: String
    var imported: Bool
    func covers(source: String, classifier: String) -> Bool {
        imported || (sourceRevision == source && classifierRevision == classifier)
    }
}

/// Four decimal places are enough for stored landmarks and keep the archive small.
func rounded4(_ value: Double) -> Double { (value * 10_000).rounded() / 10_000 }

func clockText(_ time: Double) -> String {
    let seconds = max(0, Int(time.isFinite ? time : 0))
    return String(format: "%02d:%02d", seconds / 60, seconds % 60)
}
