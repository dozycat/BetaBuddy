import Foundation

/// What is written next to a clip while it is being recorded: everything needed to make a record of it
/// without the capture screen, should the app not live to save it.
struct CaptureDraft: Codable, Equatable {
    var started: Date
    var gymId: String?
    var gymName: String
    var grade: String
    var info: CaptureInfo
    var plan: String?
    /// `Discipline` raw value the user chose on the capture screen; absent when it was left to the analysis.
    var discipline: String? = nil
}

/// Recordings are written inside the store's own directory, not the system's temporary one, so that a
/// clip survives the app being killed. At launch whatever an earlier run left there is either adopted
/// as a record (when it plays) or removed.
enum CaptureRecovery {
    static let folder = "recording"

    static func directory(in root: URL) -> URL { root.appendingPathComponent(folder, isDirectory: true) }

    /// Where a new recording and its draft go. The directory is created on the way.
    static func newClip(in root: URL) throws -> (movie: URL, draft: URL) {
        let directory = directory(in: root)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let name = UUID().uuidString
        return (directory.appendingPathComponent(name + ".mov"), directory.appendingPathComponent(name + ".json"))
    }

    static func draftURL(for movie: URL) -> URL { movie.deletingPathExtension().appendingPathExtension("json") }

    static func write(_ draft: CaptureDraft, for movie: URL) {
        guard let data = try? JSONEncoder().encode(draft) else { return }
        try? data.write(to: draftURL(for: movie), options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
    }

    /// Removes a clip and its draft.
    static func remove(_ movie: URL) {
        try? FileManager.default.removeItem(at: movie)
        try? FileManager.default.removeItem(at: draftURL(for: movie))
    }

    /// Adopts or cleans up what an earlier run left behind. Must not run while a recording is in
    /// progress; it is called once at launch. Returns the records made and the number of files removed.
    @MainActor
    @discardableResult
    static func sweep(store: ClimbStore, analysisAvailable: Bool) async -> (adopted: [UUID], removed: Int) {
        guard !store.cameraOpen else { return ([], 0) }
        let manager = FileManager.default
        var adopted: [UUID] = [], removed = 0
        // Clips of builds that still recorded into the temporary directory have nothing to go with them.
        for name in (try? manager.contentsOfDirectory(atPath: NSTemporaryDirectory())) ?? [] where name.hasPrefix("betabuddy-capture-") {
            try? manager.removeItem(at: manager.temporaryDirectory.appendingPathComponent(name))
            removed += 1
        }
        let directory = directory(in: store.root)
        let names = ((try? manager.contentsOfDirectory(atPath: directory.path)) ?? []).sorted()
        let analyzer = VideoAnalyzer()
        for name in names where name.hasSuffix(".mov") {
            let movie = directory.appendingPathComponent(name)
            let draft = (try? Data(contentsOf: draftURL(for: movie))).flatMap { try? JSONDecoder().decode(CaptureDraft.self, from: $0) }
            // A clip whose writer never finished plays up to its last complete fragment, or not at all.
            guard let playback = try? await analyzer.videoSummary(url: movie), playback.duration >= CaptureLimits.minimumDuration,
                  let thumbnail = try? await analyzer.thumbnail(url: movie) else {
                remove(movie); removed += 1
                continue
            }
            var info = draft?.info ?? CaptureInfo()
            info.analysisQueued = analysisAvailable ? true : nil
            var started = draft?.started
            if started == nil { started = await analyzer.recordingDate(url: movie) }
            do {
                let climb = try store.save(title: "", grade: draft?.grade ?? ungradedValue, sent: false, note: "", stagedVideo: movie, thumbnail: thumbnail, analysis: playback,
                                           gymId: draft?.gymId, gymName: draft?.gymName, climbedAt: started, pendingAnalysis: AnalysisQuality.balanced.rawValue,
                                           capture: info, capturePlan: draft?.plan, discipline: draft?.discipline.flatMap(Discipline.init(rawValue:)))
                try? manager.removeItem(at: draftURL(for: movie))
                store.setAnalysisMessage(CaptureText.stop(.interrupted), for: climb.id)
                adopted.append(climb.id)
            } catch {
                // The album cannot be written right now: the clip stays for the next launch.
            }
        }
        // Drafts without a clip.
        for name in names where name.hasSuffix(".json") && !manager.fileExists(atPath: directory.appendingPathComponent(name).deletingPathExtension().appendingPathExtension("mov").path) {
            if manager.fileExists(atPath: directory.appendingPathComponent(name).path) {
                try? manager.removeItem(at: directory.appendingPathComponent(name)); removed += 1
            }
        }
        return (adopted, removed)
    }
}
