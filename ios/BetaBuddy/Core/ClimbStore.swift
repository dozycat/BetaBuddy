import Foundation
import SwiftUI
import UIKit

@MainActor
final class ClimbStore: ObservableObject {
    @Published private(set) var archive = Archive()
    @Published var error: String?
    let root: URL
    private var canWrite = true
    @Published private(set) var analysisProgress: [UUID: Double] = [:]
    @Published private(set) var analysisMessages: [UUID: String] = [:]
    private var analysisWork: Task<Void, Never>?
    private var activeAnalysis: UUID?
    private var backgroundTask: UIBackgroundTaskIdentifier = .invalid
    private var mayAnalyze = true
    /// Records that have a saved result from the pack's analysis page (`<id>.analysis.json`).
    @Published private(set) var analysisResults: Set<UUID> = []
    static let analysisResultSuffix = ".analysis.json"
    static let analysisClimberKey = "analysisClimber"
    private let defaults: UserDefaults

    init(root: URL? = nil, defaults: UserDefaults = .standard) {
        self.defaults = defaults
        self.root = root ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("BetaBuddy", isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: self.root, withIntermediateDirectories: true)
            var directory = self.root
            var resourceValues = URLResourceValues()
            resourceValues.isExcludedFromBackup = true
            try directory.setResourceValues(resourceValues)
            let index = self.root.appendingPathComponent("archive.json")
            if FileManager.default.fileExists(atPath: index.path) {
                archive = try JSONDecoder().decode(Archive.self, from: Data(contentsOf: index))
                guard archive.version == 1 else { throw CocoaError(.fileReadCorruptFile) }
            }
            let names = (try? FileManager.default.contentsOfDirectory(atPath: self.root.path)) ?? []
            analysisResults = Set(names.compactMap { $0.hasSuffix(Self.analysisResultSuffix) ? UUID(uuidString: String($0.dropLast(Self.analysisResultSuffix.count))) : nil })
        } catch {
            canWrite = false // Never overwrite an unreadable archive with an empty one.
            self.error = String(localized: "本地记录无法读取，已保留原文件。请检查存储空间或重新启动。", bundle: AppLanguage.bundle)
        }
    }

    var climbs: [Climb] { archive.climbs.sorted { $0.climbingDate > $1.climbingDate } }
    func file(_ name: String) -> URL { root.appendingPathComponent(name) }

    private func persist(_ next: Archive) throws {
        guard canWrite else { throw CocoaError(.fileWriteNoPermission) }
        let data = try JSONEncoder().encode(next)
        try data.write(to: file("archive.json"), options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
        archive = next
    }

    func save(title: String, grade: String, sent: Bool, note: String, stagedVideo: URL, thumbnail: Data, analysis: PoseAnalysis, gymId: String? = nil, gymName: String? = nil, climbedAt: Date? = nil, folderId: UUID? = nil, pendingAnalysis: String? = nil) throws -> Climb {
        let id = UUID()
        let videoName = "\(id).\(stagedVideo.pathExtension.isEmpty ? "mov" : stagedVideo.pathExtension)"
        let thumbnailName = "\(id).jpg"
        let climb = Climb(id: id, date: Date(), title: String(title.trimmingCharacters(in: .whitespacesAndNewlines).prefix(60)), grade: grade, sent: sent, note: String(note.prefix(1000)), videoName: videoName, thumbnailName: thumbnailName, analysis: analysis, gymId: gymId, gymName: cleanGymName(gymName), climbedAt: climbedAt, folderId: validFolder(folderId), pendingAnalysis: pendingAnalysis)
        do {
            try FileManager.default.moveItem(at: stagedVideo, to: file(videoName))
            try FileManager.default.setAttributes([.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication], ofItemAtPath: file(videoName).path)
            try thumbnail.write(to: file(thumbnailName), options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
            var next = archive
            next.climbs.append(climb)
            try persist(next)
            return climb
        } catch {
            if FileManager.default.fileExists(atPath: file(videoName).path) {
                try? FileManager.default.moveItem(at: file(videoName), to: stagedVideo)
            }
            try? FileManager.default.removeItem(at: file(thumbnailName))
            throw error
        }
    }

    func delete(_ climb: Climb) throws {
        var next = archive
        next.climbs.removeAll { $0.id == climb.id }
        try persist(next)
        if activeAnalysis == climb.id { analysisWork?.cancel() }
        try? FileManager.default.removeItem(at: file(climb.videoName))
        try? FileManager.default.removeItem(at: file(climb.thumbnailName))
        try? FileManager.default.removeItem(at: analysisResultFile(climb.id))
        analysisResults.remove(climb.id)
    }

    // MARK: Result of the pack's analysis page

    private func analysisResultFile(_ id: UUID) -> URL { file("\(id)" + Self.analysisResultSuffix) }

    /// The page's result exactly as it handed it over, or `nil` when there is none.
    func analysisResult(for climb: Climb) -> String? {
        guard analysisResults.contains(climb.id) else { return nil }
        return try? String(contentsOf: analysisResultFile(climb.id), encoding: .utf8)
    }

    /// Kept next to the clip, written atomically, inside the directory that is excluded from backup.
    /// The page sends a result again after the user adjusts it; each one replaces the last.
    func saveAnalysisResult(_ json: String, for id: UUID) throws {
        guard archive.climbs.contains(where: { $0.id == id }) else { return }   // Deleted while the page was open.
        guard canWrite else { throw CocoaError(.fileWriteNoPermission) }
        try Data(json.utf8).write(to: analysisResultFile(id), options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
        analysisResults.insert(id)
    }

    func discardAnalysisResult(_ climb: Climb) {
        try? FileManager.default.removeItem(at: analysisResultFile(climb.id))
        analysisResults.remove(climb.id)
    }

    /// Body measurements the analysis page last used, so the next analysis starts from them.
    var analysisClimber: [String: Double]? {
        get { (defaults.dictionary(forKey: Self.analysisClimberKey) as? [String: NSNumber])?.mapValues(\.doubleValue) }
        set { defaults.set(newValue, forKey: Self.analysisClimberKey) }
    }

    func update(_ climb: Climb, note: String, sent: Bool) throws {
        var next = archive
        guard let index = next.climbs.firstIndex(where: { $0.id == climb.id }) else { return }
        next.climbs[index].note = String(note.prefix(1000))
        next.climbs[index].sent = sent
        try persist(next)
    }

    var folders: [ClimbFolder] { archive.folders }

    private func validFolder(_ id: UUID?) -> UUID? { folders.contains { $0.id == id } ? id : nil }
    private func cleanGymName(_ name: String?) -> String? {
        let clean = String((name ?? "").trimmingCharacters(in: .whitespacesAndNewlines).prefix(120))
        return clean.isEmpty ? nil : clean
    }

    func saveFolder(name: String, id: UUID? = nil) throws {
        let name = String(name.trimmingCharacters(in: .whitespacesAndNewlines).prefix(40))
        guard !name.isEmpty else { throw AlbumError.emptyFolder }
        guard !folders.contains(where: { $0.id != id && $0.name.compare(name, options: [.caseInsensitive, .diacriticInsensitive]) == .orderedSame }) else { throw AlbumError.duplicateFolder }
        var next = archive
        if let id {
            guard let index = next.folders.firstIndex(where: { $0.id == id }) else { return }
            next.folders[index].name = name
        } else { next.folders.append(ClimbFolder(id: UUID(), name: name)) }
        try persist(next)
    }

    func deleteFolder(_ folder: ClimbFolder) throws {
        var next = archive
        next.folders.removeAll { $0.id == folder.id }
        for index in next.climbs.indices where next.climbs[index].folderId == folder.id { next.climbs[index].folderId = nil }
        try persist(next)
    }

    func organize(_ climb: Climb, title: String, grade: String, gymId: String?, gymName: String?, climbedAt: Date, folderId: UUID?) throws {
        var next = archive
        guard let index = next.climbs.firstIndex(where: { $0.id == climb.id }) else { return }
        next.climbs[index].title = String(title.trimmingCharacters(in: .whitespacesAndNewlines).prefix(60))
        next.climbs[index].grade = grade
        next.climbs[index].gymId = gymId
        next.climbs[index].gymName = cleanGymName(gymName)
        next.climbs[index].climbedAt = climbedAt
        next.climbs[index].folderId = validFolder(folderId)
        try persist(next)
    }

    func replaceAnalysis(_ climb: Climb, analysis: PoseAnalysis) throws {
        var next = archive
        guard let index = next.climbs.firstIndex(where: { $0.id == climb.id }) else { return }
        next.climbs[index].analysis = analysis
        next.climbs[index].pendingAnalysis = nil
        try persist(next)
    }

    func requestAnalysis(_ climb: Climb, quality: AnalysisQuality = .balanced) throws {
        guard !archive.climbs.contains(where: { $0.id == climb.id && $0.pendingAnalysis != nil }) else { return }
        try setPendingAnalysis(climb.id, quality: quality.rawValue)
        analysisMessages[climb.id] = nil
        resumeAnalysis()
    }

    private func setPendingAnalysis(_ id: UUID, quality: String?) throws {
        var next = archive
        guard let index = next.climbs.firstIndex(where: { $0.id == id }) else { return }
        next.climbs[index].pendingAnalysis = quality
        try persist(next)
    }

    func cancelAnalysis(_ climb: Climb) {
        do {
            try setPendingAnalysis(climb.id, quality: nil)
            if activeAnalysis == climb.id { analysisWork?.cancel() }
            analysisMessages[climb.id] = String(localized: "已停止，原分析和笔记已保留。", bundle: AppLanguage.bundle)
        } catch { self.error = error.localizedDescription }
    }

    /// Called when the app becomes active, including after a process restart.
    func resumeAnalysis() {
        mayAnalyze = true
        startNextAnalysis()
    }

    private func startNextAnalysis() {
        guard mayAnalyze, analysisWork == nil,
              let climb = archive.climbs.first(where: { $0.pendingAnalysis != nil }) else { return }
        activeAnalysis = climb.id
        analysisProgress[climb.id] = 0
        backgroundTask = UIApplication.shared.beginBackgroundTask(withName: "Climb analysis") { [weak self] in
            self?.pauseAnalysisForBackgroundExpiration()
        }
        analysisWork = Task {
            defer {
                analysisProgress[climb.id] = nil
                activeAnalysis = nil
                analysisWork = nil
                endBackgroundTime()
                startNextAnalysis()
            }
            do {
                let result = try await VideoAnalyzer().analyze(url: file(climb.videoName), quality: AnalysisQuality(rawValue: climb.pendingAnalysis ?? "") ?? .balanced) { value in
                    await self.setAnalysisProgress(climb.id, value: value)
                }
                try Task.checkCancellation()
                // replaceAnalysis merges into the latest record, preserving edits made while running.
                try replaceAnalysis(climb, analysis: result)
                analysisMessages[climb.id] = String(localized: "骨架分析已更新。", bundle: AppLanguage.bundle)
            } catch is CancellationError {
                // Explicit stop clears the request; expiration keeps it for the next foreground.
            } catch {
                analysisMessages[climb.id] = error.localizedDescription
                do { try setPendingAnalysis(climb.id, quality: nil) }
                catch { self.error = error.localizedDescription; mayAnalyze = false }
            }
        }
    }

    private func setAnalysisProgress(_ id: UUID, value: Double) { analysisProgress[id] = value }

    func pauseAnalysisForBackgroundExpiration() {
        mayAnalyze = false
        analysisWork?.cancel()
        endBackgroundTime()
    }

    private func endBackgroundTime() {
        if backgroundTask != .invalid {
            UIApplication.shared.endBackgroundTask(backgroundTask)
            backgroundTask = .invalid
        }
    }

}


enum AlbumError: LocalizedError {
    case emptyFolder, duplicateFolder
    var errorDescription: String? {
        switch self {
        case .emptyFolder: return String(localized: "给文件夹起个名字吧。", bundle: AppLanguage.bundle)
        case .duplicateFolder: return String(localized: "已有同名文件夹，换个名字试试。", bundle: AppLanguage.bundle)
        }
    }
}
