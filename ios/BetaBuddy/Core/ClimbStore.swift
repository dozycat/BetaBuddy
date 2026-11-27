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
    /// The capture screen is framing or recording; nothing is analysed meanwhile.
    @Published private(set) var cameraOpen = false
    @Published private(set) var selectionOpen = false
    /// Records that have a saved result from the pack's analysis page (`<id>.analysis.json`).
    @Published private(set) var analysisResults: Set<UUID> = []
    static let analysisResultSuffix = ".analysis.json"
    /// The capture page's plan for a record made with the in-app camera (`<id>.capture.json`), kept as the page gave it.
    static let capturePlanSuffix = ".capture.json"
    /// Records that have a render plan from the analysis page (`<id>.edit.json`), kept as the page gave it.
    @Published private(set) var editPlans: Set<UUID> = []
    static let editPlanSuffix = ".edit.json"
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
            if archive.climbs.contains(where: { $0.analysis.people != nil }) { try persist(archive) }
            analysisResults = Set(archive.climbs.filter { FileManager.default.fileExists(atPath: analysisResultFile($0.id).path) }.map(\.id))
            editPlans = Set(archive.climbs.filter { FileManager.default.fileExists(atPath: editPlanFile($0.id).path) }.map(\.id))
        } catch {
            canWrite = false // Never overwrite an unreadable archive with an empty one.
            self.error = String(localized: "本地记录无法读取，已保留原文件。请检查存储空间或重新启动。", bundle: AppLanguage.bundle)
        }
    }

    var climbs: [Climb] { archive.climbs.sorted { $0.climbingDate > $1.climbingDate } }
    func file(_ name: String) -> URL { root.appendingPathComponent(name) }

    private func persist(_ next: Archive) throws {
        guard canWrite else { throw CocoaError(.fileWriteNoPermission) }
        var stored = next
        var created: [URL] = []
        do {
            for index in stored.climbs.indices {
                guard let people = stored.climbs[index].analysis.people else { continue }
                let reference = UUID()
                let destination = SubjectCandidateStorage.file(root: root, record: stored.climbs[index].id, reference: reference)
                try SubjectCandidateStorage.encode(people).write(to: destination, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
                created.append(destination)
                stored.climbs[index].analysis.people = nil
                stored.climbs[index].analysis.peopleReference = reference
            }
            let data = try JSONEncoder().encode(stored)
            try data.write(to: file("archive.json"), options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
        } catch {
            for url in created { try? FileManager.default.removeItem(at: url) }
            throw error
        }
        let previous = archive
        archive = stored
        for climb in previous.climbs {
            guard let reference = climb.analysis.peopleReference,
                  stored.climbs.first(where: { $0.id == climb.id })?.analysis.peopleReference != reference else { continue }
            try? FileManager.default.removeItem(at: SubjectCandidateStorage.file(root: root, record: climb.id, reference: reference))
        }
    }

    func subjectAnalysis(for climb: Climb) async throws -> PoseAnalysis {
        try await SubjectCandidateReader().load(climb.analysis, root: root, record: climb.id)
    }

    /// `capture` and `capturePlan` come from the in-app camera. The clip, its thumbnail, the plan and
    /// the record are all on disk when this returns; nothing is analysed here.
    func save(title: String, grade: String, sent: Bool, note: String, stagedVideo: URL, thumbnail: Data, analysis: PoseAnalysis, gymId: String? = nil, gymName: String? = nil, climbedAt: Date? = nil, folderId: UUID? = nil, pendingAnalysis: String? = nil, capture: CaptureInfo? = nil, capturePlan: String? = nil, discipline: Discipline? = nil, environment: ClimbEnvironment? = nil, sourceAssetID: String? = nil, libraryReceipt: LibraryReceipt? = nil) throws -> Climb {
        if libraryReceipt != nil, let sourceAssetID, archive.libraryLedger[sourceAssetID]?.imported == true {
            throw CocoaError(.fileWriteFileExists)
        }
        let id = UUID()
        var capture = capture
        capture?.hasPlan = capturePlan?.isEmpty == false
        let summary = capture?.summary?.stored
        capture?.summary = summary
        let videoName = "\(id).\(stagedVideo.pathExtension.isEmpty ? "mov" : stagedVideo.pathExtension)"
        let thumbnailName = "\(id).jpg"
        var climb = Climb(id: id, date: Date(), title: String(title.trimmingCharacters(in: .whitespacesAndNewlines).prefix(60)), grade: grade, sent: sent, note: String(note.prefix(1000)), videoName: videoName, thumbnailName: thumbnailName, analysis: analysis, gymId: gymId, gymName: cleanGymName(gymName), climbedAt: climbedAt, folderId: validFolder(folderId), pendingAnalysis: pendingAnalysis, capture: capture, discipline: discipline?.rawValue)
        climb.environment = environment?.rawValue
        climb.environmentAuto = environment == nil ? nil : true
        climb.sourceAssetID = sourceAssetID
        climb.gymAuto = libraryReceipt != nil && gymId != nil ? true : nil
        do {
            try FileManager.default.moveItem(at: stagedVideo, to: file(videoName))
            try FileManager.default.setAttributes([.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication], ofItemAtPath: file(videoName).path)
            try thumbnail.write(to: file(thumbnailName), options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
            if capture?.hasPlan == true, let capturePlan {
                try Data(capturePlan.utf8).write(to: capturePlanFile(id), options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
            }
            var next = archive
            next.climbs.append(climb)
            if let sourceAssetID {
                var receipt = libraryReceipt ?? LibraryReceipt(sourceRevision: "", classifierRevision: "", imported: true)
                receipt.imported = true
                next.libraryLedger[sourceAssetID] = receipt
            }
            try persist(next)
            return archive.climbs.first { $0.id == climb.id } ?? climb
        } catch {
            if FileManager.default.fileExists(atPath: file(videoName).path) {
                try? FileManager.default.moveItem(at: file(videoName), to: stagedVideo)
            }
            try? FileManager.default.removeItem(at: file(thumbnailName))
            try? FileManager.default.removeItem(at: capturePlanFile(id))
            throw error
        }
    }

    func hasHandledAsset(_ id: String, source: String, classifier: String) -> Bool {
        archive.climbs.contains { $0.sourceAssetID == id } ||
        archive.libraryLedger[id]?.covers(source: source, classifier: classifier) == true
    }

    func recordLibraryNonmatch(_ id: String, source: String, classifier: String) throws {
        guard archive.libraryLedger[id]?.imported != true else { return }
        var next = archive
        next.libraryLedger[id] = LibraryReceipt(sourceRevision: source, classifierRevision: classifier, imported: false)
        try persist(next)
    }

    func delete(_ climb: Climb) throws {
        let resultFile = analysisResultFile(climb.id), editFile = editPlanFile(climb.id)
        var next = archive
        next.climbs.removeAll { $0.id == climb.id }
        try persist(next)
        if activeAnalysis == climb.id { analysisWork?.cancel() }
        try? FileManager.default.removeItem(at: file(climb.videoName))
        try? FileManager.default.removeItem(at: file(climb.thumbnailName))
        try? FileManager.default.removeItem(at: resultFile)
        try? FileManager.default.removeItem(at: capturePlanFile(climb.id))
        try? FileManager.default.removeItem(at: editFile)
        EditFiles.remove(for: climb.id)
        analysisResults.remove(climb.id)
        editPlans.remove(climb.id)
        // A process may have stopped after advancing the selection revision but before deleting
        // old caches. Remove every file owned by this record, including those obsolete versions.
        for name in (try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? [] where name.hasPrefix("\(climb.id).") {
            try? FileManager.default.removeItem(at: file(name))
        }
    }

    // MARK: Records made with the in-app camera

    private func capturePlanFile(_ id: UUID) -> URL { file("\(id)" + Self.capturePlanSuffix) }

    /// The capture page's plan exactly as it handed it over, for the analysis page. `nil` for imported clips.
    func capturePlan(for climb: Climb) -> String? {
        guard archive.climbs.first(where: { $0.id == climb.id })?.capture?.hasPlan == true else { return nil }
        return try? String(contentsOf: capturePlanFile(climb.id), encoding: .utf8)
    }

    /// The page may hand over its plan, or a fuller one, after the clip has been saved.
    func saveCapturePlan(_ plan: String, summary: CaptureSummary?, for id: UUID) throws {
        var next = archive
        guard !plan.isEmpty, let index = next.climbs.firstIndex(where: { $0.id == id }), next.climbs[index].capture != nil else { return }
        guard canWrite else { throw CocoaError(.fileWriteNoPermission) }
        try Data(plan.utf8).write(to: capturePlanFile(id), options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
        next.climbs[index].capture?.hasPlan = true
        if let summary { next.climbs[index].capture?.summary = summary.stored }
        try persist(next)
    }

    /// What the analysis page said about the capture plan with its latest result. Only records made with the camera have a plan to speak of.
    func setPlanUse(_ use: CapturePlanUse?, for id: UUID) throws {
        var next = archive
        guard let index = next.climbs.firstIndex(where: { $0.id == id }), next.climbs[index].capture != nil, next.climbs[index].capture?.planUse != use else { return }
        next.climbs[index].capture?.planUse = use
        try persist(next)
    }

    /// Whether the analysis asked for when the clip was saved is still to be done.
    func setAnalysisQueued(_ id: UUID, _ queued: Bool) throws {
        var next = archive
        guard let index = next.climbs.firstIndex(where: { $0.id == id }), next.climbs[index].capture != nil,
              (next.climbs[index].capture?.analysisQueued ?? false) != queued else { return }
        next.climbs[index].capture?.analysisQueued = queued ? true : nil
        try persist(next)
    }

    /// The gym of the record made most recently that has one (by when the record was made, not by
    /// the climbing date the user may have set): where the next recording most likely is.
    var lastGym: (id: String?, name: String)? {
        archive.climbs.filter { !$0.gymKey.isEmpty }.max { $0.date < $1.date }.map { ($0.gymId, $0.shortGymName) }
    }

    func setAnalysisMessage(_ text: String?, for id: UUID) { analysisMessages[id] = text }

    // MARK: Result of the pack's analysis page

    private func derivedFile(_ id: UUID, suffix: String) -> URL {
        let revision = archive.climbs.first { $0.id == id }?.subjectRevision ?? 0
        return file("\(id)" + (revision == 0 ? "" : ".s\(revision)") + suffix)
    }
    private func analysisResultFile(_ id: UUID) -> URL { derivedFile(id, suffix: Self.analysisResultSuffix) }

    /// The page's result exactly as it handed it over, or `nil` when there is none.
    func analysisResult(for climb: Climb) -> String? {
        guard analysisResults.contains(climb.id) else { return nil }
        return try? String(contentsOf: analysisResultFile(climb.id), encoding: .utf8)
    }

    /// Kept next to the clip, written atomically, inside the directory that is excluded from backup.
    /// The page sends a result again after the user adjusts it; each one replaces the last.
    func saveAnalysisResult(_ json: String, for id: UUID, expectedSubjectRevision: Int? = nil) throws {
        if let expectedSubjectRevision, (archive.climbs.first { $0.id == id }?.subjectRevision ?? 0) != expectedSubjectRevision { return }
        guard archive.climbs.contains(where: { $0.id == id }) else { return }   // Deleted while the page was open.
        guard canWrite else { throw CocoaError(.fileWriteNoPermission) }
        try Data(json.utf8).write(to: analysisResultFile(id), options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
        analysisResults.insert(id)
    }

    func discardAnalysisResult(_ climb: Climb) {
        try? FileManager.default.removeItem(at: analysisResultFile(climb.id))
        analysisResults.remove(climb.id)
        try? saveEditPlan(nil, for: climb.id)
    }

    // MARK: Render plan and discipline that come with a result

    private func editPlanFile(_ id: UUID) -> URL { derivedFile(id, suffix: Self.editPlanSuffix) }

    /// The plan that came with the latest result, when it is one this build can render.
    func editPlan(for climb: Climb) -> EditPlan? {
        guard editPlans.contains(climb.id), let json = try? String(contentsOf: editPlanFile(climb.id), encoding: .utf8) else { return nil }
        return EditPlan(json: json)
    }

    /// Kept as the page gave it, next to the result it came with. A result without a plan removes the old one.
    func saveEditPlan(_ json: String?, for id: UUID, expectedSubjectRevision: Int? = nil) throws {
        if let expectedSubjectRevision, (archive.climbs.first { $0.id == id }?.subjectRevision ?? 0) != expectedSubjectRevision { return }
        guard let json, !json.isEmpty, archive.climbs.contains(where: { $0.id == id }) else {
            try? FileManager.default.removeItem(at: editPlanFile(id))
            EditFiles.remove(for: id)
            editPlans.remove(id)
            return
        }
        guard canWrite else { throw CocoaError(.fileWriteNoPermission) }
        try Data(json.utf8).write(to: editPlanFile(id), options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
        editPlans.insert(id)
    }

    /// What the analysis page made of the clip. The user's own choice is never replaced; neither is
    /// a grade they marked on the other scale, which says as much as a choice would.
    func setAutomaticDiscipline(_ kind: Discipline, for id: UUID) throws {
        var next = archive
        guard let index = next.climbs.firstIndex(where: { $0.id == id }) else { return }
        let climb = next.climbs[index]
        guard climb.chosenStyle == nil, climb.discipline != kind.rawValue || climb.disciplineAuto != true,
              GradeScale.kept(climb.grade, for: kind) == climb.grade else { return }
        next.climbs[index].discipline = kind.rawValue
        next.climbs[index].disciplineAuto = true
        try persist(next)
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

    /// `discipline` is given only when the user picked one in the form; it is theirs from then on.
    /// A grade of the other scale is cleared.
    func organize(_ climb: Climb, title: String, grade: String, gymId: String?, gymName: String?, climbedAt: Date, folderId: UUID?, discipline: Discipline? = nil, environment: ClimbEnvironment? = nil) throws {
        var next = archive
        guard let index = next.climbs.firstIndex(where: { $0.id == climb.id }) else { return }
        next.climbs[index].title = String(title.trimmingCharacters(in: .whitespacesAndNewlines).prefix(60))
        if let discipline {
            next.climbs[index].discipline = discipline.rawValue
            next.climbs[index].disciplineAuto = nil
        }
        next.climbs[index].grade = GradeScale.kept(grade, for: next.climbs[index].style)
        if next.climbs[index].gymId != gymId || next.climbs[index].gymName != cleanGymName(gymName) {
            next.climbs[index].gymAuto = nil
        }
        if let environment {
            next.climbs[index].environment = environment.rawValue
            next.climbs[index].environmentAuto = nil
        }
        next.climbs[index].gymId = gymId
        next.climbs[index].gymName = cleanGymName(gymName)
        next.climbs[index].climbedAt = climbedAt
        next.climbs[index].folderId = validFolder(folderId)
        try persist(next)
    }

    /// The saved selection governs all future analysis. Caches use the new revision even if
    /// the process stops before old files can be removed. An in-flight old result cannot revive them.
    func selectSubject(_ climb: Climb, analysis: PoseAnalysis, anchors: [SubjectAnchor]) throws {
        var next = archive
        guard let index = next.climbs.firstIndex(where: { $0.id == climb.id }),
              next.climbs[index].subjectRevision == climb.subjectRevision,
              analysis.subject != nil, analysis.hasSubjectCandidates else { throw CocoaError(.fileWriteUnknown) }
        let oldResult = analysisResultFile(climb.id), oldEdit = editPlanFile(climb.id)
        next.climbs[index].subjectRevision = (climb.subjectRevision ?? 0) + 1
        next.climbs[index].subjectAnchors = anchors.isEmpty ? nil : anchors
        next.climbs[index].analysis = analysis
        next.climbs[index].pendingAnalysis = nil
        next.climbs[index].capture?.analysisQueued = false
        try persist(next)
        if activeAnalysis == climb.id { analysisWork?.cancel() }
        analysisResults.remove(climb.id); editPlans.remove(climb.id)
        analysisMessages[climb.id] = nil
        try? FileManager.default.removeItem(at: oldResult)
        try? FileManager.default.removeItem(at: oldEdit)
        EditFiles.remove(for: climb.id)
    }

    func replaceAnalysis(_ climb: Climb, analysis: PoseAnalysis) throws {
        var next = archive
        guard let index = next.climbs.firstIndex(where: { $0.id == climb.id }) else { return }
        guard next.climbs[index].subjectRevision == climb.subjectRevision else { return }
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
    func resumeAnalysis(cameraClosed: Bool = false) {
        if cameraClosed { cameraOpen = false }
        guard !cameraOpen else { return }
        mayAnalyze = true
        startNextAnalysis()
    }

    private func startNextAnalysis() {
        guard mayAnalyze, !selectionOpen, analysisWork == nil,
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
            #if DEBUG
            let poseBegan = Date()
            #endif
            do {
                let result = try await VideoAnalyzer().analyze(url: file(climb.videoName), quality: AnalysisQuality(rawValue: climb.pendingAnalysis ?? "") ?? .balanced, subjectAnchors: climb.subjectAnchors ?? []) { value in
                    await self.setAnalysisProgress(climb.id, value: value)
                }
                try Task.checkCancellation()
                #if DEBUG
                if climb.capture != nil {
                    CaptureDiagnostics.event("pose", ["record": climb.id.uuidString, "frames": result.frames.count,
                                                      "framesWithPerson": result.frames.filter(\.hasWorldLandmarks).count, "seconds": Date().timeIntervalSince(poseBegan)])
                }
                #endif
                // replaceAnalysis merges into the latest record, preserving edits made while running.
                try replaceAnalysis(climb, analysis: result)
                analysisMessages[climb.id] = nil
            } catch is CancellationError {
                // Explicit stop clears the request; expiration keeps it for the next foreground.
            } catch {
                guard let current = archive.climbs.first(where: { $0.id == climb.id }), current.subjectRevision == climb.subjectRevision else { return }
                #if DEBUG
                if climb.capture != nil { CaptureDiagnostics.event("pose", ["record": climb.id.uuidString, "error": error.localizedDescription]) }
                #endif
                analysisMessages[climb.id] = error.localizedDescription
                do { try setPendingAnalysis(climb.id, quality: nil) }
                catch { self.error = error.localizedDescription; mayAnalyze = false }
            }
        }
    }

    private func setAnalysisProgress(_ id: UUID, value: Double) { analysisProgress[id] = value }

    func setSubjectSelectionOpen(_ open: Bool) {
        selectionOpen = open
        if open { pauseAnalysisForBackgroundExpiration() } else { resumeAnalysis() }
    }

    func pauseAnalysisForBackgroundExpiration() {
        mayAnalyze = false
        analysisWork?.cancel()
        endBackgroundTime()
    }

    /// The camera is framing or recording: pose recognition gives way to it. Requests stay and
    /// start again from the beginning once `resumeAnalysis(cameraClosed: true)` is called.
    func pauseAnalysisForCamera() {
        cameraOpen = true
        pauseAnalysisForBackgroundExpiration()
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
