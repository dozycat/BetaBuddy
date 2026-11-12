import Foundation
import SwiftUI

@MainActor
final class ClimbStore: ObservableObject {
    @Published private(set) var archive = Archive()
    @Published var error: String?
    let root: URL
    private var canWrite = true

    init(root: URL? = nil) {
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
        } catch {
            canWrite = false // Never overwrite an unreadable archive with an empty one.
            self.error = String(localized: "本地记录无法读取，已保留原文件。请检查存储空间或重新启动。")
        }
    }

    var climbs: [Climb] { archive.climbs.sorted { $0.date > $1.date } }
    func file(_ name: String) -> URL { root.appendingPathComponent(name) }

    private func persist(_ next: Archive) throws {
        guard canWrite else { throw CocoaError(.fileWriteNoPermission) }
        let data = try JSONEncoder().encode(next)
        try data.write(to: file("archive.json"), options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
        archive = next
    }

    func save(title: String, grade: String, sent: Bool, note: String, stagedVideo: URL, thumbnail: Data, analysis: PoseAnalysis) throws -> Climb {
        let id = UUID()
        let videoName = "\(id).\(stagedVideo.pathExtension.isEmpty ? "mov" : stagedVideo.pathExtension)"
        let thumbnailName = "\(id).jpg"
        let climb = Climb(id: id, date: Date(), title: title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? String(localized: "今天的攀爬") : String(title.prefix(60)), grade: grade, sent: sent, note: String(note.prefix(1000)), videoName: videoName, thumbnailName: thumbnailName, analysis: analysis)
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
        try? FileManager.default.removeItem(at: file(climb.videoName))
        try? FileManager.default.removeItem(at: file(climb.thumbnailName))
    }

    func update(_ climb: Climb, note: String, sent: Bool) throws {
        var next = archive
        guard let index = next.climbs.firstIndex(where: { $0.id == climb.id }) else { return }
        next.climbs[index].note = String(note.prefix(1000))
        next.climbs[index].sent = sent
        try persist(next)
    }

    func replaceAnalysis(_ climb: Climb, analysis: PoseAnalysis) throws {
        var next = archive
        guard let index = next.climbs.firstIndex(where: { $0.id == climb.id }) else { return }
        next.climbs[index].analysis = analysis
        try persist(next)
    }
}
