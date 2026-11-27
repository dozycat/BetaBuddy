import AVFoundation
import Photos
import SwiftUI

/// Bridges callback completion and cancellation exactly once, including cancellation before
/// Photos has returned a request id. Late Photos callbacks cannot resume a cancelled scan.
private final class LibraryAssetRequest: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<AVAsset, Error>?
    private var result: Result<AVAsset, Error>?
    private var requestID: PHImageRequestID?
    private let manager = PHImageManager.default()

    func start(_ asset: PHAsset, continuation: CheckedContinuation<AVAsset, Error>) {
        lock.lock()
        if let result { lock.unlock(); continuation.resume(with: result); return }
        self.continuation = continuation
        lock.unlock()
        let options = PHVideoRequestOptions()
        options.isNetworkAccessAllowed = false
        options.deliveryMode = .highQualityFormat
        options.version = .current
        let id = manager.requestAVAsset(forVideo: asset, options: options) { [self] value, _, info in
            if let value { finish(.success(value)) }
            else { finish(.failure(info?[PHImageErrorKey] as? Error ?? CocoaError(.fileReadNoSuchFile))) }
        }
        lock.lock()
        requestID = id
        let finished = result != nil
        lock.unlock()
        if finished { manager.cancelImageRequest(id) }
    }
    private func finish(_ value: Result<AVAsset, Error>) {
        lock.lock()
        guard result == nil else { lock.unlock(); return }
        result = value
        let pending = continuation
        continuation = nil
        lock.unlock()
        pending?.resume(with: value)
    }
    func cancel() {
        finish(.failure(CancellationError()))
        lock.lock(); let id = requestID; lock.unlock()
        if let id { manager.cancelImageRequest(id) }
    }
    static func load(_ asset: PHAsset) async throws -> AVAsset {
        let request = LibraryAssetRequest()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { request.start(asset, continuation: $0) }
        } onCancel: { request.cancel() }
    }
}

/// Copies only matching clips. Photos originals and their metadata are never written to.
actor LibraryVideoStager {
    func stage(_ asset: AVAsset) async throws -> URL {
        try Task.checkCancellation()
        if let source = asset as? AVURLAsset { return try stageVideo(source.url) }
        let target = FileManager.default.temporaryDirectory.appendingPathComponent("betabuddy-library-\(UUID()).mov")
        guard let export = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetPassthrough) else { throw AnalysisError.invalidVideo }
        export.outputURL = target
        export.outputFileType = .mov
        export.fileLengthLimit = 500 * 1024 * 1024
        do {
            try await withTaskCancellationHandler {
                await export.export()
                try Task.checkCancellation()
                guard export.status == .completed else { throw export.error ?? AnalysisError.invalidVideo }
            } onCancel: { export.cancelExport() }
            return target
        } catch { try? FileManager.default.removeItem(at: target); throw error }
    }
}

@MainActor
final class PhotoLibraryImport: NSObject, ObservableObject, PHPhotoLibraryChangeObserver {
    @Published private(set) var enabled: Bool
    @Published private(set) var authorization = PHPhotoLibrary.authorizationStatus(for: .readWrite)
    @Published private(set) var running = false
    @Published private(set) var checked = 0
    @Published private(set) var total = 0
    @Published private(set) var imported = 0
    @Published private(set) var unavailable = 0
    @Published private(set) var error: String?
    private let defaults: UserDefaults
    private let fetchVideos: (() -> PHFetchResult<PHAsset>)?
    private weak var store: ClimbStore?
    private var work: Task<Void, Never>?
    private var observing = false
    private var foreground = true
    private var rescan = false
    private var classifier: LibraryClassifier?
    private let analyzer = VideoAnalyzer()
    private let stager = LibraryVideoStager()
    private static let enabledKey = "automaticPhotoImport"

    init(store: ClimbStore, defaults: UserDefaults = .standard, fetchVideos: (() -> PHFetchResult<PHAsset>)? = nil) {
        self.fetchVideos = fetchVideos
        self.store = store; self.defaults = defaults
        enabled = defaults.bool(forKey: Self.enabledKey)
        super.init()
    }
    var hasAccess: Bool { authorization == .authorized || authorization == .limited }

    func setEnabled(_ value: Bool) async {
        enabled = value
        defaults.set(value, forKey: Self.enabledKey)
        if value {
            if authorization == .notDetermined { authorization = await PHPhotoLibrary.requestAuthorization(for: .readWrite) }
            refresh()
        } else { stop(); updateObserver() }
    }
    func setForeground(_ value: Bool) {
        foreground = value
        if value { if running { rescan = true }; refresh() } else { stop() }
    }
    func refresh() {
        authorization = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        updateObserver()
        guard enabled, foreground, hasAccess else { stop(); return }
        scan()
    }
    private func updateObserver() {
        let needed = enabled && hasAccess
        if needed && !observing { PHPhotoLibrary.shared().register(self); observing = true }
        else if !needed && observing { PHPhotoLibrary.shared().unregisterChangeObserver(self); observing = false }
    }
    nonisolated func photoLibraryDidChange(_ changeInstance: PHChange) {
        Task { @MainActor [weak self] in
            guard let self else { return }
            if running { rescan = true } else { refresh() }
        }
    }
    func stop() { rescan = false; work?.cancel() }
    func scan() {
        guard enabled, hasAccess, foreground, !running, let store, !store.cameraOpen, !store.selectionOpen else { return }
        running = true; error = nil; checked = 0; imported = 0; unavailable = 0
        work = Task { [weak self] in
            guard let self else { return }
            defer {
                running = false; work = nil
                if rescan { rescan = false; refresh() }
            }
            do {
                if classifier == nil {
                    guard let pack = GymPack.current else { throw CocoaError(.fileNoSuchFile) }
                    classifier = try await Task.detached(priority: .utility) { try LibraryClassifier(pack: pack) }.value
                    try Task.checkCancellation()
                }
                guard let classifier else { return }
                let options = PHFetchOptions()
                options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
                let assets = fetchVideos?() ?? PHAsset.fetchAssets(with: .video, options: options)
                total = assets.count
                for index in 0..<assets.count {
                    try Task.checkCancellation()
                    guard enabled, foreground, !store.cameraOpen, !store.selectionOpen else { return }
                    authorization = PHPhotoLibrary.authorizationStatus(for: .readWrite)
                    guard hasAccess else { updateObserver(); return }
                    let asset = assets.object(at: index)
                    defer { checked += 1 }
                    let source = Self.sourceRevision(asset)
                    let revision = classifier.revision
                    if store.hasHandledAsset(asset.localIdentifier, source: source, classifier: revision) { continue }
                    guard asset.duration > 0, asset.duration <= 1200 else { unavailable += 1; continue }
                    do {
                        let video = try await LibraryAssetRequest.load(asset)
                        let tag = try await classifier.classify(video)
                        try Task.checkCancellation()
                        // The source may have been edited or removed while inference was running.
                        guard let current = PHAsset.fetchAssets(withLocalIdentifiers: [asset.localIdentifier], options: nil).firstObject,
                              Self.sourceRevision(current) == source else { rescan = true; continue }
                        guard let tag else {
                            try store.recordLibraryNonmatch(asset.localIdentifier, source: source, classifier: revision)
                            continue
                        }
                        let environment = ClimbEnvironment(rawValue: tag) ?? .unknown
                        let staged = try await stager.stage(video)
                        defer { try? FileManager.default.removeItem(at: staged) }
                        let summary = try await analyzer.videoSummary(url: staged)
                        let thumbnail = try await analyzer.thumbnail(url: staged)
                        try Task.checkCancellation()
                        guard PHPhotoLibrary.authorizationStatus(for: .readWrite) == .authorized || PHPhotoLibrary.authorizationStatus(for: .readWrite) == .limited,
                              let latest = PHAsset.fetchAssets(withLocalIdentifiers: [asset.localIdentifier], options: nil).firstObject,
                              Self.sourceRevision(latest) == source else { rescan = true; continue }
                        // Never request live location for a historical recording. Only use the
                        // asset's coordinates; missing / coarse / distant positions stay unassigned.
                        let gym = Self.suggestedGym(for: asset, environment: environment)
                        _ = try store.save(title: "", grade: ungradedValue, sent: false, note: "", stagedVideo: staged,
                            thumbnail: thumbnail, analysis: summary, gymId: gym?.id, gymName: gym?.name,
                            climbedAt: asset.creationDate, environment: environment, sourceAssetID: asset.localIdentifier,
                            libraryReceipt: LibraryReceipt(sourceRevision: source, classifierRevision: revision, imported: true))
                        imported += 1
                    } catch is CancellationError { throw CancellationError() }
                    catch {
                        // Keep failures retryable. A failed copy or cloud-only asset is not a negative classification.
                        unavailable += 1
                        if (error as NSError).code == NSFileWriteOutOfSpaceError { throw error }
                    }
                }
            } catch is CancellationError { }
            catch { self.error = String(localized: "自动收录暂时不可用，请稍后重试。", bundle: AppLanguage.bundle) }
        }
    }
    private static func sourceRevision(_ asset: PHAsset) -> String {
        [String((asset.modificationDate ?? asset.creationDate ?? .distantPast).timeIntervalSince1970),
         String(asset.duration), String(asset.pixelWidth), String(asset.pixelHeight)].joined(separator: ":")
    }
    private static func suggestedGym(for asset: PHAsset, environment: ClimbEnvironment) -> DirectoryGym? {
        guard environment == .indoor, let location = asset.location,
              location.horizontalAccuracy >= 0, location.horizontalAccuracy <= GymDistance.usefulAccuracy else { return nil }
        let nearby = GymDistance.suggestions(GymDirectory.bundled?.gyms ?? [], latitude: location.coordinate.latitude,
            longitude: location.coordinate.longitude, accuracy: location.horizontalAccuracy)
        // Saved and displayed as a suggestion, never as a confirmed filming location.
        return nearby.ranked.first?.gym
    }
}
