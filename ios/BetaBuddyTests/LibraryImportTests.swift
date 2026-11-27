import AVFoundation
import Photos
import XCTest
@testable import BetaBuddy

@MainActor
final class LibraryImportTests: XCTestCase {
    func testReceiptsSurviveDeletionAndChangedSourcesButNonmatchesCanBeRetried() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ClimbStore(root: root)
        try store.recordLibraryNonmatch("other", source: "v1", classifier: "a")
        XCTAssertTrue(store.hasHandledAsset("other", source: "v1", classifier: "a"))
        XCTAssertFalse(store.hasHandledAsset("other", source: "v2", classifier: "a"))
        XCTAssertFalse(store.hasHandledAsset("other", source: "v1", classifier: "b"))
        let source = root.appendingPathComponent("staged.mov")
        try Data([1, 2, 3]).write(to: source)
        let receipt = LibraryReceipt(sourceRevision: "v1", classifierRevision: "a", imported: true)
        let climb = try store.save(title: "", grade: ungradedValue, sent: false, note: "", stagedVideo: source,
            thumbnail: Data(), analysis: PoseAnalysis(frames: [], sampledFrames: 0, duration: 1, aspectRatio: 1),
            gymId: "synthetic", gymName: "Example", environment: .indoor, sourceAssetID: "fixture", libraryReceipt: receipt)
        XCTAssertEqual(climb.setting, .indoor)
        XCTAssertEqual(climb.gymAuto, true)
        try store.organize(climb, title: "", grade: ungradedValue, gymId: nil, gymName: nil,
                           climbedAt: climb.climbingDate, folderId: nil, environment: .outdoor)
        let changed = try XCTUnwrap(store.climbs.first)
        XCTAssertEqual(changed.setting, .outdoor)
        XCTAssertNil(changed.environmentAuto)
        XCTAssertNil(changed.gymAuto)
        XCTAssertEqual(JournalDayGroup.make(store.climbs).first?.id.environment, .outdoor)
        try store.delete(climb)
        let reopened = ClimbStore(root: root)
        XCTAssertTrue(reopened.hasHandledAsset("fixture", source: "v2", classifier: "b"))
        XCTAssertTrue(reopened.climbs.isEmpty)
        try Data([9]).write(to: source)
        XCTAssertThrowsError(try reopened.save(title: "", grade: ungradedValue, sent: false, note: "", stagedVideo: source,
            thumbnail: Data(), analysis: climb.analysis, sourceAssetID: "fixture", libraryReceipt: receipt))
        XCTAssertEqual(try Data(contentsOf: source), Data([9]))
        // Explicit manual reimport is still allowed, and the user-chosen gym is not a suggestion.
        let manual = try reopened.save(title: "", grade: ungradedValue, sent: false, note: "", stagedVideo: source,
            thumbnail: Data(), analysis: climb.analysis, gymId: "chosen", sourceAssetID: "fixture")
        XCTAssertNil(manual.gymAuto)
        XCTAssertTrue(reopened.hasHandledAsset("fixture", source: "v3", classifier: "c"))
    }
    func testFailedSaveDoesNotConsumeAssetAndRestoresStagedVideo() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ClimbStore(root: root)
        // Force the atomic index replacement to fail after media files were moved.
        try FileManager.default.createDirectory(at: root.appendingPathComponent("archive.json"), withIntermediateDirectories: true)
        let source = root.appendingPathComponent("staged.mov")
        try Data([4, 5]).write(to: source)
        XCTAssertThrowsError(try store.save(title: "", grade: ungradedValue, sent: false, note: "", stagedVideo: source,
            thumbnail: Data(), analysis: PoseAnalysis(frames: [], sampledFrames: 0, duration: 1, aspectRatio: 1),
            sourceAssetID: "fixture", libraryReceipt: LibraryReceipt(sourceRevision: "v1", classifierRevision: "a", imported: true)))
        XCTAssertFalse(store.hasHandledAsset("fixture", source: "v1", classifier: "a"))
        XCTAssertTrue(store.climbs.isEmpty)
        XCTAssertEqual(try Data(contentsOf: source), Data([4, 5]))
    }
    func testGenericPolicyRejectsTiesInvalidVectorsAndInsufficientTemporalEvidence() throws {
        let policy = LibraryClassifierConfig(schema: 1, model: "LibraryImageEncoder", samples: 5, minimumVotes: 3,
            minimumScore: 0.5, minimumMargin: 0.1,
            prototypes: [.init(tag: "a", vector: [1, 0]), .init(tag: "b", vector: [0, 1]), .init(tag: nil, vector: [-1, 0])])
        try policy.validate()
        XCTAssertEqual(policy.tag(for: [1, 0]), "a")
        XCTAssertNil(policy.tag(for: [0.7, 0.7]))
        XCTAssertNil(policy.tag(for: [-1, 0]))
        XCTAssertNil(policy.tag(for: [Double.nan, 0]))
        XCTAssertNil(policy.decision(["a", "a", nil, "b", nil]))
        XCTAssertEqual(policy.decision(["a", "a", nil, "a", "b"]), "a")
        XCTAssertEqual(policy.frameVote(["a", "a"]), "a")
        XCTAssertEqual(policy.frameVote([nil, "b"]), "b")
        XCTAssertNil(policy.frameVote(["a", "b"]))
        XCTAssertNil(policy.decision([policy.frameVote(["a", "a"]), nil, nil, nil, nil]))
        var invalid = policy; invalid.schema = 2
        XCTAssertThrowsError(try invalid.validate())
        invalid.imageResize = "centerAndFit"; XCTAssertNoThrow(try invalid.validate())
        XCTAssertEqual(invalid.imageModes.count, 2)
        invalid.imageResize = "unsupported"; XCTAssertThrowsError(try invalid.validate())
    }

    /// Optional labelled media matrix; the attachment records misses as well as matches.
    func testOptionalClassifierVideoMatrix() async throws {
        guard let path = ProcessInfo.processInfo.environment["BB_LIBRARY_MATRIX"] else { throw XCTSkip("Local evaluation matrix not supplied") }
        let input = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path))) as? [String: Any])
        let clips = try XCTUnwrap(input["clips"] as? [[String: Any]])
        XCTAssertFalse(clips.isEmpty)
        let engine = try LibraryClassifier(pack: XCTUnwrap(GymPack.current))
        var rows: [[String: Any]] = []
        for clip in clips {
            let id = try XCTUnwrap(clip["id"] as? String), file = try XCTUnwrap(clip["path"] as? String)
            let expected = clip["expected"] as? String, start = Date()
            do {
                let actual = try await engine.classify(AVURLAsset(url: URL(fileURLWithPath: file)))
                rows.append(["id": id, "expected": expected as Any? ?? NSNull(), "actual": actual as Any? ?? NSNull(), "seconds": Date().timeIntervalSince(start)])
                XCTAssertEqual(actual, expected, "Evaluation clip \(id)")
            } catch {
                rows.append(["id": id, "error": error.localizedDescription]); XCTFail("Evaluation clip \(id): \(error)")
            }
        }
        let data = try JSONSerialization.data(withJSONObject: ["revision": engine.revision, "clips": rows], options: [.prettyPrinted,.sortedKeys])
        let attachment = XCTAttachment(data: data, uniformTypeIdentifier: "public.json")
        attachment.name = "Native library classifier matrix"; attachment.lifetime = .keepAlways; add(attachment)
    }

    /// Real model and real PhotoKit; optional because the private pack and footage are not in git.
    /// The simulator must already grant Photos permission. Keeps one reusable test asset in the simulator, never mutating existing library items.
    func testInstalledClassifierImportsPhotoAssetAndDoesNotReimportDeletedRecord() async throws {
        #if targetEnvironment(simulator)
        guard let path = ProcessInfo.processInfo.environment["BB_LIBRARY_CLIP"] else { throw XCTSkip("Private positive clip not supplied") }
        guard PHPhotoLibrary.authorizationStatus(for: .readWrite) == .authorized else { throw XCTSkip("Grant Photos permission to the simulator app first") }
        let pack = try XCTUnwrap(GymPack.current)
        let engine = try LibraryClassifier(pack: pack)
        let direct = try await engine.classify(AVURLAsset(url: URL(fileURLWithPath: path)))
        XCTAssertNotNil(direct, "Real positive must pass before PhotoKit testing")
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let suite = "library-tests-\(UUID())"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        let store = ClimbStore(root: root, defaults: defaults)
        var assetID = UserDefaults.standard.string(forKey: "libraryIntegrationFixture")
        let recorded = Date(timeIntervalSince1970: 1_700_000_000)
        if assetID == nil || PHAsset.fetchAssets(withLocalIdentifiers: [assetID!], options: nil).count == 0 {
        try await PHPhotoLibrary.shared().performChanges {
            let request = PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: URL(fileURLWithPath: path))!
            request.creationDate = recorded
            assetID = request.placeholderForCreatedAsset?.localIdentifier
        }
        }
        let id = try XCTUnwrap(assetID)
        UserDefaults.standard.set(id, forKey: "libraryIntegrationFixture")
        let session = PhotoLibraryImport(store: store, defaults: defaults, fetchVideos: { PHAsset.fetchAssets(withLocalIdentifiers: [id], options: nil) })
        do {
            await session.setEnabled(true)
            try await waitForScan(session)
            XCTAssertNil(session.error)
            let climb = try XCTUnwrap(store.climbs.first { $0.sourceAssetID == id }, "checked=\(session.checked), unavailable=\(session.unavailable), ledger=\(store.archive.libraryLedger.count)")
            XCTAssertEqual(climb.climbedAt?.timeIntervalSince1970 ?? 0, recorded.timeIntervalSince1970, accuracy: 1)
            XCTAssertNotEqual(climb.setting, .unknown)
            session.scan()
            session.stop()
            try await waitForScan(session)
            XCTAssertEqual(store.climbs.filter { $0.sourceAssetID == id }.count, 1)
            XCTAssertTrue(FileManager.default.fileExists(atPath: store.file(climb.videoName).path))
            try store.delete(climb)
            session.scan()
            try await waitForScan(session)
            XCTAssertFalse(store.climbs.contains { $0.sourceAssetID == id })
            await session.setEnabled(false)
        } catch {
            await session.setEnabled(false)
            defaults.removePersistentDomain(forName: suite)
            try? FileManager.default.removeItem(at: root)
            throw error
        }
        defaults.removePersistentDomain(forName: suite)
        try? FileManager.default.removeItem(at: root)
        #else
        throw XCTSkip("Photo-library fixture only runs in the simulator")
        #endif
    }
    private func waitForScan(_ session: PhotoLibraryImport) async throws {
        let end = Date().addingTimeInterval(180)
        while session.running && Date() < end { try await Task.sleep(for: .milliseconds(100)) }
        XCTAssertFalse(session.running, "Scan did not finish")
    }
}
