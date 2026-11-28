import AVFoundation
import Photos
import ImageIO
import Vision
import XCTest
@testable import BetaBuddy

@MainActor
final class LibraryImportTests: XCTestCase {
    func testOptionalEncoderImageParity() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let manifest = env["BB_ENCODER_IMAGES"], let output = env["BB_ENCODER_OUTPUT"] else {
            throw XCTSkip("Optional common decoded images not supplied")
        }
        struct Input: Decodable { var id: String; var path: String; var prepared: Bool; var expected: [Double]; var views: [String: [Double]]? }
        let inputs = try JSONDecoder().decode([Input].self, from: Data(contentsOf: URL(fileURLWithPath: manifest)))
        XCTAssertFalse(inputs.isEmpty)
        let engine = try LibraryClassifier(pack: XCTUnwrap(GymPack.current))
        var rows: [[String: Any]] = []
        for input in inputs {
            let source = try XCTUnwrap(CGImageSourceCreateWithURL(URL(fileURLWithPath: input.path) as CFURL, nil))
            let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
            let modes: [(String, VNImageCropAndScaleOption)] = input.prepared ? [("prepared", .scaleFill)] : [("center", .centerCrop), ("fit", .scaleFit)]
            for (name, mode) in modes {
                let vector = try await engine.embedding(image, mode: mode)
                XCTAssertEqual(vector.count, input.expected.count)
                XCTAssertEqual(vector.reduce(0) { $0+$1*$1 }, 1, accuracy: 0.005)
                let expected = input.views?[name] ?? input.expected
                let cosine = zip(vector, expected).reduce(0) { $0+$1.0*$1.1 }
                rows.append(["id": input.id, "mode": name, "embedding": vector, "cosine": cosine])
                XCTAssertGreaterThan(cosine, 0.999, "Native sampling and encoder must agree: \(input.id), \(name)")
            }
        }
        try JSONSerialization.data(withJSONObject: rows, options: [.sortedKeys]).write(to: URL(fileURLWithPath: output))
    }
    func testImageSamplingAveragesFineDetailAndKeepsPaddingColourAndOrientation() throws {
        let width = 448, height = 896, space = CGColorSpace(name: CGColorSpace.sRGB)!
        func image(checker: Bool) throws -> CGImage {
            var rgba = [UInt8](repeating: 0, count: width*height*4)
            for y in 0..<height { for x in 0..<width {
                let i = (y*width+x)*4
                if checker { let value: UInt8 = (x+y)%2 == 0 ? 0 : 255; rgba[i] = value; rgba[i+1] = value; rgba[i+2] = value }
                else { rgba[i+(y < height/2 ? 0 : 2)] = 255 }
                rgba[i+3] = 255
            } }
            return try XCTUnwrap(CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width*4,
                space: space, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: CGDataProvider(data: Data(rgba) as CFData)!, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        }
        let sampled = try LibraryClassifier.sampledImage(image(checker: true), mode: .scaleFit)
        let pixels = try XCTUnwrap(sampled.dataProvider?.data) as Data
        for channel in 0..<3 { XCTAssertEqual(Int(pixels[(112*224+112)*4+channel]), 128, accuracy: 1) }
        XCTAssertEqual(pixels[(112*224)*4], 0)
        let colour = try LibraryClassifier.sampledImage(image(checker: false), mode: .scaleFit)
        let colours = try XCTUnwrap(colour.dataProvider?.data) as Data
        XCTAssertEqual(colours[(10*224+112)*4], 255); XCTAssertEqual(colours[(10*224+112)*4+2], 0)
        XCTAssertEqual(colours[(210*224+112)*4], 0); XCTAssertEqual(colours[(210*224+112)*4+2], 255)
    }
    /// Run only on a dedicated simulator seeded by simctl, never on a personal library.
    func testOptionalSystemPhotoScopeAndImport() async throws {
        #if targetEnvironment(simulator)
        let env = ProcessInfo.processInfo.environment
        guard let mode = env["BB_PHOTOS_SCOPE"] else { throw XCTSkip("Dedicated simulator photo scope not supplied") }
        let fetched = PHAsset.fetchAssets(with: .video, options: nil)
        let videos = (0..<fetched.count).map { fetched.object(at: $0) }
        let authorization = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        if mode == "inventory" {
            XCTAssertEqual(authorization, .authorized)
            XCTAssertEqual(videos.count, 3, "Use a clean simulator with exactly the three supplied fixtures")
            let items = videos.map { ["id": $0.localIdentifier, "duration": $0.duration] as [String: Any] }
            let data = try JSONSerialization.data(withJSONObject: items, options: [.prettyPrinted, .sortedKeys])
            let path = try XCTUnwrap(env["BB_PHOTOS_INVENTORY"])
            try data.write(to: URL(fileURLWithPath: path))
            return
        }
        let allIDs = try XCTUnwrap(env["BB_PHOTOS_ALL_IDS"]).split(separator: ",").map(String.init)
        let expected = Set((env["BB_PHOTOS_ALLOWED_IDS"] ?? "").split(separator: ",").map(String.init))
        XCTAssertEqual(authorization, mode == "revoked" ? .denied : .limited)
        XCTAssertEqual(Set(videos.map(\.localIdentifier)), expected)
        for id in allIDs {
            XCTAssertEqual(PHAsset.fetchAssets(withLocalIdentifiers: [id], options: nil).count, expected.contains(id) ? 1 : 0)
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let suite = "photo-scope-\(UUID())"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        let store = ClimbStore(root: root, defaults: defaults)
        let session = PhotoLibraryImport(store: store, defaults: defaults)
        if mode == "revoked" {
            let staged = root.appendingPathComponent("retained.mov")
            try Data([1, 2, 3]).write(to: staged)
            let saved = try store.save(title: "Retained", grade: ungradedValue, sent: false, note: "", stagedVideo: staged,
                thumbnail: Data(), analysis: PoseAnalysis(frames: [], sampledFrames: 0, duration: 1, aspectRatio: 1))
            await session.setEnabled(true)
            XCTAssertFalse(session.hasAccess); XCTAssertFalse(session.running)
            session.setForeground(false); session.setForeground(true)
            XCTAssertFalse(session.running)
            XCTAssertEqual(store.climbs.map(\.id), [saved.id])
            XCTAssertEqual(try Data(contentsOf: store.file(saved.videoName)), Data([1, 2, 3]))
            await session.setEnabled(false)
            return
        }
        let positiveID = try XCTUnwrap(env["BB_PHOTOS_POSITIVE_ID"])
        XCTAssertTrue(expected.contains(positiveID))
        await session.setEnabled(true)
        do {
            try await waitForScan(session)
            XCTAssertNil(session.error); XCTAssertEqual(session.unavailable, 0)
            XCTAssertEqual(session.total, expected.count)
            XCTAssertEqual(session.imported, 1)
            XCTAssertEqual(store.climbs.compactMap(\.sourceAssetID), [positiveID])
            let climb = try XCTUnwrap(store.climbs.first)
            XCTAssertEqual(climb.setting, .indoor)
            XCTAssertTrue(FileManager.default.fileExists(atPath: store.file(climb.videoName).path))
            let date = try XCTUnwrap(videos.first { $0.localIdentifier == positiveID }?.creationDate)
            XCTAssertEqual(climb.climbedAt, date)
            session.setForeground(false); session.setForeground(true)
            try await waitForScan(session)
            XCTAssertEqual(session.imported, 1)
            XCTAssertEqual(store.climbs.compactMap(\.sourceAssetID), [positiveID])
            XCTAssertEqual(Set(store.archive.libraryLedger.keys), expected)
            await session.setEnabled(false)
        } catch { await session.setEnabled(false); throw error }
        #else
        throw XCTSkip("System permission fixtures only run in a simulator")
        #endif
    }
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
