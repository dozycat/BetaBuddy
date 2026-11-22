import XCTest
import CoreImage
import CoreVideo
@testable import BetaBuddy

/// Gym ordering by distance, what the store keeps for a camera record, and the words for the
/// capture page's codes. Directory data here is synthetic.
@MainActor
final class CaptureSupportTests: XCTestCase {
    private var root: URL!
    override func setUp() { super.setUp(); root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString) }
    override func tearDown() { try? FileManager.default.removeItem(at: root); super.tearDown() }

    private func directory() throws -> GymDirectory { try JSONDecoder().decode(GymDirectory.self, from: Data(StandInPack.directory.utf8)) }

    func testDirectoryCoordinatesAreOptionalAndBadOnesDoNotBreakTheEntry() throws {
        let gyms = try directory().gyms
        XCTAssertEqual(gyms.count, 7)
        let east = try XCTUnwrap(gyms.first { $0.id == "syn:east" })
        XCTAssertEqual(east.lat, 30.0027); XCTAssertEqual(east.lon, 120.0); XCTAssertEqual(east.coordPrecision, "poi")
        let none = try XCTUnwrap(gyms.first { $0.id == "syn:nowhere" })
        XCTAssertNil(none.lat); XCTAssertNil(none.lon); XCTAssertNil(none.coordPrecision)
        let broken = try XCTUnwrap(gyms.first { $0.id == "syn:broken" })
        XCTAssertEqual(broken.name, "坐标写坏了的岩馆")
        XCTAssertNil(broken.lat); XCTAssertNil(broken.lon); XCTAssertNil(broken.coordPrecision)
        // The directory in this repository decodes with or without the new fields.
        XCTAssertFalse(try XCTUnwrap(GymDirectory.bundled).gyms.isEmpty)
    }

    func testNearbyGymsAreOrderedByDistanceAndOnlyExactOnesShowIt() throws {
        let gyms = try directory().gyms
        let here = StandInPack.origin
        let near = GymDistance.nearby(gyms, latitude: here.latitude, longitude: here.longitude, accuracy: 30)
        XCTAssertEqual(near.map(\.gym.id), ["syn:east", "syn:district", "syn:city", "syn:west"])       // 0.3, 1.1, 2.9 and 4.8 km; the far one, and those without usable coordinates, are left out.
        XCTAssertEqual(near[0].meters, 300, accuracy: 5)
        XCTAssertEqual(near.map(\.showsDistance), [true, false, false, true])                           // poi, district, city, address.
        // With only an approximate position the order stays and no number is shown.
        let coarse = GymDistance.nearby(gyms, latitude: here.latitude, longitude: here.longitude, accuracy: 5_000)
        XCTAssertEqual(coarse.map(\.gym.id), near.map(\.gym.id))
        XCTAssertFalse(coarse.contains(where: \.showsDistance))
        XCTAssertEqual(GymDistance.nearby(gyms, latitude: here.latitude, longitude: here.longitude, accuracy: 30, limit: 2).map(\.gym.id), ["syn:east", "syn:district"])
        // From the other side of town the order turns round.
        XCTAssertEqual(GymDistance.nearby(gyms, latitude: 30.0, longitude: 119.94, accuracy: 30).first?.gym.id, "syn:west")
        // Somewhere with nothing within reach, and positions that are not positions.
        XCTAssertTrue(GymDistance.nearby(gyms, latitude: 10, longitude: 10, accuracy: 30).isEmpty)
        XCTAssertTrue(GymDistance.nearby(gyms, latitude: .nan, longitude: 120, accuracy: 30).isEmpty)
        XCTAssertTrue(GymDistance.nearby(gyms, latitude: 95, longitude: 120, accuracy: 30).isEmpty)
        XCTAssertEqual(GymDistance.meters(fromLatitude: 0, longitude: 0, toLatitude: 0, longitude: 1), 111_195, accuracy: 20)
        XCTAssertFalse(CaptureText.distance(320).isEmpty)
    }

    func testAGymIsOnlyTakenForAPackVenueWhenItsNameSaysSo() throws {
        let directory = root.appendingPathComponent("pack")
        try StandInPack.write(to: directory)
        let manifest = try GymPack(directory: directory.appendingPathComponent(GymPack.directoryName)).manifest
        XCTAssertEqual(manifest.venueID(forGymNamed: StandInPack.directoryGymName), StandInPack.venueID)       // Brand and branch both in the venue's name.
        XCTAssertEqual(manifest.venueID(forGymNamed: StandInPack.venueName), StandInPack.venueID)
        XCTAssertEqual(manifest.venueID(forGymNamed: "Synthetic Gym East"), StandInPack.venueID)               // One of its names in another language.
        XCTAssertNil(manifest.venueID(forGymNamed: "合成岩馆（西门）"))                                            // Same brand, another branch.
        XCTAssertNil(manifest.venueID(forGymNamed: "合成岩馆"))                                                   // A brand alone is not a branch.
        XCTAssertNil(manifest.venueID(forGymNamed: "")); XCTAssertNil(manifest.venueID(forGymNamed: nil))
        XCTAssertTrue(try GymPack(directory: directory.appendingPathComponent(GymPack.directoryName)).supportsCapture)
        XCTAssertEqual(try GymPack(directory: directory.appendingPathComponent(GymPack.directoryName)).captureURL(language: "ja")?.absoluteString, "gympack://local/capture.html?lang=ja")
        let bare = root.appendingPathComponent("bare")
        try StandInPack.write(to: bare, capture: false)
        let without = try GymPack(directory: bare.appendingPathComponent(GymPack.directoryName))
        XCTAssertFalse(without.supportsCapture); XCTAssertNil(without.captureURL(language: "en"))
    }

    private func staged() throws -> URL {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let url = root.appendingPathComponent("staged-\(UUID()).mov")
        try Data([0, 1, 2]).write(to: url)
        return url
    }
    private let playback = PoseAnalysis(frames: [], sampledFrames: 0, duration: 3, aspectRatio: 0.5625)

    func testPlanAndCaptureInfoPersistNextToTheClipAndGoWithIt() throws {
        let store = ClimbStore(root: root)
        let plan = "{\"v\":1,\"note\":\"引号 \\\" 和\\n换行\"}"
        let info = CaptureInfo(colorKey: "yellow", colorRGB: [226, 196, 40], summary: CaptureSummary(color: "yellow", holds: 4), packVersion: "9.9.9", analysisQueued: true)
        let camera = try store.save(title: "", grade: "V5", sent: false, note: "", stagedVideo: staged(), thumbnail: Data(), analysis: playback,
                                    gymId: "syn:east", gymName: "合成岩馆（东门）", climbedAt: Date(timeIntervalSince1970: 5000), pendingAnalysis: "balanced", capture: info, capturePlan: plan)
        let imported = try store.save(title: "imported", grade: "V1", sent: false, note: "", stagedVideo: staged(), thumbnail: Data(), analysis: playback, climbedAt: Date(timeIntervalSince1970: 4000))
        XCTAssertTrue(FileManager.default.fileExists(atPath: store.file("\(camera.id)" + ClimbStore.capturePlanSuffix).path))

        let reopened = ClimbStore(root: root)
        let record = try XCTUnwrap(reopened.climbs.first { $0.id == camera.id })
        XCTAssertEqual(record.capture?.colorKey, "yellow"); XCTAssertEqual(record.capture?.hasPlan, true); XCTAssertEqual(record.capture?.analysisQueued, true)
        XCTAssertEqual(record.pendingAnalysis, "balanced")
        XCTAssertEqual(reopened.capturePlan(for: record), plan)                       // Byte for byte.
        XCTAssertNil(reopened.capturePlan(for: imported)); XCTAssertNil(reopened.climbs.first { $0.id == imported.id }?.capture)
        XCTAssertEqual(reopened.lastGym?.id, "syn:east")                               // The most recent record with a gym.

        // The plan reaches the analysis page as a JavaScript string literal, whatever is in it.
        let begin = AnalysisCommand.begin(id: record.id.uuidString, width: 1080, height: 1920, climber: nil, capture: reopened.capturePlan(for: record)).script
        let quoted = try XCTUnwrap(String(data: JSONEncoder().encode(plan), encoding: .utf8))
        XCTAssertTrue(begin.contains("\"capture\":" + quoted), begin)
        XCTAssertFalse(AnalysisCommand.begin(id: "a", width: 1, height: 1, climber: nil, capture: nil).script.contains("capture"))
        XCTAssertFalse(AnalysisCommand.begin(id: "a", width: 1, height: 1, climber: nil, capture: "").script.contains("capture"))
        XCTAssertFalse(AnalysisCommand.begin(id: "a", width: 1, height: 1, climber: nil).script.contains("capture"))

        // A fuller plan that arrives later replaces it; an imported record never gets one.
        try reopened.saveCapturePlan("{\"v\":2}", summary: CaptureSummary(color: "yellow", holds: 5), for: record.id)
        try reopened.saveCapturePlan("{\"v\":2}", summary: nil, for: imported.id)
        XCTAssertEqual(ClimbStore(root: root).capturePlan(for: record), "{\"v\":2}")
        XCTAssertEqual(ClimbStore(root: root).climbs.first { $0.id == record.id }?.capture?.summary?.holds, 5)
        XCTAssertNil(ClimbStore(root: root).capturePlan(for: imported))

        try reopened.setAnalysisQueued(record.id, false)
        XCTAssertNil(ClimbStore(root: root).climbs.first { $0.id == record.id }?.capture?.analysisQueued)
        try reopened.setAnalysisQueued(imported.id, true)                              // Not a camera record: nothing to mark.
        XCTAssertNil(ClimbStore(root: root).climbs.first { $0.id == imported.id }?.capture)

        try reopened.delete(record)
        XCTAssertFalse(FileManager.default.fileExists(atPath: store.file("\(camera.id)" + ClimbStore.capturePlanSuffix).path))
        XCTAssertEqual(ClimbStore(root: root).climbs.map(\.id), [imported.id])
    }

    func testACameraRecordWithoutAPlanAndAnArchiveFromBeforeTheCamera() throws {
        let store = ClimbStore(root: root)
        let record = try store.save(title: "", grade: ungradedValue, sent: false, note: "", stagedVideo: staged(), thumbnail: Data(), analysis: playback, capture: CaptureInfo(hasPlan: true))
        XCTAssertEqual(record.capture?.hasPlan, false)            // Only true when a plan was really written.
        XCTAssertNil(store.capturePlan(for: record))
        // An archive as 1.6.0 wrote it, read by this build and written back.
        let old = root.appendingPathComponent("old")
        try FileManager.default.createDirectory(at: old, withIntermediateDirectories: true)
        let id = "7F9619FF-8B86-D011-B42D-00C04FC964FF"
        try Data("""
        {"version":1,"folders":[{"id":"11111111-2222-3333-4444-555555555555","name":"Projects"}],"climbs":[{"id":"\(id)","date":700000000,"title":"old","grade":"V2","sent":true,"note":"n",
          "videoName":"\(id).mov","thumbnailName":"\(id).jpg","gymName":"Hand typed","folderId":"11111111-2222-3333-4444-555555555555","pendingAnalysis":"balanced",
          "analysis":{"frames":[],"sampledFrames":0,"duration":12.5,"aspectRatio":0.5625}}]}
        """.utf8).write(to: old.appendingPathComponent("archive.json"))
        let loaded = ClimbStore(root: old)
        XCTAssertNil(loaded.error)
        let climb = try XCTUnwrap(loaded.climbs.first)
        XCTAssertNil(climb.capture); XCTAssertNil(loaded.capturePlan(for: climb))
        XCTAssertEqual(climb.pendingAnalysis, "balanced"); XCTAssertEqual(loaded.folders.first?.name, "Projects")
        XCTAssertEqual(loaded.lastGym?.name, "Hand typed"); XCTAssertNil(loaded.lastGym?.id)
        try loaded.update(climb, note: "still here", sent: false)
        let again = ClimbStore(root: old)
        XCTAssertEqual(again.climbs.first?.note, "still here"); XCTAssertNil(again.climbs.first?.capture)
    }

    func testAnalysisWaitsWhileTheCameraIsInUse() throws {
        let store = ClimbStore(root: root)
        _ = try store.save(title: "", grade: ungradedValue, sent: false, note: "", stagedVideo: staged(), thumbnail: Data(), analysis: playback, pendingAnalysis: "balanced")
        store.pauseAnalysisForCamera()
        XCTAssertTrue(store.cameraOpen)
        store.resumeAnalysis()                                    // The app becoming active does not override the camera.
        XCTAssertTrue(store.analysisProgress.isEmpty)
        XCTAssertEqual(ClimbStore(root: root).climbs.first?.pendingAnalysis, "balanced")
        store.resumeAnalysis(cameraClosed: true)
        XCTAssertFalse(store.cameraOpen)
        XCTAssertFalse(store.analysisProgress.isEmpty)
        store.pauseAnalysisForCamera()
    }

    /// Red over blue, in the camera's format and in the file source's.
    private func twoTone(_ format: OSType, width: Int, height: Int) throws -> CVPixelBuffer {
        var made: CVPixelBuffer?
        XCTAssertEqual(CVPixelBufferCreate(nil, width, height, format, [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &made), kCVReturnSuccess)
        let buffer = try XCTUnwrap(made)
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        if format == kCVPixelFormatType_32BGRA {
            let base = try XCTUnwrap(CVPixelBufferGetBaseAddress(buffer)).assumingMemoryBound(to: UInt8.self)
            let stride = CVPixelBufferGetBytesPerRow(buffer)
            for y in 0..<height { for x in 0..<width {
                let pixel = base + y * stride + x * 4
                pixel[0] = y < height / 2 ? 0 : 255; pixel[1] = 0; pixel[2] = y < height / 2 ? 255 : 0; pixel[3] = 255
            } }
        } else {
            // BT.709, full range: red is (54, 99, 255), blue is (18, 255, 116).
            let luma = try XCTUnwrap(CVPixelBufferGetBaseAddressOfPlane(buffer, 0)).assumingMemoryBound(to: UInt8.self)
            let lumaStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
            for y in 0..<height { memset(luma + y * lumaStride, y < height / 2 ? 54 : 18, width) }
            let chroma = try XCTUnwrap(CVPixelBufferGetBaseAddressOfPlane(buffer, 1)).assumingMemoryBound(to: UInt8.self)
            let chromaStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1)
            for y in 0..<height / 2 { for x in 0..<width / 2 {
                chroma[y * chromaStride + x * 2] = y < height / 4 ? 99 : 255
                chroma[y * chromaStride + x * 2 + 1] = y < height / 4 ? 255 : 116
            } }
        }
        return buffer
    }
    private func colour(_ image: CGImage, x: Double, y: Double) throws -> [Int] {
        var pixel = [UInt8](repeating: 0, count: 4)
        let context = try XCTUnwrap(CGContext(data: &pixel, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        // Core Graphics counts rows from the bottom.
        context.draw(image, in: CGRect(x: -x * Double(image.width), y: -(1 - y) * Double(image.height) + 1, width: Double(image.width), height: Double(image.height)))
        return pixel.prefix(3).map(Int.init)
    }

    func testFramesForThePageKeepTheirWayUpTheirColoursAndAShortSideOf360() throws {
        let context = CIContext()
        for format in [kCVPixelFormatType_32BGRA, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange] {
            let image = try XCTUnwrap(CapturePipeline.scaled(try twoTone(format, width: 1080, height: 1920), shortSide: 360, context: context))
            XCTAssertEqual(image.width, 360); XCTAssertEqual(image.height, 640)
            let top = try colour(image, x: 0.5, y: 0.2), bottom = try colour(image, x: 0.5, y: 0.8)
            XCTAssertGreaterThan(top[0], 220, "\(format) top \(top)"); XCTAssertLessThan(top[1], 40, "\(top)"); XCTAssertLessThan(top[2], 40, "\(top)")
            XCTAssertLessThan(bottom[0], 40, "\(format) bottom \(bottom)"); XCTAssertLessThan(bottom[1], 40, "\(bottom)"); XCTAssertGreaterThan(bottom[2], 220, "\(bottom)")
            // A landscape frame keeps its shape; a small one is not enlarged.
            let wide = try XCTUnwrap(CapturePipeline.scaled(try twoTone(format, width: 1920, height: 1080), shortSide: 360, context: context))
            XCTAssertEqual(wide.width, 640); XCTAssertEqual(wide.height, 360)
            let small = try XCTUnwrap(CapturePipeline.scaled(try twoTone(format, width: 180, height: 320), shortSide: 360, context: context))
            XCTAssertEqual(small.width, 180); XCTAssertEqual(small.height, 320)
        }
    }

    func testEveryCodeHasItsOwnWordsAndUnknownOnesTheGenericLine() {
        let generic = CaptureText.reason("lens_flare_2031")
        XCTAssertFalse(generic.isEmpty)
        XCTAssertEqual(CaptureText.reason(""), generic)
        let reasons = CaptureGuidance.Code.allCases.map { CaptureText.reason($0.rawValue) }
        XCTAssertFalse(reasons.contains(generic))
        XCTAssertEqual(Set(reasons).count, reasons.count)
        let actions = CaptureGuidance.Move.allCases.compactMap { CaptureText.action($0.rawValue) }
        XCTAssertEqual(Set(actions).count, CaptureGuidance.Move.allCases.count)
        XCTAssertNil(CaptureText.action("spin_around")); XCTAssertNil(CaptureText.action(nil))

        // Code and move make one line; a move without words leaves the reason alone.
        let both = CaptureText.line(.init(code: "cut_top", blocking: true, move: "back"))
        XCTAssertTrue(both.contains(CaptureText.reason("cut_top"))); XCTAssertTrue(both.contains(CaptureText.action("back")!))
        XCTAssertEqual(CaptureText.line(.init(code: "cut_top", blocking: true, move: nil)), CaptureText.reason("cut_top"))
        XCTAssertEqual(CaptureText.line(.init(code: "cut_top", blocking: true, move: "spin_around")), CaptureText.reason("cut_top"))
        XCTAssertEqual(CaptureText.line(.init(code: "lens_flare_2031", blocking: true, move: "spin_around")), generic)
        XCTAssertTrue(CaptureText.line(.init(code: "lens_flare_2031", blocking: true, move: "back")).contains(generic))

        // The same in every language the app has.
        for language in AppLanguage.supported {
            let table = AppLanguage.bundle(for: language)
            XCTAssertNotEqual(table.localizedString(forKey: "换个机位试试", value: "?", table: nil), "?", language)
            XCTAssertNotEqual(table.localizedString(forKey: "%@，%@", value: "?", table: nil), "?", language)
        }
        XCTAssertEqual(AppLanguage.bundle(for: "en").localizedString(forKey: "换个机位试试", value: nil, table: nil), "Try another camera position")
        XCTAssertEqual(String(format: AppLanguage.bundle(for: "en").localizedString(forKey: "%@，%@", value: nil, table: nil), "The top is cut off", "step back a little"),
                       "The top is cut off — step back a little")

        XCTAssertNotNil(CaptureText.colorName("yellow")); XCTAssertEqual(CaptureText.colorName("GREY"), CaptureText.colorName("gray"))
        XCTAssertNil(CaptureText.colorName("teal-ish")); XCTAssertNil(CaptureText.colorName(nil))
        XCTAssertEqual(Set(CaptureRole.allCases.map(CaptureText.role)).count, 5)
        XCTAssertNil(CaptureText.stop(.user)); XCTAssertNil(CaptureText.stop(nil))
        for reason in [CaptureStop.limit, .storage, .interrupted, .overheated] { XCTAssertNotNil(CaptureText.stop(reason)) }
    }
}
