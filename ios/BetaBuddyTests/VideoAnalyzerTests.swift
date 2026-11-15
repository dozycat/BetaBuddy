import XCTest
import AVFoundation
import UIKit
import CoreImage
@testable import BetaBuddy

final class VideoAnalyzerTests: XCTestCase {
    private func makeVideo(image:UIImage?, transform:CGAffineTransform = .identity, creationDate: String? = nil) async throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("smoke-\(UUID()).mov")
        let width = 640, height = 426
        let writer = try AVAssetWriter(outputURL:url,fileType:.mov)
        if let creationDate {
            let metadata = AVMutableMetadataItem()
            metadata.identifier = .quickTimeMetadataCreationDate
            metadata.value = creationDate as NSString
            writer.metadata = [metadata]
        }
        let input = AVAssetWriterInput(mediaType:.video,outputSettings:[AVVideoCodecKey:AVVideoCodecType.h264,AVVideoWidthKey:width,AVVideoHeightKey:height])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput:input,sourcePixelBufferAttributes:[kCVPixelBufferPixelFormatTypeKey as String:kCVPixelFormatType_32ARGB,kCVPixelBufferWidthKey as String:width,kCVPixelBufferHeightKey as String:height,kCVPixelBufferCGImageCompatibilityKey as String:true,kCVPixelBufferCGBitmapContextCompatibilityKey as String:true])
        input.transform = transform
        writer.add(input)
        XCTAssertTrue(writer.startWriting())
        writer.startSession(atSourceTime:.zero)
        for frame in 0..<30 {
            while !input.isReadyForMoreMediaData {
                if writer.status == .failed { throw writer.error! }
                try await Task.sleep(for:.milliseconds(10))
            }
            var pixel:CVPixelBuffer?
            let pool = try XCTUnwrap(adaptor.pixelBufferPool)
            XCTAssertEqual(CVPixelBufferPoolCreatePixelBuffer(nil,pool,&pixel),kCVReturnSuccess)
            let buffer = try XCTUnwrap(pixel)
            CVPixelBufferLockBaseAddress(buffer,[])
            let context = try XCTUnwrap(CGContext(data:CVPixelBufferGetBaseAddress(buffer),width:width,height:height,bitsPerComponent:8,bytesPerRow:CVPixelBufferGetBytesPerRow(buffer),space:CGColorSpaceCreateDeviceRGB(),bitmapInfo:CGImageAlphaInfo.noneSkipFirst.rawValue))
            context.setFillColor(UIColor.black.cgColor)
            context.fill(CGRect(x:0,y:0,width:width,height:height))
            if let cgImage = image?.cgImage { context.draw(cgImage,in:CGRect(x:0,y:0,width:width,height:height)) }
            CVPixelBufferUnlockBaseAddress(buffer,[])
            XCTAssertTrue(adaptor.append(buffer,withPresentationTime:CMTime(value:Int64(frame),timescale:30)))
        }
        input.markAsFinished()
        await writer.finishWriting()
        XCTAssertEqual(writer.status,.completed)
        return url
    }
    @MainActor
    func testImportWithoutAnalysisUsesRecordingTimeAndKeepsPlayableVideo() async throws {
        let url = try await makeVideo(image: nil, transform: CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: 426, ty: 0), creationDate: "2024-03-02T14:35:00+08:00")
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: url) }
        let model = ImportModel()
        model.replace(url)
        await model.readRecordingDate(for: url)
        XCTAssertTrue(model.hasRecordingDate)
        XCTAssertEqual(model.climbedAt, ISO8601DateFormatter().date(from: "2024-03-02T06:35:00Z"))
        let store = ClimbStore(root: root)
        model.run(store: store, title: "No person required", grade: "V2", sent: false, note: "Keep", quality: .balanced, gymId: nil, gymName: "", climbedAt: model.climbedAt, folderId: nil, analyze: false)
        for _ in 0..<1000 { if !model.busy { break }; try await Task.sleep(for: .milliseconds(20)) }
        XCTAssertFalse(model.busy)
        XCTAssertNil(model.error)
        let climb = try XCTUnwrap(model.saved)
        XCTAssertEqual(climb.analysis.sampledFrames, 0)
        XCTAssertEqual(climb.analysis.duration, 1, accuracy: 0.05)
        XCTAssertEqual(climb.analysis.aspectRatio, 426.0 / 640, accuracy: 0.01)
        XCTAssertEqual(climb.climbingDate, model.climbedAt)
        XCTAssertNil(climb.pendingAnalysis)
        XCTAssertTrue(FileManager.default.fileExists(atPath: store.file(climb.videoName).path))
    }

    @MainActor
    func testBackgroundExpirationRetainsRequestAndResumePreservesEdits() async throws {
        let source = try XCTUnwrap(Bundle(for: Self.self).url(forResource: "pose", withExtension: "jpg"))
        let url = try await makeVideo(image: UIImage(contentsOfFile: source.path))
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root); try? FileManager.default.removeItem(at: url) }
        let store = ClimbStore(root: root)
        let summary = try await VideoAnalyzer().videoSummary(url: url)
        let climb = try store.save(title: "Queue", grade: "V1", sent: false, note: "", stagedVideo: url, thumbnail: Data(), analysis: summary, pendingAnalysis: "balanced")
        store.resumeAnalysis()
        store.pauseAnalysisForBackgroundExpiration()
        for _ in 0..<1000 { if store.analysisProgress[climb.id] == nil { break }; try await Task.sleep(for: .milliseconds(20)) }
        XCTAssertNil(store.analysisProgress[climb.id])
        XCTAssertEqual(ClimbStore(root: root).climbs.first?.pendingAnalysis, "balanced")
        store.resumeAnalysis()
        try store.update(climb, note: "Edited during analysis", sent: true)
        let editedTime = Date(timeIntervalSince1970: 1_700_000_000)
        try store.organize(climb, title: "Edited", grade: "V4", gymId: nil, gymName: nil, climbedAt: editedTime, folderId: nil)
        for _ in 0..<1000 { if store.analysisProgress[climb.id] == nil { break }; try await Task.sleep(for: .milliseconds(20)) }
        XCTAssertNil(store.analysisProgress[climb.id])
        let saved = try XCTUnwrap(ClimbStore(root: root).climbs.first)
        XCTAssertNil(saved.pendingAnalysis)
        XCTAssertGreaterThan(saved.analysis.sampledFrames, 0)
        XCTAssertEqual(saved.note, "Edited during analysis")
        XCTAssertTrue(saved.sent)
        XCTAssertEqual(saved.climbingDate, editedTime)
        XCTAssertEqual(saved.grade, "V4")
    }

    func testMissingRecordingDateDoesNotUseFileCreationTime() async throws {
        let url = try await makeVideo(image: nil)
        defer { try? FileManager.default.removeItem(at: url) }
        // AVAssetWriter writes a container creation time even without tagged metadata.
        // Zero the movie/track/media header timestamps to model a clip with no recorded date.
        var data = try Data(contentsOf: url)
        for atom in ["mvhd", "tkhd", "mdhd"] {
            var search = data.startIndex..<data.endIndex
            while let range = data.range(of: Data(atom.utf8), in: search) {
                let version = data[range.upperBound]
                let start = range.upperBound + 4
                let count = version == 1 ? 16 : 8
                data.replaceSubrange(start..<(start + count), with: repeatElement(UInt8(0), count: count))
                search = (start + count)..<data.endIndex
            }
        }
        try data.write(to: url)
        let date = await VideoAnalyzer().recordingDate(url: url)
        XCTAssertNil(date)
    }

    func testBundledModelProcessesVideoOffline() async throws {
        let source = try XCTUnwrap(Bundle(for:Self.self).url(forResource:"pose",withExtension:"jpg"))
        let image = try XCTUnwrap(UIImage(contentsOfFile:source.path))
        let url = try await makeVideo(image:image)
        defer { try? FileManager.default.removeItem(at:url) }
        let result = try await VideoAnalyzer().analyze(url:url,quality:.balanced) { _ in }
        XCTAssertGreaterThan(result.coverage,0.5)
        XCTAssertGreaterThan(result.frames.count,2)
        XCTAssertEqual(result.frames.first?.joints.count,33)
        XCTAssertEqual(result.aspectRatio,640.0/426,accuracy:0.01)
        XCTAssertLessThanOrEqual(result.sampledFrames,11)
    }
    // Compare the actual inference pixels with Apple's playback-oriented frame,
    // using asymmetric content so a 180-degree mismatch cannot pass.
    func testInferenceOrientationMatchesPlayback() async throws {
        let source = try XCTUnwrap(Bundle(for:Self.self).url(forResource:"pose",withExtension:"jpg"))
        let image = try XCTUnwrap(UIImage(contentsOfFile:source.path))
        let transforms: [CGAffineTransform] = [
            .identity,
            CGAffineTransform(a:0,b:1,c:-1,d:0,tx:426,ty:0),
            CGAffineTransform(a:0,b:-1,c:1,d:0,tx:0,ty:640),
            CGAffineTransform(a:-1,b:0,c:0,d:-1,tx:640,ty:426),
            CGAffineTransform(a:-1,b:0,c:0,d:1,tx:640,ty:0),
            CGAffineTransform(a:0,b:1,c:1,d:0,tx:0,ty:0)
        ]
        let context = CIContext()
        for transform in transforms {
            let url = try await makeVideo(image:image,transform:transform)
            defer { try? FileManager.default.removeItem(at:url) }
            let asset = AVURLAsset(url:url)
            let tracks = try await asset.loadTracks(withMediaType:.video)
            let track = try XCTUnwrap(tracks.first)
            let storedTransform = try await track.load(.preferredTransform)
            let reader = try AVAssetReader(asset:asset)
            let output = AVAssetReaderTrackOutput(track:track,outputSettings:[kCVPixelBufferPixelFormatTypeKey as String:kCVPixelFormatType_32BGRA])
            reader.add(output)
            XCTAssertTrue(reader.startReading())
            let sample = try XCTUnwrap(output.copyNextSampleBuffer())
            let buffer = try XCTUnwrap(CMSampleBufferGetImageBuffer(sample))
            let actual = try VideoAnalyzer.inferenceImage(buffer:buffer,transform:storedTransform,maxDimension:720,context:context)
            reader.cancelReading()
            let generator = AVAssetImageGenerator(asset:asset)
            generator.appliesPreferredTrackTransform = true
            generator.requestedTimeToleranceBefore = .zero
            generator.requestedTimeToleranceAfter = .zero
            let (expected, _) = try await generator.image(at:.zero)
            XCTAssertEqual(actual.width,expected.width)
            XCTAssertEqual(actual.height,expected.height)
            func pixels(_ image:CGImage) -> [UInt8] {
                var bytes = [UInt8](repeating:0,count:32*32*4)
                bytes.withUnsafeMutableBytes { raw in
                    let canvas = CGContext(data:raw.baseAddress,width:32,height:32,bitsPerComponent:8,bytesPerRow:128,space:CGColorSpaceCreateDeviceRGB(),bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue)!
                    canvas.draw(image,in:CGRect(x:0,y:0,width:32,height:32))
                }
                return bytes
            }
            let a = pixels(actual), b = pixels(expected)
            let meanError = zip(a,b).enumerated().filter { $0.offset % 4 != 3 }
                .map { abs(Double($0.element.0)-Double($0.element.1)) }.reduce(0,+) / Double(32*32*3)
            XCTAssertLessThan(meanError,8,"Orientation mismatch: \(transform), MAE \(meanError)")
        }
    }

    func testLocalRegressionVideo() async throws {
        let url = URL(fileURLWithPath:"/tmp/bb-regression.mp4")
        guard FileManager.default.fileExists(atPath:url.path) else { throw XCTSkip("Optional private regression video is not installed") }
        let asset = AVURLAsset(url:url)
        let tracks = try await asset.loadTracks(withMediaType:.video)
        let track = try XCTUnwrap(tracks.first)
        print("REGRESSION",try await track.load(.naturalSize),try await track.load(.preferredTransform))
        for quality in AnalysisQuality.allCases {
            let result = try await VideoAnalyzer().analyze(url:url,quality:quality) { _ in }
            print("REGRESSION",quality.rawValue,"COVERAGE",result.coverage)
            try JSONEncoder().encode(result).write(to:URL(fileURLWithPath:"/tmp/bb-regression-\(quality.rawValue).json"))
            // At the user's reported 9-second point, the hips are on the climber
            // near the middle of the frame, not the pipes near the top-right corner.
            let frame = try XCTUnwrap(result.nearestFrame(to:9))
            let hip = try XCTUnwrap(frame.hip)
            XCTAssertTrue((0.40...0.60).contains(hip.x))
            XCTAssertTrue((0.30...0.50).contains(hip.y))
            XCTAssertGreaterThan(result.coverage,0.45)
            XCTAssertTrue(result.frames.allSatisfy(\.hasReliableTorso))
        }
    }

    func testNoPersonReturnsActionableError() async throws {
        let url = try await makeVideo(image:nil)
        defer { try? FileManager.default.removeItem(at:url) }
        do {
            _ = try await VideoAnalyzer().analyze(url:url,quality:.balanced) { _ in }
            XCTFail("A blank video must not receive invented results")
        } catch AnalysisError.noPose { } catch { XCTFail("Unexpected error: \(error)") }
    }
    func testCancellationDoesNotReturnPartialAnalysis() async throws {
        let url = try await makeVideo(image:nil)
        defer { try? FileManager.default.removeItem(at:url) }
        let task = Task { try await VideoAnalyzer().analyze(url:url,quality:.detailed) { _ in } }
        task.cancel()
        do { _ = try await task.value; XCTFail("Cancelled analysis returned results") }
        catch is CancellationError { }
    }
}
