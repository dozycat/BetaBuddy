import XCTest
import AVFoundation
import UIKit

final class ImportSaveUITests: XCTestCase {
    @MainActor
    func testSingleSavePersistsAndCancelDoesNotSave() async throws {
        let clip = try await makeVideo(image: nil)
        defer { try? FileManager.default.removeItem(at: clip) }
        let app = XCUIApplication()
        app.launchArguments = ["-AppleLanguages", "(en)", "-AppleLocale", "en_US", "-resetLanguage", "YES", "-initialTab", "1", "-debugImportVideo", clip.path]
        app.launch()
        let title = "Save-" + UUID().uuidString.prefix(6)
        XCTAssertTrue(app.buttons["importClimb"].waitForExistence(timeout: 10))
        app.buttons["importClimb"].tap()
        XCTAssertTrue(app.textFields["climbTitle"].waitForExistence(timeout: 5))
        app.textFields["climbTitle"].tap()
        app.textFields["climbTitle"].typeText(String(title))
        let save = app.buttons["saveClimb"]
        XCTAssertTrue(save.wait(for: \.isEnabled, toEqual: true, timeout: 20))
        XCTAssertEqual(app.buttons.matching(identifier: "saveClimb").count, 1)
        XCTAssertFalse(app.buttons["importDone"].exists)
        XCTAssertFalse(app.buttons["startAnalysis"].exists)
        save.tap()
        XCTAssertTrue(app.buttons["importClimb"].waitForExistence(timeout: 20))
        XCTAssertTrue(app.staticTexts[String(title)].waitForExistence(timeout: 5))
        app.terminate(); app.launch()
        XCTAssertTrue(app.staticTexts[String(title)].waitForExistence(timeout: 10))
        XCTAssertEqual(app.staticTexts.matching(identifier: String(title)).count, 1)
        app.staticTexts[String(title)].tap()
        XCTAssertTrue(app.staticTexts["playbackClock"].waitForExistence(timeout: 10))
        let screenshot = XCTAttachment(screenshot: app.screenshot())
        screenshot.name = "Saved clip reopened after relaunch"; screenshot.lifetime = .keepAlways; add(screenshot)
        app.buttons["More actions"].tap()
        app.buttons["deleteClimb"].tap()
        let confirm = app.sheets.buttons["Delete this climb"]
        XCTAssertTrue(confirm.waitForExistence(timeout: 5))
        confirm.tap()
        XCTAssertTrue(app.buttons["importClimb"].waitForExistence(timeout: 10))
        app.buttons["importClimb"].tap()
        XCTAssertTrue(app.textFields["climbTitle"].waitForExistence(timeout: 5))
        app.textFields["climbTitle"].tap()
        app.textFields["climbTitle"].typeText(String(title) + "-cancel")
        XCTAssertTrue(app.buttons["saveClimb"].wait(for: \.isEnabled, toEqual: true, timeout: 20))
        app.buttons["importCancel"].tap()
        XCTAssertTrue(app.buttons["importClimb"].waitForExistence(timeout: 10))
        app.terminate(); app.launch()
        XCTAssertTrue(app.buttons["importClimb"].waitForExistence(timeout: 10))
        XCTAssertFalse(app.staticTexts[String(title) + "-cancel"].exists)
    }
    private func makeVideo(image:UIImage?, transform:CGAffineTransform = .identity, creationDate: String? = nil, endTime: Double? = nil) async throws -> URL {
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
        if let endTime { writer.endSession(atSourceTime: CMTime(seconds: endTime, preferredTimescale: 600)) }
        input.markAsFinished()
        await writer.finishWriting()
        XCTAssertEqual(writer.status,.completed)
        return url
    }
}
