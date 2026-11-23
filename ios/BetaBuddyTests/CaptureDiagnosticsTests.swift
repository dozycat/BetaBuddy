#if DEBUG
import XCTest
import UIKit
@testable import BetaBuddy

/// The Debug-only diagnostics folder: line format, one file per visit, the size cap, the log file,
/// and the picture of the screen when the preview itself does not come out. All data is synthetic.
final class CaptureDiagnosticsTests: XCTestCase {
    private var root: URL!
    override func setUp() { super.setUp(); root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString) }
    override func tearDown() { try? FileManager.default.removeItem(at: root); super.tearDown() }

    private func lines(_ url: URL) throws -> [[String: Any]] {
        try String(contentsOf: url, encoding: .utf8).split(separator: "\n").map {
            try XCTUnwrap(JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any])
        }
    }
    private func size(_ url: URL) -> Int { (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0 }
    private func names() -> [String] { ((try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? []).sorted() }

    func testALineIsOneJSONObjectWithTimeAndKind() throws {
        let nothing: Double? = nil
        let when = Date(timeIntervalSince1970: 1_800_000_000.25)
        let data = try XCTUnwrap(CaptureDiagnostics.line(kind: "sample", [
            "fovH": 68.123456, "pitch": Double.nan, "poseMs": nothing as Any, "route": Optional(["id": "r1", "conf": Optional(0.5) as Any]) as Any,
            "issues": [["code": "too_close", "severity": "block", "move": Optional<String>.none as Any]], "ok": true, "holds": 12, "width": CGFloat(360),
        ], at: when))
        XCTAssertEqual(data.last, 0x0A)
        XCTAssertTrue(String(decoding: data, as: UTF8.self).contains("\"fovH\":68.1235,"))      // As written, not 68.123500000000007.
        XCTAssertEqual(data.filter { $0 == 0x0A }.count, 1)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["kind"] as? String, "sample")
        XCTAssertEqual(object["t"] as? String, "2027-01-15T08:00:00.250Z")
        XCTAssertEqual(object["fovH"] as? Double, 68.1235)                 // Four decimals are plenty.
        XCTAssertTrue(object["pitch"] is NSNull)                           // Not a number: null, not a broken file.
        XCTAssertTrue(object["poseMs"] is NSNull)
        XCTAssertEqual((object["route"] as? [String: Any])?["conf"] as? Double, 0.5)
        XCTAssertTrue(((object["issues"] as? [[String: Any]])?.first?["move"]) is NSNull)
        XCTAssertEqual(object["ok"] as? Bool, true)
        XCTAssertEqual(object["holds"] as? Int, 12)
        XCTAssertEqual(object["width"] as? Double, 360)
    }

    func testAVisitWritesSamplesEventsAndPictures() throws {
        let now = Date(timeIntervalSince1970: 1_800_000_000)
        let first = CaptureDiagnostics(directory: root, now: now)
        first.event("open", ["model": "synthetic"])
        first.sample(["ok": false, "holds": 0])
        first.sample(["ok": true, "holds": 7])
        first.event("tap", ["x": 0.25, "y": 0.75, "role": "start_hand", "hold": "h3"])
        first.write(Data([0xFF, 0xD8, 0xFF]), as: "frame-latest.jpg")
        first.flush()
        let written = try lines(first.file)
        XCTAssertEqual(written.map { $0["kind"] as? String }, ["event", "sample", "sample", "event"])
        XCTAssertEqual(written[0]["event"] as? String, "open")
        XCTAssertEqual(written[2]["holds"] as? Int, 7)
        XCTAssertEqual(written[3]["hold"] as? String, "h3")
        XCTAssertTrue(first.file.lastPathComponent.hasPrefix("session-")); XCTAssertEqual(first.file.pathExtension, "jsonl")
        XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent("frame-latest.jpg")).count, 3)

        // A second visit within the same second gets its own file, and later events go to the latest one.
        let second = CaptureDiagnostics(directory: root, now: now)
        XCTAssertNotEqual(second.file, first.file)
        CaptureDiagnostics.event("analysis", ["record": "synthetic", "status": "result"])
        second.flush()
        XCTAssertEqual(try lines(second.file).map { $0["event"] as? String }, ["analysis"])
        XCTAssertEqual(try lines(first.file).count, 4)
    }

    func testTheOldestVisitsGoWhenTheFolderIsOverItsCap() throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let block = Data(repeating: 0x61, count: 300_000)
        // Five earlier visits of 300 KB each, a picture and the log: 1.7 MB against a cap of 1 MB.
        for day in 1...5 { try block.write(to: root.appendingPathComponent("session-2026010\(day)-120000.jsonl")) }
        try Data(repeating: 1, count: 150_000).write(to: root.appendingPathComponent("preview-latest.png"))
        try Data(repeating: 2, count: 50_000).write(to: root.appendingPathComponent("log.txt"))

        let current = CaptureDiagnostics(directory: root, now: Date(timeIntervalSince1970: 1_800_000_000), cap: 1_000_000)
        current.flush()
        // The three oldest went, which brings it to 800 KB; the pictures and the log are never what goes.
        XCTAssertEqual(names(), ["log.txt", "preview-latest.png", "session-20260104-120000.jsonl", "session-20260105-120000.jsonl", current.file.lastPathComponent].sorted())
        XCTAssertLessThanOrEqual(names().reduce(0) { $0 + size(root.appendingPathComponent($1)) }, 1_000_000)

        // Under the cap nothing is touched.
        let next = CaptureDiagnostics(directory: root, now: Date(timeIntervalSince1970: 1_800_000_100), cap: 1_000_000)
        next.flush()
        XCTAssertEqual(names().count, 6)

        // The visit being written is kept even when it alone is over the cap.
        CaptureDiagnostics.prune(root, cap: 10, keeping: next.file)
        XCTAssertEqual(names(), ["log.txt", "preview-latest.png", next.file.lastPathComponent].sorted())
    }

    func testTheLogKeepsItsNewerHalfWhenItGrowsOverItsCap() throws {
        let cap = 4_000
        for index in 0..<400 { CaptureDiagnostics.log("capture", "line \(index) " + String(repeating: "x", count: 30), directory: root, cap: cap) }
        CaptureDiagnostics.flushLog()
        let file = root.appendingPathComponent("log.txt")
        let text = try String(contentsOf: file, encoding: .utf8)
        let kept = text.split(separator: "\n")
        XCTAssertLessThanOrEqual(size(file), cap + 100)                    // At most one line over.
        XCTAssertTrue(try XCTUnwrap(kept.last).contains("[capture] line 399 "))
        // Whole lines only, each with its time and where it came from, still in order.
        XCTAssertTrue(kept.allSatisfy { $0.contains("Z [capture] line ") })
        let numbers = kept.compactMap { Int($0.split(separator: " ")[3]) }
        XCTAssertEqual(numbers, Array((400 - numbers.count)..<400))
        XCTAssertGreaterThan(numbers.count, 20)
    }

    // MARK: The autopilot's choices

    func testTheAutopilotTakesTheColourAskedForOrTheOneWithMostHolds() {
        let palette = [CaptureAssessment.Swatch(key: "blue", rgb: [0, 0, 255], count: 5), .init(key: "yellow", rgb: [255, 255, 0], count: 9),
                       .init(key: "red", rgb: [255, 0, 0], count: 9)]
        XCTAssertEqual(CaptureAutopilot.colour(in: palette, wanted: nil), "yellow")        // The first of two equals.
        XCTAssertEqual(CaptureAutopilot.colour(in: palette, wanted: ""), "yellow")
        XCTAssertEqual(CaptureAutopilot.colour(in: palette, wanted: "blue"), "blue")
        XCTAssertNil(CaptureAutopilot.colour(in: palette, wanted: "green"))                // Not there: it stops and says so.
        XCTAssertNil(CaptureAutopilot.colour(in: [], wanted: nil))
    }

    func testTheAutopilotMarksTheLowestHoldAsStartAndTheHighestAsFinish() {
        func hold(_ id: String, _ y: Double, width: Double = 0.1) -> CaptureAssessment.Hold {
            .init(id: id, box: CGRect(x: 0.4, y: y, width: width, height: 0.05), poly: [], role: .hold, confidence: nil, user: false)
        }
        let marks = CaptureAutopilot.marks(in: [hold("middle", 0.5), hold("top", 0.1), hold("bottom", 0.8), hold("nothing", 0.95, width: 0)])
        XCTAssertEqual(marks?.start.id, "bottom"); XCTAssertEqual(marks?.finish?.id, "top")
        // One hold alone is a start and nothing else; no hold, nothing to tap.
        let alone = CaptureAutopilot.marks(in: [hold("only", 0.5)])
        XCTAssertEqual(alone?.start.id, "only"); XCTAssertNil(alone?.finish)
        XCTAssertNil(CaptureAutopilot.marks(in: [hold("nothing", 0.5, width: 0)]))
    }

    func testAClipNamedFromTheContainerIsFoundWhereverTheContainerIs() {
        XCTAssertEqual(CaptureSession.debugVideo("Documents/debug/clip.mp4", home: "/somewhere/container").path, "/somewhere/container/Documents/debug/clip.mp4")
        XCTAssertEqual(CaptureSession.debugVideo("/elsewhere/clip.mp4", home: "/somewhere/container").path, "/elsewhere/clip.mp4")
    }

    // MARK: The picture of the screen

    /// A stand-in for a snapshot of the hierarchy: a grey screen, and in `picture` either the black
    /// a preview layer leaves or something drawn, with a white outline and a label on top.
    private func screen(_ size: CGSize, picture: CGRect, preview: UIColor) -> CGImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1; format.opaque = true
        return UIGraphicsImageRenderer(size: size, format: format).image { context in
            UIColor(white: 0.4, alpha: 1).setFill(); context.fill(CGRect(origin: .zero, size: size))
            preview.setFill(); context.fill(picture)
            UIColor.white.setStroke()
            let outline = UIBezierPath(rect: CGRect(x: picture.minX + 40, y: picture.minY + 60, width: 50, height: 40))
            outline.lineWidth = 4; outline.stroke()
            UIColor.green.setFill(); context.fill(CGRect(x: picture.minX + 100, y: picture.minY + 200, width: 30, height: 12))
        }.cgImage!
    }
    private func solid(_ color: UIColor, _ size: CGSize) -> CGImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1; format.opaque = true
        return UIGraphicsImageRenderer(size: size, format: format).image { context in color.setFill(); context.fill(CGRect(origin: .zero, size: size)) }.cgImage!
    }
    private func pixel(_ image: CGImage, _ x: Int, _ y: Int) -> [Int] {
        var bytes = [UInt8](repeating: 0, count: 4)
        let context = CGContext(data: &bytes, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4, space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        context.draw(image.cropping(to: CGRect(x: x, y: y, width: 1, height: 1))!, in: CGRect(x: 0, y: 0, width: 1, height: 1))
        return bytes.prefix(3).map(Int.init)
    }
    private func assertColor(_ found: [Int], _ wanted: [Int], _ message: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertTrue(zip(found, wanted).allSatisfy { abs($0 - $1) <= 6 }, "\(message): \(found) is not \(wanted)", file: file, line: line)
    }

    func testABlackPreviewIsToldFromOneThatCameOut() {
        let size = CGSize(width: 300, height: 600), picture = CGRect(x: 30, y: 80, width: 240, height: 426)
        // Black with outlines over it is still black; a drawn preview is not, and neither is a card covering most of it.
        XCTAssertGreaterThan(CaptureSnapshot.darkShare(of: screen(size, picture: picture, preview: .black), in: picture), 0.9)
        XCTAssertLessThan(CaptureSnapshot.darkShare(of: screen(size, picture: picture, preview: UIColor(red: 0.5, green: 0.4, blue: 0.3, alpha: 1)), in: picture), 0.05)
        XCTAssertLessThan(CaptureSnapshot.darkShare(of: solid(.white, size), in: picture), 0.01)
    }

    func testThePictureGoesUnderWhatTheScreenDrawsOverThePreview() throws {
        let size = CGSize(width: 300, height: 600), picture = CGRect(x: 30, y: 80, width: 240, height: 426)
        let orange = UIColor(red: 1, green: 0.5, blue: 0, alpha: 1)
        let combined = try XCTUnwrap(CaptureSnapshot.composite(hierarchy: screen(size, picture: picture, preview: .black), picture: picture,
                                                               frame: solid(orange, CGSize(width: 360, height: 640)))?.cgImage)
        XCTAssertEqual(combined.width, 300); XCTAssertEqual(combined.height, 600)
        assertColor(pixel(combined, 150, 400), [255, 128, 0], "the picture shows where the preview was black")
        assertColor(pixel(combined, 35, 85), [255, 128, 0], "right into the corner of its rectangle")
        assertColor(pixel(combined, 264, 500), [255, 128, 0], "and the opposite one")
        assertColor(pixel(combined, 70, 140), [255, 255, 255], "an outline stays on top, where it was")
        assertColor(pixel(combined, 95, 160), [255, 128, 0], "with the picture inside it")
        assertColor(pixel(combined, 145, 286), [0, 255, 0], "a label stays on top")
        assertColor(pixel(combined, 10, 10), [102, 102, 102], "the rest of the screen is untouched")
        assertColor(pixel(combined, 150, 560), [102, 102, 102], "below the picture too")
    }
}
#endif
