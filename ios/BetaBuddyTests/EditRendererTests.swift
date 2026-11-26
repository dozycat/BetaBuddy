import XCTest
import AVFoundation
import CoreImage
@testable import BetaBuddy

/// The renderer against a drawn clip and made-up plans. The "people" are coloured squares and the
/// finder that reports them goes by colour: nothing here recognises a person or a climb.
final class EditRendererTests: XCTestCase {
    static let width = 360, height = 640
    static let wall = (r: 219.0, g: 214.0, b: 204.0)
    private var work: URL!

    override func setUp() {
        work = FileManager.default.temporaryDirectory.appendingPathComponent("edit-tests-\(UUID())")
        try? FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
    }
    override func tearDown() { try? FileManager.default.removeItem(at: work) }

    /// Where the three moving squares are at `time`, as centres in 0–1 from the top-left.
    static func climber(_ time: Double) -> (x: Double, y: Double) { (0.42, 0.70 - 0.45 * time / 3) }      // Green, inside the keep box.
    static func passer(_ time: Double) -> (x: Double, y: Double) { (0.1 + 0.8 * time / 3, 0.86) }         // Magenta, walks past below.
    static func thing(_ time: Double) -> (x: Double, y: Double) { (0.8, 0.3 + 0.3 * time / 3) }           // Orange, moves but is not a person.
    static let keep = EditPlan.Box(x: 0.32, y: 0.15, w: 0.2, h: 0.6)

    private func clip(seconds: Double = 3) async throws -> URL {
        let w = CGFloat(Self.width), h = CGFloat(Self.height)
        return try await SyntheticWallClip.make(width: Self.width, height: Self.height, seconds: seconds) { context, frame in
            let time = Double(frame) / 30
            func square(_ centre: (x: Double, y: Double), _ side: Double, _ r: CGFloat, _ g: CGFloat, _ b: CGFloat) {
                context.setFillColor(CGColor(red: r / 255, green: g / 255, blue: b / 255, alpha: 1))
                context.fill(CGRect(x: (centre.x - side / 2) * w, y: (1 - centre.y) * h - side * w / 2, width: side * w, height: side * w))
            }
            square(Self.climber(time), 0.08, 40, 200, 70)
            square(Self.passer(time), 0.10, 200, 30, 160)
            square(Self.thing(time), 0.08, 240, 140, 20)
        }
    }

    private func plan(_ extra: String) throws -> EditPlan {
        try XCTUnwrap(EditPlan(json: #"{"v":1,"duration":3,"trim":[0.5,2.5],"# + extra + "}"))
    }
    private var keepJSON: String {
        let b = Self.keep
        return #""keep":{"feather":0.03,"boxes":[{"t":0,"b":[[\#(b.x),\#(b.y),\#(b.w),\#(b.h)]]}]},"plate":{"times":[0,0.37,0.75,1.12,1.5,1.87,2.25,2.62,2.95]}"#
    }

    /// Reports green and magenta squares as people, by colour, from a small copy of the picture.
    final class ColourFinder: PeopleFinder {
        var calls = 0
        var failing = false
        /// Hand back an outline picture for each person instead of only a box.
        var masks = false
        /// Does not see the magenta square while its centre is in this stretch across the picture.
        var blind: ClosedRange<Double>?
        func people(in image: CIImage, context: CIContext) throws -> [FoundPerson] {
            calls += 1
            if failing { throw CocoaError(.featureUnsupported) }
            let w = 90, h = 160
            let small = image.transformed(by: CGAffineTransform(scaleX: CGFloat(w) / image.extent.width, y: CGFloat(h) / image.extent.height))
            var bytes = [UInt8](repeating: 0, count: w * h * 4)
            context.render(small, toBitmap: &bytes, rowBytes: w * 4, bounds: CGRect(x: 0, y: 0, width: w, height: h), format: .RGBA8, colorSpace: nil)
            var boxes: [Int: (x0: Int, y0: Int, x1: Int, y1: Int)] = [:]
            var outlines: [Int: [UInt8]] = [:]
            for y in 0..<h { for x in 0..<w {
                let r = bytes[(y * w + x) * 4], g = bytes[(y * w + x) * 4 + 1], b = bytes[(y * w + x) * 4 + 2]
                let kind = g > 150 && r < 110 && b < 130 ? 0 : r > 150 && g < 90 && b > 110 ? 1 : -1
                guard kind >= 0 else { continue }
                if let box = boxes[kind] { boxes[kind] = (min(box.x0, x), min(box.y0, y), max(box.x1, x), max(box.y1, y)) } else { boxes[kind] = (x, y, x, y) }
                outlines[kind, default: [UInt8](repeating: 0, count: w * h)][y * w + x] = 255
            } }
            return boxes.keys.sorted().compactMap { kind -> FoundPerson? in
                guard let found = boxes[kind] else { return nil }
                let box = EditPlan.Box(x: Double(found.x0) / Double(w), y: Double(found.y0) / Double(h), w: Double(found.x1 - found.x0 + 1) / Double(w), h: Double(found.y1 - found.y0 + 1) / Double(h))
                if kind == 1, let blind, blind.contains(box.x + box.w / 2) { return nil }
                let mask = masks ? outlines[kind].map { CIImage(bitmapData: Data($0), bytesPerRow: w, size: CGSize(width: w, height: h), format: .R8, colorSpace: nil) } : nil
                return FoundPerson(box: box, mask: mask)
            }
        }
    }

    /// The colour of the result at `time`, at a point in 0–1 from the top-left.
    private func colour(_ url: URL, at time: Double, _ point: (x: Double, y: Double)) async throws -> (r: Double, g: Double, b: Double) {
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        generator.requestedTimeToleranceBefore = .zero; generator.requestedTimeToleranceAfter = .zero
        let image = try await generator.image(at: CMTime(seconds: time, preferredTimescale: 600)).image
        var pixel = [UInt8](repeating: 0, count: 4)
        let context = try XCTUnwrap(CGContext(data: &pixel, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        // One pixel of the picture, drawn so that the wanted point lands on it.
        context.draw(image, in: CGRect(x: -point.x * Double(image.width), y: -(1 - point.y) * Double(image.height) + 1, width: Double(image.width), height: Double(image.height)))
        return (Double(pixel[0]), Double(pixel[1]), Double(pixel[2]))
    }
    private func isWall(_ c: (r: Double, g: Double, b: Double)) -> Bool { abs(c.r - Self.wall.r) < 30 && abs(c.g - Self.wall.g) < 30 && abs(c.b - Self.wall.b) < 30 }
    private func isMagenta(_ c: (r: Double, g: Double, b: Double)) -> Bool { c.r > 150 && c.g < 100 && c.b > 110 }
    private func isGreen(_ c: (r: Double, g: Double, b: Double)) -> Bool { c.g > 150 && c.r < 110 }
    private func isOrange(_ c: (r: Double, g: Double, b: Double)) -> Bool { c.r > 190 && c.g > 90 && c.g < 190 && c.b < 90 }

    func testMedianAcrossFramesTakesTheMiddleValueOfEveryByte() {
        // Five "pictures" of two rows by three one-byte pixels; the middle value of each byte is known.
        let frames: [[UInt8]] = [[9, 0, 200, 1, 1, 1], [1, 0, 100, 2, 255, 1], [5, 0, 150, 3, 0, 1], [7, 255, 50, 4, 7, 9], [3, 0, 250, 5, 8, 1]]
        let packed = frames.flatMap { $0 }
        var out = [UInt8](repeating: 0, count: 8)   // Rows four bytes apart: the padding is left alone.
        XCTAssertTrue(packed.withUnsafeBytes { PlateMedian.median(frames: $0.baseAddress!, count: 5, width: 3, pixelBytes: 1, height: 2, out: &out, outRowBytes: 4) })
        XCTAssertEqual(out, [5, 0, 150, 0, 3, 7, 1, 0])
        var one = [UInt8](repeating: 0, count: 6)
        XCTAssertTrue(frames[0].withUnsafeBytes { PlateMedian.median(frames: $0.baseAddress!, count: 1, width: 3, pixelBytes: 1, height: 2, out: &one, outRowBytes: 3) })
        XCTAssertEqual(one, frames[0])
        XCTAssertFalse(packed.withUnsafeBytes { PlateMedian.median(frames: $0.baseAddress!, count: 5, width: 3, pixelBytes: 1, height: 2, out: &out, outRowBytes: 4) { true } })
    }

    func testMedianLeavesOutWhatIsInsideAKeepBoxAndSaysWhereTooLittleIsLeft() {
        // Seven pictures of one row, four pixels of two bytes. In the first four something (value 200) sits on
        // pixels 1 and 2 inside a keep box; in the last three the box has moved on to pixel 3 and they show the wall (10).
        func picture(_ values: [UInt8]) -> [UInt8] { values.flatMap { [$0, $0] } }
        let frames = Array(repeating: picture([10, 200, 200, 10]), count: 4) + Array(repeating: picture([10, 10, 10, 200]), count: 3)
        let packed = frames.flatMap { $0 }
        let holes: [[PlateMedian.Hole]] = Array(repeating: [(x0: 1, x1: 3, y0: 0, y1: 1)], count: 4) + Array(repeating: [(x0: 3, x1: 9, y0: -2, y1: 5)], count: 3)
        var out = [UInt8](repeating: 0, count: 8), valid = [UInt8](repeating: 7, count: 4)
        XCTAssertTrue(packed.withUnsafeBytes { PlateMedian.median(frames: $0.baseAddress!, count: 7, width: 4, pixelBytes: 2, height: 1, holes: holes, out: &out, outRowBytes: 8, valid: &valid, validRowBytes: 4) })
        // Pixels 1 and 2: the wall from the three pictures that count, though most pictures show something else there.
        XCTAssertEqual(out, picture([10, 10, 10, 10]))
        XCTAssertEqual(valid, [255, 255, 255, 255])
        // Without the boxes it is what most pictures show.
        XCTAssertTrue(packed.withUnsafeBytes { PlateMedian.median(frames: $0.baseAddress!, count: 7, width: 4, pixelBytes: 2, height: 1, out: &out, outRowBytes: 8, valid: &valid, validRowBytes: 4) })
        XCTAssertEqual(out, picture([10, 200, 200, 10])); XCTAssertEqual(valid, [255, 255, 255, 255])
        // A box that stays on pixels 1 and 2 in five of the seven: two pictures are too few, and that is said.
        let staying: [[PlateMedian.Hole]] = Array(repeating: [(x0: 1, x1: 3, y0: 0, y1: 1)], count: 5) + [[], []]
        XCTAssertTrue(packed.withUnsafeBytes { PlateMedian.median(frames: $0.baseAddress!, count: 7, width: 4, pixelBytes: 2, height: 1, holes: staying, out: &out, outRowBytes: 8, valid: &valid, validRowBytes: 4) })
        XCTAssertEqual(valid, [255, 0, 0, 255])
        XCTAssertEqual(out[0], 10); XCTAssertEqual(out[6], 10)
        // Every count of pictures left out, odd or even, gives the middle of the rest: the lower of the two
        // middle ones when the rest is even (here 1…9 with the top k left out).
        for left in 0...6 {
            let values = (1...9).map { UInt8($0) }
            let single: [[PlateMedian.Hole]] = (0..<9).map { $0 >= 9 - left ? [(x0: 0, x1: 1, y0: 0, y1: 1)] : [] }
            var middle = [UInt8](repeating: 0, count: 1), known = [UInt8](repeating: 0, count: 1)
            XCTAssertTrue(values.withUnsafeBytes { PlateMedian.median(frames: $0.baseAddress!, count: 9, width: 1, pixelBytes: 1, height: 1, holes: single, out: &middle, outRowBytes: 1, valid: &known, validRowBytes: 1) })
            XCTAssertEqual(known, [255]); XCTAssertEqual(Int(middle[0]), (9 - left - 1) / 2 + 1, "\(left) left out")
        }
        func median(_ values: [UInt8], holes: [[PlateMedian.Hole]] = []) -> (value: UInt8, valid: UInt8) {
            var out = [UInt8](repeating: 9, count: 1), known = [UInt8](repeating: 9, count: 1)
            XCTAssertTrue(values.withUnsafeBytes { PlateMedian.median(frames: $0.baseAddress!, count: values.count, width: 1, pixelBytes: 1, height: 1, holes: holes, out: &out, outRowBytes: 1, valid: &known, validRowBytes: 1) })
            return (out[0], known[0])
        }
        let hole: PlateMedian.Hole = (x0: 0, x1: 1, y0: 0, y1: 1)
        // Four pictures, one left out: the middle of 100, 102, 200 is 102, wherever the one left out stands.
        for skippedAt in 0..<4 {
            var values: [UInt8] = [100, 102, 200]; values.insert(7, at: skippedAt)
            var holes = [[PlateMedian.Hole]](repeating: [], count: 4); holes[skippedAt] = [hole]
            XCTAssertEqual(median(values, holes: holes).value, 102, "left out at \(skippedAt)"); XCTAssertEqual(median(values, holes: holes).valid, 255)
        }
        // An even number that count, none left out: the lower of the two in the middle.
        XCTAssertEqual(median([4, 1, 3, 2]).value, 2)
        // Two holes over the same pixel in one picture leave that picture out once, not twice: five pictures, two left
        // out by doubled holes, three count, and their middle is the answer.
        let doubled = median([50, 0, 60, 255, 70], holes: [[], [hole, hole], [], [hole, (x0: -5, x1: 9, y0: 0, y1: 3)], []])
        XCTAssertEqual(doubled.value, 60); XCTAssertEqual(doubled.valid, 255)
        // Everything left out: nothing known there.
        XCTAssertEqual(median([1, 2, 3], holes: [[hole], [hole], [hole]]).valid, 0)
        // Real 255s among the values are not mistaken for pictures that were left out.
        XCTAssertEqual(median([255, 255, 10, 3], holes: [[], [], [], [hole]]).value, 255)
    }

    func testTheChosenPeopleAreReplacedAndKeepBoxesWin() {
        let context = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
        let extent = CGRect(x: 0, y: 0, width: 200, height: 400)
        let live = CIImage(color: CIColor(red: 1, green: 0, blue: 0)).cropped(to: extent)
        let plate = CIImage(color: CIColor(red: 0, green: 0, blue: 1)).cropped(to: extent)
        let keep = [EditPlan.Box(x: 0.4, y: 0.1, w: 0.3, h: 0.4)]
        // The masks are worked out on a small picture: here a quarter of the frame's size.
        let small = SmallPictures(width: 50, height: 100, valid: [], plate: [], reach: 0)
        func mask(_ boxes: [EditPlan.Box], feather: Double) -> CIImage {
            CIImage(bitmapData: Data(small.soft(boxes.map { small.outline(of: $0) }, feather: feather)), bytesPerRow: 50, size: CGSize(width: 50, height: 100), format: .R8, colorSpace: nil)
        }
        // Who is replaced has been decided before; here: one to the left, and one along the bottom right.
        let passers = [EditPlan.Box(x: 0.05, y: 0.6, w: 0.25, h: 0.3), EditPlan.Box(x: 0.5, y: 0.9, w: 0.5, h: 0.1)]
        let image = EditFrame.replacing(live: live, plate: plate, mask: mask(passers, feather: 0.03), keep: keep)
        XCTAssertEqual(image.extent, extent)
        var bytes = [UInt8](repeating: 0, count: 200 * 400 * 4)
        context.render(image, toBitmap: &bytes, rowBytes: 800, bounds: extent, format: .RGBA8, colorSpace: nil)
        func at(_ x: Double, _ y: Double) -> (r: Int, b: Int) {   // 0–1 from the top-left; the bitmap's first row is the top.
            let offset = (min(399, Int(y * 400)) * 200 + min(199, Int(x * 200))) * 4
            return (Int(bytes[offset]), Int(bytes[offset + 2]))
        }
        XCTAssertEqual(at(0.17, 0.75).b, 255); XCTAssertEqual(at(0.17, 0.75).r, 0)       // The passer-by: the plate.
        XCTAssertEqual(at(0.55, 0.3).r, 255); XCTAssertEqual(at(0.55, 0.3).b, 0)         // In the keep box: live.
        XCTAssertEqual(at(0.85, 0.5).r, 255)                                              // Nobody there: live, not the plate.
        XCTAssertEqual(at(0.2, 0.2).r, 255)
        XCTAssertEqual(at(0.75, 0.96).b, 255)                                             // The other one: the plate, where they are (the top of the picture is the top).
        XCTAssertEqual(at(0.2, 0.96).r, 255); XCTAssertEqual(at(0.75, 0.04).r, 255)
        let edge = at(0.37, 0.75)                                                         // A little outside the passer-by: a soft edge, some of each.
        XCTAssertGreaterThan(edge.b, 20); XCTAssertGreaterThan(edge.r, 20)
        XCTAssertEqual(at(0.45, 0.75).r, 255)                                             // Further out: live again.
        // A passer-by whose grown shape reaches into the keep box does not show through it.
        let near = EditFrame.replacing(live: live, plate: plate, mask: mask([EditPlan.Box(x: 0.05, y: 0.2, w: 0.3, h: 0.2)], feather: 0.1), keep: keep)
        context.render(near, toBitmap: &bytes, rowBytes: 800, bounds: extent, format: .RGBA8, colorSpace: nil)
        XCTAssertEqual(at(0.41, 0.3).r, 255); XCTAssertEqual(at(0.41, 0.3).b, 0)
        XCTAssertGreaterThan(at(0.38, 0.3).b, 100)
        // Where the plate is not known the picture stays live, whoever is there: here the left half is unknown.
        let known = CIImage(color: .white).cropped(to: CGRect(x: 100, y: 0, width: 100, height: 400)).composited(over: CIImage(color: .black).cropped(to: extent))
        let partly = EditFrame.replacing(live: live, plate: plate, valid: known, mask: mask(passers, feather: 0.03), keep: keep)
        context.render(partly, toBitmap: &bytes, rowBytes: 800, bounds: extent, format: .RGBA8, colorSpace: nil)
        XCTAssertEqual(at(0.17, 0.75).r, 255); XCTAssertEqual(at(0.17, 0.75).b, 0)       // On the unknown half: left as filmed.
        XCTAssertEqual(at(0.75, 0.96).b, 255)                                             // On the known half: the plate.

        // The mask itself: full inside the outline, soft outside, and nothing beyond the reach that is checked beforehand.
        let side = SmallPictures(width: 60, height: 100, valid: [], plate: [], reach: 0)
        let person = side.outline(of: EditPlan.Box(x: 0.4, y: 0.4, w: 0.2, h: 0.2))        // Columns 24..<36, rows 40..<60.
        let soft = side.soft([person], feather: 0.05)
        let spread = SmallPictures.spread(feather: 0.05, side: 60)
        XCTAssertEqual(spread.grow, 3); XCTAssertEqual(spread.blur, 1); XCTAssertEqual(spread.reach, 6)
        XCTAssertEqual(soft[50 * 60 + 30], 255); XCTAssertEqual(soft[40 * 60 + 24], 255)   // The middle and a corner of the outline.
        for index in person.indices where person[index] != 0 { XCTAssertEqual(soft[index], 255) }   // Everything inside the outline goes, corners included.
        XCTAssertTrue((1..<255).contains(soft[50 * 60 + 39]))                               // A little beyond what it was grown to: partly.
        let reached = side.grown(person, by: spread.reach)
        for index in soft.indices where reached[index] == 0 { XCTAssertEqual(soft[index], 0) }
        XCTAssertEqual(side.soft([], feather: 0.05), [UInt8](repeating: 0, count: 6000))
    }

    func testWindowShowsItsPartOfThePictureEnlarged() {
        let context = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
        let extent = CGRect(x: 0, y: 0, width: 100, height: 200)
        // Red everywhere, a blue block in the top-right quarter's middle.
        let image = CIImage(color: CIColor(red: 0, green: 0, blue: 1)).cropped(to: EditFrame.rect(EditPlan.Box(x: 0.7, y: 0.2, w: 0.1, h: 0.1), in: extent))
            .composited(over: CIImage(color: CIColor(red: 1, green: 0, blue: 0)).cropped(to: extent))
        XCTAssertEqual(EditFrame.framed(image, window: .full), image)
        let framed = EditFrame.framed(image, window: EditWindow(cx: 0.75, cy: 0.25, s: 2))
        XCTAssertEqual(framed.extent, extent)
        var bytes = [UInt8](repeating: 0, count: 100 * 200 * 4)
        context.render(framed, toBitmap: &bytes, rowBytes: 400, bounds: extent, format: .RGBA8, colorSpace: nil)
        func blue(_ x: Double, _ y: Double) -> Bool { bytes[(Int(y * 200) * 100 + Int(x * 100)) * 4 + 2] > 200 }
        XCTAssertTrue(blue(0.5, 0.5))                 // The block is now in the middle, twice the size: 0.4…0.6 across.
        XCTAssertTrue(blue(0.42, 0.42)); XCTAssertTrue(blue(0.58, 0.58))
        XCTAssertFalse(blue(0.35, 0.5)); XCTAssertFalse(blue(0.65, 0.5)); XCTAssertFalse(blue(0.75, 0.25))
    }

    func testUprightTransformPutsARotatedFrameTheRightWayUp() {
        // A clip stored 200×100 and shown turned a quarter: 100×200 on screen, rendered at half size.
        let preferred = CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: 100, ty: 0)
        let upright = EditFrame.upright(natural: CGSize(width: 200, height: 100), preferred: preferred, render: CGSize(width: 50, height: 100))
        XCTAssertEqual(CGRect(x: 0, y: 0, width: 200, height: 100).applying(upright).standardized, CGRect(x: 0, y: 0, width: 50, height: 100))
        // The stored picture's top-left corner (Core Image: x 0, y 100) is shown at the top-right (Core Image: x 50, y 100).
        let corner = CGPoint(x: 0, y: 100).applying(upright)
        XCTAssertEqual(corner.x, 50, accuracy: 1e-9); XCTAssertEqual(corner.y, 100, accuracy: 1e-9)
        let plain = EditFrame.upright(natural: CGSize(width: 200, height: 100), preferred: .identity, render: CGSize(width: 200, height: 100))
        XCTAssertEqual(CGPoint(x: 30, y: 70).applying(plain), CGPoint(x: 30, y: 70))
    }

    func testRenderFollowsThePlanAndReplacesOnlyThePasserBy() async throws {
        let source = try await clip()
        defer { try? FileManager.default.removeItem(at: source) }
        // 0.5–1.5 as it is, 1.5–2.0 at half speed, then 1.0–2.0 again at double speed: 1 + 1 + 0.5 seconds.
        let plan = try plan(#""segments":[{"t0":0.5,"t1":1.5,"speed":1,"kind":"run"},{"t0":1.5,"t1":2,"speed":0.5,"kind":"slow"},{"t0":1,"t1":2,"speed":2,"kind":"replay"}],"# + keepJSON)
        let finder = ColourFinder()
        let renderer = EditRenderer(source: source, plan: plan, directory: work, finder: finder)
        let seen = Progress()
        let url = try await renderer.render(options: EditOptions()) { seen.note($0) }
        let asset = AVURLAsset(url: url)
        XCTAssertEqual(url.pathExtension, "mp4")
        do { let seconds = try await asset.load(.duration).seconds; XCTAssertEqual(seconds, 2.5, accuracy: 0.1) }
        let audio = try await asset.loadTracks(withMediaType: .audio)
        XCTAssertTrue(audio.isEmpty)
        let video = try await asset.loadTracks(withMediaType: .video)
        let size = try await XCTUnwrap(video.first).load(.naturalSize)
        XCTAssertEqual(size, CGSize(width: Self.width, height: Self.height))
        XCTAssertGreaterThan(finder.calls, 20)
        XCTAssertEqual(renderer.removedPeople, 1)                                              // The passer-by; the climber's track touches the keep box.
        XCTAssertEqual(seen.last, 1); XCTAssertTrue(seen.rising)

        // Half a second in is one second of the clip: the passer-by is gone, the wall behind them is there.
        do { let c = try await colour(url, at: 0.5, Self.passer(1.0)); XCTAssertTrue(isWall(c), "\(c)") }
        do { let c = try await colour(url, at: 0.5, Self.climber(1.0)); XCTAssertTrue(isGreen(c), "\(c)") }            // The climber is live.
        do { let c = try await colour(url, at: 0.5, Self.thing(1.0)); XCTAssertTrue(isOrange(c), "\(c)") }             // What moves but is not a person stays.
        // The slowed part: 1.5 s of the result is 1.75 s of the clip.
        do { let c = try await colour(url, at: 1.5, Self.climber(1.75)); XCTAssertTrue(isGreen(c), "\(c)") }
        do { let c = try await colour(url, at: 1.5, Self.climber(1.5 + 0.5)); XCTAssertFalse(isGreen(c), "\(c)") }     // Not where it would be at full speed.
        do { let c = try await colour(url, at: 1.5, Self.passer(1.75)); XCTAssertTrue(isWall(c), "\(c)") }
        // The replay: 2.25 s of the result is 1.5 s of the clip again.
        do { let c = try await colour(url, at: 2.25, Self.climber(1.5)); XCTAssertTrue(isGreen(c), "\(c)") }
        do { let c = try await colour(url, at: 2.25, Self.thing(1.5)); XCTAssertTrue(isOrange(c), "\(c)") }

        // Every switch off: the attempt alone, nothing replaced, nobody looked for.
        let calls = finder.calls
        let plain = try await renderer.render(options: EditOptions(removePassersBy: false, slowMotion: false, cameraMoves: false)) { _ in }
        do { let seconds = try await AVURLAsset(url: plain).load(.duration).seconds; XCTAssertEqual(seconds, 2.0, accuracy: 0.1) }
        do { let c = try await colour(plain, at: 0.5, Self.passer(1.0)); XCTAssertTrue(isMagenta(c), "\(c)") }
        do { let c = try await colour(plain, at: 1.5, Self.climber(2.0)); XCTAssertTrue(isGreen(c), "\(c)") }
        XCTAssertEqual(finder.calls, calls)
        XCTAssertNotEqual(plain, url)
    }

    func testAClimberWhoStaysInOnePlaceIsNotPartOfTheEmptyWall() async throws {
        // The "climber" sits at one place for two thirds of the clip, then moves up; the keep box goes with it.
        // Afterwards a "passer-by" walks over the place it left. Most frames show the climber there, but those
        // are the frames in which that place is inside the keep box, so the empty wall there is the wall.
        let w = CGFloat(Self.width), h = CGFloat(Self.height)
        let low = (x: 0.42, y: 0.5), high = (x: 0.42, y: 0.2)
        func walker(_ time: Double) -> (x: Double, y: Double) { (0.42 + (time - 2.6) * 0.6, 0.5) }
        let source = try await SyntheticWallClip.make(width: Self.width, height: Self.height, seconds: 3) { context, frame in
            let time = Double(frame) / 30
            func square(_ centre: (x: Double, y: Double), _ side: Double, _ r: CGFloat, _ g: CGFloat, _ b: CGFloat) {
                context.setFillColor(CGColor(red: r / 255, green: g / 255, blue: b / 255, alpha: 1))
                context.fill(CGRect(x: (centre.x - side / 2) * w, y: (1 - centre.y) * h - side * w / 2, width: side * w, height: side * w))
            }
            square(time < 2.1 ? low : high, 0.08, 40, 200, 70)
            square(walker(time), 0.10, 200, 30, 160)
        }
        defer { try? FileManager.default.removeItem(at: source) }
        let plan = try XCTUnwrap(EditPlan(json: #"{"v":1,"duration":3,"trim":[1,3],"segments":[{"t0":1,"t1":3,"speed":1}],"plate":{"times":[0,0.37,0.75,1.12,1.5,1.87,2.25,2.62,2.95]},"# +
            #""keep":{"feather":0.03,"boxes":[{"t":0,"b":[[0.36,0.4,0.12,0.2]]},{"t":2.0,"b":[[0.36,0.4,0.12,0.2]]},{"t":2.2,"b":[[0.36,0.05,0.12,0.25]]},{"t":3,"b":[[0.36,0.05,0.12,0.25]]}]}}"#))
        // The attempt starts one second in: three of the nine plate times are before it, and the keep box
        // of each of those moments is left out of the plate like any other. The result runs one second behind the clip.
        XCTAssertEqual(plan.plateTimes.filter { !plan.trim.contains($0) }.count, 3)
        let renderer = EditRenderer(source: source, plan: plan, directory: work, finder: ColourFinder())
        let url = try await renderer.render(options: EditOptions()) { _ in }
        XCTAssertEqual(renderer.removedPeople, 1)
        do { let c = try await colour(url, at: 1.6, low); XCTAssertTrue(isWall(c), "\(c)") }          // Neither the passer-by nor a ghost of the climber.
        do { let c = try await colour(url, at: 1.6, high); XCTAssertTrue(isGreen(c), "\(c)") }        // The climber, live, where it is now.
        do { let c = try await colour(url, at: 0.2, low); XCTAssertTrue(isGreen(c), "\(c)") }         // And where it was then.
        do { let c = try await colour(url, at: 1.3, walker(2.3)); XCTAssertTrue(isWall(c), "\(c)") }  // The passer-by on the way in is gone too.
    }

    func testCameraWindowIsAppliedAndCanBeSwitchedOff() async throws {
        let source = try await clip()
        defer { try? FileManager.default.removeItem(at: source) }
        // Three times closer on a blue blob near the top right (a made-up position from the drawn wall).
        let hold = try XCTUnwrap(StandInPack.holds.first { $0.id == "b5" })
        let plan = try plan(#""segments":[{"t0":0.5,"t1":2.5,"speed":1}],"camera":[{"t":0.5,"cx":\#(hold.x),"cy":\#(hold.y),"s":3}]"#)
        XCTAssertFalse(plan.offersRemoval); XCTAssertFalse(plan.offersSlowMotion); XCTAssertTrue(plan.offersCamera)
        let renderer = EditRenderer(source: source, plan: plan, directory: work, finder: ColourFinder())
        let close = try await renderer.render(options: EditOptions()) { _ in }
        let centre = try await colour(close, at: 1, (0.5, 0.5))
        XCTAssertTrue(centre.b > 150 && centre.r < 110, "\(centre)")
        let wide = try await renderer.render(options: EditOptions(cameraMoves: false)) { _ in }
        let there = try await colour(wide, at: 1, (hold.x, hold.y))
        XCTAssertTrue(there.b > 150 && there.r < 110, "\(there)")
        let middle = try await colour(wide, at: 1, (0.5, 0.5))
        XCTAssertFalse(middle.b > 150 && middle.r < 110)
        do { let seconds = try await AVURLAsset(url: close).load(.duration).seconds; XCTAssertEqual(seconds, 2.0, accuracy: 0.1) }
    }

    func testAClipStoredTurnedIsRenderedUprightAndThePlanIsReadInUprightCoordinates() async throws {
        // Stored 640×360 with a quarter turn, shown 360×640. A cyan block in the stored picture's top-left
        // is at the top-right once upright: about 0.70–0.95 across and 0.02–0.50 down.
        let turn = CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: 360, ty: 0)
        let source = try await SyntheticWallClip.make(width: 640, height: 360, seconds: 2, transform: turn) { context, _ in
            context.setFillColor(CGColor(red: 0, green: 200 / 255, blue: 220 / 255, alpha: 1))
            context.fill(CGRect(x: 0.02 * 640, y: 0.70 * 360, width: 0.48 * 640, height: 0.25 * 360))
        }
        defer { try? FileManager.default.removeItem(at: source) }
        func isCyan(_ c: (r: Double, g: Double, b: Double)) -> Bool { c.r < 80 && c.g > 150 && c.b > 170 }
        let plan = try XCTUnwrap(EditPlan(json: #"{"v":1,"duration":2,"trim":[0,2],"segments":[{"t0":0,"t1":2,"speed":1}],"camera":[{"t":0,"cx":0.825,"cy":0.26,"s":3}],"# + keepJSON + "}"))
        let renderer = EditRenderer(source: source, plan: plan, directory: work, finder: ColourFinder())
        let wide = try await renderer.render(options: EditOptions(cameraMoves: false)) { _ in }
        let video = try await AVURLAsset(url: wide).loadTracks(withMediaType: .video)
        let track = try XCTUnwrap(video.first)
        let size = try await track.load(.naturalSize), stored = try await track.load(.preferredTransform)
        XCTAssertEqual(size, CGSize(width: 360, height: 640)); XCTAssertEqual(stored, .identity)      // Upright in the file itself.
        do { let c = try await colour(wide, at: 1, (0.825, 0.26)); XCTAssertTrue(isCyan(c), "\(c)") }
        do { let c = try await colour(wide, at: 1, (0.2, 0.26)); XCTAssertFalse(isCyan(c), "\(c)") }
        do { let c = try await colour(wide, at: 1, (0.825, 0.8)); XCTAssertFalse(isCyan(c), "\(c)") }
        do { let c = try await colour(wide, at: 1, (0.5, 0.5)); XCTAssertFalse(isCyan(c), "\(c)") }
        // The plate (made with passer-by removal on) is the same picture the same way up: the block is in it.
        let close = try await renderer.render(options: EditOptions()) { _ in }
        do { let c = try await colour(close, at: 1, (0.5, 0.5)); XCTAssertTrue(isCyan(c), "\(c)") }   // The camera looks where the plan says, upright.
    }

    func testRenderStopsWhenCancelledAndIsLeftAsFilmedWhenNobodyCanBeLookedFor() async throws {
        let source = try await clip()
        defer { try? FileManager.default.removeItem(at: source) }
        let plan = try plan(#""segments":[{"t0":0.5,"t1":2.5,"speed":0.25}],"# + keepJSON)
        let renderer = EditRenderer(source: source, plan: plan, directory: work, finder: ColourFinder())
        let task = Task { try await renderer.render(options: EditOptions()) { _ in } }
        try await Task.sleep(for: .milliseconds(150))
        task.cancel()
        do { _ = try await task.value; XCTFail("a cancelled render returned a file") } catch is CancellationError {} catch { XCTFail("\(error)") }
        // Nothing of it is left behind: no movie and no working file.
        let left = FileManager.default.enumerator(at: work, includingPropertiesForKeys: nil)?.compactMap { $0 as? URL }.filter { ["mp4", "raw"].contains($0.pathExtension) } ?? []
        XCTAssertEqual(left, [])

        // Nobody can be looked for (the finder fails on every frame): nobody is taken out, and that is all.
        // The edit is the picture as filmed, without an error.
        let broken = ColourFinder(); broken.failing = true
        let quiet = EditRenderer(source: source, plan: plan, directory: work, finder: broken)
        let url = try await quiet.render(options: EditOptions()) { _ in }
        XCTAssertGreaterThan(broken.calls, 10); XCTAssertEqual(quiet.removedPeople, 0)
        do { let seconds = try await AVURLAsset(url: url).load(.duration).seconds; XCTAssertEqual(seconds, 8, accuracy: 0.2) }
        do { let c = try await colour(url, at: 2, Self.passer(1.0)); XCTAssertTrue(isMagenta(c), "\(c)") }     // 2 s of the result is 1 s of the clip.
    }

    /// The made-up passer-by of these clips at one moment after another, in the plain result of `plan`.
    private func passerSeen(_ url: URL, at times: [Double], offset: Double = 0) async throws -> [Bool] {
        var seen: [Bool] = []
        for time in times { seen.append(isMagenta(try await colour(url, at: time - offset, Self.passer(time)))) }
        return seen
    }

    func testAPasserByWhoComesTooCloseToTheClimberIsLeftAloneFromFirstToLast() async throws {
        let source = try await clip()
        defer { try? FileManager.default.removeItem(at: source) }
        // The keep box reaches down to where the passer-by walks: they touch it in the middle of their way, not at the ends.
        let keep = #""keep":{"feather":0.03,"boxes":[{"t":0,"b":[[0.32,0.15,0.2,0.69]]}]},"plate":{"times":[0,0.37,0.75,1.12,1.5,1.87,2.25,2.62,2.95]}"#
        let renderer = EditRenderer(source: source, plan: try plan(#""segments":[{"t0":0.5,"t1":2.5,"speed":1}],"# + keep), directory: work, finder: ColourFinder())
        let url = try await renderer.render(options: EditOptions()) { _ in }
        XCTAssertEqual(renderer.removedPeople, 0)
        // Far to the left, right below the climber, far to the right: there all the way. No half bodies, no coming and going.
        let seen = try await passerSeen(url, at: [0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4], offset: 0.5)
        XCTAssertEqual(seen, [Bool](repeating: true, count: 7))
        // The same walk with the keep box ending above it: gone all the way.
        let clear = EditRenderer(source: source, plan: try plan(#""segments":[{"t0":0.5,"t1":2.5,"speed":1}],"# + keepJSON), directory: work, finder: ColourFinder())
        let gone = try await passerSeen(try await clear.render(options: EditOptions()) { _ in }, at: [0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4], offset: 0.5)
        XCTAssertEqual(gone, [Bool](repeating: false, count: 7))
        XCTAssertEqual(clear.removedPeople, 1)
    }

    func testAPasserByNotFoundForAMomentDoesNotFlickerBackAndOutlinesComeBackTheRightWayUp() async throws {
        let source = try await clip()
        defer { try? FileManager.default.removeItem(at: source) }
        let plan = try plan(#""segments":[{"t0":0.5,"t1":2.5,"speed":1}],"# + keepJSON)
        // Not found while crossing a narrow stretch (about a frame, around 1.89 s of the clip), and found by outline, not by box.
        let finder = ColourFinder(); finder.blind = 0.60...0.61; finder.masks = true
        let renderer = EditRenderer(source: source, plan: plan, directory: work, finder: finder)
        let url = try await renderer.render(options: EditOptions()) { _ in }
        XCTAssertEqual(renderer.removedPeople, 1)                         // One person, one track, across the gap.
        let times = [0.6, 1.0, 1.4, 1.80, 1.84, 1.87, 1.90, 1.93, 1.97, 2.0, 2.4]
        let seen = try await passerSeen(url, at: times, offset: 0.5)
        XCTAssertEqual(seen, [Bool](repeating: false, count: times.count))
        do { let c = try await colour(url, at: 1.4, Self.passer(1.9)); XCTAssertTrue(isWall(c), "\(c)") }      // The wall, where they were not found.
        do { let c = try await colour(url, at: 1.4, Self.climber(1.9)); XCTAssertTrue(isGreen(c), "\(c)") }
        do { let c = try await colour(url, at: 1.4, Self.thing(1.9)); XCTAssertTrue(isOrange(c), "\(c)") }
        // Lost for longer than a person may go unseen (0.5 s), in the middle of the picture: taking them out only
        // where they were found would have them appear and vanish, so they are left as filmed throughout.
        let long = ColourFinder(); long.blind = 0.40...0.62
        let split = EditRenderer(source: source, plan: plan, directory: work, finder: long)
        let whole = try await passerSeen(try await split.render(options: EditOptions()) { _ in }, at: times, offset: 0.5)
        XCTAssertEqual(split.removedPeople, 0)
        XCTAssertEqual(whole, [Bool](repeating: true, count: times.count))
    }

    func testAPasserByOverAPlaceWhereTheWallWasNeverSeenIsLeftAloneFromFirstToLast() async throws {
        // As in the test of the climber who stays in one place, but the wall is only looked at while the
        // climber is still there: where it sat, the empty wall is not known. The passer-by who later walks
        // over that place is not taken out anywhere, not even where the wall is known.
        let w = CGFloat(Self.width), h = CGFloat(Self.height)
        let low = (x: 0.42, y: 0.5), high = (x: 0.42, y: 0.2)
        func walker(_ time: Double) -> (x: Double, y: Double) { (0.42 + (time - 2.6) * 0.6, 0.5) }
        let source = try await SyntheticWallClip.make(width: Self.width, height: Self.height, seconds: 3) { context, frame in
            let time = Double(frame) / 30
            func square(_ centre: (x: Double, y: Double), _ side: Double, _ r: CGFloat, _ g: CGFloat, _ b: CGFloat) {
                context.setFillColor(CGColor(red: r / 255, green: g / 255, blue: b / 255, alpha: 1))
                context.fill(CGRect(x: (centre.x - side / 2) * w, y: (1 - centre.y) * h - side * w / 2, width: side * w, height: side * w))
            }
            square(time < 2.1 ? low : high, 0.08, 40, 200, 70)
            square(walker(time), 0.10, 200, 30, 160)
        }
        defer { try? FileManager.default.removeItem(at: source) }
        func plan(times: String) throws -> EditPlan {
            try XCTUnwrap(EditPlan(json: #"{"v":1,"duration":3,"trim":[1,3],"segments":[{"t0":1,"t1":3,"speed":1}],"plate":{"times":\#(times)},"# +
                #""keep":{"feather":0.03,"boxes":[{"t":0,"b":[[0.36,0.4,0.12,0.2]]},{"t":2.0,"b":[[0.36,0.4,0.12,0.2]]},{"t":2.2,"b":[[0.36,0.05,0.12,0.25]]},{"t":3,"b":[[0.36,0.05,0.12,0.25]]}]}}"#))
        }
        let unknown = EditRenderer(source: source, plan: try plan(times: "[0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,1.9]"), directory: work, finder: ColourFinder())
        let left = try await unknown.render(options: EditOptions()) { _ in }
        XCTAssertEqual(unknown.removedPeople, 0)
        for time in [2.3, 2.45, 2.6, 2.75, 2.9] {       // Before the place, on it, after it: there throughout.
            let c = try await colour(left, at: time - 1, walker(time)); XCTAssertTrue(isMagenta(c), "\(time): \(c)")
        }
        // With the wall also looked at after the climber has moved on, the same passer-by goes, throughout.
        let known = EditRenderer(source: source, plan: try plan(times: "[0,0.37,0.75,1.12,1.5,1.87,2.25,2.62,2.95]"), directory: work, finder: ColourFinder())
        let taken = try await known.render(options: EditOptions()) { _ in }
        XCTAssertEqual(known.removedPeople, 1)
        for time in [2.3, 2.45, 2.6, 2.75, 2.9] {
            let c = try await colour(taken, at: time - 1, walker(time)); XCTAssertTrue(isWall(c), "\(time): \(c)")
        }
        do { let c = try await colour(taken, at: 1.6, high); XCTAssertTrue(isGreen(c), "\(c)") }
    }

    func testAPersonWhoIsInThePlateOrWhoseSurroundingsDoNotMatchItIsLeftAlone() {
        // The small pictures the decision is made on: a grey wall, known everywhere but in one corner.
        let w = 36, h = 64
        var valid = [UInt8](repeating: 255, count: w * h)
        for y in 0..<10 { for x in 0..<10 { valid[y * w + x] = 0 } }
        let wall = [UInt8](repeating: 120, count: w * h * 4)
        let small = SmallPictures(width: w, height: h, valid: valid, plate: wall, reach: 3)
        let person = small.outline(of: EditPlan.Box(x: 0.4, y: 0.5, w: 0.2, h: 0.2))
        let region = small.grown(person, by: small.reach)
        XCTAssertEqual(region.filter { $0 != 0 }.count, (8 + 6) * (13 + 6))                        // Grown by three on every side.
        let keep = [EditPlan.Box(x: 0.7, y: 0.1, w: 0.2, h: 0.3)]
        XCTAssertTrue(small.clean(region: region, keep: keep, frame: wall))
        XCTAssertTrue(small.clean(region: region, keep: keep, frame: nil))
        // The surroundings are a little different all over (the light changed): not clean. A slight difference is.
        XCTAssertFalse(small.clean(region: region, keep: keep, frame: [UInt8](repeating: 160, count: w * h * 4)))
        XCTAssertTrue(small.clean(region: region, keep: keep, frame: [UInt8](repeating: 128, count: w * h * 4)))
        // Another person right next to them says nothing about the wall and is not held against them.
        var crowded = wall
        let neighbour = small.grown(small.outline(of: EditPlan.Box(x: 0.62, y: 0.5, w: 0.2, h: 0.2)), by: small.reach)
        for index in neighbour.indices where neighbour[index] != 0 { for channel in 0..<3 { crowded[index * 4 + channel] = 250 } }
        XCTAssertFalse(small.clean(region: region, keep: keep, frame: crowded))
        XCTAssertTrue(small.clean(region: region, keep: keep, frame: crowded, others: [neighbour]))
        // Touching a keep box (grown a little), or lying over a place where the plate is not known: not clean.
        XCTAssertFalse(small.clean(region: region, keep: [EditPlan.Box(x: 0.6, y: 0.4, w: 0.2, h: 0.3)], frame: wall))
        XCTAssertFalse(small.clean(region: small.grown(small.outline(of: EditPlan.Box(x: 0.2, y: 0.1, w: 0.2, h: 0.2)), by: small.reach), keep: keep, frame: wall))
        // A box with wild numbers is an outline of nothing or of what is inside the picture; it is never a crash.
        XCTAssertFalse(small.outline(of: EditPlan.Box(x: 1e30, y: 0.1, w: 0.2, h: 0.2)).contains(1))
        XCTAssertFalse(small.outline(of: EditPlan.Box(x: .nan, y: .infinity, w: 1, h: 1)).contains(1))
        XCTAssertEqual(small.outline(of: EditPlan.Box(x: -1e30, y: -5, w: 2e30, h: 9)).filter { $0 != 0 }.count, w * h)
        // Outlines survive being packed away.
        XCTAssertEqual(small.unpacked(SmallPictures.packed(person)), person)
    }

    func testAPasserByWhoCrossesSomeoneWhoStaysIsKeptAndThatPersonIsNeverPaintedOver() async throws {
        // A passer-by (magenta) walks along the bottom. A second person (green) is in the picture for a moment,
        // right above the passer-by's path; they appear and vanish in the middle of the picture, so they stay.
        // Everything else about the passer-by is clean, but what would replace them would lie over the other.
        let w = CGFloat(Self.width), h = CGFloat(Self.height)
        func make(other: (x: Double, y: Double)) async throws -> URL {
            try await SyntheticWallClip.make(width: Self.width, height: Self.height, seconds: 3) { context, frame in
                let time = Double(frame) / 30
                func square(_ centre: (x: Double, y: Double), _ side: Double, _ r: CGFloat, _ g: CGFloat, _ b: CGFloat) {
                    context.setFillColor(CGColor(red: r / 255, green: g / 255, blue: b / 255, alpha: 1))
                    context.fill(CGRect(x: (centre.x - side / 2) * w, y: (1 - centre.y) * h - side * w / 2, width: side * w, height: side * w))
                }
                if time >= 1.2, time < 1.8 { square(other, 0.08, 40, 200, 70) }
                square(Self.passer(time), 0.10, 200, 30, 160)
            }
        }
        // The keep box is up on the wall, away from both; nobody is in it.
        let plan = try plan(#""segments":[{"t0":0.5,"t1":2.5,"speed":1}],"keep":{"feather":0.03,"boxes":[{"t":0,"b":[[0.6,0.1,0.2,0.3]]}]},"plate":{"times":[0,0.37,0.75,1.12,1.5,1.87,2.25,2.62,2.95]}"#)
        let times = [0.6, 0.9, 1.2, 1.35, 1.5, 1.65, 1.8, 2.1, 2.4]
        let above = (x: 0.5, y: 0.8)                                // Its lower edge is just above the passer-by's upper edge.
        let crossing = try await make(other: above)
        defer { try? FileManager.default.removeItem(at: crossing) }
        let renderer = EditRenderer(source: crossing, plan: plan, directory: work, finder: ColourFinder())
        let url = try await renderer.render(options: EditOptions()) { _ in }
        XCTAssertEqual(renderer.removedPeople, 0)
        let seen = try await passerSeen(url, at: times, offset: 0.5)
        XCTAssertEqual(seen, [Bool](repeating: true, count: times.count))          // The passer-by: there throughout.
        for time in [1.25, 1.35, 1.45, 1.5, 1.55, 1.65, 1.75] {                      // The other, down to the edge nearest the passer-by: untouched in every frame.
            for point in [above, (x: 0.5, y: 0.818), (x: 0.47, y: 0.818), (x: 0.53, y: 0.818)] {
                let c = try await colour(url, at: time - 0.5, point); XCTAssertTrue(isGreen(c), "\(time) \(point): \(c)")
            }
        }
        // The same two people with the other one well away from the path: the passer-by goes, the other stays as filmed.
        let apart = try await make(other: (x: 0.3, y: 0.45))
        defer { try? FileManager.default.removeItem(at: apart) }
        let clear = EditRenderer(source: apart, plan: plan, directory: work, finder: ColourFinder())
        let gone = try await clear.render(options: EditOptions()) { _ in }
        XCTAssertEqual(clear.removedPeople, 1)
        let left = try await passerSeen(gone, at: times, offset: 0.5)
        XCTAssertEqual(left, [Bool](repeating: false, count: times.count))
        do { let c = try await colour(gone, at: 1.0, (x: 0.3, y: 0.45)); XCTAssertTrue(isGreen(c), "\(c)") }
    }

    /// Switches flipped one after another while a render is under way: each render waits for the one before
    /// it to stop, the last choice is the one that is shown, and nothing of the others is left behind.
    @MainActor
    func testFlippingSwitchesQuicklyEndsWithOneResultForTheLastChoice() async throws {
        let source = try await clip()
        defer { try? FileManager.default.removeItem(at: source) }
        let plan = try plan(#""segments":[{"t0":0.5,"t1":1.5,"speed":1},{"t0":1.5,"t1":2.5,"speed":0.5}],"camera":[{"t":0.5,"cx":0.5,"cy":0.5,"s":1},{"t":2.5,"cx":0.5,"cy":0.4,"s":1.5}],"# + keepJSON)
        let climb = Climb(id: UUID(), date: Date(), title: "", grade: ungradedValue, sent: false, note: "", videoName: "a.mov", thumbnailName: "a.jpg",
                          analysis: PoseAnalysis(frames: [], sampledFrames: 0, duration: 3, aspectRatio: 0.5625))
        let folder = EditFiles.directory(for: climb.id)
        defer { EditFiles.remove(for: climb.id) }
        let session = EditSession(climb: climb, plan: plan, source: source)
        session.start()
        session.start()                                           // Coming back to the page does not start another.
        for flip in 0..<6 {
            try await Task.sleep(for: .milliseconds(flip.isMultiple(of: 2) ? 30 : 120))
            switch flip % 3 {
            case 0: session.options.slowMotion.toggle()
            case 1: session.options.cameraMoves.toggle()
            default: session.options.removePassersBy.toggle()
            }
        }
        XCTAssertEqual(session.options, EditOptions.offered)       // Each switch flipped twice: back where it began.
        session.options.slowMotion = false
        let deadline = Date().addingTimeInterval(120)
        var result: URL?
        while result == nil, Date() < deadline {
            if case .ready(let url) = session.phase { result = url } else { try await Task.sleep(for: .milliseconds(50)) }
            if case .failed(let message) = session.phase { return XCTFail(message) }
        }
        let url = try XCTUnwrap(result)
        do { let seconds = try await AVURLAsset(url: url).load(.duration).seconds; XCTAssertEqual(seconds, 2.0, accuracy: 0.1) }   // The attempt alone: slow motion is off.
        func files(_ kind: String) -> [URL] { (FileManager.default.enumerator(at: folder, includingPropertiesForKeys: nil)?.compactMap { $0 as? URL } ?? []).filter { $0.pathExtension == kind } }
        XCTAssertEqual(files("mp4").count, 1); XCTAssertEqual(files("raw").count, 0)
        // Leaving the page takes the folder with it once the work has stopped.
        session.options.slowMotion = true
        session.close()
        let gone = Date().addingTimeInterval(60)
        while FileManager.default.fileExists(atPath: folder.path), Date() < gone { try await Task.sleep(for: .milliseconds(50)) }
        XCTAssertFalse(FileManager.default.fileExists(atPath: folder.path))
        if case .ready = session.phase { XCTFail("a render finished after the page was closed") }
    }

    /// The system's own people finder on a photo of a person. Skipped where the simulator cannot run it.
    func testSystemPeopleFinderOnAPhoto() throws {
        let url = try XCTUnwrap(Bundle(for: Self.self).url(forResource: "pose", withExtension: "jpg"))
        let image = try XCTUnwrap(CIImage(contentsOf: url))
        let context = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
        let people: [FoundPerson]
        do { people = try VisionPeopleFinder().people(in: image, context: context) }
        catch { throw XCTSkip("Vision cannot look for people here: \(error.localizedDescription)") }
        let found = XCTAttachment(string: people.map { "box \($0.box) mask \($0.mask?.extent.debugDescription ?? "none")" }.joined(separator: "\n"))
        found.name = "People found in the photo"; found.lifetime = .keepAlways; add(found)
        guard let person = people.max(by: { $0.box.w * $0.box.h < $1.box.w * $1.box.h }) else { throw XCTSkip("Vision ran and found nobody in the photo on this simulator") }
        XCTAssertGreaterThan(person.box.w * person.box.h, 0.02)
        XCTAssertTrue((0...1).contains(person.box.x) && (0...1).contains(person.box.y) && person.box.x + person.box.w <= 1.001 && person.box.y + person.box.h <= 1.001)
    }

    final class Progress: @unchecked Sendable {
        private let lock = NSLock()
        private var values: [Double] = []
        func note(_ value: Double) { lock.lock(); values.append(value); lock.unlock() }
        var last: Double? { lock.lock(); defer { lock.unlock() }; return values.last }
        var rising: Bool { lock.lock(); defer { lock.unlock() }; return zip(values, values.dropFirst()).allSatisfy { $0 <= $1 + 1e-9 } }
    }
}
