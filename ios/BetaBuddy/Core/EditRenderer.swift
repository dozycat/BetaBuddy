import AVFoundation
import CoreImage
import Vision

// Renders an edit from a plan (`EditPlan`). Everything here is general picture work: speed changes,
// a moving window, and putting the empty wall over people who are not the climber. When to slow
// down, where to look and what must stay live all come from the plan; nothing here decides them.
//
// A passer-by is taken out for the whole time they are seen, or not at all: the clip is looked
// through first, people are followed from frame to frame, and one moment that cannot be replaced
// cleanly leaves that person as filmed throughout. Nothing is forced.

/// A person found in one frame. `box` is in the upright picture's 0–1 coordinates, origin top-left.
/// `mask` is white where the person is, over any extent that stands for the whole picture; without
/// one the box itself is used.
struct FoundPerson {
    var box: EditPlan.Box
    var mask: CIImage?
}

/// Finds the people in an upright picture. Injected so the compositing can be tested without a model.
protocol PeopleFinder: AnyObject {
    func people(in image: CIImage, context: CIContext) throws -> [FoundPerson]
}

/// The system's own person segmentation: one mask per person where that is available, otherwise
/// rectangles around people. No model file and no network.
final class VisionPeopleFinder: PeopleFinder {
    /// Per-person outlines could not be had and rectangles are used instead.
    private(set) var instancesUnavailable = false
    /// Long side of the picture handed to Vision; masks are grown and softened afterwards anyway.
    static let side: CGFloat = 640

    func people(in image: CIImage, context: CIContext) throws -> [FoundPerson] {
        let extent = image.extent
        let scale = min(1, Self.side / max(extent.width, extent.height, 1))
        let small = image.transformed(by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY).concatenating(CGAffineTransform(scaleX: scale, y: scale)))
        let handler = VNImageRequestHandler(ciImage: small, options: [.ciContext: context])
        if !instancesUnavailable {
            do {
                let request = VNGeneratePersonInstanceMaskRequest()
                try handler.perform([request])
                guard let observation = request.results?.first else { return [] }
                return try Self.people(observation)
            } catch { instancesUnavailable = true }
        }
        let request = VNDetectHumanRectanglesRequest()
        request.upperBodyOnly = false
        try handler.perform([request])
        return (request.results ?? []).map {
            let box = $0.boundingBox   // Vision counts y from the bottom.
            return FoundPerson(box: EditPlan.Box(x: box.minX, y: 1 - box.maxY, w: box.width, h: box.height), mask: nil)
        }
    }

    /// One person per instance: the instance's own soft mask, and the box of what is at least half
    /// person in it. The observation's label picture is not used for the box: on real footage it
    /// marks stray specks all over the frame as part of a person, where the soft mask is clean.
    private static func people(_ observation: VNInstanceMaskObservation) throws -> [FoundPerson] {
        try observation.allInstances.compactMap { label -> FoundPerson? in
            let mask = try observation.generateMask(forInstances: IndexSet(integer: label))
            guard CVPixelBufferGetPixelFormatType(mask) == kCVPixelFormatType_OneComponent32Float else { return nil }
            CVPixelBufferLockBaseAddress(mask, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(mask, .readOnly) }
            let width = CVPixelBufferGetWidth(mask), height = CVPixelBufferGetHeight(mask), stride = CVPixelBufferGetBytesPerRow(mask)
            guard width > 0, height > 0, let base = CVPixelBufferGetBaseAddress(mask) else { return nil }
            var x0 = width, y0 = height, x1 = -1, y1 = -1
            for y in 0..<height {
                let row = base.advanced(by: y * stride).assumingMemoryBound(to: Float32.self)
                for x in 0..<width where row[x] >= 0.5 {
                    x0 = min(x0, x); x1 = max(x1, x); y0 = min(y0, y); y1 = max(y1, y)
                }
            }
            guard x1 >= x0, y1 >= y0 else { return nil }     // Nothing of this instance is clearly a person.
            return FoundPerson(box: EditPlan.Box(x: Double(x0) / Double(width), y: Double(y0) / Double(height),
                                                 w: Double(x1 - x0 + 1) / Double(width), h: Double(y1 - y0 + 1) / Double(height)),
                               mask: CIImage(cvPixelBuffer: mask, options: [.colorSpace: NSNull()]))
        }
    }
}

/// One frame of the result, as Core Image recipes. Pictures are upright; boxes and windows are in
/// 0–1 coordinates from the top-left, Core Image counts from the bottom-left.
enum EditFrame {
    /// How far a passer-by's shape is grown before its edge is softened, as a fraction of the short side.
    static let growth = 0.015

    static func rect(_ box: EditPlan.Box, in extent: CGRect) -> CGRect {
        CGRect(x: extent.minX + box.x * extent.width, y: extent.minY + (1 - box.y - box.h) * extent.height, width: box.w * extent.width, height: box.h * extent.height)
    }

    /// `live` with `plate` put where `mask` is white (a small picture standing for the whole frame,
    /// already grown and softened: `SmallPictures.soft`). Who is in the mask has been decided before
    /// (`EditRemoval`). Inside a keep box the picture is always live, and so is every pixel where
    /// the plate is not known (`valid` black there; `nil` = known everywhere).
    static func replacing(live: CIImage, plate: CIImage, valid: CIImage? = nil, mask: CIImage, keep: [EditPlan.Box]) -> CIImage {
        let extent = live.extent, size = mask.extent
        guard size.width > 0, size.height > 0, !size.isInfinite, extent.width > 0, extent.height > 0 else { return live }
        var mask = mask.clampedToExtent()
            .transformed(by: CGAffineTransform(translationX: -size.minX, y: -size.minY)
                .concatenating(CGAffineTransform(scaleX: extent.width / size.width, y: extent.height / size.height))
                .concatenating(CGAffineTransform(translationX: extent.minX, y: extent.minY)))
            .cropped(to: extent)
        for box in keep { mask = CIImage(color: .black).cropped(to: rect(box, in: extent)).composited(over: mask) }
        if let valid { mask = mask.applyingFilter("CIMinimumCompositing", parameters: [kCIInputBackgroundImageKey: valid]).cropped(to: extent) }
        return plate.applyingFilter("CIBlendWithRedMask", parameters: [kCIInputBackgroundImageKey: live, kCIInputMaskImageKey: mask]).cropped(to: extent)
    }

    /// The part of `image` inside `window`, enlarged to fill the picture.
    static func framed(_ image: CIImage, window: EditWindow) -> CIImage {
        let window = window.clamped, extent = image.extent
        guard window.s > 1.0001 else { return image }
        let crop = CGRect(x: extent.minX + window.x * extent.width, y: extent.minY + (1 - window.y - window.side) * extent.height,
                          width: window.side * extent.width, height: window.side * extent.height)
        return image.clampedToExtent()
            .transformed(by: CGAffineTransform(translationX: -crop.minX, y: -crop.minY).concatenating(CGAffineTransform(scaleX: window.s, y: window.s))
                .concatenating(CGAffineTransform(translationX: extent.minX, y: extent.minY)))
            .cropped(to: extent)
    }

    /// From a decoded frame (Core Image coordinates of the stored picture) to the upright picture at
    /// the render size. `preferred` is the track's transform, which counts y downwards.
    static func upright(natural: CGSize, preferred: CGAffineTransform, render: CGSize) -> CGAffineTransform {
        let display = CGRect(origin: .zero, size: natural).applying(preferred).standardized
        return CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: natural.height)
            .concatenating(preferred)
            .concatenating(CGAffineTransform(translationX: -display.minX, y: -display.minY))
            .concatenating(CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: display.height))
            .concatenating(CGAffineTransform(scaleX: render.width / max(1, display.width), y: render.height / max(1, display.height)))
    }
}

/// Per-pixel median across frames, for the empty wall.
enum PlateMedian {
    /// A pixel needs this many pictures in which it is not inside a keep box; with fewer it has no plate.
    static let minimumSamples = 3
    /// A part of one picture that does not count: columns `x0..<x1` of rows `y0..<y1`.
    typealias Hole = (x0: Int, x1: Int, y0: Int, y1: Int)

    /// `frames` holds `count` pictures back to back, each `height` rows of `width` pixels of
    /// `pixelBytes` bytes. Writes to `out` (rows `outRowBytes` apart), byte by byte, the median of
    /// the pictures that count for that pixel: the middle one, the lower of the two middle ones when
    /// their number is even. `holes[i]` are the parts of picture `i` that do not count; `valid` (one
    /// byte a pixel, rows `validRowBytes` apart) is set to 255 where at least `minimumSamples`
    /// pictures counted and 0 elsewhere. Returns `false` when stopped early.
    static func median(frames: UnsafeRawPointer, count: Int, width: Int, pixelBytes: Int = 4, height: Int, holes: [[Hole]] = [],
                       out: UnsafeMutableRawPointer, outRowBytes: Int, valid: UnsafeMutableRawPointer? = nil, validRowBytes: Int = 0, stop: () -> Bool = { false }) -> Bool {
        let rowBytes = width * pixelBytes
        guard count > 0, count < 256, rowBytes > 0, height > 0 else { return false }
        let padded = (rowBytes + 63) / 64 * 64
        let scratch = UnsafeMutableRawPointer.allocate(byteCount: padded * count, alignment: 64)
        defer { scratch.deallocate() }
        scratch.initializeMemory(as: UInt8.self, repeating: 0, count: padded * count)
        var skipped = [UInt8](repeating: 0, count: width)      // In how many pictures this pixel does not count.
        var marked = [Int32](repeating: -1, count: width)      // The last picture in which it was marked, so that overlapping holes count once.
        for y in 0..<height {
            if y % 32 == 0, stop() { return false }
            var any = false
            for x in 0..<width { skipped[x] = 0; marked[x] = -1 }
            for index in 0..<count {
                let row = scratch + index * padded
                memcpy(row, frames + (index * height + y) * rowBytes, rowBytes)
                guard index < holes.count else { continue }
                // A pixel that does not count is given the highest value: sorted, the ones that count come first.
                for hole in holes[index] where y >= hole.y0 && y < hole.y1 {
                    let from = min(width, max(0, hole.x0)), to = min(width, max(from, hole.x1))
                    for x in from..<to where marked[x] != Int32(index) {
                        marked[x] = Int32(index)
                        skipped[x] += 1
                        memset(row + x * pixelBytes, 255, pixelBytes)
                        any = true
                    }
                }
            }
            // Bubble the largest values up, a whole row at a time: all the way when the middle differs
            // from pixel to pixel, otherwise only until the middle one is in place.
            let middle = (count - 1) / 2
            for pass in 0..<(any ? count - 1 : count - middle) {
                for index in 0..<(count - 1 - pass) {
                    let a = scratch + index * padded, b = a + padded
                    var offset = 0
                    while offset < padded {
                        let low = (a + offset).load(as: SIMD64<UInt8>.self), high = (b + offset).load(as: SIMD64<UInt8>.self)
                        (a + offset).storeBytes(of: pointwiseMin(low, high), as: SIMD64<UInt8>.self)
                        (b + offset).storeBytes(of: pointwiseMax(low, high), as: SIMD64<UInt8>.self)
                        offset += 64
                    }
                }
            }
            let target = out + y * outRowBytes
            if any {
                for x in 0..<width {
                    let usable = count - Int(skipped[x])
                    if usable > 0 { memcpy(target + x * pixelBytes, scratch + ((usable - 1) / 2) * padded + x * pixelBytes, pixelBytes) }
                    else { memset(target + x * pixelBytes, 0, pixelBytes) }
                }
            } else {
                memcpy(target, scratch + middle * padded, rowBytes)
            }
            if let valid {
                let row = (valid + y * validRowBytes).assumingMemoryBound(to: UInt8.self)
                for x in 0..<width { row[x] = count - Int(skipped[x]) >= minimumSamples ? 255 : 0 }
            }
        }
        return true
    }
}

/// The empty wall and where it is known: `valid` is white where the wall was seen often enough.
struct EditPlate {
    var picture: CVPixelBuffer
    var valid: CVPixelBuffer
}

/// Small copies of pictures and outlines, on which it is decided whether a person can be replaced
/// cleanly. One byte a pixel for outlines (0 or 1), four for pictures (RGBA); the first row is the top.
struct SmallPictures {
    let width: Int, height: Int
    /// 255 where the plate is known.
    let valid: [UInt8]
    let plate: [UInt8]
    /// How far (pixels here) what is put in a person's place can reach beyond their outline.
    let reach: Int
    /// How much the plate and the frame may differ on average just outside a person, per colour channel out of 255.
    static let tolerance = 14.0
    /// Short side of the small copies.
    static let side = 144.0

    /// How what is put in a person's place spreads beyond their outline, in pixels here: grown by
    /// `grow`, then softened by two passes of a blur `blur` wide either side, which leaves everything
    /// inside the outline fully replaced and an edge about `feather` of the short side (`side`) wide.
    /// Nothing changes further than `reach` from the outline.
    static func spread(feather: Double, side: Int) -> (grow: Int, blur: Int, reach: Int) {
        let blur = max(1, Int((feather * Double(side) / 4).rounded()))
        let grow = Int((EditFrame.growth * Double(side)).rounded(.up)) + 2 * blur
        return (grow, blur, grow + 2 * blur + 1)
    }

    /// The mask for a frame: `shapes` together, grown a little and given a soft edge of `feather`
    /// (fraction of the short side). 255 where the plate goes, 0 where the frame stays. Worked out
    /// here, on a small picture, so that rendering a frame is one plain pass.
    func soft(_ shapes: [[UInt8]], feather: Double) -> [UInt8] {
        var all = [UInt8](repeating: 0, count: width * height)
        for shape in shapes where shape.count == all.count { for index in shape.indices where shape[index] != 0 { all[index] = 1 } }
        let spread = Self.spread(feather: feather, side: min(width, height))
        var picture = grown(all, by: spread.grow).map { $0 == 0 ? UInt8(0) : 255 }
        // Two passes of an average over `blur` pixels either side, across and then down.
        for _ in 0..<2 {
            var across = picture
            for y in 0..<height {
                var sum = 0
                for x in -spread.blur...spread.blur { sum += Int(picture[y * width + min(width - 1, max(0, x))]) }
                for x in 0..<width {
                    across[y * width + x] = UInt8(sum / (2 * spread.blur + 1))
                    sum += Int(picture[y * width + min(width - 1, x + spread.blur + 1)]) - Int(picture[y * width + max(0, x - spread.blur)])
                }
            }
            for x in 0..<width {
                var sum = 0
                for y in -spread.blur...spread.blur { sum += Int(across[min(height - 1, max(0, y)) * width + x]) }
                for y in 0..<height {
                    picture[y * width + x] = UInt8(sum / (2 * spread.blur + 1))
                    sum += Int(across[min(height - 1, y + spread.blur + 1) * width + x]) - Int(across[max(0, y - spread.blur) * width + x])
                }
            }
        }
        return picture
    }

    /// Every pixel within `radius` (in both directions) of a set pixel.
    func grown(_ shape: [UInt8], by radius: Int) -> [UInt8] {
        guard radius > 0 else { return shape }
        var across = [UInt8](repeating: 0, count: shape.count), result = across
        for y in 0..<height {
            for x in 0..<width where shape[y * width + x] != 0 {
                for near in max(0, x - radius)...min(width - 1, x + radius) { across[y * width + near] = 1 }
            }
        }
        for x in 0..<width {
            for y in 0..<height where across[y * width + x] != 0 {
                for near in max(0, y - radius)...min(height - 1, y + radius) { result[near * width + x] = 1 }
            }
        }
        return result
    }

    func outline(of box: EditPlan.Box) -> [UInt8] {
        var shape = [UInt8](repeating: 0, count: width * height)
        let columns = EditPlan.Box.cells(box.x, box.w, of: width)
        for y in EditPlan.Box.cells(box.y, box.h, of: height) { for x in columns { shape[y * width + x] = 1 } }
        return shape
    }

    /// Whether everything that would be put in this person's place is clean: `region` (their outline
    /// grown by `reach`) touches no keep box (each grown a little), the plate is known all over it,
    /// and, when the frame itself is given, plate and frame agree just outside it. `others` are the
    /// regions of the other people in the frame, which say nothing about the wall.
    func clean(region: [UInt8], keep: [EditPlan.Box], frame: [UInt8]?, others: [[UInt8]] = []) -> Bool {
        verdict(region: region, keep: keep, frame: frame, others: others) == nil
    }

    /// `nil` when clean; otherwise what stands in the way.
    func verdict(region: [UInt8], keep: [EditPlan.Box], frame: [UInt8]?, others: [[UInt8]] = []) -> EditTracks.Failure? {
        let boxes = keep.map { $0.grown(by: EditTracks.margin) }.map { outline(of: $0) }
        var unknown = false
        for index in region.indices where region[index] != 0 {
            if boxes.contains(where: { $0[index] != 0 }) { return .keepBox }
            if valid[index] < 250 { unknown = true }
        }
        if unknown { return .noPlate }
        guard let frame else { return nil }
        let ring = grown(region, by: 3)
        var around = 0, compared = 0, difference = 0
        for index in ring.indices where ring[index] != 0 && region[index] == 0 {
            around += 1
            guard valid[index] >= 250, !others.contains(where: { $0[index] != 0 }) else { continue }
            compared += 1
            for channel in 0..<3 { difference += abs(Int(frame[index * 4 + channel]) - Int(plate[index * 4 + channel])) }
        }
        guard around > 0 else { return region.contains(1) ? .plateMismatch : nil }     // A region that fills the picture has nothing around it to go by.
        guard compared >= max(6, around / 4) else { return .plateMismatch }             // Too little of the surroundings can be compared.
        return Double(difference) / Double(compared * 3) <= Self.tolerance ? nil : .plateMismatch
    }

    static func packed(_ shape: [UInt8]) -> Data {
        var data = Data(count: (shape.count + 7) / 8)
        data.withUnsafeMutableBytes { bytes in
            for index in shape.indices where shape[index] != 0 { bytes[index / 8] |= 1 << UInt8(index % 8) }
        }
        return data
    }
    func unpacked(_ data: Data) -> [UInt8] {
        var shape = [UInt8](repeating: 0, count: width * height)
        guard data.count * 8 >= width * height else { return shape }
        for index in shape.indices where data[data.startIndex + index / 8] & (1 << UInt8(index % 8)) != 0 { shape[index] = 1 }
        return shape
    }
}

/// Who is taken out of the picture and what goes in their place. Made once per clip, before any
/// frame of the result; the frames only read it.
final class EditRemoval: @unchecked Sendable {
    let plate: CIImage
    let valid: CIImage
    let tracks: EditTracks
    /// The outlines the tracks point at, packed; empty for people who are left as filmed.
    let shapes: [Data]
    let small: SmallPictures
    /// How far apart the moments looked at were, in seconds.
    let step: Double

    /// How many frames were looked at for people, and in how many of them looking failed. For reports.
    let looked: Int, unseen: Int

    init(plate: EditPlate, tracks: EditTracks, shapes: [Data], small: SmallPictures, step: Double, looked: Int = 0, unseen: Int = 0) {
        self.looked = looked; self.unseen = unseen
        self.plate = CIImage(cvPixelBuffer: plate.picture, options: [.colorSpace: NSNull()])
        valid = CIImage(cvPixelBuffer: plate.valid, options: [.colorSpace: NSNull()])
        self.tracks = tracks; self.shapes = shapes; self.small = small; self.step = step
    }

    /// How many people are taken out somewhere in the clip.
    var removed: Int { tracks.tracks.filter { $0.removable && !$0.samples.isEmpty }.count }

    /// The mask for the frame of the clip at `time`: everyone who is taken out then, each by the
    /// outline found at that moment or, where they went unseen for a moment, a box between the ones
    /// before and after; grown and softened. `nil` when there is nobody to take out in this frame.
    /// Nothing here changes after `init`: any number of renders may read it at once.
    func mask(at time: Double, feather: Double) -> CIImage? {
        let people = tracks.shapes(at: time, step: step)
        guard !people.isEmpty else { return nil }
        let outlines = people.map { shape -> [UInt8] in
            guard let index = shape.shape, shapes.indices.contains(index), !shapes[index].isEmpty else { return small.outline(of: shape.box) }
            return small.unpacked(shapes[index])
        }
        return CIImage(bitmapData: Data(small.soft(outlines, feather: feather)), bytesPerRow: small.width, size: CGSize(width: small.width, height: small.height), format: .R8, colorSpace: nil)
    }
}

/// What the frames of one render share: made for that render and used by nothing else. The one
/// thing in it that changes, the last frame's people, is only touched on the rendering queue.
final class EditJob: @unchecked Sendable {
    let upright: CGAffineTransform
    let renderSize: CGSize
    let keep: EditPlan.Keep?
    /// `nil` when passers-by are not being removed.
    let removal: EditRemoval?
    let context: CIContext
    private var last: (time: Double, mask: CIImage?)?

    /// The mask of who is replaced at `time`. A slowed frame is asked for more than once and worked out once.
    func mask(at time: Double, feather: Double) -> CIImage? {
        guard let removal else { return nil }
        if let last, abs(last.time - time) < 0.002 { return last.mask }
        let mask = removal.mask(at: time, feather: feather)
        last = (time, mask)
        return mask
    }

    init(upright: CGAffineTransform, renderSize: CGSize, keep: EditPlan.Keep? = nil, removal: EditRemoval? = nil, context: CIContext) {
        self.upright = upright; self.renderSize = renderSize; self.keep = keep; self.removal = removal; self.context = context
    }
}

final class EditInstruction: NSObject, AVVideoCompositionInstructionProtocol {
    let timeRange: CMTimeRange
    let enablePostProcessing = false
    let containsTweening = true
    let requiredSourceTrackIDs: [NSValue]?
    let passthroughTrackID = kCMPersistentTrackID_Invalid
    let track: CMPersistentTrackID
    let job: EditJob
    /// The stretch of the clip playing in this range; `nil` to hand the upright frame through as it is (for the plate).
    let piece: EditTimeline.Piece?

    init(timeRange: CMTimeRange, track: CMPersistentTrackID, job: EditJob, piece: EditTimeline.Piece?) {
        self.timeRange = timeRange; self.track = track; self.job = job; self.piece = piece
        requiredSourceTrackIDs = [NSNumber(value: track)]
    }
}

final class EditCompositor: NSObject, AVVideoCompositing, @unchecked Sendable {
    private static let format: [String: any Sendable] = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA), kCVPixelBufferMetalCompatibilityKey as String: true]
    var sourcePixelBufferAttributes: [String: any Sendable]? { Self.format }
    var requiredPixelBufferAttributesForRenderContext: [String: any Sendable] { Self.format }
    /// One queue for every picture this file renders, whichever compositor or pass asks: the frames
    /// of a render, the frames read for the plate, and the looking-through before it. Nothing here
    /// renders two pictures at once.
    static let work = DispatchQueue(label: "com.betabuddy.ios.edit")
    private var queue: DispatchQueue { Self.work }
    private let lock = NSLock()
    /// Goes up with every cancellation; a request asked for before the latest one is not rendered.
    private var generation = 0

    func renderContextChanged(_ newRenderContext: AVVideoCompositionRenderContext) {}

    /// Returns at once. Every request still waiting is finished as cancelled when its turn comes,
    /// without being rendered; the one being rendered finishes.
    func cancelAllPendingVideoCompositionRequests() {
        lock.lock(); generation += 1; lock.unlock()
    }

    private var current: Int { lock.lock(); defer { lock.unlock() }; return generation }

    func startRequest(_ request: AVAsynchronousVideoCompositionRequest) {
        let asked = current
        queue.async {
            guard asked == self.current else { request.finishCancelledRequest(); return }
            autoreleasepool {
                guard let instruction = request.videoCompositionInstruction as? EditInstruction,
                      let source = request.sourceFrame(byTrackID: instruction.track),
                      let target = request.renderContext.newPixelBuffer() else {
                    request.finish(with: CocoaError(.fileReadCorruptFile))
                    return
                }
                let job = instruction.job
                let bounds = CGRect(origin: .zero, size: job.renderSize)
                var image = CIImage(cvPixelBuffer: source, options: [.colorSpace: NSNull()]).transformed(by: job.upright).cropped(to: bounds)
                if let piece = instruction.piece {
                    let time = piece.sourceTime(atOutput: request.compositionTime.seconds)
                    if let removal = job.removal, let keep = job.keep, let mask = job.mask(at: time, feather: keep.feather) {
                        image = EditFrame.replacing(live: image, plate: removal.plate, valid: removal.valid, mask: mask, keep: keep.boxes(at: time))
                    }
                    image = EditFrame.framed(image, window: EditWindow.at(time, keys: piece.camera, linear: piece.linear))
                }
                job.context.render(image, to: target, bounds: bounds, colorSpace: nil)
                request.finish(withComposedVideoFrame: target)
            }
        }
    }
}

/// Makes the edit of one clip. The plate is worked out once and reused when only a switch changes.
final class EditRenderer: @unchecked Sendable {
    private let source: URL
    private let plan: EditPlan
    private let directory: URL
    private let finder: PeopleFinder
    private let context = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull(), .cacheIntermediates: false])
    private let lock = NSLock()
    /// Worked out once per clip and never changed afterwards; only handed from one render to the next.
    private var surveyed: EditRemoval?
    private var removal: EditRemoval? {
        get { lock.lock(); defer { lock.unlock() }; return surveyed }
        set { lock.lock(); surveyed = newValue; lock.unlock() }
    }

    init(source: URL, plan: EditPlan, directory: URL, finder: PeopleFinder = VisionPeopleFinder()) {
        self.source = source; self.plan = plan; self.directory = directory; self.finder = finder
    }

    private struct Clip {
        var asset: AVURLAsset
        var track: AVAssetTrack
        /// The video track's own length: what there is to show.
        var duration: CMTime
        /// The whole file's length, which an instruction for reading it has to cover.
        var whole: CMTime
        var upright: CGAffineTransform
        var renderSize: CGSize
        var frame: CMTime
    }

    private func clip() async throws -> Clip {
        let asset = AVURLAsset(url: source)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw CocoaError(.fileReadCorruptFile) }
        let (natural, preferred, rate, range) = try await track.load(.naturalSize, .preferredTransform, .nominalFrameRate, .timeRange)
        let whole = try await asset.load(.duration)
        let display = CGRect(origin: .zero, size: natural).applying(preferred).standardized.size
        guard display.width >= 16, display.height >= 16, range.duration.seconds > 0 else { throw CocoaError(.fileReadCorruptFile) }
        let scale = min(1, 1920 / max(display.width, display.height))
        func even(_ value: CGFloat) -> CGFloat { max(2, (value * scale / 2).rounded() * 2) }
        let size = CGSize(width: even(display.width), height: even(display.height))
        let fps = Int32(min(60, max(1, rate.isFinite && rate > 0 ? rate.rounded() : 30)))
        return Clip(asset: asset, track: track, duration: range.duration, whole: max(whole, range.end), upright: EditFrame.upright(natural: natural, preferred: preferred, render: size),
                    renderSize: size, frame: CMTime(value: 1, timescale: fps))
    }

    private func composition(_ clip: Clip, instructions: [EditInstruction]) -> AVMutableVideoComposition {
        let composition = AVMutableVideoComposition()
        composition.customVideoCompositorClass = EditCompositor.self
        composition.renderSize = clip.renderSize
        composition.frameDuration = clip.frame
        composition.instructions = instructions
        // Frames reach the compositor, and leave it, as ordinary 8-bit Rec. 709.
        composition.colorPrimaries = AVVideoColorPrimaries_ITU_R_709_2
        composition.colorTransferFunction = AVVideoTransferFunction_ITU_R_709_2
        composition.colorYCbCrMatrix = AVVideoYCbCrMatrix_ITU_R_709_2
        return composition
    }

    /// The upright frames at the plan's plate times, written to a file one after another, then their
    /// median, leaving out of each frame what is inside that moment's keep boxes (grown a little):
    /// whoever the plan keeps live is not part of the empty wall, even where they stay for a while.
    private func makePlate(_ clip: Clip, keep: EditPlan.Keep, progress: @escaping @Sendable (Double) -> Void) async throws -> EditPlate {
        let width = Int(clip.renderSize.width), height = Int(clip.renderSize.height), rowBytes = width * 4
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let file = directory.appendingPathComponent("plate-\(UUID().uuidString).raw")
        defer { try? FileManager.default.removeItem(at: file) }
        guard FileManager.default.createFile(atPath: file.path, contents: nil) else { throw CocoaError(.fileWriteUnknown) }
        let handle = try FileHandle(forWritingTo: file)
        var count = 0
        var holes: [[PlateMedian.Hole]] = []
        let job = EditJob(upright: clip.upright, renderSize: clip.renderSize, context: context)
        let whole = EditInstruction(timeRange: CMTimeRange(start: .zero, duration: clip.whole), track: clip.track.trackID, job: job, piece: nil)
        let times = Array(Set(plan.plateTimes.map { ($0 * 100).rounded() / 100 })).sorted()
        do {
            for (index, time) in times.enumerated() {
                try Task.checkCancellation()
                let reader = try AVAssetReader(asset: clip.asset)
                let output = AVAssetReaderVideoCompositionOutput(videoTracks: [clip.track], videoSettings: [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)])
                output.videoComposition = composition(clip, instructions: [whole])
                output.alwaysCopiesSampleData = false
                guard reader.canAdd(output) else { throw CocoaError(.fileReadCorruptFile) }
                reader.add(output)
                // A short stretch from that moment, never past the end of the clip: the first frame in it is the one.
                let start = CMTime(seconds: min(max(0, time), max(0, clip.duration.seconds - 0.2)), preferredTimescale: 600)
                reader.timeRange = CMTimeRange(start: start, end: min(clip.duration, start + CMTime(seconds: 0.5, preferredTimescale: 600)))
                guard reader.startReading() else { throw reader.error ?? CocoaError(.fileReadCorruptFile) }
                if let sample = output.copyNextSampleBuffer(), let buffer = CMSampleBufferGetImageBuffer(sample),
                   CVPixelBufferGetWidth(buffer) == width, CVPixelBufferGetHeight(buffer) == height {
                    CVPixelBufferLockBaseAddress(buffer, .readOnly)
                    if let base = CVPixelBufferGetBaseAddress(buffer) {
                        let stride = CVPixelBufferGetBytesPerRow(buffer)
                        var packed = Data(count: rowBytes * height)
                        packed.withUnsafeMutableBytes { bytes in
                            for row in 0..<height { memcpy(bytes.baseAddress! + row * rowBytes, base + row * stride, rowBytes) }
                        }
                        try handle.write(contentsOf: packed)
                        count += 1
                        // The frame that came is the first one at or after the moment asked for.
                        let shown = CMSampleBufferGetPresentationTimeStamp(sample).seconds
                        holes.append(keep.boxes(at: shown.isFinite ? shown : time).map { $0.grown(by: EditTracks.margin) }.map {
                            let columns = EditPlan.Box.cells($0.x, $0.w, of: width), rows = EditPlan.Box.cells($0.y, $0.h, of: height)
                            return (x0: columns.lowerBound, x1: columns.upperBound, y0: rows.lowerBound, y1: rows.upperBound)
                        })
                    }
                    CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
                }
                reader.cancelReading()
                progress(Double(index + 1) / Double(times.count) * 0.8)
            }
            try handle.close()
        } catch {
            try? handle.close()
            throw error
        }
        // Too few pictures and whoever stands in them is in the plate as well.
        guard count >= 3 else { throw CocoaError(.fileReadCorruptFile) }
        var made: CVPixelBuffer?, mask: CVPixelBuffer?
        let surface = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary] as CFDictionary
        CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, surface, &made)
        CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_OneComponent8, surface, &mask)
        guard let plate = made, let valid = mask else { throw CocoaError(.fileWriteUnknown) }
        let frames = try Data(contentsOf: file, options: .alwaysMapped)
        CVPixelBufferLockBaseAddress(plate, []); CVPixelBufferLockBaseAddress(valid, [])
        let finished = frames.withUnsafeBytes { bytes -> Bool in
            guard let base = bytes.baseAddress, let out = CVPixelBufferGetBaseAddress(plate), let known = CVPixelBufferGetBaseAddress(valid) else { return false }
            return PlateMedian.median(frames: base, count: count, width: width, height: height, holes: holes, out: out, outRowBytes: CVPixelBufferGetBytesPerRow(plate),
                                      valid: known, validRowBytes: CVPixelBufferGetBytesPerRow(valid)) { Task.isCancelled }
        }
        CVPixelBufferUnlockBaseAddress(plate, []); CVPixelBufferUnlockBaseAddress(valid, [])
        try Task.checkCancellation()
        guard finished else { throw CocoaError(.fileReadCorruptFile) }
        progress(1)
        return EditPlate(picture: plate, valid: valid)
    }

    private func bitmap(_ image: CIImage, _ small: SmallPictures, format: CIFormat) -> [UInt8] {
        let extent = image.extent
        let fitted = image.transformed(by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY)
            .concatenating(CGAffineTransform(scaleX: CGFloat(small.width) / extent.width, y: CGFloat(small.height) / extent.height)))
        let bytes = format == .R8 ? 1 : 4
        var pixels = [UInt8](repeating: 0, count: small.width * small.height * bytes)
        context.render(fitted, toBitmap: &pixels, rowBytes: small.width * bytes, bounds: CGRect(x: 0, y: 0, width: small.width, height: small.height), format: format, colorSpace: nil)
        return pixels
    }

    /// Looks through every part of the clip the result can show, follows the people in it and decides,
    /// person by person, whether they can be taken out cleanly for as long as they are seen.
    private func survey(_ clip: Clip, keep: EditPlan.Keep, plate: EditPlate, progress: @escaping @Sendable (Double) -> Void) async throws -> EditRemoval {
        let short = min(clip.renderSize.width, clip.renderSize.height), q = min(1, SmallPictures.side / short)
        let width = max(8, Int((clip.renderSize.width * q).rounded())), height = max(8, Int((clip.renderSize.height * q).rounded()))
        let reach = SmallPictures.spread(feather: keep.feather, side: min(width, height)).reach
        let sizes = SmallPictures(width: width, height: height, valid: [], plate: [], reach: reach)
        let plateImage = CIImage(cvPixelBuffer: plate.picture, options: [.colorSpace: NSNull()])
        // Someone who hardly moved is in the plate as well. Where the plate itself shows a person it is not the empty wall.
        let standing = EditCompositor.work.sync { (try? finder.people(in: plateImage, context: context)) ?? [] }
        if !standing.isEmpty {
            CVPixelBufferLockBaseAddress(plate.valid, [])
            if let base = CVPixelBufferGetBaseAddress(plate.valid) {
                let full = (w: CVPixelBufferGetWidth(plate.valid), h: CVPixelBufferGetHeight(plate.valid), stride: CVPixelBufferGetBytesPerRow(plate.valid))
                let margin = EditFrame.growth + 1.5 * keep.feather + EditTracks.margin
                for person in standing {
                    let box = person.box.grown(by: margin)
                    let columns = EditPlan.Box.cells(box.x, box.w, of: full.w)
                    guard !columns.isEmpty else { continue }
                    for y in EditPlan.Box.cells(box.y, box.h, of: full.h) { memset(base + y * full.stride + columns.lowerBound, 0, columns.count) }
                }
            }
            CVPixelBufferUnlockBaseAddress(plate.valid, [])
        }
        let small = EditCompositor.work.sync {
            SmallPictures(width: width, height: height, valid: bitmap(CIImage(cvPixelBuffer: plate.valid, options: [.colorSpace: NSNull()]), sizes, format: .R8),
                          plate: bitmap(plateImage, sizes, format: .RGBA8), reach: reach)
        }

        // Everything the result can show, whichever switches are on: the attempt and every segment.
        var ranges: [(Double, Double)] = []
        for range in ([(plan.trim.lowerBound, plan.trim.upperBound)] + plan.segments.map { ($0.t0, $0.t1) }).map({ (max(0, $0.0), min(clip.duration.seconds, $0.1)) }).filter({ $0.1 > $0.0 }).sorted(by: { $0.0 < $1.0 }) {
            if let last = ranges.last, range.0 <= last.1 { ranges[ranges.count - 1].1 = max(last.1, range.1) } else { ranges.append(range) }
        }
        let total = max(0.001, ranges.reduce(0) { $0 + $1.1 - $1.0 })
        let step = max(clip.frame.seconds, 1.0 / 30)
        var tracks = EditTracks()
        var shapes: [Data] = []
        var done = 0.0
        var looked = 0, unseen = 0      // Frames looked at, and those in which looking failed. For reports.
        let job = EditJob(upright: clip.upright, renderSize: clip.renderSize, context: context)
        let whole = EditInstruction(timeRange: CMTimeRange(start: .zero, duration: clip.whole), track: clip.track.trackID, job: job, piece: nil)
        for range in ranges {
            let reader = try AVAssetReader(asset: clip.asset)
            let output = AVAssetReaderVideoCompositionOutput(videoTracks: [clip.track], videoSettings: [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)])
            output.videoComposition = composition(clip, instructions: [whole])
            output.alwaysCopiesSampleData = false
            guard reader.canAdd(output) else { throw CocoaError(.fileReadCorruptFile) }
            reader.add(output)
            reader.timeRange = CMTimeRange(start: CMTime(seconds: range.0, preferredTimescale: 600), end: min(clip.duration, CMTime(seconds: range.1, preferredTimescale: 600)))
            guard reader.startReading() else { throw reader.error ?? CocoaError(.fileReadCorruptFile) }
            defer { reader.cancelReading() }
            var latest = -Double.infinity
            while let sample = output.copyNextSampleBuffer() {
                try Task.checkCancellation()
                let time = CMSampleBufferGetPresentationTimeStamp(sample).seconds
                guard time.isFinite, time - latest >= step - 0.25 * clip.frame.seconds, let buffer = CMSampleBufferGetImageBuffer(sample) else { continue }
                latest = time
                // On the queue the frames are rendered on, so that looking at one frame and rendering the next do not overlap.
                EditCompositor.work.sync { autoreleasepool {
                    let image = CIImage(cvPixelBuffer: buffer, options: [.colorSpace: NSNull()])
                    // A frame in which nobody could be looked for is a moment in which nobody was seen.
                    looked += 1
                    guard let people = try? finder.people(in: image, context: context) else { unseen += 1; return }
                    guard !people.isEmpty else { return }
                    let frame = bitmap(image, small, format: .RGBA8)
                    let outlines = people.map { person -> [UInt8] in
                        guard let mask = person.mask, mask.extent.width > 0, mask.extent.height > 0, !mask.extent.isInfinite else { return small.outline(of: person.box) }
                        return bitmap(mask, small, format: .R8).map { $0 >= 128 ? 1 : 0 }
                    }
                    let regions = outlines.map { small.grown($0, by: small.reach) }
                    let boxes = keep.boxes(at: time)
                    for (number, track) in tracks.add(time: time, boxes: people.map(\.box)).enumerated() {
                        guard tracks.tracks[track].removable else { continue }
                        if let reason = small.verdict(region: regions[number], keep: boxes, frame: frame, others: regions.indices.filter { $0 != number }.map { regions[$0] }) {
                            // One moment that would not come out clean: this person stays as filmed, from first to last.
                            tracks.fail(track, reason)
                        } else {
                            shapes.append(SmallPictures.packed(outlines[number]))
                            tracks.setShape(shapes.count - 1, track: track)
                        }
                    }
                } }
                progress(min(1, (done + time - range.0) / total))
            }
            if reader.status == .failed { throw reader.error ?? CocoaError(.fileReadCorruptFile) }
            done += range.1 - range.0
        }
        // Someone who was only found for part of the time they were in the picture is not taken out for that part.
        tracks.failUnseenComingOrGoing(ranges: ranges, step: step)
        // Moments inside a track at which the person was not found: a box stands in, and it has to be clean as well.
        for bridge in tracks.bridges(step: step) where tracks.tracks[bridge.track].removable {
            if let reason = small.verdict(region: small.grown(small.outline(of: bridge.box), by: small.reach), keep: keep.boxes(at: bridge.time), frame: nil) {
                tracks.fail(bridge.track, reason)
            }
        }
        // Nobody who stays may be painted over by someone who goes: whoever would cover a person that is kept,
        // at any moment they are replaced, is kept as well. First by boxes, then by the outline itself.
        let margin = Double(small.reach) / Double(min(small.width, small.height))
        tracks.failThoseOverKeptPeople(step: step) { moment, kept in
            guard moment.box.grown(by: margin).intersects(kept) else { return false }
            let outline = moment.shape.flatMap { shapes.indices.contains($0) && !shapes[$0].isEmpty ? small.unpacked(shapes[$0]) : nil } ?? small.outline(of: moment.box)
            let region = small.grown(outline, by: small.reach), other = small.outline(of: kept)
            return region.indices.contains { region[$0] != 0 && other[$0] != 0 }
        }
        // The outlines of people who stay are not needed any more.
        for track in tracks.tracks where !track.removable { for sample in track.samples { if let shape = sample.shape { shapes[shape] = Data() } } }
        progress(1)
        return EditRemoval(plate: plate, tracks: tracks, shapes: shapes, small: small, step: step, looked: looked, unseen: unseen)
    }

    /// What the last render with passer-by removal worked out about the clip; `nil` before one. For reports and tests.
    var survey: EditRemoval? { removal }

    /// How many people the last render with passer-by removal took out; `nil` before one. For tests and the local log.
    var removedPeople: Int? { removal?.removed }

    /// Renders with the given switches and returns the file. `progress` runs from 0 to 1.
    /// Cancelling the task stops the work and leaves no file behind. When nobody can be taken out
    /// cleanly the picture is simply left as it is.
    func render(options: EditOptions, progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        let clip = try await clip()
        let timeline = EditTimeline(plan: plan, options: options, sourceDuration: clip.duration.seconds)
        guard !timeline.pieces.isEmpty else { throw CocoaError(.fileReadCorruptFile) }
        let removing = options.removePassersBy && plan.keep != nil
        // This render's own copy of who is taken out: read once here, never looked up again while it runs.
        var removal = removing ? self.removal : nil
        if removing, removal == nil, let keep = plan.keep {
            let plate = try await makePlate(clip, keep: keep) { progress($0 * 0.15) }
            let made = try await survey(clip, keep: keep, plate: plate) { progress(0.15 + $0 * 0.35) }
            self.removal = made
            removal = made
        }
        let base = removing ? 0.5 : 0.0
        // With nobody to take out there is nothing to replace: the frames are the clip's own.
        let job = EditJob(upright: clip.upright, renderSize: clip.renderSize, keep: removing ? plan.keep : nil,
                          removal: (removal?.removed ?? 0) > 0 ? removal : nil, context: context)

        let movie = AVMutableComposition()
        guard let track = movie.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid) else { throw CocoaError(.fileWriteUnknown) }
        var cursor = CMTime.zero
        var instructions: [EditInstruction] = []
        for piece in timeline.pieces {
            let range = CMTimeRange(start: CMTime(seconds: piece.t0, preferredTimescale: 600), end: CMTime(seconds: piece.t1, preferredTimescale: 600))
            let length = CMTime(seconds: range.duration.seconds / piece.speed, preferredTimescale: 600)
            guard range.duration > .zero, length > .zero else { continue }
            try track.insertTimeRange(range, of: clip.track, at: cursor)
            track.scaleTimeRange(CMTimeRange(start: cursor, duration: range.duration), toDuration: length)
            // The piece as it really sits in the movie, after rounding to the movie's clock.
            var placed = piece
            placed.t0 = range.start.seconds; placed.t1 = range.end.seconds; placed.start = cursor.seconds
            instructions.append(EditInstruction(timeRange: CMTimeRange(start: cursor, duration: length), track: track.trackID, job: job, piece: placed))
            cursor = cursor + length
        }
        guard !instructions.isEmpty else { throw CocoaError(.fileReadCorruptFile) }

        let folder = directory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let url = folder.appendingPathComponent("BetaBuddy.mp4")
        guard let export = AVAssetExportSession(asset: movie, presetName: AVAssetExportPresetHighestQuality) else { throw CocoaError(.fileWriteUnknown) }
        export.videoComposition = composition(clip, instructions: instructions)
        export.outputURL = url
        export.outputFileType = .mp4
        let watch = Task {
            while !Task.isCancelled {
                progress(base + Double(export.progress) * (1 - base))
                try? await Task.sleep(for: .milliseconds(150))
            }
        }
        defer { watch.cancel() }
        await withTaskCancellationHandler { await export.export() } onCancel: { export.cancelExport() }
        guard export.status == .completed, !Task.isCancelled else {
            try? FileManager.default.removeItem(at: folder)
            try Task.checkCancellation()
            throw export.error ?? CocoaError(.fileWriteUnknown)
        }
        progress(1)
        return url
    }
}
