import AVFoundation
import CoreGraphics
import Foundation

/// A drawn climbing wall as a short video, standing in for the camera: a pale wall with panel
/// seams, coloured blobs where `StandInPack.holds` says, a mat along the bottom and a dot that
/// moves so that frames differ. Nothing in it comes from a real gym.
enum SyntheticWallClip {
    static func draw(in context: CGContext, width: Int, height: Int, frame: Int, photo: CGImage? = nil) {
        let w = CGFloat(width), h = CGFloat(height)
        context.setFillColor(CGColor(red: 0.86, green: 0.84, blue: 0.80, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: w, height: h))
        context.setStrokeColor(CGColor(red: 0.70, green: 0.68, blue: 0.64, alpha: 1))
        context.setLineWidth(max(1, w / 270))
        for column in 1..<4 { context.stroke(CGRect(x: w * CGFloat(column) / 4, y: 0, width: 0, height: h)) }
        for row in 1..<6 { context.stroke(CGRect(x: 0, y: h * CGFloat(row) / 6, width: w, height: 0)) }
        // Core Graphics counts y from the bottom; the layout counts from the top.
        context.setFillColor(CGColor(red: 0.20, green: 0.24, blue: 0.32, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: w, height: h * 0.06))
        if let photo {
            let scale = min(w / CGFloat(photo.width), h * 0.8 / CGFloat(photo.height))
            let size = CGSize(width: CGFloat(photo.width) * scale, height: CGFloat(photo.height) * scale)
            context.draw(photo, in: CGRect(x: (w - size.width) / 2, y: h * 0.08, width: size.width, height: size.height))
        }
        for hold in StandInPack.holds {
            let color = StandInPack.colors[hold.color] ?? (128, 128, 128)
            context.setFillColor(CGColor(red: color.red / 255, green: color.green / 255, blue: color.blue / 255, alpha: 1))
            context.fillEllipse(in: CGRect(x: (hold.x - 0.035) * w, y: (1 - hold.y - 0.02) * h, width: 0.07 * w, height: 0.04 * h))
        }
        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        let swing = CGFloat(frame % 60) / 60
        context.fillEllipse(in: CGRect(x: (0.05 + 0.9 * swing) * w, y: h * 0.965, width: w * 0.02, height: w * 0.02))
    }

    /// Writes the clip to a temporary file and returns it. The caller removes it.
    /// `overlay` draws on top of each frame (Core Graphics coordinates, y from the bottom), for things that move.
    /// `transform` is stored with the track, as a phone does for a clip it recorded turned.
    static func make(width: Int = 540, height: Int = 960, seconds: Double = 3, fps: Int32 = 30, photo: CGImage? = nil,
                     transform: CGAffineTransform = .identity, overlay: ((CGContext, Int) -> Void)? = nil) async throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("synthetic-wall-\(UUID()).mov")
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB, kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferCGImageCompatibilityKey as String: true, kCVPixelBufferCGBitmapContextCompatibilityKey as String: true])
        input.transform = transform
        writer.add(input)
        guard writer.startWriting() else { throw writer.error ?? CocoaError(.fileWriteUnknown) }
        writer.startSession(atSourceTime: .zero)
        for frame in 0..<Int(seconds * Double(fps)) {
            while !input.isReadyForMoreMediaData {
                if writer.status == .failed { throw writer.error ?? CocoaError(.fileWriteUnknown) }
                try await Task.sleep(for: .milliseconds(5))
            }
            var pixels: CVPixelBuffer?
            guard let pool = adaptor.pixelBufferPool, CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pixels) == kCVReturnSuccess, let buffer = pixels else { throw CocoaError(.fileWriteUnknown) }
            CVPixelBufferLockBaseAddress(buffer, [])
            if let context = CGContext(data: CVPixelBufferGetBaseAddress(buffer), width: width, height: height, bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                       space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) {
                draw(in: context, width: width, height: height, frame: frame, photo: photo)
                overlay?(context, frame)
            }
            CVPixelBufferUnlockBaseAddress(buffer, [])
            guard adaptor.append(buffer, withPresentationTime: CMTime(value: Int64(frame), timescale: fps)) else { throw writer.error ?? CocoaError(.fileWriteUnknown) }
        }
        input.markAsFinished()
        await writer.finishWriting()
        guard writer.status == .completed else { throw writer.error ?? CocoaError(.fileWriteUnknown) }
        return url
    }
}
