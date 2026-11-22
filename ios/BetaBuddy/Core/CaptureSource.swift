import AVFoundation
import Accelerate
import CoreImage
import CoreMotion
import MediaPipeTasksVision
import UIKit
import os

enum CaptureSourceError: Error { case denied, unavailable }

/// Where the capture screen's pictures come from. On an iPhone that is the back camera; tests
/// and the simulator, which has no camera, play a file instead. Everything after this point
/// (the checks, the recording, saving) is the same code for both.
protocol CaptureSource: AnyObject {
    /// Upright frames with increasing presentation times, on the source's own queue.
    var onFrame: ((CVPixelBuffer, CMTime) -> Void)? { get set }
    /// The source stopped delivering by itself: a call, another app took the camera, an error.
    var onInterruption: (() -> Void)? { get set }
    /// Horizontal field of view of the upright frames in degrees, when known.
    var horizontalFieldOfView: Double? { get }
    /// Shows what the source delivers, uncropped.
    var previewLayer: CALayer { get }
    /// Asks for permission when needed. Throws `CaptureSourceError`.
    func start() async throws
    func stop()
    /// Encoder settings for a clip of these frames.
    func writerSettings(width: Int, height: Int) -> [String: Any]
}

extension CaptureSource {
    func writerSettings(width: Int, height: Int) -> [String: Any] { ClipRecorder.defaultSettings(width: width, height: height) }
}

/// The back wide camera at 1080p, video only: no audio input is ever added, so the microphone
/// is not asked for. Frames are rotated to portrait by the capture pipeline.
final class CameraSource: NSObject, CaptureSource, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    var onFrame: ((CVPixelBuffer, CMTime) -> Void)?
    var onInterruption: (() -> Void)?
    private(set) var horizontalFieldOfView: Double?
    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "com.betabuddy.capture.video", qos: .userInitiated)
    private var configured = false
    private var observers: [NSObjectProtocol] = []
    private lazy var preview: AVCaptureVideoPreviewLayer = {
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspect
        return layer
    }()
    var previewLayer: CALayer { preview }

    func start() async throws {
        // Asked before permission: where there is no camera at all, there is nothing to ask for.
        guard AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) != nil else { throw CaptureSourceError.unavailable }
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: break
        case .notDetermined: guard await AVCaptureDevice.requestAccess(for: .video) else { throw CaptureSourceError.denied }
        default: throw CaptureSourceError.denied
        }
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            queue.async {
                do {
                    try self.configure()
                    if !self.session.isRunning { self.session.startRunning() }
                    continuation.resume()
                } catch { continuation.resume(throwing: error) }
            }
        }
        if let connection = preview.connection, connection.isVideoRotationAngleSupported(90) { connection.videoRotationAngle = 90 }
        let center = NotificationCenter.default
        guard observers.isEmpty else { return }
        for name in [AVCaptureSession.wasInterruptedNotification, AVCaptureSession.runtimeErrorNotification] {
            observers.append(center.addObserver(forName: name, object: session, queue: .main) { [weak self] _ in self?.onInterruption?() })
        }
    }

    private func configure() throws {
        guard !configured else { return }
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else { throw CaptureSourceError.unavailable }
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = session.canSetSessionPreset(.hd1920x1080) ? .hd1920x1080 : .high
        guard session.canAddInput(input), session.canAddOutput(output) else { throw CaptureSourceError.unavailable }
        session.addInput(input)
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
        // A late frame is dropped rather than held: the clip keeps real time, the checks only need a few frames a second.
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        session.addOutput(output)
        if let connection = output.connection(with: .video) {
            if connection.isVideoRotationAngleSupported(90) { connection.videoRotationAngle = 90 }
            // The phone stands still; stabilisation would only crop the picture and change its field of view.
            if connection.isVideoStabilizationSupported { connection.preferredVideoStabilizationMode = .off }
        }
        let size = CMVideoFormatDescriptionGetDimensions(device.activeFormat.formatDescription)
        horizontalFieldOfView = CaptureMotion.uprightFieldOfView(sensorDegrees: Double(device.activeFormat.videoFieldOfView),
                                                                 longSide: Double(max(size.width, size.height)), shortSide: Double(min(size.width, size.height)))
        configured = true
    }

    func stop() {
        observers.forEach(NotificationCenter.default.removeObserver)
        observers = []
        queue.async { if self.session.isRunning { self.session.stopRunning() } }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        onFrame?(buffer, CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
    }

    func writerSettings(width: Int, height: Int) -> [String: Any] {
        let codec: AVVideoCodecType = output.availableVideoCodecTypesForAssetWriter(writingTo: .mov).contains(.hevc) ? .hevc : .h264
        guard var settings = output.recommendedVideoSettings(forVideoCodecType: codec, assetWriterOutputFileType: .mov) else {
            return ClipRecorder.defaultSettings(width: width, height: height)
        }
        // The frames that arrive are what gets written, whatever the recommendation assumed.
        settings[AVVideoWidthKey] = width
        settings[AVVideoHeightKey] = height
        return settings
    }
}

/// Carries a value that is only ever used on one queue across to that queue.
private struct Unchecked<Value>: @unchecked Sendable { let value: Value }

#if DEBUG
/// Plays a video file in a loop, at its own pace, in place of the camera. Only reachable with
/// `-debugCaptureVideo <path>` on the launch command line of a Debug build, and from tests.
final class FileCaptureSource: CaptureSource, @unchecked Sendable {
    var onFrame: ((CVPixelBuffer, CMTime) -> Void)?
    var onInterruption: (() -> Void)?
    var horizontalFieldOfView: Double?
    private let url: URL
    private let display = AVSampleBufferDisplayLayer()
    private let queue = DispatchQueue(label: "com.betabuddy.capture.file", qos: .userInitiated)
    private let lock = NSLock()
    private var running = false
    var previewLayer: CALayer { display }

    init(url: URL, horizontalFieldOfView: Double? = nil) {
        self.url = url
        self.horizontalFieldOfView = horizontalFieldOfView
        display.videoGravity = .resizeAspect
    }

    private var isRunning: Bool {
        get { lock.lock(); defer { lock.unlock() }; return running }
        set { lock.lock(); running = newValue; lock.unlock() }
    }

    func start() async throws {
        let asset = AVURLAsset(url: url)
        guard FileManager.default.fileExists(atPath: url.path),
              let track = try? await asset.loadTracks(withMediaType: .video).first else { throw CaptureSourceError.unavailable }
        guard !isRunning else { return }
        isRunning = true
        let media = Unchecked(value: track)
        queue.async { self.play(asset, media.value) }
    }

    func stop() { isRunning = false }

    /// Lets a test end the pictures the way a phone call would.
    func interrupt() {
        isRunning = false
        DispatchQueue.main.async { self.onInterruption?() }
    }

    private func play(_ asset: AVURLAsset, _ track: AVAssetTrack) {
        let began = Date()
        var offset = CMTime.zero
        while isRunning {
            guard let reader = try? AVAssetReader(asset: asset) else { break }
            let output = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
            guard reader.canAdd(output) else { break }
            reader.add(output)
            guard reader.startReading() else { break }
            var last = CMTime.zero
            while isRunning, let sample = output.copyNextSampleBuffer() {
                guard let buffer = CMSampleBufferGetImageBuffer(sample) else { continue }
                last = CMSampleBufferGetPresentationTimeStamp(sample)
                let time = offset + last
                let wait = time.seconds - Date().timeIntervalSince(began)
                if wait > 0 { Thread.sleep(forTimeInterval: wait) }
                if let attachments = CMSampleBufferGetSampleAttachmentsArray(sample, createIfNecessary: true) as? [NSMutableDictionary] {
                    attachments.first?[kCMSampleAttachmentKey_DisplayImmediately] = true
                }
                if display.status == .failed { display.flush() }
                display.enqueue(sample)
                onFrame?(buffer, time)
            }
            reader.cancelReading()
            guard last > .zero else { break }   // Nothing could be read; do not spin.
            offset = offset + last + CMTime(value: 1, timescale: 30)
        }
    }
}
#endif

/// Writes the frames it is given into a movie, in real time. Appending and finishing may come
/// from different queues.
final class ClipRecorder: @unchecked Sendable {
    struct Result {
        /// `nil` when nothing usable was written.
        var url: URL?
        var duration: Double
        var frames: Int
        var dropped: Int
    }
    let url: URL
    private let writer: AVAssetWriter
    private let input: AVAssetWriterInput
    private let adaptor: AVAssetWriterInputPixelBufferAdaptor
    private let lock = NSLock()
    private var first: CMTime?
    private var last = CMTime.zero
    private var frames = 0
    private var dropped = 0
    private var closed = false

    static func defaultSettings(width: Int, height: Int) -> [String: Any] {
        [AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height,
         AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: max(1_000_000, width * height * 4), AVVideoMaxKeyFrameIntervalKey: 60]]
    }

    init(url: URL, width: Int, height: Int, settings: [String: Any], date: Date = Date()) throws {
        self.url = url
        writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        guard writer.canApply(outputSettings: settings, forMediaType: .video) else { throw CaptureSourceError.unavailable }
        input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = true
        adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: nil)
        guard writer.canAdd(input) else { throw CaptureSourceError.unavailable }
        writer.add(input)
        // The same key an imported video's recording time is read from.
        let created = AVMutableMetadataItem()
        created.identifier = .quickTimeMetadataCreationDate
        created.value = ISO8601DateFormatter().string(from: date) as NSString
        writer.metadata = [created]
        guard writer.startWriting() else { throw writer.error ?? CaptureSourceError.unavailable }
    }

    /// Seconds recorded so far, or `nil` once the clip is closed or the writer has failed.
    @discardableResult
    func append(_ buffer: CVPixelBuffer, at time: CMTime) -> Double? {
        lock.lock(); defer { lock.unlock() }
        guard !closed, writer.status == .writing, time.isNumeric else { return nil }
        guard let first else {
            writer.startSession(atSourceTime: time)
            first = time; last = time
            if adaptor.append(buffer, withPresentationTime: time) { frames = 1 } else { dropped += 1 }
            return 0
        }
        guard time > last else { return (last - first).seconds }
        // The encoder is behind: this frame is left out instead of making the camera wait.
        if input.isReadyForMoreMediaData, adaptor.append(buffer, withPresentationTime: time) { frames += 1; last = time } else { dropped += 1 }
        return (last - first).seconds
    }

    var failed: Bool { writer.status == .failed }

    private func close() -> (alreadyClosed: Bool, duration: Double, counts: (Int, Int)) {
        lock.lock(); defer { lock.unlock() }
        let alreadyClosed = closed
        closed = true
        return (alreadyClosed, first.map { (last - $0).seconds } ?? 0, (frames, dropped))
    }

    /// Closes the file with what has been written so far.
    func finish() async -> Result {
        let (alreadyClosed, duration, counts) = close()
        guard !alreadyClosed, counts.0 > 0, writer.status == .writing else {
            if writer.status == .writing { writer.cancelWriting() }
            if !alreadyClosed { try? FileManager.default.removeItem(at: url) }
            return Result(url: nil, duration: 0, frames: counts.0, dropped: counts.1)
        }
        input.markAsFinished()
        await writer.finishWriting()
        guard writer.status == .completed else {
            try? FileManager.default.removeItem(at: url)
            return Result(url: nil, duration: 0, frames: counts.0, dropped: counts.1)
        }
        return Result(url: url, duration: duration, frames: counts.0, dropped: counts.1)
    }
}

/// The phone's attitude and how much it is moving, from Core Motion. Not available in the simulator.
final class CaptureMotionMonitor: @unchecked Sendable {
    private let manager = CMMotionManager()
    private let queue = OperationQueue()
    private let lock = NSLock()
    private var gravity: (x: Double, y: Double, z: Double)?
    private var squares = 0.0
    private var samples = 0

    func start() {
        guard manager.isDeviceMotionAvailable, !manager.isDeviceMotionActive else { return }
        queue.maxConcurrentOperationCount = 1
        manager.deviceMotionUpdateInterval = 1.0 / 30
        manager.startDeviceMotionUpdates(to: queue) { [weak self] motion, _ in
            guard let self, let motion else { return }
            let rate = motion.rotationRate
            self.lock.lock()
            self.gravity = (motion.gravity.x, motion.gravity.y, motion.gravity.z)
            self.squares += rate.x * rate.x + rate.y * rate.y + rate.z * rate.z
            self.samples += 1
            self.lock.unlock()
        }
    }
    func stop() { if manager.isDeviceMotionActive { manager.stopDeviceMotionUpdates() } }

    /// Attitude now and shake since the last call. All `nil` without a motion sensor.
    func take() -> CaptureFrameInfo {
        lock.lock(); defer { lock.unlock() }
        guard let gravity else { return CaptureFrameInfo() }
        let shake = samples > 0 ? (squares / Double(samples)).squareRoot() : nil
        squares = 0; samples = 0
        return CaptureFrameInfo(pitch: CaptureMotion.pitch(gravity: gravity), roll: CaptureMotion.roll(gravity: gravity), shake: shake)
    }
}

/// The bundled pose model on preview frames. It is optional help for the capture page, so it is
/// dropped for the rest of the session as soon as it cannot keep up. Used from one queue only.
final class LivePose {
    /// Average time per frame above which live pose is given up.
    static let budget = 0.15
    private var landmarker: PoseLandmarker?
    private var lastTimestamp = -1
    private var total = 0.0
    private var samples = 0
    private(set) var enabled = true

    /// 33 × `[x, y, visibility]`, or `nil` when nobody was found. The time spent is returned either way.
    func detect(_ image: CGImage, time: Double) -> (pose: [[Double]]?, seconds: Double) {
        guard enabled else { return (nil, 0) }
        if landmarker == nil {
            guard let model = Bundle.main.path(forResource: "pose_landmarker_full", ofType: "task") else { enabled = false; return (nil, 0) }
            let options = PoseLandmarkerOptions()
            options.baseOptions.modelAssetPath = model
            options.baseOptions.delegate = .CPU
            options.runningMode = .video
            options.numPoses = 1
            landmarker = try? PoseLandmarker(options: options)
            guard landmarker != nil else { enabled = false; return (nil, 0) }
        }
        let began = Date()
        let timestamp = max(lastTimestamp + 1, Int(time * 1000))
        lastTimestamp = timestamp
        let result = try? landmarker?.detect(videoFrame: MPImage(uiImage: UIImage(cgImage: image)), timestampInMilliseconds: timestamp)
        let seconds = Date().timeIntervalSince(began)
        // The first run also loads the model; it does not say how fast the following ones are.
        if samples > 0 { total += seconds }
        samples += 1
        if samples >= 6, total / Double(samples - 1) > Self.budget || ProcessInfo.processInfo.thermalState.rawValue >= ProcessInfo.ThermalState.serious.rawValue {
            enabled = false
            landmarker = nil
        }
        guard let points = result?.landmarks.first, points.count == 33 else { return (nil, seconds) }
        return (points.map { [Double($0.x), Double($0.y), $0.visibility?.doubleValue ?? 0] }, seconds)
    }
    var averageSeconds: Double { samples > 1 ? total / Double(samples - 1) : 0 }
}

/// What handing preview frames to the capture page costs, for the local log and for tests.
struct CaptureMetrics: Equatable, Sendable {
    var frames = 0
    var prepareSeconds = 0.0     // Scaling to the hand-over size and JPEG encoding.
    var poseSeconds = 0.0
    var poseFrames = 0
    var posesFound = 0
    var handOverSeconds = 0.0    // From the call into the page until it returns.
    var jpegBytes = 0
    var poseDropped = false
    func average(_ total: Double, over count: Int? = nil) -> Double { (count ?? frames) > 0 ? total / Double(count ?? frames) : 0 }
    var summary: String {
        String(format: "frames %d, prepare %.1f ms, pose %.1f ms over %d (found %d%@), hand-over %.1f ms, jpeg %d B",
               frames, average(prepareSeconds) * 1000, average(poseSeconds, over: poseFrames) * 1000, poseFrames, posesFound,
               poseDropped ? ", dropped" : "", average(handOverSeconds) * 1000, frames > 0 ? jpegBytes / frames : 0)
    }
}

/// Sits between the source and everything that uses its frames: the recorder gets every frame,
/// the capture page gets a few a second, scaled down and with what the sensors know.
final class CapturePipeline: @unchecked Sendable {
    struct Frame {
        var time: Double
        var jpegBase64: String
        var width: Int
        var height: Int
        var info: CaptureFrameInfo
        var prepareSeconds: Double
        var poseSeconds: Double?
        var bytes: Int
    }
    static let shortSide: CGFloat = 360
    let motion = CaptureMotionMonitor()
    /// Called on the main queue. The second argument must be called when the frame has been dealt with.
    var deliver: ((Frame, @escaping () -> Void) -> Void)?
    /// Seconds recorded so far, on the source's queue.
    var onRecorded: ((Double) -> Void)?
    /// Pixel size of the source's frames, once, on the source's queue.
    var onFirstFrame: ((Int, Int) -> Void)?
    private let work = DispatchQueue(label: "com.betabuddy.capture.checks", qos: .userInitiated)
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let pose = LivePose()
    private let lock = NSLock()
    private var recorder: ClipRecorder?
    private var busy = false
    private var feeding = false
    private var wantsPose = true
    private var interval = 1.0 / 3
    private var lastFed = -Double.infinity
    private var origin: Double?
    private var size: (width: Int, height: Int)?

    /// Pixel size of the frames the source delivers, once one has arrived.
    var frameSize: (width: Int, height: Int)? { lock.lock(); defer { lock.unlock() }; return size }
    var poseDropped: Bool { !pose.enabled }

    func setRecorder(_ recorder: ClipRecorder?) { lock.lock(); self.recorder = recorder; lock.unlock() }
    /// `interval` is the least time between two frames handed to the page.
    func setFeeding(_ on: Bool, interval: Double? = nil, pose: Bool? = nil) {
        lock.lock()
        feeding = on
        if let interval { self.interval = interval }
        if let pose { wantsPose = pose }
        lock.unlock()
    }

    func handle(_ buffer: CVPixelBuffer, _ time: CMTime) {
        let seconds = time.seconds
        lock.lock()
        size = (CVPixelBufferGetWidth(buffer), CVPixelBufferGetHeight(buffer))
        let isFirst = origin == nil
        if origin == nil { origin = seconds }
        let recorder = self.recorder
        let feed = feeding && !busy && seconds - lastFed >= interval - 0.005
        if feed { busy = true; lastFed = seconds }
        let wantsPose = self.wantsPose
        let elapsed = seconds - (origin ?? seconds)
        lock.unlock()
        if isFirst { onFirstFrame?(CVPixelBufferGetWidth(buffer), CVPixelBufferGetHeight(buffer)) }
        if let recorder, let recorded = recorder.append(buffer, at: time) { onRecorded?(recorded) }
        guard feed else { return }
        work.async { [self] in
            let began = Date()
            guard let image = Self.scaled(buffer, shortSide: Self.shortSide, context: context),
                  let jpeg = UIImage(cgImage: image).jpegData(compressionQuality: 0.7) else { release(); return }
            let base64 = jpeg.base64EncodedString()
            let prepared = Date().timeIntervalSince(began)
            var info = motion.take()
            var poseSeconds: Double?
            if wantsPose, pose.enabled {
                let found = pose.detect(image, time: elapsed)
                info.pose = found.pose
                poseSeconds = found.seconds
            }
            let frame = Frame(time: elapsed, jpegBase64: base64, width: image.width, height: image.height, info: info,
                              prepareSeconds: prepared, poseSeconds: poseSeconds, bytes: jpeg.count)
            DispatchQueue.main.async { [self] in
                guard let deliver else { release(); return }
                deliver(frame) { [self] in release() }
            }
        }
    }

    private func release() { lock.lock(); busy = false; lock.unlock() }

    /// The frame as the user sees it, with its shorter side scaled down to `shortSide`. The two
    /// formats the camera and the file source deliver are scaled on the processor, which costs about
    /// the same everywhere; anything else goes through Core Image.
    static func scaled(_ buffer: CVPixelBuffer, shortSide: CGFloat, context: CIContext) -> CGImage? {
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        guard width > 1, height > 1 else { return nil }
        let scale = min(1, shortSide / CGFloat(min(width, height)))
        // Even sizes: the colour planes of a camera frame are half as large.
        let target = (width: max(2, Int((CGFloat(width) * scale).rounded()) & ~1), height: max(2, Int((CGFloat(height) * scale).rounded()) & ~1))
        switch CVPixelBufferGetPixelFormatType(buffer) {
        case kCVPixelFormatType_32BGRA:
            if let image = scaledBGRA(buffer, target) { return image }
        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
            if let image = scaledBiPlanar(buffer, target, fullRange: true) { return image }
        case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
            if let image = scaledBiPlanar(buffer, target, fullRange: false) { return image }
        default: break
        }
        var image = CIImage(cvPixelBuffer: buffer)
        if scale < 1 {
            image = image.applyingFilter("CILanczosScaleTransform", parameters: [kCIInputScaleKey: scale, kCIInputAspectRatioKey: 1.0])
        }
        let extent = CGRect(x: image.extent.minX, y: image.extent.minY, width: image.extent.width.rounded(.down), height: image.extent.height.rounded(.down))
        return context.createCGImage(image, from: extent)
    }

    private static func image(fromBGRA pixels: inout vImage_Buffer) -> CGImage? {
        guard var format = vImage_CGImageFormat(bitsPerComponent: 8, bitsPerPixel: 32, colorSpace: CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB(),
                                                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)) else { return nil }
        var error = vImage_Error(kvImageNoError)
        let image = vImageCreateCGImageFromBuffer(&pixels, &format, nil, nil, vImage_Flags(kvImageNoFlags), &error)?.takeRetainedValue()
        return error == kvImageNoError ? image : nil
    }

    private static func scaledBGRA(_ buffer: CVPixelBuffer, _ target: (width: Int, height: Int)) -> CGImage? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        var source = vImage_Buffer(data: base, height: vImagePixelCount(CVPixelBufferGetHeight(buffer)), width: vImagePixelCount(CVPixelBufferGetWidth(buffer)),
                                   rowBytes: CVPixelBufferGetBytesPerRow(buffer))
        var scaled = vImage_Buffer()
        guard vImageBuffer_Init(&scaled, vImagePixelCount(target.height), vImagePixelCount(target.width), 32, vImage_Flags(kvImageNoFlags)) == kvImageNoError else { return nil }
        defer { free(scaled.data) }
        guard vImageScale_ARGB8888(&source, &scaled, nil, vImage_Flags(kvImageHighQualityResampling)) == kvImageNoError else { return nil }
        return image(fromBGRA: &scaled)
    }

    /// Scales luma and chroma separately while they are still small, then converts once at the target size.
    private static func scaledBiPlanar(_ buffer: CVPixelBuffer, _ target: (width: Int, height: Int), fullRange: Bool) -> CGImage? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard CVPixelBufferGetPlaneCount(buffer) == 2, let lumaBase = CVPixelBufferGetBaseAddressOfPlane(buffer, 0),
              let chromaBase = CVPixelBufferGetBaseAddressOfPlane(buffer, 1) else { return nil }
        var luma = vImage_Buffer(data: lumaBase, height: vImagePixelCount(CVPixelBufferGetHeightOfPlane(buffer, 0)), width: vImagePixelCount(CVPixelBufferGetWidthOfPlane(buffer, 0)),
                                 rowBytes: CVPixelBufferGetBytesPerRowOfPlane(buffer, 0))
        var chroma = vImage_Buffer(data: chromaBase, height: vImagePixelCount(CVPixelBufferGetHeightOfPlane(buffer, 1)), width: vImagePixelCount(CVPixelBufferGetWidthOfPlane(buffer, 1)),
                                   rowBytes: CVPixelBufferGetBytesPerRowOfPlane(buffer, 1))
        var smallLuma = vImage_Buffer(), smallChroma = vImage_Buffer(), pixels = vImage_Buffer()
        let flags = vImage_Flags(kvImageNoFlags)
        guard vImageBuffer_Init(&smallLuma, vImagePixelCount(target.height), vImagePixelCount(target.width), 8, flags) == kvImageNoError else { return nil }
        defer { free(smallLuma.data) }
        guard vImageBuffer_Init(&smallChroma, vImagePixelCount(target.height / 2), vImagePixelCount(target.width / 2), 16, flags) == kvImageNoError else { return nil }
        defer { free(smallChroma.data) }
        guard vImageBuffer_Init(&pixels, vImagePixelCount(target.height), vImagePixelCount(target.width), 32, flags) == kvImageNoError else { return nil }
        defer { free(pixels.data) }
        let quality = vImage_Flags(kvImageHighQualityResampling)
        guard vImageScale_Planar8(&luma, &smallLuma, nil, quality) == kvImageNoError,
              vImageScale_CbCr8(&chroma, &smallChroma, nil, quality) == kvImageNoError else { return nil }
        var range = fullRange
            ? vImage_YpCbCrPixelRange(Yp_bias: 0, CbCr_bias: 128, YpRangeMax: 255, CbCrRangeMax: 255, YpMax: 255, YpMin: 1, CbCrMax: 255, CbCrMin: 0)
            : vImage_YpCbCrPixelRange(Yp_bias: 16, CbCr_bias: 128, YpRangeMax: 235, CbCrRangeMax: 240, YpMax: 235, YpMin: 16, CbCrMax: 240, CbCrMin: 16)
        var conversion = vImage_YpCbCrToARGB()
        guard vImageConvert_YpCbCrToARGB_GenerateConversion(kvImage_YpCbCrToARGBMatrix_ITU_R_709_2, &range, &conversion, kvImage420Yp8_CbCr8, kvImageARGB8888, flags) == kvImageNoError,
              // The permute map writes the result as B, G, R, A.
              vImageConvert_420Yp8_CbCr8ToARGB8888(&smallLuma, &smallChroma, &pixels, &conversion, [3, 2, 1, 0], 255, flags) == kvImageNoError else { return nil }
        return image(fromBGRA: &pixels)
    }

    /// Free space where the clip is being written.
    static func freeBytes() -> Int64? {
        let values = try? FileManager.default.temporaryDirectory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage
    }
}
