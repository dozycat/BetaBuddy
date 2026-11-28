import AVFoundation
import CoreImage
import UIKit
import MediaPipeTasksVision

enum AnalysisError: LocalizedError {
    case missingModel, invalidVideo, tooLong, tooLarge, noPose, overheated
    var errorDescription: String? {
        switch self {
        case .missingModel: return String(localized: "安装包缺少姿态模型，请重新安装完整版本。", bundle: AppLanguage.bundle)
        case .invalidVideo: return String(localized: "无法读取这个视频，请选择未损坏的 MOV 或 MP4 文件。", bundle: AppLanguage.bundle)
        case .tooLong: return String(localized: "视频超过 20 分钟。请先在相册中裁剪。", bundle: AppLanguage.bundle)
        case .tooLarge: return String(localized: "视频超过 500 MB，请先在相册中裁剪或导出较小版本。", bundle: AppLanguage.bundle)
        case .noPose: return String(localized: "未识别到足够清晰的人体。请使用全身入镜、光线充足、单人攀爬的视频。", bundle: AppLanguage.bundle)
        case .overheated: return String(localized: "手机温度较高，已停止分析。稍后可重新开始。", bundle: AppLanguage.bundle)
        }
    }
}

enum AnalysisQuality: String, CaseIterable, Identifiable {
    case balanced, detailed
    var id: String { rawValue }
    var title: String { self == .balanced ? String(localized: "日常回看", bundle: AppLanguage.bundle) : String(localized: "动作细看", bundle: AppLanguage.bundle) }
    var subtitle: String { self == .balanced ? String(localized: "最高 10 帧/秒 · 较少计算", bundle: AppLanguage.bundle) : String(localized: "最高 15 帧/秒 · 保留更多细节", bundle: AppLanguage.bundle) }
    var fps: Double { self == .balanced ? 10 : 15 }
    var maxDimension: CGFloat { self == .balanced ? 720 : 960 }
}

struct ClipStills: Sendable {
    struct Frame: Sendable {
        var time: Double
        var jpegBase64: String
    }
    /// Pixel size of the video as displayed, after its orientation is applied.
    var width: Int
    var height: Int
    var frames: [Frame]
}

// A single worker owns decoder, Core Image context, and model. Never infer on MainActor.
actor VideoAnalyzer {
    private static let maximumDurationSeconds: Double = 20 * 60
    func analyze(url: URL, quality: AnalysisQuality, subjectAnchors: [SubjectAnchor] = [],
                 progress: @escaping @Sendable (Double) async -> Void) async throws -> PoseAnalysis {
        try Task.checkCancellation()
        guard let model = Bundle.main.path(forResource: "pose_landmarker_full", ofType: "task") else { throw AnalysisError.missingModel }
        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration).seconds
        guard duration.isFinite, duration >= 0.5,
              let track = try await asset.loadTracks(withMediaType: .video).first else { throw AnalysisError.invalidVideo }
        guard duration <= Self.maximumDurationSeconds else { throw AnalysisError.tooLong }
        let transform = try await track.load(.preferredTransform)
        let natural = try await track.load(.naturalSize)
        let displayRect = CGRect(origin: .zero, size: natural).applying(transform)
        let aspect = abs(displayRect.width / displayRect.height)
        guard aspect.isFinite, aspect > 0 else { throw AnalysisError.invalidVideo }
        let options = PoseLandmarkerOptions()
        options.baseOptions.modelAssetPath = model
        // Fixed CPU delegate: safe on simulator and deterministic baseline for thermal profiling.
        options.baseOptions.delegate = .CPU
        options.runningMode = .video
        let chooseSubject = SubjectResolver.available
        let regions = chooseSubject ? subjectAnchors.filter { $0.region == true } : []
        let passes = Double(1 + regions.count)
        let recover = chooseSubject && subjectAnchors.isEmpty
        let initialShare = recover ? 0.7 : 1.0
        options.numPoses = chooseSubject ? 4 : 1
        // MediaPipe defaults: raising these can miss the climber
        // and initialize video tracking on a high-contrast background false positive.
        options.minPoseDetectionConfidence = 0.5
        options.minPosePresenceConfidence = 0.5
        options.minTrackingConfidence = 0.5
        var landmarker: PoseLandmarker? = try PoseLandmarker(options: options)
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw AnalysisError.invalidVideo }
        reader.add(output)
        guard reader.startReading() else { throw reader.error ?? AnalysisError.invalidVideo }
        defer { reader.cancelReading() }
        let context = CIContext(options: [.cacheIntermediates: false])
        var frames: [PoseFrame] = []
        var people: [SubjectSample] = []
        var nextTime = 0.0, sampled = 0, lastTimestamp = -1
        while reader.status == .reading {
            try Task.checkCancellation()
            if ProcessInfo.processInfo.thermalState == .critical { throw AnalysisError.overheated }
            let step: (finished: Bool, time: Double?) = try autoreleasepool {
                guard let sample = output.copyNextSampleBuffer() else { return (true, nil) }
                let time = CMSampleBufferGetPresentationTimeStamp(sample).seconds
                guard time.isFinite, time + 0.000001 >= nextTime, let buffer = CMSampleBufferGetImageBuffer(sample) else { return (false, nil) }
                // Skip before color conversion and inference. Never queue decoded images.
                nextTime = (floor(time * quality.fps + 0.000001) + 1) / quality.fps
                let cgImage = try Self.inferenceImage(buffer: buffer, transform: transform,
                                                      maxDimension: quality.maxDimension, context: context)
                let timestamp = max(lastTimestamp + 1, Int(time * 1000))
                lastTimestamp = timestamp
                let result = try landmarker!.detect(videoFrame: MPImage(uiImage: UIImage(cgImage: cgImage)), timestampInMilliseconds: timestamp)
                sampled += 1
                var candidates: [SubjectCandidate] = []
                for (person, points) in result.landmarks.enumerated() where points.count == 33 {
                    let joints = points.map { Joint(x: rounded4(Double($0.x)), y: rounded4(Double($0.y)), confidence: rounded4(min($0.visibility?.doubleValue ?? 0, $0.presence?.doubleValue ?? 0))) }
                    var frame = PoseFrame(time: time, joints: joints)
                    if result.worldLandmarks.indices.contains(person), result.worldLandmarks[person].count == 33 {
                        frame.world = result.worldLandmarks[person].map { point in
                            [Double(point.x), Double(point.y), Double(point.z), point.visibility?.doubleValue ?? 0].map { $0.isFinite ? rounded4($0) : 0 }
                        }
                    }
                    guard frame.hasReliableTorso else { continue }
                    if !chooseSubject { frames.append(frame); continue }
                    let valid = joints.filter(\.valid)
                    guard let left = valid.map(\.x).min(), let right = valid.map(\.x).max(),
                          let top = valid.map(\.y).min(), let bottom = valid.map(\.y).max(), right > left, bottom > top else { continue }
                    let bounds = SubjectBounds(x: left, y: top, width: right-left, height: bottom-top)
                    // Samples come from the torso rather than face; used only for local continuity.
                    let torso = [joints[11], joints[12], joints[23], joints[24]]
                    let tx = torso.map(\.x).min()!, ty = torso.map(\.y).min()!
                    let tw = torso.map(\.x).max()! - tx, th = torso.map(\.y).max()! - ty
                    let colour = Self.appearance(image: cgImage, bounds: SubjectBounds(x: tx, y: ty, width: tw, height: th))
                    candidates.append(SubjectCandidate(id: "\(timestamp)-\(person)", bounds: bounds, appearance: colour, pose: frame))
                }
                if chooseSubject { people.append(SubjectSample(time: time, candidates: candidates)) }
                return (false, time)
            }
            if step.finished { break }
            if let time = step.time, sampled % 5 == 0 { await progress(min(0.99, time / duration) / passes * initialShare) }
        }
        try Task.checkCancellation()
        if reader.status == .failed { throw reader.error ?? AnalysisError.invalidVideo }
        // Recovery uses another detector/decoder; release the full-frame pass first.
        landmarker = nil
        reader.cancelReading()
        var analysis = PoseAnalysis(frames: frames, sampledFrames: sampled, duration: duration, aspectRatio: aspect)
        if chooseSubject {
            analysis.people = people
            for (index, anchor) in regions.enumerated() {
                analysis = try await SubjectRegionAnalyzer().refine(url: url, analysis: analysis, bounds: anchor.bounds, at: anchor.time) { value in
                    await progress((Double(index + 1) + value) / passes)
                }.analysis
            }
            analysis = try await SubjectResolver().resolve(analysis, anchors: subjectAnchors)
            if recover && analysis.needsSubjectRecovery {
                analysis=try await recoverSubject(url:url,analysis:analysis) { value in await progress(0.7+0.3*value) }
            }
            guard analysis.hasSubjectCandidates else { throw AnalysisError.noPose }
        } else if frames.count < 3 { throw AnalysisError.noPose }
        await progress(1)
        return analysis
    }

    /// Also used when Automatic is requested for a cached, unresolved candidate set.
    func recoverSubject(url: URL, analysis: PoseAnalysis, progress: @escaping @Sendable (Double) async -> Void) async throws -> PoseAnalysis {
        let worker=SubjectRegionAnalyzer()
        let proposals=try await worker.discover(url:url,analysis:analysis) { value in await progress(value/3) }
        let seeds=try await SubjectResolver().proposedRegions(proposals)
        var result=analysis
        for (index,seed) in seeds.enumerated() {
            result=try await worker.refine(url:url,analysis:result,bounds:seed.bounds,at:seed.time,preserveExistingRegions:false) { value in
                await progress(1.0/3+2.0/3*(Double(index)+value)/Double(seeds.count))
            }.analysis
        }
        result=try await SubjectResolver().resolve(result,anchors:[])
        await progress(1)
        return result
    }

    static func appearance(image: CGImage, bounds: SubjectBounds) -> [Double] {
        guard bounds.valid, let crop = image.cropping(to: CGRect(x: bounds.x * Double(image.width), y: bounds.y * Double(image.height),
            width: bounds.width * Double(image.width), height: bounds.height * Double(image.height))) else { return [] }
        var pixels = [UInt8](repeating: 0, count: 4 * 4 * 4)
        let valid = pixels.withUnsafeMutableBytes { raw -> Bool in
            guard let context = CGContext(data: raw.baseAddress, width: 4, height: 4, bitsPerComponent: 8, bytesPerRow: 16,
                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
            context.interpolationQuality = .medium
            context.draw(crop, in: CGRect(x: 0, y: 0, width: 4, height: 4))
            return true
        }
        guard valid else { return [] }
        // Average each row: stable under small pose shifts, no retained crop or facial descriptor.
        return (0..<4).flatMap { row in (0..<3).map { channel in
            rounded4((0..<4).reduce(0.0) { $0 + Double(pixels[(row * 4 + $1) * 4 + channel]) } / (4 * 255))
        } }
    }

    static func inferenceImage(buffer: CVPixelBuffer, transform: CGAffineTransform,
                               maxDimension: CGFloat, context: CIContext) throws -> CGImage {
        // Track transforms use top-left video coordinates; Core Image uses bottom-left.
        // Conjugate the linear transform by a Y flip before normalizing the extent.
        let ciTransform = CGAffineTransform(a: transform.a, b: -transform.b,
                                           c: -transform.c, d: transform.d, tx: 0, ty: 0)
        var image = CIImage(cvPixelBuffer: buffer).transformed(by: ciTransform)
        image = image.transformed(by: CGAffineTransform(translationX: -image.extent.minX, y: -image.extent.minY))
        let scale = min(1, maxDimension / max(image.extent.width, image.extent.height))
        image = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        guard let result = context.createCGImage(image, from: image.extent) else { throw AnalysisError.invalidVideo }
        return result
    }

    /// Reads playback facts without loading the pose model or decoding the full clip.
    func videoSummary(url: URL) async throws -> PoseAnalysis {
        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration).seconds
        guard duration.isFinite, duration >= 0.5,
              let track = try await asset.loadTracks(withMediaType: .video).first else { throw AnalysisError.invalidVideo }
        guard duration <= Self.maximumDurationSeconds else { throw AnalysisError.tooLong }
        let size = try await track.load(.naturalSize)
        let transform = try await track.load(.preferredTransform)
        let rect = CGRect(origin: .zero, size: size).applying(transform)
        let aspect = abs(rect.width / rect.height)
        guard aspect.isFinite, aspect > 0 else { throw AnalysisError.invalidVideo }
        return PoseAnalysis(frames: [], sampledFrames: 0, duration: duration, aspectRatio: aspect)
    }

    /// Upright stills spread evenly over the clip, for the pack's analysis page. They are made when
    /// the page opens and only ever live in memory.
    func stills(url: URL, count: Int = 60, width: CGFloat = 540, quality: CGFloat = 0.7) async throws -> ClipStills {
        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration).seconds
        guard duration.isFinite, duration > 0,
              let track = try await asset.loadTracks(withMediaType: .video).first else { throw AnalysisError.invalidVideo }
        let display = CGRect(origin: .zero, size: try await track.load(.naturalSize)).applying(try await track.load(.preferredTransform))
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true   // Same orientation as the pose coordinates.
        // `width` bounds the shorter side, so a landscape clip keeps as much detail as a portrait one.
        generator.maximumSize = abs(display.width) > abs(display.height) ? CGSize(width: width * 8, height: width) : CGSize(width: width, height: width * 8)
        // A very short clip does not have that many different frames.
        let count = min(count, max(8, Int(duration * 15)))
        let spacing = duration / Double(count)
        let tolerance = CMTime(seconds: min(0.1, spacing / 3), preferredTimescale: 600)
        generator.requestedTimeToleranceBefore = tolerance
        generator.requestedTimeToleranceAfter = tolerance
        var stills = ClipStills(width: Int(abs(display.width).rounded()), height: Int(abs(display.height).rounded()), frames: [])
        for index in 0..<count {
            try Task.checkCancellation()
            let wanted = CMTime(seconds: (Double(index) + 0.5) * spacing, preferredTimescale: 600)
            // A frame that cannot be decoded is left out; the page decides whether enough remain.
            guard let (image, actual) = try? await generator.image(at: wanted),
                  let data = UIImage(cgImage: image).jpegData(compressionQuality: quality) else { continue }
            stills.frames.append(.init(time: actual.seconds.isFinite ? actual.seconds : wanted.seconds, jpegBase64: data.base64EncodedString()))
        }
        guard !stills.frames.isEmpty else { throw AnalysisError.invalidVideo }
        return stills
    }

    /// Small, frequent upright stills of the whole clip (about three a second, shorter side 270), for
    /// the page to find the climber in a wide fixed shot by what moves. In memory only, like `stills`.
    func trackStills(url: URL, perSecond: Double = 3, limit: Int = 900) async throws -> ClipStills {
        let duration = try await AVURLAsset(url: url).load(.duration).seconds
        guard duration.isFinite, duration > 0 else { throw AnalysisError.invalidVideo }
        return try await stills(url: url, count: min(limit, max(8, Int(duration * perSecond))), width: 270, quality: 0.5)
    }

    func recordingDate(url: URL) async -> Date? {
        let asset = AVURLAsset(url: url)
        guard let item = try? await asset.load(.creationDate) else { return nil }
        if let date = try? await item.load(.dateValue), date.timeIntervalSince1970 > 0 { return date }
        guard let value = try? await item.load(.stringValue) else { return nil }
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let date = formatter.date(from: value) ?? ISO8601DateFormatter().date(from: value)
        return date.flatMap { $0.timeIntervalSince1970 > 0 ? $0 : nil }
    }

    func thumbnail(url: URL) async throws -> Data {
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 600, height: 600)
        let (image, _) = try await generator.image(at: .zero)
        guard let data = UIImage(cgImage: image).jpegData(compressionQuality: 0.8) else { throw AnalysisError.invalidVideo }
        return data
    }
}
