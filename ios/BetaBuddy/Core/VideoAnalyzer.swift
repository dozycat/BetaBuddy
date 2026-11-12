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
        case .tooLong: return String(localized: "视频超过 3 分钟。请先在相册中裁剪成单趟攀爬。", bundle: AppLanguage.bundle)
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

// A single worker owns decoder, Core Image context, and model. Never infer on MainActor.
actor VideoAnalyzer {
    func analyze(url: URL, quality: AnalysisQuality,
                 progress: @escaping @Sendable (Double) async -> Void) async throws -> PoseAnalysis {
        try Task.checkCancellation()
        let start = Date()
        guard let model = Bundle.main.path(forResource: "pose_landmarker_full", ofType: "task") else { throw AnalysisError.missingModel }
        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration).seconds
        guard duration.isFinite, duration >= 0.5,
              let track = try await asset.loadTracks(withMediaType: .video).first else { throw AnalysisError.invalidVideo }
        guard duration <= 180 else { throw AnalysisError.tooLong }
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
        options.numPoses = 1
        // MediaPipe defaults: raising these can miss the climber
        // and initialize video tracking on a high-contrast background false positive.
        options.minPoseDetectionConfidence = 0.5
        options.minPosePresenceConfidence = 0.5
        options.minTrackingConfidence = 0.5
        let landmarker = try PoseLandmarker(options: options)
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw AnalysisError.invalidVideo }
        reader.add(output)
        guard reader.startReading() else { throw reader.error ?? AnalysisError.invalidVideo }
        defer { reader.cancelReading() }
        let context = CIContext(options: [.cacheIntermediates: false])
        var frames: [PoseFrame] = []
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
                let result = try landmarker.detect(videoFrame: MPImage(uiImage: UIImage(cgImage: cgImage)), timestampInMilliseconds: timestamp)
                sampled += 1
                if let points = result.landmarks.first, points.count == 33 {
                    let joints = points.map { Joint(x: Double($0.x), y: Double($0.y), confidence: min($0.visibility?.doubleValue ?? 0, $0.presence?.doubleValue ?? 0)) }
                    let frame = PoseFrame(time: time, joints: joints)
                    if frame.hasReliableTorso { frames.append(frame) }
                }
                return (false, time)
            }
            if step.finished { break }
            if let time = step.time, sampled % 5 == 0 { await progress(min(0.99, time / duration)) }
        }
        try Task.checkCancellation()
        if reader.status == .failed { throw reader.error ?? AnalysisError.invalidVideo }
        guard frames.filter({ $0.hip != nil }).count >= 3 else { throw AnalysisError.noPose }
        await progress(1)
        return PoseAnalysis(frames: frames, sampledFrames: sampled, duration: duration, aspectRatio: aspect,
                            processingSeconds: Date().timeIntervalSince(start), moments: PoseMath.moments(frames: frames, aspect: aspect))
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

    func thumbnail(url: URL) async throws -> Data {
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 600, height: 600)
        let (image, _) = try await generator.image(at: .zero)
        guard let data = UIImage(cgImage: image).jpegData(compressionQuality: 0.8) else { throw AnalysisError.invalidVideo }
        return data
    }
}
