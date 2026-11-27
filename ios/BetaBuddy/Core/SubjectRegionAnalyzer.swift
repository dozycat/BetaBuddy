import AVFoundation
import MediaPipeTasksVision
import UIKit

/// Tracks a user-drawn image region independently of pose visibility. The same region supplies
/// a closer view for pose detection; only actual, reliable poses enter movement analysis.
actor SubjectRegionAnalyzer {
    struct Result: Sendable { var analysis: PoseAnalysis; var anchor: SubjectAnchor }
    func refine(url: URL, analysis: PoseAnalysis, bounds: SubjectBounds, at time: Double,
                progress: @escaping @Sendable (Double) async -> Void) async throws -> Result {
        guard bounds.valid, let path = Bundle.main.path(forResource: "pose_landmarker_full", ofType: "task") else { throw AnalysisError.missingModel }
        let options = PoseLandmarkerOptions()
        options.baseOptions.modelAssetPath = path; options.baseOptions.delegate = .CPU
        options.runningMode = .video; options.numPoses = 4
        var detector = try PoseLandmarker(options: options)
        var timestamp = 0
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 1920, height: 1920)
        generator.requestedTimeToleranceBefore = .zero; generator.requestedTimeToleranceAfter = .zero
        defer { generator.cancelAllCGImageGeneration() }
        var samples = analysis.people ?? []
        if samples.isEmpty {
            samples = (0..<max(1, Int(ceil(analysis.duration * 10)))).map { SubjectSample(time: Double($0)/10, candidates: []) }
        }
        let seedIndex = samples.indices.min { abs(samples[$0].time-time) < abs(samples[$1].time-time) }!
        let seedTime = samples[seedIndex].time
        let resolver = SubjectResolver(), token = UUID().uuidString
        func image(at index: Int) async throws -> CGImage {
            try Task.checkCancellation()
            if ProcessInfo.processInfo.thermalState == .critical { throw AnalysisError.overheated }
            return try await generator.image(at: CMTime(seconds: samples[index].time, preferredTimescale: 600)).image
        }
        func detect(_ index: Int, image: CGImage, around region: SubjectBounds) throws -> [SubjectCandidate] {
            try autoreleasepool {
                let crop = region.expanded(by: 0.45)
                let pixels = CGRect(x: crop.x*Double(image.width), y: crop.y*Double(image.height), width: crop.width*Double(image.width), height: crop.height*Double(image.height)).integral.intersection(CGRect(x: 0, y: 0, width: image.width, height: image.height))
                guard let close = image.cropping(to: pixels) else { return [] }
                let area = SubjectBounds(x: pixels.minX/Double(image.width), y: pixels.minY/Double(image.height), width: pixels.width/Double(image.width), height: pixels.height/Double(image.height))
                timestamp += 100
                let poses = try detector.detect(videoFrame: MPImage(uiImage: UIImage(cgImage: close)), timestampInMilliseconds: timestamp)
                return poses.landmarks.enumerated().compactMap { person, points in
                    guard points.count == 33 else { return nil }
                    let joints = points.map { point in Joint(x: rounded4(area.x+Double(point.x)*area.width), y: rounded4(area.y+Double(point.y)*area.height), confidence: rounded4(min(point.visibility?.doubleValue ?? 0, point.presence?.doubleValue ?? 0))) }
                    var pose = PoseFrame(time: samples[index].time, joints: joints)
                    guard pose.hasReliableTorso else { return nil }
                    if poses.worldLandmarks.indices.contains(person) {
                        pose.world = poses.worldLandmarks[person].map { [Double($0.x), Double($0.y), Double($0.z), $0.visibility?.doubleValue ?? 0].map { $0.isFinite ? rounded4($0) : 0 } }
                    }
                    let visible = joints.filter(\.valid)
                    guard let left = visible.map(\.x).min(), let right = visible.map(\.x).max(), let top = visible.map(\.y).min(), let bottom = visible.map(\.y).max(), right > left, bottom > top else { return nil }
                    return SubjectCandidate(id: "pose-\(index)-\(person)", bounds: .init(x:left,y:top,width:right-left,height:bottom-top), appearance: [], pose: pose)
                }
            }
        }
        func candidate(_ index: Int, observation: SubjectBoxTracker.Observation, image: CGImage) async throws -> SubjectCandidate {
            let box = observation.bounds
            let centre = SubjectBounds(x:box.x+box.width*0.2,y:box.y+box.height*0.2,width:box.width*0.6,height:box.height*0.6)
            var candidate = SubjectCandidate(id:"region-\(token)-\(index)",bounds:box,
                appearance:VideoAnalyzer.appearance(image:image,bounds:centre),pose:PoseFrame(time:samples[index].time,joints:[]),
                tracking:SubjectTracking(token:token,confidence:observation.confidence))
            let poses = try detect(index, image:image, around:box)
            let match = try await resolver.matchRegion(previous:candidate,candidates:poses,delta:0)
            if let pose = poses.first(where: { $0.id == match.id })?.pose { candidate.pose = pose }
            return candidate
        }
        func keep(_ candidate: SubjectCandidate, at index: Int) {
            samples[index].candidates.removeAll { $0.bounds.overlap(candidate.bounds) > 0.5 }
            samples[index].candidates.append(candidate)
        }
        let seedImage = try await image(at:seedIndex)
        let first = try await candidate(seedIndex,observation:.init(bounds:bounds,confidence:1),image:seedImage)
        keep(first,at:seedIndex)
        var done = 1
        for (direction,indices) in [Array((seedIndex+1)..<samples.count),Array((0..<seedIndex).reversed())].enumerated() {
            if indices.isEmpty { continue }
            let tracker = SubjectBoxTracker(bounds:bounds)
            _ = try tracker.next(image:seedImage)
            if direction == 1 {
                detector = try PoseLandmarker(options:options); timestamp = 0
                _ = try detect(seedIndex,image:seedImage,around:bounds)
            }
            var previous = first
            for index in indices {
                try Task.checkCancellation()
                let currentImage = try await image(at:index)
                guard let observation = try tracker.next(image:currentImage) else { break }
                let current = try await candidate(index,observation:observation,image:currentImage)
                let accepted = try await resolver.acceptTracking(previous:previous,candidate:current,seed:first,delta:abs(samples[index].time-previous.pose.time))
                #if DEBUG
                if ProcessInfo.processInfo.environment["BB_SUBJECT_TRACE"] == "1" {
                    print("Region sample \(samples[index].time): confidence=\(observation.confidence), pose=\(current.pose.hasReliableTorso), accepted=\(accepted)")
                }
                #endif
                guard accepted else { break }
                keep(current,at:index); previous = current
                done += 1
                if done % 5 == 0 { await progress(Double(done)/Double(samples.count)) }
            }
        }
        try Task.checkCancellation()
        var refined = analysis; refined.sampledFrames = samples.count; refined.people = samples; refined.peopleReference = nil
        await progress(1)
        return Result(analysis:refined,anchor:SubjectAnchor(time:seedTime,candidate:first))
    }
}
