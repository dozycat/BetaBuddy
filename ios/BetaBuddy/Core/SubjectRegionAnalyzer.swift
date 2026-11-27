import AVFoundation
import MediaPipeTasksVision
import UIKit

/// A user-drawn area gives the pose model a closer view. It is sampled in both directions from
/// that moment. The private policy decides continuity; missing poses never become invented ones.
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
            samples = (0..<max(1, Int(analysis.duration * 10))).map { SubjectSample(time: Double($0)/10, candidates: []) }
        }
        let seedIndex = samples.indices.min { abs(samples[$0].time-time) < abs(samples[$1].time-time) }!
        let seedTime = samples[seedIndex].time
        let resolver = SubjectResolver()
        let seed = SubjectCandidate(id: "seed", bounds: bounds, appearance: [], pose: PoseFrame(time: seedTime, joints: []))
        let token = UUID().uuidString
        func detect(_ index: Int, around region: SubjectBounds) async throws -> [SubjectCandidate] {
            try Task.checkCancellation()
            let time = samples[index].time
            let (image, _) = try await generator.image(at: CMTime(seconds: time, preferredTimescale: 600))
            return try autoreleasepool {
                let crop = region.expanded(by: 0.45)
                let pixels = CGRect(x: crop.x*Double(image.width), y: crop.y*Double(image.height), width: crop.width*Double(image.width), height: crop.height*Double(image.height)).integral.intersection(CGRect(x: 0, y: 0, width: image.width, height: image.height))
                guard let close = image.cropping(to: pixels) else { return [] }
                let area = SubjectBounds(x: pixels.minX/Double(image.width), y: pixels.minY/Double(image.height), width: pixels.width/Double(image.width), height: pixels.height/Double(image.height))
                timestamp += 100
                let poses = try detector.detect(videoFrame: MPImage(uiImage: UIImage(cgImage: close)), timestampInMilliseconds: timestamp)
                return poses.landmarks.enumerated().compactMap { person, points in
                    guard points.count == 33 else { return nil }
                    let joints = points.map { point in Joint(x: rounded4(area.x+Double(point.x)*area.width), y: rounded4(area.y+Double(point.y)*area.height), confidence: rounded4(min(point.visibility?.doubleValue ?? 0, point.presence?.doubleValue ?? 0))) }
                    var pose = PoseFrame(time: time, joints: joints)
                    guard pose.hasReliableTorso else { return nil }
                    if poses.worldLandmarks.indices.contains(person) {
                        pose.world = poses.worldLandmarks[person].map { [Double($0.x), Double($0.y), Double($0.z), $0.visibility?.doubleValue ?? 0].map { $0.isFinite ? rounded4($0) : 0 } }
                    }
                    let visible = joints.filter(\.valid)
                    guard let left = visible.map(\.x).min(), let right = visible.map(\.x).max(), let top = visible.map(\.y).min(), let bottom = visible.map(\.y).max(), right > left, bottom > top else { return nil }
                    let box = SubjectBounds(x: left, y: top, width: right-left, height: bottom-top)
                    let torso = [joints[11], joints[12], joints[23], joints[24]]
                    let tx = torso.map(\.x).min()!, ty = torso.map(\.y).min()!
                    let colour = VideoAnalyzer.appearance(image: image, bounds: .init(x: tx, y: ty, width: torso.map(\.x).max()!-tx, height: torso.map(\.y).max()!-ty))
                    return SubjectCandidate(id: "region-\(token)-\(index)-\(person)", bounds: box, appearance: colour, pose: pose)
                }
            }
        }
        let seedCandidates = try await detect(seedIndex, around: bounds)
        let initial = try await resolver.matchRegion(previous: seed, candidates: seedCandidates, delta: 0)
        guard let first = seedCandidates.first(where: { $0.id == initial.id }) else { throw AnalysisError.noPose }
        func keep(_ candidate: SubjectCandidate, at index: Int) {
            // The same person may now have two detections, one from the full image and one from
            // the close view. Keep the close view instead of feeding duplicate bodies to association.
            samples[index].candidates.removeAll { $0.bounds.overlap(candidate.bounds) > 0.5 }
            samples[index].candidates.append(candidate)
        }
        keep(first, at: seedIndex)
        var done = 1
        for (direction, indices) in [Array((seedIndex+1)..<samples.count), Array((0..<seedIndex).reversed())].enumerated() {
            if direction == 1 && !indices.isEmpty {
                // Reverse playback is a new sequence for the detector, initialized at the same seed.
                detector = try PoseLandmarker(options: options); timestamp = 0
                _ = try await detect(seedIndex, around: bounds)
            }
            var previous = first
            for index in indices {
                try Task.checkCancellation()
                // Keep the user's initial context as a minimum. Visible-joint bounds shrink
                // when an arm or foot is hidden; using that smaller crop can exclude the body
                // context the detector needs in the very next frame.
                let width = max(bounds.width, previous.bounds.width), height = max(bounds.height, previous.bounds.height)
                let region = SubjectBounds(x: max(0, min(1-width, previous.bounds.x+previous.bounds.width/2-width/2)), y: max(0, min(1-height, previous.bounds.y+previous.bounds.height/2-height/2)), width: width, height: height)
                let candidates = try await detect(index, around: region)
                let match = try await resolver.matchRegion(previous: previous, candidates: candidates, delta: abs(samples[index].time-previous.pose.time))
                #if DEBUG
                if ProcessInfo.processInfo.environment["BB_SUBJECT_TRACE"] == "1" {
                    print("Region sample \(samples[index].time): candidates=\(candidates.count), matched=\(match.id != nil), stop=\(match.stop)")
                }
                #endif
                if match.stop { break }
                if let candidate = candidates.first(where: { $0.id == match.id }) { keep(candidate, at: index); previous = candidate }
                done += 1
                if done % 5 == 0 { await progress(Double(done)/Double(samples.count)) }
            }
        }
        try Task.checkCancellation()
        var refined = analysis; refined.people = samples; refined.peopleReference = nil
        await progress(1)
        return Result(analysis: refined, anchor: SubjectAnchor(time: seedTime, candidate: first))
    }
}
