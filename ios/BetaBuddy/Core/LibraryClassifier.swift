import AVFoundation
import CoreML
import CryptoKit
import Vision

/// Generic embedding comparison. Labels, prototypes and decision thresholds are supplied by
/// the private pack, never inferred from gym names or hard-coded into the native application.
struct LibraryClassifierConfig: Decodable {
    struct Prototype: Decodable { var tag: String?; var vector: [Double] }
    var schema: Int
    var model: String
    var samples: Int
    var minimumVotes: Int
    var minimumScore: Double
    var minimumMargin: Double
    var prototypes: [Prototype]

    func validate() throws {
        guard schema == 1, model == "LibraryImageEncoder", (3...20).contains(samples),
              (1...samples).contains(minimumVotes), minimumScore.isFinite, minimumMargin.isFinite,
              (-1...1).contains(minimumScore), (0...2).contains(minimumMargin),
              let width = prototypes.first?.vector.count, width > 0, width <= 4096,
              prototypes.contains(where: { $0.tag == nil }), prototypes.contains(where: { $0.tag != nil }),
              prototypes.allSatisfy({ $0.vector.count == width && $0.vector.allSatisfy(\.isFinite) }) else {
            throw CocoaError(.fileReadCorruptFile)
        }
    }
    func tag(for embedding: [Double]) -> String? {
        guard let width = prototypes.first?.vector.count, embedding.count == width, embedding.allSatisfy(\.isFinite) else { return nil }
        let ranked = prototypes.map { prototype in
            (tag: prototype.tag, score: zip(embedding, prototype.vector).reduce(0) { $0 + $1.0 * $1.1 })
        }.sorted { $0.score > $1.score }
        guard let best = ranked.first, let tag = best.tag, best.score >= minimumScore,
              let other = ranked.first(where: { $0.tag != tag }), best.score - other.score >= minimumMargin else { return nil }
        return tag
    }
    func decision(_ votes: [String?]) -> String? {
        let counts = Dictionary(grouping: votes.compactMap { $0 }, by: { $0 }).mapValues(\.count)
        let sorted = counts.sorted { $0.value > $1.value }
        guard let best = sorted.first, best.value >= minimumVotes,
              sorted.count == 1 || best.value > sorted[1].value else { return nil }
        return best.key
    }
}

actor LibraryClassifier {
    private let model: VNCoreMLModel
    let config: LibraryClassifierConfig
    nonisolated let revision: String

    init(pack: GymPack, resources: URL? = Bundle.main.resourceURL) throws {
        let data = try Data(contentsOf: pack.directory.appendingPathComponent("library-classifier.json"))
        config = try JSONDecoder().decode(LibraryClassifierConfig.self, from: data)
        try config.validate()
        revision = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard let resources else { throw CocoaError(.fileNoSuchFile) }
        let settings = MLModelConfiguration()
        #if targetEnvironment(simulator)
        settings.computeUnits = .cpuOnly
        #else
        settings.computeUnits = .all
        #endif
        model = try VNCoreMLModel(for: MLModel(contentsOf: resources.appendingPathComponent(config.model + ".mlmodelc"), configuration: settings))
    }

    func classify(_ asset: AVAsset) async throws -> String? {
        let duration = try await asset.load(.duration).seconds
        guard duration.isFinite, duration > 0, duration <= 1200 else { throw AnalysisError.invalidVideo }
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 960, height: 960)
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        var votes: [String?] = []
        for sample in 0..<config.samples {
            try Task.checkCancellation()
            let time = duration * (Double(sample) + 0.5) / Double(config.samples)
            let (image, _) = try await generator.image(at: CMTime(seconds: time, preferredTimescale: 600))
            let request = VNCoreMLRequest(model: model)
            request.imageCropAndScaleOption = .centerCrop
            try VNImageRequestHandler(cgImage: image).perform([request])
            guard let result = request.results?.compactMap({ $0 as? VNCoreMLFeatureValueObservation }).first,
                  let vector = result.featureValue.multiArrayValue else { throw AnalysisError.invalidVideo }
            let embedding = (0..<vector.count).map { vector[$0].doubleValue }
            votes.append(config.tag(for: embedding))
        }
        try Task.checkCancellation()
        return config.decision(votes)
    }
}
