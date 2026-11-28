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
    var imageResize: String? = nil

    var imageModes: [VNImageCropAndScaleOption] {
        switch imageResize {
        case "fit": return [.scaleFit]
        case "stretch": return [.scaleFill]
        case "centerAndFit": return [.centerCrop, .scaleFit]
        default: return [.centerCrop]
        }
    }

    func validate() throws {
        guard (1...2).contains(schema), schema == 1 || imageResize != nil,
              imageResize == nil || ["center", "fit", "stretch", "centerAndFit"].contains(imageResize!),
              model == "LibraryImageEncoder", (3...20).contains(samples),
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
    /// Several views of one frame still count as one observation; disagreement abstains.
    func frameVote(_ votes: [String?]) -> String? {
        let tags = Set(votes.compactMap { $0 })
        return tags.count == 1 ? tags.first : nil
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
        revision = SHA256.hash(data: data + Data("|image-sampling-2".utf8)).map { String(format: "%02x", $0) }.joined()
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
        generator.maximumSize = CGSize(width: 3840, height: 3840)
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        var votes: [String?] = []
        for sample in 0..<config.samples {
            try Task.checkCancellation()
            let time = duration * (Double(sample) + 0.5) / Double(config.samples)
            let (image, _) = try await generator.image(at: CMTime(seconds: time, preferredTimescale: 600))
            var views: [String?] = []
            for mode in config.imageModes {
                try Task.checkCancellation()
                views.append(config.tag(for: try embedding(image, mode: mode)))
            }
            votes.append(config.frameVote(views))
        }
        try Task.checkCancellation()
        return config.decision(votes)
    }

    func embedding(_ image: CGImage, mode: VNImageCropAndScaleOption) throws -> [Double] {
        let request = VNCoreMLRequest(model: model)
        request.imageCropAndScaleOption = .scaleFill
        try VNImageRequestHandler(cgImage: try Self.sampledImage(image, mode: mode)).perform([request])
        guard let result = request.results?.compactMap({ $0 as? VNCoreMLFeatureValueObservation }).first,
              let vector = result.featureValue.multiArrayValue else { throw AnalysisError.invalidVideo }
        return (0..<vector.count).map { vector[$0].doubleValue }
    }

    /// Scale-aware triangle filtering uses all source pixels covered by a reduced pixel.
    /// Integer padding and RGB rounding match the other native encoder.
    static func sampledImage(_ image: CGImage, mode: VNImageCropAndScaleOption) throws -> CGImage {
        try Task.checkCancellation()
        let width = image.width, height = image.height, side = min(width,height)
        let sw = mode == .centerCrop ? side : width, sh = mode == .centerCrop ? side : height
        let left = (width-sw)/2, top = (height-sh)/2
        let scale = 224.0/Double(max(sw,sh))
        let dw = mode == .scaleFit ? max(1,Int((Double(sw)*scale).rounded(.toNearestOrEven))) : 224
        let dh = mode == .scaleFit ? max(1,Int((Double(sh)*scale).rounded(.toNearestOrEven))) : 224
        let padX = Int((Double(224-dw)/2).rounded(.toNearestOrEven)), padY = Int((Double(224-dh)/2).rounded(.toNearestOrEven))
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        var pixels = [UInt8](repeating: 0, count: width*height*4)
        let drawn = pixels.withUnsafeMutableBytes { bytes -> Bool in
            guard let context = CGContext(data: bytes.baseAddress, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width*4, space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
            context.interpolationQuality = .none
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { throw AnalysisError.invalidVideo }
        let wx = weights(sw,dw), wy = weights(sh,dh)
        var horizontal = [UInt8](repeating: 0, count: sh*dw*3)
        for y in 0..<sh {
            try Task.checkCancellation()
            for x in 0..<dw { for channel in 0..<3 {
                var value = 0.0
                for (index, weight) in wx[x] { value += Double(pixels[((top+y)*width+left+index)*4+channel])*weight }
                horizontal[(y*dw+x)*3+channel] = UInt8(clamping: Int(value+0.5))
            } }
        }
        var result = [UInt8](repeating: 0, count: 224*224*4)
        for i in 0..<(224*224) { result[i*4+3] = 255 }
        for y in 0..<dh {
            try Task.checkCancellation()
            for x in 0..<dw { for channel in 0..<3 {
                var value = 0.0
                for (index, weight) in wy[y] { value += Double(horizontal[(index*dw+x)*3+channel])*weight }
                result[((y+padY)*224+x+padX)*4+channel] = UInt8(clamping: Int(value+0.5))
            } }
        }
        guard let provider = CGDataProvider(data: Data(result) as CFData),
              let output = CGImage(width: 224, height: 224, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 224*4,
                space: space, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else { throw AnalysisError.invalidVideo }
        return output
    }

    private static func weights(_ source: Int, _ target: Int) -> [[(Int,Double)]] {
        let scale = Double(source)/Double(target), radius = max(1,scale)
        return (0..<target).map { output in
            let center = (Double(output)+0.5)*scale
            let lo = max(0,Int(floor(center-radius))), hi = min(source-1,Int(ceil(center+radius)))
            let values: [(Int,Double)] = (lo...hi).compactMap { input in
                let weight = max(0,1-abs(Double(input)+0.5-center)/radius)
                return weight > 0 ? (input,weight) : nil
            }
            let sum = values.reduce(0) { $0+$1.1 }
            return values.map { ($0.0,$0.1/sum) }
        }
    }

}
