import Foundation
import JavaScriptCore

/// Executes a pure, locally bundled policy with no native bridges or networking facilities.
/// Climbing-specific automatic selection and identity association stay in the private pack.
actor SubjectResolver {
    private var regionContext: JSContext?
    static var available: Bool { GymPack.current?.manifest.files.contains { $0.path == "subject.js" } == true }
    private struct Input: Encodable {
        var schema = 1
        var samples: [SubjectSample]
        var anchors: [SubjectAnchor]
    }
    struct RegionMatch: Decodable, Sendable { var id: String?; var stop: Bool }
    func matchRegion(previous: SubjectCandidate, candidates: [SubjectCandidate], delta: Double) throws -> RegionMatch {
        struct RegionInput: Encodable { var previous: SubjectCandidate; var candidates: [SubjectCandidate]; var dt: Double }
        if regionContext == nil {
            guard let pack = GymPack.current, let context = JSContext() else { throw CocoaError(.fileReadCorruptFile) }
            context.evaluateScript(try String(contentsOf: pack.directory.appendingPathComponent("subject.js"), encoding: .utf8))
            guard context.exception == nil else { throw CocoaError(.fileReadCorruptFile) }
            regionContext = context
        }
        let json = String(decoding: try JSONEncoder().encode(RegionInput(previous: previous, candidates: candidates, dt: delta)), as: UTF8.self)
        let function = regionContext?.evaluateScript("(function(json) { return JSON.stringify(BetaBuddySubject.matchRegion(JSON.parse(json))); })")
        let result = function?.call(withArguments: [json])
        guard regionContext?.exception == nil else { throw CocoaError(.fileReadCorruptFile) }
        guard let text = result?.toString() else { throw CocoaError(.fileReadCorruptFile) }
        return try JSONDecoder().decode(RegionMatch.self, from: Data(text.utf8))
    }
    func resolve(_ analysis: PoseAnalysis, anchors: [SubjectAnchor], pack: GymPack? = GymPack.current) throws -> PoseAnalysis {
        try Task.checkCancellation()
        guard let people = analysis.people, let pack,
              pack.manifest.files.contains(where: { $0.path == "subject.js" }) else { throw CocoaError(.fileNoSuchFile) }
        let source = try String(contentsOf: pack.directory.appendingPathComponent("subject.js"), encoding: .utf8)
        guard let context = JSContext() else { throw CocoaError(.fileReadCorruptFile) }
        context.evaluateScript(source)
        guard context.exception == nil else { throw CocoaError(.fileReadCorruptFile) }
        // Association uses image-space joints. Keep the larger world poses out of the JS heap;
        // the chosen ids still select the original, complete poses after resolution.
        let samples = people.map { sample in
            SubjectSample(time: sample.time, candidates: sample.candidates.map { candidate in
                var copy = candidate; copy.pose.world = nil; return copy
            })
        }
        let data = try JSONEncoder().encode(Input(samples: samples, anchors: anchors))
        guard let json = String(data: data, encoding: .utf8) else { throw CocoaError(.fileReadCorruptFile) }
        // Argument is passed as a JS string, never interpolated into source code.
        let function = context.evaluateScript("(function(json) { return JSON.stringify(BetaBuddySubject.resolve(JSON.parse(json))); })")
        guard let output = function?.call(withArguments: [json])?.toString(), context.exception == nil,
              let resultData = output.data(using: .utf8) else { throw CocoaError(.fileReadCorruptFile) }
        let result = try JSONDecoder().decode(SubjectResolution.self, from: resultData)
        try Task.checkCancellation()
        return try analysis.selecting(result)
    }
}
