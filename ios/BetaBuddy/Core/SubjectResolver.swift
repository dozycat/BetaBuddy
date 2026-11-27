import Foundation
import JavaScriptCore

/// Executes a pure, locally bundled policy with no native bridges or networking facilities.
/// Climbing-specific automatic selection and identity association stay in the private pack.
actor SubjectResolver {
    static var available: Bool { GymPack.current?.manifest.files.contains { $0.path == "subject.js" } == true }
    private struct Input: Encodable {
        var schema = 1
        var samples: [SubjectSample]
        var anchors: [SubjectAnchor]
    }
    func resolve(_ analysis: PoseAnalysis, anchors: [SubjectAnchor], pack: GymPack? = GymPack.current) throws -> PoseAnalysis {
        try Task.checkCancellation()
        guard let people = analysis.people, let pack,
              pack.manifest.files.contains(where: { $0.path == "subject.js" }) else { throw CocoaError(.fileNoSuchFile) }
        let source = try String(contentsOf: pack.directory.appendingPathComponent("subject.js"), encoding: .utf8)
        guard let context = JSContext() else { throw CocoaError(.fileReadCorruptFile) }
        context.evaluateScript(source)
        guard context.exception == nil else { throw CocoaError(.fileReadCorruptFile) }
        let data = try JSONEncoder().encode(Input(samples: people, anchors: anchors))
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
