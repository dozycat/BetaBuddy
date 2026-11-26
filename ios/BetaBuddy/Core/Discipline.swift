import Foundation

/// Stored sentinel for a climb without a grade. Kept as is so existing archives stay readable.
let ungradedValue = "未标级"

/// How an attempt was climbed. The raw value is what is stored and what the pack's pages use.
enum Discipline: String, CaseIterable, Sendable {
    case boulder, toprope, lead
}

/// The grades offered for each discipline, as the raw strings that are stored.
enum GradeScale {
    static let boulder = (0...16).map { "V\($0)" }
    /// YDS: 5.5 … 5.9, then 5.10a … 5.14d.
    static let rope = (5...9).map { "5.\($0)" } + (10...14).flatMap { number in ["a", "b", "c", "d"].map { "5.\(number)\($0)" } }

    /// No discipline is treated as bouldering wherever a scale is needed.
    static func grades(for discipline: Discipline?) -> [String] { (discipline ?? .boulder) == .boulder ? boulder : rope }

    /// A grade of the other scale goes back to ungraded. Anything else, including grades older
    /// builds wrote in another form, is kept as it is.
    static func kept(_ grade: String, for discipline: Discipline?) -> String {
        let other = (discipline ?? .boulder) == .boulder ? rope : boulder
        return other.contains(grade) ? ungradedValue : grade
    }
}

/// What the analysis page made of the clip: a guess, unless the user's own choice was handed to it.
struct DisciplineGuess: Equatable, Sendable {
    var kind: Discipline
    var sure: Bool

    init(kind: Discipline, sure: Bool) { self.kind = kind; self.sure = sure }

    /// The `discipline` field of an `analysisResult` message. `nil` for anything else.
    init?(body: Any?) {
        guard let body = body as? [String: Any], let raw = body["kind"] as? String, let kind = Discipline(rawValue: raw) else { return nil }
        self.kind = kind
        sure = (body["sure"] as? NSNumber)?.boolValue ?? false
    }
}
