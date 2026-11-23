import Foundation

/// Finds the bundle these files were compiled into.
final class TestSupportMarker {}

/// A synthetic gym pack for tests: an entry page that only says it is ready, and stand-ins for the
/// capture and analysis pages that answer from a script. No venue, route or algorithm is in it.
enum StandInPack {
    static let version = "0.0.0-standin"
    /// The made-up wall the capture stand-in reports, as `(colour key, hold id, x, y)` normalised to the picture.
    static let holds: [(color: String, id: String, x: Double, y: Double)] = [
        ("yellow", "y1", 0.30, 0.80), ("yellow", "y2", 0.42, 0.78), ("yellow", "y3", 0.36, 0.90), ("yellow", "y4", 0.50, 0.60), ("yellow", "y5", 0.40, 0.42), ("yellow", "y6", 0.55, 0.22),
        ("blue", "b1", 0.70, 0.82), ("blue", "b2", 0.78, 0.66), ("blue", "b3", 0.66, 0.50), ("blue", "b4", 0.80, 0.36), ("blue", "b5", 0.72, 0.18),
        ("red", "r1", 0.15, 0.70), ("red", "r2", 0.20, 0.50), ("red", "r3", 0.12, 0.30)
    ]
    static let colors: [String: (red: Double, green: Double, blue: Double)] = [
        "yellow": (226, 196, 40), "blue": (40, 110, 220), "red": (214, 52, 44)
    ]
    /// A venue of the synthetic pack and the directory entry that stands for it.
    static let venueID = "synthetic-east"
    static let venueName = "合成岩馆 抱石 东门店"
    static let directoryGymName = "合成岩馆（东门）"

    /// Writes `<root>/GymPack` and returns `root`. `step` is how many frames each scripted state of the
    /// capture stand-in lasts.
    @discardableResult
    static func write(to root: URL, capture: Bool = true, analysis: Bool = true, step: Int = 5, directoryIds: Bool = true) throws -> URL {
        let pack = root.appendingPathComponent("GymPack", isDirectory: true)
        try FileManager.default.createDirectory(at: pack, withIntermediateDirectories: true)
        let bundle = Bundle(for: TestSupportMarker.self)
        var files = ["index.html": "<!doctype html><meta charset=\"utf-8\"><title>synthetic pack</title><p>synthetic pack</p><script>window.BetaBuddyGym={openWall(){},openRoute(){},openVenue(){},state(){return {ready:true}}};try{window.webkit.messageHandlers.gym.postMessage({type:'ready',packVersion:'\(version)',routeCount:0})}catch(e){}</script>"]
        func page(_ name: String) throws -> String {
            guard let url = bundle.url(forResource: name, withExtension: "html") else { throw CocoaError(.fileNoSuchFile) }
            return try String(contentsOf: url, encoding: .utf8)
        }
        if capture { files["capture.html"] = try page("capture-standin").replacingOccurrences(of: "const STEP = 5;", with: "const STEP = \(step);") }
        if analysis { files["analysis.html"] = try page("analysis-standin") }
        for (name, text) in files { try Data(text.utf8).write(to: pack.appendingPathComponent(name)) }
        let manifest: [String: Any] = [
            "schema": 1, "packVersion": version, "generatedAt": "2026-01-01T00:00:00Z", "gymName": "合成岩馆", "routeCount": 0, "entry": "index.html",
            "files": files.keys.sorted().map { ["path": $0, "sha256": "00", "bytes": files[$0]!.utf8.count] },
            // Newer packs say which directory gym a venue is; older ones leave the app to go by name.
            "venues": [["id": venueID, "files": [String](), "name": venueName, "names": ["en": "Synthetic Gym East"]]
                .merging(directoryIds ? ["directoryIds": ["syn:east"]] : [:]) { first, _ in first }]
        ]
        try JSONSerialization.data(withJSONObject: manifest).write(to: pack.appendingPathComponent("manifest.json"))
        return root
    }

    /// A synthetic gym directory with the optional coordinate fields, around a made-up position.
    static let origin = (latitude: 30.0, longitude: 120.0)
    static let directory = """
    {"version":1,"updatedAt":"2026-01-01","gyms":[
     {"id":"syn:east","name":"\(directoryGymName)","province":"合成省","city":"合成市","district":"东区","address":"东门路 1 号","sourceURL":"https://example.invalid/","lat":30.0018,"lon":120.0,"coordPrecision":"poi"},
     {"id":"syn:station","name":"只知道最近车站的岩馆","province":"合成省","city":"合成市","district":"北区","address":"","sourceURL":"https://example.invalid/","lat":30.0115,"lon":120.0,"coordPrecision":"station"},
     {"id":"syn:west","name":"合成岩馆（西门）","province":"合成省","city":"合成市","district":"西区","address":"西门路 2 号","sourceURL":"https://example.invalid/","lat":30.0,"lon":119.95,"coordPrecision":"address"},
     {"id":"syn:district","name":"只知道在哪个区的岩馆","province":"合成省","city":"合成市","district":"南区","address":"","sourceURL":"https://example.invalid/","lat":29.99,"lon":120.0,"coordPrecision":"district"},
     {"id":"syn:city","name":"只知道在哪个市的岩馆","province":"合成省","city":"合成市","address":"","sourceURL":"https://example.invalid/","lat":30.02,"lon":120.02,"coordPrecision":"city"},
     {"id":"syn:far","name":"很远的岩馆","province":"远方省","city":"远方市","address":"","sourceURL":"https://example.invalid/","lat":40.0,"lon":116.0,"coordPrecision":"poi"},
     {"id":"syn:nowhere","name":"没有坐标的岩馆","province":"合成省","city":"合成市","address":"","sourceURL":"https://example.invalid/"},
     {"id":"syn:odd","name":"精度写法不认识的岩馆","province":"合成省","city":"合成市","address":"","sourceURL":"https://example.invalid/","lat":30.0001,"lon":120.0,"coordPrecision":"planet"},
     {"id":"syn:broken","name":"坐标写坏了的岩馆","province":"合成省","city":"合成市","address":"","sourceURL":"https://example.invalid/","lat":"north","lon":null,"coordPrecision":7}
    ]}
    """
}
