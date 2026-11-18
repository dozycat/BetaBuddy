import Foundation

/// `manifest.json` of a gym pack. The pack itself (model, routes, solver, web page) is built
/// outside this repository and is never checked in; this type is the whole native-side contract.
/// See docs/ios/gym-design.md. Both camelCase and snake_case keys are accepted.
struct GymManifest: Decodable, Equatable, Sendable {
    struct File: Equatable, Sendable {
        var path: String
        var sha256: String
    }
    static let supportedSchema = 1
    var schema: Int
    var packVersion: String
    var generatedAt: String
    var gymName: String
    /// Optional per-language names, keyed by the app's language codes.
    var gymNameLocalized: [String: String] = [:]
    var routeCount: Int
    var entry: String
    var files: [File]

    private struct Key: CodingKey {
        var stringValue: String
        var intValue: Int? { nil }
        init(_ value: String) { stringValue = value }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { nil }
    }
    private struct FileEntry: Decodable {
        var path: String?
        var sha256: String
    }

    init(from decoder: Decoder) throws {
        let box = try decoder.container(keyedBy: Key.self)
        func first<T: Decodable>(_ type: T.Type, _ names: String...) throws -> T? {
            for name in names { if let value = try box.decodeIfPresent(type, forKey: Key(name)) { return value } }
            return nil
        }
        func required<T>(_ value: T?, _ name: String) throws -> T {
            guard let value else {
                throw DecodingError.keyNotFound(Key(name), .init(codingPath: box.codingPath, debugDescription: "manifest 缺少 \(name)"))
            }
            return value
        }
        schema = try first(Int.self, "schema", "schemaVersion", "schema_version") ?? 1
        packVersion = try required(first(String.self, "packVersion", "pack_version", "version"), "packVersion")
        generatedAt = try required(first(String.self, "generatedAt", "generated_at"), "generatedAt")
        var name = try first(String.self, "gymName", "gym_name")
        if name == nil, let gym = try? box.nestedContainer(keyedBy: Key.self, forKey: Key("gym")) {
            name = try gym.decodeIfPresent(String.self, forKey: Key("name"))
        }
        gymName = try required(name, "gymName")
        gymNameLocalized = (try? box.decodeIfPresent([String: String].self, forKey: Key("gymNameLocalized"))) ?? [:]
        routeCount = try required(first(Int.self, "routeCount", "route_count"), "routeCount")
        entry = try first(String.self, "entry") ?? "index.html"
        // Either [{path, sha256}] or {path: sha256} / {path: {sha256}}.
        if let list = try? box.decode([FileEntry].self, forKey: Key("files")) {
            files = try list.map { File(path: try required($0.path, "files[].path"), sha256: $0.sha256) }
        } else if let map = try? box.decode([String: String].self, forKey: Key("files")) {
            files = map.map { File(path: $0.key, sha256: $0.value) }
        } else {
            let map = try box.decode([String: FileEntry].self, forKey: Key("files"))
            files = map.map { File(path: $0.key, sha256: $0.value.sha256) }
        }
        files.sort { $0.path < $1.path }
    }
}

enum GymPackError: Error, Equatable {
    case unreadableManifest
    case unsupportedSchema(Int)
    case unsafePath(String)
    case missingFile(String)
}

/// A gym pack found on disk. Hashes are verified by `ios/scripts/install-gym-pack.sh` at build
/// time and the bundle is code signed afterwards, so launch only checks structure and presence.
struct GymPack: Equatable, Sendable {
    enum Status: Equatable, Sendable {
        case missing            // This build ships without a pack, e.g. a clone of the public repository.
        case invalid(GymPackError)
        case ready(GymPack)
    }
    static let directoryName = "GymPack"
    var directory: URL
    var manifest: GymManifest
    /// The page is served to the web view as `gympack://local/<path>` rather than `file://`,
    /// so `fetch`, loaders and workers inside the pack behave as on a normal origin.
    static let scheme = "gympack"
    static let host = "local"
    var entryURL: URL {
        var parts = URLComponents()
        parts.scheme = Self.scheme; parts.host = Self.host; parts.path = "/" + manifest.entry
        return parts.url!
    }
    /// Entry URL for a load. `hint=0` tells the page its first-run gesture hint has already been seen;
    /// `lang` is the app's interface language. `extraQuery` is appended as given (Debug builds only
    /// pass one); if it does not form a URL it is dropped.
    func entryURL(hintSeen: Bool, language: String? = nil, extraQuery: String? = nil) -> URL {
        let extra = (extraQuery ?? "").trimmingCharacters(in: CharacterSet(charactersIn: "?&"))
        let known = (hintSeen ? ["hint=0"] : []) + (language.map { ["lang=\($0)"] } ?? [])
        func url(_ items: [String]) -> URL? {
            items.isEmpty ? entryURL : URL(string: entryURL.absoluteString + "?" + items.joined(separator: "&")).flatMap { allows($0) ? $0 : nil }
        }
        return (extra.isEmpty ? nil : url(known + [extra])) ?? url(known) ?? entryURL
    }

    static let status: Status = status(in: Bundle.main.resourceURL)
    /// `nil` when this build has no usable pack.
    static var current: GymPack? { if case .ready(let pack) = status { pack } else { nil } }

    static func status(in resources: URL?) -> Status {
        guard let directory = resources?.appendingPathComponent(directoryName, isDirectory: true),
              FileManager.default.fileExists(atPath: directory.appendingPathComponent("manifest.json").path) else { return .missing }
        do { return .ready(try GymPack(directory: directory)) }
        catch let error as GymPackError { return .invalid(error) }
        catch { return .invalid(.unreadableManifest) }
    }

    init(directory: URL) throws {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("manifest.json")),
              let manifest = try? JSONDecoder().decode(GymManifest.self, from: data) else { throw GymPackError.unreadableManifest }
        guard manifest.schema == GymManifest.supportedSchema else { throw GymPackError.unsupportedSchema(manifest.schema) }
        for path in [manifest.entry] + manifest.files.map(\.path) {
            guard Self.isSafe(path) else { throw GymPackError.unsafePath(path) }
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: directory.appendingPathComponent(path).path, isDirectory: &isDirectory),
                  !isDirectory.boolValue else { throw GymPackError.missingFile(path) }
        }
        self.directory = directory.standardizedFileURL
        self.manifest = manifest
    }

    /// Relative, inside the pack, no parent traversal.
    static func isSafe(_ path: String) -> Bool {
        !path.isEmpty && !path.hasPrefix("/") && !path.contains("\\")
            && !path.split(separator: "/", omittingEmptySubsequences: false).contains { $0.isEmpty || $0 == "." || $0 == ".." }
    }

    /// Only the pack's own origin may be navigated to. Everything else, including any network URL, is refused.
    func allows(_ url: URL?) -> Bool {
        url?.scheme?.lowercased() == Self.scheme && url?.host?.lowercased() == Self.host
    }

    /// The file a pack URL stands for, or `nil` when it is not an existing regular file that is
    /// still inside the pack directory after `..` and symlinks are resolved.
    func file(for url: URL?) -> URL? {
        guard allows(url), let path = url?.path, path.count > 1, !path.contains("\0") else { return nil }
        let root = directory.resolvingSymlinksInPath().path
        let target = URL(fileURLWithPath: root + path).standardizedFileURL.resolvingSymlinksInPath()
        var isDirectory: ObjCBool = false
        guard target.path.hasPrefix(root + "/"),
              FileManager.default.fileExists(atPath: target.path, isDirectory: &isDirectory), !isDirectory.boolValue else { return nil }
        return target
    }

    static func mimeType(forExtension pathExtension: String) -> String {
        switch pathExtension.lowercased() {
        case "html", "htm": "text/html; charset=utf-8"
        case "js", "mjs": "text/javascript; charset=utf-8"
        case "json", "map": "application/json; charset=utf-8"
        case "css": "text/css; charset=utf-8"
        case "glb": "model/gltf-binary"
        case "gltf": "model/gltf+json"
        case "wasm": "application/wasm"
        case "png": "image/png"
        case "jpg", "jpeg": "image/jpeg"
        case "webp": "image/webp"
        case "svg": "image/svg+xml"
        case "ktx2": "image/ktx2"
        case "txt": "text/plain; charset=utf-8"
        case "md": "text/markdown; charset=utf-8"
        default: "application/octet-stream"   // .bin and anything unknown
        }
    }
}

/// A route the user picked inside the pack page. Only the id and display name cross the bridge.
struct GymRoute: Codable, Equatable, Sendable {
    var id: String
    var name: String
}

/// Page → app messages on the `gym` script message handler.
enum GymMessage: Equatable, Sendable {
    case ready
    case hintSeen
    case openSettings
    case routeSelected(GymRoute?)
    case error(String)

    init?(body: Any) {
        guard let body = body as? [String: Any], let type = body["type"] as? String else { return nil }
        switch type {
        case "ready": self = .ready
        case "hintSeen": self = .hintSeen
        case "openSettings": self = .openSettings
        case "routeSelected":
            if let id = body["id"] as? String, !id.isEmpty {
                self = .routeSelected(GymRoute(id: String(id.prefix(120)), name: String((body["name"] as? String ?? id).prefix(120))))
            } else { self = .routeSelected(nil) }
        case "error": self = .error(String((body["message"] as? String ?? "").prefix(300)))
        default: return nil
        }
    }
}

/// App → page calls. The page exposes `window.BetaBuddyGym.openWall(id)` / `.openRoute(id)`.
enum GymCommand: Equatable, Sendable {
    case openWall(String)
    case openRoute(String)

    var script: String {
        let (function, id) = switch self {
        case .openWall(let id): ("openWall", id)
        case .openRoute(let id): ("openRoute", id)
        }
        // JSON string encoding keeps ids from breaking out of the call.
        let argument = (try? JSONEncoder().encode(id)).flatMap { String(data: $0, encoding: .utf8) } ?? "\"\""
        return "window.BetaBuddyGym && typeof window.BetaBuddyGym.\(function) === 'function' ? (window.BetaBuddyGym.\(function)(\(argument)), true) : false"
    }
}

extension GymManifest {
    /// The gym's name in the interface language when the pack provides one; the pack's own name otherwise.
    func displayName(language: String = AppLanguage.current) -> String {
        if let name = gymNameLocalized[language], !name.isEmpty { return name }
        return gymName
    }
}

// MARK: - Video analysis page

extension GymPack {
    static let analysisPage = "analysis.html"
    /// Whether this pack carries the video analysis page. Without it the app shows no entry for it.
    var supportsAnalysis: Bool { manifest.files.contains { $0.path == Self.analysisPage } }
    /// The analysis page is loaded the same way as the entry page, from the same directory.
    func analysisURL(language: String) -> URL? {
        guard supportsAnalysis else { return nil }
        var parts = URLComponents()
        parts.scheme = Self.scheme; parts.host = Self.host; parts.path = "/" + Self.analysisPage
        parts.queryItems = [URLQueryItem(name: "lang", value: language)]
        return parts.url.flatMap { allows($0) ? $0 : nil }
    }
}

/// Analysis page → app messages. They arrive on the same `gym` handler as `GymMessage`,
/// from the analysis page's own web view. What the page computes is the pack's business:
/// the app keeps the result as an opaque string and hands it back unchanged.
enum AnalysisMessage: Equatable, Sendable {
    case ready
    /// `id` is the record id given in `begin`, or empty when the page was showing a saved result.
    case result(id: String, json: String, climber: [String: Double])
    case seek(Double)
    /// The page could not analyse the clip and is showing its own explanation.
    case failed(code: String)
    /// A script error. Before `ready` it means the page never came up.
    case pageError(String)

    static let maximumResultLength = 8_000_000

    init?(body: Any) {
        guard let body = body as? [String: Any], let type = body["type"] as? String else { return nil }
        switch type {
        case "analysisReady": self = .ready
        case "analysisResult":
            guard let json = body["json"] as? String, !json.isEmpty, json.utf8.count <= Self.maximumResultLength else { return nil }
            var climber: [String: Double] = [:]
            for (key, value) in body["climber"] as? [String: Any] ?? [:] where key.count <= 32 && climber.count < 16 {
                // Numbers only; a JSON boolean is not a body measurement.
                guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(), number.doubleValue.isFinite else { continue }
                climber[key] = number.doubleValue
            }
            self = .result(id: String((body["id"] as? String ?? "").prefix(80)), json: json, climber: climber)
        case "analysisSeek":
            guard let time = (body["t"] as? NSNumber)?.doubleValue, time.isFinite else { return nil }
            self = .seek(max(0, time))
        case "analysisError": self = .failed(code: String((body["code"] as? String ?? "").prefix(80)))
        case "error": self = .pageError(String((body["message"] as? String ?? "").prefix(300)))
        default: return nil   // Progress and anything newer: the page shows it itself.
        }
    }
}

/// App → analysis page calls on `window.BetaBuddyAnalysis`.
enum AnalysisCommand: Sendable {
    case begin(id: String, width: Int, height: Int, climber: [String: Double]?)
    case addFrame(index: Int, time: Double, jpegBase64: String)
    case addPoses([PoseFrame])
    case run
    case show(String)

    static let poseBatch = 40

    private struct Meta: Encodable {
        var id: String
        var w: Int
        var h: Int
        var climber: [String: Double]?
    }
    private struct Pose: Encodable {
        var t: Double
        var img: [[Double]]
        var lm: [[Double]]
    }

    /// Every argument becomes a JavaScript literal through JSON encoding, never by joining quotes.
    private static func literal<T: Encodable>(_ value: T) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .withoutEscapingSlashes   // Base64 is full of them.
        return (try? encoder.encode(value)).flatMap { String(data: $0, encoding: .utf8) } ?? "null"
    }

    var script: String {
        let (function, arguments): (String, [String]) = switch self {
        case .begin(let id, let width, let height, let climber):
            ("begin", [Self.literal(Meta(id: id, w: width, h: height, climber: climber?.isEmpty == false ? climber : nil))])
        case .addFrame(let index, let time, let jpegBase64):
            ("addFrame", [Self.literal(index), Self.literal(rounded4(time)), Self.literal(jpegBase64)])
        case .addPoses(let frames):
            // Only samples that have world landmarks. Their fourth value is the landmarker's own
            // visibility, which the page expects for the image points as well.
            ("addPoses", [Self.literal(frames.compactMap { frame -> Pose? in
                guard let world = frame.world, world.count == 33, frame.joints.count == 33, world.allSatisfy({ $0.count == 4 }) else { return nil }
                return Pose(t: rounded4(frame.time), img: zip(frame.joints, world).map { $0.x.isFinite && $0.y.isFinite ? [$0.x, $0.y, $1[3]] : [0, 0, 0] }, lm: world)
            })])
        case .run: ("run", [])
        case .show(let json): ("show", [Self.literal(json)])
        }
        return "window.BetaBuddyAnalysis && typeof window.BetaBuddyAnalysis.\(function) === 'function' ? (window.BetaBuddyAnalysis.\(function)(\(arguments.joined(separator: ","))), true) : false"
    }
}
