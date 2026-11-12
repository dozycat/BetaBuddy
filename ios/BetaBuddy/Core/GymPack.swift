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
    case routeSelected(GymRoute?)
    case error(String)

    init?(body: Any) {
        guard let body = body as? [String: Any], let type = body["type"] as? String else { return nil }
        switch type {
        case "ready": self = .ready
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
