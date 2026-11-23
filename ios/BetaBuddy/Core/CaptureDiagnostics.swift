#if DEBUG
import Foundation
import UIKit

/// Debug builds only: what the capture screen saw and did, left on disk so that a first run on a
/// phone can be read afterwards without anyone watching the screen. Everything goes into
/// `Documents/capture-diagnostics/`; nothing is sent anywhere, and none of this exists in a Release build.
///
/// - `session-<yyyyMMdd-HHmmss>.jsonl`: one JSON object per line. `kind` is `sample` (once a second
///   while the screen is open) or `event`; `t` is an ISO 8601 time with milliseconds.
/// - `frame-latest.jpg`: the upright JPEG last handed to the capture page; `frame-at-lock.jpg`: the one at Record.
/// - `preview-latest.png`: the capture screen as it is drawn.
/// - `log.txt`: the lines that also go to the system log.
final class CaptureDiagnostics: @unchecked Sendable {
    static let folder = "capture-diagnostics"
    /// The folder is kept under this; the oldest sessions go first.
    static let defaultCap = 30 * 1024 * 1024
    static let logCap = 2 * 1024 * 1024

    static var defaultDirectory: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent(folder, isDirectory: true)
    }
    /// The session whose file later events (pose recognition, analysis) are added to.
    private(set) static var current: CaptureDiagnostics?
    private static let shared = DispatchQueue(label: "com.betabuddy.capture.diagnostics", qos: .utility)

    let directory: URL
    let file: URL
    private let queue = CaptureDiagnostics.shared
    private var handle: FileHandle?

    private static let stamp: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    init(directory: URL = CaptureDiagnostics.defaultDirectory, now: Date = Date(), cap: Int = CaptureDiagnostics.defaultCap) {
        self.directory = directory
        let name = DateFormatter()
        name.locale = Locale(identifier: "en_US_POSIX")
        name.dateFormat = "yyyyMMdd-HHmmss"
        var file = directory.appendingPathComponent("session-\(name.string(from: now)).jsonl")
        var suffix = 1
        while FileManager.default.fileExists(atPath: file.path) {      // Two sessions within a second.
            suffix += 1
            file = directory.appendingPathComponent("session-\(name.string(from: now))-\(suffix).jsonl")
        }
        self.file = file
        queue.sync {
            try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            Self.prune(directory, cap: cap, keeping: file)
            FileManager.default.createFile(atPath: file.path, contents: nil)
            handle = try? FileHandle(forWritingTo: file)
        }
        Self.current = self
    }

    deinit { try? handle?.close() }

    /// Drops the oldest session files until the folder is under `cap`. The images and the log are bounded by themselves.
    static func prune(_ directory: URL, cap: Int, keeping: URL? = nil) {
        let manager = FileManager.default
        func size(_ url: URL) -> Int { (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0 }
        let all = ((try? manager.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.fileSizeKey])) ?? [])
        var total = all.reduce(0) { $0 + size($1) }
        // Names carry the time, so name order is age order.
        for old in all.filter({ $0.lastPathComponent.hasPrefix("session-") && $0 != keeping }).sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) where total > cap {
            total -= size(old)
            try? manager.removeItem(at: old)
        }
    }

    /// Makes any value something `JSONSerialization` takes: numbers that are not finite become null.
    static func clean(_ value: Any) -> Any {
        // An optional that found its way into a dictionary of `Any`.
        let mirror = Mirror(reflecting: value)
        if mirror.displayStyle == .optional { return mirror.children.first.map { clean($0.value) } ?? NSNull() }
        switch value {
        case let flag as Bool: return flag
        case let number as Double: return number.isFinite ? (number * 10_000).rounded() / 10_000 : NSNull()
        case let number as CGFloat: return clean(Double(number))
        case let number as Float: return clean(Double(number))
        case let list as [Any]: return list.map(clean)
        case let object as [String: Any]: return object.mapValues(clean)
        default: return value
        }
    }

    static func line(kind: String, _ fields: [String: Any], at time: Date = Date()) -> Data? {
        var object = clean(fields) as? [String: Any] ?? [:]
        object["t"] = stamp.string(from: time)
        object["kind"] = kind
        guard JSONSerialization.isValidJSONObject(object),
              var data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes]) else { return nil }
        data.append(0x0A)
        return data
    }

    private func append(_ data: Data?) {
        guard let data else { return }
        queue.async { [self] in
            guard let handle else { return }
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: data)
        }
    }

    /// Once a second: the state of the screen.
    func sample(_ fields: [String: Any]) { append(Self.line(kind: "sample", fields)) }

    /// Something that happened. `event` is its name.
    func event(_ name: String, _ fields: [String: Any] = [:]) {
        var fields = fields
        fields["event"] = name
        append(Self.line(kind: "event", fields))
    }

    /// For what happens after the capture screen has gone: added to the latest session, or to a new one.
    static func event(_ name: String, _ fields: [String: Any] = [:]) {
        (current ?? CaptureDiagnostics()).event(name, fields)
    }

    func write(_ data: Data, as name: String) {
        queue.async { [directory] in try? data.write(to: directory.appendingPathComponent(name), options: .atomic) }
    }
    /// Encoded off the main thread.
    func write(png image: UIImage, as name: String) {
        queue.async { [directory] in try? image.pngData()?.write(to: directory.appendingPathComponent(name), options: .atomic) }
    }

    /// Waits until everything asked for so far is on disk.
    func flush() { queue.sync { try? handle?.synchronize() } }

    /// One line into `log.txt`, next to the system log's.
    static func log(_ category: String, _ text: String, directory: URL = defaultDirectory, cap: Int = logCap, at time: Date = Date()) {
        let line = Data("\(stamp.string(from: time)) [\(category)] \(text)\n".utf8)
        shared.async {
            let file = directory.appendingPathComponent("log.txt")
            try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            // Over the cap the older half goes.
            if let size = try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize, size > cap, let old = try? Data(contentsOf: file) {
                let kept = old.suffix(cap / 2)
                let start = kept.firstIndex(of: 0x0A).map { kept.index(after: $0) } ?? kept.startIndex
                try? Data(kept[start...]).write(to: file, options: .atomic)
            }
            if let handle = try? FileHandle(forWritingTo: file) {
                _ = try? handle.seekToEnd()
                try? handle.write(contentsOf: line)
                try? handle.close()
            } else {
                try? line.write(to: file, options: .atomic)
            }
        }
    }
    static func flushLog() { shared.sync {} }

    /// What this phone and build are.
    static var environment: [String: Any] {
        var system = utsname()
        uname(&system)
        let machine = withUnsafeBytes(of: &system.machine) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
        let info = Bundle.main.infoDictionary ?? [:]
        return ["model": machine, "system": UIDevice.current.systemVersion,
                "app": "\(info["CFBundleShortVersionString"] as? String ?? "?") (\(info["CFBundleVersion"] as? String ?? "?"))",
                "bundle": Bundle.main.bundleIdentifier ?? "?"]
    }

    static func thermal(_ state: ProcessInfo.ThermalState = ProcessInfo.processInfo.thermalState) -> String {
        switch state {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }
}

/// A picture of the capture screen as it is drawn. A camera preview layer does not come out in a
/// snapshot of the view hierarchy (it is left black); where that happens, the picture last handed to
/// the capture page is put underneath, in the same rectangle the preview fills, and everything the
/// screen draws over the preview stays on top of it.
enum CaptureSnapshot {
    /// `hierarchy`: the preview came out in the snapshot itself. `composite`: it did not, and the last
    /// handed-over picture stands in for it. `hierarchy-no-picture`: it did not, and there was nothing to put there.
    enum Source: String { case hierarchy, composite, missing = "hierarchy-no-picture" }
    /// Pixels darker than this in every channel count as the black a preview layer leaves.
    static let black: UInt8 = 16

    /// The share of `rect` (pixels, origin top left) that is black, sampled on a grid.
    static func darkShare(of image: CGImage, in rect: CGRect) -> Double {
        let columns = 24, rows = 40
        guard let part = image.cropping(to: rect.integral),
              let context = CGContext(data: nil, width: columns, height: rows, bitsPerComponent: 8, bytesPerRow: columns * 4,
                                      space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return 0 }
        context.interpolationQuality = .none       // Samples, not averages: a thin outline must not lift a black cell.
        context.draw(part, in: CGRect(x: 0, y: 0, width: columns, height: rows))
        guard let data = context.data else { return 0 }
        let bytes = data.bindMemory(to: UInt8.self, capacity: columns * rows * 4)
        var dark = 0
        for index in 0..<(columns * rows) where bytes[index * 4] < black && bytes[index * 4 + 1] < black && bytes[index * 4 + 2] < black { dark += 1 }
        return Double(dark) / Double(columns * rows)
    }

    /// `frame` drawn into `picture` (pixels, origin top left), with what `hierarchy` has there on top of it, its black taken out.
    static func composite(hierarchy: CGImage, picture: CGRect, frame: CGImage, cornerRadius: CGFloat = 0) -> UIImage? {
        let size = CGSize(width: hierarchy.width, height: hierarchy.height)
        // The key needs an image without an alpha channel.
        guard let plain = CGContext(data: nil, width: hierarchy.width, height: hierarchy.height, bitsPerComponent: 8, bytesPerRow: hierarchy.width * 4,
                                    space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        plain.draw(hierarchy, in: CGRect(origin: .zero, size: size))
        let limit = CGFloat(black)
        guard let keyed = plain.makeImage()?.copy(maskingColorComponents: [0, limit, 0, limit, 0, limit]) else { return nil }
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1; format.opaque = true
        return UIGraphicsImageRenderer(size: size, format: format).image { context in
            let whole = CGRect(origin: .zero, size: size)
            UIImage(cgImage: hierarchy).draw(in: whole)
            UIBezierPath(roundedRect: picture, cornerRadius: cornerRadius).addClip()
            context.cgContext.interpolationQuality = .high
            UIImage(cgImage: frame).draw(in: picture)
            UIImage(cgImage: keyed).draw(in: whole)
        }
    }

    /// The window as drawn, at two pixels a point. `picture` is where the preview is, in the window's
    /// coordinates; `nil` when something covers it. `frame` is the JPEG last handed to the page.
    @MainActor
    static func take(window: UIWindow, picture: CGRect?, frame: Data?, cornerRadius: CGFloat = 18) -> (image: UIImage, source: Source, dark: Double?)? {
        let scale: CGFloat = 2
        let format = UIGraphicsImageRendererFormat()
        format.scale = scale; format.opaque = true
        let shot = UIGraphicsImageRenderer(bounds: window.bounds, format: format).image { _ in
            window.drawHierarchy(in: window.bounds, afterScreenUpdates: false)
        }
        guard let drawn = shot.cgImage else { return nil }
        guard let picture, picture.width > 1, picture.height > 1 else { return (shot, .hierarchy, nil) }
        let pixels = CGRect(x: picture.minX * scale, y: picture.minY * scale, width: picture.width * scale, height: picture.height * scale)
        let dark = darkShare(of: drawn, in: pixels)
        guard dark >= 0.5 else { return (shot, .hierarchy, dark) }
        guard let frame, let still = UIImage(data: frame)?.cgImage,
              let combined = composite(hierarchy: drawn, picture: pixels, frame: still, cornerRadius: cornerRadius * scale) else { return (shot, .missing, dark) }
        return (combined, .composite, dark)
    }
}
#endif
