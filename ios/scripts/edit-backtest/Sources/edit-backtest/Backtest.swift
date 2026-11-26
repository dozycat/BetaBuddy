import AVFoundation
import CoreGraphics
import CoreImage
import CoreText
import Foundation
import ImageIO
import UniformTypeIdentifiers

// A backtest of the app's edit renderer on a Mac, where the system's people finder (Vision) runs.
// It renders a clip from a plan exactly as the app would (the renderer and the plan reader are the
// app's own files) and writes down what it found and decided, for a person to look at.
//
//     swift run -c release --package-path ios/scripts/edit-backtest edit-backtest <video> <plan.json> <outdir>
//
// Writes full.mp4 (all three switches on), plain.mp4 (all off), removal-only.mp4, report.txt and
// sheet.jpg (the clip above, removal only below, at the same moments). Nothing here decides
// anything about the edit: the plan comes from the pack, the rendering from the app.

@main
struct Backtest {
    static func fail(_ message: String) -> Never {
        FileHandle.standardError.write(Data((message + "\n").utf8))
        exit(2)
    }

    static func main() async {
        let arguments = CommandLine.arguments
        guard arguments.count == 4 else { fail("usage: edit-backtest <video> <plan.json> <outdir>") }
        let video = URL(fileURLWithPath: arguments[1]), planFile = URL(fileURLWithPath: arguments[2]), out = URL(fileURLWithPath: arguments[3], isDirectory: true)
        guard FileManager.default.fileExists(atPath: video.path) else { fail("no such video: \(video.path)") }
        guard let text = try? String(contentsOf: planFile, encoding: .utf8) else { fail("cannot read the plan: \(planFile.path)") }
        guard let plan = EditPlan(json: text) else { fail("not a plan this renderer accepts (v 1 with valid segments): \(planFile.path)") }
        do { try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true) } catch { fail("cannot make \(out.path): \(error.localizedDescription)") }
        let work = out.appendingPathComponent("work", isDirectory: true)
        try? FileManager.default.removeItem(at: work)

        var report: [String] = []
        func note(_ line: String) { report.append(line); print(line) }
        let asset = AVURLAsset(url: video)
        let duration = (try? await asset.load(.duration).seconds) ?? 0
        note("clip: \(video.lastPathComponent), \(format(duration)) s")
        note("plan: \(planFile.lastPathComponent), duration \(format(plan.duration)), trim \(format(plan.trim.lowerBound))–\(format(plan.trim.upperBound)), \(plan.segments.count) segments (speeds \(plan.segments.map { format($0.speed) }.joined(separator: " ")))")
        note("      camera keys \(plan.camera.count) (+\(plan.segments.filter { $0.camera != nil }.count) segments with their own), keep samples \(plan.keep?.samples.count ?? 0), plate times \(plan.plateTimes.count)")
        note("      offers: removal \(plan.offersRemoval), slow motion \(plan.offersSlowMotion), camera \(plan.offersCamera)")

        let finder = VisionPeopleFinder()
        let renderer = EditRenderer(source: video, plan: plan, directory: work, finder: finder)
        let renders: [(name: String, options: EditOptions)] = [
            ("removal-only", EditOptions(removePassersBy: true, slowMotion: false, cameraMoves: false)),
            ("full", EditOptions()),
            ("plain", EditOptions(removePassersBy: false, slowMotion: false, cameraMoves: false))
        ]
        note("")
        for render in renders {
            let began = Date()
            do {
                let url = try await renderer.render(options: render.options) { _ in }
                let target = out.appendingPathComponent(render.name + ".mp4")
                try? FileManager.default.removeItem(at: target)
                try FileManager.default.moveItem(at: url, to: target)
                let seconds = (try? await AVURLAsset(url: target).load(.duration).seconds) ?? 0
                let expected = EditTimeline(plan: plan, options: render.options, sourceDuration: duration).duration
                note("\(render.name).mp4: \(format(seconds)) s (timeline \(format(expected)) s), rendered in \(format(Date().timeIntervalSince(began))) s")
            } catch {
                note("\(render.name).mp4: FAILED — \(error.localizedDescription)")
            }
        }
        try? FileManager.default.removeItem(at: work)

        note("")
        if let survey = renderer.survey {
            let valid = survey.small.valid
            let known = valid.isEmpty ? 0 : Double(valid.filter { $0 >= 250 }.count) / Double(valid.count)
            note("people finder: Vision, \(finder.instancesUnavailable ? "rectangles only (per-person outlines were not available)" : "per-person outlines"); looked at \(survey.looked) frames, looking failed in \(survey.unseen)")
            note("plate coverage: \(format(known * 100)) % of the picture has an empty wall to put in someone's place")
            note("people tracks: \(survey.tracks.tracks.count), taken out: \(survey.removed)")
            for (index, track) in survey.tracks.tracks.enumerated() {
                guard let first = track.samples.first, let last = track.samples.last else { continue }
                let middle = track.samples[track.samples.count / 2].box
                let verdict = track.removable ? "REMOVED" : "kept — \(track.reason?.rawValue ?? "no reason recorded")"
                note("  #\(index + 1): \(format(first.time))–\(format(last.time)) s, \(track.samples.count) sightings, around x \(format(middle.x + middle.w / 2)) y \(format(middle.y + middle.h / 2)) (size \(format(middle.w))×\(format(middle.h))): \(verdict)")
            }
            if survey.tracks.tracks.isEmpty { note("  (nobody was found in any frame)") }
        } else {
            note("passer-by removal: not offered by this plan (no keep boxes or no plate times); removal-only.mp4 is the plain attempt")
        }

        // The sheet: the clip above, removal only below, at the same moments of the clip.
        let source = out.appendingPathComponent("removal-only.mp4")
        if FileManager.default.fileExists(atPath: source.path) {
            let times = instants(plan: plan, survey: renderer.survey, clip: duration)
            if let sheet = await sheet(video: video, removal: source, plan: plan, survey: renderer.survey, times: times) {
                let target = out.appendingPathComponent("sheet.jpg")
                if let destination = CGImageDestinationCreateWithURL(target as CFURL, UTType.jpeg.identifier as CFString, 1, nil) {
                    CGImageDestinationAddImage(destination, sheet, [kCGImageDestinationLossyCompressionQuality: 0.85] as CFDictionary)
                    CGImageDestinationFinalize(destination)
                    note("")
                    note("sheet.jpg: top row the clip, bottom row removal only, at \(times.map { format($0) }.joined(separator: " ")) s of the clip")
                    note("           on the top row: green = the plan's keep box, red = a person taken out, yellow = a person found and kept")
                }
            }
        }
        try? (report.joined(separator: "\n") + "\n").write(to: out.appendingPathComponent("report.txt"), atomically: true, encoding: .utf8)
    }

    static func format(_ value: Double) -> String { String(format: "%.2f", value) }

    /// About eight moments of the attempt: where people are taken out first, the rest spread evenly.
    static func instants(plan: EditPlan, survey: EditRemoval?, clip: Double) -> [Double] {
        let low = plan.trim.lowerBound, high = min(clip, plan.trim.upperBound)
        var times: [Double] = []
        for track in survey?.tracks.tracks ?? [] where track.removable {
            guard let first = track.samples.first?.time, let last = track.samples.last?.time else { continue }
            times += [0.25, 0.5, 0.75].map { first + (last - first) * $0 }
        }
        times = Array(times.filter { $0 >= low && $0 <= high }.prefix(5))
        let wanted = 8
        for index in 0..<wanted where times.count < wanted {
            let time = low + (high - low) * (Double(index) + 0.5) / Double(wanted)
            if !times.contains(where: { abs($0 - time) < 0.15 }) { times.append(time) }
        }
        return times.sorted()
    }

    static func frame(_ url: URL, at time: Double, width: Int) async -> CGImage? {
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero; generator.requestedTimeToleranceAfter = .zero
        generator.maximumSize = CGSize(width: width * 2, height: width * 4)
        return try? await generator.image(at: CMTime(seconds: max(0, time), preferredTimescale: 600)).image
    }

    static func sheet(video: URL, removal: URL, plan: EditPlan, survey: EditRemoval?, times: [Double]) async -> CGImage? {
        let cell = 300
        var pairs: [(time: Double, top: CGImage, bottom: CGImage)] = []
        for time in times {
            // Removal only plays the attempt at its own speed: a moment of the clip is that much after the trim's start.
            guard let top = await frame(video, at: time, width: cell), let bottom = await frame(removal, at: time - plan.trim.lowerBound, width: cell) else { continue }
            pairs.append((time, top, bottom))
        }
        guard let first = pairs.first else { return nil }
        let height = Int((Double(cell) * Double(first.top.height) / Double(first.top.width)).rounded()), label = 26
        let width = cell * pairs.count, total = 2 * height + label
        guard let context = CGContext(data: nil, width: width, height: total, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        context.setFillColor(CGColor(gray: 0.1, alpha: 1)); context.fill(CGRect(x: 0, y: 0, width: width, height: total))
        for (column, pair) in pairs.enumerated() {
            let x = column * cell
            // Core Graphics counts y from the bottom: the clip's row is the upper one.
            let top = CGRect(x: x, y: height, width: cell, height: height), bottom = CGRect(x: x, y: 0, width: cell, height: height)
            context.draw(pair.top, in: top); context.draw(pair.bottom, in: bottom)
            func outline(_ box: EditPlan.Box, _ colour: CGColor) {
                context.setStrokeColor(colour); context.setLineWidth(2)
                context.stroke(CGRect(x: top.minX + box.x * top.width, y: top.minY + (1 - box.y - box.h) * top.height, width: box.w * top.width, height: box.h * top.height))
            }
            for box in plan.keep?.boxes(at: pair.time) ?? [] { outline(box, CGColor(red: 0.2, green: 0.9, blue: 0.3, alpha: 1)) }
            if let survey {
                for index in survey.tracks.tracks.indices {
                    guard let shape = survey.tracks.shape(of: index, at: pair.time, step: survey.step) else { continue }
                    outline(shape.box, survey.tracks.tracks[index].removable ? CGColor(red: 1, green: 0.2, blue: 0.2, alpha: 1) : CGColor(red: 1, green: 0.85, blue: 0.1, alpha: 1))
                }
            }
            let text = NSAttributedString(string: String(format: "%.2f s", pair.time), attributes: [
                kCTFontAttributeName as NSAttributedString.Key: CTFontCreateWithName("Menlo" as CFString, 15, nil),
                kCTForegroundColorAttributeName as NSAttributedString.Key: CGColor(gray: 1, alpha: 1)])
            context.textPosition = CGPoint(x: x + 8, y: 2 * height + 7)
            CTLineDraw(CTLineCreateWithAttributedString(text), context)
        }
        return context.makeImage()
    }
}
