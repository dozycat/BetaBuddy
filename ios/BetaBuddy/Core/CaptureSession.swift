import AVFoundation
import SwiftUI
import WebKit
import os

/// One visit to the capture screen: frames from the source go to the pack's capture page a few
/// times a second and to the recorder while recording; the page's replies are published for the
/// screen to draw. When recording stops the clip is written into the album first, and only then
/// is anything analysed. Without a capture page the camera records and saves all the same.
@MainActor
final class CaptureSession: NSObject, ObservableObject, WKNavigationDelegate, WKScriptMessageHandler {
    /// The capture page. `.unavailable`: this build's pack has none (or there is no pack).
    enum Page: Equatable { case unavailable, loading, ready, failed }
    enum Camera: Equatable { case starting, running, denied, unavailable }
    enum Phase: Equatable {
        case framing
        case recording
        case saving
        /// The record is in the album.
        case saved(UUID)
        /// The clip was recorded but could not be written into the album; it is still held.
        case saveFailed(String)
    }
    /// Something the user should be told once, as a type so the screen picks the words.
    enum Notice: Equatable { case tooShort, lowStorage, recorderFailed }

    @Published private(set) var page: Page
    @Published private(set) var camera = Camera.starting
    @Published private(set) var phase = Phase.framing
    @Published private(set) var assessment: CaptureAssessment?
    /// The colour the user wants to climb, as one of the page's palette keys.
    @Published private(set) var color: String?
    @Published private(set) var recordingStarted: Date?
    @Published private(set) var stopReason: CaptureStop?
    @Published private(set) var summary: CaptureSummary?
    /// Pixel size of the frames handed to the page; the picture on screen has the same shape.
    @Published private(set) var frameSize = CGSize(width: 9, height: 16)
    @Published private(set) var webView: WKWebView?
    @Published var notice: Notice?
    /// The last tap on the picture found no hold there. Cleared by itself after a moment.
    @Published private(set) var tapMissed = false
    /// Problems the page recognised and the user turned down in this take.
    @Published private(set) var rejectedRoutes: Set<String> = []
    /// The grade shown was taken from the recognised problem, not picked by the user.
    @Published private(set) var gradePrefilled = false
    private var gradePicked = false
    @Published var gymId: String? { didSet { if gymId != oldValue { venueChanged() } } }
    @Published var gymName = "" { didSet { if gymName != oldValue { venueChanged() } } }
    /// Set with `pickGrade`. Until the user has picked one, a recognised problem's grade is filled in.
    @Published private(set) var grade = ungradedValue

    let source: CaptureSource
    private let store: ClimbStore
    private let pack: GymPack?
    /// The pack can analyse the clip afterwards, so saving asks for that too.
    private let analysisAvailable: Bool
    private let pipeline = CapturePipeline()
    private var started = false
    private var captureID = UUID().uuidString
    private var begun: (id: String, width: Int, height: Int, venue: String?)?
    private var pageVersion: String?
    private var plan: String?
    private var recorder: ClipRecorder?
    private var heldClip: (url: URL, started: Date)?
    private var watch: Task<Void, Never>?
    private var background: NSObjectProtocol?
    private(set) var metrics = CaptureMetrics()
    private var firstAssessment: Date?
    /// When the page last answered (or was last asked to start over). A page that has gone quiet
    /// must not leave an old judgement on the screen.
    private var lastReply: Date?
    private var lastHandOver: Date?
    var quietLimit: TimeInterval = 5
    /// How long the page may take to come up before the screen stops waiting for it.
    var loadingLimit: Duration = .seconds(8)
    /// Without a frame for this long the camera is taken to have stopped: recording is not offered and the source is started again.
    var stallLimit: TimeInterval = 2
    private var stallWatch: Task<Void, Never>?
    /// Seconds between frames handed to the page while framing. Nothing is handed over once recording has started.
    var framingInterval = 0.5
    var maximumDuration = CaptureLimits.maximumDuration
    /// How often a running recording is checked for free space, heat and a failed writer.
    var watchInterval: Duration = .seconds(3)
    /// Free space in bytes; replaced in tests.
    var freeBytes: () -> Int64? = CapturePipeline.freeBytes
    // Local system log only.
    private let log = Logger(subsystem: "com.betabuddy.ios", category: "capture")
    /// The local system log; in Debug builds also a file that can be pulled off a phone.
    private func note(_ text: String, error: Bool = false) {
        if error { log.error("\(text, privacy: .public)") } else { log.info("\(text, privacy: .public)") }
        #if DEBUG
        CaptureDiagnostics.log("capture", text)
        #endif
    }
    #if DEBUG
    /// What this visit to the screen leaves on disk, see `CaptureDiagnostics`.
    private(set) var diagnostics: CaptureDiagnostics?
    private var sampler: Task<Void, Never>?
    private var lastFrameJPEG: String?
    private var lastFrameWritten = Date.distantPast
    private var previewSource: String?
    /// Filled in by the screen: what it knows that the session does not (location access).
    var diagnosticsExtra: () -> [String: Any] = { [:] }

    func mark(_ name: String, _ fields: [String: Any] = [:]) { diagnostics?.event(name, fields) }
    /// The picture last handed to the page, and how long ago: what a composite preview is made from.
    var diagnosticsFrame: (jpeg: Data, age: Double)? {
        guard let lastFrameJPEG, let data = Data(base64Encoded: lastFrameJPEG) else { return nil }
        return (data, Date().timeIntervalSince(lastFrameHanded))
    }
    private var lastFrameHanded = Date.distantPast
    private var previewFields: [String: Any] = [:]

    /// The capture screen as drawn, from the screen itself; `source` says how the picture inside it was obtained.
    func diagnosticsPreview(_ image: UIImage, source: String, fields: [String: Any]) {
        diagnostics?.write(png: image, as: "preview-latest.png")
        previewFields = fields.merging(["source": source, "pixels": [Int(image.size.width * image.scale), Int(image.size.height * image.scale)]]) { _, new in new }
        if previewSource != source { previewSource = source; mark("snapshot", previewFields) }
    }

    /// One line a second: everything the overlay shows, and the rest of the screen's state.
    private func sampleFields() -> [String: Any] {
        let now = Date()
        let replies = recentReplies.filter { now.timeIntervalSince($0) <= 4 }
        var fields: [String: Any] = [
            "phase": "\(phase)".components(separatedBy: "(").first ?? "", "camera": "\(camera)", "page": "\(page)", "pack": pageVersion as Any,
            "fovH": source.horizontalFieldOfView as Any,
            "frame": [Int(frameSize.width), Int(frameSize.height)],
            "source": pipeline.frameSize.map { [$0.width, $0.height] } as Any,
            "pitch": lastInfo.pitch as Any, "roll": lastInfo.roll as Any, "shake": lastInfo.shake as Any,
            "repliesPerSecond": Double(replies.count) / 4, "replies": metrics.assessments,
            "handOverMs": lastHandOverMs, "prepareMs": lastPrepareMs, "poseMs": lastPoseMs as Any,
            "posesFound": metrics.posesFound, "poseFrames": metrics.poseFrames, "poseDropped": metrics.poseDropped,
            "jpegKB": metrics.frames > 0 ? Double(metrics.jpegBytes) / Double(metrics.frames) / 1024 : 0,
            "framesHanded": metrics.frames,
            "gym": gymId as Any, "venue": venue as Any, "grade": grade, "gradePrefilled": gradePrefilled,
            "color": color as Any, "thermal": CaptureDiagnostics.thermal(),
            "preview": previewFields, "debugOverlay": showsDebug, "notice": notice.map { "\($0)" } as Any,
        ]
        if let assessment {
            let tapped = Set(assessment.holds.filter(\.user).map(\.id))
            fields["ok"] = assessment.ok; fields["steady"] = assessment.steady; fields["score"] = assessment.score as Any; fields["ready"] = assessment.ready
            fields["person"] = assessment.person
            fields["issues"] = assessment.issues.map { ["code": $0.code, "severity": $0.blocking ? "block" : "warn", "move": $0.move as Any] }
            fields["hint"] = assessment.hint.map { ["code": $0.code, "move": $0.move as Any] } as Any
            fields["palette"] = assessment.palette.map { ["key": $0.key, "rgb": $0.rgb, "n": $0.count] }
            fields["holds"] = assessment.holds.count
            fields["holdsWithOutline"] = assessment.holds.filter { !$0.poly.isEmpty }.count
            fields["start"] = ["hands": assessment.startHands, "feet": assessment.startFeet, "conf": assessment.startConfidence as Any,
                               "tapped": assessment.startConfirmed]
            fields["finish"] = assessment.finish.map { ["id": $0, "conf": assessment.finishConfidence as Any, "tapped": tapped.contains($0)] } as Any
            fields["route"] = assessment.route.map { ["id": $0.id, "conf": $0.confidence as Any] } as Any
        }
        if phase == .recording, let recorder, let began = recordingStarted {
            let counts = recorder.counts
            fields["recording"] = ["seconds": now.timeIntervalSince(began), "frames": counts.frames, "dropped": counts.dropped]
        }
        return fields.merging(diagnosticsExtra()) { own, _ in own }
    }

    /// Numbers over the picture for a first run on a phone: `-debugCaptureOverlay YES`, or a long press on the guidance line.
    @Published var showsDebug = UserDefaults.standard.bool(forKey: "debugCaptureOverlay")
    private var lastInfo = CaptureFrameInfo()
    private var lastHandOverMs = 0.0, lastPrepareMs = 0.0, lastPoseMs: Double?
    private var recentReplies: [Date] = []
    /// `-debugFailSaves <n>` makes the next n attempts to write a clip into the album fail, so that the screen for that can be seen.
    private var debugSaveFailures = UserDefaults.standard.integer(forKey: "debugFailSaves")
    private var lockSent: Date?

    var debugText: String {
        let now = Date()
        let replies = recentReplies.filter { now.timeIntervalSince($0) <= 4 }
        func number(_ value: Double?, _ format: String) -> String { value.map { String(format: format, $0) } ?? "–" }
        var lines = [
            "fovH \(number(source.horizontalFieldOfView, "%.1f°"))  frame \(Int(frameSize.width))×\(Int(frameSize.height))  page \(pageVersion ?? "–") \(page)",
            "pitch \(number(lastInfo.pitch, "%+.1f°")) roll \(number(lastInfo.roll, "%+.1f°")) shake \(number(lastInfo.shake, "%.3f"))",
            String(format: "replies %.1f/s (%d)  hand-over %.1f ms  prepare %.1f ms", Double(replies.count) / 4, metrics.assessments, lastHandOverMs, lastPrepareMs),
            "pose \(number(lastPoseMs, "%.0f ms")) found \(metrics.posesFound)/\(metrics.poseFrames)\(metrics.poseDropped ? " DROPPED" : "")  jpeg \(metrics.frames > 0 ? metrics.jpegBytes / metrics.frames / 1024 : 0) KB",
        ]
        if let assessment {
            lines.append("ok \(assessment.ok) steady \(assessment.steady) score \(number(assessment.score, "%.2f")) person \(assessment.person) holds \(assessment.holds.count)")
            lines.append("issues " + (assessment.issues.isEmpty ? "none" : assessment.issues.map { $0.code + ($0.blocking ? "!" : "") + ($0.move.map { "→" + $0 } ?? "") }.joined(separator: " ")))
            if let route = assessment.route { lines.append("route \(route.id) \(number(route.confidence, "%.2f"))") }
        }
        if phase == .recording, let recorder, let began = recordingStarted {
            let counts = recorder.counts
            lines.append(String(format: "rec %.0f s  frames %d  dropped %d  plan %@", now.timeIntervalSince(began), counts.frames, counts.dropped, plan.map { "\($0.utf8.count / 1024) KB" } ?? "–"))
        }
        return lines.joined(separator: "\n")
    }
    #endif

    init(store: ClimbStore, pack: GymPack?, source: CaptureSource) {
        self.store = store
        self.pack = pack?.supportsCapture == true ? pack : nil
        self.source = source
        analysisAvailable = pack?.supportsAnalysis == true
        page = self.pack == nil ? .unavailable : .loading
        super.init()
        if let last = store.lastGym { gymId = last.id; gymName = last.name }
    }

    /// The pack in use and where pictures come from. A Debug build can be pointed at a video file
    /// with `-debugCaptureVideo <path>`, which is how tests and the simulator get past having no camera.
    nonisolated static func makeSource() -> CaptureSource {
        #if DEBUG
        if let path = UserDefaults.standard.string(forKey: "debugCaptureVideo") {
            // `-debugCaptureFieldOfView <degrees>` says how wide the lens that shot the file was, when that is known.
            let fieldOfView = UserDefaults.standard.double(forKey: "debugCaptureFieldOfView")
            return FileCaptureSource(url: URL(fileURLWithPath: path), horizontalFieldOfView: fieldOfView > 0 ? fieldOfView : nil)
        }
        #endif
        return CameraSource()
    }

    // MARK: Lifetime

    func start() {
        guard !started else { return }
        started = true
        // Pose recognition of earlier clips waits while the camera is in use.
        store.pauseAnalysisForCamera()
        pipeline.deliver = { [weak self] frame, done in
            guard let self else { done(); return }
            self.hand(frame, done)
        }
        pipeline.onFrames = { [weak self] width, height in
            Task { @MainActor [weak self] in
                guard let self, self.started else { return }
                if self.camera != .running { self.frameSize = CGSize(width: width, height: height) }
                #if DEBUG
                self.mark("frames", ["source": [width, height], "fovH": self.source.horizontalFieldOfView as Any, "details": (self.source as? CameraSource)?.diagnostics as Any])
                #endif
                self.camera = .running
                // Frames are back after a pause (a call, the background): the page has had nothing to answer meanwhile.
                if self.lastReply != nil { self.lastReply = Date() }
            }
        }
        source.onFrame = { [pipeline] buffer, time in pipeline.handle(buffer, time) }
        source.onInterruption = { [weak self] in
            Task { @MainActor in
                #if DEBUG
                self?.mark("sourceInterruption")
                #endif
                await self?.stop(.interrupted)
            }
        }
        background = NotificationCenter.default.addObserver(forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: .main) { [weak self] _ in
            // Asked for here and now, before this returns: closing the clip must not depend on a task that has yet to start.
            MainActor.assumeIsolated {
                guard let self, self.phase == .recording else { return }
                let task = UIApplication.shared.beginBackgroundTask(withName: "Close recording")
                Task { @MainActor in
                    await self.stop(.interrupted)
                    if task != .invalid { UIApplication.shared.endBackgroundTask(task) }
                }
            }
        }
        stallWatch = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(500))
                guard let self, !Task.isCancelled, self.started else { return }
                // The preview has frozen (a camera error, or the pictures simply stopped): there is nothing to record.
                guard self.camera == .running, self.phase == .framing, let age = self.pipeline.secondsSinceFrame, age > self.stallLimit else { continue }
                self.camera = .starting
                #if DEBUG
                self.mark("stall", ["secondsWithoutFrame": age])
                #endif
                self.pipeline.announceNextFrame()
                try? await self.source.start()
            }
        }
        #if DEBUG
        // `-debugCaptureAttitude "<pitch>,<roll>,<shake>"` stands in for the motion sensor the simulator lacks.
        if let text = UserDefaults.standard.string(forKey: "debugCaptureAttitude") {
            let parts = text.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
            if parts.count == 3 { pipeline.motion.fixed = CaptureFrameInfo(pitch: parts[0], roll: parts[1], shake: parts[2]) }
        }
        #endif
        #if DEBUG
        let diagnostics = CaptureDiagnostics()
        self.diagnostics = diagnostics
        var opened = CaptureDiagnostics.environment
        opened["pack"] = pack?.manifest.packVersion as Any
        opened["capturePage"] = pack != nil
        opened["source"] = "\(type(of: source))"
        opened["cameraAccess"] = AVCaptureDevice.authorizationStatus(for: .video).rawValue      // 0 not asked, 1 restricted, 2 denied, 3 allowed
        opened["gym"] = gymId as Any
        diagnostics.event("open", opened)
        sampler = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1))
                guard let self, !Task.isCancelled, self.started else { return }
                self.diagnostics?.sample(self.sampleFields())
            }
        }
        #endif
        loadPage()
        Task {
            do {
                try await source.start()
                guard started else { source.stop(); return }
                // `camera` turns to `.running` with the first frame: only then is there something to record.
                pipeline.motion.start()
                pipeline.setFeeding(page != .unavailable, interval: framingInterval)
                #if DEBUG
                mark("sourceStarted", ["cameraAccess": AVCaptureDevice.authorizationStatus(for: .video).rawValue, "details": (source as? CameraSource)?.diagnostics as Any])
                #endif
            } catch CaptureSourceError.denied {
                camera = .denied
                #if DEBUG
                mark("sourceError", ["error": "denied", "cameraAccess": AVCaptureDevice.authorizationStatus(for: .video).rawValue])
                #endif
            } catch {
                camera = .unavailable
                #if DEBUG
                mark("sourceError", ["error": "unavailable"])
                #endif
            }
        }
    }

    /// Leaves the screen. A recording still running is closed and saved first.
    func close() async {
        if phase == .recording { await stop(.user) }
        guard started else { return }
        started = false
        watch?.cancel()
        stallWatch?.cancel()
        if let background { NotificationCenter.default.removeObserver(background) }
        background = nil
        pipeline.setFeeding(false)
        pipeline.setRecorder(nil)
        pipeline.motion.stop()
        source.onFrame = nil
        source.onInterruption = nil
        source.stop()
        webView?.stopLoading()
        webView?.configuration.userContentController.removeScriptMessageHandler(forName: GymSession.handlerName)
        webView = nil
        // A clip that could not be saved is not thrown away by leaving: it stays where recordings are
        // written and is taken into the album the next time the app starts. Discarding is the user's own choice.
        heldClip = nil
        note("capture closed: \(metrics.summary)")
        #if DEBUG
        sampler?.cancel()
        mark("close", ["summary": metrics.summary])
        diagnostics?.flush()
        #endif
        store.resumeAnalysis(cameraClosed: true)
    }

    // MARK: Page

    private func loadPage() {
        guard let pack, let url = pack.captureURL(language: AppLanguage.current) else { return }
        page = .loading
        Task {
            guard let view = await GymSession.makeWebView(pack: pack, handler: WeakMessageHandler(self), delegate: self) else {
                page = .failed
                return
            }
            guard started else { return }
            webView = view
            view.load(URLRequest(url: url))
            // A page that never says it is ready is treated like one that is not there.
            try? await Task.sleep(for: loadingLimit)
            if started, page == .loading {
                #if DEBUG
                mark("loadingTimeout")
                #endif
                pageFailed("no captureReady")
            }
        }
    }

    @discardableResult
    private func call(_ command: CaptureCommand) async -> Bool {
        await answer(command) as? Bool == true
    }

    /// What the page's function gave back; `false` when the page does not have it.
    private func answer(_ command: CaptureCommand) async -> Any? {
        guard let webView else { return nil }
        return await withCheckedContinuation { continuation in
            webView.evaluateJavaScript(command.script) { value, _ in continuation.resume(returning: value) }
        }
    }

    private var venue: String? { pack?.manifest.venueID(gymID: gymId, name: gymName) }

    /// The page is told about a gym only when it is one of the pack's own; choosing another gym
    /// starts its judgement over.
    private func venueChanged() {
        guard phase == .framing, let begun, begun.venue != venue else { return }
        self.begun = nil
    }

    private func hand(_ frame: CapturePipeline.Frame, _ done: @escaping () -> Void) {
        let size = CGSize(width: frame.width, height: frame.height)
        if frameSize != size { frameSize = size }
        guard started, page == .ready || page == .failed, webView != nil else { done(); return }
        // The page can only have gone quiet while it was being asked. After any pause in what it is
        // given (a call, the background, a camera that stopped) its clock starts again.
        let now = Date()
        if let lastHandOver, now.timeIntervalSince(lastHandOver) > 1, lastReply != nil { lastReply = now }
        lastHandOver = now
        Task {
            let venue = self.venue
            if begun?.id != captureID || begun?.width != frame.width || begun?.height != frame.height {
                guard phase == .framing else { done(); return }   // A locked judgement is not started over.
                begun = (captureID, frame.width, frame.height, venue)
                assessment = nil
                rejectedRoutes = []          // The page forgets them with `begin` as well.
                lastReply = Date()
                await call(.begin(id: captureID, width: frame.width, height: frame.height, fieldOfView: source.horizontalFieldOfView, venue: venue))
                #if DEBUG
                mark("begin", ["id": captureID, "w": frame.width, "h": frame.height, "fovH": source.horizontalFieldOfView as Any, "venue": venue as Any, "gym": gymId as Any])
                #endif
                if let color { await call(.setColor(color)) }
            }
            let began = Date()
            await call(.addFrame(time: frame.time, jpegBase64: frame.jpegBase64, info: frame.info))
            metrics.frames += 1
            metrics.handOverSeconds += Date().timeIntervalSince(began)
            metrics.prepareSeconds += frame.prepareSeconds
            metrics.jpegBytes += frame.bytes
            if let seconds = frame.poseSeconds {
                metrics.poseFrames += 1
                metrics.poseSeconds += seconds
                if frame.info.pose != nil { metrics.posesFound += 1 }
            }
            metrics.poseDropped = pipeline.poseDropped
            #if DEBUG
            lastInfo = frame.info; lastHandOverMs = Date().timeIntervalSince(began) * 1000; lastPrepareMs = frame.prepareSeconds * 1000
            lastPoseMs = frame.poseSeconds.map { $0 * 1000 }
            // The very picture the page was given, every couple of seconds.
            lastFrameJPEG = frame.jpegBase64
            lastFrameHanded = Date()
            if Date().timeIntervalSince(lastFrameWritten) >= 2, let data = Data(base64Encoded: frame.jpegBase64) {
                lastFrameWritten = Date()
                diagnostics?.write(data, as: "frame-latest.jpg")
            }
            #endif
            if phase == .framing, page == .ready, let lastReply, Date().timeIntervalSince(lastReply) > quietLimit {
                note("capture page has not answered for \(Int(quietLimit)) s", error: true)
                #if DEBUG
                mark("quiet", ["seconds": quietLimit])
                #endif
                page = .failed
                assessment = nil
            }
            done()
        }
    }

    func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction) async -> WKNavigationActionPolicy {
        GymSession.navigationPolicy(pack: pack, for: action)
    }
    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) { loadFailed(error) }
    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) { loadFailed(error) }
    private func loadFailed(_ error: Error) {
        guard (error as NSError).code != NSURLErrorCancelled else { return }
        pageFailed(error.localizedDescription)
    }
    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) { pageFailed("web content process ended") }
    /// The page is gone for this visit; frames are no longer prepared for it.
    private func pageFailed(_ text: String) {
        note("capture page: \(text)", error: true)
        #if DEBUG
        mark("pageFailed", ["text": text])
        #endif
        page = .failed
        assessment = nil
        pipeline.setFeeding(false)
    }

    func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage) {
        guard message.frameInfo.isMainFrame, let parsed = CaptureMessage(body: message.body) else { return }
        switch parsed {
        case .ready(let version):
            pageVersion = version.isEmpty ? pack?.manifest.packVersion : version
            begun = nil
            page = .ready
            #if DEBUG
            mark("captureReady", ["pack": pageVersion as Any])
            #endif
            if phase == .framing { pipeline.setFeeding(started, interval: framingInterval) }   // Also when it came up late.
        case .assessment(let assessment):
            guard phase == .framing, begun?.id == captureID else { return }
            if page == .failed { page = .ready }
            lastReply = Date()
            #if DEBUG
            recentReplies = (recentReplies + [Date()]).suffix(20)
            #endif
            self.assessment = assessment
            fillInGrade()
            metrics.assessments += 1
            if let firstAssessment { metrics.assessmentSpan = Date().timeIntervalSince(firstAssessment) } else { firstAssessment = Date() }
        case .locked(let id, let plan, let summary):
            guard id.isEmpty || id == captureID else { return }
            self.summary = summary
            if let plan { self.plan = plan }
            writeDraft()
            #if DEBUG
            mark("captureLocked", ["ms": lockSent.map { Date().timeIntervalSince($0) * 1000 } as Any, "planBytes": plan?.utf8.count as Any,
                                   "summary": ["color": summary.color as Any, "rgb": summary.rgb as Any, "holds": summary.holds, "startHands": summary.startHands,
                                               "startFeet": summary.startFeet, "finish": summary.finish, "ok": summary.ok, "issues": summary.issues,
                                               "route": summary.route.map { ["id": $0.id, "conf": $0.confidence as Any] } as Any]])
            #endif
            // The page may answer, or answer again with a fuller plan, after the clip is in the album.
            if case .saved(let climb) = phase, let plan {
                do { try store.saveCapturePlan(plan, summary: summary, for: climb) }
                catch { note("late plan not kept: \(error.localizedDescription)", error: true) }
            }
        case .failed(let code, let message):
            // Some of these are passing (one frame it could not read, a venue file it could not get) and
            // the page carries on. Whether it still answers is what decides what the screen says, see `hand`.
            note("capture page reported: \(code) — \(message)")
            #if DEBUG
            mark("captureError", ["code": code, "message": message])
            #endif
        case .pageError(let text):
            note("capture page error (\(page == .loading ? "before" : "after") ready): \(text)", error: true)
            #if DEBUG
            mark("pageError", ["text": text, "beforeReady": page == .loading])
            #endif
            if page == .loading { page = .failed }
        }
    }

    // MARK: What the user chooses

    var palette: [CaptureAssessment.Swatch] { assessment?.palette ?? [] }

    /// `nil`, or the colour already chosen, clears the choice.
    func selectColor(_ key: String?) {
        guard phase == .framing else { return }
        color = key == color ? nil : key
        #if DEBUG
        mark("setColor", ["key": color as Any])
        #endif
        Task { await call(.setColor(color)) }
    }

    /// The user says what the hold at this spot of the picture is. The page finds the hold; what
    /// changed shows with its next reply. When it finds none there, the screen says so for a moment.
    func correct(at point: CGPoint, role: CaptureRole) {
        guard phase == .framing, page == .ready, color != nil else { return }
        Task {
            let found = await answer(.tap(x: point.x, y: point.y, role: role))
            #if DEBUG
            mark("tap", ["x": Double(point.x), "y": Double(point.y), "role": role.rawValue, "hold": found as? String as Any, "answered": !(found is Bool)])
            #endif
            // An empty answer is “no hold here”. A page that answers nothing at all gives the same; `false` means it has no `tap`.
            guard let text = found as? String, text.isEmpty else { tapMissed = false; return }
            tapMissed = true
            try? await Task.sleep(for: .seconds(3))
            tapMissed = false
        }
    }

    func resetCorrections() {
        guard phase == .framing, page == .ready else { return }
        #if DEBUG
        mark("resetTaps")
        #endif
        Task { await call(.resetTaps) }
    }

    /// The problem the page recognised, unless the user has said it is not that one.
    var recognised: CaptureAssessment.Route? {
        (summary?.route ?? assessment?.route).flatMap { rejectedRoutes.contains($0.id) ? nil : $0 }
    }

    /// “Not this one”. A page that has `rejectRoute` moves on to its next candidate, or to none; an
    /// older one keeps reporting the same problem, which is then simply not shown and not stored.
    func rejectRoute() {
        guard phase == .framing, let route = recognised else { return }
        rejectedRoutes.insert(route.id)
        #if DEBUG
        mark("rejectRoute", ["id": route.id])
        #endif
        fillInGrade()
        Task { await call(.rejectRoute(route.id)) }
    }

    /// The user's own choice of grade. It is never replaced afterwards.
    func pickGrade(_ value: String) {
        #if DEBUG
        mark("pickGrade", ["grade": value])
        #endif
        grade = value
        gradePicked = true
        gradePrefilled = false
    }

    /// Until the user has picked a grade, the recognised problem's is filled in, visibly; when
    /// that problem goes away again, so does its grade.
    private func fillInGrade() {
        guard !gradePicked, phase == .framing else { return }
        if let known = recognised?.albumGrade {
            if grade != known { grade = known }
            gradePrefilled = true
        } else if gradePrefilled {
            grade = ungradedValue
            gradePrefilled = false
        }
    }

    /// The hint blocks: the user is asked before recording from this position.
    var needsConfirmation: Bool { page == .ready && assessment?.ok == false }

    // MARK: Recording

    func startRecording() {
        guard phase == .framing, camera == .running, let size = pipeline.frameSize else { return }
        if let free = freeBytes(), free < CaptureLimits.startingFreeBytes { notice = .lowStorage; return }
        let began = Date()
        guard let url = try? CaptureRecovery.newClip(in: store.root).movie,
              let recorder = try? ClipRecorder(url: url, width: size.width, height: size.height,
                                               settings: source.writerSettings(width: size.width, height: size.height), date: began) else {
            notice = .recorderFailed
            return
        }
        self.recorder = recorder
        plan = nil; summary = nil; stopReason = nil; notice = nil
        recordingStarted = began
        phase = .recording
        let limit = maximumDuration
        pipeline.onRecorded = { [weak self] seconds in
            guard seconds >= limit else { return }
            Task { @MainActor [weak self] in await self?.stop(.limit) }
        }
        #if DEBUG
        (source as? FileCaptureSource)?.rewind()       // A file in place of the camera is recorded from its first frame.
        #endif
        pipeline.setRecorder(recorder)
        // The judgement of this moment is what the analysis will be given. The page uses nothing
        // that comes after `lock`, so no more frames are prepared for it and live pose stops.
        pipeline.setFeeding(false)
        // Only a judgement of this take, for the gym now chosen, is locked. Right after the gym was
        // changed the page is still on the old one: then this take is recorded without a plan.
        let locks = page == .ready && begun?.id == captureID && begun?.venue == venue
        if locks { Task { await call(.lock) } }
        writeDraft()
        #if DEBUG
        lockSent = locks ? Date() : nil
        mark("recordStart", ["source": [size.width, size.height], "lockSent": locks, "page": "\(page)", "file": url.lastPathComponent,
                             "codec": (source.writerSettings(width: size.width, height: size.height)[AVVideoCodecKey] as? AVVideoCodecType)?.rawValue as Any])
        if let lastFrameJPEG, let data = Data(base64Encoded: lastFrameJPEG) { diagnostics?.write(data, as: "frame-at-lock.jpg") }
        #endif
        watch?.cancel()
        watch = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: self?.watchInterval ?? .seconds(3))
                guard let self, !Task.isCancelled, self.phase == .recording else { return }
                if self.recorder?.failed == true { await self.stop(.interrupted) }
                else if let free = self.freeBytes(), free < CaptureLimits.stoppingFreeBytes { await self.stop(.storage) }
                else if ProcessInfo.processInfo.thermalState == .critical { await self.stop(.overheated) }
            }
        }
    }

    /// Closes the clip with what has been recorded, writes it into the album, and only then lets
    /// analysis start. Safe to call from several places at once: only the first call acts.
    func stop(_ reason: CaptureStop) async {
        guard phase == .recording, let recorder, let began = recordingStarted else { return }
        phase = .saving
        stopReason = reason
        watch?.cancel()
        self.recorder = nil
        pipeline.setRecorder(nil)
        pipeline.setFeeding(false)
        // Closing the file and writing the record must not be cut short by the app going to the background.
        let task = UIApplication.shared.beginBackgroundTask(withName: "Save recording")
        defer { if task != .invalid { UIApplication.shared.endBackgroundTask(task) } }
        let result = await recorder.finish()
        note(String(format: "recorded %.1f s, %d frames, %d dropped, stop: %@", result.duration, result.frames, result.dropped, reason.rawValue))
        #if DEBUG
        mark("recordStop", ["reason": reason.rawValue, "seconds": result.duration, "frames": result.frames, "dropped": result.dropped, "written": result.url != nil,
                            "bytes": (try? result.url?.resourceValues(forKeys: [.fileSizeKey]).fileSize) as Any])
        #endif
        guard let url = result.url, result.duration >= CaptureLimits.minimumDuration else {
            CaptureRecovery.remove(recorder.url)
            notice = result.url == nil && result.frames > 0 ? .recorderFailed : .tooShort
            startFraming()
            return
        }
        heldClip = (url, began)
        await save()
    }

    /// Writes the held clip into the album. Also the retry after a failed save.
    func save() async {
        guard let clip = heldClip else { return }
        if phase != .saving { phase = .saving }
        do {
            #if DEBUG
            if debugSaveFailures > 0 { debugSaveFailures -= 1; throw CocoaError(.fileWriteOutOfSpace) }
            #endif
            let analyzer = VideoAnalyzer()
            let playback = try await analyzer.videoSummary(url: clip.url)
            let thumbnail = try await analyzer.thumbnail(url: clip.url)
            let info = makeInfo()
            // One write: the clip, its thumbnail, the plan, what was chosen and the request for
            // analysis. Nothing is analysed until this has succeeded.
            let climb = try store.save(title: "", grade: grade, sent: false, note: "", stagedVideo: clip.url, thumbnail: thumbnail, analysis: playback,
                                       gymId: gymId, gymName: gymName, climbedAt: clip.started, pendingAnalysis: AnalysisQuality.balanced.rawValue,
                                       capture: info, capturePlan: plan)
            heldClip = nil
            try? FileManager.default.removeItem(at: CaptureRecovery.draftURL(for: clip.url))
            #if DEBUG
            mark("save", ["record": climb.id.uuidString, "seconds": playback.duration, "aspect": playback.aspectRatio, "hasPlan": climb.capture?.hasPlan as Any,
                          "analysisQueued": climb.capture?.analysisQueued as Any, "video": climb.videoName])
            #endif
            phase = .saved(climb.id)
            store.resumeAnalysis(cameraClosed: true)
        } catch {
            #if DEBUG
            mark("save", ["error": error.localizedDescription])
            #endif
            phase = .saveFailed(error.localizedDescription)
        }
    }

    /// What is known about this take besides the clip itself.
    private func makeInfo() -> CaptureInfo {
        let swatch = assessment?.palette.first { $0.key == color }
        return CaptureInfo(colorKey: color, colorRGB: swatch?.rgb ?? (summary?.color == color ? summary?.rgb : nil), summary: summary?.stored,
                           packVersion: pack == nil ? nil : pageVersion, analysisQueued: analysisAvailable ? true : nil,
                           routeId: recognised?.id, routeName: recognised?.nameZh)
    }

    /// Kept next to the clip while it is being recorded, so that it can become a record even if the app does not live to save it.
    private func writeDraft() {
        guard let movie = recorder?.url ?? heldClip?.url, let began = recordingStarted else { return }
        CaptureRecovery.write(CaptureDraft(started: began, gymId: gymId, gymName: gymName, grade: grade, info: makeInfo(), plan: plan), for: movie)
    }

    /// Gives up a clip that could not be saved.
    func discardHeldClip() {
        if let heldClip { CaptureRecovery.remove(heldClip.url) }
        heldClip = nil
        startFraming()
    }

    /// Back to framing for another attempt, with the same gym, grade and colour.
    func frameAgain() {
        guard phase != .recording, phase != .saving else { return }
        startFraming()
    }

    private func startFraming() {
        store.pauseAnalysisForCamera()
        captureID = UUID().uuidString
        begun = nil
        assessment = nil
        plan = nil; summary = nil
        recordingStarted = nil
        rejectedRoutes = []
        phase = .framing
        pipeline.setFeeding(page != .unavailable && started, interval: framingInterval)
    }
}
