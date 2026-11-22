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
    @Published var gymId: String? { didSet { if gymId != oldValue { venueChanged() } } }
    @Published var gymName = "" { didSet { if gymName != oldValue { venueChanged() } } }
    @Published var grade = ungradedValue

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
    /// Seconds between frames handed to the page while framing, and while recording.
    var framingInterval = 1.0 / 3
    var recordingInterval = 1.0
    var maximumDuration = CaptureLimits.maximumDuration
    /// How often a running recording is checked for free space, heat and a failed writer.
    var watchInterval: Duration = .seconds(3)
    /// Free space in bytes; replaced in tests.
    var freeBytes: () -> Int64? = CapturePipeline.freeBytes
    // Local system log only.
    private let log = Logger(subsystem: "com.betabuddy.ios", category: "capture")

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
        if let path = UserDefaults.standard.string(forKey: "debugCaptureVideo") { return FileCaptureSource(url: URL(fileURLWithPath: path)) }
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
        pipeline.onFirstFrame = { [weak self] width, height in
            Task { @MainActor [weak self] in
                guard let self, self.started else { return }
                self.frameSize = CGSize(width: width, height: height)
                self.camera = .running
            }
        }
        source.onFrame = { [pipeline] buffer, time in pipeline.handle(buffer, time) }
        source.onInterruption = { [weak self] in Task { @MainActor in await self?.stop(.interrupted) } }
        background = NotificationCenter.default.addObserver(forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: .main) { [weak self] _ in
            Task { @MainActor in await self?.stop(.interrupted) }
        }
        loadPage()
        Task {
            do {
                try await source.start()
                guard started else { source.stop(); return }
                // `camera` turns to `.running` with the first frame: only then is there something to record.
                pipeline.motion.start()
                pipeline.setFeeding(page != .unavailable, interval: framingInterval)
            } catch CaptureSourceError.denied { camera = .denied }
            catch { camera = .unavailable }
        }
    }

    /// Leaves the screen. A recording still running is closed and saved first.
    func close() async {
        if phase == .recording { await stop(.user) }
        guard started else { return }
        started = false
        watch?.cancel()
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
        if let heldClip { try? FileManager.default.removeItem(at: heldClip.url) }
        heldClip = nil
        log.info("capture closed: \(self.metrics.summary, privacy: .public)")
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
        }
    }

    @discardableResult
    private func call(_ command: CaptureCommand) async -> Bool {
        guard let webView else { return false }
        return await withCheckedContinuation { continuation in
            webView.evaluateJavaScript(command.script) { value, _ in continuation.resume(returning: value as? Bool == true) }
        }
    }

    private var venue: String? { pack?.manifest.venueID(forGymNamed: gymName) }

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
        Task {
            let venue = self.venue
            if begun?.id != captureID || begun?.width != frame.width || begun?.height != frame.height {
                guard phase == .framing else { done(); return }   // A locked judgement is not started over.
                begun = (captureID, frame.width, frame.height, venue)
                assessment = nil
                await call(.begin(id: captureID, width: frame.width, height: frame.height, fieldOfView: source.horizontalFieldOfView, venue: venue))
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
        log.error("capture page: \(text, privacy: .public)")
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
        case .assessment(let assessment):
            guard phase == .framing, begun?.id == captureID else { return }
            if page == .failed { page = .ready }
            self.assessment = assessment
        case .locked(let id, let plan, let summary):
            guard id.isEmpty || id == captureID else { return }
            self.summary = summary
            if let plan { self.plan = plan }
            // The page may answer, or answer again with a fuller plan, after the clip is in the album.
            if case .saved(let climb) = phase, let plan {
                do { try store.saveCapturePlan(plan, summary: summary, for: climb) }
                catch { log.error("late plan not kept: \(error.localizedDescription, privacy: .public)") }
            }
        case .failed(let code):
            // Recording goes on without the page; the screen says the checks are not available.
            log.info("capture page cannot assess: \(code, privacy: .public)")
            if phase == .framing { page = .failed; assessment = nil }
        case .pageError(let text):
            log.error("capture page error (\(self.page == .loading ? "before" : "after", privacy: .public) ready): \(text, privacy: .public)")
            if page == .loading { page = .failed }
        }
    }

    // MARK: What the user chooses

    var palette: [CaptureAssessment.Swatch] { assessment?.palette ?? [] }

    /// `nil`, or the colour already chosen, clears the choice.
    func selectColor(_ key: String?) {
        guard phase == .framing else { return }
        color = key == color ? nil : key
        Task { await call(.setColor(color)) }
    }

    /// The user says what the hold at this spot of the picture is. The page finds the hold.
    func correct(at point: CGPoint, role: CaptureRole) {
        guard phase == .framing, page == .ready, color != nil else { return }
        Task { await call(.tap(x: point.x, y: point.y, role: role)) }
    }

    func resetCorrections() {
        guard phase == .framing, page == .ready else { return }
        Task { await call(.resetTaps) }
    }

    /// The hint blocks: the user is asked before recording from this position.
    var needsConfirmation: Bool { page == .ready && assessment?.ok == false }

    // MARK: Recording

    func startRecording() {
        guard phase == .framing, camera == .running, let size = pipeline.frameSize else { return }
        if let free = freeBytes(), free < CaptureLimits.startingFreeBytes { notice = .lowStorage; return }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("betabuddy-capture-\(UUID()).mov")
        let began = Date()
        guard let recorder = try? ClipRecorder(url: url, width: size.width, height: size.height,
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
        pipeline.setRecorder(recorder)
        // The judgement of this moment is what the analysis will be given. Frames keep going to the
        // page at a slower pace; it only uses them to see the wall without the climber.
        pipeline.setFeeding(page == .ready, interval: recordingInterval)
        if page == .ready { Task { await call(.lock) } }
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
        log.info("recorded \(result.duration, format: .fixed(precision: 1)) s, \(result.frames) frames, \(result.dropped) dropped, stop: \(reason.rawValue, privacy: .public)")
        guard let url = result.url, result.duration >= CaptureLimits.minimumDuration else {
            if let url = result.url { try? FileManager.default.removeItem(at: url) }
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
            let analyzer = VideoAnalyzer()
            let playback = try await analyzer.videoSummary(url: clip.url)
            let thumbnail = try await analyzer.thumbnail(url: clip.url)
            let swatch = assessment?.palette.first { $0.key == color }
            let info = CaptureInfo(colorKey: color, colorRGB: swatch?.rgb ?? (summary?.color == color ? summary?.rgb : nil), summary: summary,
                                   packVersion: pack == nil ? nil : pageVersion, analysisQueued: analysisAvailable ? true : nil)
            // One write: the clip, its thumbnail, the plan, what was chosen and the request for
            // analysis. Nothing is analysed until this has succeeded.
            let climb = try store.save(title: "", grade: grade, sent: false, note: "", stagedVideo: clip.url, thumbnail: thumbnail, analysis: playback,
                                       gymId: gymId, gymName: gymName, climbedAt: clip.started, pendingAnalysis: AnalysisQuality.balanced.rawValue,
                                       capture: info, capturePlan: plan)
            heldClip = nil
            phase = .saved(climb.id)
            store.resumeAnalysis(cameraClosed: true)
        } catch {
            phase = .saveFailed(error.localizedDescription)
        }
    }

    /// Gives up a clip that could not be saved.
    func discardHeldClip() {
        if let heldClip { try? FileManager.default.removeItem(at: heldClip.url) }
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
        phase = .framing
        pipeline.setFeeding(page != .unavailable && started, interval: framingInterval)
    }
}
