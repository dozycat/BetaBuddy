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
                self.camera = .running
                // Frames are back after a pause (a call, the background): the page has had nothing to answer meanwhile.
                if self.lastReply != nil { self.lastReply = Date() }
            }
        }
        source.onFrame = { [pipeline] buffer, time in pipeline.handle(buffer, time) }
        source.onInterruption = { [weak self] in Task { @MainActor in await self?.stop(.interrupted) } }
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
            // A page that never says it is ready is treated like one that is not there.
            try? await Task.sleep(for: loadingLimit)
            if started, page == .loading { pageFailed("no captureReady") }
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
            if phase == .framing, page == .ready, let lastReply, Date().timeIntervalSince(lastReply) > quietLimit {
                log.error("capture page has not answered for \(self.quietLimit, format: .fixed(precision: 0)) s")
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
            if phase == .framing { pipeline.setFeeding(started, interval: framingInterval) }   // Also when it came up late.
        case .assessment(let assessment):
            guard phase == .framing, begun?.id == captureID else { return }
            if page == .failed { page = .ready }
            lastReply = Date()
            self.assessment = assessment
            fillInGrade()
            metrics.assessments += 1
            if let firstAssessment { metrics.assessmentSpan = Date().timeIntervalSince(firstAssessment) } else { firstAssessment = Date() }
        case .locked(let id, let plan, let summary):
            guard id.isEmpty || id == captureID else { return }
            self.summary = summary
            if let plan { self.plan = plan }
            writeDraft()
            // The page may answer, or answer again with a fuller plan, after the clip is in the album.
            if case .saved(let climb) = phase, let plan {
                do { try store.saveCapturePlan(plan, summary: summary, for: climb) }
                catch { log.error("late plan not kept: \(error.localizedDescription, privacy: .public)") }
            }
        case .failed(let code):
            // Some of these are passing (one frame it could not read, a venue file it could not get) and
            // the page carries on. Whether it still answers is what decides what the screen says, see `hand`.
            log.info("capture page reported: \(code, privacy: .public)")
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

    /// The user says what the hold at this spot of the picture is. The page finds the hold; what
    /// changed shows with its next reply. When it finds none there, the screen says so for a moment.
    func correct(at point: CGPoint, role: CaptureRole) {
        guard phase == .framing, page == .ready, color != nil else { return }
        Task {
            let found = await answer(.tap(x: point.x, y: point.y, role: role))
            // An empty answer is “no hold here”. A page that answers nothing at all gives the same; `false` means it has no `tap`.
            guard let text = found as? String, text.isEmpty else { tapMissed = false; return }
            tapMissed = true
            try? await Task.sleep(for: .seconds(3))
            tapMissed = false
        }
    }

    func resetCorrections() {
        guard phase == .framing, page == .ready else { return }
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
        fillInGrade()
        Task { await call(.rejectRoute(route.id)) }
    }

    /// The user's own choice of grade. It is never replaced afterwards.
    func pickGrade(_ value: String) {
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
        pipeline.setRecorder(recorder)
        // The judgement of this moment is what the analysis will be given. The page uses nothing
        // that comes after `lock`, so no more frames are prepared for it and live pose stops.
        pipeline.setFeeding(false)
        // Only a judgement of this take, for the gym now chosen, is locked. Right after the gym was
        // changed the page is still on the old one: then this take is recorded without a plan.
        if page == .ready, let begun, begun.id == captureID, begun.venue == venue { Task { await call(.lock) } }
        writeDraft()
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
            phase = .saved(climb.id)
            store.resumeAnalysis(cameraClosed: true)
        } catch {
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
