import Combine
import SwiftUI
import WebKit

/// Runs the pack's analysis for records that asked for it when they were saved: the ones made with
/// the in-app camera. Pose recognition comes first, on the store's own queue; when a record has its
/// poses, the analysis page is loaded out of sight, given the clip and the capture plan, and its
/// result is kept exactly as when the user opens the page. One record at a time, never while the
/// camera is in use or the app is in the background. The request is part of the record, so it
/// is picked up again after the app has been closed.
@MainActor
final class AnalysisQueue: ObservableObject {
    /// The page being used, for the root view to keep in the window where nobody sees it.
    @Published private(set) var webView: WKWebView?
    @Published private(set) var running: UUID?
    private let store: ClimbStore
    private let pack: GymPack?
    private var session: AnalysisSession?
    private var watching: Set<AnyCancellable> = []
    private var pageWatch: AnyCancellable?
    private var timeout: Task<Void, Never>?
    private var suspended = false
    private var runningRevision: Int?
    /// Records whose request could not be cleared; they are not tried again in this run of the app.
    private var settled: Set<UUID> = []
    /// Long enough for a twenty-minute clip on a slow phone; a page that takes longer is given up on.
    var patience: Duration = .seconds(600)

    init(store: ClimbStore, pack: GymPack? = GymPack.current) {
        self.store = store
        self.pack = pack?.supportsAnalysis == true ? pack : nil
        // Both publish before they change; look again once they have.
        store.$archive.receive(on: DispatchQueue.main).sink { [weak self] _ in self?.advance() }.store(in: &watching)
        store.$selectionOpen.receive(on: DispatchQueue.main).sink { [weak self] open in
            if open { self?.halt() } else { self?.advance() }
        }.store(in: &watching)
        store.$cameraOpen.receive(on: DispatchQueue.main).sink { [weak self] open in
            if open { self?.halt() } else { self?.advance() }
        }.store(in: &watching)
    }

    /// The app left the foreground, or came back.
    func setSuspended(_ suspended: Bool) {
        self.suspended = suspended
        if suspended { halt() } else { advance() }
    }

    /// The user opened the analysis page for this record themselves; that visit does the work.
    func cancel(_ id: UUID) {
        if running == id { halt() }
        try? store.setAnalysisQueued(id, false)
    }

    private func halt() {
        timeout?.cancel(); timeout = nil
        pageWatch = nil
        session?.onOutcome = nil
        session?.stop()
        session = nil
        webView = nil
        running = nil
        runningRevision = nil
    }

    private func advance() {
        if let running, store.climbs.first(where: { $0.id == running })?.subjectRevision != runningRevision { halt() }
        guard !suspended, !store.cameraOpen, !store.selectionOpen, session == nil else { return }
        for climb in store.climbs where climb.capture?.analysisQueued == true && climb.pendingAnalysis == nil && !settled.contains(climb.id) {
            // Already analysed, no page to do it, or pose recognition found nobody (or was stopped):
            // the request is done with. The record keeps its own “分析这一趟”.
            guard let pack, !store.analysisResults.contains(climb.id), climb.analysis.hasWorldLandmarks else {
                #if DEBUG
                CaptureDiagnostics.event("analysisQueue", ["record": climb.id.uuidString, "outcome": "skipped",
                                                           "why": pack == nil ? "no analysis page" : store.analysisResults.contains(climb.id) ? "already analysed" : "nobody found in the clip"])
                #endif
                settled.insert(climb.id)
                try? store.setAnalysisQueued(climb.id, false)
                continue
            }
            run(climb, pack)
            return
        }
    }

    private func run(_ climb: Climb, _ pack: GymPack) {
        let session = AnalysisSession(pack: pack, climb: climb, store: store)
        self.session = session
        running = climb.id
        runningRevision = climb.subjectRevision
        session.onOutcome = { [weak self] succeeded in self?.finish(climb.id, succeeded) }
        pageWatch = session.$webView.sink { [weak self] view in self?.webView = view }
        session.start()
        timeout = Task { [weak self, patience] in
            try? await Task.sleep(for: patience)
            guard !Task.isCancelled else { return }
            #if DEBUG
            CaptureDiagnostics.event("analysisQueue", ["record": climb.id.uuidString, "outcome": "timeout"])
            #endif
            self?.finish(climb.id, false)
        }
    }

    private func finish(_ id: UUID, _ succeeded: Bool) {
        guard running == id else { return }
        #if DEBUG
        CaptureDiagnostics.event("analysisQueue", ["record": id.uuidString, "outcome": succeeded ? "succeeded" : "failed",
                                                   "planUsed": store.climbs.first { $0.id == id }?.capture?.planUse?.used as Any])
        #endif
        halt()
        settled.insert(id)
        // Analysed without the marks made while framing is still analysed, but it is said.
        let unused = store.climbs.first { $0.id == id }?.capture?.planUse?.used == false
        store.setAnalysisMessage(!succeeded ? String(localized: "这一趟没能自动分析，可以在回看页再试。", bundle: AppLanguage.bundle)
                                 : unused ? CaptureText.planNotUsed
                                 : String(localized: "这一趟分析好了，打开就能看。", bundle: AppLanguage.bundle), for: id)
        do { try store.setAnalysisQueued(id, false) }
        catch { store.error = error.localizedDescription }
        advance()
    }
}

/// Keeps the queue's page in the window, behind everything, so its scripts run as on a visible page.
struct AnalysisQueueHost: View {
    @EnvironmentObject private var queue: AnalysisQueue
    var body: some View {
        if let webView = queue.webView {
            GymWebView(webView: webView).frame(width: 390, height: 760).allowsHitTesting(false).accessibilityHidden(true)
        }
    }
}
