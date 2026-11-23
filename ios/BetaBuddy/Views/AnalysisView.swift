import SwiftUI
import WebKit
import os

/// Keeps the web view's content controller from owning the session, so closing the page frees both.
final class WeakMessageHandler: NSObject, WKScriptMessageHandler {
    weak var target: WKScriptMessageHandler?
    init(_ target: WKScriptMessageHandler) { self.target = target }
    func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage) {
        target?.userContentController(controller, didReceive: message)
    }
}

/// Hosts the pack's analysis page for one record. The page does the analysis and shows it; the app
/// prepares stills and poses from the clip, hands them over, keeps the result the page gives back,
/// and answers its messages. One web view per presentation, loaded the same way as the gym.
@MainActor
final class AnalysisSession: NSObject, ObservableObject, WKNavigationDelegate, WKScriptMessageHandler {
    enum Phase: Equatable { case loading, preparing, shown, failed(String) }
    @Published private(set) var phase = Phase.loading
    @Published private(set) var webView: WKWebView?
    /// A moment in the clip the page asked to see, in seconds.
    @Published private(set) var seekRequest: Double?
    private let pack: GymPack
    private let climb: Climb
    private let store: ClimbStore
    private var started = false
    private var pageReady = false
    private var feed: Task<Void, Never>?
    /// Called once per load: `true` when the page handed back a result, `false` when it could not
    /// analyse or never came up. Used when the analysis runs out of sight after a recording.
    var onOutcome: ((Bool) -> Void)?
    private var reported = false
    // Local system log only.
    private let log = Logger(subsystem: "com.betabuddy.ios", category: "analysis")

    init(pack: GymPack, climb: Climb, store: ClimbStore) {
        self.pack = pack; self.climb = climb; self.store = store
    }

    func start() {
        guard !started, let url = pack.analysisURL(language: AppLanguage.current) else { return }
        started = true
        phase = .loading
        Task {
            guard let view = await GymSession.makeWebView(pack: pack, handler: WeakMessageHandler(self), delegate: self) else {
                phase = .failed("")
                report(false)
                return
            }
            guard started else { return }
            webView = view
            view.load(URLRequest(url: url))
        }
    }

    private func report(_ succeeded: Bool) {
        guard !reported else { return }
        reported = true
        onOutcome?(succeeded)
    }

    func reload() {
        feed?.cancel()
        pageReady = false
        reported = false
        guard let webView, let url = pack.analysisURL(language: AppLanguage.current) else { started = false; start(); return }
        phase = .loading
        webView.load(URLRequest(url: url))
    }

    func stop() {
        started = false
        feed?.cancel()
        webView?.stopLoading()
        webView?.configuration.userContentController.removeScriptMessageHandler(forName: GymSession.handlerName)
        webView = nil
    }

    /// `false` when the page does not provide the call.
    @discardableResult
    private func call(_ command: AnalysisCommand) async -> Bool {
        guard let webView else { return false }
        return await withCheckedContinuation { continuation in
            webView.evaluateJavaScript(command.script) { value, _ in continuation.resume(returning: value as? Bool == true) }
        }
    }

    /// A saved result is shown as it is. Otherwise the clip is prepared and handed to the page.
    private func present() async {
        if let saved = store.analysisResult(for: climb) {
            phase = .shown
            await call(.show(saved))
            return
        }
        phase = .preparing
        do {
            let stills = try await VideoAnalyzer().stills(url: store.file(climb.videoName))
            try Task.checkCancellation()
            // A record made with the in-app camera brings the capture page's plan; any other has none.
            guard await call(.begin(id: climb.id.uuidString, width: stills.width, height: stills.height, climber: store.analysisClimber,
                                    capture: store.capturePlan(for: climb))) else {
                phase = .failed("")
                report(false)
                return
            }
            phase = .shown
            for (index, frame) in stills.frames.enumerated() {
                try Task.checkCancellation()
                await call(.addFrame(index: index, time: frame.time, jpegBase64: frame.jpegBase64))
            }
            let poses = climb.analysis.frames
            for start in stride(from: 0, to: poses.count, by: AnalysisCommand.poseBatch) {
                try Task.checkCancellation()
                await call(.addPoses(Array(poses[start..<min(poses.count, start + AnalysisCommand.poseBatch)])))
            }
            try Task.checkCancellation()
            await call(.run)
        } catch is CancellationError {
        } catch { phase = .failed(error.localizedDescription); report(false) }
    }

    func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction) async -> WKNavigationActionPolicy {
        GymSession.navigationPolicy(pack: pack, for: action)
    }
    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) { fail(error) }
    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) { fail(error) }
    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        feed?.cancel()
        phase = .failed("")
        report(false)
    }
    private func fail(_ error: Error) {
        guard (error as NSError).code != NSURLErrorCancelled else { return }
        log.error("load failed: \(error.localizedDescription, privacy: .public)")
        phase = .failed(error.localizedDescription)
        report(false)
    }

    func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage) {
        guard message.frameInfo.isMainFrame, let parsed = AnalysisMessage(body: message.body) else { return }
        switch parsed {
        case .ready:
            guard !pageReady else { return }
            pageReady = true
            feed?.cancel()
            feed = Task { await present() }
        case .result(let id, let json, let climber, let planUse):
            // After showing a saved result the page has no id to give back; this web view only ever serves this record.
            guard id.isEmpty || id == climb.id.uuidString else { return }
            do { try store.saveAnalysisResult(json, for: climb.id) }
            catch { store.error = error.localizedDescription }
            if !climber.isEmpty { store.analysisClimber = climber }
            try? store.setPlanUse(planUse, for: climb.id)
            report(store.analysisResults.contains(climb.id))
        case .seek(let time): seekRequest = time
        case .failed(let code):
            // The page is showing its own explanation; nothing is saved and the page stays open.
            log.info("page could not analyse: \(code, privacy: .public)")
            report(false)
        case .pageError(let text):
            log.error("page error (\(self.pageReady ? "after" : "before", privacy: .public) ready): \(text, privacy: .public)")
            if !pageReady { phase = .failed(text); report(false) }
        }
    }
}

struct AnalysisView: View {
    @StateObject private var session: AnalysisSession
    @Environment(\.dismiss) private var dismiss
    private let onSeek: (Double) -> Void

    init(pack: GymPack, climb: Climb, store: ClimbStore, onSeek: @escaping (Double) -> Void) {
        _session = StateObject(wrappedValue: AnalysisSession(pack: pack, climb: climb, store: store))
        self.onSeek = onSeek
    }

    var body: some View {
        ZStack {
            BB.paper.ignoresSafeArea()
            if let webView = session.webView {
                // The page fills the screen and keeps clear of the safe areas itself.
                GymWebView(webView: webView).ignoresSafeArea()
                    .opacity(session.phase == .shown ? 1 : 0).accessibilityIdentifier("analysisPage")
            }
            switch session.phase {
            case .loading, .preparing:
                VStack(spacing: 18) {
                    ProgressView().tint(BB.ink)
                    Text("正在准备画面…").font(.headline)
                }.foregroundStyle(BB.ink)
            case .failed(let message):
                VStack(alignment: .leading, spacing: 18) {
                    Image(systemName: "exclamationmark.circle").font(.system(size: 44, weight: .light))
                    Text("分析页面没有打开").font(.title.bold())
                    if !message.isEmpty { Text(verbatim: message).foregroundStyle(BB.muted).accessibilityIdentifier("analysisError") }
                    BigButton(title: "重新加载", icon: "arrow.clockwise") { session.reload() }
                }.padding(24).frame(maxWidth: .infinity, alignment: .leading)
                    .background(BB.lavender, in: RoundedRectangle(cornerRadius: 24)).padding(24).foregroundStyle(BB.ink)
            case .shown: EmptyView()
            }
        }
        // The page scrolls; keep its content from running under the status bar.
        .overlay(alignment: .top) {
            GeometryReader { proxy in
                BB.paper.frame(height: proxy.safeAreaInsets.top).ignoresSafeArea(edges: .top)
            }.allowsHitTesting(false)
        }
        .overlay(alignment: .topTrailing) {
            Button { dismiss() } label: {
                Text("完成").font(.subheadline.bold()).padding(.horizontal, 16).frame(minHeight: 44)
                    .background(BB.paper, in: Capsule()).overlay(Capsule().stroke(BB.line, lineWidth: 1.5))
                    .contentShape(Rectangle())
            }.buttonStyle(.plain).foregroundStyle(BB.ink).padding(.trailing, 12).padding(.top, 4)
                .accessibilityIdentifier("analysisDone")
        }
        .onAppear { session.start() }
        .onDisappear { session.stop() }
        .onChange(of: session.seekRequest) { _, time in if let time { onSeek(time) } }
    }
}
