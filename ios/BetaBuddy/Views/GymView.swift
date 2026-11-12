import SwiftUI
import WebKit
import os

/// What the gym feature is honest about. Shown before first entry and kept in 关于.
enum GymNotice {
    static let items = [
        "非官方整理，与场馆没有合作或授权关系。",
        "岩点位置和线路由爱好者根据公开视频整理，可能与现场不符，也可能已经换线。以现场为准。",
        "“不同身材怎么爬”是模型推演，不是教练建议，不保证动作可行或安全。",
        "难度档来自线路牌色条，不是定级结论。",
        "场馆数据随安装包内置，在手机上打开；不联网，不上传。"
    ]
}

/// Serves `gympack://local/…` from the installed pack directory and nothing else.
final class GymSchemeHandler: NSObject, WKURLSchemeHandler {
    // No remote source of any kind. The pack uses inline scripts, Blob workers and data: URLs.
    static let contentSecurityPolicy = "default-src 'self' gympack: 'unsafe-inline' 'wasm-unsafe-eval' blob: data:; object-src 'none'; base-uri 'self'; form-action 'none'"
    let pack: GymPack
    init(pack: GymPack) { self.pack = pack }

    func webView(_ webView: WKWebView, start task: WKURLSchemeTask) {
        guard let url = task.request.url else { return task.didFailWithError(URLError(.badURL)) }
        // Memory-mapped: a large model is paged in by the kernel instead of being copied onto the heap.
        guard let file = pack.file(for: url), let data = try? Data(contentsOf: file, options: .mappedIfSafe) else {
            return respond(task, url: url, status: 404, type: "text/plain; charset=utf-8", data: Data("Not found".utf8))
        }
        respond(task, url: url, status: 200, type: GymPack.mimeType(forExtension: file.pathExtension), data: data)
    }
    func webView(_ webView: WKWebView, stop task: WKURLSchemeTask) {}   // Responses complete synchronously.

    private func respond(_ task: WKURLSchemeTask, url: URL, status: Int, type: String, data: Data) {
        let headers = ["Content-Type": type, "Content-Length": String(data.count), "Cache-Control": "no-store",
                       "Content-Security-Policy": Self.contentSecurityPolicy, "X-Content-Type-Options": "nosniff"]
        guard let response = HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers) else {
            return task.didFailWithError(URLError(.badServerResponse))
        }
        task.didReceive(response)
        if task.request.httpMethod != "HEAD" { task.didReceive(data) }
        task.didFinish()
    }
}

/// Owns the one web view for the app's lifetime, so switching tabs keeps the scene and selection.
@MainActor
final class GymSession: NSObject, ObservableObject, WKNavigationDelegate, WKScriptMessageHandler {
    enum Phase: Equatable { case idle, loading, ready, failed(String) }
    static let handlerName = "gym"
    // Navigation is limited to the pack's own scheme by the delegate; these rules also stop subresource and fetch/XHR/WebSocket loads.
    static let blockRules = #"""
    [{"trigger":{"url-filter":"^https?://"},"action":{"type":"block"}},
     {"trigger":{"url-filter":"^wss?://"},"action":{"type":"block"}},
     {"trigger":{"url-filter":"^ftp://"},"action":{"type":"block"}}]
    """#

    let status: GymPack.Status
    @Published private(set) var phase = Phase.idle
    @Published private(set) var webView: WKWebView?
    /// Last route picked in the page. Kept for linking a recorded climb to a route (id and name only).
    @Published private(set) var selectedRoute: GymRoute?
    /// Set by the page's `ready`. Before it, a page error means the scene never came up; after it,
    /// errors (an unknown route id, say) are the page's own to show and must not cover it.
    private var pageReady = false
    private var loadStarted = Date()
    // Local system log only: load timings and the text of page-reported errors.
    private let log = Logger(subsystem: "com.betabuddy.ios", category: "gym")
    private var pack: GymPack? { if case .ready(let pack) = status { pack } else { nil } }

    init(status: GymPack.Status = GymPack.status) { self.status = status }

    func start() {
        guard phase == .idle, let pack else { return }
        phase = .loading
        Task {
            // Fail closed: without the network block the page is not shown at all.
            guard let rules = try? await WKContentRuleListStore.default()
                .compileContentRuleList(forIdentifier: "gym-offline", encodedContentRuleList: Self.blockRules) else {
                phase = .failed("无法启用离线保护，岩馆页面没有打开。")
                return
            }
            let configuration = WKWebViewConfiguration()
            configuration.websiteDataStore = .nonPersistent()
            configuration.allowsInlineMediaPlayback = true
            configuration.setURLSchemeHandler(GymSchemeHandler(pack: pack), forURLScheme: GymPack.scheme)
            configuration.userContentController.add(rules)
            configuration.userContentController.add(self, name: Self.handlerName)
            let view = WKWebView(frame: .zero, configuration: configuration)
            view.navigationDelegate = self
            view.allowsBackForwardNavigationGestures = false
            view.allowsLinkPreview = false
            view.isOpaque = false
            view.backgroundColor = UIColor(BB.paper)
            view.scrollView.backgroundColor = UIColor(BB.paper)
            view.scrollView.bounces = false
            view.scrollView.alwaysBounceVertical = false
            view.scrollView.bouncesZoom = false
            view.scrollView.minimumZoomScale = 1
            view.scrollView.maximumZoomScale = 1
            view.scrollView.pinchGestureRecognizer?.isEnabled = false
            view.scrollView.showsVerticalScrollIndicator = false
            view.scrollView.showsHorizontalScrollIndicator = false
            // The page lays itself out with env(safe-area-inset-*).
            view.scrollView.contentInsetAdjustmentBehavior = .never
            #if DEBUG
            view.isInspectable = true
            #endif
            webView = view
            loadStarted = Date()
            view.load(URLRequest(url: pack.entryURL))
        }
    }

    func reload() {
        guard let pack, let webView else { phase = .idle; start(); return }
        phase = .loading
        pageReady = false
        loadStarted = Date()
        webView.load(URLRequest(url: pack.entryURL))
    }

    func send(_ command: GymCommand) {
        guard phase == .ready else { return }
        webView?.evaluateJavaScript(command.script)
    }

    func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction) async -> WKNavigationActionPolicy {
        let blankSubframe = action.targetFrame?.isMainFrame == false && action.request.url?.scheme == "about"
        return action.targetFrame != nil && (pack?.allows(action.request.url) == true || blankSubframe) ? .allow : .cancel
    }
    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        log.info("document loaded in \(Date().timeIntervalSince(self.loadStarted), format: .fixed(precision: 2)) s")
        if phase == .loading { phase = .ready }
    }
    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) { fail(error) }
    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) { fail(error) }
    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        phase = .failed("岩馆页面被系统回收了，可能是内存不足。")
    }
    private func fail(_ error: Error) {
        guard (error as NSError).code != NSURLErrorCancelled else { return }
        log.error("load failed: \(error.localizedDescription, privacy: .public)")
        phase = .failed("岩馆页面没有打开：\(error.localizedDescription)")
    }

    func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage) {
        guard message.frameInfo.isMainFrame, let parsed = GymMessage(body: message.body) else { return }
        switch parsed {
        case .ready:
            pageReady = true
            log.info("page ready in \(Date().timeIntervalSince(self.loadStarted), format: .fixed(precision: 2)) s")
            if phase == .loading { phase = .ready }
            #if DEBUG
            // `-gymOpenWall <id>` / `-gymOpenRoute <id>` on the launch command line, for screenshots and bridge checks.
            if let wall = UserDefaults.standard.string(forKey: "gymOpenWall") { send(.openWall(wall)) }
            if let route = UserDefaults.standard.string(forKey: "gymOpenRoute") { send(.openRoute(route)) }
            #endif
        case .routeSelected(let route): selectedRoute = route
        case .error(let text):
            log.error("page error (\(self.pageReady ? "after" : "before", privacy: .public) ready): \(text, privacy: .public)")
            if !pageReady { phase = .failed(text.isEmpty ? "岩馆页面报告了一个错误。" : text) }
        }
    }
}

struct GymWebView: UIViewRepresentable {
    let webView: WKWebView
    func makeUIView(context: Context) -> WKWebView { webView }
    func updateUIView(_ view: WKWebView, context: Context) {}
}

struct GymView: View {
    @ObservedObject var session: GymSession
    @AppStorage("gymNoticeSeen") private var noticeSeen = false
    var body: some View {
        switch session.status {
        case .missing:
            unavailable(title:"这个构建没有\n内置岩馆数据",
                        detail:"场馆模型、线路和攀爬模拟随正式安装包提供，不在公开源码里。回看、攀爬册和练习不受影响。")
        case .invalid(let error):
            unavailable(title:"岩馆数据不完整", detail:"\(Self.describe(error))请重新安装完整版本。")
        case .ready(let pack):
            if noticeSeen { stage } else { intro(pack) }
        }
    }

    private func unavailable(title: String, detail: String) -> some View {
        PaperPage {
            Stamp(text:"GYM / NOT IN THIS BUILD")
            VStack(alignment:.leading,spacing:18) {
                Image(systemName:"map").font(.system(size:44,weight:.light))
                Text(title).font(.title.bold()).accessibilityIdentifier("gymUnavailable")
                Text(detail).foregroundStyle(BB.muted)
            }.padding(24).frame(maxWidth:.infinity,alignment:.leading).background(BB.lavender,in:RoundedRectangle(cornerRadius:24))
        }
    }

    private func intro(_ pack: GymPack) -> some View {
        PaperPage {
            Stamp(text:"GYM / UNOFFICIAL")
            Text(pack.manifest.gymName).font(.system(size:34,weight:.heavy)).accessibilityIdentifier("gymName")
            Text("\(pack.manifest.routeCount) 条线路 · 数据版本 \(pack.manifest.packVersion)")
                .font(.system(size:12,weight:.bold,design:.monospaced))
            VStack(alignment:.leading,spacing:10) {
                Text("换个身材，\n看这条线怎么爬。").font(.title2.bold())
                Text("在 3D 岩馆里找到一条线，换一个身材，看看模型推演出的爬法有什么不同。").font(.subheadline).lineSpacing(4)
            }.padding(22).frame(maxWidth:.infinity,alignment:.leading).background(BB.lime,in:RoundedRectangle(cornerRadius:22))
            GymNoticeList()
            BigButton(title:"知道了，进入岩馆") { noticeSeen = true }.accessibilityIdentifier("enterGym")
        }
    }

    private var stage: some View {
        ZStack {
            BB.paper.ignoresSafeArea()
            if let webView = session.webView {
                GymWebView(webView:webView).ignoresSafeArea(edges:.top)
                    .opacity(session.phase == .ready ? 1 : 0).accessibilityIdentifier("gymPage")
            }
            switch session.phase {
            case .idle, .loading:
                VStack(spacing:18) {
                    ProgressView().tint(BB.ink)
                    Text("正在打开岩馆…").font(.headline)
                    Text("ON YOUR IPHONE").font(.system(size:10,weight:.bold,design:.monospaced)).foregroundStyle(BB.muted)
                }.foregroundStyle(BB.ink)
            case .failed(let message):
                VStack(alignment:.leading,spacing:18) {
                    Image(systemName:"exclamationmark.circle").font(.system(size:44,weight:.light))
                    Text("岩馆没有打开").font(.title.bold())
                    Text(message).foregroundStyle(BB.muted).accessibilityIdentifier("gymError")
                    BigButton(title:"重新加载",icon:"arrow.clockwise") { session.reload() }
                }.padding(24).frame(maxWidth:.infinity,alignment:.leading)
                    .background(BB.lavender,in:RoundedRectangle(cornerRadius:24)).padding(24).foregroundStyle(BB.ink)
            case .ready: EmptyView()
            }
        }.onAppear { session.start() }
    }

    static func describe(_ error: GymPackError) -> String {
        switch error {
        case .unreadableManifest: "数据包清单无法读取。"
        case .unsupportedSchema(let schema): "数据包格式（\(schema)）比这个版本的应用新。"
        case .unsafePath, .missingFile: "数据包缺少文件。"
        }
    }
}

struct GymNoticeList: View {
    var body: some View {
        VStack(alignment:.leading,spacing:14) {
            SectionLabel(title:"进去之前",trailing:"KEEP IT HONEST")
            ForEach(Array(GymNotice.items.enumerated()),id:\.offset) { index, item in
                HStack(alignment:.top,spacing:14) {
                    Text(String(format:"%02d",index+1)).font(.system(size:15,weight:.black,design:.rounded)).foregroundStyle(BB.muted)
                    Text(item).font(.subheadline).lineSpacing(4).frame(maxWidth:.infinity,alignment:.leading)
                }
            }
        }
    }
}
