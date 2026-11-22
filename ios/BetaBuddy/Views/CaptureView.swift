import SwiftUI
import UIKit

/// Shows a source's preview layer, uncropped.
private struct CapturePreview: UIViewRepresentable {
    let layer: CALayer
    final class Surface: UIView {
        var content: CALayer? { didSet { oldValue?.removeFromSuperlayer(); if let content { layer.addSublayer(content) } } }
        override func layoutSubviews() {
            super.layoutSubviews()
            CATransaction.begin(); CATransaction.setDisableActions(true)
            content?.frame = bounds
            CATransaction.commit()
        }
    }
    func makeUIView(context: Context) -> Surface {
        let view = Surface(); view.backgroundColor = .black; view.content = layer
        return view
    }
    func updateUIView(_ view: Surface, context: Context) { if view.content !== layer { view.content = layer } }
}

/// What the capture page recognised, drawn over the picture: the chosen colour's holds, with the
/// start and finish marked. Dashed outlines are the page's own guesses; solid ones were set by the user.
private struct CaptureOverlay: View {
    var holds: [CaptureAssessment.Hold]
    var size: CGSize
    private func path(_ hold: CaptureAssessment.Hold) -> Path {
        if hold.poly.count >= 3 {
            var path = Path()
            path.addLines(hold.poly.map { CGPoint(x: $0.x * size.width, y: $0.y * size.height) })
            path.closeSubpath()
            return path
        }
        let rect = CGRect(x: hold.box.minX * size.width, y: hold.box.minY * size.height, width: hold.box.width * size.width, height: hold.box.height * size.height)
        return Path(roundedRect: rect.insetBy(dx: -2, dy: -2), cornerRadius: 6)
    }
    private func color(_ role: CaptureRole) -> Color {
        switch role {
        case .startHand, .startFoot: BB.lime
        case .finish: BB.lavender
        default: .white
        }
    }
    private func symbol(_ role: CaptureRole) -> String {
        switch role {
        case .startHand: "hand.raised.fill"
        case .startFoot: "shoeprints.fill"
        default: "flag.checkered"
        }
    }
    var body: some View {
        ZStack(alignment: .topLeading) {
            Canvas { context, _ in
                for hold in holds {
                    let outline = path(hold)
                    let marked = hold.role != .hold
                    context.stroke(outline, with: .color(.black.opacity(0.7)), style: StrokeStyle(lineWidth: marked ? 6 : 4.5, lineJoin: .round))
                    context.stroke(outline, with: .color(color(hold.role)),
                                   style: StrokeStyle(lineWidth: marked ? 3 : 2, lineJoin: .round, dash: hold.user ? [] : [6, 4]))
                }
            }
            // Start and finish also carry a word: colour alone does not say which is which.
            ForEach(holds.filter { $0.role != .hold }) { hold in
                HStack(spacing: 3) {
                    Image(systemName: symbol(hold.role)).font(.system(size: 9, weight: .bold))
                    Text(verbatim: CaptureText.role(hold.role)).font(.system(size: 10, weight: .bold)).lineLimit(1)
                    if hold.user { Image(systemName: "checkmark").font(.system(size: 8, weight: .black)) }
                }
                .padding(.horizontal, 6).padding(.vertical, 3)
                .foregroundStyle(BB.ink).background(color(hold.role), in: Capsule())
                .fixedSize()
                .position(x: min(max(hold.box.midX * size.width, 40), size.width - 40),
                          y: max(hold.box.minY * size.height - 13, 10))
            }
        }
        .frame(width: size.width, height: size.height)
        .allowsHitTesting(false).accessibilityHidden(true)
    }
}

/// Recommends gyms by distance when the position is known, and leads to the full directory.
private struct CaptureGymSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Binding var gymId: String?
    @Binding var gymName: String
    @ObservedObject var locator: GymLocator
    var body: some View {
        NavigationStack {
            List {
                if locator.access == .allowed {
                    let nearby = locator.nearby()
                    Section {
                        ForEach(nearby) { item in
                            Button { gymId = item.gym.id; gymName = item.gym.name; dismiss() } label: {
                                HStack {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(verbatim: item.gym.name).foregroundStyle(BB.ink)
                                        Text(verbatim: item.gym.location).font(.caption).foregroundStyle(BB.muted)
                                    }
                                    Spacer()
                                    if item.showsDistance {
                                        Text(verbatim: CaptureText.distance(item.meters)).font(.caption.monospacedDigit()).foregroundStyle(BB.muted)
                                    }
                                    if gymId == item.gym.id { Image(systemName: "checkmark").foregroundStyle(BB.ink) }
                                }.padding(.vertical, 3)
                            }.accessibilityIdentifier("nearbyGym-\(item.gym.id)")
                        }
                        if locator.locating { ProgressView() }
                        else if nearby.isEmpty { Text("附近没有找到目录里的岩馆。").foregroundStyle(BB.muted) }
                    } header: { Text("离你近的") } footer: { Text("位置只在这台 iPhone 上用来给岩馆排序，不保存，也不离开设备。") }
                } else if locator.access == .notAsked {
                    Section {
                        Button { locator.ask() } label: { Label("按距离找附近的岩馆", systemImage: "location") }
                            .accessibilityIdentifier("askLocation")
                    } footer: { Text("位置只在这台 iPhone 上用来给岩馆排序，不保存，也不离开设备。") }
                }
                Section {
                    NavigationLink { GymPickerView(gymId: $gymId, gymName: $gymName) } label: {
                        Label("全部岩馆", systemImage: "list.bullet")
                    }.accessibilityIdentifier("allGyms")
                }
            }
            .scrollContentBackground(.hidden).background(BB.paper)
            .navigationTitle("选择岩馆").navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .cancellationAction) { Button("取消") { dismiss() } } }
            .onAppear { locator.locateIfAllowed() }
            // A choice made in the full directory closes the whole sheet.
            .onChange(of: gymName) { _, _ in dismiss() }
        }.presentationDetents([.medium, .large])
    }
}

struct CaptureView: View {
    @EnvironmentObject private var store: ClimbStore
    @Environment(\.dismiss) private var dismiss
    @StateObject private var session: CaptureSession
    @StateObject private var locator = GymLocator()
    @State private var choosingGym = false
    @State private var confirming = false
    @State private var tapped: CGPoint?
    @State private var sent = false
    @State private var viewing: Climb?
    /// Leaves the camera for the photo library import.
    private let onImport: () -> Void

    init(store: ClimbStore, pack: GymPack? = GymPack.current, source: CaptureSource = CaptureSession.makeSource(), onImport: @escaping () -> Void) {
        _session = StateObject(wrappedValue: CaptureSession(store: store, pack: pack, source: source))
        self.onImport = onImport
    }

    private var saved: Climb? {
        guard case .saved(let id) = session.phase else { return nil }
        return store.climbs.first { $0.id == id }
    }
    private var framing: Bool { session.phase == .framing && session.camera == .running }

    var body: some View {
        ZStack {
            BB.ink.ignoresSafeArea()
            VStack(spacing: 10) {
                topBar
                picture
                // Below the picture, not on it: the start of a problem is at the bottom of the wall.
                if framing { colours }
                bottomBar
            }.padding(.horizontal, 12).padding(.vertical, 8)
            // The capture page has nothing to show; it only needs to be alive.
            if let webView = session.webView {
                GymWebView(webView: webView).frame(width: 1, height: 1).opacity(0.01)
                    .allowsHitTesting(false).accessibilityHidden(true)
            }
        }
        .foregroundStyle(BB.paper)
        .statusBarHidden(true)
        .onAppear {
            UIApplication.shared.isIdleTimerDisabled = true   // The phone stands there while the user climbs.
            session.start()
            locator.locateIfAllowed()
        }
        .onDisappear {
            UIApplication.shared.isIdleTimerDisabled = false
            Task { await session.close() }
        }
        .sheet(isPresented: $choosingGym) { CaptureGymSheet(gymId: $session.gymId, gymName: $session.gymName, locator: locator) }
        .fullScreenCover(item: $viewing) { climb in
            if let pack = GymPack.current, pack.supportsAnalysis { AnalysisView(pack: pack, climb: climb, store: store) { _ in viewing = nil } }
        }
        .confirmationDialog("从这个机位拍，多半分析不了", isPresented: $confirming, titleVisibility: .visible) {
            Button("仍然开始录") { session.startRecording() }.accessibilityIdentifier("recordAnyway")
            Button("再调一下机位", role: .cancel) {}
        } message: { Text(verbatim: session.assessment?.hint.map(CaptureText.line) ?? "") }
        .confirmationDialog("这个岩点是…", isPresented: Binding(get: { tapped != nil }, set: { if !$0 { tapped = nil } }), titleVisibility: .visible) {
            ForEach(CaptureRole.allCases, id: \.self) { role in
                Button { if let tapped { session.correct(at: tapped, role: role) } } label: { Text(verbatim: CaptureText.role(role)) }
            }
            Button("恢复自动识别") { session.resetCorrections() }
            Button("取消", role: .cancel) {}
        }
        .alert(Text(verbatim: session.notice.map(CaptureText.notice) ?? ""), isPresented: Binding(get: { session.notice != nil }, set: { if !$0 { session.notice = nil } })) {
            Button("知道了", role: .cancel) { session.notice = nil }
        }
    }

    // MARK: Top: close, gym, grade

    private var topBar: some View {
        HStack(spacing: 8) {
            Button { Task { await session.close(); dismiss() } } label: {
                Image(systemName: "xmark").font(.system(size: 16, weight: .bold)).frame(width: 44, height: 44)
                    .background(.white.opacity(0.14), in: Circle())
            }.buttonStyle(.plain).accessibilityLabel(Text("关闭")).accessibilityIdentifier("captureClose")
                .disabled(session.phase == .recording || session.phase == .saving)
                .opacity(session.phase == .recording || session.phase == .saving ? 0.3 : 1)
            if session.phase == .recording, let began = session.recordingStarted {
                Spacer(minLength: 0)
                TimelineView(.periodic(from: began, by: 0.5)) { context in
                    HStack(spacing: 8) {
                        Circle().fill(.red).frame(width: 10, height: 10)
                        Text("正在录").font(.subheadline.bold())
                        Text(verbatim: clockText(context.date.timeIntervalSince(began))).font(.system(.subheadline, design: .monospaced).bold()).monospacedDigit()
                            .accessibilityIdentifier("captureElapsed")
                    }.padding(.horizontal, 14).frame(minHeight: 44).background(.white.opacity(0.14), in: Capsule())
                }
                Spacer(minLength: 0)
                Color.clear.frame(width: 44, height: 44)
            } else {
                Button { choosingGym = true } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "mappin.and.ellipse")
                        if session.gymName.isEmpty { Text("选择岩馆") } else { Text(verbatim: session.gymName) }
                        Image(systemName: "chevron.down").font(.caption2.bold())
                    }.font(.subheadline.bold()).lineLimit(1).padding(.horizontal, 12).frame(minHeight: 44)
                        .frame(maxWidth: .infinity, alignment: .leading).background(.white.opacity(0.14), in: Capsule())
                }.buttonStyle(.plain).disabled(session.phase != .framing).accessibilityIdentifier("captureGym")
                Menu {
                    Picker("线路难度", selection: $session.grade) {
                        Text("未标级").tag(ungradedValue)
                        ForEach(0..<13) { Text(verbatim: "V\($0)").tag("V\($0)") }
                    }
                } label: {
                    HStack(spacing: 6) {
                        Text(verbatim: gradeText(session.grade))
                        Image(systemName: "chevron.down").font(.caption2.bold())
                    }.font(.subheadline.bold()).lineLimit(1).padding(.horizontal, 12).frame(minHeight: 44)
                        .background(.white.opacity(0.14), in: Capsule())
                }.disabled(session.phase != .framing).accessibilityLabel(Text("线路难度")).accessibilityValue(Text(verbatim: gradeText(session.grade)))
                    .accessibilityIdentifier("captureGrade")
            }
        }
    }

    // MARK: Middle: the picture and what is drawn on it

    private var picture: some View {
        GeometryReader { proxy in
            let shape = session.frameSize
            let scale = min(proxy.size.width / shape.width, proxy.size.height / shape.height)
            let size = CGSize(width: shape.width * scale, height: shape.height * scale)
            ZStack {
                switch session.camera {
                case .denied: unavailable(title: "没有相机权限", detail: "在系统设置里允许 BetaBuddy 使用相机，就能在这里拍。画面和录像只留在这台 iPhone 上。", settings: true)
                case .unavailable: unavailable(title: "这台设备没有可用的相机", detail: "仍然可以从相册导入拍好的视频。", settings: false)
                case .starting, .running:
                    CapturePreview(layer: session.source.previewLayer)
                    if session.phase == .framing || session.phase == .recording, session.color != nil, let holds = session.assessment?.holds {
                        CaptureOverlay(holds: holds, size: size).opacity(session.phase == .recording ? 0.45 : 1)
                    }
                    if framing {
                        // A tap on the picture says what the hold there is.
                        Color.clear.contentShape(Rectangle())
                            .gesture(SpatialTapGesture().onEnded { value in
                                guard session.page == .ready, session.color != nil else { return }
                                tapped = CGPoint(x: value.location.x / size.width, y: value.location.y / size.height)
                            })
                            .accessibilityElement().accessibilityLabel(Text("这个岩点是…")).accessibilityIdentifier("capturePicture")
                        VStack(spacing: 8) {
                            guidance
                            Spacer(minLength: 0)
                        }.padding(10)
                    } else if session.phase == .recording {
                        VStack {
                            Spacer()
                            Text("手机放稳，去爬吧。爬完回来点停止。").font(.subheadline.bold()).multilineTextAlignment(.center)
                                .padding(.horizontal, 14).padding(.vertical, 10).foregroundStyle(BB.ink)
                                .background(BB.paper, in: RoundedRectangle(cornerRadius: 14)).padding(10)
                        }
                    } else if session.phase == .saving {
                        VStack(spacing: 14) { ProgressView().tint(BB.ink); Text("正在保存…").font(.headline) }
                            .padding(24).foregroundStyle(BB.ink).background(BB.paper, in: RoundedRectangle(cornerRadius: 22))
                    } else if case .saveFailed(let message) = session.phase {
                        saveFailed(message)
                    } else if let saved {
                        savedCard(saved)
                    }
                }
            }
            .frame(width: size.width, height: size.height)
            .clipShape(RoundedRectangle(cornerRadius: 18))
            .position(x: proxy.size.width / 2, y: proxy.size.height / 2)
        }
    }

    /// One line: what to change, or that the position is good. Plus an indicator for the score.
    @ViewBuilder
    private var guidance: some View {
        let assessment = session.assessment
        switch session.page {
        case .unavailable:
            pill(icon: "video", text: Text("这个版本没有机位检查和岩点识别，照常可以录。"), tone: .plain).accessibilityIdentifier("captureNoChecks")
        case .failed:
            pill(icon: "exclamationmark.triangle", text: Text("机位检查暂时用不了，照常可以录。"), tone: .plain).accessibilityIdentifier("captureNoChecks")
        case .loading:
            pill(icon: "viewfinder", text: Text("正在看画面…"), tone: .plain)
        case .ready:
            if let assessment {
                VStack(spacing: 6) {
                    if assessment.ready {
                        pill(icon: "checkmark.circle.fill", text: Text("机位可以，开始吧"), tone: .ready)
                            .accessibilityIdentifier("captureReady")
                        // Nothing blocks, but the page still has something to say.
                        if let hint = assessment.hint {
                            pill(icon: "info.circle", text: Text(verbatim: CaptureText.line(hint)), tone: .plain)
                                .accessibilityIdentifier("captureHint").accessibilityValue(Text(verbatim: hint.code))
                        }
                    } else if let hint = assessment.hint {
                        pill(icon: hint.blocking ? "exclamationmark.triangle.fill" : "info.circle", text: Text(verbatim: CaptureText.line(hint)), tone: hint.blocking ? .blocked : .plain)
                            .accessibilityIdentifier("captureHint").accessibilityValue(Text(verbatim: hint.code))
                    } else {
                        pill(icon: "hand.raised", text: Text("放稳手机，马上就好"), tone: .plain).accessibilityIdentifier("captureSettling")
                    }
                    if let score = assessment.score { quality(score) }
                }
            } else {
                pill(icon: "viewfinder", text: Text("正在看画面…"), tone: .plain)
            }
        }
    }

    private enum Tone { case plain, blocked, ready }
    private func pill(icon: String, text: Text, tone: Tone) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: icon).font(.subheadline.bold())
            text.font(.subheadline.bold()).fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, 14).padding(.vertical, 10).frame(maxWidth: .infinity, alignment: .leading)
        .foregroundStyle(BB.ink)
        .background(tone == .ready ? BB.lime : BB.paper, in: RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(tone == .blocked ? Color.orange : .clear, lineWidth: 2.5))
        .accessibilityElement(children: .combine)
    }

    /// Five segments, never a number: the score only orders camera positions.
    private func quality(_ score: Double) -> some View {
        let level = min(5, max(1, Int((score * 5).rounded(.up))))
        return HStack(spacing: 6) {
            Text("机位").font(.caption2.bold())
            HStack(spacing: 3) {
                ForEach(1...5, id: \.self) { index in
                    Capsule().fill(index <= level ? BB.lime : .white.opacity(0.35)).frame(width: 16, height: 6)
                }
            }
        }
        .padding(.horizontal, 10).padding(.vertical, 6).background(.black.opacity(0.55), in: Capsule())
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .ignore).accessibilityLabel(Text("机位"))
        .accessibilityValue(level >= 4 ? Text("好") : level >= 3 ? Text("一般") : Text("差"))
        .accessibilityIdentifier("captureQuality")
    }

    /// Chips for the colours the page sees on the wall, and the reminder that what is drawn is a guess.
    @ViewBuilder
    private var colours: some View {
        if session.page == .ready, !session.palette.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                Text(session.color == nil ? "选你要爬的颜色" : "虚线是自动认的，可能认错。点一下岩点可以改。")
                    .font(.caption.bold()).fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("captureColourNote")
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(session.palette) { swatch in
                            let selected = session.color == swatch.key
                            Button { session.selectColor(swatch.key) } label: {
                                HStack(spacing: 6) {
                                    Circle().fill(Color(swatch.rgb)).frame(width: 22, height: 22)
                                        .overlay(Circle().stroke(.white, lineWidth: 1.5))
                                    if let name = CaptureText.colorName(swatch.key) { Text(verbatim: name).font(.caption.bold()) }
                                    Text(verbatim: "\(swatch.count)").font(.caption.monospacedDigit())
                                    if selected { Image(systemName: "checkmark").font(.caption2.bold()) }
                                }
                                .padding(.horizontal, 10).frame(minHeight: 44)
                                .foregroundStyle(selected ? BB.ink : BB.paper)
                                .background(selected ? BB.lime : .white.opacity(0.14), in: Capsule())
                            }.buttonStyle(.plain)
                                .accessibilityLabel(Text(verbatim: CaptureText.colorName(swatch.key) ?? swatch.key))
                                .accessibilityAddTraits(selected ? [.isSelected] : [])
                                .accessibilityIdentifier("captureColor-\(swatch.key)")
                        }
                    }
                }
            }
        }
    }

    private func unavailable(title: LocalizedStringKey, detail: LocalizedStringKey, settings: Bool) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Image(systemName: "video.slash").font(.system(size: 40, weight: .light))
            Text(title).font(.title2.bold()).accessibilityIdentifier("captureUnavailable")
            Text(detail).foregroundStyle(BB.muted).fixedSize(horizontal: false, vertical: true)
            if settings {
                Button("打开系统设置") {
                    if let url = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(url) }
                }.buttonStyle(RouteButtonStyle())
            }
            Button("改从相册导入") { Task { await session.close(); onImport() } }.buttonStyle(RouteButtonStyle(tone: .paper))
        }
        .padding(22).frame(maxWidth: .infinity, alignment: .leading).foregroundStyle(BB.ink)
        .background(BB.lavender, in: RoundedRectangle(cornerRadius: 22)).padding(12)
    }

    private func saveFailed(_ message: String) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("这一趟还没存进攀爬册").font(.title3.bold())
            Text(verbatim: message).font(.subheadline).foregroundStyle(BB.muted).fixedSize(horizontal: false, vertical: true)
            Button("再存一次") { Task { await session.save() } }.buttonStyle(RouteButtonStyle())
            Button("放弃这段录像", role: .destructive) { session.discardHeldClip() }.buttonStyle(RouteButtonStyle(tone: .paper))
        }
        .padding(22).frame(maxWidth: .infinity, alignment: .leading).foregroundStyle(BB.ink)
        .background(BB.paper, in: RoundedRectangle(cornerRadius: 22)).padding(12)
    }

    /// Saved first. What happens to the clip afterwards is reported here but never waited for.
    private func savedCard(_ climb: Climb) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("已保存到攀爬册", systemImage: "checkmark.circle.fill").font(.title3.bold()).accessibilityIdentifier("captureSaved")
            HStack(alignment: .top, spacing: 12) {
                if let image = UIImage(contentsOfFile: store.file(climb.thumbnailName).path) {
                    Image(uiImage: image).resizable().scaledToFill().frame(width: 64, height: 88).clipShape(RoundedRectangle(cornerRadius: 12))
                }
                VStack(alignment: .leading, spacing: 5) {
                    Text(verbatim: "\(gradeText(climb.grade))  /  \(clockText(climb.analysis.duration))").font(.system(size: 13, weight: .bold, design: .monospaced))
                    if !climb.shortGymName.isEmpty { Text(verbatim: climb.shortGymName).font(.caption).foregroundStyle(BB.muted) }
                    CaptureColourLabel(capture: climb.capture)
                    if let reason = CaptureText.stop(session.stopReason) {
                        Text(verbatim: reason).font(.caption).foregroundStyle(BB.muted).fixedSize(horizontal: false, vertical: true)
                            .accessibilityIdentifier("captureStopReason")
                    }
                }
            }
            analysisStatus(climb)
            Toggle("这趟完攀了", isOn: $sent).tint(BB.ink).font(.subheadline)
                .onChange(of: sent) { _, value in
                    do { try store.update(climb, note: climb.note, sent: value) } catch { store.error = error.localizedDescription }
                }
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 10) { savedButtons }
                VStack(spacing: 8) { savedButtons }
            }
        }
        .padding(18).frame(maxWidth: .infinity, alignment: .leading).foregroundStyle(BB.ink)
        .background(BB.paper, in: RoundedRectangle(cornerRadius: 22)).padding(10)
    }

    @ViewBuilder
    private var savedButtons: some View {
        Button { sent = false; session.frameAgain() } label: { Text("再拍一趟").frame(maxWidth: .infinity) }
            .buttonStyle(RouteButtonStyle()).accessibilityIdentifier("captureAgain")
        Button { Task { await session.close(); dismiss() } } label: { Text("完成").frame(maxWidth: .infinity) }
            .buttonStyle(RouteButtonStyle(tone: .paper)).accessibilityIdentifier("captureDone")
    }

    @ViewBuilder
    private func analysisStatus(_ climb: Climb) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            if climb.pendingAnalysis != nil {
                Text("先存好了。姿态识别接着在这台 iPhone 上做，可以放下手机。").font(.caption).foregroundStyle(BB.muted)
                    .fixedSize(horizontal: false, vertical: true).accessibilityIdentifier("captureQueued")
                ProgressView(value: store.analysisProgress[climb.id] ?? 0).tint(BB.ink)
            } else if climb.capture?.analysisQueued == true {
                Text("姿态识别好了，正在分析这一趟…").font(.caption).foregroundStyle(BB.muted).accessibilityIdentifier("captureAnalysing")
                ProgressView().tint(BB.ink)
            } else if store.analysisResults.contains(climb.id) {
                Button { viewing = climb } label: { Text("看分析").frame(maxWidth: .infinity) }
                    .buttonStyle(RouteButtonStyle(tone: .lime)).accessibilityIdentifier("captureViewAnalysis")
            } else if let message = store.analysisMessages[climb.id] {
                Text(verbatim: message).font(.caption).foregroundStyle(BB.muted).fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("captureAnalysisMessage")
            }
        }
    }

    // MARK: Bottom: record / stop

    private var bottomBar: some View {
        HStack {
            Spacer()
            if session.phase == .recording {
                Button { Task { await session.stop(.user) } } label: {
                    ZStack {
                        Circle().stroke(BB.paper, lineWidth: 4).frame(width: 72, height: 72)
                        RoundedRectangle(cornerRadius: 7).fill(.red).frame(width: 30, height: 30)
                    }.contentShape(Circle())
                }.buttonStyle(.plain).accessibilityLabel(Text("停止并保存")).accessibilityIdentifier("captureStop")
            } else {
                Button {
                    if session.needsConfirmation { confirming = true } else { session.startRecording() }
                } label: {
                    ZStack {
                        Circle().stroke(session.assessment?.ready == true ? BB.lime : BB.paper, lineWidth: 4).frame(width: 72, height: 72)
                        Circle().fill(.red).frame(width: 58, height: 58)
                    }.contentShape(Circle())
                }.buttonStyle(.plain).disabled(!framing).opacity(framing ? 1 : 0.3)
                    .accessibilityLabel(Text("开始录")).accessibilityIdentifier("captureRecord")
            }
            Spacer()
        }.frame(height: 78)
    }
}

/// The colour chosen on the capture screen, as a swatch with its name when the key has one.
struct CaptureColourLabel: View {
    var capture: CaptureInfo?
    var body: some View {
        if let capture, capture.colorKey != nil || capture.colorRGB != nil {
            HStack(spacing: 6) {
                Circle().fill(Color(capture.colorRGB ?? [128, 128, 128])).frame(width: 12, height: 12)
                    .overlay(Circle().stroke(BB.ink.opacity(0.35), lineWidth: 1))
                if let name = CaptureText.colorName(capture.colorKey) { Text(verbatim: name).font(.caption).foregroundStyle(BB.muted) }
            }.accessibilityElement(children: .combine)
                .accessibilityLabel(Text(verbatim: CaptureText.colorName(capture.colorKey) ?? capture.colorKey ?? ""))
        }
    }
}

extension Color {
    /// A colour from 0–255 red, green and blue, as the capture page reports them.
    init(_ rgb: [Int]) {
        let parts = rgb.count == 3 ? rgb : [128, 128, 128]
        self.init(red: Double(parts[0]) / 255, green: Double(parts[1]) / 255, blue: Double(parts[2]) / 255)
    }
}
