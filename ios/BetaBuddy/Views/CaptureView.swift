import SwiftUI
import UIKit
import AudioToolbox

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

#if DEBUG
/// Where the picture is on the screen, for the diagnostics snapshot.
private struct CapturePictureFrame: PreferenceKey {
    static var defaultValue = CGRect.zero
    static func reduce(value: inout CGRect, nextValue: () -> CGRect) { let next = nextValue(); if next != .zero { value = next } }
}
#endif
private extension View {
    /// Debug builds note where the picture is; a Release build has nothing of this.
    @ViewBuilder func notingPictureFrame() -> some View {
        #if DEBUG
        background(GeometryReader { proxy in Color.clear.preference(key: CapturePictureFrame.self, value: proxy.frame(in: .global)) })
        #else
        self
        #endif
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
                    let suggestions = locator.suggestions()
                    Section {
                        ForEach(suggestions.ranked) { item in
                            row(item.gym, distance: CaptureText.distance(item)).accessibilityIdentifier("nearbyGym-\(item.gym.id)")
                        }
                        // Three different things: still waiting for a position, no position to be had, and a position with no gym near it.
                        if locator.fix == nil {
                            if locator.failed {
                                Text("暂时取不到位置。").foregroundStyle(BB.muted).accessibilityIdentifier("locationUnavailable")
                                Button("再试一次") { locator.retry() }
                            } else {
                                HStack(spacing: 10) { ProgressView(); Text("正在定位…").foregroundStyle(BB.muted) }.accessibilityIdentifier("locationWaiting")
                            }
                        } else if suggestions.ranked.isEmpty { Text("附近没有找到目录里的岩馆。").foregroundStyle(BB.muted) }
                    } header: { Text("离你近的") } footer: {
                        VStack(alignment: .leading, spacing: 5) {
                            Text("位置只在这台 iPhone 上用来给岩馆排序，不保存，也不离开设备。")
                            if GymDirectory.bundled?.hasCoordinates == true { Text(verbatim: GymDirectory.coordinateCredit) }
                        }
                    }
                    // Gyms the directory cannot place more exactly than their district or city: listed, never measured.
                    if !suggestions.sameCity.isEmpty {
                        Section {
                            ForEach(suggestions.sameCity) { gym in row(gym, distance: nil).accessibilityIdentifier("cityGym-\(gym.id)") }
                        } header: { Text("同城的岩馆") }
                    }
                } else if locator.access == .notAsked {
                    Section {
                        Button { locator.ask() } label: { Label("按距离找附近的岩馆", systemImage: "location") }
                            .accessibilityIdentifier("askLocation")
                    } footer: { Text("位置只在这台 iPhone 上用来给岩馆排序，不保存，也不离开设备。") }
                } else {
                    Section {
                        Text("没有位置权限。在系统设置里允许 BetaBuddy 使用位置，就能按距离找岩馆。").foregroundStyle(BB.muted)
                            .accessibilityIdentifier("locationRefused")
                        Button("打开系统设置") {
                            if let url = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(url) }
                        }
                    }
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

    private func row(_ gym: DirectoryGym, distance: String?) -> some View {
        Button { gymId = gym.id; gymName = gym.name; dismiss() } label: {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(verbatim: gym.name).foregroundStyle(BB.ink)
                    Text(verbatim: gym.location).font(.caption).foregroundStyle(BB.muted)
                }
                Spacer()
                if let distance { Text(verbatim: distance).font(.caption.monospacedDigit()).foregroundStyle(BB.muted) }
                if gymId == gym.id { Image(systemName: "checkmark").foregroundStyle(BB.ink) }
            }.padding(.vertical, 3)
        }
    }
}

struct CaptureView: View {
    @EnvironmentObject private var store: ClimbStore
    @Environment(\.dismiss) private var dismiss
    @StateObject private var session: CaptureSession
    @StateObject private var locator = GymLocator()
    @State private var choosingGym = false
    @State private var choosingDiscipline = false
    @State private var confirming = false
    /// What a tap on the picture marks. Starts on the start holds: those are what the page gets wrong most.
    @State private var tool = CaptureRole.startHand
    @State private var sent = false
    @State private var viewing: Climb?
    /// The gym the user declined when it was offered as the one they are standing in.
    @State private var declined: String?
    @State private var confirmingDiscard = false
    /// Leaves the camera for the photo library import.
    private let onImport: () -> Void

    /// `source` is only asked for once, when the screen first appears: the view itself is rebuilt
    /// whenever the album behind it changes, and a camera must not be set up each time.
    init(store: ClimbStore, pack: GymPack? = GymPack.current, source: @escaping @autoclosure () -> CaptureSource = CaptureSession.makeSource(), onImport: @escaping () -> Void) {
        _session = StateObject(wrappedValue: CaptureSession(store: store, pack: pack, source: source()))
        self.onImport = onImport
    }

    private var saved: Climb? {
        guard case .saved(let id) = session.phase else { return nil }
        return store.climbs.first { $0.id == id }
    }
    private var framing: Bool { session.phase == .framing && session.camera == .running }
    /// What the screen offers about the gym without being asked. It is only ever an offer: the
    /// selection, which starts as the last record's gym, never changes by itself.
    private enum GymOffer {
        /// A gym right where the phone is, other than the one selected.
        case here(NearbyGym)
        /// No gym is selected yet: the nearest one whose own position is known…
        case nearest(NearbyGym)
        /// …or the nearest one known only by its station, or with only a rough position: a guess, and worded as one.
        case guess(NearbyGym)
        /// No gym is selected and the position has never been asked for.
        case ask
    }
    private var offered: GymOffer? {
        guard framing else { return nil }
        let suggestions = locator.suggestions()
        if let here = suggestions.here, here.gym.id != session.gymId, here.gym.id != declined { return .here(here) }
        guard session.gymId == nil, session.gymName.isEmpty else { return nil }
        if locator.access == .notAsked { return declined == "ask" ? nil : .ask }
        guard let first = suggestions.ranked.first, first.gym.id != declined else { return nil }
        return first.shown == .exact ? .nearest(first) : .guess(first)
    }

    // MARK: What the controls do. A debug build's autopilot goes through these same functions.

    private func choose(_ gym: DirectoryGym) { session.gymId = gym.id; session.gymName = gym.name }
    private func pick(colour key: String) { tool = .startHand; session.selectColor(key) }
    /// One tap on the picture, in the picture's own 0–1 coordinates.
    private func tapPicture(_ point: CGPoint) {
        guard session.page == .ready, session.color != nil else { return }
        session.correct(at: point, role: tool)
    }
    private func pressRecord() { if session.needsConfirmation { confirming = true } else { session.startRecording() } }
    private func recordAnyway() { session.startRecording() }
    private func pressStop() { Task { await session.stop(.user) } }

    #if DEBUG
    @State private var pictureFrame = CGRect.zero

    /// Debug builds leave a picture of this screen and what the location service said in `Documents/capture-diagnostics/`.
    var body: some View {
        screen
            .onPreferenceChange(CapturePictureFrame.self) { pictureFrame = $0 }
            .task { await snapshots() }
            .task {
                // `-debugCaptureAutopilot YES`: nobody is there to tap.
                await CaptureAutopilot(session: session, store: store, screen: CaptureAutopilot.Screen(
                    chooseGym: choose, pickColour: { pick(colour: $0) }, chooseTool: { tool = $0 }, tapPicture: tapPicture,
                    pressRecord: pressRecord, asksToConfirm: { confirming }, recordAnyway: { confirming = false; recordAnyway() },
                    pressStop: pressStop, shot: { snapshot(as: $0) })).run()
            }
            .onAppear { session.diagnosticsExtra = { [locator] in ["location": locator.diagnostics] } }
            .onChange(of: locator.access) { session.mark("location", locator.diagnostics) }
            .onChange(of: locator.fix) { session.mark("location", locator.diagnostics) }
            .onChange(of: locator.failed) { session.mark("location", locator.diagnostics) }
    }

    private func snapshots() async {
        var wait = 1.0      // The first one soon, then every three seconds.
        while !Task.isCancelled {
            try? await Task.sleep(for: .seconds(wait))
            wait = 3
            if !Task.isCancelled { snapshot() }
        }
    }

    /// The screen as it is now, as `preview-latest.png` and, for a moment worth keeping, under `name` as well.
    private func snapshot(as name: String? = nil) {
        guard let window = UIApplication.shared.connectedScenes.compactMap({ $0 as? UIWindowScene }).flatMap(\.windows).first(where: \.isKeyWindow) else { return }
        // With the gym list or the analysis page on top there is no preview to look for.
        let showsPreview = viewing == nil && !choosingGym && (session.camera == .running || session.camera == .starting)
        let frame = session.diagnosticsFrame
        guard let shot = CaptureSnapshot.take(window: window, picture: showsPreview ? pictureFrame : nil, frame: frame?.jpeg) else { return }
        let offer: String? = switch offered {
        case .here(let gym): "here:\(gym.gym.id)"
        case .nearest(let gym): "nearest:\(gym.gym.id)"
        case .guess(let gym): "guess:\(gym.gym.id)"
        case .ask: "ask"
        case nil: nil
        }
        session.diagnosticsPreview(shot.image, source: shot.source.rawValue, fields: [
            "picture": [pictureFrame.minX, pictureFrame.minY, pictureFrame.width, pictureFrame.height].map(Double.init),
            "screen": [Double(window.bounds.width), Double(window.bounds.height)],
            "pictureDark": shot.dark as Any, "frameAge": (shot.source == .composite ? frame?.age : nil) as Any,
            "covered": !showsPreview, "offer": offer as Any, "tool": tool.rawValue,
        ], alsoAs: name)
    }
    #else
    var body: some View { screen }
    #endif

    private var recording: Bool { session.phase == .recording }
    /// The latest record, shown on the album button.
    private var latest: Climb? { store.climbs.first }

    private var screen: some View {
        ZStack {
            // The capture page has nothing to show; it only has to stay alive and keep answering. It is
            // kept in the window at full size, underneath everything, like a page nobody happens to see.
            if let webView = session.webView {
                GymWebView(webView: webView).allowsHitTesting(false).accessibilityHidden(true)
            }
            Color.black.ignoresSafeArea()
            // The whole picture, as large as the screen allows. Every control floats over its edges;
            // nothing is drawn over the middle, where the climber is.
            picture.ignoresSafeArea()
            LinearGradient(stops: [.init(color: .black.opacity(0.55), location: 0), .init(color: .clear, location: 0.2),
                                   .init(color: .clear, location: 0.6), .init(color: .black.opacity(0.75), location: 1)],
                           startPoint: .top, endPoint: .bottom)
                .ignoresSafeArea().allowsHitTesting(false)
            VStack(spacing: 8) {
                topBar
                Spacer(minLength: 0)
                card
                Spacer(minLength: 0)
                if session.phase == .framing || recording { bottom }
            }
            .padding(.horizontal, 14).padding(.top, 4).padding(.bottom, 6)
            if choosingDiscipline {
                // A tap anywhere else closes the list.
                Color.black.opacity(0.001).ignoresSafeArea()
                    .onTapGesture { withAnimation(.snappy(duration: 0.2)) { choosingDiscipline = false } }
                disciplineList
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
                    .padding(.top, 60).padding(.trailing, 14)
            }
        }
        .foregroundStyle(.white)
        .statusBarHidden(true)
        .onAppear {
            UIApplication.shared.isIdleTimerDisabled = true   // The phone stands there while the user climbs.
            session.start()
            locator.locateIfAllowed()
            pickNearest()
        }
        .onChange(of: locator.fix) { pickNearest() }
        .onDisappear {
            // Not when the analysis page or the gym list merely covers this screen.
            guard viewing == nil, !choosingGym else { return }
            UIApplication.shared.isIdleTimerDisabled = false
            Task { await session.close() }
        }
        .sheet(isPresented: $choosingGym) { CaptureGymSheet(gymId: $session.gymId, gymName: $session.gymName, locator: locator) }
        .fullScreenCover(item: $viewing) { climb in
            if let pack = GymPack.current, pack.supportsAnalysis { AnalysisView(pack: pack, climb: climb, store: store) { _ in viewing = nil } }
        }
        .confirmationDialog("从这个机位拍，多半分析不了", isPresented: $confirming, titleVisibility: .visible) {
            Button("仍然开始录") { recordAnyway() }.accessibilityIdentifier("recordAnyway")
            Button("再调一下机位", role: .cancel) {}
        } message: { Text(verbatim: session.assessment?.hint.map(CaptureText.line) ?? "") }
        .confirmationDialog("这一趟还没存进攀爬册", isPresented: $confirmingDiscard, titleVisibility: .visible) {
            Button("再存一次") { Task { await session.save() } }
            Button("放弃这段录像", role: .destructive) { Task { session.discardHeldClip(); await session.close(); dismiss() } }
                .accessibilityIdentifier("discardAndClose")
            Button("取消", role: .cancel) {}
        }
        .alert(Text(verbatim: session.notice.map(CaptureText.notice) ?? ""), isPresented: Binding(get: { session.notice != nil }, set: { if !$0 { session.notice = nil } })) {
            Button("知道了", role: .cancel) { session.notice = nil }
        }
    }

    // MARK: Top: close, gym, discipline; while recording, the clock with what the take is labelled

    private func close() {
        // A clip that has not made it into the album is not dropped by closing the screen without a word.
        if case .saveFailed = session.phase { confirmingDiscard = true } else { Task { await session.close(); dismiss() } }
    }

    /// How far the selected gym is, when the directory knows where it is and the phone where it is.
    private var gymDistance: (text: String, nearest: Bool)? {
        guard let id = session.gymId else { return nil }
        let ranked = locator.suggestions().ranked
        guard let item = ranked.first(where: { $0.gym.id == id }), let text = CaptureText.distance(item) else { return nil }
        return (text, ranked.first?.gym.id == id)
    }

    /// With no gym chosen yet and the position already allowed, the nearest gym whose own position
    /// the directory knows is selected. A gym already chosen is never replaced.
    private func pickNearest() {
        guard session.phase == .framing, session.gymId == nil, session.gymName.isEmpty,
              let first = locator.suggestions().ranked.first, first.shown == .exact else { return }
        choose(first.gym)
    }

    private var topBar: some View {
        HStack(spacing: 8) {
            Button { close() } label: {
                Image(systemName: "xmark").font(.system(size: 14, weight: .bold)).frame(width: 38, height: 38)
                    .background(.black.opacity(0.5), in: Circle()).overlay(Circle().stroke(.white.opacity(0.12)))
            }.buttonStyle(.plain).accessibilityLabel(Text("关闭")).accessibilityIdentifier("captureClose")
                .disabled(recording || session.phase == .saving)
                .opacity(recording ? 0.25 : session.phase == .saving ? 0.3 : 1)
            if recording, let began = session.recordingStarted {
                Spacer(minLength: 0)
                TimelineView(.periodic(from: began, by: 0.5)) { context in
                    HStack(spacing: 8) {
                        Circle().fill(Self.recordRed).frame(width: 9, height: 9)
                        Text(verbatim: clockText(context.date.timeIntervalSince(began))).font(.system(.headline, design: .monospaced).bold()).monospacedDigit()
                            .accessibilityLabel(Text("正在录")).accessibilityValue(Text(verbatim: clockText(context.date.timeIntervalSince(began))))
                            .accessibilityIdentifier("captureElapsed")
                        Text(verbatim: takeLabel).font(.caption).foregroundStyle(.white.opacity(0.75)).lineLimit(1)
                    }.padding(.horizontal, 14).frame(minHeight: 38).background(.black.opacity(0.5), in: Capsule())
                }
                Spacer(minLength: 0)
                Color.clear.frame(width: 38, height: 38)
            } else {
                Button { choosingGym = true } label: {
                    HStack(spacing: 7) {
                        Image(systemName: "mappin.and.ellipse").font(.system(size: 17, weight: .semibold))
                        VStack(alignment: .leading, spacing: 0) {
                            if let distance = gymDistance {
                                Group { if distance.nearest { Text("最近 · \(distance.text)") } else { Text(verbatim: distance.text) } }
                                    .font(.caption2).foregroundStyle(.white.opacity(0.7))
                            }
                            Group { if session.gymName.isEmpty { Text("选择岩馆") } else { Text(verbatim: session.gymName) } }
                                .font(.subheadline.bold())
                        }.lineLimit(1)
                        Image(systemName: "chevron.down").font(.caption2.bold()).opacity(0.7)
                    }.padding(.horizontal, 14).frame(minHeight: 48).background(.black.opacity(0.5), in: Capsule())
                        .overlay(Capsule().stroke(.white.opacity(0.12)))
                }.buttonStyle(.plain).disabled(session.phase != .framing).accessibilityIdentifier("captureGym")
                    .layoutPriority(-1)
                Spacer(minLength: 0)
                disciplineMenu
            }
        }
    }

    /// Automatic until the user picks one. “Automatic” only means the analysis page will guess after
    /// the take; nothing on this screen recognises the discipline, and the menu says so.
    private var disciplineMenu: some View {
        Button { withAnimation(.snappy(duration: 0.2)) { choosingDiscipline.toggle() } } label: {
            HStack(spacing: 8) {
                Group { if session.discipline == nil { Text(verbatim: "AUTO") } else { Text("手动") } }
                    .font(.system(size: 11, weight: .heavy)).tracking(session.discipline == nil ? 1.2 : 0)
                    .padding(.horizontal, 10).frame(minHeight: 32)
                    .foregroundStyle(session.discipline == nil ? BB.ink : .white)
                    .background(session.discipline == nil ? Self.accent : .white.opacity(0.18), in: Capsule())
                Group { if let kind = session.discipline { Text(Self.name(kind)) } else { Text("自动判断") } }
                    .font(.subheadline.bold()).lineLimit(1).fixedSize()
            }.padding(.leading, 7).padding(.trailing, 14).frame(minHeight: 48)
                .background(.black.opacity(0.5), in: Capsule()).overlay(Capsule().stroke(.white.opacity(0.12)))
        }
        .buttonStyle(.plain)
        .disabled(session.phase != .framing)
        .accessibilityLabel(Text("攀爬方式"))
        .accessibilityValue(session.discipline.map { Text(Self.name($0)) } ?? Text("自动判断"))
        .accessibilityIdentifier("captureDiscipline")
    }

    /// Drops down under the discipline pill. Automatic first, with what it means.
    private var disciplineList: some View {
        VStack(alignment: .leading, spacing: 2) {
            disciplineRow(nil) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("自动判断").font(.subheadline.bold())
                    Text("录完由分析来认，可能认错").font(.caption2).foregroundStyle(.white.opacity(0.6)).fixedSize(horizontal: false, vertical: true)
                }
            }
            Divider().overlay(.white.opacity(0.15)).padding(.vertical, 2)
            ForEach(Discipline.allCases, id: \.self) { kind in
                disciplineRow(kind) {
                    HStack(spacing: 6) {
                        Text(Self.name(kind)).font(.subheadline.bold())
                        Group { if kind == .boulder { Text("V 等级") } else { Text(verbatim: "YDS") } }
                            .font(.caption2).foregroundStyle(.white.opacity(0.5))
                    }
                }
            }
        }
        .padding(6).frame(width: 220)
        .background(Color(white: 0.11).opacity(0.96), in: RoundedRectangle(cornerRadius: 18))
        .overlay(RoundedRectangle(cornerRadius: 18).stroke(.white.opacity(0.1)))
        .shadow(color: .black.opacity(0.5), radius: 18, y: 10)
        .transition(.scale(scale: 0.94, anchor: .topTrailing).combined(with: .opacity))
    }

    private func disciplineRow(_ kind: Discipline?, @ViewBuilder label: () -> some View) -> some View {
        Button {
            session.pickDiscipline(kind)
            withAnimation(.snappy(duration: 0.2)) { choosingDiscipline = false }
        } label: {
            HStack {
                label()
                Spacer(minLength: 8)
                if session.discipline == kind { Image(systemName: "checkmark").font(.footnote.bold()).foregroundStyle(Self.accent) }
            }
            .padding(.horizontal, 12).frame(minHeight: 44).contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(session.discipline == kind ? [.isSelected] : [])
        .accessibilityIdentifier("discipline-\(kind?.rawValue ?? "auto")")
    }

    private static func name(_ kind: Discipline) -> LocalizedStringKey {
        switch kind {
        case .boulder: "抱石"
        case .toprope: "顶绳"
        case .lead: "先锋"
        }
    }
    private static func nameText(_ kind: Discipline) -> String {
        switch kind {
        case .boulder: String(localized: "抱石", bundle: AppLanguage.bundle)
        case .toprope: String(localized: "顶绳", bundle: AppLanguage.bundle)
        case .lead: String(localized: "先锋", bundle: AppLanguage.bundle)
        }
    }
    /// What the take is labelled with, on the recording clock.
    private var takeLabel: String {
        [session.discipline.map(Self.nameText) ?? "", gradeText(session.grade), session.gymName].filter { !$0.isEmpty }.joined(separator: " · ")
    }

    // MARK: Middle: the picture, and what replaces the controls after a take

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
                        .accessibilityElement().accessibilityLabel(Text("相机画面")).accessibilityIdentifier("capturePicture")
                    #if DEBUG
                    // Numbers for a first run on a phone. Debug builds only; never part of the release interface.
                    if session.showsDebug {
                        TimelineView(.periodic(from: .now, by: 1)) { _ in
                            Text(verbatim: session.debugText).font(.system(size: 9, weight: .semibold, design: .monospaced))
                                .foregroundStyle(.white).padding(5).background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 6))
                                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading).padding(6)
                        }.allowsHitTesting(false).accessibilityIdentifier("captureDebug")
                    }
                    #endif
                }
            }
            .frame(width: size.width, height: size.height)
            .notingPictureFrame()
            .position(x: proxy.size.width / 2, y: proxy.size.height / 2)
        }
    }

    @ViewBuilder
    private var card: some View {
        if session.camera == .starting || session.camera == .running {
            if session.phase == .saving {
                VStack(spacing: 14) { ProgressView().tint(BB.ink); Text("正在保存…").font(.headline) }
                    .padding(24).foregroundStyle(BB.ink).background(BB.paper, in: RoundedRectangle(cornerRadius: 22))
            } else if case .saveFailed(let message) = session.phase {
                saveFailed(message)
            } else if let saved {
                savedCard(saved)
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

    // MARK: Bottom: the grade wheel, then album and shutter

    static let accent = Color(red: 1, green: 0.48, blue: 0.24)
    static let recordRed = Color(red: 1, green: 0.35, blue: 0.21)

    private var bottom: some View {
        VStack(spacing: 10) {
            if !recording {
                VStack(spacing: 4) {
                    HStack(spacing: 6) {
                        Text("难度")
                        Text(verbatim: "·").opacity(0.6)
                        if (session.discipline ?? .boulder) == .boulder { Text("V 等级") } else { Text(verbatim: "YDS") }
                        // The grade was filled in from the problem the page recognised, not picked.
                        if session.gradePrefilled {
                            Image(systemName: "wand.and.stars").foregroundStyle(BB.lime).accessibilityIdentifier("captureGradePrefilled")
                        }
                    }.font(.caption).foregroundStyle(.white.opacity(0.7)).accessibilityHidden(true)
                    Capsule().fill(Self.accent).frame(width: 2, height: 8).accessibilityHidden(true)
                    GradeWheel(grades: [ungradedValue] + GradeScale.grades(for: session.discipline), selection: session.grade, accent: Self.accent) {
                        session.pickGrade($0)
                    }
                    .frame(height: 52)
                    .padding(.horizontal, -14)
                }
                .disabled(!framing)
            }
            HStack {
                albumButton.opacity(recording ? 0 : 1).disabled(recording)
                Spacer()
                shutter
                Spacer()
                Color.clear.frame(width: 54, height: 54)
            }.padding(.horizontal, 18)
        }
    }

    /// The latest record, with its grade: a way back to the album.
    private var albumButton: some View {
        Button { Task { await session.close(); dismiss() } } label: {
            ZStack {
                if let latest, let image = UIImage(contentsOfFile: store.file(latest.thumbnailName).path) {
                    Image(uiImage: image).resizable().scaledToFill().frame(width: 54, height: 54).clipped().overlay(Color.black.opacity(0.35))
                    if latest.grade != ungradedValue {
                        Text(verbatim: latest.grade).font(.system(size: 15, weight: .heavy)).monospacedDigit().minimumScaleFactor(0.6).padding(3)
                    }
                } else {
                    Image(systemName: "photo").font(.system(size: 20, weight: .semibold))
                }
            }
            .frame(width: 54, height: 54).background(.white.opacity(0.08))
            .clipShape(RoundedRectangle(cornerRadius: 13))
            .overlay(RoundedRectangle(cornerRadius: 13).stroke(.white.opacity(0.85), lineWidth: 2))
        }.buttonStyle(.plain).accessibilityLabel(Text("攀爬册")).accessibilityIdentifier("captureAlbum")
    }

    /// The camera-position check shows only on the record button's ring: yellow-green when the page
    /// finds nothing in the way, orange when something would keep the analysis from working (pressing
    /// record then asks first, with the reason). Read out in full as the button's value.
    private var ring: Color {
        if recording { return Self.recordRed }
        guard framing, session.page == .ready, let assessment = session.assessment else { return .white.opacity(0.9) }
        if assessment.ready { return BB.lime }
        return assessment.hint?.blocking == true ? Self.accent : .white.opacity(0.9)
    }
    private var positionNote: Text {
        switch session.page {
        case .unavailable: return Text("这个版本没有机位检查和岩点识别，照常可以录。")
        case .failed: return Text("机位检查暂时用不了，照常可以录。")
        case .loading: return Text("正在看画面…")
        case .ready:
            guard let assessment = session.assessment else { return Text("正在看画面…") }
            if assessment.ready { return Text("机位可以，开始吧") }
            if let hint = assessment.hint { return Text(verbatim: CaptureText.line(hint)) }
            return Text("放稳手机，马上就好")
        }
    }

    /// A white jug with its bolt hole; while recording it turns red and the hole becomes a stop square.
    private var shutter: some View {
        Button { if recording { pressStop() } else { pressRecord() } } label: {
            ZStack {
                if recording {
                    TimelineView(.animation) { context in
                        let phase = (sin(context.date.timeIntervalSinceReferenceDate * .pi / 0.9) + 1) / 2
                        Circle().stroke(Self.recordRed, lineWidth: 2).frame(width: 92, height: 92)
                            .scaleEffect(0.96 + 0.08 * phase).opacity(0.15 + 0.6 * phase)
                    }
                }
                Circle().stroke(ring, lineWidth: 3)
                    .frame(width: 80, height: 80)
                JugShape().fill(recording ? Self.recordRed : Color(red: 0.96, green: 0.95, blue: 0.92))
                    .frame(width: 60, height: 60).shadow(color: .black.opacity(0.35), radius: 4, y: 2)
                if recording { RoundedRectangle(cornerRadius: 3).fill(.white).frame(width: 13, height: 13) }
                else { Circle().fill(BB.ink).frame(width: 12, height: 12) }
            }
            .frame(width: 92, height: 92).contentShape(Circle())
            .animation(.easeOut(duration: 0.2), value: recording)
        }
        .buttonStyle(.plain)
        .disabled(!(framing || recording)).opacity(framing || recording ? 1 : 0.3)
        .accessibilityLabel(recording ? Text("停止并保存") : Text("开始录"))
        .accessibilityValue(recording ? Text(verbatim: "") : positionNote)
        .accessibilityIdentifier(recording ? "captureStop" : "captureRecord")
    }
}

/// A jug hold, drawn in a 64 × 64 box.
private struct JugShape: Shape {
    func path(in rect: CGRect) -> Path {
        let s = min(rect.width, rect.height) / 64
        func p(_ x: CGFloat, _ y: CGFloat) -> CGPoint { CGPoint(x: rect.minX + x * s, y: rect.minY + y * s) }
        var path = Path()
        path.move(to: p(8, 33))
        path.addCurve(to: p(32, 6), control1: p(7, 18), control2: p(17, 7))
        path.addCurve(to: p(57, 25), control1: p(45, 5), control2: p(56, 13))
        path.addCurve(to: p(56, 39), control1: p(57.6, 31), control2: p(55.4, 34))
        path.addCurve(to: p(35, 58), control1: p(57, 48), control2: p(49, 57))
        path.addCurve(to: p(8, 33), control1: p(19, 59), control2: p(9, 47))
        path.closeSubpath()
        return path
    }
}

/// The grades as a turning drum: the one in the middle is chosen; the further from it, the smaller
/// and fainter. Drag it, or tap a grade to turn to it. Each step clicks and taps the hand.
private struct GradeWheel: View {
    var grades: [String]
    var selection: String
    var accent: Color
    var pick: (String) -> Void
    /// How far the drum is being dragged, in points; zero at rest.
    @State private var drag: CGFloat = 0
    /// The grade last clicked past, so each step sounds once.
    @State private var ticked: Int?
    @State private var haptic = UISelectionFeedbackGenerator()
    private let width: CGFloat = 78

    private var index: Int { grades.firstIndex(of: selection) ?? 0 }
    private func clamp(_ value: Int) -> Int { min(max(value, 0), grades.count - 1) }
    /// Where the middle of the drum is, in grades.
    private func position(_ drag: CGFloat) -> CGFloat { CGFloat(index) - drag / width }

    private func tick(_ value: Int) {
        guard value != ticked else { return }
        ticked = value
        AudioServicesPlaySystemSound(1104)
        haptic.selectionChanged()
    }
    private func choose(_ value: Int) {
        let value = clamp(value)
        tick(value)
        withAnimation(.snappy(duration: 0.3)) { drag = 0; pick(grades[value]) }
    }

    var body: some View {
        let middle = position(drag)
        let chosen = clamp(Int(middle.rounded()))
        ZStack {
            ForEach(Array(grades.enumerated()), id: \.element) { offset, grade in
                let distance = CGFloat(offset) - middle
                if abs(distance) < 4.5 {
                    let far = min(abs(distance), 3)
                    Text(verbatim: gradeText(grade))
                        .font(.system(size: offset == chosen ? 28 : 21, weight: offset == chosen ? .heavy : .bold)).monospacedDigit()
                        .minimumScaleFactor(0.5).lineLimit(1)
                        .foregroundStyle(offset == chosen ? accent : .white)
                        .frame(width: width, height: 52)
                        .contentShape(Rectangle())
                        .onTapGesture { choose(offset) }
                        .rotation3DEffect(.degrees(Double(-distance) * 20), axis: (x: 0, y: 1, z: 0), perspective: 0.5)
                        .scaleEffect(1 - far * 0.12)
                        .opacity(1 - far * 0.3)
                        .offset(x: distance * width * (1 - far * 0.05))
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .contentShape(Rectangle())
        .gesture(DragGesture(minimumDistance: 4)
            .onChanged { value in
                var moved = value.translation.width
                // Past either end the drum gives a little and no more.
                let low = CGFloat(index) * width, high = CGFloat(index - grades.count + 1) * width
                if moved > low { moved = low + (moved - low) * 0.3 }
                if moved < high { moved = high + (moved - high) * 0.3 }
                drag = moved
                tick(clamp(Int(position(moved).rounded())))
            }
            .onEnded { value in
                let fling = max(-4 * width, min(4 * width, value.predictedEndTranslation.width - value.translation.width))
                choose(Int(position(drag + fling).rounded()))
            })
        .onAppear { ticked = index; haptic.prepare() }
        .onChange(of: selection) { _, _ in ticked = index }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text("线路难度"))
        .accessibilityValue(Text(verbatim: gradeText(selection)))
        .accessibilityAdjustableAction { direction in
            choose(direction == .increment ? index + 1 : index - 1)
        }
        .accessibilityIdentifier("captureGrade")
    }
}

/// The colour chosen on the capture screen, as the swatch the page reported. No name: the page's
/// names for colours are not reliable enough to print.
struct CaptureColourLabel: View {
    var capture: CaptureInfo?
    var body: some View {
        if let rgb = capture?.colorRGB {
            Circle().fill(Color(rgb)).frame(width: 12, height: 12)
                .overlay(Circle().stroke(BB.ink.opacity(0.35), lineWidth: 1))
                .accessibilityElement().accessibilityLabel(Text("拍摄时选的颜色"))
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
