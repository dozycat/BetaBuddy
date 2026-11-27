import SwiftUI
import AVFoundation

/// One visit to the edit page: renders the record's clip from its plan, again whenever a switch
/// changes, and plays the result in a loop. The file is temporary and goes when the page closes.
@MainActor
final class EditSession: ObservableObject {
    enum Phase: Equatable { case rendering(Double), ready(URL), failed(String) }
    @Published private(set) var phase = Phase.rendering(0)
    @Published var options = EditOptions.offered { didSet { if options != oldValue { render() } } }
    let plan: EditPlan
    let player = AVQueuePlayer()
    private let id: UUID
    private let renderer: EditRenderer
    private var looper: AVPlayerLooper?
    private var work: Task<Void, Never>?
    private var closed = false
    private var started = false
    private var generation = 0

    /// The first render, when the page comes up. Coming back from the share sheet does not start another.
    func start() {
        guard !started else { return }
        started = true
        render()
    }

    init(climb: Climb, plan: EditPlan, source: URL) {
        id = climb.id; self.plan = plan
        renderer = EditRenderer(source: source, plan: plan, directory: EditFiles.directory(for: climb.id))
        player.isMuted = true
    }

    func render() {
        guard !closed else { return }
        generation += 1
        let version = generation
        let earlier = work
        earlier?.cancel()
        looper = nil
        player.removeAllItems()
        let previous: URL? = if case .ready(let url) = phase { url } else { nil }
        phase = .rendering(0)
        let options = options
        work = Task { [renderer] in
            // The render before this one has stopped and let go of everything before this one begins.
            await earlier?.value
            guard !Task.isCancelled else { return }
            if let previous { try? FileManager.default.removeItem(at: previous.deletingLastPathComponent()) }
            do {
                let url = try await renderer.render(options: options) { value in
                    Task { @MainActor [weak self] in
                        guard let self, !self.closed, self.generation == version, case .rendering = self.phase else { return }
                        self.phase = .rendering(value)
                    }
                }
                guard !Task.isCancelled else { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()); return }
                phase = .ready(url)
                looper = AVPlayerLooper(player: player, templateItem: AVPlayerItem(url: url))
                player.play()
            } catch is CancellationError {
            } catch {
                guard !Task.isCancelled else { return }
                phase = .failed(String(localized: "没能剪出来：\(error.localizedDescription)，可以再试一次。", bundle: AppLanguage.bundle))
            }
        }
    }

    func close() {
        closed = true
        generation += 1
        work?.cancel()
        looper = nil
        player.removeAllItems()
        // After the render has let go of its files.
        let id = id, work = work
        Task { await work?.value; EditFiles.remove(for: id) }
    }
}

/// The edit of one attempt: the result, the switches the plan has data for, and sharing the file.
struct EditView: View {
    @StateObject private var session: EditSession
    @Environment(\.dismiss) private var dismiss

    init(climb: Climb, plan: EditPlan, source: URL) {
        _session = StateObject(wrappedValue: EditSession(climb: climb, plan: plan, source: source))
    }

    private var file: URL? { if case .ready(let url) = session.phase { url } else { nil } }

    var body: some View {
        VStack(spacing: 16) {
            HStack {
                Spacer()
                Button { dismiss() } label: {
                    Text(file == nil ? "取消" : "完成").font(.subheadline.bold()).padding(.horizontal, 16).frame(minHeight: 44)
                        .background(BB.paper, in: Capsule()).overlay(Capsule().stroke(BB.line, lineWidth: 1.5))
                        .contentShape(Rectangle())
                }.buttonStyle(.plain).accessibilityIdentifier("editDone")
            }
            ZStack {
                RoundedRectangle(cornerRadius: 20).fill(.black)
                switch session.phase {
                case .ready:
                    PlayerSurface(player: session.player).clipShape(RoundedRectangle(cornerRadius: 20)).accessibilityIdentifier("editResult")
                case .rendering(let value):
                    BelayProgressView(progress: value).accessibilityIdentifier("editRendering")
                case .failed(let message):
                    VStack(spacing: 14) {
                        Text(verbatim: message).font(.subheadline).multilineTextAlignment(.center).accessibilityIdentifier("editError")
                        Button("再试一次") { session.render() }.buttonStyle(RouteButtonStyle(tone: .lime)).accessibilityIdentifier("editRetry")
                    }.foregroundStyle(BB.paper).padding(24)
                }
            }.frame(maxHeight: .infinity)
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 8) { switches }
                VStack(spacing: 8) { switches }
            }
            if let file {
                ShareLink(item: file) { Text("分享视频").frame(maxWidth: .infinity) }
                    .buttonStyle(RouteButtonStyle()).accessibilityIdentifier("editShare")
            } else {
                Button {} label: { Text("分享视频").frame(maxWidth: .infinity) }
                    .buttonStyle(RouteButtonStyle()).disabled(true).opacity(0.4)
            }
            Text("自动剪辑，可能剪错。").font(.caption).foregroundStyle(BB.muted)
        }.padding(20).background(BB.paper.ignoresSafeArea()).foregroundStyle(BB.ink)
            .onAppear { session.start() }
            .onDisappear { session.close() }
    }

    @ViewBuilder
    private var switches: some View {
        if EditOptions.removalOffered && session.plan.offersRemoval { chip("消除路人", $session.options.removePassersBy, "editRemove") }
        if session.plan.offersSlowMotion { chip("慢镜头", $session.options.slowMotion, "editSlow") }
        if session.plan.offersCamera { chip("推拉镜头", $session.options.cameraMoves, "editCamera") }
    }

    private func chip(_ title: LocalizedStringKey, _ isOn: Binding<Bool>, _ identifier: String) -> some View {
        Button { isOn.wrappedValue.toggle() } label: {
            Text(title).font(.subheadline.bold()).lineLimit(1).padding(.horizontal, 14).frame(minHeight: 44)
                .frame(maxWidth: .infinity)
                .background(isOn.wrappedValue ? BB.lime : .white.opacity(0.65), in: Capsule())
                .overlay(Capsule().stroke(BB.ink.opacity(isOn.wrappedValue ? 0.05 : 0.16), lineWidth: 1))
                .contentShape(Rectangle())
        }.buttonStyle(.plain).accessibilityAddTraits(isOn.wrappedValue ? .isSelected : [])
            .accessibilityIdentifier(identifier)
    }
}
