import SwiftUI

@MainActor
final class SubjectSelectionSession: ObservableObject {
    @Published private(set) var analysis: PoseAnalysis?
    @Published private(set) var anchors: [SubjectAnchor]
    @Published private(set) var busy = false
    @Published private(set) var progress = 0.0
    @Published private(set) var error: String?
    private var work: Task<Void, Never>?
    private var generation = 0
    private let climb: Climb
    private let source: URL
    init(climb: Climb, source: URL) {
        self.climb = climb; self.source = source; anchors = climb.subjectAnchors ?? []
    }
    func prepare() {
        guard analysis == nil, !busy else { return }
        busy = true; error = nil
        work = Task {
            defer { busy = false }
            do {
                if climb.analysis.hasSubjectCandidates {
                    analysis = try await SubjectResolver().resolve(climb.analysis, anchors: anchors)
                } else {
                    analysis = try await VideoAnalyzer().analyze(url: source, quality: .balanced, subjectAnchors: anchors) { [weak self] value in
                        await self?.setProgress(value)
                    }
                }
            } catch is CancellationError { }
            catch { self.error = error.localizedDescription }
        }
    }
    private func setProgress(_ value: Double) { progress = value }
    func choose(_ candidate: SubjectCandidate, at time: Double) {
        var next = anchors.filter { abs($0.time - time) > 0.2 }
        next.append(SubjectAnchor(time: time, candidate: candidate))
        resolve(next.sorted { $0.time < $1.time })
    }
    func automatic() { resolve([]) }
    private func resolve(_ next: [SubjectAnchor]) {
        guard let analysis else { return }
        work?.cancel(); generation += 1
        let version = generation
        busy = true; error = nil
        work = Task {
            defer { if generation == version { busy = false } }
            do {
                let result = try await SubjectResolver().resolve(analysis, anchors: next)
                try Task.checkCancellation()
                guard version == generation else { return }
                self.analysis = result; anchors = next
            } catch is CancellationError { }
            catch { if version == generation { self.error = error.localizedDescription } }
        }
    }
    func stop() { generation += 1; work?.cancel() }
}

/// A thin outline keeps the original background visible. Every tap is a draft until Save;
/// revisiting another time and tapping again supplies an explicit correction after occlusion.
struct SubjectSelectionView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var session: SubjectSelectionSession
    @StateObject private var playback = Playback()
    @State private var saveError: String?
    let climb: Climb
    let store: ClimbStore
    init(climb: Climb, store: ClimbStore) {
        self.climb = climb; self.store = store
        _session = StateObject(wrappedValue: SubjectSelectionSession(climb: climb, source: store.file(climb.videoName)))
    }
    private var sample: SubjectSample? { session.analysis?.subjectSample(at: playback.time) }
    private var selectedID: String? {
        guard let sample else { return nil }
        return session.analysis?.subject?.picks.first { abs($0.time-sample.time) < 0.0001 }?.personID
    }
    var body: some View {
        NavigationStack {
            PaperPage {
                Text("点选要分析的人").font(.title2.bold())
                Text("拖到看清人物的位置，点一下圈线。遮挡后认不准时，可在新的位置再点一次纠正。")
                    .font(.subheadline).foregroundStyle(BB.muted)
                ZStack {
                    PlayerSurface(player: playback.player)
                    GeometryReader { geometry in
                        let aspect = session.analysis?.aspectRatio ?? climb.analysis.aspectRatio
                        let width = min(geometry.size.width, geometry.size.height * aspect)
                        let height = width / max(0.01, aspect)
                        let origin = CGPoint(x: (geometry.size.width-width)/2, y: (geometry.size.height-height)/2)
                        ForEach(sample?.candidates ?? []) { candidate in
                            let rect = candidate.bounds
                            Button {
                                playback.pause()
                                if let sample { session.choose(candidate, at: sample.time) }
                            } label: {
                                Ellipse().stroke(.black.opacity(0.75), lineWidth: selectedID == candidate.id ? 5 : 3)
                                    .overlay(Ellipse().stroke(selectedID == candidate.id ? BB.lime : .white.opacity(0.9), lineWidth: selectedID == candidate.id ? 3 : 1))
                                    .contentShape(Rectangle())
                            }.buttonStyle(.plain)
                                .frame(width: max(30, width*rect.width+12), height: max(44, height*rect.height+12))
                                .position(x: origin.x+width*(rect.x+rect.width/2), y: origin.y+height*(rect.y+rect.height/2))
                                .accessibilityLabel(Text("人物 \((sample?.candidates.firstIndex { $0.id == candidate.id } ?? 0) + 1)"))
                                .accessibilityAddTraits(selectedID == candidate.id ? .isSelected : [])
                                .accessibilityIdentifier("subject-\(candidate.id)")
                        }
                    }
                }.frame(height: 390).clipShape(RoundedRectangle(cornerRadius: 20))
                Slider(value: Binding(get: { playback.time }, set: { playback.pause(); playback.seek($0) }), in: 0...max(0.1, climb.analysis.duration))
                    .tint(BB.ink).accessibilityLabel("视频进度")
                HStack {
                    Button { playback.toggle(duration: climb.analysis.duration) } label: {
                        Image(systemName: playback.playing ? "pause.fill" : "play.fill").frame(width: 48, height: 44)
                    }.buttonStyle(.plain).accessibilityLabel(playback.playing ? Text("暂停") : Text("播放"))
                    Text(verbatim: clockText(playback.time)).monospacedDigit()
                    Spacer()
                    Button("自动选择") { playback.pause(); session.automatic() }
                        .disabled(session.analysis == nil || session.busy).buttonStyle(RouteButtonStyle(tone: .paper))
                }
                if session.busy {
                    if session.analysis == nil { ProgressView(value: session.progress); Text("正在找视频里的人…") }
                    else { ProgressView() }
                } else if let analysis = session.analysis {
                    if analysis.frames.isEmpty { Text("暂未选定人物，请在画面里点选。").foregroundStyle(BB.muted) }
                    else if selectedID == nil { Text("这一刻未确认人物，不会自动换成别人。").font(.caption).foregroundStyle(BB.muted) }
                    else { Text(session.anchors.isEmpty ? "自动选中，可点选其他人。" : "已按你的选择跟踪。").font(.caption).foregroundStyle(BB.muted) }
                    if (analysis.subject?.unresolved ?? 0) > 0 { Text("有选人标记未能对上，请在看清人物的位置重新点选。").font(.caption).foregroundStyle(BB.muted) }
                }
                if store.analysisResults.contains(climb.id) || store.editPlans.contains(climb.id) { Text("更换人物后，需要重新分析。").font(.caption).foregroundStyle(BB.muted) }
                if let error = saveError ?? session.error { Text(verbatim: error).foregroundStyle(.red) }
            }.navigationTitle("选择攀爬者").navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) { Button("取消") { dismiss() } }
                    ToolbarItem(placement: .confirmationAction) {
                        Button("保存") {
                            guard let analysis = session.analysis else { return }
                            do { try store.selectSubject(climb, analysis: analysis, anchors: session.anchors); dismiss() }
                            catch { saveError = error.localizedDescription }
                        }.disabled(session.busy || session.analysis?.frames.isEmpty != false)
                            .accessibilityIdentifier("saveSubject")
                    }
                }
        }.onAppear { store.setSubjectSelectionOpen(true); playback.load(store.file(climb.videoName)); session.prepare() }
            .onDisappear { playback.unload(); session.stop(); store.setSubjectSelectionOpen(false) }
    }
}
