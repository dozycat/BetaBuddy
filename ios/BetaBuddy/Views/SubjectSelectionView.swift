import SwiftUI

@MainActor
final class SubjectSelectionSession: ObservableObject {
    @Published private(set) var analysis: PoseAnalysis?
    @Published private(set) var anchors: [SubjectAnchor]
    @Published private(set) var busy = false
    @Published private(set) var progress = 0.0
    @Published private(set) var refining = false
    @Published private(set) var error: String?
    private var work: Task<Void, Never>?
    private var generation = 0
    private let climb: Climb
    private let source: URL
    private let store: ClimbStore
    init(climb: Climb, store: ClimbStore) {
        self.climb = climb; self.store = store; source = store.file(climb.videoName); anchors = climb.subjectAnchors ?? []
    }
    private enum Operation { case prepare, resolve([SubjectAnchor]), refine(SubjectBounds, Double) }
    private var lastOperation: Operation = .prepare
    private var resumeOperation: Operation?

    func prepare() { if analysis == nil && !busy { run(.prepare) } }
    func choose(_ candidate: SubjectCandidate, at time: Double) {
        let next = (anchors.filter { abs($0.time-time) > 0.2 } + [SubjectAnchor(time: time, candidate: candidate)]).sorted { $0.time < $1.time }
        run(.resolve(next))
    }
    func automatic() { run(.resolve([])) }
    func refine(_ bounds: SubjectBounds, at time: Double) { run(.refine(bounds, time)) }
    func retry() { run(lastOperation) }

    private func run(_ operation: Operation) {
        stop(); lastOperation = operation
        let version = generation
        busy = true; error = nil; progress = 0
        if case .refine = operation { refining = true }
        work = Task {
            defer { if generation == version { busy = false; refining = false } }
            do {
                var result: PoseAnalysis
                var next = anchors
                switch operation {
                case .prepare:
                    let cached = try? await store.subjectAnalysis(for: climb)
                    try Task.checkCancellation()
                    if let cached, cached.people?.contains(where: { !$0.candidates.isEmpty }) == true {
                        result = try await SubjectResolver().resolve(cached, anchors: anchors)
                        if anchors.isEmpty && result.needsSubjectRecovery {
                            result=try await VideoAnalyzer().recoverSubject(url:source,analysis:result) { [weak self] value in await self?.setProgress(value,version:version) }
                        }
                    } else {
                        do {
                            result = try await VideoAnalyzer().analyze(url: source, quality: .balanced, subjectAnchors: anchors) { [weak self] value in
                                await self?.setProgress(value, version: version)
                            }
                        } catch AnalysisError.noPose {
                            var empty = climb.analysis; empty.people = []; empty.peopleReference = nil
                            result = try await SubjectResolver().resolve(empty, anchors: anchors)
                        }
                    }
                case .resolve(let choices):
                    guard let analysis else { return }
                    result = try await SubjectResolver().resolve(analysis, anchors: choices)
                    if choices.isEmpty && result.needsSubjectRecovery {
                        result=try await VideoAnalyzer().recoverSubject(url:source,analysis:result) { [weak self] value in await self?.setProgress(value,version:version) }
                    }
                    next = choices
                case .refine(let bounds, let time):
                    let refined = try await SubjectRegionAnalyzer().refine(url: source, analysis: analysis ?? climb.analysis, bounds: bounds, at: time) { [weak self] value in
                        await self?.setProgress(value, version: version)
                    }
                    next = (anchors.filter { abs($0.time-refined.anchor.time) > 0.2 } + [refined.anchor]).sorted { $0.time < $1.time }
                    result = try await SubjectResolver().resolve(refined.analysis, anchors: next)
                }
                try Task.checkCancellation()
                guard generation == version else { return }
                analysis = result; anchors = next
            } catch is CancellationError { }
            catch { if generation == version { self.error = error.localizedDescription } }
        }
    }
    private func setProgress(_ value: Double, version: Int) { if generation == version { progress = value } }
    func stop() { generation += 1; work?.cancel(); busy = false; refining = false; resumeOperation = nil }
    func suspend() { guard busy else { return }; let operation = lastOperation; stop(); resumeOperation = operation }
    func resume() { if let operation = resumeOperation { run(operation) } }

}

/// A thin outline keeps the original background visible. Every tap is a draft until Save;
/// revisiting another time and tapping again supplies an explicit correction after occlusion.
struct SubjectSelectionView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize
    @StateObject private var session: SubjectSelectionSession
    @StateObject private var playback = Playback()
    @State private var saveError: String?
    @State private var drawingRegion = false
    @State private var adjustingRegion = false
    @State private var draft = SubjectBounds(x: 0.35, y: 0.25, width: 0.3, height: 0.5)
    let climb: Climb
    let store: ClimbStore
    init(climb: Climb, store: ClimbStore) {
        self.climb = climb; self.store = store
        _session = StateObject(wrappedValue: SubjectSelectionSession(climb: climb, store: store))
    }
    private var sample: SubjectSample? { session.analysis?.subjectSample(at: playback.time) }
    private var selectedID: String? {
        guard let sample else { return nil }
        return session.analysis?.subject?.picks.first { abs($0.time-sample.time) < 0.0001 }?.personID
    }
    var body: some View {
        NavigationStack {
            GeometryReader { layout in
                VStack(spacing: 0) {
                    subjectStage.frame(height: min(520, layout.size.height * (adjustingRegion ? 0.38 : (dynamicTypeSize.isAccessibilitySize ? 0.48 : 0.72))))
                        .padding(.horizontal,20).padding(.top,12)
                    PaperPage(spacing:16) {
                        if drawingRegion {
                            Button("微调") { adjustingRegion.toggle() }.buttonStyle(.plain).accessibilityIdentifier("adjustSubjectRegion")
                                .accessibilityAddTraits(adjustingRegion ? .isSelected : [])
                            if adjustingRegion {
                                adjustment("左右位置", value: Binding(get: { draft.x }, set: { draft.x = $0 }), range: 0...max(0.0001,1-draft.width), enabled: draft.width < 1)
                                adjustment("上下位置", value: Binding(get: { draft.y }, set: { draft.y = $0 }), range: 0...max(0.0001,1-draft.height), enabled: draft.height < 1)
                                adjustment("圈线宽度", value: Binding(get: { draft.width }, set: { draft.width = $0; draft.x = min(draft.x,1-$0) }), range: 0.02...1)
                                adjustment("圈线高度", value: Binding(get: { draft.height }, set: { draft.height = $0; draft.y = min(draft.y,1-$0) }), range: 0.02...1)
                            }
                            Button("跟踪") { playback.pause(); drawingRegion = false; session.refine(draft, at: playback.time) }
                                .buttonStyle(RouteButtonStyle(tone: .lime)).accessibilityIdentifier("trackSubjectRegion")
                        }
                        Slider(value: Binding(get: { playback.time }, set: { playback.pause(); playback.seek($0) }), in: 0...max(0.1, climb.analysis.duration))
                            .tint(BB.ink).accessibilityLabel("视频进度")
                        ViewThatFits(in: .horizontal) {
                            HStack(spacing: 12) { playbackControls; Spacer(); selectionControls }
                            VStack(spacing: 12) {
                                HStack { playbackControls; Spacer() }
                                HStack { selectionControls; Spacer() }
                            }
                        }
                        if session.busy {
                            if session.analysis == nil || session.refining { ProgressView(value: session.progress).accessibilityLabel("正在找视频里的人…") }
                            else { ProgressView() }
                        } else if let analysis = session.analysis {
                            if (analysis.subject?.unresolved ?? 0) > 0 { Text("请重新选人").font(.caption).foregroundStyle(BB.muted) }
                            else if !analysis.hasSelectedSubject { Text("未选人").foregroundStyle(BB.muted) }
                            else if selectedID == nil || analysis.subject?.picks.contains(where: { $0.personID == nil }) == true { Text("待补选").font(.caption).foregroundStyle(BB.muted) }
                        }
                        if let error = saveError ?? session.error {
                            Text(verbatim: error).foregroundStyle(.red)
                            if session.error != nil { Button("再试一次") { session.retry() }.disabled(session.busy).accessibilityIdentifier("retrySubject") }
                        }
                    }.accessibilityIdentifier("subjectControls")
                }.background(BB.paper)
            }.navigationTitle("选人").navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) { Button("取消") { dismiss() } }
                    ToolbarItem(placement: .confirmationAction) {
                        Button("保存") {
                            guard let analysis = session.analysis else { return }
                            do { try store.selectSubject(climb, analysis: analysis, anchors: session.anchors); dismiss() }
                            catch { saveError = error.localizedDescription }
                        }.disabled(drawingRegion || session.busy || session.analysis?.hasSelectedSubject != true)
                            .accessibilityIdentifier("saveSubject")
                    }
                }
        }.onAppear { store.setSubjectSelectionOpen(true); playback.load(store.file(climb.videoName)); session.prepare() }
            .onDisappear { playback.unload(); session.stop(); store.setSubjectSelectionOpen(false) }
            .onChange(of: scenePhase) { _, phase in
                if phase == .active { session.resume() }
                else { playback.pause(); session.suspend() }
            }
    }

    private var playbackControls: some View {
        HStack(spacing: 8) {
            Button { playback.toggle(duration: climb.analysis.duration) } label: {
                Image(systemName: playback.playing ? "pause.fill" : "play.fill").frame(width: 48, height: 44)
            }.buttonStyle(.plain).accessibilityLabel(playback.playing ? Text("暂停") : Text("播放"))
            Text(verbatim: clockText(playback.time)).monospacedDigit().fixedSize()
        }
    }

    private var selectionControls: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 12) { selectionButtons }.fixedSize(horizontal: true, vertical: false)
            VStack(alignment: .leading, spacing: 12) { selectionButtons }
        }
    }

    @ViewBuilder private var selectionButtons: some View {
        let automatic = !drawingRegion && session.anchors.isEmpty && session.analysis?.hasSelectedSubject == true
        let manual = drawingRegion || !session.anchors.isEmpty
        Group {
            Button("自动") { playback.pause(); session.automatic() }
                .disabled(session.analysis == nil || session.busy)
                .buttonStyle(RouteButtonStyle(tone: automatic ? .lime : .paper))
                .accessibilityIdentifier("automaticSubject")
                .accessibilityAddTraits(automatic ? .isSelected : [])
            Button(drawingRegion ? "取消圈选" : "圈选") {
                playback.pause(); session.stop(); drawingRegion.toggle(); adjustingRegion = false
                if !drawingRegion { session.prepare() }
            }.buttonStyle(RouteButtonStyle(tone: manual ? .lime : .paper))
                .accessibilityIdentifier("drawSubjectRegion")
                .accessibilityAddTraits(manual ? .isSelected : [])
        }
    }

    private var subjectStage: some View {
        ZStack {
            PlayerSurface(player: playback.player)
            GeometryReader { geometry in
                let aspect = session.analysis?.aspectRatio ?? climb.analysis.aspectRatio
                let width = min(geometry.size.width, geometry.size.height * aspect)
                let height = width / max(0.01, aspect)
                let origin = CGPoint(x: (geometry.size.width-width)/2, y: (geometry.size.height-height)/2)
                ForEach(drawingRegion ? [] : (sample?.candidates ?? [])) { candidate in
                    let rect = candidate.bounds
                    Button {
                        playback.pause()
                        if let sample { session.choose(candidate, at: sample.time) }
                    } label: {
                        Ellipse().stroke(.black.opacity(0.75), lineWidth: selectedID == candidate.id ? 5 : 3)
                            .overlay(Ellipse().stroke(selectedID == candidate.id ? BB.lime : .white.opacity(0.9), lineWidth: selectedID == candidate.id ? 3 : 1))
                            .opacity(selectedID == nil || selectedID == candidate.id ? 1 : 0)
                            .contentShape(Rectangle())
                    }.buttonStyle(.plain).disabled(session.busy)
                        .frame(width: max(44, width*rect.width+12), height: max(44, height*rect.height+12))
                        .position(x: origin.x+width*(rect.x+rect.width/2), y: origin.y+height*(rect.y+rect.height/2))
                        .zIndex(selectedID == candidate.id ? 1 : 0)
                        .accessibilityLabel(Text("人物 \((sample?.candidates.firstIndex { $0.id == candidate.id } ?? 0) + 1)"))
                        .accessibilityAddTraits(selectedID == candidate.id ? .isSelected : [])
                        .accessibilityIdentifier("subject-\(candidate.id)")
                }
                if drawingRegion {
                    Rectangle().fill(.clear).contentShape(Rectangle())
                        .gesture(DragGesture(minimumDistance: 3)
                            .onChanged { value in
                                let x = max(0, min(1, (min(value.startLocation.x,value.location.x)-origin.x)/width))
                                let y = max(0, min(1, (min(value.startLocation.y,value.location.y)-origin.y)/height))
                                let right = max(0, min(1, (max(value.startLocation.x,value.location.x)-origin.x)/width))
                                let bottom = max(0, min(1, (max(value.startLocation.y,value.location.y)-origin.y)/height))
                                guard right-x > 0.02, bottom-y > 0.02 else { return }
                                draft = .init(x: x, y: y, width: right-x, height: bottom-y)
                            })
                        .accessibilityLabel("拖动圈住人物全身")
                        .accessibilityIdentifier("subjectRegionCanvas")
                        .allowsHitTesting(!adjustingRegion)
                    Ellipse().stroke(.black.opacity(0.75), lineWidth: 5)
                        .overlay(Ellipse().stroke(BB.lime, lineWidth: 3))
                        .frame(width: width*draft.width, height: height*draft.height)
                        .position(x: origin.x+width*(draft.x+draft.width/2), y: origin.y+height*(draft.y+draft.height/2))
                        .allowsHitTesting(false).accessibilityHidden(true)
                }
            }
        }.clipShape(RoundedRectangle(cornerRadius: 20))
    }

    private func adjustment(_ title: LocalizedStringKey, value: Binding<Double>, range: ClosedRange<Double>, enabled: Bool = true) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption).foregroundStyle(BB.muted)
            Slider(value: value, in: range).tint(BB.ink).disabled(!enabled).accessibilityLabel(Text(title))
                .accessibilityValue(Text(value.wrappedValue, format: .percent.precision(.fractionLength(0))))
        }
    }
}
