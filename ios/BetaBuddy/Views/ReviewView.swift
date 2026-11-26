import SwiftUI
import AVFoundation

@MainActor
final class Playback: ObservableObject {
    let player = AVPlayer()
    @Published var time = 0.0
    @Published var playing = false
    @Published var failure: String?
    private var observer: Any?
    private var statusObserver: NSKeyValueObservation?
    var rate: Float = 1
    func load(_ url: URL) {
        guard player.currentItem == nil else { return }
        guard FileManager.default.fileExists(atPath:url.path) else { failure = String(localized: "原片文件不存在，仍可查看分析与笔记。", bundle: AppLanguage.bundle); return }
        let item = AVPlayerItem(url:url)
        player.replaceCurrentItem(with:item)
        statusObserver = item.observe(\.status,options:[.new]) { [weak self] item, _ in
            if item.status == .failed { Task { @MainActor in self?.failure = String(localized: "原片无法播放，请检查视频文件。", bundle: AppLanguage.bundle) } }
        }
        observer = player.addPeriodicTimeObserver(forInterval:CMTime(seconds:0.05,preferredTimescale:600),queue:.main) { [weak self] value in
            MainActor.assumeIsolated {
                self?.time = max(0,value.seconds.isFinite ? value.seconds : 0)
                self?.playing = self?.player.timeControlStatus == .playing
            }
        }
    }
    func toggle(duration:Double) {
        if playing { player.pause(); playing = false }
        else {
            if time >= duration-0.1 { seek(0) }
            player.playImmediately(atRate:rate); playing = true
        }
    }
    func seek(_ value:Double) { time = max(0,value); player.seek(to:CMTime(seconds:time,preferredTimescale:600),toleranceBefore:.zero,toleranceAfter:.zero) }
    func pause() { player.pause(); playing = false }
    func unload() {
        pause()
        if let observer { player.removeTimeObserver(observer); self.observer = nil }
        statusObserver = nil
        player.replaceCurrentItem(with:nil)
    }
}

struct PlayerSurface: UIViewRepresentable {
    let player: AVPlayer
    final class Surface: UIView {
        override class var layerClass: AnyClass { AVPlayerLayer.self }
        var playerLayer: AVPlayerLayer { layer as! AVPlayerLayer }
    }
    func makeUIView(context:Context) -> Surface {
        let view = Surface(); view.playerLayer.player = player; view.playerLayer.videoGravity = .resizeAspect
        view.backgroundColor = .black
        return view
    }
    func updateUIView(_ view:Surface,context:Context) { view.playerLayer.player = player }
}

struct ReviewView: View {
    @EnvironmentObject private var store: ClimbStore
    @EnvironmentObject private var analysisQueue: AnalysisQueue
    @Environment(\.dismiss) private var dismiss
    @StateObject private var playback = Playback()
    @State private var slow = false
    @State private var sent = false
    @State private var confirmDelete = false
    @State private var organizing = false
    @State private var analysing = false
    /// Pose recognition was started from “分析这一趟”; open the analysis page when it finishes.
    @State private var openAfterPose = false
    @State private var seekAfterAnalysis: Double?
    /// The render plan that came with the saved analysis, when this build can render it.
    @State private var editPlan: EditPlan?
    @State private var editing = false
    var climb: Climb
    private var current: Climb { store.climbs.first { $0.id == climb.id } ?? climb }
    /// Shown only when the installed pack carries the analysis page.
    private var analysisPack: GymPack? { GymPack.current.flatMap { $0.supportsAnalysis ? $0 : nil } }
    private var report: String {
        let status = current.sent ? String(localized: "已完攀", bundle: AppLanguage.bundle) : String(localized: "还在尝试", bundle: AppLanguage.bundle)
        let head = "BetaBuddy · \(current.displayTitle)\n\(gradeText(current.grade)) · \(status) · \(clockText(current.analysis.duration))"
        return current.note.isEmpty ? head : head + "\n" + current.note
    }
    var body: some View {
        PaperPage {
            VStack(alignment: .leading, spacing: 8) {
                Text(verbatim: current.displayTitle).font(.title.bold())
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 12) { climbMetadata }.fixedSize(horizontal: true, vertical: false)
                    VStack(alignment: .leading, spacing: 8) { climbMetadata }
                }
                if !current.displayGymName.isEmpty {
                    Text(verbatim: current.displayGymName).font(.caption).foregroundStyle(BB.muted)
                }
                CaptureColourLabel(capture: current.capture)
            }
            if let failure = playback.failure {
                Label { Text(verbatim: failure) } icon: { Image(systemName: "exclamationmark.circle") }
                    .font(.subheadline).foregroundStyle(.red)
            } else {
                replay
            }
            Toggle("这趟完攀了", isOn: $sent).tint(BB.ink).accessibilityIdentifier("sentToggle")
            if analysisPack != nil { analysisEntry }
            if current.pendingAnalysis != nil {
                VStack(alignment: .leading, spacing: 12) {
                    Text(store.analysisProgress[climb.id] == nil ? "等待分析…" : "正在分析，可离开此页面。")
                    ProgressView(value: store.analysisProgress[climb.id] ?? 0)
                    Button("停止重新分析") { store.cancelAnalysis(current) }
                        .buttonStyle(RouteButtonStyle(tone: .paper))
                }
            }
            if let message = store.analysisMessages[climb.id] {
                Text(verbatim: message).font(.caption).foregroundStyle(BB.muted)
            }
        }.navigationTitle("动作回看").navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        Button { playback.pause(); organizing = true } label: {
                            Label("整理这一趟", systemImage: "folder")
                        }.accessibilityIdentifier("organizeClimb")
                        ShareLink(item: report) { Label("分享复盘文字", systemImage: "square.and.arrow.up") }
                        Button("删除这一趟", role: .destructive) { confirmDelete = true }
                            .accessibilityIdentifier("deleteClimb")
                    } label: {
                        Image(systemName: "ellipsis").frame(width: 44, height: 44)
                    }.accessibilityLabel("更多操作")
                }
            }
            .sheet(isPresented: $organizing) { OrganizeClimbView(climb: current) }
            .fullScreenCover(isPresented: $analysing, onDismiss: {
                // “看视频里这一刻”: back on the clip, at that moment, paused.
                guard let time = seekAfterAnalysis else { return }
                seekAfterAnalysis = nil
                playback.load(store.file(climb.videoName))
                playback.pause(); playback.seek(time)
            }) {
                if let pack = analysisPack {
                    AnalysisView(pack: pack, climb: current, store: store) { time in seekAfterAnalysis = time; analysing = false }
                }
            }
            .fullScreenCover(isPresented: $editing) {
                if let editPlan { EditView(climb: current, plan: editPlan, source: store.file(climb.videoName)) }
            }
            .onChange(of: store.editPlans.contains(climb.id)) { _, _ in editPlan = store.editPlan(for: current) }
            // A fresh result may bring another plan under the same name.
            .onChange(of: analysing) { _, open in if !open { editPlan = store.editPlan(for: current) } }
            .onChange(of: current.pendingAnalysis) { _, pending in
                guard pending == nil, openAfterPose else { return }
                openAfterPose = false
                if current.analysis.hasWorldLandmarks { playback.pause(); analysing = true }
            }
            .onAppear { sent = current.sent; editPlan = store.editPlan(for: current); playback.load(store.file(climb.videoName)) }
            .onDisappear { playback.unload() }
            .onChange(of:sent) { _, value in
                guard value != current.sent else { return }
                do { try store.update(climb, note: current.note, sent: value) }
                catch { store.error = String(localized: "未能保存：\(error.localizedDescription)", bundle: AppLanguage.bundle) }
            }
            .confirmationDialog("删除原片、分析和笔记？此操作无法撤销。",isPresented:$confirmDelete,titleVisibility:.visible) {
                Button("删除这一趟",role:.destructive) {
                    do { playback.unload(); try store.delete(climb); dismiss() }
                    catch { store.error = String(localized: "未能删除：\(error.localizedDescription)", bundle: AppLanguage.bundle) }
                }
            }
    }
    @ViewBuilder
    private var climbMetadata: some View {
        Text(verbatim: gradeText(current.grade)).font(.headline)
        Text(current.climbingDate, format: .dateTime.year().month().day().hour().minute())
            .font(.caption).foregroundStyle(BB.muted)
    }
    private var replay: some View {
        VStack(spacing: 12) {
            PlayerSurface(player: playback.player)
                .frame(height: 340).clipShape(RoundedRectangle(cornerRadius: 20))
            Slider(value: Binding(get: { playback.time }, set: { playback.pause(); playback.seek($0) }),
                   in: 0...max(0.1, current.analysis.duration))
                .tint(BB.ink).accessibilityLabel("视频进度")
            HStack(spacing: 4) {
                Text(verbatim: clockText(playback.time) + " / " + clockText(current.analysis.duration))
                    .font(.system(.caption, design: .monospaced)).monospacedDigit().foregroundStyle(BB.muted)
                    .lineLimit(1).minimumScaleFactor(0.7)
                    .accessibilityIdentifier("playbackClock")
                Spacer(minLength: 4)
                transportButtons
                Spacer(minLength: 4)
                speedButton
            }.buttonStyle(.plain)
        }
    }
    @ViewBuilder
    private var transportButtons: some View {
        Button { playback.pause(); playback.seek(max(0, playback.time - 0.1)) } label: {
            Image(systemName: "backward.frame.fill").font(.callout)
                .frame(width: 44, height: 48).contentShape(Rectangle())
        }.accessibilityLabel("后退零点一秒")
        Button { playback.toggle(duration: current.analysis.duration) } label: {
            Image(systemName: playback.playing ? "pause.fill" : "play.fill")
                .font(.title3).frame(width: 52, height: 52)
                .background(BB.lime, in: HoldShape()).contentShape(Rectangle())
        }.accessibilityLabel(playback.playing ? Text("暂停") : Text("播放"))
        Button { playback.pause(); playback.seek(min(current.analysis.duration, playback.time + 0.1)) } label: {
            Image(systemName: "forward.frame.fill").font(.callout)
                .frame(width: 44, height: 48).contentShape(Rectangle())
        }.accessibilityLabel("前进零点一秒")
    }
    private var speedButton: some View {
        Button {
            slow.toggle(); playback.rate = slow ? 0.5 : 1
            if playback.playing { playback.player.rate = playback.rate }
        } label: {
            Text(verbatim: slow ? "0.5×" : "1×").font(.subheadline.bold().monospacedDigit())
                .padding(.horizontal, 12).frame(minWidth: 48, minHeight: 44)
                .background(slow ? BB.lime : BB.lavender, in: Capsule()).contentShape(Rectangle())
        }.buttonStyle(.plain).accessibilityLabel(slow ? Text("0.5× 慢放") : Text("1× 原速"))
    }
    private var analysisEntry: some View {
        VStack(alignment: .leading, spacing: 10) {
            if store.analysisResults.contains(climb.id) {
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 12) { savedAnalysisButtons(expanded: false) }
                    VStack(spacing: 8) { savedAnalysisButtons(expanded: true) }
                }
            } else {
                HStack(spacing: 12) {
                    Button { startAnalysis() } label: { Text("分析这一趟").frame(maxWidth: .infinity) }
                        .buttonStyle(RouteButtonStyle())
                        .disabled(openAfterPose && current.pendingAnalysis != nil)
                        .accessibilityIdentifier("analyseClimb")
                    // The page could not analyse this clip but still made a plan (following the climber).
                    if editPlan != nil {
                        Button { playback.pause(); editing = true } label: { Text("自动剪辑") }
                            .buttonStyle(RouteButtonStyle(tone: .lime)).accessibilityIdentifier("autoEdit")
                    }
                }
            }
            if store.analysisResults.contains(climb.id), current.capture?.planUse?.used == false {
                Text(verbatim: CaptureText.planNotUsed).font(.caption).foregroundStyle(BB.muted).fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("planNotUsed")
            }
            Text("自动识别，可能认错。")
                .font(.caption).foregroundStyle(BB.muted).fixedSize(horizontal: false, vertical: true)
        }
    }
    @ViewBuilder
    private func savedAnalysisButtons(expanded: Bool) -> some View {
        Button { playback.pause(); analysisQueue.cancel(climb.id); analysing = true } label: { Text("看分析").frame(maxWidth: .infinity) }
            .buttonStyle(RouteButtonStyle()).accessibilityIdentifier("viewAnalysis")
        if editPlan != nil {
            Button { playback.pause(); editing = true } label: { Text("自动剪辑").frame(maxWidth: expanded ? .infinity : nil) }
                .buttonStyle(RouteButtonStyle(tone: .lime)).accessibilityIdentifier("autoEdit")
        }
        Button { store.discardAnalysisResult(current); startAnalysis() } label: {
            Text("重新分析").frame(maxWidth: expanded ? .infinity : nil)
        }.buttonStyle(RouteButtonStyle(tone: .paper)).accessibilityIdentifier("analyseAgain")
    }
    /// Records without usable pose data go through pose recognition first, on the existing queue.
    private func startAnalysis() {
        playback.pause()
        // Opening the page here does the work; the same record is not also analysed out of sight.
        analysisQueue.cancel(climb.id)
        if current.analysis.hasWorldLandmarks { analysing = true; return }
        do { try store.requestAnalysis(current); openAfterPose = true }
        catch { store.error = error.localizedDescription }
    }
}
