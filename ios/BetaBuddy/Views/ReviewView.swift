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

struct SkeletonOverlay: View {
    var analysis: PoseAnalysis
    var time: Double
    var body: some View {
        Canvas { context,size in
            guard let frame = analysis.nearestFrame(to:time) else { return }
            let width = min(size.width,size.height*analysis.aspectRatio)
            let height = width/analysis.aspectRatio
            let origin = CGPoint(x:(size.width-width)/2,y:(size.height-height)/2)
            func point(_ joint:Joint) -> CGPoint { CGPoint(x:origin.x+joint.x*width,y:origin.y+joint.y*height) }
            for (a,b) in PoseMath.connections where frame.joints.count == 33 && frame.joints[a].valid && frame.joints[b].valid {
                var line = Path(); line.move(to:point(frame.joints[a])); line.addLine(to:point(frame.joints[b]))
                context.stroke(line,with:.color(.black.opacity(0.6)),style:StrokeStyle(lineWidth:5,lineCap:.round))
                context.stroke(line,with:.color(BB.lime),style:StrokeStyle(lineWidth:2.5,lineCap:.round))
            }
            for joint in frame.joints.dropFirst(11) where joint.valid {
                let p = point(joint)
                context.fill(Path(ellipseIn:CGRect(x:p.x-3,y:p.y-3,width:6,height:6)),with:.color(.white))
            }
            // Only connect consecutive reliable samples; break at occlusions / large temporal gaps.
            let trail = analysis.frames.filter { $0.time <= time && $0.time >= time-2 }
            var path = Path(), previous:Double?
            for sample in trail {
                guard let hip = sample.hip else { previous = nil; continue }
                if let previous, sample.time-previous <= 0.3 { path.addLine(to:point(hip)) }
                else { path.move(to:point(hip)) }
                previous = sample.time
            }
            context.stroke(path,with:.color(BB.lavender),style:StrokeStyle(lineWidth:3,lineCap:.round))
            if let hip = frame.hip {
                let p = point(hip)
                context.fill(Path(ellipseIn:CGRect(x:p.x-5,y:p.y-5,width:10,height:10)),with:.color(BB.lavender))
            }
        }.allowsHitTesting(false).accessibilityHidden(true)
    }
}

@MainActor
final class Reanalysis: ObservableObject {
    @Published var busy = false
    @Published var progress = 0.0
    @Published var message: String?
    private var work: Task<Void, Never>?
    func run(climb: Climb, store: ClimbStore) {
        guard !busy else { return }
        busy = true; progress = 0; message = nil
        work = Task {
            defer { busy = false }
            do {
                let result = try await VideoAnalyzer().analyze(url: store.file(climb.videoName), quality: .balanced) { [weak self] value in
                    await self?.setProgress(value)
                }
                try Task.checkCancellation()
                try store.replaceAnalysis(climb, analysis: result)
                message = String(localized: "骨架分析已更新。", bundle: AppLanguage.bundle)
            } catch is CancellationError { message = String(localized: "已停止，原分析和笔记已保留。", bundle: AppLanguage.bundle) }
            catch { message = error.localizedDescription }
        }
    }
    private func setProgress(_ value: Double) { progress = value }
    func cancel() { work?.cancel() }
}

struct ReviewView: View {
    @EnvironmentObject private var store: ClimbStore
    @Environment(\.dismiss) private var dismiss
    @StateObject private var playback = Playback()
    @StateObject private var reanalysis = Reanalysis()
    @Environment(\.scenePhase) private var phase
    @State private var skeleton = true
    @State private var slow = false
    @State private var note = ""
    @State private var sent = false
    @State private var confirmDelete = false
    @State private var savedNotice = false
    @State private var organizing = false
    var climb: Climb
    private var current: Climb { store.climbs.first { $0.id == climb.id } ?? climb }
    private var report: String {
        let status = current.sent ? String(localized: "已完攀", bundle: AppLanguage.bundle) : String(localized: "还在尝试", bundle: AppLanguage.bundle)
        let coverage = String(localized: "可用髋部检测：\(percentText(current.analysis.coverage)) 的采样帧", bundle: AppLanguage.bundle)
        let caveat = String(localized: "二维姿态辅助回看；候选动作未经人工确认。", bundle: AppLanguage.bundle)
        return "BetaBuddy · \(current.displayTitle)\n\(gradeText(current.grade)) · \(status) · \(clockText(current.analysis.duration))\n\(coverage)\n\(current.note)\n\n\(caveat)"
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
            }
            if let failure = playback.failure {
                Label { Text(verbatim: failure) } icon: { Image(systemName: "exclamationmark.circle") }
                    .font(.subheadline).foregroundStyle(.red)
            } else {
                replay
            }
            if reanalysis.busy {
                VStack(alignment: .leading, spacing: 12) {
                    ProgressView(value: reanalysis.progress)
                    Button("停止重新分析") { reanalysis.cancel() }
                        .buttonStyle(RouteButtonStyle(tone: .paper))
                }
            }
            if let message = reanalysis.message {
                Text(verbatim: message).font(.caption).foregroundStyle(BB.muted)
            }
            if !current.analysis.moments.isEmpty {
                VStack(alignment: .leading, spacing: 12) {
                    SectionLabel(title: "值得再看一眼")
                    ForEach(current.analysis.moments) { moment in
                        Button { playback.pause(); playback.seek(moment.time) } label: {
                            HStack(alignment: .top, spacing: 14) {
                                Text(verbatim: clockText(moment.time))
                                    .font(.system(.caption, design: .monospaced).bold())
                                    .padding(8).background(BB.lime, in: Capsule())
                                VStack(alignment: .leading, spacing: 7) {
                                    Text(verbatim: moment.displayTitle).font(.subheadline.bold())
                                    Text(verbatim: moment.displayDetail).font(.caption).foregroundStyle(BB.muted)
                                }
                            }.padding(16).frame(maxWidth: .infinity, alignment: .leading)
                                .background(.white.opacity(0.6), in: RoundedRectangle(cornerRadius: 18))
                        }.buttonStyle(.plain)
                    }
                    Text("二维姿态辅助回看；候选动作未经人工确认。")
                        .font(.caption).foregroundStyle(BB.muted)
                }
            }
            VStack(alignment: .leading, spacing: 16) {
                SectionLabel(title: "下一次，我想…")
                TextField("只记一个想尝试的改变", text: $note, axis: .vertical)
                    .lineLimit(3...6).padding(14)
                    .background(.white.opacity(0.65), in: RoundedRectangle(cornerRadius: 12))
                Toggle("这趟完攀了", isOn: $sent).tint(BB.ink)
                Button {
                    do { try store.update(climb, note: note, sent: sent); savedNotice = true }
                    catch { store.error = String(localized: "笔记未能保存：\(error.localizedDescription)", bundle: AppLanguage.bundle) }
                } label: {
                    Text(savedNotice ? "已保存 ✓" : "保存复盘").frame(maxWidth: .infinity)
                }.buttonStyle(RouteButtonStyle())
            }.padding(20).background(BB.lavender, in: RoundedRectangle(cornerRadius: 22))
            analysisDetails
        }.navigationTitle("动作回看").navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        Button { playback.pause(); organizing = true } label: {
                            Label("整理这一趟", systemImage: "folder")
                        }.accessibilityIdentifier("organizeClimb")
                        Button {
                            playback.pause()
                            reanalysis.run(climb: current, store: store)
                        } label: {
                            Label("重新分析骨架", systemImage: "arrow.clockwise")
                        }.disabled(reanalysis.busy)
                        ShareLink(item: report) { Label("分享复盘文字", systemImage: "square.and.arrow.up") }
                        Button("删除这一趟", role: .destructive) { confirmDelete = true }
                            .accessibilityIdentifier("deleteClimb")
                    } label: {
                        Image(systemName: "ellipsis").frame(width: 44, height: 44)
                    }.accessibilityLabel("更多操作")
                }
            }
            .sheet(isPresented: $organizing) { OrganizeClimbView(climb: current) }
            .onAppear { note = current.note; sent = current.sent; playback.load(store.file(climb.videoName)) }
            .onDisappear { playback.unload(); reanalysis.cancel() }
            .onChange(of: phase) { _, value in if value == .background { reanalysis.cancel() } }
            .onChange(of:note) { _,_ in savedNotice = false }
            .onChange(of:sent) { _,_ in savedNotice = false }
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
        Text(current.climbingDate, format: .dateTime.year().month().day())
            .font(.caption).foregroundStyle(BB.muted)
    }
    private var replay: some View {
        VStack(spacing: 12) {
            ZStack(alignment: .topLeading) {
                PlayerSurface(player: playback.player)
                if skeleton { SkeletonOverlay(analysis: current.analysis, time: playback.time) }
                if skeleton && current.analysis.nearestFrame(to: playback.time) == nil {
                    Text("暂未可靠识别").font(.caption.bold())
                        .padding(.horizontal, 12).padding(.vertical, 8)
                        .background(BB.paper, in: Capsule()).padding(12)
                }
            }.frame(height: 340).clipShape(RoundedRectangle(cornerRadius: 20))
            Slider(value: Binding(get: { playback.time }, set: { playback.pause(); playback.seek($0) }),
                   in: 0...max(0.1, current.analysis.duration))
                .tint(BB.ink).accessibilityLabel("视频进度")
            Text(verbatim: clockText(playback.time) + " / " + clockText(current.analysis.duration))
                .font(.system(.caption, design: .monospaced)).monospacedDigit().foregroundStyle(BB.muted)
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 16) { transportButtons }.fixedSize(horizontal: true, vertical: false)
                VStack(spacing: 8) { transportButtons }
            }.buttonStyle(.plain)
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 20) { playbackOptions }.fixedSize(horizontal: true, vertical: false)
                VStack(alignment: .leading, spacing: 8) { playbackOptions }
            }.frame(maxWidth: .infinity, alignment: .leading)
        }
    }
    @ViewBuilder
    private var transportButtons: some View {
        Button { playback.pause(); playback.seek(max(0, playback.time - 0.1)) } label: {
            Text(verbatim: "−0.1s").font(.callout.monospacedDigit())
                .padding(.horizontal, 8).frame(minWidth: 48, minHeight: 48).contentShape(Rectangle())
        }.accessibilityLabel("后退零点一秒")
        Button { playback.toggle(duration: current.analysis.duration) } label: {
            Image(systemName: playback.playing ? "pause.fill" : "play.fill")
                .font(.title3).frame(width: 52, height: 52)
                .background(BB.lime, in: HoldShape()).contentShape(Rectangle())
        }.accessibilityLabel(playback.playing ? Text("暂停") : Text("播放"))
        Button { playback.pause(); playback.seek(min(current.analysis.duration, playback.time + 0.1)) } label: {
            Text(verbatim: "+0.1s").font(.callout.monospacedDigit())
                .padding(.horizontal, 8).frame(minWidth: 48, minHeight: 48).contentShape(Rectangle())
        }.accessibilityLabel("前进零点一秒")
    }
    @ViewBuilder
    private var playbackOptions: some View {
        Toggle("骨架", isOn: $skeleton).font(.subheadline.bold()).tint(BB.ink).fixedSize()
            .frame(minHeight: 44)
        Button {
            slow.toggle(); playback.rate = slow ? 0.5 : 1
            if playback.playing { playback.player.rate = playback.rate }
        } label: {
            Text(slow ? "0.5× 慢放" : "1× 原速").font(.subheadline.bold())
                .padding(.horizontal, 14).padding(.vertical, 10).frame(minHeight: 44)
                .background(BB.lavender, in: Capsule()).contentShape(Rectangle())
        }.buttonStyle(.plain)
    }
    private var analysisDetails: some View {
        DisclosureGroup("分析详情") {
            VStack(alignment: .leading, spacing: 14) {
                VStack(alignment: .leading, spacing: 10) {
                    if current.analysis.sampledFrames > 0 {
                        LabeledContent("可用髋部检测", value: percentText(current.analysis.coverage))
                    }
                    if let angle = current.analysis.averageElbowAngle {
                        LabeledContent("平均二维肘角", value: "\(Int(angle))°")
                    }
                    if current.analysis.sampledFrames == 0 && current.analysis.averageElbowAngle == nil {
                        Text("暂未可靠识别").foregroundStyle(BB.muted)
                    }
                }.font(.subheadline)
                Text("检测比例反映采样帧中髋部是否清晰，不是动作评分。角度为画面投影，不能直接比较不同拍摄角度。")
                    .font(.caption).foregroundStyle(BB.muted)
                Text("FULL / FLOAT16 · \(current.analysis.sampledFrames) 采样帧 · 分析用时 \(Int(current.analysis.processingSeconds)) 秒")
                    .font(.caption).foregroundStyle(BB.muted)
            }.padding(.top, 12)
        }
    }
}
