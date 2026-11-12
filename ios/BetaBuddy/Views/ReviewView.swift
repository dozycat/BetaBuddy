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
        guard FileManager.default.fileExists(atPath:url.path) else { failure = String(localized: "原片文件不存在，仍可查看分析与笔记。"); return }
        let item = AVPlayerItem(url:url)
        player.replaceCurrentItem(with:item)
        statusObserver = item.observe(\.status,options:[.new]) { [weak self] item, _ in
            if item.status == .failed { Task { @MainActor in self?.failure = String(localized: "原片无法播放，请检查视频文件。") } }
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
                message = String(localized: "骨架分析已更新。")
            } catch is CancellationError { message = String(localized: "已停止，原分析和笔记已保留。") }
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
    var climb: Climb
    private var current: Climb { store.climbs.first { $0.id == climb.id } ?? climb }
    private var report: String {
        let status = current.sent ? String(localized: "已完攀") : String(localized: "还在尝试")
        let coverage = String(localized: "可用髋部检测：\(percentText(current.analysis.coverage)) 的采样帧")
        let caveat = String(localized: "二维姿态辅助回看；候选动作未经人工确认。")
        return "BetaBuddy · \(current.title)\n\(gradeText(current.grade)) · \(status) · \(clockText(current.analysis.duration))\n\(coverage)\n\(current.note)\n\n\(caveat)"
    }
    var body: some View {
        PaperPage {
            HStack { Stamp(text:"THE REPLAY"); Spacer(); Text(verbatim:gradeText(climb.grade)).font(BB.display(28)).lineLimit(1).minimumScaleFactor(0.6) }
            VStack(alignment:.leading,spacing:8) {
                Text(verbatim:climb.title).font(.title.bold())
                Text(climb.date,format:.dateTime.year().month().day()).font(.caption).foregroundStyle(BB.muted)
            }
            VStack(spacing:0) {
                ZStack(alignment:.topLeading) {
                    PlayerSurface(player:playback.player)
                    if skeleton { SkeletonOverlay(analysis:current.analysis,time:playback.time) }
                    Text(verbatim:!skeleton ? "ORIGINAL" : current.analysis.nearestFrame(to: playback.time) == nil ? String(localized: "暂未可靠识别") : "33 POINTS / LOCAL").font(.system(size:10,weight:.bold,design:.monospaced)).tracking(1)
                        .padding(9).background(BB.lime,in:Capsule()).padding(12)
                }.frame(height:340).clipShape(RoundedRectangle(cornerRadius:20))
                VStack(spacing:12) {
                    Slider(value:Binding(get:{ playback.time },set:{ playback.pause(); playback.seek($0) }),in:0...max(0.1,current.analysis.duration)).tint(BB.ink).accessibilityLabel("视频进度")
                    HStack {
                        Text(verbatim:clockText(playback.time)+" / "+clockText(current.analysis.duration)).font(.system(size:11,weight:.medium,design:.monospaced)).monospacedDigit()
                        Spacer()
                        Button { playback.pause(); playback.seek(max(0,playback.time-0.1)) } label: { Image(systemName:"backward.end").frame(width:36,height:44) }.accessibilityLabel("后退零点一秒")
                        Button { playback.toggle(duration:current.analysis.duration) } label: { Image(systemName:playback.playing ? "pause.fill" : "play.fill").frame(width:44,height:44).background(BB.lime,in:Circle()) }.accessibilityLabel(playback.playing ? Text("暂停") : Text("播放"))
                        Button { playback.pause(); playback.seek(min(current.analysis.duration,playback.time+0.1)) } label: { Image(systemName:"forward.end").frame(width:36,height:44) }.accessibilityLabel("前进零点一秒")
                    }
                    HStack {
                        Toggle("骨架",isOn:$skeleton).font(.caption.bold()).tint(BB.ink).fixedSize()
                        Spacer()
                        Button {
                            slow.toggle(); playback.rate = slow ? 0.5 : 1
                            if playback.playing { playback.player.rate = playback.rate }
                        } label: { Text(slow ? "0.5× 慢放" : "1× 原速").font(.caption.bold()).padding(10).background(BB.lavender,in:Capsule()) }
                    }
                }.padding(.top,12)
            }
            if let failure = playback.failure { Text(verbatim:failure).font(.caption).foregroundStyle(.red) }
            VStack(alignment: .leading, spacing: 10) {
                if reanalysis.busy {
                    ProgressView(value: reanalysis.progress)
                    Button("停止重新分析") { reanalysis.cancel() }
                } else {
                    Button("重新分析骨架") {
                        playback.pause()
                        reanalysis.run(climb: current, store: store)
                    }.buttonStyle(.bordered)
                }
                if let message = reanalysis.message { Text(verbatim:message).font(.caption).foregroundStyle(BB.muted) }
            }
            HStack(alignment:.top) {
                Stat(value:percentText(current.analysis.coverage),label:"可用髋部检测")
                Stat(value:current.analysis.averageElbowAngle.map { "\(Int($0))°" } ?? "—",label:"平均二维肘角")
            }
            Text("检测比例反映采样帧中髋部是否清晰，不是动作评分。角度为画面投影，不能直接比较不同拍摄角度。").font(.caption).foregroundStyle(BB.muted)
            SectionLabel(title:"值得再看一眼",trailing:"REPLAY CUES")
            if current.analysis.moments.isEmpty {
                Text("这一趟没有触发候选片段。试着慢放，自己寻找最想调整的一步。").font(.subheadline).foregroundStyle(BB.muted)
            }
            ForEach(current.analysis.moments) { moment in
                Button { playback.pause(); playback.seek(moment.time) } label: {
                    HStack(alignment:.top,spacing:14) {
                        Text(verbatim:clockText(moment.time)).font(.system(size:12,weight:.bold,design:.monospaced)).padding(8).background(BB.lime,in:RoundedRectangle(cornerRadius:8))
                        VStack(alignment:.leading,spacing:7) {
                            Text(verbatim:moment.displayTitle).font(.subheadline.bold())
                            Text(verbatim:moment.displayDetail).font(.caption).lineSpacing(4).foregroundStyle(BB.muted)
                        }
                    }.padding(16).frame(maxWidth:.infinity,alignment:.leading).background(.white.opacity(0.6),in:RoundedRectangle(cornerRadius:18))
                }.buttonStyle(.plain)
            }
            VStack(alignment:.leading,spacing:16) {
                SectionLabel(title:"下一次，我想…",trailing:"YOUR NOTE")
                TextField("只记一个想尝试的改变",text:$note,axis:.vertical).lineLimit(3...6).padding(14).background(.white.opacity(0.65),in:RoundedRectangle(cornerRadius:12))
                Toggle("这趟完攀了",isOn:$sent).tint(BB.ink)
                Button(savedNotice ? "已保存 ✓" : "保存复盘") {
                    do { try store.update(climb,note:note,sent:sent); savedNotice = true }
                    catch { store.error = String(localized: "笔记未能保存：\(error.localizedDescription)") }
                }.buttonStyle(.borderedProminent).foregroundStyle(BB.paper)
            }.padding(20).background(BB.lavender,in:RoundedRectangle(cornerRadius:22))
            ShareLink(item:report) { Label("分享复盘文字",systemImage:"square.and.arrow.up").frame(maxWidth:.infinity,minHeight:44) }.buttonStyle(.bordered)
            Button("删除这一趟",role:.destructive) { confirmDelete = true }.font(.caption).frame(maxWidth:.infinity,minHeight:44)
            Text("FULL / FLOAT16 · \(current.analysis.sampledFrames) 采样帧 · 分析用时 \(Int(current.analysis.processingSeconds)) 秒").font(.system(size:9,design:.monospaced)).foregroundStyle(BB.muted)
        }.navigationTitle("动作回看").navigationBarTitleDisplayMode(.inline)
            .onAppear { note = current.note; sent = current.sent; playback.load(store.file(climb.videoName)) }
            .onDisappear { playback.unload(); reanalysis.cancel() }
            .onChange(of: phase) { _, value in if value == .background { reanalysis.cancel() } }
            .onChange(of:note) { _,_ in savedNotice = false }
            .onChange(of:sent) { _,_ in savedNotice = false }
            .confirmationDialog("删除原片、分析和笔记？此操作无法撤销。",isPresented:$confirmDelete,titleVisibility:.visible) {
                Button("删除这一趟",role:.destructive) {
                    do { playback.unload(); try store.delete(climb); dismiss() }
                    catch { store.error = String(localized: "未能删除：\(error.localizedDescription)") }
                }
            }
    }
}
