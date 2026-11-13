import SwiftUI
import PhotosUI
import UniformTypeIdentifiers
import AVFoundation

struct ImportedMovie: Transferable {
    let url: URL
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType:.movie) { movie in SentTransferredFile(movie.url) } importing: { received in
            ImportedMovie(url: try stageVideo(received.file))
        }
    }
}

func stageVideo(_ source: URL) throws -> URL {
    let size = try source.resourceValues(forKeys:[.fileSizeKey]).fileSize ?? 0
    guard size <= 500 * 1024 * 1024 else { throw AnalysisError.tooLarge }
    let target = FileManager.default.temporaryDirectory.appendingPathComponent("betabuddy-\(UUID()).\(source.pathExtension.isEmpty ? "mov" : source.pathExtension)")
    try FileManager.default.copyItem(at:source,to:target)
    return target
}

@MainActor
final class ImportModel: ObservableObject {
    @Published var selectedURL: URL?
    @Published var progress = 0.0
    @Published var busy = false
    @Published var importing = false
    @Published var error: String?
    @Published var saved: Climb?
    private var work: Task<Void,Never>?
    private let analyzer = VideoAnalyzer()
    func replace(_ url: URL) {
        if let old = selectedURL { try? FileManager.default.removeItem(at:old) }
        selectedURL = url
        error = nil
    }
    func run(store:ClimbStore,title:String,grade:String,sent:Bool,note:String,quality:AnalysisQuality) {
        guard !busy, let url = selectedURL else { return }
        busy = true; progress = 0; error = nil
        work = Task {
            defer { busy = false }
            do {
                let analysis = try await analyzer.analyze(url:url,quality:quality) { [weak self] value in
                    await self?.setProgress(value)
                }
                let thumbnail = try await analyzer.thumbnail(url:url)
                try Task.checkCancellation()
                saved = try store.save(title:title,grade:grade,sent:sent,note:note,stagedVideo:url,thumbnail:thumbnail,analysis:analysis)
                selectedURL = nil
                try? FileManager.default.removeItem(at:url)
            } catch is CancellationError {
                error = String(localized: "分析已停止。视频仍已选中，可以重新开始。", bundle: AppLanguage.bundle)
            } catch { self.error = error.localizedDescription }
        }
    }
    private func setProgress(_ value: Double) { progress = value }
    func cancel() { work?.cancel() }
    func clean() {
        cancel()
        // The view cannot dismiss while running; cancellation releases reader before cleanup.
        if !busy, let url = selectedURL { try? FileManager.default.removeItem(at:url); selectedURL = nil }
    }
}

struct ImportView: View {
    @EnvironmentObject private var store: ClimbStore
    @Environment(\.dismiss) private var dismiss
    @Environment(\.scenePhase) private var phase
    @StateObject private var model = ImportModel()
    @State private var photo: PhotosPickerItem?
    @State private var files = false
    @State private var title = ""
    @State private var grade = ungradedValue
    @State private var sent = false
    @State private var note = ""
    @State private var quality = AnalysisQuality.balanced
    var body: some View {
        NavigationStack {
            Group {
                if let saved = model.saved { ReviewView(climb:saved) }
                else {
                    PaperPage {
                        Stamp(text:"A NEW ATTEMPT")
                        Text(model.busy ? "正在读懂\n你的这一趟。" : "把这一趟，\n变成下一次的灵感。").font(.system(size:29,weight:.heavy)).lineSpacing(4)
                        if model.busy { processing }
                        else { selection }
                        if let error = model.error {
                            Label { Text(verbatim:error) } icon: { Image(systemName:"exclamationmark.circle") }.font(.subheadline).foregroundStyle(.red)
                                .accessibilityIdentifier("importError")
                        }
                    }
                }
            }
            .navigationTitle(model.saved == nil ? Text("记录攀爬") : Text("这一趟已保存"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement:.topBarTrailing) {
                    Button("完成") { dismiss() }.disabled(model.busy || model.importing).accessibilityIdentifier("importDone")
                }
            }
        }.interactiveDismissDisabled(model.busy || model.importing)
            .onDisappear { model.clean() }
            #if DEBUG
            // `-debugImportVideo <path>` on the launch command line selects a video without the system picker, for tests.
            .onAppear {
                if model.selectedURL == nil, let path = UserDefaults.standard.string(forKey: "debugImportVideo"),
                   let staged = try? stageVideo(URL(fileURLWithPath: path)) { model.replace(staged) }
            }
            #endif
            .onChange(of:phase) { _, state in if state == .background && model.busy { model.cancel() } }
            .task(id:photo) {
                guard let photo else { return }
                model.importing = true
                defer { model.importing = false }
                do {
                    if let movie = try await photo.loadTransferable(type:ImportedMovie.self) {
                        if Task.isCancelled { try? FileManager.default.removeItem(at:movie.url) }
                        else { model.replace(movie.url) }
                    } else { model.error = String(localized: "未能取得视频。若原片在 iCloud，请先在相册中下载。", bundle: AppLanguage.bundle) }
                } catch { model.error = String(localized: "导入失败：\(error.localizedDescription)", bundle: AppLanguage.bundle) }
            }
            .fileImporter(isPresented:$files,allowedContentTypes:[.movie],allowsMultipleSelection:false) { result in
                switch result {
                case .success(let urls):
                    guard let url = urls.first else { return }
                    model.importing = true
                    Task {
                        defer { model.importing = false }
                        do {
                            let staged = try await Task.detached(priority:.userInitiated) {
                                let access = url.startAccessingSecurityScopedResource()
                                defer { if access { url.stopAccessingSecurityScopedResource() } }
                                return try stageVideo(url)
                            }.value
                            model.replace(staged)
                        } catch { model.error = error.localizedDescription }
                    }
                case .failure(let error): model.error = error.localizedDescription
                }
            }
    }
    private var selection: some View {
        VStack(alignment:.leading,spacing:22) {
            VStack(alignment:.leading,spacing:14) {
                HStack {
                    Image(systemName:model.selectedURL == nil ? "video.badge.plus" : "checkmark.circle.fill").font(.title)
                    VStack(alignment:.leading,spacing:4) {
                        Text(model.importing ? "正在取回原片…" : model.selectedURL == nil ? "选择攀爬视频" : "视频已就绪").font(.headline)
                        Text("单人全身 · 固定机位 · 最长 3 分钟").font(.caption).foregroundStyle(BB.muted)
                    }
                }
                HStack {
                    PhotosPicker(selection:$photo,matching:.videos) { Label("从相册选择",systemImage:"photo").foregroundStyle(BB.paper) }
                        .buttonStyle(RouteButtonStyle())
                    Button { files = true } label: { Label("文件",systemImage:"folder") }
                        .buttonStyle(RouteButtonStyle(tone:.paper))
                }.disabled(model.importing)
                if model.importing { ProgressView() }
            }.padding(20).frame(maxWidth:.infinity,alignment:.leading).background(BB.lime,in:RoundedRectangle(cornerRadius:22))
            VStack(alignment:.leading,spacing:14) {
                Text("这一趟的名字").font(.caption.bold())
                TextField("例如：周日的紫色线路",text:$title).textFieldStyle(.roundedBorder).accessibilityIdentifier("climbTitle")
                HStack {
                    Text("线路难度").font(.subheadline)
                    Spacer()
                    Picker("线路难度",selection:$grade) {
                        Text("未标级").tag(ungradedValue)
                        ForEach(0..<13) { Text(verbatim:"V\($0)").tag("V\($0)") }
                    }.tint(BB.ink)
                }
                Toggle("这趟完攀了",isOn:$sent).tint(BB.ink)
                TextField("记住一个感受（可选）",text:$note,axis:.vertical).textFieldStyle(.roundedBorder).lineLimit(2...4)
            }
            VStack(alignment:.leading,spacing:12) {
                SectionLabel(title:"回看精度",trailing:"SAME FULL MODEL")
                ForEach(AnalysisQuality.allCases) { option in
                    Button { quality = option } label: {
                        HStack {
                            VStack(alignment:.leading,spacing:4) {
                                Text(verbatim:option.title).font(.subheadline.bold())
                                Text(verbatim:option.subtitle).font(.caption).foregroundStyle(BB.muted)
                            }
                            Spacer()
                            Image(systemName:quality == option ? "checkmark.circle.fill" : "circle")
                        }.padding(16).background(quality == option ? BB.lavender : .white.opacity(0.5),in:RoundedRectangle(cornerRadius:16))
                    }.buttonStyle(.plain)
                }
            }
            BigButton(title:"开始本地分析",icon:"sparkle") { model.run(store:store,title:title,grade:grade,sent:sent,note:note,quality:quality) }
                .disabled(model.selectedURL == nil || model.importing).opacity(model.selectedURL == nil ? 0.4 : 1)
                .accessibilityIdentifier("startAnalysis")
            Text("分析时请保持应用在前台。视频和模型都在手机上处理；iCloud 原片可能需由系统先下载。").font(.caption).foregroundStyle(BB.muted)
        }
    }
    private var processing: some View {
        VStack(alignment:.leading,spacing:24) {
            ClimbingArt().frame(height:220).padding(24).background(BB.lime,in:RoundedRectangle(cornerRadius:24))
            HStack(alignment:.firstTextBaseline) {
                Text(verbatim:percentText(model.progress)).font(BB.display(54)).monospacedDigit()
                Spacer()
                Text(verbatim:"ON YOUR IPHONE").font(.system(size:10,weight:.bold,design:.monospaced))
            }
            ProgressView(value:model.progress).tint(BB.ink)
            Text("逐帧读取 → 33 点姿态 → 整理回看片段\n无需上传。分析完成后自动保存。").font(.subheadline).lineSpacing(5).foregroundStyle(BB.muted)
            Button("停止分析",role:.cancel) { model.cancel() }
                .buttonStyle(RouteButtonStyle(tone:.paper)).frame(maxWidth:.infinity)
        }
    }
}
