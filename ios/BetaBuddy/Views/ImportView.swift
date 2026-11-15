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
    @Published var climbedAt = Date()
    @Published var hasRecordingDate = false
    @Published var readingDate = false
    @Published var busy = false
    @Published var importing = false
    @Published var error: String?
    @Published var saved: Climb?
    private var work: Task<Void,Never>?
    private let analyzer = VideoAnalyzer()
    func replace(_ url: URL) {
        if let old = selectedURL { try? FileManager.default.removeItem(at:old) }
        readingDate = true
        selectedURL = url
        error = nil
        climbedAt = Date()
        hasRecordingDate = false
    }
    func readRecordingDate(for url: URL) async {
        defer { if selectedURL == url { readingDate = false } }
        let originalDate = climbedAt
        let date = await VideoAnalyzer().recordingDate(url: url)
        guard !Task.isCancelled, selectedURL == url, climbedAt == originalDate else { return }
        hasRecordingDate = date != nil
        if let date { climbedAt = date }
    }
    func run(store:ClimbStore,title:String,grade:String,sent:Bool,note:String,quality:AnalysisQuality,gymId:String?,gymName:String,climbedAt:Date,folderId:UUID?,analyze:Bool) {
        guard !busy, let url = selectedURL else { return }
        busy = true; error = nil
        work = Task {
            defer { busy = false }
            do {
                let analysis = try await analyzer.videoSummary(url: url)
                let thumbnail = try await analyzer.thumbnail(url:url)
                try Task.checkCancellation()
                saved = try store.save(title:title,grade:grade,sent:sent,note:note,stagedVideo:url,thumbnail:thumbnail,analysis:analysis,gymId:gymId,gymName:gymName,climbedAt:climbedAt,folderId:folderId,pendingAnalysis:analyze ? quality.rawValue : nil)
                if analyze { store.resumeAnalysis() }
                selectedURL = nil
                try? FileManager.default.removeItem(at:url)
            } catch is CancellationError {
                error = String(localized: "分析已停止。视频仍已选中，可以重新开始。", bundle: AppLanguage.bundle)
            } catch { self.error = error.localizedDescription }
        }
    }
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
    @StateObject private var model = ImportModel()
    @State private var photo: PhotosPickerItem?
    @State private var files = false
    @State private var title = ""
    @State private var grade = ungradedValue
    @State private var sent = false
    @State private var note = ""
    @State private var gymId: String?
    @State private var gymName = ""
    @State private var analyze = false
    @State private var folderId: UUID?
    @State private var quality = AnalysisQuality.balanced
    var body: some View {
        NavigationStack {
            Group {
                if let saved = model.saved { ReviewView(climb:saved) }
                else {
                    PaperPage {
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
            .task(id: model.selectedURL) {
                if let url = model.selectedURL { await model.readRecordingDate(for: url) }
            }
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
                    Text(model.importing ? "正在取回原片…" : model.selectedURL == nil ? "选择攀爬视频" : "视频已就绪")
                        .font(.headline).fixedSize(horizontal: false, vertical: true)
                }
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 12) { sourceButtons(expanded: false) }
                        .fixedSize(horizontal: true, vertical: false)
                    VStack(alignment: .leading, spacing: 12) { sourceButtons(expanded: true) }
                }.disabled(model.importing)
                if model.importing { ProgressView() }
                DisclosureGroup("视频要求") {
                    Text("单人全身 · 固定机位 · 最长 3 分钟")
                        .font(.subheadline).foregroundStyle(BB.muted)
                        .frame(maxWidth: .infinity, alignment: .leading).padding(.top, 8)
                }.font(.subheadline)
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
            DatePicker("攀爬时间", selection: $model.climbedAt, displayedComponents: [.date, .hourAndMinute])
                .accessibilityIdentifier("climbedAt")
            if model.readingDate { ProgressView() }
            else { Text(model.hasRecordingDate ? "已读取视频拍摄时间，可手动修改。" : "未读取到拍摄时间，默认使用当前时间，可手动修改。")
                .font(.caption).foregroundStyle(BB.muted) }
            DisclosureGroup("收好这一趟（可选）") {
                VStack(spacing: 14) {
                    AlbumMetadataFields(gymId: $gymId, gymName: $gymName, climbedAt: $model.climbedAt, folderId: $folderId, showsDate: false)
                }.padding(.top, 12)
            }.tint(BB.ink)
            Toggle("保存后分析骨架", isOn: $analyze).tint(BB.ink).accessibilityIdentifier("analyzeAfterSaving")
            if analyze {
                DisclosureGroup("分析设置") {
                    VStack(alignment: .leading, spacing: 12) {
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
                    Text("视频先保存，分析可在离开页面后继续。系统暂停后台任务时，下次打开应用会重新分析。")
                        .font(.caption).foregroundStyle(BB.muted)
                    }.padding(.top, 12)
                }
            }
            BigButton(title:"保存这一趟",icon:"checkmark") { model.run(store:store,title:title,grade:grade,sent:sent,note:note,quality:quality,gymId:gymId,gymName:gymName,climbedAt:model.climbedAt,folderId:folderId,analyze:analyze) }
                .disabled(model.selectedURL == nil || model.importing || model.readingDate).opacity(model.selectedURL == nil ? 0.4 : 1)
                .accessibilityIdentifier("startAnalysis")
        }
    }
    @ViewBuilder
    private func sourceButtons(expanded: Bool) -> some View {
        PhotosPicker(selection: $photo, matching: .videos, preferredItemEncoding: .current) {
            Label("相册", systemImage: "photo")
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: expanded ? .infinity : nil)
        }.buttonStyle(RouteButtonStyle()).accessibilityLabel("从相册选择")
        Button { files = true } label: {
            Label("文件", systemImage: "folder")
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: expanded ? .infinity : nil)
        }.buttonStyle(RouteButtonStyle(tone: .paper))
    }
    private var processing: some View {
        VStack(alignment:.leading,spacing:24) {
            Text("正在保存视频…").font(.title2.bold())
            ProgressView().tint(BB.ink)
        }
    }
}
