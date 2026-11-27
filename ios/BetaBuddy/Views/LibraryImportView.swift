import Photos
import SwiftUI

struct LibraryImportView: View {
    @EnvironmentObject private var library: PhotoLibraryImport
    @Environment(\.dismiss) private var dismiss
    @State private var requesting = false
    @State private var managing = false

    var body: some View {
        NavigationStack {
            PaperPage {
                VStack(alignment: .leading, spacing: 20) {
                    Image(systemName: "photo.on.rectangle.angled").font(.largeTitle).accessibilityHidden(true)
                    Text("把相册里的攀爬收好").font(.title2.bold())
                    Text("开启后，在 App 打开时检查你授权的视频，将识别到的攀爬复制到攀爬册。原片不改动，视频不上传。")
                        .font(.subheadline).foregroundStyle(BB.muted)
                    Toggle("自动收录", isOn: Binding(get: { library.enabled }, set: { value in
                        requesting = true
                        Task { await library.setEnabled(value); requesting = false }
                    })).tint(BB.ink).disabled(requesting).accessibilityIdentifier("libraryImportToggle")
                    if library.enabled && !library.hasAccess {
                        Text("请允许访问相册，也可以只选择部分视频。")
                        Button("打开系统设置") { openSettings() }.buttonStyle(RouteButtonStyle(tone: .paper))
                    }
                    if library.authorization == .limited {
                        Text("只检查你已授权的视频。")
                        Button("调整可访问的视频") { managing = true }.buttonStyle(RouteButtonStyle(tone: .paper))
                    }
                }.padding(20).background(BB.lime, in: RoundedRectangle(cornerRadius: 22))
                if library.enabled && library.hasAccess {
                    VStack(alignment: .leading, spacing: 12) {
                        if library.running {
                            Text("正在检查相册…").font(.headline)
                            ProgressView(value: Double(library.checked), total: Double(max(1, library.total)))
                            Text("已检查 \(library.checked) / \(library.total) 个视频")
                            Button("停止检查") { library.stop() }.buttonStyle(RouteButtonStyle(tone: .paper))
                        } else {
                            Text("本次已收录 \(library.imported) 个视频").font(.headline)
                            Button("检查新视频") { library.scan() }.buttonStyle(RouteButtonStyle())
                        }
                        if library.unavailable > 0 {
                            Text("\(library.unavailable) 个视频暂未处理。云端原片需先在相册下载，下载后可重试。")
                                .font(.caption).foregroundStyle(BB.muted)
                        }
                        if let error = library.error { Text(verbatim: error).foregroundStyle(.red) }
                    }.accessibilityIdentifier("libraryImportStatus")
                }
                Text("按视频拍摄日、室内或室外及推荐岩馆整理。推荐只使用视频已有的位置；识别和推荐可能有误，可在记录中修改。")
                    .font(.footnote).foregroundStyle(BB.muted)
                Text("删除攀爬册中的记录后，不会自动再次收录。手动导入仍然可用。")
                    .font(.footnote).foregroundStyle(BB.muted)
                Text(verbatim: GymDirectory.coordinateCredit).font(.caption2).foregroundStyle(BB.muted)
            }.navigationTitle("相册自动收录").navigationBarTitleDisplayMode(.inline)
                .toolbar { ToolbarItem(placement: .confirmationAction) { Button("完成") { dismiss() } } }
        }.sheet(isPresented: $managing) { LimitedLibraryPicker() }
    }
    private func openSettings() {
        if let url = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(url) }
    }
}

private struct LimitedLibraryPicker: UIViewControllerRepresentable {
    @Environment(\.dismiss) private var dismiss
    final class Controller: UIViewController {
        var finished: (() -> Void)?
        private var presented = false
        override func viewDidAppear(_ animated: Bool) {
            super.viewDidAppear(animated)
            guard !presented else { return }
            presented = true
            PHPhotoLibrary.shared().presentLimitedLibraryPicker(from: self) { [weak self] _ in self?.finished?() }
        }
    }
    func makeUIViewController(context: Context) -> Controller {
        let controller = Controller()
        controller.finished = { dismiss() }
        return controller
    }
    func updateUIViewController(_ uiViewController: Controller, context: Context) { }
}
