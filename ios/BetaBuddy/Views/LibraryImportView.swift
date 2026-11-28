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
                    Toggle("自动收录", isOn: Binding(get: { library.enabled }, set: { value in
                        requesting = true
                        Task { await library.setEnabled(value); requesting = false }
                    })).tint(BB.ink).disabled(requesting).accessibilityIdentifier("libraryImportToggle")
                    if library.enabled && !library.hasAccess {
                        Text("需要相册权限").foregroundStyle(BB.muted)
                        Button("打开系统设置") { openSettings() }.buttonStyle(RouteButtonStyle(tone: .paper))
                    }
                    if library.authorization == .limited {
                        Button("授权范围") { managing = true }.buttonStyle(RouteButtonStyle(tone: .paper)).accessibilityIdentifier("libraryAllowedVideos")
                    }
                }.padding(20).background(BB.lime, in: RoundedRectangle(cornerRadius: 22))
                if library.enabled && library.hasAccess {
                    VStack(alignment: .leading, spacing: 12) {
                        if library.running {
                            ProgressView(value: Double(library.checked), total: Double(max(1, library.total)))
                                .accessibilityLabel("正在检查相册…")
                            Text(verbatim: "\(library.checked) / \(library.total)").monospacedDigit()
                            Button("停止检查") { library.stop() }.buttonStyle(RouteButtonStyle(tone: .paper))
                        } else {
                            HStack {
                                Text("已收录 \(library.imported)").font(.headline)
                                Spacer()
                                Button("扫描") { library.scan() }.buttonStyle(RouteButtonStyle()).accessibilityIdentifier("libraryScan")
                            }
                        }
                        if library.unavailable > 0 {
                            Text("待处理 \(library.unavailable)")
                                .font(.caption).foregroundStyle(BB.muted)
                        }
                        if let error = library.error { Text(verbatim: error).foregroundStyle(.red) }
                    }.accessibilityElement(children: .contain).accessibilityIdentifier("libraryImportStatus")
                }
            }.navigationTitle("扫相册").navigationBarTitleDisplayMode(.inline)
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
