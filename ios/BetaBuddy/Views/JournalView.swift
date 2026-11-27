import SwiftUI

struct JournalView: View {
    @EnvironmentObject private var store: ClimbStore
    @EnvironmentObject private var membership: MembershipStore
    @State private var filter = 0
    @State private var importing = false
    @State private var showingLibrary = false
#if DEBUG
    /// `-debugOpenCapture YES` opens the camera straight away: a first run on a phone can then be started from a computer.
    @State private var capturing = UserDefaults.standard.bool(forKey: "debugOpenCapture")
#else
    @State private var capturing = false
#endif
    @State private var importAfterCapture = false
    @State private var search = ""
    @State private var albumFilter = JournalFilter()
    @State private var period = 0
    @State private var showingFilters = false
    @State private var showingFolders = false
    private var filterCount: Int {
        [filter != 0, albumFilter.gymKey != nil, albumFilter.grade != nil, period != 0].filter { $0 }.count
    }
    private var climbs: [Climb] {
        store.climbs.filter { albumFilter.matches($0) && (filter == 0 || (filter == 1 ? $0.sent : !$0.sent)) && (search.isEmpty || $0.displayTitle.localizedCaseInsensitiveContains(search) || gradeText($0.grade).localizedCaseInsensitiveContains(search) || ($0.gymName ?? "").localizedCaseInsensitiveContains(search)) }
    }
    private var activeFilters: String {
        var parts: [String] = []
        if filter != 0 { parts.append(String(localized: filter == 1 ? "已完攀" : "还在尝试", bundle: AppLanguage.bundle)) }
        if let key = albumFilter.gymKey {
            let name = store.climbs.first { $0.gymKey == key }?.shortGymName ?? ""
            parts.append(name.isEmpty ? String(localized: "未标岩馆", bundle: AppLanguage.bundle) : name)
        }
        if let grade = albumFilter.grade { parts.append(gradeText(grade)) }
        if period == 7 { parts.append(String(localized: "近 7 天", bundle: AppLanguage.bundle)) }
        else if period == 30 { parts.append(String(localized: "近 30 天", bundle: AppLanguage.bundle)) }
        else if let start = albumFilter.startDate, let end = albumFilter.endDate {
            parts.append(start.formatted(.dateTime.year().month().day().locale(AppLanguage.locale)) + " – " + end.formatted(.dateTime.year().month().day().locale(AppLanguage.locale)))
        }
        return parts.joined(separator: " · ")
    }
    var body: some View {
        PaperPage(spacing: 18) {
            VStack(alignment: .leading, spacing: 14) {
                HStack(spacing: 6) {
                    BrandMark().fill(BB.ink).frame(width: 20, height: 20).accessibilityHidden(true)
                    Text(verbatim: "betabuddy")
                        .font(.system(size: 20, weight: .black, design: .rounded)).tracking(-0.6)
                        .lineLimit(1).fixedSize(horizontal: true, vertical: false)
                        .accessibilityIdentifier("brandWordmark")
                    Spacer(minLength: 12)
                    if membership.isMember {
                        MemberMark().frame(width: 24, height: 24)
                            .accessibilityHidden(false).accessibilityLabel(Text("终生会员"))
                            .accessibilityIdentifier("lifetimeBadge")
                    }
                }
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 10) {
                        Text("攀爬册").font(.largeTitle.bold()).fixedSize()
                        Spacer(minLength: 0)
                        importButton.fixedSize(horizontal: true, vertical: false)
                        recordButton.fixedSize(horizontal: true, vertical: false)
                    }
                    VStack(alignment: .leading, spacing: 12) {
                        Text("攀爬册").font(.largeTitle.bold())
                        HStack(spacing: 10) { recordButton; importButton }
                    }
                }
            }
            if !store.climbs.isEmpty {
                HStack(spacing: 8) {
                    HStack(spacing: 8) {
                        Image(systemName: "magnifyingglass").foregroundStyle(BB.muted).accessibilityHidden(true)
                        TextField("搜索记录", text: $search)
                            .accessibilityHint(Text("找线路、难度或岩馆"))
                            .accessibilityIdentifier("journalSearch")
                        if !search.isEmpty {
                            Button { search = "" } label: {
                                Image(systemName: "xmark.circle.fill").foregroundStyle(BB.muted)
                                    .frame(width: 32, height: 44)
                            }.buttonStyle(.plain).accessibilityLabel(Text("清除搜索"))
                        }
                    }.padding(.horizontal, 12).frame(minHeight: 48)
                        .background(.white.opacity(0.6), in: RoundedRectangle(cornerRadius: 14))
                    Button { showingFilters = true } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "line.3.horizontal.decrease")
                            if filterCount > 0 { Text(verbatim: "\(filterCount)").font(.caption.bold()) }
                        }.frame(minWidth: 48, minHeight: 48)
                            .background(filterCount > 0 ? BB.lime : .white.opacity(0.6), in: RoundedRectangle(cornerRadius: 14))
                    }.buttonStyle(.plain).accessibilityLabel(Text("筛选记录"))
                        .accessibilityValue(activeFilters).accessibilityIdentifier("albumFilters")
                }
                if filterCount > 0 {
                    HStack(spacing: 8) {
                        Text(verbatim: activeFilters).font(.caption).foregroundStyle(BB.muted)
                            .fixedSize(horizontal: false, vertical: true).accessibilityIdentifier("activeAlbumFilters")
                        Spacer(minLength: 0)
                        Button { filter = 0; albumFilter.gymKey = nil; albumFilter.grade = nil; albumFilter.startDate = nil; albumFilter.endDate = nil; period = 0 } label: {
                            Image(systemName: "xmark.circle.fill").frame(width: 44, height: 44)
                        }.buttonStyle(.plain).foregroundStyle(BB.muted).accessibilityLabel(Text("清除筛选"))
                    }
                }
            }
            Button { showingLibrary = true } label: {
                Label("扫相册", systemImage: "photo.on.rectangle.angled").font(.subheadline)
            }.buttonStyle(.plain).accessibilityIdentifier("libraryImport")
            folderBar
            if store.climbs.isEmpty {
                VStack(alignment: .leading, spacing: 16) {
                    Image(systemName: "square.stack.3d.up").font(.system(size: 36, weight: .light)).accessibilityHidden(true)
                    Text("还没有攀爬记录").font(.title3.bold())
                }.padding(24).frame(maxWidth: .infinity, alignment: .leading)
                    .background(BB.lavender, in: RoundedRectangle(cornerRadius: 22))
            } else if climbs.isEmpty {
                VStack(alignment: .leading, spacing: 16) {
                    Text("这里还没有匹配的记录。").font(.title3.bold())
                    Button("查看全部记录") { clearFilters() }.buttonStyle(RouteButtonStyle(tone: .paper))
                }.padding(20).frame(maxWidth: .infinity, alignment: .leading)
                    .background(BB.lavender, in: RoundedRectangle(cornerRadius: 22))
            } else {
                LazyVStack(spacing: 12) {
                    ForEach(JournalDayGroup.make(climbs)) { group in
                        VStack(alignment: .leading, spacing: 10) {
                            HStack(alignment: .firstTextBaseline) {
                                Text(group.id.day, format: .dateTime.year().month().day()).font(.headline)
                                Spacer(minLength: 8)
                                Text(verbatim: String(group.climbs.count)).font(.caption.monospacedDigit()).foregroundStyle(BB.muted)
                            }
                            Group {
                                if group.id.environment == .outdoor { Text("室外") }
                                else if let name = group.gymName {
                                    HStack { Text(verbatim: name); if group.climbs.allSatisfy({ $0.gymAuto == true }) { Text("推荐").font(.caption2) } }
                                }
                                else { Text("未标岩馆") }
                            }.font(.subheadline).foregroundStyle(BB.muted)
                            ForEach(group.climbs) { climb in
                                NavigationLink { ReviewView(climb: climb) } label: { ClimbRow(climb: climb) }.buttonStyle(.plain)
                            }
                        }.padding(.top, 8).accessibilityIdentifier("journalDayGroup")
                    }
                }
            }
        }.toolbar(.hidden, for: .navigationBar)
            .sheet(isPresented: $showingFolders) { FolderManagerView() }
            .sheet(isPresented: $showingFilters) { AlbumFilterView(filter: $albumFilter, period: $period, sendStatus: $filter) }
            .onChange(of: store.folders) { _, folders in
                if let selected = albumFilter.folderId, !folders.contains(where: { $0.id == selected }) { albumFilter.folderId = nil }
            }
            .sheet(isPresented: $importing) { ImportView() }
            .sheet(isPresented: $showingLibrary) { LibraryImportView() }
            .fullScreenCover(isPresented: $capturing, onDismiss: {
                if importAfterCapture { importAfterCapture = false; importing = true }
            }) {
                CaptureView(store: store) { importAfterCapture = true; capturing = false }
            }
    }
    /// Recording with the in-app camera is the main way in; a clip shot elsewhere is imported.
    private var recordButton: some View {
        Button { capturing = true } label: { Label("拍一趟", systemImage: "video.fill") }
            .buttonStyle(RouteButtonStyle()).accessibilityLabel(Text("打开相机拍一趟")).accessibilityIdentifier("recordClimb")
    }
    private var importButton: some View {
        Button { importing = true } label: { Label("导入", systemImage: "plus") }
            .buttonStyle(RouteButtonStyle(tone: .paper)).accessibilityLabel(Text("导入这一趟")).accessibilityIdentifier("importClimb")
    }
    private func clearFilters() { filter = 0; search = ""; albumFilter = JournalFilter(); period = 0 }
    private var folderBar: some View {
        HStack(spacing: 6) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    folderChip(title: String(localized: "所有记录", bundle: AppLanguage.bundle), selected: albumFilter.folderId == nil && !albumFilter.unfiled) {
                        albumFilter.folderId = nil; albumFilter.unfiled = false
                    }
                    folderChip(title: String(localized: "未归档", bundle: AppLanguage.bundle), selected: albumFilter.unfiled) {
                        albumFilter.folderId = nil; albumFilter.unfiled = true
                    }
                    ForEach(store.folders) { folder in
                        folderChip(title: folder.name, selected: albumFilter.folderId == folder.id) {
                            albumFilter.folderId = folder.id; albumFilter.unfiled = false
                        }
                    }
                }
            }
            Button { showingFolders = true } label: {
                Image(systemName: "ellipsis").frame(width: 44, height: 44)
                    .background(.white.opacity(0.6), in: Circle())
            }.buttonStyle(.plain).accessibilityLabel(Text("管理文件夹")).accessibilityIdentifier("manageFolders")
        }
    }
    private func folderChip(title: String, selected: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 5) {
                if selected { Image(systemName: "checkmark").font(.caption2.bold()) }
                Text(verbatim: title).font(.subheadline.bold()).lineLimit(1)
            }.padding(.horizontal, 14).padding(.vertical, 10).frame(minHeight: 44)
                .background(selected ? BB.lime : .white.opacity(0.6), in: Capsule())
        }.buttonStyle(.plain).accessibilityAddTraits(selected ? [.isSelected] : [])
    }
}

struct ClimbRow: View {
    @EnvironmentObject private var store: ClimbStore
    var climb: Climb
    var body: some View {
        HStack(spacing:14) {
            Group {
                if let image = UIImage(contentsOfFile:store.file(climb.thumbnailName).path) { Image(uiImage:image).resizable().scaledToFill() }
                else { BB.lavender.overlay(Image(systemName:"figure.climbing").font(.title)) }
            }.frame(width:72,height:88).clipped().clipShape(RoundedRectangle(cornerRadius:14))
            VStack(alignment:.leading,spacing:6) {
                Text(verbatim:climb.displayTitle).font(.headline).lineLimit(2)
                Text(verbatim:"\(gradeText(climb.grade))  /  \(clockText(climb.analysis.duration))").font(.system(size:12,weight:.bold,design:.monospaced))
                Text(climb.climbingDate,format:.dateTime.month().day()).font(.caption).foregroundStyle(BB.muted)
                if !climb.shortGymName.isEmpty { Text(verbatim: climb.shortGymName).font(.caption).foregroundStyle(BB.muted).lineLimit(1) }
                CaptureColourLabel(capture: climb.capture)
                if climb.pendingAnalysis != nil {
                    Text(store.analysisProgress[climb.id] == nil ? "等待分析…" : "正在分析，可离开此页面。").font(.caption).foregroundStyle(BB.muted)
                    ProgressView(value: store.analysisProgress[climb.id] ?? 0)
                }
                else if climb.capture?.analysisQueued == true {
                    Text("正在分析这一趟…").font(.caption).foregroundStyle(BB.muted)
                }
                if let message = store.analysisMessages[climb.id] { Text(verbatim: message).font(.caption).foregroundStyle(BB.muted) }
                Text(climb.sent ? "✓ 已完攀" : "↗ 还在尝试").font(.caption.bold())
            }
            Spacer(minLength:0)
            Image(systemName:"arrow.up.right").font(.subheadline)
        }.padding(14).background(.white.opacity(0.6),in:RoundedRectangle(cornerRadius:22))
    }
}
