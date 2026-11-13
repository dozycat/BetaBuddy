import SwiftUI

struct JournalView: View {
    @EnvironmentObject private var store: ClimbStore
    @EnvironmentObject private var membership: MembershipStore
    @State private var filter = 0
    @State private var importing = false
    @State private var showingMembership = false
    @State private var search = ""
    @State private var albumFilter = JournalFilter()
    @State private var period = 0
    @State private var showingFilters = false
    @State private var showingFolders = false
    private var hasFilters: Bool { filter != 0 || !search.isEmpty || albumFilter.folderId != nil || albumFilter.unfiled || albumFilter.gymKey != nil || albumFilter.grade != nil || period != 0 }
    private var climbs: [Climb] {
        store.climbs.filter { albumFilter.matches($0) && (filter == 0 || (filter == 1 ? $0.sent : !$0.sent)) && (search.isEmpty || $0.displayTitle.localizedCaseInsensitiveContains(search) || gradeText($0.grade).localizedCaseInsensitiveContains(search) || ($0.gymName ?? "").localizedCaseInsensitiveContains(search)) }
    }
    var body: some View {
        PaperPage {
            HStack(alignment:.center) {
                HStack(spacing:6) {
                    BrandMark().fill(BB.ink).frame(width:30,height:30).accessibilityHidden(true)
                    Text(verbatim:"betabuddy").font(.system(size:23,weight:.black,design:.rounded)).tracking(-1)
                }
                Spacer()
                if membership.isMember {
                    HStack(spacing:5) {
                        MemberMark().frame(width:24,height:24)
                        Text("终生会员").font(.caption.bold())
                    }.padding(.horizontal,9).padding(.vertical,5)
                        .background(BB.lime,in:HoldShape()).accessibilityIdentifier("lifetimeBadge")
                } else {
                    Button { showingMembership = true } label: {
                        HStack(spacing:5) {
                            MemberMark().frame(width:24,height:24)
                            Text("会员权益").font(.caption.bold())
                            Image(systemName:"arrow.up.right").font(.system(size:9,weight:.bold))
                        }.padding(.horizontal,9).padding(.vertical,5)
                            .background(BB.lime,in:HoldShape())
                    }.buttonStyle(.plain).accessibilityIdentifier("membershipEntry")
                }
                SettingsButton()
            }
            VStack(alignment:.leading,spacing:9) {
                Text("攀爬册").font(.system(size:34,weight:.heavy))
                Text("没有白爬的一趟。").foregroundStyle(BB.muted)
            }
            folderBar
            if store.climbs.isEmpty {
                VStack(alignment:.leading,spacing:18) {
                    Image(systemName:"square.stack.3d.up").font(.system(size:44,weight:.light))
                    Text("第一趟，\n从这里开始。").font(.title.bold())
                    Text("选一段攀爬视频，在手机上回看骨架和动作细节，把感受和下一次想试的 beta 留下来。").foregroundStyle(BB.muted)
                    BigButton(title:"记录第一趟",icon:"plus") { importing = true }.accessibilityIdentifier("importClimb")
                }.padding(24).frame(maxWidth:.infinity,alignment:.leading).background(BB.lavender,in:RoundedRectangle(cornerRadius:24))
                Label("视频留在手机里 · 无需账号",systemImage:"lock.fill")
                    .font(.system(size:11,weight:.medium)).foregroundStyle(BB.muted)
            } else {
                VStack(spacing:12) {
                    BigButton(title:"导入这一趟",icon:"plus") { importing = true }.accessibilityIdentifier("importClimb")
                    Label("视频留在手机里 · 无需账号",systemImage:"lock.fill")
                        .font(.system(size:11,weight:.medium)).foregroundStyle(BB.muted)
                }
                Picker("记录筛选",selection:$filter) {
                    Text("全部").tag(0); Text("已完攀").tag(1); Text("还在尝试").tag(2)
                }.pickerStyle(.segmented)
                HStack {
                    Image(systemName:"magnifyingglass").foregroundStyle(BB.muted)
                    TextField("找线路、难度或岩馆",text:$search).accessibilityIdentifier("journalSearch")
                }.padding(14).background(.white.opacity(0.6),in:RoundedRectangle(cornerRadius:14))
                HStack {
                    Button { showingFilters = true } label: {
                        Label("岩馆 · 难度 · 时间", systemImage: "line.3.horizontal.decrease")
                            .font(.subheadline.bold()).padding(.horizontal, 12).frame(minHeight: 44)
                            .background(albumFilter.gymKey != nil || albumFilter.grade != nil || period != 0 ? BB.lime : .white.opacity(0.6), in: Capsule())
                    }.buttonStyle(.plain).accessibilityIdentifier("albumFilters")
                    Spacer()
                    if hasFilters { Button("清除筛选") { clearFilters() }.font(.caption).frame(minHeight: 44) }
                }
                if albumFilter.gymKey != nil || albumFilter.grade != nil || period != 0 {
                    VStack(alignment: .leading, spacing: 5) {
                        if let gym = albumFilter.gymKey {
                            if gym.isEmpty { Text("未标岩馆") }
                            else { Text(verbatim: store.climbs.first { $0.gymKey == gym }?.displayGymName ?? gym) }
                        }
                        if let grade = albumFilter.grade { Text(verbatim: gradeText(grade)) }
                        if let start = albumFilter.startDate, let end = albumFilter.endDate {
                            Text(verbatim: start.formatted(.dateTime.year().month().day().locale(AppLanguage.locale)) + " – " + end.formatted(.dateTime.year().month().day().locale(AppLanguage.locale)))
                        }
                    }.font(.caption).foregroundStyle(BB.muted).accessibilityIdentifier("activeAlbumFilters")
                }
                if climbs.isEmpty {
                    VStack(alignment:.leading,spacing:12) {
                        Text("这里还没有匹配的记录。").font(.title3.bold())
                        Text("试试其他筛选条件或搜索词。").foregroundStyle(BB.muted)
                        Button("查看全部记录") { clearFilters() }.buttonStyle(RouteButtonStyle(tone: .paper))
                    }.padding(24).frame(maxWidth:.infinity,alignment:.leading).background(BB.lavender,in:RoundedRectangle(cornerRadius:24))
                }
                ForEach(climbs) { climb in
                    NavigationLink { ReviewView(climb:climb) } label: { ClimbRow(climb:climb) }.buttonStyle(.plain)
                }
            }
        }.toolbar(.hidden,for:.navigationBar)
            .sheet(isPresented:$showingFolders) { FolderManagerView() }
            .sheet(isPresented:$showingFilters) { AlbumFilterView(filter: $albumFilter, period: $period) }
            .onChange(of: store.folders) { _, folders in
                if let selected = albumFilter.folderId, !folders.contains(where: { $0.id == selected }) { albumFilter.folderId = nil }
            }
            .sheet(isPresented:$importing) { ImportView() }
            .sheet(isPresented:$showingMembership) {
                NavigationStack { MembershipOfferView() }
            }
    }
    private func clearFilters() { filter = 0; search = ""; albumFilter = JournalFilter(); period = 0 }
    private var folderBar: some View {
        HStack(spacing: 6) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    folderChip(title: String(localized: "全部", bundle: AppLanguage.bundle), selected: albumFilter.folderId == nil && !albumFilter.unfiled) {
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
                Image(systemName: "folder.badge.plus").frame(width: 44, height: 44)
            }.buttonStyle(.plain).accessibilityLabel(Text("管理文件夹")).accessibilityIdentifier("manageFolders")
        }
    }
    private func folderChip(title: String, selected: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 5) {
                if selected { Image(systemName: "checkmark").font(.caption2.bold()) }
                Text(verbatim: title).font(.subheadline.bold()).lineLimit(1)
            }.padding(.horizontal, 13).frame(minHeight: 44)
                .background(selected ? BB.lime : .white.opacity(0.6), in: Capsule())
        }.buttonStyle(.plain).accessibilityAddTraits(selected ? [.isSelected] : [])
    }
}

struct ClimbRow: View {
    @EnvironmentObject private var store: ClimbStore
    var climb: Climb
    var body: some View {
        HStack(spacing:16) {
            Group {
                if let image = UIImage(contentsOfFile:store.file(climb.thumbnailName).path) { Image(uiImage:image).resizable().scaledToFill() }
                else { BB.lavender.overlay(Image(systemName:"figure.climbing").font(.title)) }
            }.frame(width:84,height:106).clipped().clipShape(RoundedRectangle(cornerRadius:14))
            VStack(alignment:.leading,spacing:9) {
                Text(verbatim:climb.displayTitle).font(.headline).lineLimit(2)
                Text(verbatim:"\(gradeText(climb.grade))  /  \(clockText(climb.analysis.duration))").font(.system(size:12,weight:.bold,design:.monospaced))
                Text(climb.climbingDate,format:.dateTime.month().day()).font(.caption).foregroundStyle(BB.muted)
                if !climb.displayGymName.isEmpty { Text(verbatim: climb.displayGymName).font(.caption).foregroundStyle(BB.muted).lineLimit(1) }
                Text(climb.sent ? "✓ 已完攀" : "↗ 还在尝试").font(.caption.bold())
            }
            Spacer(minLength:0)
            Image(systemName:"arrow.up.right").font(.subheadline)
        }.padding(14).background(.white.opacity(0.6),in:RoundedRectangle(cornerRadius:22))
    }
}
