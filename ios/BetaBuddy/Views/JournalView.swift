import SwiftUI

struct JournalView: View {
    @EnvironmentObject private var store: ClimbStore
    @EnvironmentObject private var membership: MembershipStore
    @State private var filter = 0
    @State private var importing = false
    @State private var search = ""
    private var climbs: [Climb] {
        store.climbs.filter { (filter == 0 || (filter == 1 ? $0.sent : !$0.sent)) && (search.isEmpty || $0.displayTitle.localizedCaseInsensitiveContains(search) || gradeText($0.grade).localizedCaseInsensitiveContains(search)) }
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
                    Label("终生会员",systemImage:"star.fill").font(.caption.bold())
                        .padding(.horizontal,10).padding(.vertical,6)
                        .background(BB.lime,in:Capsule()).accessibilityIdentifier("lifetimeBadge")
                }
                SettingsButton()
            }
            VStack(alignment:.leading,spacing:9) {
                Text("攀爬册").font(.system(size:34,weight:.heavy))
                Text("没有白爬的一趟。").foregroundStyle(BB.muted)
            }
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
                    TextField("找线路或难度",text:$search).accessibilityIdentifier("journalSearch")
                }.padding(14).background(.white.opacity(0.6),in:RoundedRectangle(cornerRadius:14))
                if climbs.isEmpty {
                    VStack(alignment:.leading,spacing:12) {
                        Text("这里还没有匹配的记录。").font(.title3.bold())
                        Text("试试其他筛选条件或搜索词。").foregroundStyle(BB.muted)
                    }.padding(24).frame(maxWidth:.infinity,alignment:.leading).background(BB.lavender,in:RoundedRectangle(cornerRadius:24))
                }
                ForEach(climbs) { climb in
                    NavigationLink { ReviewView(climb:climb) } label: { ClimbRow(climb:climb) }.buttonStyle(.plain)
                }
            }
        }.toolbar(.hidden,for:.navigationBar).sheet(isPresented:$importing) { ImportView() }
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
                Text(climb.date,format:.dateTime.month().day()).font(.caption).foregroundStyle(BB.muted)
                Text(climb.sent ? "✓ 已完攀" : "↗ 还在尝试").font(.caption.bold())
            }
            Spacer(minLength:0)
            Image(systemName:"arrow.up.right").font(.subheadline)
        }.padding(14).background(.white.opacity(0.6),in:RoundedRectangle(cornerRadius:22))
    }
}
