import SwiftUI

struct JournalView: View {
    @EnvironmentObject private var store: ClimbStore
    @State private var filter = 0
    @State private var importing = false
    @State private var search = ""
    private var climbs: [Climb] {
        store.climbs.filter { (filter == 0 || (filter == 1 ? $0.sent : !$0.sent)) && (search.isEmpty || $0.title.localizedCaseInsensitiveContains(search) || $0.grade.localizedCaseInsensitiveContains(search)) }
    }
    var body: some View {
        PaperPage {
            HStack {
                Text("攀爬册").font(.system(size:34,weight:.heavy))
                Spacer()
                Button { importing = true } label: { Image(systemName:"plus").font(.title3.bold()).frame(width:44,height:44).background(BB.lime,in:Circle()) }.accessibilityLabel("新增攀爬")
            }
            Text("没有白爬的一趟。").foregroundStyle(BB.muted)
            Picker("记录筛选",selection:$filter) {
                Text("全部").tag(0); Text("已完攀").tag(1); Text("还在尝试").tag(2)
            }.pickerStyle(.segmented)
            HStack {
                Image(systemName:"magnifyingglass").foregroundStyle(BB.muted)
                TextField("找线路或难度",text:$search)
            }.padding(14).background(.white.opacity(0.6),in:RoundedRectangle(cornerRadius:14))
            if climbs.isEmpty {
                VStack(alignment:.leading,spacing:18) {
                    Image(systemName:"square.stack.3d.up").font(.system(size:44,weight:.light))
                    Text(store.climbs.isEmpty ? "第一趟，\n从这里开始。" : "这里还没有匹配的记录。").font(.title.bold())
                    Text(store.climbs.isEmpty ? "选一段攀爬视频，把动作、感受和下一次想试的 beta 留下来。" : "试试其他筛选条件或搜索词。").foregroundStyle(BB.muted)
                    if store.climbs.isEmpty { BigButton(title:"记录第一趟",icon:"plus") { importing = true } }
                }.padding(24).frame(maxWidth:.infinity,alignment:.leading).background(BB.lavender,in:RoundedRectangle(cornerRadius:24))
            } else {
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
                Text(climb.title).font(.headline).lineLimit(2)
                Text("\(climb.grade)  /  \(clockText(climb.analysis.duration))").font(.system(size:12,weight:.bold,design:.monospaced))
                Text(climb.date,format:.dateTime.month().day()).font(.caption).foregroundStyle(BB.muted)
                Text(climb.sent ? "✓ 已完攀" : "↗ 还在尝试").font(.caption.bold())
            }
            Spacer(minLength:0)
            Image(systemName:"arrow.up.right").font(.subheadline)
        }.padding(14).background(.white.opacity(0.6),in:RoundedRectangle(cornerRadius:22))
    }
}

struct TrainingView: View {
    @EnvironmentObject private var store: ClimbStore
    var body: some View {
        PaperPage {
            Stamp(text:"LESS RUSH. MORE FEEL.")
            Text("今天，\n只进步一点点。").font(.system(size:34,weight:.heavy)).lineSpacing(4)
            Text("选一个小目标。带上墙，试试看。\n这些练习独立于模型分析，由你决定节奏。").font(.subheadline).lineSpacing(5).foregroundStyle(BB.muted)
            ForEach(Drill.all) { drill in
                NavigationLink { DrillView(drill:drill) } label: {
                    VStack(alignment:.leading,spacing:22) {
                        HStack {
                            Text(drill.number).font(BB.display(46)).foregroundStyle(BB.ink.opacity(0.4))
                            Spacer()
                            Image(systemName:drill.symbol).font(.system(size:30))
                        }
                        VStack(alignment:.leading,spacing:8) {
                            Text(drill.title).font(.title2.bold())
                            Text(drill.subtitle).font(.subheadline).foregroundStyle(BB.muted)
                        }
                        HStack {
                            Text(store.completedToday(drill) ? "✓ 今天已练习" : "约 \(drill.minutes) 分钟").font(.caption.bold())
                            Spacer()
                            Image(systemName:"arrow.up.right")
                        }
                    }.padding(24).background(drill.id == "quiet-feet" ? BB.lime : drill.id == "straight-arms" ? BB.lavender : .white.opacity(0.6),in:RoundedRectangle(cornerRadius:24))
                }.buttonStyle(.plain)
            }
        }.toolbar(.hidden,for:.navigationBar)
    }
}

struct DrillView: View {
    @EnvironmentObject private var store: ClimbStore
    var drill: Drill
    var body: some View {
        PaperPage {
            Stamp(text:"PRACTICE / \(drill.number)")
            Text(drill.title).font(.system(size:34,weight:.heavy))
            Text(drill.subtitle).foregroundStyle(BB.muted)
            Image(systemName:drill.symbol).font(.system(size:80)).frame(maxWidth:.infinity).padding(.vertical,44).background(BB.lime,in:RoundedRectangle(cornerRadius:26))
            ForEach(Array(drill.steps.enumerated()),id:\.offset) { index, step in
                HStack(alignment:.top,spacing:18) {
                    Text(String(format:"%02d",index+1)).font(.system(size:22,weight:.black,design:.rounded)).foregroundStyle(BB.muted)
                    Text(step).lineSpacing(6).frame(maxWidth:.infinity,alignment:.leading)
                }
            }
            BigButton(title:store.completedToday(drill) ? "今天已练习 ✓" : "我练完了",icon:"checkmark") {
                do { try store.complete(drill) } catch { store.error = "练习记录未能保存：\(error.localizedDescription)" }
            }.disabled(store.completedToday(drill))
        }.navigationTitle("一个小练习").navigationBarTitleDisplayMode(.inline)
    }
}
