import SwiftUI

struct DirectoryGym: Decodable, Identifiable {
    let id: String
    let name: String
    let province: String
    let city: String
    let address: String
    let sourceURL: String
    let district: String?
    let aliases: [String]?
    var location: String { [province, city, district ?? ""].filter { !$0.isEmpty }.joined(separator: " · ") }
}
struct GymDirectory: Decodable {
    let updatedAt: String
    let gyms: [DirectoryGym]
    static let bundled: GymDirectory? = {
        guard let url = Bundle.main.url(forResource: "gyms-cn", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(GymDirectory.self, from: data)
    }()
}

extension Climb {
    var displayGymName: String {
        guard let id = gymId, let gym = GymDirectory.bundled?.gyms.first(where: { $0.id == id }) else { return gymName ?? "" }
        return [gymName ?? gym.name, gym.city, gym.address].filter { !$0.isEmpty }.joined(separator: " · ")
    }
}

struct GymPickerView: View {
    @Environment(\.dismiss) private var dismiss
    @Binding var gymId: String?
    @Binding var gymName: String
    @State private var search = ""
    @State private var custom = ""
    private var gyms: [DirectoryGym] {
        let terms = search.split(whereSeparator: { $0.isWhitespace }).map(String.init)
        return (GymDirectory.bundled?.gyms ?? []).filter { gym in
            let fields = [gym.name, gym.city, gym.province, gym.address, gym.district ?? ""] + (gym.aliases ?? [])
            return terms.allSatisfy { term in fields.contains { $0.localizedCaseInsensitiveContains(term) } }
        }
    }
    var body: some View {
        List {
            Section {
                Button("暂不填写") { gymId = nil; gymName = ""; dismiss() }
                HStack {
                    TextField("没找到？手填岩馆名", text: $custom).accessibilityIdentifier("customGymName")
                    Button("使用") {
                        gymId = nil; gymName = String(custom.trimmingCharacters(in: .whitespacesAndNewlines).prefix(120)); dismiss()
                    }.disabled(custom.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }
            Section {
                ForEach(gyms) { gym in
                    Button {
                        gymId = gym.id; gymName = gym.name; dismiss()
                    } label: {
                        HStack {
                            VStack(alignment: .leading, spacing: 5) {
                                Text(verbatim: gym.name).foregroundStyle(BB.ink)
                                Text(verbatim: gym.location).font(.caption).foregroundStyle(BB.muted)
                                if !gym.address.isEmpty { Text(verbatim: gym.address).font(.caption2).foregroundStyle(BB.muted) }
                            }
                            Spacer()
                            if gymId == gym.id { Image(systemName: "checkmark").foregroundStyle(BB.ink) }
                        }.padding(.vertical, 3)
                    }
                }
                if gyms.isEmpty { Text("没有找到，手填也可以。").foregroundStyle(BB.muted) }
            } header: { Text("岩馆目录") } footer: {
                VStack(alignment: .leading, spacing: 5) {
                    Text("公开资料整理，可能有遗漏或变动。")
                    if let directory = GymDirectory.bundled { Text("目录更新于 \(directory.updatedAt)") }
                }
            }
        }.navigationTitle("选择岩馆")
            .searchable(text: $search, prompt: "搜城市或岩馆")
            .scrollContentBackground(.hidden).background(BB.paper)
    }
}

struct AlbumMetadataFields: View {
    @EnvironmentObject private var store: ClimbStore
    @Binding var gymId: String?
    @Binding var gymName: String
    @Binding var climbedAt: Date
    @Binding var folderId: UUID?
    var body: some View {
        NavigationLink {
            GymPickerView(gymId: $gymId, gymName: $gymName)
        } label: {
            HStack {
                Label("岩馆", systemImage: "mappin.and.ellipse")
                Spacer()
                if gymName.isEmpty { Text("暂不填写").foregroundStyle(BB.muted) }
                else { Text(verbatim: gymName).foregroundStyle(BB.muted).lineLimit(1) }
            }.frame(minHeight: 44)
        }.buttonStyle(.plain)
        DatePicker("攀爬日期", selection: $climbedAt, in: ...Date(), displayedComponents: .date)
        Picker("放入文件夹", selection: $folderId) {
            Text("未归档").tag(nil as UUID?)
            ForEach(store.folders) { folder in Text(verbatim: folder.name).tag(Optional(folder.id)) }
        }.tint(BB.ink)
    }
}

struct OrganizeClimbView: View {
    @EnvironmentObject private var store: ClimbStore
    @Environment(\.dismiss) private var dismiss
    let climb: Climb
    @State private var title: String
    @State private var grade: String
    @State private var gymId: String?
    @State private var gymName: String
    @State private var climbedAt: Date
    @State private var folderId: UUID?
    @State private var error: String?
    init(climb: Climb) {
        self.climb = climb
        _title = State(initialValue: climb.title); _grade = State(initialValue: climb.grade)
        _gymId = State(initialValue: climb.gymId); _gymName = State(initialValue: climb.gymName ?? "")
        _climbedAt = State(initialValue: climb.climbingDate); _folderId = State(initialValue: climb.folderId)
    }
    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("这一趟的名字", text: $title)
                    Picker("线路难度", selection: $grade) {
                        Text("未标级").tag(ungradedValue)
                        ForEach(0..<13) { Text(verbatim: "V\($0)").tag("V\($0)") }
                        if grade != ungradedValue && !(0..<13).map({ "V\($0)" }).contains(grade) { Text(verbatim: grade).tag(grade) }
                    }
                }
                Section { AlbumMetadataFields(gymId: $gymId, gymName: $gymName, climbedAt: $climbedAt, folderId: $folderId) }
                if let error { Text(verbatim: error).foregroundStyle(.red) }
            }.scrollContentBackground(.hidden).background(BB.paper)
                .navigationTitle("整理这一趟").navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) { Button("取消") { dismiss() } }
                    ToolbarItem(placement: .confirmationAction) {
                        Button("保存") {
                            do {
                                try store.organize(climb, title: title, grade: grade, gymId: gymId, gymName: gymName, climbedAt: climbedAt, folderId: folderId)
                                dismiss()
                            } catch { self.error = error.localizedDescription }
                        }
                    }
                }
        }
    }
}

private struct FolderDraft: Identifiable {
    let id = UUID()
    var folder: ClimbFolder?
}

private struct FolderNameView: View {
    @EnvironmentObject private var store: ClimbStore
    @Environment(\.dismiss) private var dismiss
    let folder: ClimbFolder?
    @State private var name: String
    @State private var error: String?
    init(folder: ClimbFolder?) { self.folder = folder; _name = State(initialValue: folder?.name ?? "") }
    var body: some View {
        NavigationStack {
            Form {
                TextField("文件夹名称", text: $name).autocorrectionDisabled().accessibilityIdentifier("folderName")
                if let error { Text(verbatim: error).foregroundStyle(.red) }
            }.scrollContentBackground(.hidden).background(BB.paper)
                .navigationTitle(folder == nil ? Text("新建文件夹") : Text("重命名"))
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) { Button("取消") { dismiss() } }
                    ToolbarItem(placement: .confirmationAction) {
                        Button("保存") {
                            do { try store.saveFolder(name: name, id: folder?.id); dismiss() }
                            catch { self.error = error.localizedDescription }
                        }.accessibilityIdentifier("saveFolder")
                    }
                }
        }.presentationDetents([.height(260), .medium])
    }
}

struct FolderManagerView: View {
    @EnvironmentObject private var store: ClimbStore
    @Environment(\.dismiss) private var dismiss
    @State private var draft: FolderDraft?
    @State private var deleting: ClimbFolder?
    @State private var error: String?
    var body: some View {
        NavigationStack {
            List {
                Section {
                    Button { draft = FolderDraft() } label: {
                        Label("新建文件夹", systemImage: "folder.badge.plus")
                    }.accessibilityIdentifier("createFolder")
                    ForEach(store.folders) { folder in
                        HStack {
                            Image(systemName: "folder").foregroundStyle(BB.muted)
                            Text(verbatim: folder.name)
                            Spacer()
                            Text(verbatim: "\(store.climbs.filter { $0.folderId == folder.id }.count)").font(.caption.monospacedDigit()).foregroundStyle(BB.muted)
                            Menu {
                                Button("重命名") { draft = FolderDraft(folder: folder) }
                                Button("删除文件夹", role: .destructive) { deleting = folder }
                            } label: { Image(systemName: "ellipsis").frame(width: 44, height: 44) }
                                .accessibilityLabel(Text("管理文件夹"))
                                .accessibilityIdentifier("folderMenu-\(folder.name)")
                        }
                    }
                } footer: { Text("按自己的节奏收好：抱石、难度，或一个还想再试的计划。") }
                if let error { Text(verbatim: error).foregroundStyle(.red).accessibilityIdentifier("folderError") }
            }.scrollContentBackground(.hidden).background(BB.paper)
                .navigationTitle("文件夹").navigationBarTitleDisplayMode(.inline)
                .toolbar { ToolbarItem(placement: .confirmationAction) { Button("完成") { dismiss() } } }
                .sheet(item: $draft) { FolderNameView(folder: $0.folder) }
                .confirmationDialog("删除文件夹？里面的视频会保留。", isPresented: Binding(get: { deleting != nil }, set: { if !$0 { deleting = nil } }), titleVisibility: .visible) {
                    Button("删除文件夹", role: .destructive) {
                        guard let deleting else { return }
                        do { try store.deleteFolder(deleting); error = nil }
                        catch { self.error = error.localizedDescription }
                        self.deleting = nil
                    }
                }
        }
    }
}

struct AlbumFilterView: View {
    @EnvironmentObject private var store: ClimbStore
    @Environment(\.dismiss) private var dismiss
    @Binding var filter: JournalFilter
    @Binding var period: Int
    private var gymKeys: [String] { Array(Set(store.climbs.map(\.gymKey).filter { !$0.isEmpty })).sorted() }
    private var grades: [String] { Array(Set(store.climbs.map(\.grade))).sorted { $0.localizedStandardCompare($1) == .orderedAscending } }
    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Picker("岩馆", selection: $filter.gymKey) {
                        Text("全部岩馆").tag(nil as String?)
                        Text("未标岩馆").tag(Optional(""))
                        ForEach(gymKeys, id: \.self) { key in
                            Text(verbatim: store.climbs.first { $0.gymKey == key }?.displayGymName ?? key).tag(Optional(key))
                        }
                    }
                    Picker("线路难度", selection: $filter.grade) {
                        Text("全部难度").tag(nil as String?)
                        ForEach(grades, id: \.self) { grade in Text(verbatim: gradeText(grade)).tag(Optional(grade)) }
                    }
                }
                Section {
                    Picker("攀爬日期", selection: $period) {
                        Text("不限时间").tag(0); Text("近 7 天").tag(7); Text("近 30 天").tag(30); Text("自选日期").tag(-1)
                    }.onChange(of: period) { _, value in
                        if value == 0 { filter.startDate = nil; filter.endDate = nil }
                        else if value > 0 {
                            filter.startDate = Calendar.current.date(byAdding: .day, value: -(value - 1), to: Calendar.current.startOfDay(for: Date()))
                            filter.endDate = Date()
                        } else { filter.startDate = filter.startDate ?? Date(); filter.endDate = filter.endDate ?? Date() }
                    }
                    if period == -1 {
                        DatePicker("开始日期", selection: Binding(get: { filter.startDate ?? Date() }, set: {
                            filter.startDate = $0
                            if $0 > (filter.endDate ?? Date()) { filter.endDate = $0 }
                        }), in: ...Date(), displayedComponents: .date)
                        DatePicker("结束日期", selection: Binding(get: { filter.endDate ?? Date() }, set: { filter.endDate = $0 }), in: (filter.startDate ?? Date())...Date(), displayedComponents: .date)
                    }
                } footer: { Text("按攀爬当天筛选；旧记录默认使用导入日期，可以在回看页修改。") }
            }.scrollContentBackground(.hidden).background(BB.paper)
                .navigationTitle("筛选这一趟").navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) {
                        Button("清除筛选") { filter.gymKey = nil; filter.grade = nil; filter.startDate = nil; filter.endDate = nil; period = 0 }
                    }
                    ToolbarItem(placement: .confirmationAction) { Button("完成") { dismiss() } }
                }
        }.presentationDetents([.medium, .large])
    }
}
