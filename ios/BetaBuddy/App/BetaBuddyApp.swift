import SwiftUI

@main
struct BetaBuddyApp: App {
    @StateObject private var store = ClimbStore()
    var body: some Scene {
        WindowGroup {
            RootView().environmentObject(store).tint(BB.ink).preferredColorScheme(.light)
                .environment(\.locale, Locale(identifier: "zh_Hans_CN"))
                .alert("暂时无法完成", isPresented: Binding(get: { store.error != nil }, set: { if !$0 { store.error = nil } })) {
                    Button("知道了", role: .cancel) { store.error = nil }
                } message: { Text(store.error ?? "") }
        }
    }
}

struct RootView: View {
    // `-initialTab 3` on the launch command line opens a specific entry; nothing writes this key.
    @State private var tab = UserDefaults.standard.integer(forKey: "initialTab")
    @State private var importing = false
    @StateObject private var gym = GymSession()
    var body: some View {
        VStack(spacing:0) {
            Group {
                switch tab {
                case 1: NavigationStack { JournalView() }
                case 2: NavigationStack { TrainingView() }
                case 3: GymView(session: gym)
                default: NavigationStack { HomeView(importing: $importing, tab: $tab) }
                }
            }.frame(maxWidth:.infinity,maxHeight:.infinity)
            HStack(spacing:0) {
                tabButton("上墙", symbol:"figure.climbing", index:0)
                tabButton("攀爬册", symbol:"square.stack", index:1)
                tabButton("练习", symbol:"scope", index:2)
                tabButton("岩馆", symbol:"map", index:3)
            }.padding(.top,12).padding(.bottom,6).background(BB.paper)
                .overlay(alignment:.top) { Rectangle().fill(BB.line).frame(height:1) }
        }.background(BB.paper).sheet(isPresented:$importing) { ImportView() }
    }
    private func tabButton(_ title: String, symbol: String, index: Int) -> some View {
        Button { tab = index } label: {
            VStack(spacing:5) {
                Image(systemName:symbol).font(.system(size:20,weight:.semibold)).frame(width:54,height:30)
                    .background(tab == index ? BB.lime : .clear,in:Capsule())
                Text(title).font(.system(size:11,weight:tab == index ? .bold : .medium))
            }.foregroundStyle(tab == index ? BB.ink : BB.muted).frame(maxWidth:.infinity).contentShape(Rectangle())
        }.buttonStyle(.plain).accessibilityAddTraits(tab == index ? .isSelected : [])
    }
}
