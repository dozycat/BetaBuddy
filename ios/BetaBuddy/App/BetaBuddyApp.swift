import SwiftUI

@main
struct BetaBuddyApp: App {
    @StateObject private var store = ClimbStore()
    var body: some Scene {
        WindowGroup {
            RootView().environmentObject(store).tint(BB.ink).preferredColorScheme(.light)
                .alert("暂时无法完成", isPresented: Binding(get: { store.error != nil }, set: { if !$0 { store.error = nil } })) {
                    Button("知道了", role: .cancel) { store.error = nil }
                } message: { Text(verbatim: store.error ?? "") }
        }
    }
}

struct RootView: View {
    // 0 = 岩馆, 1 = 攀爬册. `-initialTab 1` on the launch command line opens the logbook; nothing writes this key.
    @State private var tab = min(1, max(0, UserDefaults.standard.integer(forKey: "initialTab")))
    @StateObject private var gym = GymSession()
    var body: some View {
        VStack(spacing:0) {
            Group {
                switch tab {
                case 1: NavigationStack { JournalView() }
                default: GymView(session: gym)
                }
            }.frame(maxWidth:.infinity,maxHeight:.infinity)
            HStack(spacing:0) {
                tabButton("岩馆", symbol:"map", index:0, id:"tabGym")
                tabButton("攀爬册", symbol:"square.stack", index:1, id:"tabJournal")
            }.padding(.top,12).padding(.bottom,6).background(BB.paper)
                .overlay(alignment:.top) { Rectangle().fill(BB.line).frame(height:1) }
        }.background(BB.paper)
    }
    private func tabButton(_ title: LocalizedStringKey, symbol: String, index: Int, id: String) -> some View {
        Button { tab = index } label: {
            VStack(spacing:5) {
                Image(systemName:symbol).font(.system(size:20,weight:.semibold)).frame(width:54,height:30)
                    .background(tab == index ? BB.lime : .clear,in:Capsule())
                Text(title).font(.system(size:11,weight:tab == index ? .bold : .medium)).lineLimit(1).minimumScaleFactor(0.8)
            }.foregroundStyle(tab == index ? BB.ink : BB.muted).frame(maxWidth:.infinity).contentShape(Rectangle())
        }.buttonStyle(.plain).accessibilityAddTraits(tab == index ? .isSelected : []).accessibilityIdentifier(id)
    }
}
