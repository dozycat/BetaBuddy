import SwiftUI

/// Publishes the interface language so the whole view tree follows an in-app change at once.
@MainActor
final class LanguageSettings: ObservableObject {
    @Published private(set) var selection: String?
    @Published private(set) var language: String
    private let defaults: UserDefaults
    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        #if DEBUG
        // `-resetLanguage YES` on the launch command line starts from “follow system”, for tests.
        if defaults.bool(forKey: "resetLanguage") { AppLanguage.select(nil, in: defaults) }
        #endif
        selection = AppLanguage.selection(in: defaults)
        language = AppLanguage.resolve(selection: AppLanguage.selection(in: defaults), system: Locale.preferredLanguages)
    }
    /// `nil` follows the system.
    func select(_ choice: String?) {
        AppLanguage.select(choice, in: defaults)
        selection = AppLanguage.selection(in: defaults)
        language = AppLanguage.resolve(selection: selection, system: Locale.preferredLanguages)
    }
    var locale: Locale { AppLanguage.locale(for: language) }
}

private struct OpenSettingsKey: EnvironmentKey { static let defaultValue: () -> Void = {} }
extension EnvironmentValues {
    /// Opens settings (language first, then About) from anywhere.
    var openSettings: () -> Void {
        get { self[OpenSettingsKey.self] }
        set { self[OpenSettingsKey.self] = newValue }
    }
}

@main
struct BetaBuddyApp: App {
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var store: ClimbStore
    @StateObject private var analysisQueue: AnalysisQueue
    @StateObject private var library: PhotoLibraryImport
    @StateObject private var language = LanguageSettings()
    @StateObject private var membership = MembershipStore()
    init() {
        // Edits a run that ended with the edit page open left behind; they are made again on demand.
        try? FileManager.default.removeItem(at: EditFiles.root)
        let store = ClimbStore()
        _store = StateObject(wrappedValue: store)
        _library = StateObject(wrappedValue: PhotoLibraryImport(store: store))
        _analysisQueue = StateObject(wrappedValue: AnalysisQueue(store: store))
    }
    var body: some Scene {
        WindowGroup {
            RootView().environmentObject(library).environmentObject(store).environmentObject(analysisQueue).environmentObject(language).environmentObject(membership).tint(BB.ink).preferredColorScheme(.light)
                .alert("暂时无法完成", isPresented: Binding(get: { store.error != nil }, set: { if !$0 { store.error = nil } })) {
                    Button("知道了", role: .cancel) { store.error = nil }
                } message: { Text(verbatim: store.error ?? "") }
                .environment(\.locale, language.locale)
                .task {
                    // A recording an earlier run did not live to save becomes a record now, before anything is analysed.
                    await CaptureRecovery.sweep(store: store, analysisAvailable: GymPack.current?.supportsAnalysis == true)
                    store.resumeAnalysis(); library.refresh(); await membership.refreshEntitlement()
                }
                .onChange(of: store.selectionOpen) { _, open in
                    if open { library.stop() } else { library.refresh() }
                }
                .onChange(of: store.cameraOpen) { _, open in
                    if open { library.stop() } else { library.refresh() }
                }
                .onChange(of: scenePhase) { _, phase in
                    library.setForeground(phase == .active)
                    if phase == .active { store.resumeAnalysis(); analysisQueue.setSuspended(false); Task { await membership.load() } }
                    else if phase == .background { analysisQueue.setSuspended(true) }
                }
        }
    }
}

struct RootView: View {
    @EnvironmentObject private var membership: MembershipStore
    @EnvironmentObject private var language: LanguageSettings
    // 0 = 岩馆, 1 = 攀爬册. `-initialTab 1` on the launch command line opens the logbook; nothing writes this key.
    // A Debug build asked to open the camera straight away (`-debugOpenCapture YES`) starts on the logbook, where the camera is.
    #if DEBUG
    @State private var tab = UserDefaults.standard.bool(forKey: "debugOpenCapture") ? 1 : min(1, max(0, UserDefaults.standard.integer(forKey: "initialTab")))
    #else
    @State private var tab = min(1, max(0, UserDefaults.standard.integer(forKey: "initialTab")))
    #endif
    @State private var settings = false
    @StateObject private var gym = GymSession()
    var body: some View {
        VStack(spacing:0) {
            // Rebuilt on a language change so text cached in view state is produced again.
            Group {
                switch tab {
                case 1: NavigationStack { JournalView() }
                default: GymView(session: gym)
                }
            }.id(language.language).frame(maxWidth:.infinity,maxHeight:.infinity)
            HStack(spacing:0) {
                tabButton("岩馆", symbol:"map", index:0, id:"tabGym")
                tabButton("攀爬册", symbol:"square.stack", index:1, id:"tabJournal")
                // Always reachable, including over the gym page, whose top bar belongs to the pack.
                Button { settings = true } label: {
                    Image(systemName:"gearshape").font(.system(size:19,weight:.semibold)).frame(width:64,height:52).foregroundStyle(BB.muted).contentShape(Rectangle())
                }.buttonStyle(.plain).accessibilityLabel("设置与关于").accessibilityIdentifier("settings")
            }.padding(.top,12).padding(.bottom,6).background(BB.paper)
                .overlay(alignment:.top) { Rectangle().fill(BB.line).frame(height:1) }
        }.background(BB.paper)
            // Out of sight behind the paper: the page that analyses a recording after it has been saved.
            .background(AnalysisQueueHost())
            .environment(\.openSettings, { settings = true })
            .sheet(isPresented:$settings) {
                NavigationStack { AboutView() }.id(language.language).environment(\.locale, language.locale).tint(BB.ink)
            }
            .sheet(isPresented: $gym.showingMembership, onDismiss: { gym.cancelMembershipRequest() }) {
                NavigationStack { MembershipOfferView() }
                    .environment(\.locale, language.locale).tint(BB.ink)
            }
            .onAppear { gym.updateMembership(membership.isMember) }
            .onChange(of: membership.isMember) { _, active in gym.updateMembership(active) }
            .onChange(of: language.language) { _, _ in gym.languageChanged() }
            .onReceive(gym.$settingsRequests.dropFirst()) { _ in settings = true }
    }
    private func tabButton(_ title: LocalizedStringKey, symbol: String, index: Int, id: String) -> some View {
        Button { tab = index } label: {
            VStack(spacing:5) {
                Image(systemName:symbol).font(.system(size:20,weight:.semibold)).frame(width:54,height:30)
                    .background(tab == index ? BB.lime : .clear,in:HoldShape())
                Text(title).font(.system(size:11,weight:tab == index ? .bold : .medium)).lineLimit(1).minimumScaleFactor(0.8)
            }.foregroundStyle(tab == index ? BB.ink : BB.muted).frame(maxWidth:.infinity).contentShape(Rectangle())
        }.buttonStyle(.plain).accessibilityAddTraits(tab == index ? .isSelected : []).accessibilityIdentifier(id)
    }
}

/// The round settings button used at the top right of native pages.
struct SettingsButton: View {
    @Environment(\.openSettings) private var openSettings
    var body: some View {
        Button(action: openSettings) {
            Image(systemName:"slider.horizontal.3").frame(width:44,height:44)
                .overlay(HoldShape().stroke(BB.line,lineWidth:1.5))
        }.buttonStyle(.plain).accessibilityLabel("设置与关于").accessibilityIdentifier("about")
    }
}
