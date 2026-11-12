import SwiftUI

struct AboutView: View {
    @EnvironmentObject private var store: ClimbStore
    @EnvironmentObject private var language: LanguageSettings
    @EnvironmentObject private var membership: MembershipStore
    @Environment(\.dismiss) private var dismiss
    @State private var showingMembership = false
    private var version: String { Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "" }
    var body: some View {
        PaperPage {
            VStack(alignment:.leading,spacing:12) {
                Label("语言",systemImage:"globe").font(.title3.bold())
                VStack(spacing:0) {
                    languageRow(nil)
                    ForEach(AppLanguage.supported,id:\.self) { languageRow($0) }
                }.background(.white.opacity(0.6),in:RoundedRectangle(cornerRadius:18))
            }
            Divider()
            VStack(alignment:.leading,spacing:12) {
                HStack(spacing:10) {
                    MemberMark().frame(width:30,height:30)
                    Text("终生会员").font(.title3.bold())
                }
                if membership.isMember {
                    Text("终生会员已解锁").font(.headline).accessibilityIdentifier("membershipActive")
                } else {
                    Text("支持独立开发，现在获得会员标识。")
                        .foregroundStyle(BB.muted)
                    Button { showingMembership = true } label: {
                        HStack(spacing:12) {
                            BoltMark()
                            Text("查看会员权益").font(.headline)
                            Spacer()
                            Image(systemName:"arrow.up.right")
                        }.padding(14)
                    }.buttonStyle(RouteButtonStyle(tone:.lime))
                        .accessibilityIdentifier("membershipDetails")
                }
            }.padding(18).frame(maxWidth:.infinity,alignment:.leading)
                .background(BB.lavender,in:RoundedRectangle(cornerRadius:20))
            Divider()
            Text(verbatim:"YOUR BETA.\nYOUR DEVICE.").font(BB.display(37))
            Label("完全在这台 iPhone 上分析",systemImage:"lock.shield").font(.title3.bold())
            Text("核心功能免费 · 终生会员为一次性内购 · 无订阅 · 无广告").font(.subheadline.bold())
            Text("姿态模型随安装包提供。没有模型下载，没有账号，也没有云端 AI。导入的视频与分析记录保存在应用内，并排除在系统云备份之外。").foregroundStyle(BB.muted)
            Divider()
            SectionLabel(title:"分析边界",trailing:"KEEP IT HONEST")
            Text("33 个关键点用于骨架回放与画面内几何分析。角度是二维投影角度；髋部轨迹不是物理重心。遮挡、镜头移动、多人入镜会影响结果。候选片段需要结合原片确认。")
            Text("适合单人、全身入镜、固定机位的视频；单段 0.5–180 秒，最大 500 MB。建议先在相册里裁掉无关片段。").foregroundStyle(BB.muted)
            Divider()
            if let pack = GymPack.current {
                GymNoticeList()
                Text("\(pack.manifest.displayName()) · \(pack.manifest.routeCount) 条线路 · 数据版本 \(pack.manifest.packVersion)（\(String(pack.manifest.generatedAt.prefix(10)))）").font(.caption).foregroundStyle(BB.muted)
            } else {
                SectionLabel(title:"岩馆",trailing:"NOT IN THIS BUILD")
                Text("这个构建没有内置岩馆数据。场馆模型、线路和攀爬模拟随正式安装包提供，不在公开源码里。").foregroundStyle(BB.muted)
            }
            Divider()
            Text("记录管理").font(.headline)
            Text("已保存 \(store.climbs.count) 段攀爬。可在每条记录中删除原片和分析；卸载应用会移除本地记录。需要保留的复盘文字可通过分享导出。").foregroundStyle(BB.muted)
            NavigationLink("隐私政策") { PrivacyPolicyView() }
                .font(.headline).padding(.vertical, 12).accessibilityIdentifier("privacyPolicy")
            Link("在线隐私政策", destination: URL(string: "https://betabuddy-support.dozycfs.chatgpt.site/privacy")!)
            Link("帮助与支持", destination: URL(string: "https://betabuddy-support.dozycfs.chatgpt.site")!)
            Link(destination: URL(string: "mailto:hi@dozycat.ai")!) { Text(verbatim:"hi@dozycat.ai") }
            Text(verbatim:"BETABUDDY / \(version)\nBuilt for the next attempt.").font(.system(size:12,design:.monospaced)).lineSpacing(6)
        }.navigationTitle("设置与关于").navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement:.topBarTrailing) { Button("完成") { dismiss() }.accessibilityIdentifier("settingsDone") }
            }
            .sheet(isPresented:$showingMembership) { NavigationStack { MembershipOfferView() } }
    }
    // Each language is written in itself; only “follow system” is translated.
    private func languageRow(_ code: String?) -> some View {
        Button { language.select(code) } label: {
            HStack {
                if let code { Text(verbatim:AppLanguage.nativeNames[code] ?? code) } else { Text("跟随系统") }
                Spacer()
                if language.selection == code { Image(systemName:"checkmark").font(.subheadline.bold()) }
            }.padding(.horizontal,16).frame(minHeight:48).contentShape(Rectangle())
        }.buttonStyle(.plain).accessibilityIdentifier("language-\(code ?? "system")")
            .accessibilityAddTraits(language.selection == code ? .isSelected : [])
            .overlay(alignment:.bottom) { if code != AppLanguage.supported.last { Rectangle().fill(BB.line).frame(height:1).padding(.leading,16) } }
    }
}

struct MembershipOfferView: View {
    @EnvironmentObject private var membership: MembershipStore
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        PaperPage {
            HStack {
                Stamp(text:"BETABUDDY / LIFETIME")
                Spacer()
                MemberMark().frame(width:54,height:54)
            }
            VStack(alignment:.leading,spacing:12) {
                Text("终生会员").font(BB.display(38))
                Text("一次购买，支持 BetaBuddy 继续迭代。")
                    .font(.title3.bold())
                Text("岩馆与攀爬册仍可免费使用；会员购买是可选的。")
                    .foregroundStyle(BB.muted)
            }
            VStack(alignment:.leading,spacing:16) {
                SectionLabel(title:"购买后立即获得",trailing:"AVAILABLE NOW")
                    .accessibilityIdentifier("membershipBenefitsNow")
                HStack(spacing:12) {
                    MemberMark().frame(width:32,height:32)
                    Text("攀爬册终生会员标识").font(.headline)
                }
                Text("购买成功后，标识会显示在攀爬册顶部。")
                    .foregroundStyle(BB.muted)
            }.padding(20).frame(maxWidth:.infinity,alignment:.leading)
                .background(BB.lavender,in:RoundedRectangle(cornerRadius:22))
            if membership.isMember {
                Label("终生会员已解锁",systemImage:"checkmark.seal.fill")
                    .font(.headline).accessibilityIdentifier("membershipActive")
            } else {
                VStack(alignment:.leading,spacing:12) {
                    if let product = membership.product {
                        Button { Task { await membership.purchase() } } label: {
                            HStack {
                                BoltMark()
                                Text("解锁终生会员")
                                Spacer()
                                Text(verbatim:product.displayPrice)
                            }.font(.headline).padding(18).frame(maxWidth:.infinity)
                        }.buttonStyle(RouteButtonStyle(tone:.lime))
                            .disabled(membership.isBusy).accessibilityIdentifier("buyLifetime")
                    } else {
                        Text("暂时无法连接 App Store。请稍后重试。").foregroundStyle(BB.muted)
                        Button("重试连接 App Store") { Task { await membership.load() } }
                            .accessibilityIdentifier("retryMembershipProduct")
                    }
                    Button("恢复购买") { Task { await membership.restore() } }
                        .disabled(membership.isBusy).accessibilityIdentifier("restoreLifetime")
                    Text("一次性购买，无订阅。购买由 Apple 处理，可在同一 Apple 账户的设备上恢复。")
                        .font(.caption).foregroundStyle(BB.muted)
                    if membership.isBusy { ProgressView() }
                }
            }
            VStack(alignment:.leading,spacing:12) {
                SectionLabel(title:"未来计划",trailing:"PLANNED")
                    .accessibilityIdentifier("membershipBenefitsLater")
                Label("训练建议",systemImage:"figure.climbing")
                Label("新岩馆数据",systemImage:"map")
                Text("这些功能尚未上线，推出时间与具体内容可能调整。")
                    .font(.subheadline).foregroundStyle(BB.muted)
            }.padding(20).frame(maxWidth:.infinity,alignment:.leading)
                .background(.white.opacity(0.6),in:RoundedRectangle(cornerRadius:22))
        }.navigationTitle("终生会员").navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement:.topBarTrailing) { Button("完成") { dismiss() }.accessibilityIdentifier("membershipDone") }
            }
            .alert("暂时无法完成", isPresented: Binding(get: { membership.errorMessage != nil }, set: { if !$0 { membership.errorMessage = nil } })) {
                Button("知道了", role:.cancel) { membership.errorMessage = nil }
            } message: { Text(verbatim:membership.errorMessage ?? "") }
    }
}

struct PrivacyPolicyView: View {
    /// Simplified and Traditional Chinese and English are bundled; other languages read the English text.
    static func resourceName(for language: String) -> String {
        ["zh-Hans", "zh-Hant"].contains(language) ? "PrivacyPolicy.\(language)" : "PrivacyPolicy.en"
    }
    private var policy: String {
        guard let url = Bundle.main.url(forResource: Self.resourceName(for: AppLanguage.current), withExtension: "txt"),
              let text = try? String(contentsOf: url, encoding: .utf8) else {
            return String(localized: "隐私政策无法读取，请重新安装完整版本。", bundle: AppLanguage.bundle)
        }
        return text
    }

    var body: some View {
        PaperPage {
            Text(verbatim:policy).font(.body).lineSpacing(7).textSelection(.enabled)
        }.navigationTitle("隐私政策").navigationBarTitleDisplayMode(.inline)
    }
}
