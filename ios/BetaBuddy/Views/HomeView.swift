import SwiftUI

struct HomeView: View {
    @EnvironmentObject private var store: ClimbStore
    @Binding var importing: Bool
    @Binding var tab: Int
    var body: some View {
        PaperPage {
            HStack(alignment:.center) {
                HStack(spacing:6) {
                    BrandMark().fill(BB.ink).frame(width:30,height:30).accessibilityHidden(true)
                    Text("betabuddy").font(.system(size:23,weight:.black,design:.rounded)).tracking(-1)
                }
                Spacer()
                NavigationLink { AboutView() } label: {
                    Image(systemName:"slider.horizontal.3").frame(width:44,height:44).overlay(Circle().stroke(BB.line))
                }.accessibilityLabel("关于与本地存储")
            }
            VStack(alignment:.leading,spacing:9) {
                Text("YOUR NEXT MOVE").font(.system(size:11,weight:.bold,design:.monospaced)).tracking(2).foregroundStyle(BB.muted)
                Text("再试一次。\n这次，有点不一样。").font(.system(size:31,weight:.heavy)).tracking(-1).lineSpacing(3)
            }
            ZStack(alignment:.topLeading) {
                BB.lime
                HStack(spacing:0) {
                    VStack(alignment:.leading,spacing:15) {
                        Stamp(text:"CLIMB / REPLAY")
                        Spacer(minLength:12)
                        Text("MAKE\nYOUR\nBETA.").font(BB.display(39)).tracking(-1.8).lineSpacing(-5).minimumScaleFactor(0.75)
                        Text("每一趟，都有新发现。").font(.system(size:12,weight:.medium))
                    }.frame(maxWidth:.infinity,alignment:.leading)
                    ClimbingArt().frame(width:145)
                }.padding(22)
            }.frame(height:268).clipShape(RoundedRectangle(cornerRadius:26))
            VStack(spacing:12) {
                BigButton(title:"导入这一趟", icon:"plus") { importing = true }
                    .accessibilityIdentifier("importClimb")
                Label("视频留在手机里 · 无需账号",systemImage:"lock.fill")
                    .font(.system(size:11,weight:.medium)).foregroundStyle(BB.muted)
            }
            HStack {
                Stat(value:"\(store.climbs.count)",label:"记录的攀爬")
                Stat(value:"\(store.climbs.filter(\.sent).count)",label:"自己标记完攀")
                Stat(value:"\(store.archive.training.count)",label:"完成的练习")
            }.padding(.vertical,8)
            SectionLabel(title:"留给下一趟",trailing:"ONE THING AT A TIME")
            Button { tab = 2 } label: {
                HStack(alignment:.top,spacing:16) {
                    Text("01").font(.system(size:36,weight:.black,design:.rounded)).foregroundStyle(BB.ink.opacity(0.4))
                    VStack(alignment:.leading,spacing:8) {
                        Text("先练一次，安静落脚。").font(.headline)
                        Text("别急着追难度。让每一次踩点\n更有把握。").font(.subheadline).lineSpacing(4).foregroundStyle(BB.muted)
                        Text("8 分钟的小练习  ↗").font(.caption.bold()).padding(.top,4)
                    }
                    Spacer(minLength:0)
                }.padding(22).background(BB.lavender,in:RoundedRectangle(cornerRadius:22))
            }.buttonStyle(.plain)
            if let climb = store.climbs.first {
                SectionLabel(title:"接着回看",trailing:"LATEST CLIMB")
                NavigationLink { ReviewView(climb:climb) } label: { ClimbRow(climb:climb) }.buttonStyle(.plain)
            }
        }.toolbar(.hidden,for:.navigationBar)
    }
}

struct AboutView: View {
    @EnvironmentObject private var store: ClimbStore
    var body: some View {
        PaperPage {
            Text("YOUR BETA.\nYOUR DEVICE.").font(BB.display(37))
            Label("完全在这台 iPhone 上分析",systemImage:"lock.shield").font(.title3.bold())
            Text("全部功能免费 · 无内购 · 无订阅 · 无广告").font(.subheadline.bold())
            Text("MediaPipe Full 模型随安装包提供。没有模型下载，没有账号，也没有 LLM。导入的视频与分析记录保存在应用内，并排除在系统云备份之外。").foregroundStyle(BB.muted)
            Divider()
            SectionLabel(title:"分析边界",trailing:"KEEP IT HONEST")
            Text("33 个关键点用于骨架回放与画面内几何分析。角度是二维投影角度；髋部轨迹不是物理重心。遮挡、镜头移动、多人入镜会影响结果。候选片段需要结合原片确认。")
            Text("第一版：单人、全身入镜、固定机位；单段 0.5–180 秒，最大 500 MB。建议先在相册里裁掉无关片段。").foregroundStyle(BB.muted)
            Divider()
            Text("记录管理").font(.headline)
            Text("已保存 \(store.climbs.count) 段攀爬。可在每条记录中删除原片和分析；卸载应用会移除本地记录。需要保留的复盘文字可通过分享导出。").foregroundStyle(BB.muted)
            NavigationLink("隐私政策") { PrivacyPolicyView() }
                .font(.headline).padding(.vertical, 12)
            Link("在线隐私政策", destination: URL(string: "https://betabuddy-support.dozycfs.chatgpt.site/privacy")!)
            Link("帮助与支持", destination: URL(string: "https://betabuddy-support.dozycfs.chatgpt.site")!)
            Link("hi@dozycat.ai", destination: URL(string: "mailto:hi@dozycat.ai")!)
            Text("BETABUDDY / \(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "1.0.0")\nBuilt for the next attempt.").font(.system(size:12,design:.monospaced)).lineSpacing(6)
        }.navigationTitle("关于 BetaBuddy").navigationBarTitleDisplayMode(.inline)
    }
}

struct PrivacyPolicyView: View {
    private var policy: String {
        guard let url = Bundle.main.url(forResource: "PrivacyPolicy", withExtension: "txt"),
              let text = try? String(contentsOf: url, encoding: .utf8) else {
            return "隐私政策无法读取，请重新安装完整版本。"
        }
        return text
    }

    var body: some View {
        PaperPage {
            Text(policy).font(.body).lineSpacing(7).textSelection(.enabled)
        }.navigationTitle("隐私政策").navigationBarTitleDisplayMode(.inline)
    }
}
