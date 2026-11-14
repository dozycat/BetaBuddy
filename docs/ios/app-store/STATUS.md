# BetaBuddy 发布状态

2026-11-14 10:16（中国时间）：按用户要求，将现有可安装的 TestFlight **1.3.0 (13)** 提交 App Store 正式审核。Connect 确认 `1 Item Submitted`，已建立 1.3.0 正式版本并关联构建 13；简体中文与英语（美国）栏目继续使用英文文案，更新说明、相册功能描述和审核备注已保存。保留审核通过后自动发布、立即向全部用户供应更新和现有评分。提交前确认旧正式版 1.2.0 已为 `Ready for Distribution`。本次提交**不包含构建 14 的布局修正**；构建 14 在解锁后重试上传仍报账户权限错误，未出现在 Connect 构建列表。提交成功截图保存在本机忽略目录 `ios/build/appstore-1.3.0-13-submitted.png`。

2026-11-14 09:30（中国时间）：布局打磨已从独立 worktree 合入并推送 main（`f23e947`），签名归档 **1.3.0 (14)** 已完成，包内版本、加密声明、GymPack **1.3.3** / **`8db5933`** 及离线岩馆目录校验通过。上传配置与构建 13 成功时完全一致，但本次 Xcode 在账户检查阶段返回 `Failed to Use Accounts`，尚未上传成功；已请账户持有人解锁并检查 Xcode 账户认证。当前可安装的外测版本仍为构建 13，不能将构建 14 的归档成功写成已分发。测试范围和安装后检查项见 [release-1.3.0-build14.md](release-1.3.0-build14.md)。

2026-11-14 08:59（中国时间）：**1.3.0 (13) 已在 TestFlight 外测群组显示 Testing，可安装。** 本次在独立 worktree 准备并合入 main（发布提交 `1020d05`），固定使用 GymPack **1.3.3** / climber **`8db5933`**。Xcode 重新认证后于 08:53 上传成功（`EXPORT SUCCEEDED`），Apple 已完成处理。构建已自动加入现有内测群组；外测群组已加入构建 13，测试说明保存成功，已开启自动通知测试者。用户指定的邮件邀请于 08:54 发出，后台确认状态 Invited；邀请地址不写入仓库。现有 [TestFlight 邀请链接](https://testflight.apple.com/join/WZPw69Vx) 继续可用。

模拟器构建、核心测试和 iOS 单元／集成测试通过，签名归档成功，包内文件与固定的 1.3.3 快照逐字节一致。按用户要求跳过受锁屏影响的模拟器手动播放验证；该项及真机性能均未验证，不能将 TestFlight 可安装等同于动作或性能验证通过。测试重点见 [release-1.3.0-build13.md](release-1.3.0-build13.md)。本次仅分发 TestFlight，没有提交 App Store 正式版审核。发布及邀请截图保存在本机忽略目录 `ios/build/`。

2026-11-14 00:50（中国时间）：攀爬册文件夹、组合筛选和离线岩馆目录已合入远端 main（功能提交 `9907175`，发布提交 `7034d9b`）。iOS **1.3.0 (12)** 已签名归档，于 00:38 上传成功（Xcode `EXPORT SUCCEEDED`）并完成 Apple 处理。现有「BetaBuddy 内部测试」群组已自动分发，状态 **Testing**，可安装；「BetaBuddy 邀请测试」已加入构建 12 并提交 Beta App Review，状态 **Waiting for Review**，已勾选审核通过后自动通知测试者。公开邀请链接继续使用 [TestFlight](https://testflight.apple.com/join/WZPw69Vx)，新版本需待外测审核通过后才对外可用。英文 What to Test 已保存，内容见 [release-1.3.0.md](release-1.3.0.md)。本次没有提交 App Store 正式版本审核。

归档内版本、非豁免加密声明、342 条岩馆目录以及本机已安装的 GymPack 均已核对。构建 ID：`69e51629-9e4d-4ee1-90e8-c955ca803cd1`；[TestFlight 构建](https://appstoreconnect.apple.com/teams/5ed1fb00-fca4-4901-b156-1f383d335b4e/apps/6809249574/testflight/ios/69e51629-9e4d-4ee1-90e8-c955ca803cd1)。外测状态截图保存在本机忽略目录 `ios/build/testflight-1.3.0-12-review.png`。

2026-11-13 13:37（中国时间）：1.2.0 (10) 与首个非消耗型内购 `com.betabuddy.ios.lifetime` 已在同一份新提交中正式送审，App Store Connect 显示两项均“等待审核”。提交 ID：`d559c99f-190c-43e0-99f0-c77d22159051`；[审核记录](https://appstoreconnect.apple.com/apps/6809249574/distribution/reviewsubmissions/details/d559c99f-190c-43e0-99f0-c77d22159051)。构建 10 已成功上传、完成出口合规并关联到 1.2.0；英文商店文案、截图、内购本地化与审核截图均已保存。Cloudflare 技术支持站点 `https://betabuddy.dozycat.ai` 正常返回 200。旧的构建 9 提交已撤回；当前可分发正式版仍为 1.0.0，1.2.0 尚待 Apple 审核。

2026-11-13 下午早些时候：Apple 以 Guideline 2.2（Beta Testing）拒绝了 1.2.0 (9)，认为 App 看起来像功能有限的预发布版本；截图是“设置与关于”页。已在审核消息中说明攀岩术语 beta 的含义，并询问具体未完成的功能。修正版 1.2.0 (10) 将歧义标题 `YOUR BETA` 改为 `YOUR CLIMB`，移除会员页的未来计划，并使 App 内离线隐私文案只陈述当前权益。App Store Connect 的简体中文及英语（美国）描述、更新说明、审核备注与副标题已改为英文并保存；内购的五种地区本地化名称和描述已统一为英文，明确一次性购买立即获得永久徽章。新的 1206 × 2622 英文内购审核截图已上传。Cloudflare 支持页已部署新版。Xcode 账户重新登录后，构建 10 上传成功；最新提交状态见上方。

2026-11-13：按用户要求，1.1.0 (5) 的审核提交已取消，App Store Connect 显示“被开发者拒绝”；版本记录改为 1.2.0。1.2.0 的简体中文和英语（美国）商店文案、更新说明、关键词、技术支持 URL 和中／大屏 iPhone 截图均已改成英文。两种语言的隐私政策 URL 均改为 `https://betabuddy.dozycat.ai/privacy`，会员内购的两种商店本地化也都使用英文。英文支持与隐私站点已部署在 Cloudflare 静态 Worker `betabuddy-support`，`https://betabuddy.dozycat.ai` 和 `/privacy` 均返回 200。App 内的帮助与隐私外链已改为新域名；岩馆内嵌页面只能导航到安装包内的 `gympack://local`。1.2.0 (9) 已签名归档、上传并完成出口合规，归档内确认版本 9、GymPack 与新 URL；构建 9 和首个终生会员内购已在同一个审核提交中正式送审，均显示“等待审核”。提交 ID：`f851acda-7e1b-4c10-aaca-b5de4e7caeba`，提交时间 2026-11-13 08:16（中国时间）。现有 1.0.0 的商店素材仍会在 1.2.0 发布前展示；Apple 不允许在缺少 1.0.0 英语截图时把 App 的主要语言从简体中文切成英语，但 1.2.0 的两种商店语言内容都已是英文。

2026-11-13：1.2.0 (8) 已签名归档并上传 App Store Connect，Apple 处理完成；出口合规答复已保存，构建状态“准备提交”。攀爬册顶部以“会员权益”引导到详情页，明确区分购买后立即获得的岩点会员标识和仍在计划中的功能；购买按钮直接调用 StoreKit 2 的系统购买流程。主按钮、播放等控件采用岩点轮廓。英文界面定向 UI 测试通过，归档内 GymPack 1.2.1 清单与本地安装包 SHA-256 一致。首个内购的审核截图已换成新版会员页，审核备注已同步并显示“已保存”。1.1.0 仍显示“正在审核”，1.2.0 暂无法建立商店版本。Apple 账户密码／Face ID 验证由系统及设备设置决定，App 不能跳过。

2026-11-13：1.2.0 (7) 终生会员版本已签名归档并成功上传 App Store Connect，二进制文件状态“已验证”；尚未提审。App Store Connect 已创建非消耗型内购 `com.betabuddy.ios.lifetime`，美国价格 $10，状态“准备提交”；内购审核截图已上传，仍需随新版 App 一同提交。Apple 要求当前 1.1.0 状态先成为“可供分发”，才能建立 1.2.0 商店版本；首个非消耗型内购必须随新版一起送审，不能单独提交。本地实现使用 StoreKit 2 的已验证权益决定会员标识，购买和恢复入口在攀爬册顶部“会员权益”和“设置与关于”通往的详情页；训练建议和新岩馆数据仍属后续计划。线上支持与隐私站点已更新发布。付费 App 协议状态“有效”；收款银行账户状态“可用”。美国税表 U.S. Certificate of Foreign Status of Beneficial Owner 和 W-8BEN 均于 11 月 13 日提交，状态“使用中”；中国大陆国务院令第 810 号合规资料已显示“有效”。域名 `dozycat.ai` 已切到 Cloudflare 权威服务器；支付核验 Worker 代码与 `pay.dozycat.ai` 自定义域名已部署，健康检查及无效签名拒绝测试通过。Apple 内购 API 的 Issuer ID、Key ID 和私钥已保存为 `betabuddy-pay` Production Secret；真实交易核验仍待沙盒交易验证，当前 App 不调用此服务。下文 1.1.0 的“无内购”描述指该已提交构建。

2026-11-13：`pay.dozycat.ai` 的普通 DNS 访问已恢复，`GET /health` 返回 200，伪造交易请求返回 `active:false`；先前本机连接失败未改 DNS 即自行消除。已加入独立的 `BetaBuddyStoreKit` 本地模拟商品 scheme；Xcode 27.2 命令行测试在 iOS 26.5 和 27.0 模拟器均未加载商品，待图形界面或真机沙盒测试。

2026-11-13：Xcode 27.2 图形界面用 `BetaBuddyStoreKit` scheme 在 iOS 27.0 的 iPhone 18 Pro 模拟器显示本地商品 $10.00；模拟购买后“终生会员已解锁”与攀爬册徽章立即出现，停止并重启 App 后徽章仍在。购买页的 1206 × 2622 截图已上传 App Store Connect 的内购审核信息，副本保存在本机忽略目录 `ios/build/lifetime-iap-review-1.2.0.png`；徽章截图副本为 `ios/build/lifetime-member-badge-1.2.0.png`。这验证的是 Xcode 本地 StoreKit 配置，尚需在付费协议生效后做真机沙盒购买、恢复和 Worker 交易核验。

应用内语言设置（底部齿轮 → 设置与关于 → 语言，选了立即生效）、系统语言不在五种之内时回落英语、未命名记录按当前语言显示，这些改动已包含在 1.2.0 (7) 构建中。

2026-11-13：1.1.0 (5) 已签名归档并上传，关联到 App Store Connect 的 1.1.0 版本，出口合规已完成。构建包含场馆数据包 1.2.1、72 条线路和五种界面语言；在 iPhone 17 Pro Max 上验证岩馆、线路、视频导入、分析与回看。中英商店文案、截图、审核备注、支持与隐私页面已更新。真机操作录屏已作为审核附件上传；版本已正式提交 Apple 审核。

## 商店送审历史（2026-11-13 13:37）

**当时正式版为 1.0.0；1.2.0 (10) 与终生会员内购已联合提交，均等待 Apple 审核。** 当前提交 ID：`d559c99f-190c-43e0-99f0-c77d22159051`；[审核记录](https://appstoreconnect.apple.com/apps/6809249574/distribution/reviewsubmissions/details/d559c99f-190c-43e0-99f0-c77d22159051)。构建 9 曾被拒绝，其[旧提交](https://appstoreconnect.apple.com/apps/6809249574/distribution/reviewsubmissions/details/f851acda-7e1b-4c10-aaca-b5de4e7caeba)已撤回。下方构建 3–5 为 1.1.0 历史记录，该版本的审核已取消。

- 构建 3（2026-11-12 23:35 CST 上传成功）：四个入口的界面，只有简体中文。
- 构建 4：两个入口——岩馆在前，攀爬册在后，导入并入攀爬册；支持简体中文、繁体中文、英语、日语、韩语。已上传，不提审。
- 构建 5：补齐岩馆名称翻译，装入场馆数据包 1.2.1，2026-11-13 上传成功并选为提审构建。

设计见 [../product-design.md](../product-design.md) 和 [../gym-design.md](../gym-design.md)，文案与提审清单见 [release-1.1.0.md](release-1.1.0.md)，流程见 [RELEASE.md](RELEASE.md)。

审核附件为 44 秒 iPhone 17 Pro Max 实机操作录屏，覆盖岩馆选墙、选线、播放和身材切换，以及攀爬册视频导入、机上分析、回看和保存记录。审核备注说明了操作路径、离线功能、地区供应及示例视频中攀爬者较小造成的识别局限。

1.0.0 已在所选销售地区上架；1.2.0 尚未上架。

## 经过

2026-11-12：1.1.0 (3) 的代码合入 main。装入场馆数据包后的 Release 归档构建通过（app 约 55 MB，其中数据包 11 MB）；开发签名版装到账号持有人的 iPhone 上用于真机验收。同日签名归档、导出并上传成功。导出时需要系统自带的 rsync，见 [RELEASE.md](RELEASE.md)。

2026-09-07：1.0.0 (1) 已被 Apple 以 Guideline 2.1「Information Needed」拒绝。要求最新 iOS 真机全流程录屏及用途、操作、外部服务和地区差异说明；尚未上架。

已建立「BetaBuddy 内部测试」群组，账户持有人已加入测试，启用自动分发。构建 1 已可测试。构建 2 修复本次用户视频的背景误检、旋转视频预处理，并支持保留笔记重新分析；签名归档及上传成功（16:23），出口合规已保存，内部群组已自动获得构建 2，后台状态为「正在测试」。构建 2 测试说明已保存。

## 后台与版本

- App Store Connect 的应用链接和 Apple ID：见账户持有人的私有记录。
- Bundle ID：com.betabuddy.ios
- SKU：betabuddy-ios-001
- 个人团队：团队 ID 不写在仓库里，构建时用 `DEVELOPMENT_TEAM=<团队 ID>` 传入。
- 发布方式：审核通过后自动发布。
- App Store 初始价格：所有货币均为 0；无内购、订阅和广告。
- 首发 147 个国家／地区：排除中国大陆与欧盟 27 国；未启用将来新增地区自动供应。用户已确认先发布，之后补齐资料再扩展。
- 首发聚焦 iPhone，关闭 Apple Silicon Mac 和 Apple Vision Pro 的兼容 App 供应。

## 已完成的发布资料

- 中文名称、副标题、描述、关键词和推广文本；主要类别为体育。
- 审核联系人已按用户提供的信息填写，支持邮箱 hi@dozycat.ai。
- 支持网站：https://betabuddy.dozycat.ai
- 隐私政策：https://betabuddy.dozycat.ai/privacy
- 两个公开网址无需登录，HTTP 200，应用内已链接；同时提供离线政策。
- 隐私标签“不收集数据”已在后台发布。
- 1.1.0 中英文各 4 张截图已上传至 6.9 英寸（1320 × 2868）和必需的中尺寸（1206 × 2622）栏目。
- 年龄分级问卷主要地区结果为 9+；低于 iOS 26 的系统为 4+，区域例外以 Apple 为准。
- 已保存内容版权答复及本版出口加密问卷。
- 用户明确授权接受更新协议；个人账户确认接受 2026-08-18 版 Apple Developer Program License Agreement，接受日期 2026-09-07。

## 1.0.0 的构建与网站源

- 最终签名归档：ios/build/BetaBuddy-1.0.0-final.xcarchive
- App Store IPA：ios/build/AppStore-final/BetaBuddy.ipa（约 24.2 MiB）
- 通过 Cloud Managed Apple Distribution 和 App Store 描述文件导出并上传成功。
- 导出配置：ios/scripts/ExportOptions.plist
- 网站独立版本库：release-site/，由 Sites 保存并公开部署。
- 交付资源检查通过：Bundle ID、版本、支持邮箱、隐私政策与 required-reason 声明。
- 隐私清单为静态链接代码中的 stat/fstat、NSUserDefaults 声明 C617.1 与 CA92.1。

## 地区的后续

审核通过后自动在已选地区上架。中国大陆需补齐适用的 App 备案信息后再增加，备案资料含个人信息，不放在这个仓库里；欧盟需先完成交易商身份声明，再开放供应。

真机上的攀岩准确性、长期稳定性和热状态还没有成文的验收结论。模拟器和集成测试的范围见 [../validation.md](../validation.md)。Apple 接受提审不代表这些测试已经完成。
