# BetaBuddy 发布状态

2026-11-13：1.2.0 (7) 终生会员版本已签名归档并成功上传 App Store Connect，二进制文件状态“已验证”；尚未提审。App Store Connect 已创建非消耗型内购 `com.betabuddy.ios.lifetime`，美国价格 $10，状态“准备提交”，仍需补审核截图并随新版 App 一同提交。Apple 要求当前 1.1.0 状态先成为“可供分发”，才能建立 1.2.0 商店版本；首个非消耗型内购必须随新版一起送审，不能单独提交。本地实现使用 StoreKit 2 的已验证权益决定会员标识，购买和恢复入口在“设置与关于”；训练建议和新岩馆数据仍属后续计划。线上支持与隐私站点已更新发布。付费 App 协议已接受，当前状态为“等待用户信息”；收款银行资料“正在处理”，美国税表待签署，中国大陆国务院令第 810 号合规资料已显示“有效”。域名 `dozycat.ai` 已切到 Cloudflare 权威服务器；支付核验 Worker 代码与 `pay.dozycat.ai` 自定义域名已部署，健康检查及无效签名拒绝测试通过，但 Apple 私钥尚未作为 Worker 密钥保存，真实交易核验未完成；当前 App 不调用此服务。下文 1.1.0 的“无内购”描述指该已提交构建。

应用内语言设置（底部齿轮 → 设置与关于 → 语言，选了立即生效）、系统语言不在五种之内时回落英语、未命名记录按当前语言显示，这些改动已包含在 1.2.0 (7) 构建中。

2026-11-13：1.1.0 (5) 已签名归档并上传，关联到 App Store Connect 的 1.1.0 版本，出口合规已完成。构建包含场馆数据包 1.2.1、72 条线路和五种界面语言；在 iPhone 17 Pro Max 上验证岩馆、线路、视频导入、分析与回看。中英商店文案、截图、审核备注、支持与隐私页面已更新。真机操作录屏已作为审核附件上传；版本已正式提交 Apple 审核。

## 现在

**1.1.0 (5) 已提交 Apple，App Store Connect 状态为“等待审核”。** 提交 ID：`5e045882-a264-43ba-9819-269ad676f6b0`；[审核记录](https://appstoreconnect.apple.com/apps/6809249574/distribution/reviewsubmissions/details/5e045882-a264-43ba-9819-269ad676f6b0)。审核通过后按后台设置自动发布。

- 构建 3（2026-11-12 23:35 CST 上传成功）：四个入口的界面，只有简体中文。
- 构建 4：两个入口——岩馆在前，攀爬册在后，导入并入攀爬册；支持简体中文、繁体中文、英语、日语、韩语。已上传，不提审。
- 构建 5：补齐岩馆名称翻译，装入场馆数据包 1.2.1，2026-11-13 上传成功并选为提审构建。

设计见 [../product-design.md](../product-design.md) 和 [../gym-design.md](../gym-design.md)，文案与提审清单见 [release-1.1.0.md](release-1.1.0.md)，流程见 [RELEASE.md](RELEASE.md)。

审核附件为 44 秒 iPhone 17 Pro Max 实机操作录屏，覆盖岩馆选墙、选线、播放和身材切换，以及攀爬册视频导入、机上分析、回看和保存记录。审核备注说明了操作路径、离线功能、地区供应及示例视频中攀爬者较小造成的识别局限。

应用目前没有在任何地区上架。

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
- 支持网站：https://betabuddy-support.dozycfs.chatgpt.site
- 隐私政策：https://betabuddy-support.dozycfs.chatgpt.site/privacy；英文版：https://betabuddy-support.dozycfs.chatgpt.site/privacy/en
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
