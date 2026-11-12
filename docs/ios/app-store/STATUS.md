# BetaBuddy 发布状态

2026-11-13：1.1.0 (5) 已签名归档并上传，关联到 App Store Connect 的 1.1.0 版本，出口合规已完成。构建包含场馆数据包 1.2.1、72 条线路和五种界面语言；在 iPhone 17 Pro Max 上确认英语岩馆与线路显示正常。中英商店文案、截图、审核备注、支持与隐私页面已更新。待附上 Apple 上次要求的真机全流程录屏后提交审核。

## 现在

**1.1.0 (5) 已上传并选为提审构建；版本仍为“准备提交”。**

- 构建 3（2026-11-12 23:35 CST 上传成功）：四个入口的界面，只有简体中文。
- 构建 4：两个入口——岩馆在前，攀爬册在后，导入并入攀爬册；支持简体中文、繁体中文、英语、日语、韩语。已上传，不提审。
- 构建 5：补齐岩馆名称翻译，装入场馆数据包 1.2.1，2026-11-13 上传成功并选为提审构建。

设计见 [../product-design.md](../product-design.md) 和 [../gym-design.md](../gym-design.md)，文案与提审清单见 [release-1.1.0.md](release-1.1.0.md)，流程见 [RELEASE.md](RELEASE.md)。

提审前还差：真机全流程录屏和说明（Apple 上次按 Guideline 2.1 要求的材料），上传为审核附件，再将版本提交审核。

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
- 隐私政策：https://betabuddy-support.dozycfs.chatgpt.site/privacy
- 两个公开网址无需登录，HTTP 200，应用内已链接；同时提供离线政策。
- 隐私标签“不收集数据”已在后台发布。
- 3 张真实截图已上传至 6.9 英寸栏目，1320 × 2868，小尺寸自动使用这些文件。
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
