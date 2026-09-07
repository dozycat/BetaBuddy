# BetaBuddy 1.0.0 免费发布

最新状态与网站地址见 [STATUS.md](STATUS.md)。下文保留操作参考，签名与网站步骤现已由代理完成。现已提交 Apple 审核，审核通过后自动发布；尚未上架。免费是应用价格与功能策略，不代表免除 Apple 开发者会员费用。

## 已准备

- iPhone 原生应用、内置 Full 姿态模型、新版攀岩扣图标。
- 版本 1.0.0，构建号 1。
- 全功能免费，无内购、订阅或广告；关于页明确说明。
- 应用内离线隐私政策，支持邮箱 hi@dozycat.ai。
- `metadata-zh-Hans.md`：商店描述、关键词、审核备注与免费价格设置。
- `privacy-policy.txt`、`support.md`：公开网页的内容源，尚未托管。

## 1. 完成真机验收

连接自己的 iPhone，在 Xcode 的 Signing & Capabilities 选择个人开发者 Team。先执行 Build & Run，用真实攀岩视频验证导入、骨架回放、笔记、删除、取消分析和后台切换。测试较长视频的内存、温度与速度。已有模拟器与集成测试结果不等于真机验收，详见 `../validation.md`。

## 2. 建立公开网页

将隐私政策和支持资料放到自己的公开 HTTPS 网站，可以使用已有网站或免费的静态托管。页面无需登录即可访问；支持页包含 hi@dozycat.ai。将真实的隐私网页链接加到应用关于页，并在 App Store Connect 填写 Privacy Policy URL 与 Support URL。当前离线政策已可在应用内阅读，但不能替代后台要求的网址。

## 3. 签名归档

在 Xcode 打开 `ios/BetaBuddy.xcodeproj`：

1. 选择 BetaBuddy target → Signing & Capabilities，选自己的 Team，开启 Automatically manage signing。
2. 确认 Bundle ID `com.betabuddy.ios` 可被该团队注册；如不可用，修改 `ios/project.yml` 后重新生成项目，并在后台使用相同 ID。
3. 选择 Any iOS Device (arm64)，Product → Archive。
4. Organizer → Validate App，解决 Apple 返回的签名、隐私清单与 SDK 校验问题。
5. Distribute App → App Store Connect → Upload。上传后的构建等待 Apple 处理。

本地 `ios/build/BetaBuddy-1.0.0-unsigned.xcarchive` 是**未签名的构建验证归档**，不是可直接上架的 IPA。不要把它当成已签名分发包。现已通过 Xcode 的个人团队完成自动签名，并使用 Cloud Managed Apple Distribution 导出；当前产物请使用 STATUS.md 中的最终路径。

每次重新上传同一版本必须递增 `CURRENT_PROJECT_VERSION`，并同步保存到 `ios/project.yml`。

## 4. App Store Connect

1. Apps → 新建 iOS App，填写名称 BetaBuddy、简体中文、正确 Bundle ID、自己的唯一 SKU。
2. 粘贴同目录商店文案，上传符合 Apple 尺寸要求的真实应用截图。使用 screenshots-6.9/ 下新准备的三张 1320 × 2868 真实界面截图；不要用非攀岩模型验证片宣传攀岩效果。
3. Pricing and Availability：设置 Free / 0，选择实际计划发布的地区；不创建任何内购或订阅。
4. App Privacy：按最终实现填写。目前应用不向开发者或合作伙伴传输个人数据，拟选择“不收集数据”；提交前复核最终 SDK、隐私清单和 Apple 验证报告。
5. 完成年龄分级、内容权利、加密出口等问卷。中国大陆如适用，需完成对应备案信息；其他地区也以后台实际要求为准。
6. 填写开发者真实审核联系人与电话，支持邮箱使用 hi@dozycat.ai。隐私网址和支持网址填已上线地址。
7. 选择已处理成功的构建，建议先通过 TestFlight 真机验证，然后提交审核。发布方式可选审核通过后手动发布。

## 官方依据（2026-09-07 查阅）

- 开发者会员：https://developer.apple.com/programs/
- 免费定价：https://developer.apple.com/help/app-store-connect/manage-app-pricing/set-a-price
- 隐私说明：https://developer.apple.com/help/app-store-connect/manage-app-information/manage-app-privacy/
- 审核要求：https://developer.apple.com/app-store/review/guidelines/
- 截图尺寸：https://developer.apple.com/help/app-store-connect/reference/app-information/screenshot-specifications/
- 地区资料：https://developer.apple.com/help/app-store-connect/reference/app-information/app-information/
