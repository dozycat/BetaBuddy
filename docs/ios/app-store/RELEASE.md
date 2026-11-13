# 发布流程

从一份通过测试的代码到提交 App Store 审核，每个版本都走这几步。后台现在的状态见 [STATUS.md](STATUS.md)；每个版本自己的文案和清单在 `release-<版本>.md`；1.2.0 商店英文资料在 [metadata-en.md](metadata-en.md)。

BetaBuddy 的岩馆和攀爬册核心功能免费，没有订阅和广告。1.2.0 起提供可选的一次性终生会员内购；购买后立即显示会员标识。训练建议和新岩馆数据是后续计划，不应在审核资料中描述为已经提供。

## 1. 准备构建

1. 在 `ios/project.yml` 里设好 `MARKETING_VERSION` 和 `CURRENT_PROJECT_VERSION`。同一版本每次重新上传都要递增构建号。改完重新生成工程并提交。
2. 要内置岩馆的版本：`./ios/scripts/install-gym-pack.sh <数据包目录>`。脚本会核对清单和每个文件的 SHA-256。数据包不进仓库。
3. 跑一遍 [ios/README.md](../../../ios/README.md) 里的测试和未签名的真机构建。
4. 真机验收：用真实攀岩视频走一遍导入、分析、回看、笔记、删除、取消分析、切到后台，以及岩馆的选墙、选线、播放、换身材。模拟器和自动化测试的结果不等于真机验收，范围见 [../validation.md](../validation.md)。

## 2. 签名归档

需要在 Xcode → Settings → Accounts 里登录有权限的 Apple 账号，否则归档会以「No Accounts」失败。

在 Xcode 里：BetaBuddy target → Signing & Capabilities，选团队，开启 Automatically manage signing；目标选 Any iOS Device (arm64)，Product → Archive。

或用命令行：

```sh
xcodebuild -project ios/BetaBuddy.xcodeproj -scheme BetaBuddy -configuration Release \
  -destination 'generic/platform=iOS' -archivePath ios/build/BetaBuddy-<版本>-<构建号>.xcarchive \
  -allowProvisioningUpdates archive
```

Bundle ID 是 `com.betabuddy.ios`。换团队时如果这个 ID 不可注册，改 `ios/project.yml` 后重新生成工程，并在后台用同一个 ID。

## 3. 导出与上传

```sh
PATH=/usr/bin:/bin:/usr/sbin:/sbin xcodebuild -exportArchive \
  -archivePath ios/build/BetaBuddy-<版本>-<构建号>.xcarchive \
  -exportOptionsPlist ios/scripts/ExportOptions.plist \
  -exportPath ios/build/AppStore-<版本>-<构建号> -allowProvisioningUpdates
```

**导出时要用系统自带的 rsync。** 如果 `PATH` 里排在前面的是 Homebrew 的 rsync，Xcode 会报「Copy failed」。上面的命令把 `PATH` 限制在系统目录来避开它。

`ExportOptions.plist` 的方式是 `app-store-connect`。把其中的 `destination` 从 `export` 改成 `upload`（另存一份临时的 plist 即可），`xcodebuild -exportArchive` 会在导出后直接上传到 App Store Connect，1.1.0 (3) 就是这样上传的；需要 Xcode 的 Settings → Accounts 里已登录账号持有人的 Apple ID。也可以保持 `export`，再用 Xcode Organizer 或 Transporter 上传导出的 IPA。上传后等 Apple 处理完成，构建才会出现在后台。

`ios/build/` 被 Git 忽略。归档和 IPA 不进仓库。

**团队 ID 不写在仓库里。** 归档时在命令行加 `DEVELOPMENT_TEAM=<团队 ID>`（或在 Xcode 的 Signing & Capabilities 里选自己的团队，但不要把改动后的工程提交上来）；导出用的 plist 需要时自行加 `teamID`。证书、描述文件、Apple ID、App Store Connect 密钥同样不进仓库。

## 4. App Store Connect

首次建立应用时：

1. Apps → 新建 iOS App：名称 BetaBuddy、简体中文、Bundle ID、唯一的 SKU。
2. Pricing and Availability：App 保持免费，选择实际要发布的地区。1.2.0 起把终生会员非消耗型内购与新版 App 一同提交审核；美国价格为 $10，其他地区以 App Store Connect 的价格矩阵为准。
3. App Privacy：按最终实现填写。应用不向开发者或合作方传输个人数据，对应「不收集数据」；每次提交前对照打包的 SDK、隐私清单和 Apple 的验证报告复核。
4. 年龄分级、内容权利、出口加密问卷。
5. Support URL 填 `https://betabuddy.dozycat.ai`，Privacy Policy URL 填 `https://betabuddy.dozycat.ai/privacy`。站点由 Cloudflare 静态 Worker 托管，源文件在 `services/support-site`；应用内的离线政策不能替代这两个网址。
6. 审核联系人填真实的姓名和电话。

每个版本：

1. 新建版本，填「此版本的新增内容」，按需更新宣传文本、描述和审核备注。文案在该版本的 `release-<版本>.md`。
2. 截图用真实界面，6.9 英寸栏目为 1320 × 2868。不要用非攀岩的模型验证样片宣传攀岩效果。含场馆内容的截图不放进仓库。
3. 如果行为变了，同步线上的隐私政策和支持页。
4. 选中处理完成的构建。建议先经 TestFlight 在真机上验证。
5. 附上审核要求的材料。Apple 曾按 Guideline 2.1 要求提供最新 iOS 真机上从启动开始的全流程录屏，以及用途、操作步骤、外部服务和地区差异的说明；之后每次提审都备好。
6. 提交审核。发布方式是审核通过后自动发布。

## 5. 地区

- 中国大陆：上架需要 App 备案号。备案资料含个人信息，不放在这个仓库里；不要填造假的编号。
- 欧盟：需要先完成交易商身份声明。

## 6. 发布之后

更新 [STATUS.md](STATUS.md)：哪个构建、什么时候、到了哪一步、还差什么。只写实际发生的事。

## 官方依据

- 开发者会员：https://developer.apple.com/programs/
- 免费定价：https://developer.apple.com/help/app-store-connect/manage-app-pricing/set-a-price
- 隐私说明：https://developer.apple.com/help/app-store-connect/manage-app-information/manage-app-privacy/
- 审核要求：https://developer.apple.com/app-store/review/guidelines/
- 截图尺寸：https://developer.apple.com/help/app-store-connect/reference/app-information/screenshot-specifications/
- 地区资料：https://developer.apple.com/help/app-store-connect/reference/app-information/app-information/
