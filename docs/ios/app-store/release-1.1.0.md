# BetaBuddy 1.1.0 (3) 发布资料

本文件是 1.1.0 的文案源和提审清单。后台实际状态见 [STATUS.md](STATUS.md)。场馆数据包不在本仓库里，构建前用 `ios/scripts/install-gym-pack.sh <数据包目录>` 装入（见 [../gym-design.md](../gym-design.md)）。

## 此版本的新增内容（What's New）

新增「岩馆」入口：

• 在手机上看一座岩馆的 3D 线路图，按岩壁、难度档和岩点颜色筛选线路，查看起步点和结束点。
• 换个身材看这条线：选择或调整身高、体重、臂展、力量、爆发和柔韧，看模型推演出的一种走法，可逐步回放。
• 全部内置、离线可用，不联网、不上传。

岩馆内容为非官方整理，位置和线路可能与现场不符；推演结果不是教练建议，请以现场和自身判断为准。

## 宣传文本（更新）

再试一次，这次有点不一样。导入一趟攀爬回看骨架与动作细节；在「岩馆」里看看同一条线换个身材可以怎么爬。全部功能免费，无广告、内购或订阅。

## 描述（在原描述的功能列表后追加）

• 岩馆：内置一座岩馆的非官方 3D 线路图，可按岩壁、难度档和颜色筛选线路；选择不同身材，查看模型推演出的一种走法并逐步回放。位置和线路为爱好者根据公开资料整理，可能与现场不符；推演不是教练建议。

## 审核备注（英文，追加到原备注后）

New in 1.1.0: a fourth tab “岩馆” (Gym). It shows a bundled, offline, unofficial 3D layout of one climbing gym with its boulder problems, and a kinematic what-if simulation of how climbers of different body sizes might climb a selected problem. To test: open the “岩馆” tab, read the one-time notice and tap “知道了，进入岩馆”, pick a wall and a problem from the bottom sheet, then press play to step through the simulated sequence; the “攀爬者” tab in the sheet changes the body parameters.

The content of this tab is rendered in an embedded web view from files bundled inside the app. It makes no network requests: all http/https/ws navigation and subresource loads are blocked, and the pack is served from a custom local scheme. No account, purchase, tracking or data collection is involved. The gym layout is an unofficial, hobbyist reconstruction and is labelled as such in the app; it is not affiliated with or endorsed by the gym. The simulation is a model-based illustration, not coaching or safety advice.

## 提审前清单

- [ ] 用装好数据包的工程在真机上全流程验收（导入 → 分析 → 回看 → 手账 → 练习 → 岩馆：选墙、选线、播放、换身材），并按 Apple 上次 2.1 的要求录屏：最新 iOS 真机、从启动开始、覆盖全部入口。
- [x] 签名归档并上传构建 3（2026-11-12 完成；重新上传要递增 `CURRENT_PROJECT_VERSION`）。流程见 [RELEASE.md](RELEASE.md)。
- [ ] App Store Connect：新建版本 1.1.0，填“此版本的新增内容”、更新宣传文本和描述、追加审核备注、附上录屏和 2.1 要求的说明（用途、操作步骤、无外部服务、无地区差异）。
- [ ] 6.9 英寸截图补一张「岩馆」界面（1320 × 2868）。截图含场馆内容，不放进本仓库。
- [ ] 隐私：仍为“不收集数据”；线上隐私政策页补上「岩馆」一节（文本源 `privacy-policy.txt` 已更新）。
- [ ] 内容权利：确认对内置场馆内容的使用方式（非官方、不含第三方照片）。
- [ ] 选择构建 3，提交审核。
