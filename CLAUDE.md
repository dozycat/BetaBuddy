# BetaBuddy

iPhone 应用：本地攀岩视频回看、训练手账、岩馆线路与「换个身材怎么爬」的推演。全部离线。产品说明见 `README.md`，设计见 `docs/ios/`。

## 技术栈

- Swift、SwiftUI，iOS 17+，仅 iPhone、竖屏、浅色。
- 姿态识别：MediaPipe Tasks Vision 的 Pose Landmarker Full（33 点），模型随应用打包，在设备上以 CPU 推理。
- 视频：AVFoundation 顺序解码，Core Image 预处理。
- 存储：Application Support 下的 JSON 与文件，原子写入，排除云备份。
- 岩馆：`WKWebView` 加载随包内置的场馆数据包，经自定义协议 `gympack://local/` 提供文件。
- 工程由 XcodeGen 从 `ios/project.yml` 生成。没有 CocoaPods，没有 Swift 包运行时依赖，没有后台服务。

## 目录

```
ios/BetaBuddy/App/        BetaBuddyApp.swift：入口、四个入口的导航
ios/BetaBuddy/Core/       Models、PoseMath、VideoAnalyzer、ClimbStore、GymPack
ios/BetaBuddy/Views/      DesignSystem、BrandMark、Home、Import、Review、Journal、Gym
ios/BetaBuddy/Resources/  图标、pose_landmarker_full.task、PrivacyInfo、PrivacyPolicy、Licenses
ios/BetaBuddyTests/       模拟器上的单元与集成测试
ios/BetaBuddyUITests/     界面冒烟测试
ios/CoreTests/            Mac 上可跑的核心测试（Package.swift 只含 Models 与 PoseMath）
ios/LocalMediaPipe/       SDK 指标上报工厂的空实现
ios/scripts/              bootstrap.sh、install-gym-pack.sh、prepare-local-graph.sh
docs/ios/                 product-design、gym-design、validation、app-store/
legacy/web/               早期网页实现，不再维护，不要改
```

## 构建与测试

```sh
./ios/scripts/bootstrap.sh          # 下载并校验 SDK，生成工程；改了 project.yml 或增删文件后重跑 xcodegen
swift test --package-path ios
xcodebuild -project ios/BetaBuddy.xcodeproj -scheme BetaBuddy \
  -destination 'platform=iOS Simulator,name=<模拟器>' -derivedDataPath ios/.derivedData test
xcodebuild -project ios/BetaBuddy.xcodeproj -scheme BetaBuddy \
  -destination 'generic/platform=iOS' -derivedDataPath ios/.derivedData CODE_SIGNING_ALLOWED=NO build
```

增删源文件后用 `xcodegen generate --spec ios/project.yml --project ios` 重新生成工程，并把 `project.pbxproj` 一起提交。版本号在 `ios/project.yml`（`MARKETING_VERSION`、`CURRENT_PROJECT_VERSION`）；同一版本重新上传要递增构建号。

## 约定

**场馆数据包不进仓库。** 模型、线路、求解器、网页都在仓库之外生成，用 `ios/scripts/install-gym-pack.sh <目录>` 装到被忽略的 `ios/GymPack/`。仓库里的任何文件，包括代码、测试夹具、文档、截图和提交信息，都不写场馆名、墙名、线路名、岩点坐标或算法细节。测试用合成数据。工程必须在没有数据包时照常生成、编译、通过测试。

**不引入联网依赖。** 不加会访问网络的 SDK、遥测、远程配置或远程资源。岩馆页面只能读数据包目录，`http(s)` / `ws(s)` / `ftp` 被内容规则拦截。新增第三方代码要固定版本和校验值，并更新 `ios/THIRD_PARTY.md`。

**分析诚实性。** 不展示没验证过的评分或诊断。髋部中点不叫重心，二维角度不叫关节角度，候选片段不下结论，骨架不跨检测空白延续。岩馆内容标明非官方，推演不说成建议。新增文案和指标先对照 `docs/ios/product-design.md` 和 `docs/ios/gym-design.md`。验证记录只写实际做过的事，没做的列在「尚未验证」里。

**代码风格。** 跟随现有文件：视图用 `DesignSystem.swift` 里的 `BB` 颜色与组件，界面文字为中文，错误信息要告诉用户能做什么。不为假设的需求加抽象。

**不要写进仓库的东西。** 本机绝对路径、新的个人信息、签名产物、`ios/Vendor/`、`ios/build/`、`ios/.derivedData/`。

## 提交前检查

1. `git status` 和 `git diff --cached` 里没有 `ios/GymPack/`、Vendor、构建产物、场馆内容或本机路径。
2. `swift test --package-path ios` 通过；动了应用代码就跑模拟器测试和未签名的真机构建。
3. 改了行为就同步 `docs/ios/` 里对应的设计和验证记录；改了隐私相关行为就同步 `PrivacyPolicy.txt`、`docs/ios/app-store/privacy-policy.txt` 和 `PrivacyInfo.xcprivacy`。
4. 除非被明确要求，不 push、不上传构建、不改签名团队。
