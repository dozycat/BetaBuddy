# BetaBuddy

iPhone 与 Android 应用：本地攀岩视频回看、训练手账、岩馆线路与「换个身材怎么爬」的推演。全部离线。两个平台是同一个产品，各自原生实现。产品说明见 `README.md`，设计见 `docs/ios/`，Android 的落地见 `docs/android/`。

## 技术栈

iOS：

- Swift、SwiftUI，iOS 17+，仅 iPhone、竖屏、浅色。
- 姿态识别：MediaPipe Tasks Vision 的 Pose Landmarker Full（33 点），模型随应用打包，在设备上以 CPU 推理。
- 视频：AVFoundation 顺序解码，Core Image 预处理。
- 存储：Application Support 下的 JSON 与文件，原子写入，排除云备份。
- 岩馆：`WKWebView` 加载随包内置的场馆数据包，经自定义协议 `gympack://local/` 提供文件。
- 工程由 XcodeGen 从 `ios/project.yml` 生成。没有 CocoaPods，没有 Swift 包运行时依赖，没有后台服务。

Android：

- Kotlin、Jetpack Compose，Android 8.0（API 26）+，单 Activity、竖屏、浅色。应用 ID `com.betabuddy.android`。
- 姿态识别：同一个 Pose Landmarker Full 模型，MediaPipe Tasks Vision，`MediaExtractor` + `MediaCodec` 顺序解码，CPU 推理。
- 回看：Media3 ExoPlayer 画到 `TextureView`，骨架用 Compose `Canvas` 叠加。
- 存储：`filesDir/BetaBuddy/` 下的 JSON 与文件，原子写入，排除云备份和设备间迁移。
- 岩馆：`WebView` + `WebViewAssetLoader`，本地地址 `https://appassets.androidplatform.net/gympack/`，只提供清单列出的文件；垫片把页面的 `window.webkit.messageHandlers.gym` 接到 `addJavascriptInterface`。
- 语言：AppCompat 的按应用语言接口；`values/` 是英语。
- Gradle（wrapper）+ AGP 内置 Kotlin，版本全部固定在 `android/gradle/libs.versions.toml`。不申请任何权限。

## 目录

```
ios/BetaBuddy/App/        BetaBuddyApp.swift：入口、两个入口的导航（0 岩馆，1 攀爬册）
ios/BetaBuddy/Core/       Models、PoseMath、VideoAnalyzer、ClimbStore、GymPack、AppLanguage
ios/BetaBuddy/Views/      DesignSystem、BrandMark、Journal、Import、Review、About、Gym、Localization
ios/BetaBuddy/Resources/  图标、pose_landmarker_full.task、Localizable.xcstrings、InfoPlist.xcstrings、PrivacyInfo、PrivacyPolicy.<语言>.txt、Licenses
ios/BetaBuddyTests/       模拟器上的单元与集成测试
ios/BetaBuddyUITests/     界面冒烟测试
ios/CoreTests/            Mac 上可跑的核心测试（Package.swift 只含 Models 与 PoseMath）
ios/LocalMediaPipe/       SDK 指标上报工厂的空实现
ios/scripts/              bootstrap.sh、install-gym-pack.sh、prepare-local-graph.sh
android/app/src/main/java/com/betabuddy/android/
  MainActivity.kt         唯一的 Activity
  ui/theme、ui/common     BB 颜色与字体、公共组件、BBIcons
  ui/nav/AppShell.kt      两个入口（0 岩馆，1 攀爬册）、底部导航与齿轮、设置页的导航
  gym/                    GymManifest、GymPack、GymMessage（纯 Kotlin）、GymSession、GymScreen
  journal/ pose/ review/ data/   攀爬册：列表与导入、VideoAnalyzer 与 PoseMath、回看、ClimbStore
  settings/ i18n/         设置与关于、隐私政策、AppLanguage
android/app/src/main/res/ values*/strings.xml（导航、岩馆、设置）、strings_journal.xml（攀爬册）、raw*/privacy_policy.txt、xml/
android/app/src/test/     JVM 单元测试（纯 Kotlin）
android/app/src/androidTest/  设备测试
android/scripts/          install-gym-pack.sh、check-resources.py
docs/android/             README（构建、版本组合、验证状态）、design
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

Android（需要 JDK 17 和 Android SDK；SDK 许可要由使用者本人接受，不要替人接受）：

```sh
python3 android/scripts/check-resources.py     # 不需要 SDK：资源、五种语言的文字、Manifest
cd android
./gradlew :app:testDebugUnitTest
./gradlew :app:assembleDebug :app:lintDebug
./gradlew :app:assembleRelease                 # R8 + 资源压缩
```

Android 的版本号在 `android/app/build.gradle.kts`（`versionName`、`versionCode`）；依赖版本只在 `android/gradle/libs.versions.toml` 改，改之前对一遍 AGP、Kotlin、Gradle 三者的兼容表。

增删源文件后用 `xcodegen generate --spec ios/project.yml --project ios` 重新生成工程，并把 `project.pbxproj` 一起提交。版本号在 `ios/project.yml`（`MARKETING_VERSION`、`CURRENT_PROJECT_VERSION`）；同一版本重新上传要递增构建号。

## 约定

**场馆数据包不进仓库。** 模型、线路、求解器、网页都在仓库之外生成，用 `ios/scripts/install-gym-pack.sh <目录>` 装到被忽略的 `ios/GymPack/`，Android 用 `android/scripts/install-gym-pack.sh <目录>` 装到被忽略的 `android/app/src/main/assets/gympack/`。仓库里的任何文件，包括代码、测试夹具、文档、截图和提交信息，都不写场馆名、墙名、线路名、岩点坐标或算法细节。测试用合成数据。工程必须在没有数据包时照常生成、编译、通过测试。

**全国岩馆目录例外。** 用户要求为攀爬册离线筛选整理全国岩馆信息：公开来源的馆名、行政区、地址与来源链接可以进入独立的 `gyms-cn.json` 目录，保留来源与覆盖说明，使用可复跑脚本更新。这不授权把上述 3D GymPack、墙名、线路或求解器内容提交进来。目录测试仍使用合成数据；不得把目录快照描述为实时或完整名录。

**不引入联网依赖。** 不加会访问网络的 SDK、遥测、远程配置或远程资源。岩馆页面只能读数据包目录，`http(s)` / `ws(s)` / `ftp` 被内容规则拦截。新增第三方代码要固定版本和校验值，并更新 `ios/THIRD_PARTY.md`。Android 不声明任何权限：新增依赖先看它的 Manifest 会合并进来什么，权限和上报组件在应用的 Manifest 里用 `tools:node="remove"` 去掉，并更新 `android/THIRD_PARTY.md`。

**分析诚实性。** 不展示没验证过的评分或诊断。髋部中点不叫重心，二维角度不叫关节角度，候选片段不下结论，骨架不跨检测空白延续。岩馆内容标明非官方，推演不说成建议。新增文案和指标先对照 `docs/ios/product-design.md` 和 `docs/ios/gym-design.md`。验证记录只写实际做过的事，没做的列在「尚未验证」里。

**多语言。** 开发语言是简体中文，另有繁体中文、英语、日语、韩语。面向用户的文字一律走 `Localizable.xcstrings`：视图里用字符串字面量（键就是简体中文原文），非视图代码用 `String(localized: "…", bundle: AppLanguage.bundle)`，格式化用 `AppLanguage.locale`——语言可以在应用内切换，不能依赖进程启动时的语言；不翻译的品牌标记和数据用 `Text(verbatim:)`。新增或改动文字时，五种语言一起补齐，占位符个数和换行数保持一致，`LocalizationTests` 会检查。用攀岩的人实际说的话，不加原文没有的承诺。不要把界面语言写进存档：存类型或原值，显示时再取文字。 Android 的文字在 `res/values*/strings.xml` 和 `strings_journal.xml`，`values/` 是英语；与 iOS 是同一句话就用同一份译文，五种语言的键、占位符和换行一致，`ShellStringsTest`、`JournalStringsTest` 和 `check-resources.py` 会检查。

**代码风格。** 跟随现有文件：视图用 `DesignSystem.swift`（Android 是 `ui/theme`、`ui/common`）里的 `BB` 颜色与组件，错误信息要告诉用户能做什么。不为假设的需求加抽象。

**不要写进仓库的东西。** 本机绝对路径、新的个人信息、签名产物、`ios/Vendor/`、`ios/build/`、`ios/.derivedData/`、`android/local.properties`、`android/keystore.properties`、keystore 文件、`android/**/build/`。

## 提交前检查

1. `git status` 和 `git diff --cached` 里没有 `ios/GymPack/`、`android/app/src/main/assets/gympack/`、Vendor、构建产物、签名材料、场馆内容或本机路径。
2. `swift test --package-path ios` 通过；动了应用代码就跑模拟器测试和未签名的真机构建。动了 Android 就跑 `check-resources.py`、单元测试和 `assembleDebug`。
3. 改了行为就同步 `docs/ios/`、`docs/android/` 里对应的设计和验证记录；改了隐私相关行为就同步 `PrivacyPolicy.txt`、`docs/ios/app-store/privacy-policy.txt` 和 `PrivacyInfo.xcprivacy`。Android 的内置隐私政策是 `android/app/src/main/res/raw*/privacy_policy.txt`（三种语言一起改）。
4. 除非被明确要求，不 push、不上传构建、不改签名团队。
- 开发者账户信息（团队 ID、App 的 Apple ID、证书、描述文件、备案资料、个人姓名和邮箱）不进仓库；签名用 `DEVELOPMENT_TEAM=<团队 ID>` 在命令行传入。
