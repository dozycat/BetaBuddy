# BetaBuddy · iOS 工程

SwiftUI 原生应用，iOS 17+，仅 iPhone、竖屏。产品介绍见[仓库首页](../README.md)；这里讲怎么构建、怎么测，以及姿态识别这条管线是怎么搭的。

## 构建

1. 安装 Xcode（含 iOS SDK）和 [XcodeGen](https://github.com/yonaskolb/XcodeGen)。
2. 在仓库根目录运行 `./ios/scripts/bootstrap.sh`。它下载固定版本的官方 MediaPipe SDK 到被忽略的 `ios/Vendor/`，校验 SHA-256，准备本地 graph 库，然后生成 `BetaBuddy.xcodeproj`。姿态模型已在源码里，同样会被校验。
3. 打开 `ios/BetaBuddy.xcodeproj`，选择 BetaBuddy scheme。
4. 真机：在 Signing & Capabilities 选自己的 Development Team，选中连接的 iPhone，Build & Run。
5. 模拟器：先安装 iOS runtime。适合看界面和验证 CPU 管线，不能代替真机上的速度、温度和耗电。

没有 CocoaPods，没有 Swift 包运行时依赖，没有后台服务。SDK 按官方 podspec 的方式以静态 XCFramework 和 graph archive 链接；其中的指标上报工厂被替换成上游的空实现，见 [LocalMediaPipe/README.md](LocalMediaPipe/README.md)。

增删源文件或改了 `project.yml` 之后，重新生成工程：

```sh
xcodegen generate --spec ios/project.yml --project ios
```

### 岩馆数据包（可选）

「岩馆」入口显示的场馆模型、线路和推演来自一个在仓库之外生成的数据包。它**不在这个仓库里，也不要提交进来**。没有它，工程照常生成、编译和测试，入口显示「这个构建没有内置岩馆数据」。

```sh
./ios/scripts/install-gym-pack.sh /path/to/GymPack   # 核对 manifest 与每个文件的 SHA-256 后拷到 ios/GymPack/
./ios/scripts/install-gym-pack.sh --remove           # 回到不含数据包的构建
```

`ios/GymPack/` 被 Git 忽略。构建时由一个脚本阶段把它拷进应用包，所以工程文件与有没有数据包无关。接口约定见 [../docs/ios/gym-design.md](../docs/ios/gym-design.md)。

## 测试

```sh
# 几何与时序的核心测试，在 Mac 上直接跑
swift test --package-path ios

# 单元、集成与界面测试；把设备名换成已安装的模拟器
xcodebuild -project ios/BetaBuddy.xcodeproj -scheme BetaBuddy \
  -destination 'platform=iOS Simulator,name=<模拟器>' -derivedDataPath ios/.derivedData test

# 未签名的真机构建
xcodebuild -project ios/BetaBuddy.xcodeproj -scheme BetaBuddy \
  -destination 'generic/platform=iOS' -derivedDataPath ios/.derivedData \
  CODE_SIGNING_ALLOWED=NO build
```

第三方 SDK 可能输出反馈张量或投影相关的诊断提示，模拟器也会输出系统框架提示；这些不等于测试失败。做过和没做过的验证记在 [../docs/ios/validation.md](../docs/ios/validation.md)。

## 语言

开发语言是简体中文（`project.yml` 的 `developmentLanguage: zh-Hans`），另有繁体中文、英语、日语、韩语。全部文字在 `BetaBuddy/Resources/Localizable.xcstrings`，相册权限说明在 `InfoPlist.xcstrings`；键是简体中文原文。XcodeGen 没有 `knownRegions` 选项，工程文件里只列出开发语言，其余语言由 String Catalog 在构建时生成对应的 `.lproj`，测试会确认五种语言都进了应用包。

界面语言由 `AppLanguage.current` 决定：应用内的选择（`UserDefaults` 的 `appLanguage`）优先，否则按系统的首选语言匹配，都不匹配用英语。文字不依赖进程启动时的语言：SwiftUI 的文字跟随根视图注入的 `\.locale`，非视图代码一律写 `String(localized: "…", bundle: AppLanguage.bundle)`，格式化用 `AppLanguage.locale`。这样在应用内切换语言不用重启。

用某种语言启动：

```sh
xcrun simctl launch booted com.betabuddy.ios -AppleLanguages "(ja)"
```

系统语言用 `-AppleLanguages` 指定时，之前在应用内选过的语言仍然优先。调试用的启动参数：`-initialTab 0`（岩馆，默认）或 `-initialTab 1`（攀爬册）；Debug 构建里 `-resetLanguage YES` 清掉应用内的语言选择，`-debugImportVideo <路径>` 在导入表单里直接选中一段视频（不经过系统选择器）；Debug 构建另有 `-gymQuery`、`-gymOpenWall`、`-gymOpenRoute`，见 [../docs/ios/gym-design.md](../docs/ios/gym-design.md)。

核对源码里的文字是否都在目录里：`xcodebuild -exportLocalizations -project ios/BetaBuddy.xcodeproj -localizationPath <输出目录> -exportLanguage en` 会按编译器看到的键导出，和 `Localizable.xcstrings` 的键比对即可。注意这条命令会顺手改写仓库里的两个 `.xcstrings` 文件（重排格式，并往 `InfoPlist.xcstrings` 里加未翻译的条目），比对完用 `git checkout` 还原。

## 姿态模型与推理管线

模型是 MediaPipe **Pose Landmarker Full / float16**，33 点，固定为官方版本 `1`，权重未改动。SHA-256：

`5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1`

- 单实例串行 `.video` 模式，使用模型自带的时间追踪。
- `AVAssetReader` 顺序解码，跳帧发生在图像变换和推理之前；不会把整段视频转成图片数组。
- 视频的方向和镜像变换先烘焙进输入，宽高比与播放画面一致。
- 两档采样：日常最高 10fps / 720px 长边，细看最高 15fps / 960px 长边。两档用同一份权重，区别只在采样和预处理。
- 推理在独立 actor 上，界面只收低频进度。每帧一个 autoreleasepool，Core Image 不缓存中间图。
- 用 CPU 推理，作为可复现的基线，避开 GPU 线程亲和与模拟器支持的差异。这不代表它是最快的 delegate；GPU 对照和实际能耗要等真机基准。
- 设备进入 critical 热状态时终止；取消在帧间检查；应用退到后台即停止，不在后台长时间占用计算。
- 置信度不足或出界的关键点不进入角度和骨架；不跨检测空白计算位移或画轨迹。

这些是工程上的管线选择。模型没有针对攀岩重新训练，也没有证据表明准确率或性能因此提升。

## 目录

- `BetaBuddy/App`：应用入口与两个入口（岩馆、攀爬册）的导航。
- `BetaBuddy/Core`：数据结构、几何、MediaPipe 推理、原子写入的本地存储，以及岩馆数据包的定位、清单读取和页面消息解析。
- `BetaBuddy/Views`：设计系统、攀爬册、导入、回看、关于、岩馆页面容器，以及存储值到界面文字的转换。
- `BetaBuddy/Resources`：图标、姿态模型、`Localizable.xcstrings` 与 `InfoPlist.xcstrings`、隐私清单、三种语言的离线隐私政策、第三方许可。
- `CoreTests`：可在 macOS 直接运行的几何与时序边界测试。
- `BetaBuddyTests`：本地存储、模型打包、视频推理、数据包加载、本地化完整性的测试。
- `BetaBuddyUITests`：在简体中文和英语下各走一遍两个入口的冒烟测试。
- `LocalMediaPipe`：SDK 指标上报工厂的本地空实现及其头文件依赖。
- `scripts`：`bootstrap.sh`、`install-gym-pack.sh`、`prepare-local-graph.sh`、`generate-icon.swift`、`ExportOptions.plist`。
- [THIRD_PARTY.md](THIRD_PARTY.md)：第三方 SDK、模型与素材的来源和校验值。

## 网络

应用自己不下载模型，不上传视频，不调用任何云端服务。岩馆页面被禁止访问网络。两种情况由系统发起：用户选了只存在 iCloud 的视频时，照片选择器可能先下载原片；「关于」页里的支持与隐私链接会在浏览器里打开。
