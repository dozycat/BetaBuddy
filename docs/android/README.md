# BetaBuddy · Android 工程

Kotlin + Jetpack Compose 原生应用，Android 8.0（API 26）及以上，竖屏，只有浅色。和 iOS 版是同一个产品，全部离线：

- **岩馆**：场馆数据包里的 3D 页面（线路、推演），装在 `WebView` 里离线加载；没有数据包的构建显示「这个构建没有内置岩馆数据」。
- **攀爬册**：主按钮「拍一趟」用应用内相机录一趟，取景时由数据包的拍摄页面检查机位、认颜色和岩点；也可以从系统照片选择器导入。保存在先；点「分析这一趟」时先做姿态识别（MediaPipe Pose Landmarker Full，设备上、CPU），在本地前台服务里排队做完；回看只有画面、时间轴、0.1 秒步进、慢放和完攀开关，不叠骨架、不写笔记；姿态识别只是「分析这一趟」的第一步。「分析这一趟」由数据包里的分析页面完成，应用内相机拍的记录在姿态识别之后自动排队分析。分析页面随结果给了剪辑计划的记录，回看页多一个「自动剪辑」：按计划在手机上剪出一段（慢镜头与回放、推拉镜头、能消干净时消除路人），循环预览，可以经系统分享面板把这段视频发出去。记录可以标抱石 / 顶绳 / 先锋（分析页面也会判断），难度选择器跟着换体系。文件夹、岩馆 / 难度 / 日期组合筛选、随应用打包的全国岩馆目录。
- **设置与关于**：应用内选语言（「跟随系统」加简体中文、繁体中文、英语、日语、韩语，立即生效），本地与分析边界说明、内置隐私政策、开源许可。

产品介绍见[仓库首页](../../README.md)，Android 上的设计与和 iOS 的差异见 [design.md](design.md)，数据包接口见 [../ios/gym-design.md](../ios/gym-design.md)。

应用 ID `com.betabuddy.android`，版本 `1.2.0`（`versionCode` 2），在 `android/app/build.gradle.kts`。1.2.0 还没有发布过，发布说明的草稿在 [releases/1.2.0.md](releases/1.2.0.md)。

## 权限

`AndroidManifest.xml` 声明的全部权限：

| 权限 | 用途 | 什么时候问 |
| --- | --- | --- |
| `FOREGROUND_SERVICE`、`FOREGROUND_SERVICE_DATA_SYNC`、`FOREGROUND_SERVICE_MEDIA_PROCESSING` | 姿态识别的本地前台服务（Android 15+ 用 `mediaProcessing`，更早的系统用 `dataSync`） | 不需要询问 |
| `POST_NOTIFICATIONS` | 通知栏里显示分析进度，可选 | Android 13+，分析这一趟时还没允许；拒绝不影响分析 |
| `CAMERA` | 应用内相机，只录画面 | 第一次打开拍摄界面时 |
| `ACCESS_COARSE_LOCATION`、`ACCESS_FINE_LOCATION` | 在拍摄界面按距离给内置岩馆目录排序，取一次、不保存 | 点「按距离找附近的岩馆」时一起问，Android 12+ 由用户选精确或大致 |

Manifest 用 `tools:node="remove"` 去掉依赖库会合并进来的 `INTERNET`、`ACCESS_NETWORK_STATE`、`WAKE_LOCK`，并防止 `RECORD_AUDIO` 被带进来。合并之后另有 AndroidX Core 自动加的签名级权限 `com.betabuddy.android.DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION`（只有应用自己能持有）。不申请麦克风或网络权限。相册自动收录默认关闭，启用才请求 `READ_MEDIA_VIDEO`（Android 13+）、`READ_MEDIA_VISUAL_USER_SELECTED`（Android 14+ 可只授权部分视频）、`ACCESS_MEDIA_LOCATION`（视频已有的位置）；Android 12 及以前用 `READ_EXTERNAL_STORAGE`，声明最高 SDK 32。`python3 android/scripts/check-resources.py` 按这份名单检查 Manifest。依赖带进来的组件和移除的理由见 [android/THIRD_PARTY.md](../../android/THIRD_PARTY.md)。

## 构建

需要 JDK 17 和 Android SDK（平台 37、Build-Tools 36.0.0）。Gradle 用仓库里的 wrapper，不用另装。

```sh
brew install openjdk@17
export JAVA_HOME="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"

# Android SDK：用 Android Studio 的 SDK Manager 装，或者用命令行工具装。
# SDK 的许可协议要由你本人阅读并接受，构建不会也不应该替你接受。
sdkmanager "platform-tools" "platforms;android-37.0" "build-tools;36.0.0"
export ANDROID_HOME=<SDK 目录>          # 或者在 android/local.properties 里写 sdk.dir=<SDK 目录>

cd android
./gradlew :app:assembleDebug            # 产物在 app/build/outputs/apk/debug/
./gradlew :app:testDebugUnitTest        # JVM 上的单元测试
./gradlew :app:lintDebug
./gradlew :app:assembleRelease          # R8 + 资源压缩
./gradlew :app:connectedDebugAndroidTest   # 需要连着设备或开着模拟器
```

`android/local.properties` 被 Git 忽略，不要提交。SDK 管理器里平台包的名字以它列出的为准（`sdkmanager --list | grep platforms`）。

不需要 Android SDK 就能跑的检查：

```sh
python3 android/scripts/check-resources.py   # 资源 XML、五种语言的键与占位符、代码引用、Manifest 与备份规则
```

### 版本组合

所有版本都固定在 `android/gradle/libs.versions.toml`。

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| Gradle（wrapper） | 9.7.1 | 分发包的 SHA-256 写在 `gradle-wrapper.properties` 里，下载后校验 |
| Android Gradle Plugin | 9.3.3 | 用它内置的 Kotlin 支持，不另外应用 `kotlin-android` 插件 |
| Kotlin（Compose 与序列化编译器插件） | 2.4.21 | |
| `compileSdk` / `targetSdk` / `minSdk` | 37 / 37 / 26 | |
| Compose BOM | 2026.09.00 | |
| core-ktx / activity-compose / appcompat | 1.19.1 / 1.13.0 / 1.8.0 | |
| lifecycle / navigation-compose / webkit | 2.11.0 / 2.10.2 / 1.18.0 | |
| Media3 ExoPlayer / Transformer / Effect | 1.11.1 | 回看播放；自动剪辑的重新编码与 OpenGL 效果，三个同一版本 |
| CameraX（camera-core / camera2 / lifecycle / video / view） | 1.7.0 | 应用内相机 |
| MediaPipe Tasks Vision | 0.10.35 | 0.10 线的最后一个版本；姿态识别，以及自动剪辑里找人（`ObjectDetector`） |
| kotlinx-serialization-json / kotlinx-coroutines | 1.13.0 / 1.11.0 | |

选这一组的依据：

- 每个版本都在 Google Maven 或 Maven Central 的发布索引里。
- 这些 AndroidX 库的 AAR 元数据要求 `compileSdk` 至少 37、AGP 至少 9.1.0；它们的 `minSdk` 最高是 24，低于应用的 26。
- AGP 9.3 支持到 API 37，要求 Gradle 9.5 以上、JDK 17。
- Kotlin 2.4.21 的兼容表列出的范围是 Gradle 到 9.7、AGP 到 9.3，所以 AGP 取 9.3 线的最新补丁、Gradle 取 9.7 线的最新补丁，而不是各自的最新版。升级其中一个之前先对一遍另外两个的兼容表。

### 签名

仓库里没有、也不应该有任何密钥、keystore 或密码。Release 构建的签名从这两处读，哪个都没有时用 debug 密钥签名（可以安装测试，不能发布）：

- `android/keystore.properties`（被 Git 忽略）：`storeFile`、`storePassword`、`keyAlias`、`keyPassword`。`storeFile` 可以是绝对路径，或相对 `android/` 的路径。
- 环境变量：`BETABUDDY_KEYSTORE_FILE`、`BETABUDDY_KEYSTORE_PASSWORD`、`BETABUDDY_KEY_ALIAS`、`BETABUDDY_KEY_PASSWORD`。

`*.jks`、`*.keystore` 被 Git 忽略。

### 岩馆数据包（可选）

岩馆页面、拍摄页面和分析页面都来自一个在仓库之外生成的数据包。它**不在这个仓库里，也不要提交进来**。没有它，工程照常编译和测试：岩馆入口显示「这个构建没有内置岩馆数据」，拍摄界面照常录、照常存（没有机位检查），回看页没有「分析这一趟」。

```sh
./android/scripts/install-gym-pack.sh <数据包目录>   # 核对 manifest 与每个文件的 SHA-256 后拷到 app/src/main/assets/gympack/
./android/scripts/install-gym-pack.sh --remove      # 回到不含数据包的构建
```

脚本的校验和 iOS 的 `ios/scripts/install-gym-pack.sh` 相同：清单读得了、格式版本认识、每个文件的 SHA-256 相符、目录里没有清单之外的文件；只拷贝清单列出的文件。另外多一条 Android 特有的：打包工具不会把点开头的文件和下划线开头的目录放进 APK，用了这类名字的数据包在这里就被拒绝。`app/src/main/assets/gympack/` 被 Git 忽略；装完之后重新构建即可。

## 目录

```
android/
  settings.gradle.kts, build.gradle.kts, gradle.properties
  gradle/libs.versions.toml            依赖和版本都在这里
  gradle/wrapper/, gradlew, gradlew.bat
  scripts/install-gym-pack.sh          校验并安装 / 移除场馆数据包
  scripts/check-resources.py           不依赖 SDK 的资源与 Manifest 检查
  scripts/edit-backtest.sh             拿自己的一段片子和剪辑计划在设备上跑一遍自动剪辑，取回成片、报告和对照图
  THIRD_PARTY.md
  app/
    build.gradle.kts, proguard-rules.pro
    src/main/AndroidManifest.xml
    src/main/java/com/betabuddy/android/
      MainActivity.kt                  唯一的 Activity；边到边、调试启动参数
      ui/theme/                        BB 颜色与字体、Material 3 底座
      ui/common/                       PaperPage、Stamp、BigButton、SectionLabel、ColorCard、BBIcons…
      ui/nav/                          AppShell：两个入口、底部导航与齿轮、设置页的导航
      gym/                             数据包、WebView 容器（三个数据包页面共用）、消息桥、岩馆界面
      settings/                        设置与关于、隐私政策、开源许可
      i18n/                            AppLanguage：当前界面语言与应用内语言选择
      capture/                         应用内相机：拍摄页面的消息、画面来源（CameraX / 文件）、取帧、录像、拍摄界面、岩馆建议、取一次位置
      journal/                         攀爬册：列表、导入、文件夹与筛选、文字、岩馆目录与距离、姿态识别服务、相机开关
      pose/                            解码、姿态识别、几何
      review/                          回看：播放；分析页面的宿主；拍过的记录的分析队列
      edit/                            自动剪辑：读剪辑计划、时间与镜头的换算、谁能被消掉、找人、渲染、剪辑页面、缓存文件与分享
      data/                            记录的数据结构与本地存储
    src/main/res/
      values*/strings.xml              导航、岩馆、设置的文字
      values*/strings_album.xml        文件夹、筛选、岩馆与日期的文字（名字以 album_ 开头）
      values*/strings_journal.xml      攀爬册的文字（名字以 journal_ 开头）
      values*/strings_capture.xml      应用内相机的文字（名字以 capture_ 开头）
      raw*/privacy_policy.txt          内置隐私政策（简中、繁中、英）
      xml/locales_config.xml           应用支持的语言
      xml/data_extraction_rules.xml    不进云备份，也不随换机迁移
      xml/edit_file_paths.xml          分享剪出来的视频时 FileProvider 能给出的唯一目录（缓存里的 edits/）
    src/main/assets/
      pose_landmarker_full.task        姿态模型
      efficientdet_lite2.tflite        找人的检测模型（自动剪辑消除路人时用）
      gyms-cn.json                     全国岩馆目录（来源与覆盖见 docs/gym-directory.md）
      licenses/                        第三方许可全文（设置里可以查看）
      gympack/                         场馆数据包（被忽略，由脚本装入）
    src/test/                          JVM 单元测试（纯 Kotlin，不依赖 Android 类）
    src/androidTest/                   设备测试：解码与推理；拍摄界面（替身页面、画出来的墙）；自动剪辑（合成的片子与计划、按颜色找方块的替身、真模型一例）
```

## 与 iOS 的对应

| iOS | Android |
| --- | --- |
| `App/BetaBuddyApp.swift`（`RootView`） | `MainActivity.kt`、`ui/nav/AppShell.kt` |
| `Views/DesignSystem.swift`、`BrandMark.swift` | `ui/theme/Theme.kt`、`ui/common/Components.kt`、`ui/common/Icons.kt` |
| `Core/GymPack.swift` | `gym/GymManifest.kt`、`gym/GymPack.kt`、`gym/GymMessage.kt` |
| `Views/GymView.swift`（`GymSession`、`GymView`） | `gym/PackWebView.kt`、`gym/GymSession.kt`、`gym/GymScreen.kt` |
| `Core/AppLanguage.swift` | `i18n/LanguageTags.kt`、`i18n/AppLanguage.kt` |
| `Views/AboutView.swift` | `settings/SettingsScreens.kt` |
| `Core/Models.swift`、`ClimbStore.swift` | `data/Models.kt`、`data/ClimbStore.kt`、`data/CaptureSweep.kt` |
| `Core/VideoAnalyzer.swift` | `pose/VideoAnalyzer.kt`、`pose/AnalysisTypes.kt` |
| `Views/JournalView.swift`、`ImportView.swift` | `journal/JournalScreen.kt`、`journal/ImportScreen.kt`、`journal/AlbumControls.kt`、`journal/AlbumFilters.kt` |
| `Views/ReviewView.swift` | `review/ReviewScreen.kt`、`review/Playback.kt` |
| `Views/AnalysisView.swift` | `review/AnalysisScreen.kt`、`review/AnalysisSession.kt`、`review/AnalysisBridge.kt`、`review/AnalysisFrames.kt` |
| `Core/Capture.swift`、`Views/CaptureText.swift` | `capture/Capture.kt`、`capture/CaptureText.kt` |
| `Core/CaptureSource.swift` | `capture/CaptureSource.kt`、`capture/FileCaptureSource.kt`、`capture/CapturePipeline.kt`、`capture/FrameRecorder.kt`、`capture/RecorderSwitch.kt` |
| `Core/CaptureSession.swift`、`Views/CaptureView.swift` | `capture/CaptureSession.kt`、`capture/CaptureScreen.kt`、`capture/CaptureGymSheet.kt` |
| `Core/GymLocator.swift` | `journal/GymDirectory.kt`（目录与距离）、`capture/GymLocator.kt`（取一次位置） |
| `Core/AnalysisQueue.swift` | `journal/CameraGate.kt`、`review/AnalysisQueue.kt` |
| 自动剪辑与攀爬方式（iOS 的文件见 `docs/ios/`） | `edit/EditPlan.kt`、`EditTimeline.kt`、`Passers.kt`、`PeopleFinder.kt`、`EditRenderer.kt`、`EditEffect.kt`、`EditScreen.kt`、`EditFiles.kt`；`data/Models.kt` 里的 `Discipline`、`GradeScale` |
| `TestSupport/`（替身页面、合成的墙） | `src/androidTest/assets/*-standin.html`、`src/androidTest/.../capture/StandIns.kt` |
| `Resources/Localizable.xcstrings` | `res/values*/strings*.xml` |
| `Resources/PrivacyPolicy.<语言>.txt` | `res/raw*/privacy_policy.txt` |
| `ios/GymPack/`（被忽略） | `app/src/main/assets/gympack/`（被忽略） |
| `ios/scripts/install-gym-pack.sh` | `android/scripts/install-gym-pack.sh` |
| `BetaBuddyTests/GymPackTests.swift`、`LocalizationTests.swift` | `src/test/.../gym/GymPackTest.kt`、`StringResources.kt` 与各文件的 `*StringsTest.kt` |

## 约定

- **界面用同一套组件。** 颜色和字体取 `ui/theme` 的 `BB`（`Paper/Ink/Lime/Lavender/Muted/Line`、`display()`、`mono()`、`Title…Caption`），页面和控件取 `ui/common`（`PaperPage`、`Stamp`、`BigButton`、`SectionLabel`、`Stat`、`CircleIconButton`、`ColorCard`、`Hairline`、`BBIcons`）。岩馆、攀爬册、设置都依赖这些名字，改名要一起改。
- **攀爬册只有一个入口** `JournalScreen(onOpenSettings)`。它内部的页面用 `rememberSaveable` 的路由加 `BackHandler` 切换，不往 NavHost 里加路由。拍摄界面和分析页面是全屏对话框。姿态识别由服务持有，切到岩馆、离开回看页或应用退到后台均不取消；系统结束服务时保留请求，下一次打开应用重试。
- **存储**都在应用私有目录：攀爬册在 `filesDir/BetaBuddy/`（`archive.json`、原片、缩略图、分析页面交回的 `<记录 id>.analysis.json`、拍摄页面交回的 `<记录 id>.capture.json`、`staging/`——导入和录像中的临时文件，下次启动时清理），外加 `SharedPreferences` 里分析页面上一次用的身材参数和相机的两项设备记录；岩馆只有 `SharedPreferences` 里的两个布尔值。不写外部存储。
- **文字**分四个文件：导航、岩馆、设置在 `strings.xml`；文件夹与筛选在 `strings_album.xml`（`album_`）；攀爬册在 `strings_journal.xml`（`journal_`）；应用内相机在 `strings_capture.xml`（`capture_`）。文件之间不重名。
- **不加联网依赖。** 新增依赖先看它的 Manifest 会合并进来什么权限和组件，该移除的在应用的 Manifest 里移除，并在 `THIRD_PARTY.md` 补一行；依赖有增减时同步 `assets/licenses/components.txt`。权限名单变了要同步 `check-resources.py`、上面的权限表和隐私政策。
- **单元测试是纯 JVM 测试**，读 `src/main/res` 和 `src/main/assets` 用相对模块目录的路径（Gradle 运行测试时的工作目录）。测试夹具用合成数据。

## 语言

五种语言：`values/`（英语，也是其他系统语言的回落）、`values-b+zh+Hans`、`values-b+zh+Hant`、`values-ja`、`values-ko`。新增或改动文字时五种一起改，占位符和换行保持一致；和 iOS 同一句话就用 `Localizable.xcstrings` 里的译文。应用内的语言设置是「设置与关于」的第一项，选了立即生效并记住。

用某种语言启动（不改系统语言）：

```sh
adb shell cmd locale set-app-locales com.betabuddy.android --locales ja     # Android 13+
adb shell cmd locale set-app-locales com.betabuddy.android --locales ""     # 回到跟随系统
```

## 调试与截图

Debug 构建接受这些启动参数（Release 构建忽略）：

```sh
A=com.betabuddy.android/.MainActivity
adb shell am start -n $A --ei initialTab 1                       # 落在攀爬册
adb shell am start -n $A --ez gymNoticeSeen true                 # 跳过岩馆的进入前说明
adb shell am start -n $A --ez gymNoticeSeen true --es gymQuery "<查询串>"
adb shell am start -n $A --ez gymNoticeSeen true --es gymOpenRoute <线路 id>
adb shell am start -n $A --ez gymNoticeSeen true --es gymOpenWall <岩壁 id>
adb exec-out screencap -p > shot.png

# 应用内相机，在没有像样相机的地方（模拟器）：
adb shell am start -n $A --ei initialTab 1 --ez openCapture true          # 直接打开拍摄界面
adb shell "am start -n $A --ei initialTab 1 --ez openCapture true \
    --es debugCaptureVideo <文件> --es debugCaptureFov 50 \
    --es debugCaptureAttitude '6,0.5,0.01' --es debugLocation '<纬度>,<经度>,<误差米>'"
```

`debugCaptureVideo` 用一个视频文件循环播放代替相机（绝对路径，或者应用 `files/` 目录下的文件名；用 `adb push` 加 `run-as <应用 ID> cp` 放进去）；`debugCaptureFov` 是替它报的水平视场角；`debugCaptureAttitude` 是没有运动传感器时替它报的仰角、侧倾和晃动；`debugLocation` 给一个位置，不出现系统的询问。另有 `--ez debugCaptureOverlay true`（在画面上画出交给页面的数和页面回话的快慢）和 `--es debugCaptureKeepFrame <文件名>`（把最近交给页面的一帧另存到应用的 `files/` 目录）。

参数只在全新启动时读取，换参数前先 `adb shell am force-stop com.betabuddy.android`。`gymQuery` 原样接在入口 URL 的 `hint`、`lang` 之后，页面自己定义其中的参数。数据包页面可以用 Chrome 的 `chrome://inspect` 调试（只有 Debug 包打开了 WebView 调试）。日志标签：岩馆 `gym`、拍摄界面 `capture`（离开时有一行各环节的平均耗时，每段片子停下时记下是哪个录像器写的）、分析页面 `analysis`；只写本机 logcat。

## 相册、选人与剪辑界面

独立开发分支的入口统一为「扫相册」「选人」「剪辑」。按用户要求去掉常驻说明，选择状态用圈线和按钮高亮表达，剪辑只显示绳线和真实百分比；保留必要错误和无障碍标签。字号增大时视频缩小、控件换行，时间轴及操作区仍可滚动。详细开发验证见 `../ios/camera-library-selection.md`。

## 验证状态

只记实际做过的；没做的列在「尚未验证」里。

### 当前代码（数据包 1.10.0，加上自动剪辑与攀爬方式）

在一台 Apple 芯片的 Mac 上，JDK 17，Android SDK 平台 37、Build-Tools 36.0.0，工作区里装着数据包 1.10.0：

| 命令 | 结果 |
| --- | --- |
| `python3 android/scripts/check-resources.py` | 通过：271 条文字 × 5 种语言 |
| `./gradlew :app:testDebugUnitTest` | 188 个测试全部通过 |
| `./gradlew :app:assembleDebug :app:lintDebug` | 通过；lint 0 error（warning 是固定版本的更新提示、复数资源建议、KTX 写法之类） |
| `./gradlew :app:compileDebugAndroidTestKotlin` | 设备测试能编译 |
| `./gradlew :app:assembleRelease` | 通过（R8 与资源压缩）；`aapt2 dump permissions` 列出的权限与上面的权限表一致，新增的组件只有不导出的 `FileProvider` |
| `./gradlew :app:connectedDebugAndroidTest`（Android 15 模拟器，`-camera-back virtualscene -qemu -append "arm64.nosme"`） | 37 个：最近一遍全套 33 个通过、3 个跳过（2 个要真从系统取位置，1 个要给片子才跑的 `EditBacktest`）、1 个没过——`CaptureScreenTest` 里量应用自己的编码器帧率的那一条量到每秒 7.3 帧，当时这台 Mac 的负载很高，单独重跑通过。更早的几遍里「空间不足 / 太短」那一条有两遍等界面超时，单独跑和整类跑都通过，原因没有查清。测试里切应用语言的命令原来不等它执行完就启动界面，机器慢时界面会在测试中途被重建（落在岩馆、英语）；现在等命令结束、设置生效再启动 |

设备测试是 `CaptureScreenTest` 13 个、`VideoAnalyzerInstrumentedTest` 9 个、`EditRendererTest` 11 个、`EditScreenTest` 2 个，另有一个平时跳过的 `EditBacktest`（见下面「拿真实片子跑过的」），这一轮只在 Android 15 模拟器上跑过；没有真机。自动剪辑的设备测试做了这些（片子和计划都是合成的：灰墙、顶上一条蓝杠、一个往上走的绿方块当攀爬者、一个横着走的洋红方块当路人；找人用按颜色找方块的替身）：

- 输出在每个抽查的时刻显示的正是计划要的源时刻（原速、0.5 倍、2 倍和一段往回的回放），方向是正的；带旋转信息、横着存的片子也一样。
- 镜头窗口的位置和放大倍数与计划一致；「推拉镜头」关掉回到整幅画面。
- 路人在所有抽查的帧里都不见了，攀爬者在；「消除路人」关掉时不找人、路人在。
- 攀爬者在一处停了大半段片子、之后路人走过那里：换上去的是墙，不是攀爬者的残影（把「每张画面不计自己时刻的 `keep` 框」关掉重跑，这一条会失败）。
- 路人的路线穿过攀爬者：路人在所有抽查的帧里都完整地在，一个也没被消。
- 路人从一个留下来的人身上走过（一个更小、来回走动、只在中间一段被替身「认出」的黄方块）：路人和这个人在所有抽查的帧里都完整地在，一个也没被消（把「不盖到留下来的人」这条检查关掉重跑，会失败）。
- 原速的片段接了三段，输出的帧数和源里这一段的帧数一样（±3）；慢放一帧不少；2 倍速的片段大约是一半。
- 取消后同一个渲染器还能再用；不是视频的文件被拒绝；换一个开关不重新找人。
- 随应用打包的检测模型在一张画出来的画面上（MediaPipe 的示例照片缩小后贴在灰墙前）找到了人，空墙上没找到。只说明模型能加载、能跑，不说明找路人的效果。
- 剪辑页面：三个开关都出现、渲染完开始播放（从 `TextureView` 取图核对不是黑的）、换开关后重新渲染且只留新文件、`FileProvider` 给出的 `content://` 地址读出来就是这个文件、点「完成」后缓存目录清空。回看页：存着没有计划的结果时没有「自动剪辑」按钮，存上计划后出现，点开能渲染（这一条要构建里有带分析页面的数据包，没有时跳过）。
- 拍摄流程的测试里，替身分析页面随结果交回一份编出来的计划和「顶绳」的判断：计划原样存下、能读；这条记录用户标过 V 级难度，所以这个判断没有存，攀爬方式和难度都没变（自动的判断从不动用户标的难度）。

Release 包这一轮只编译、看了权限和组件，没有在设备上跑；更早的代码上跑过的见下一节。

#### 拿真实片子跑过的

`android/scripts/edit-backtest.sh <视频> <plan.json> <输出目录>` 把一段片子和一份剪辑计划放进应用自己的缓存（经 Debug 包的 `run-as`，不涉及存储权限），用随应用打包的检测模型跑可选的设备测试 `EditBacktest`（不给片子时跳过），取回三段成片（开关全开、全关、只开消除路人）、一份报告（找到并跟住了谁、每个人消掉还是留下以及为什么、底图有多少像素是有的、每一步用了多久）和一张对照图（原片与「只开消除路人」在这一趟里 8 个时刻的并排，需要 ffmpeg）。片子和产物都放在仓库之外。计划文件写着没有计划（`{"edit": null}`）时脚本直接说明并停下。

到现在只跑过一段真实的片子，在 Android 15 模拟器上（4 核，软件渲染的 OpenGL）：

- 片子：1080×1920、30 fps、16.5 秒，一个人爬抱石，画面里没有路人；片子本身是剪过的，开头和结尾各有一个别的镜头。
- 数据包 1.10.0 对这段片子不给剪辑计划（因为片子里有切换镜头），所以应用里这条记录不会有「自动剪辑」按钮。为了让渲染和找人仍然在真实画面上跑一遍，用的是一份手工写的计划：`keep` 框取自这段片子的姿态关键点的包围框，`trim`、一段 0.5 倍慢放、一个 1.4 倍的镜头窗口和 20 个底图时刻都是随手定的。**这不是数据包的计划**，只用来走通原生这一侧。
- 找人：15.1 秒里跟出 12 条轨迹。一条是攀爬者（1.97–14.63 秒，92 次），其余 11 条是攀爬者身上或身边的零碎框（多数只出现一两次），以及结尾那个别的镜头里的东西；12 条全都碰到 `keep` 框，所以没有候选、没有做底图、没有人被消掉。
- 结果：「只开消除路人」与「全关」两段成片逐字节相同；对照图上 8 个时刻原片和成片看不出差别，方向和时间对得上。没有任何东西被错消——也因为这段片子里本来就没有能消的人，**消除路人本身在真实画面上还没有被检验过**。
- 耗时（模拟器）：找人 11.2 秒；每次导出 13–14 秒；第一次全开的渲染合计 25.3 秒，之后换开关每次 13 秒左右。成片 1920×1080（带旋转标记，播放时是竖的）、453 帧、约 16 MB。

有路人的片子还没有找到，所以**「消除路人」这次不开放**：`EditOptions.REMOVAL_OFFERED` 关着，剪辑页不显示这个开关，剪辑时不加载检测模型；代码、测试和模型文件都留着。

2026-11-26 关掉之后的复跑：`check-resources.py` 通过；单元测试 183 个通过；`assembleDebug`、`lintDebug`、`compileDebugAndroidTestKotlin` 通过；剪辑这一组设备测试（`EditRendererTest`、`EditScreenTest`、平时跳过的 `EditBacktest`）在 Android 15 模拟器上 14 个里 13 个通过、1 个跳过，剪辑页上确认没有「消除路人」开关。完整设备测试这一轮没有重跑。模拟器要按下面「模拟器」一节带 `arm64.nosme` 启动，不带的话 MediaPipe 的任何模型（包括原有的姿态识别）都会以非法指令崩溃。

顶绳和先锋：数据包对 8 段下载来的真实绳攀片子的判断见 `docs/ios/validation.md` 同一天的记录（顶绳 6 段认出、都标不确定，先锋 2 段没认出）；这些片子没有在 Android 应用里走过，Android 这边只有 `analysisError` 带 `discipline` 的消息解析单元测试。

### 模拟器上做过的

两台模拟器（Pixel 外形 1080×2400）：Android 15（API 35，系统 WebView 124，启动时带 `arm64.nosme`）和 Android 12（API 31，系统 WebView 91）。**没有任何真机。**

- **构建与权限。** `assembleRelease`（R8 与资源压缩）通过；`aapt2 dump permissions` 在 debug 与 release 包上列出的正是上面的权限表加 AndroidX Core 的签名级权限，没有 `INTERNET`、`ACCESS_NETWORK_STATE`、`WAKE_LOCK`、`RECORD_AUDIO`。合并后的 Manifest 里没有 datatransport 的组件、没有 EmojiCompat 的启动项；CameraX 带来的只有不导出、默认停用的 `MetadataHolderService`。
- **设备测试。** API 35 和 API 31 上 22 个设备测试里 20 个运行并通过；另外两个要真从系统取位置，平时跳过，在 API 35 上各自单独跑过（先 `adb emu geo fix`，再 `pm grant` 对应的位置权限，带 `-e geoFix 1`），都通过。拍摄界面的测试用替身数据包（只会照脚本回话的拍摄页面和分析页面，与 iOS 测试是同一份）和一段画出来的墙，从攀爬册的按钮走到保存、姿态识别、后台分析和「看分析」，也覆盖了被拦的机位、画面来源中断、没有拍摄页面、空间不足、太短、保存失败、换岩馆、位置的六种状态、录像上限。
- **岩馆。** 有、无数据包两种构建；说明页只出现一次，3D 场景加载并发出 `ready`；`hintSeen`、`openSettings`、换语言后带新 `lang` 重载并回到原线路、切标签不重载、调试参数。网络审计（经 WebView 调试端口抓整次重新加载）：请求全部是数据包里的文件和 `blob:`；页面主动发起的 `fetch`、远程图片、WebSocket 全部被拦，数据包目录之外的本地路径和清单里没有的文件得到 403。
- **语言。** 五种语言各看了岩馆、攀爬册、设置；应用内切换立即生效；「跟随系统」在系统语言不受支持时落到英语；API 31 上应用内选的语言在创建 `WebView` 之后仍然有效。
- **攀爬册。** 照片选择器不需要权限；导入、姿态识别、回看（骨架与人重合、带旋转信息的视频摆正）、笔记、分享文字、删除；没有人的视频给出「没有清楚地找到攀爬者」；保存后退到桌面，后台服务把姿态识别做完；文件夹、组合筛选和岩馆目录的选择用合成记录走过。Release 包上导入、识别、回看能跑完（R8 没有裁坏 MediaPipe）。
- **分析这一趟**（当时的数据包 1.4.x–1.5.x 的分析页面，真实攀爬视频放在仓库之外）：先做姿态识别再自动打开分析页面、「正在准备画面…」、页面显示结果、「看视频里这一刻」回到回看并暂停、「看分析」显示保存的结果而不重算、在页面里改身高后覆盖保存、「重新分析」、杀进程后结果仍在、删除记录时结果文件一起删；竖屏和横屏视频；页面说「分析不了」时不保存结果。
- **拍一趟**（真实数据包 1.9.1–1.9.3）：拍摄页面在看不见的 `WebView` 里就绪并逐帧回话；在岩馆表里选中数据包里的一家；选颜色、点岩点、被拦时确认后照录；保存后姿态识别、后台分析页面交回结果，机位固定的文件来源下结果里写明用上了拍摄计划（`capture.planUse.used` 为真），手持晃动的片子写明没用上；录过 211.9 秒的一段。用模拟器相机的虚拟场景走过 Debug 和 Release 包，交给页面的那一帧另存出来是直立的、和预览同一个范围。只说明接口在 Android 的 `WebView` 里通了，不说明识别效果。
- **录像器换路。** API 31 模拟器的相机给帧打的时间从零数起，CameraX 的 `Recorder` 把它当开机时间，一帧都不写；应用 3 秒后换成自己的编码器，片子是直立的 H.264、只有画面轨（`ffprobe` 看过）。API 35 上 `Recorder` 正常写。

### 尚未验证

- **自动剪辑在真实的片子上**：除了合成的片子，只在模拟器上跑过一段没有路人的真实片子，而且用的是手工写的计划（见上面「拿真实片子跑过的」）。数据包 1.10.0 给出的真实计划没有在 Android 上渲染过；真实场景里找路人的效果（远处的小人、背对镜头、被遮挡）、底图在真实灯光下对不对得上、放大圈和羽化合不合适，都没有看过——那一段片子里没有人被消掉，所以这些一样也没碰到。那一次检测模型在攀爬者身上和身边多报了十来个零碎的框；它们都碰到 `keep` 框所以没有后果，但同样的零碎框落在别处时会被当成「留下来的人」，挡住从旁边走过的路人不被消掉，这一点有多常见还不知道。检测模型的选型只在模拟器上拿一张示例图比过。
- **自动剪辑的耗时、内存和发热**：模拟器上 3 秒 360×640 的片子几秒钟；真机上 1080p、几十秒的片子要多久（找人大约每秒片子 8 次推理，加一遍导出）、会不会被系统因内存回收、硬件编码器对输出尺寸和帧率的限制，都不知道。HDR 片子、60 fps 片子、带编辑列表（第一帧时间不是 0）的片子没有试过；Android 10 以下 HDR 转 SDR 走不了 OpenGL 的那条路没有试过。
- **自动剪辑在 Release 包上**：只编译过。R8 之后 Transformer、着色器和 `ObjectDetector` 的路径没有在设备上跑过。
- **分享视频**：只核对了 `FileProvider` 的地址能读出文件；没有真的经系统分享面板发给另一个应用。分享之后马上点「完成」（文件被删）时对方还读不读得到，没有试过。
- **剪辑页面、导入页和「整理这一趟」里的三段选择**：只有英语的截图看过剪辑页面和回看页的按钮；攀爬方式的三段选择和按体系换的难度选择器没有截图看过，繁体中文、日语、韩语下三个开关的文字会不会折行、最大字号、TalkBack 都没有看过。
- **数据包 1.10.0 的分析页面在 Android 上**：这一轮没有打开过；真实结果里的 `edit` / `discipline` 只由替身页面走通了接口。
- **去掉骨架和笔记之后的回看页、导入页**：只跑了单元测试、`assembleDebug`、`lintDebug` 和设备测试的编译；没有在模拟器上打开看过排版（尤其是窄屏上时间、三个按钮和速度在一行里放不放得下）。
- **任何真机。** 相机预览与录像在各厂商相机上的表现、视场角和姿态传感器的数值、发热与过热中止、内存、硬件解码器、HDR 视频、长片子的取帧和存档写入耗时；应用自己的编码器在真机上会不会被用到、用到时的帧率。
- **数据包 1.9.6–1.9.8 的分析页面在 Android 上**（一步一步的几屏、剪纸小人、这一趟的建议）：没有在 Android 模拟器或真机上打开过；Android 这边只确认了这些文件装进了 APK。
- 这一轮整理之后的 Release 包：只编译过，没有重跑；设备测试只在 Android 15 模拟器上重跑过，Android 12 模拟器上没有。
- 岩馆页面改用共用的 `PackWebView` 之后：只重看了加载、`ready` 和 `hintSeen`；网络审计、`openSettings`、换语言后回到原线路没有重做。
- 系统的位置询问对话框（授权都是从外面 `pm grant` 给的）；在室内、没有网络定位的机型上取不取得到位置、要多久。
- 用户打开「分析这一趟」的同时后台队列也在分析同一条记录：只有规则的单元测试。下次启动时把没存进去的片子补进攀爬册：只有合成文件的单元测试。
- 繁体中文、日语、韩语下拍摄界面和分析入口的排版；最大字号、TalkBack。
- Android 8–11、13–14 和 16 以上；折叠屏和平板。
- 渲染进程被回收后的恢复、数据包不完整时的提示、原片缺失时的回看提示：代码路径在，没有触发过。
- 备份与换机迁移确实不带走数据（规则写了，没有做过真实的备份或迁移）。
- 安装包体积没有处理：带四种 CPU 架构的原生库和数据包。

### 第一次上真机时要看的

Debug 包，带上调试读数启动：`adb shell am start -n com.betabuddy.android/.MainActivity --ei initialTab 1 --ez openCapture true --ez debugCaptureOverlay true`，另开一个 `adb logcat -s capture`。

1. `first frame …: camera clock … s, uptime … s`：两个数应该差不多。差得很远时 `Recorder` 写不进画面，3 秒后应用会换成自己的编码器（日志里有 `switching to the app's own recorder`）。每段片子停下时有一行写明是哪个录像器写的。
2. 读数第一行的 `fovH`：竖着拿、主摄，应该在 40–60 之间；`frame` 应该是竖的 16:9（例如 720x1280）。
3. `pitch` / `roll`：手机竖直立着时都接近 0；镜头往上抬，`pitch` 变正；顺时针转（从屏幕这边看），`roll` 变正。`shake` 放稳时接近 0。
4. `replies/s` 在 2 上下；`hand-over` 几十毫秒以内；`pose` 有人入镜时是 `yes`。姿态识别跟不上会自己停掉，日志最后那行会写 `dropped`。
5. 对着墙：`issues` 里的代码和画面上方那句话对得上；色块、岩点描边和预览里的岩点重合。
6. 录 30 秒以上再停：片子能回看、是竖的、没有声音轨；再录一段超过 3 分钟的，看姿态识别做不做得完。
7. 录像中按 Home、来电、锁屏、被别的相机应用抢走：回来时都应该已经保存，卡片上写着被打断。
8. 第一次点「按距离找附近的岩馆」，系统的询问里选「大致位置」和「精确位置」各试一次。
9. 把手机架好拍完整的一趟，等「看分析」：分析页面应该写着用上了取景时标的点；架好之后故意挪一下再拍一趟，应该写没用上。

### 在 Apple 芯片的 Mac 上跑模拟器

较新的 Apple 芯片支持 SME 指令集，模拟器里的 Android 15 内核会把它报给应用，但实际执行会触发非法指令：MediaPipe 的原生库在第一次推理时以 `SIGILL` 崩溃。这是模拟器环境的问题，启动时让内核屏蔽 SME 即可；Android 12 的镜像不需要。

```sh
emulator -avd <名称> -qemu -append "arm64.nosme"
```

不带这个参数时，设备测试的第一个就会让应用进程崩溃，后面的测试一个都跑不到。另外，两个用设备相机的拍摄测试要模拟器有相机：AVD 里没配相机时，系统照样报有相机功能，测试不会跳过，而是等不到画面而超时。可以在启动时加 `-camera-back virtualscene`。

### 重新验证的步骤

```sh
python3 android/scripts/check-resources.py
cd android
./gradlew :app:assembleDebug :app:testDebugUnitTest :app:lintDebug
./gradlew :app:assembleRelease
"$ANDROID_HOME/build-tools/36.0.0/aapt2" dump permissions app/build/outputs/apk/release/app-release.apk
./gradlew :app:connectedDebugAndroidTest          # 需要连着设备或开着模拟器
```

预期：资源检查通过；单元测试全部通过；lint 没有 error；权限与上面的权限表一致；设备测试 37 个里 34 个运行并通过，跳过的 3 个是 2 个要真从系统取位置的和 1 个要给片子才跑的 `EditBacktest`（没有带分析页面的数据包时，回看页那一条自动剪辑的测试也跳过）。设备测试会往应用的外部文件目录写各个状态的截图（`capture-shots/`、`edit-shots/`），用 `adb shell am instrument` 直接跑（不经 Gradle 卸载应用）就能 `adb pull` 出来；截图不进仓库。然后用 Release 包在设备上走一遍：

1. 不装数据包的构建：落在岩馆，显示「这个构建没有内置岩馆数据」；底部齿轮、右上角按钮都能进「设置与关于」，隐私政策和开源许可能打开；拍摄界面能录能存；回看页没有「分析这一趟」。
2. 装了数据包的构建：说明页只出现一次 → 场景出现、能转动缩放、选线 → 切到攀爬册再回来不重载 → 页面菜单里打开原生设置 → 换一种语言，返回后页面是新语言并回到刚才的线路 → 开不开飞行模式表现一致。
3. 语言：五种语言加「跟随系统」逐个选；杀掉应用重开仍是所选语言；系统语言设成五种之外的语言时界面是英语。
4. 攀爬册导入：一段竖拍、一段横拍 → 选抱石 / 顶绳 / 先锋，难度选择器跟着换 → 保存 → 回看一打开就有画面、步进、慢放 → 完攀开关、分享文字 → 「分析这一趟」→ 「自动剪辑」：等它剪完、循环播放，三个开关各关一次，有路人走过的片子看被消的人是不是全程都消了、没被消的是不是全程都在，「分享视频」发给另一个应用 → 「完成」→ 删除。
5. 分析这一趟：没识别过姿态的记录点「分析这一趟」→ 先跑姿态识别 → 自动进入分析页面 →「正在准备画面…」→ 页面显示结果 → 页面里请求看视频里的某一刻，回看停在那一刻 →「看分析」和「重新分析」→ 杀掉应用重开，「看分析」直接显示保存的结果 → 删除记录后 `filesDir/BetaBuddy/` 里没有这条记录的结果文件。
6. 拍一趟：第一次出现相机询问（拒绝一次，看说明卡片和「改从相册导入」；再从系统设置里打开）→ 对着墙：提示、色块、岩点描边 → 选颜色、点一下标起步和结束 → 岩馆（「按距离找附近的岩馆」）→ 录 → 停 →「已保存到攀爬册」→ 放着等「看分析」→ 录像中按 Home、来电、锁屏各试一次，回来都应该已经保存。

岩馆页面的网络审计只能在 Debug 包上做：`adb forward tcp:9222 localabstract:webview_devtools_remote_<应用的 pid>`，再用 Chrome 的 `chrome://inspect` 或 DevTools 协议看 Network。
