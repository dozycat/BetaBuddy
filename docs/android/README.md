# BetaBuddy · Android 工程

Kotlin + Jetpack Compose 原生应用，Android 8.0（API 26）及以上，竖屏，只有浅色。和 iOS 版是同一个产品：两个入口（岩馆、攀爬册）加「设置与关于」，五种语言并可在应用内切换，全部离线，不向用户申请任何权限。产品介绍见[仓库首页](../../README.md)，Android 上的设计见 [design.md](design.md)，岩馆的数据包接口见 [../ios/gym-design.md](../ios/gym-design.md)。

应用 ID `com.betabuddy.android`，版本 `1.1.0`（`android/app/build.gradle.kts` 里的 `versionName` / `versionCode`）。

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
| Compose BOM | 2026.09.00 | ui 与 foundation 1.12.1，material3 1.4.0 |
| core-ktx / activity-compose / appcompat | 1.19.1 / 1.13.0 / 1.8.0 | |
| lifecycle / navigation-compose / webkit | 2.11.0 / 2.10.2 / 1.18.0 | |
| Media3 ExoPlayer | 1.11.1 | |
| MediaPipe Tasks Vision | 0.10.35 | 0.10 线的最后一个版本 |
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

「岩馆」入口显示的场馆模型、线路和推演来自一个在仓库之外生成的数据包。它**不在这个仓库里，也不要提交进来**。没有它，工程照常编译和测试，入口显示「这个构建没有内置岩馆数据」。

```sh
./android/scripts/install-gym-pack.sh /path/to/GymPack   # 核对 manifest 与每个文件的 SHA-256 后拷到 app/src/main/assets/gympack/
./android/scripts/install-gym-pack.sh --remove           # 回到不含数据包的构建
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
  THIRD_PARTY.md
  app/
    build.gradle.kts, proguard-rules.pro
    src/main/AndroidManifest.xml
    src/main/java/com/betabuddy/android/
      MainActivity.kt                  唯一的 Activity；边到边、调试启动参数
      ui/theme/                        BB 颜色与字体、Material 3 底座
      ui/common/                       PaperPage、Stamp、BigButton、SectionLabel、ColorCard、BBIcons…
      ui/nav/                          AppShell：两个入口、底部导航与齿轮、设置页的导航
      gym/                             数据包、WebView 容器、消息桥、岩馆界面
      settings/                        设置与关于、隐私政策
      i18n/                            AppLanguage：当前界面语言与应用内语言选择
      journal/                         攀爬册：列表、导入、文字
      pose/                            解码、姿态识别、几何
      review/                          回看：播放与骨架叠加
      data/                            记录的数据结构与本地存储
    src/main/res/
      values*/strings.xml              导航、岩馆、设置的文字（五种语言）
      values*/strings_journal.xml      攀爬册的文字（五种语言，名字都以 journal_ 开头）
      raw*/privacy_policy.txt          内置隐私政策（简中、繁中、英）
      xml/locales_config.xml           应用支持的语言
      xml/data_extraction_rules.xml    不进云备份，也不随换机迁移
    src/main/assets/
      pose_landmarker_full.task        姿态模型
      gympack/                         场馆数据包（被忽略，由脚本装入）
    src/test/                          JVM 单元测试（纯 Kotlin，不依赖 Android 类）
    src/androidTest/                   设备测试（解码与推理）
```

## 与 iOS 的对应

| iOS | Android |
| --- | --- |
| `App/BetaBuddyApp.swift`（`RootView`） | `MainActivity.kt`、`ui/nav/AppShell.kt` |
| `Views/DesignSystem.swift`、`BrandMark.swift` | `ui/theme/Theme.kt`、`ui/common/Components.kt`、`ui/common/Icons.kt` |
| `Core/GymPack.swift` | `gym/GymManifest.kt`、`gym/GymPack.kt`、`gym/GymMessage.kt` |
| `Views/GymView.swift`（`GymSession`、`GymView`） | `gym/GymSession.kt`、`gym/GymScreen.kt` |
| `Core/AppLanguage.swift` | `i18n/LanguageTags.kt`、`i18n/AppLanguage.kt` |
| `Views/AboutView.swift` | `settings/SettingsScreens.kt` |
| `Core/Models.swift`、`ClimbStore.swift` | `data/Models.kt`、`data/ClimbStore.kt` |
| `Core/PoseMath.swift`、`VideoAnalyzer.swift` | `pose/PoseMath.kt`、`pose/VideoAnalyzer.kt`、`pose/AnalysisTypes.kt` |
| `Views/JournalView.swift`、`ImportView.swift` | `journal/JournalScreen.kt`、`journal/ImportScreen.kt` |
| `Views/ReviewView.swift` | `review/ReviewScreen.kt`、`review/Playback.kt`、`review/SkeletonOverlay.kt` |
| `Resources/Localizable.xcstrings` | `res/values*/strings.xml`、`strings_journal.xml` |
| `Resources/PrivacyPolicy.<语言>.txt` | `res/raw*/privacy_policy.txt` |
| `ios/GymPack/`（被忽略） | `app/src/main/assets/gympack/`（被忽略） |
| `ios/scripts/install-gym-pack.sh` | `android/scripts/install-gym-pack.sh` |
| `BetaBuddyTests/GymPackTests.swift`、`LocalizationTests.swift` | `src/test/.../gym/GymPackTest.kt`、`i18n/ShellStringsTest.kt`、`journal/JournalStringsTest.kt` |

## 约定

- **界面用同一套组件。** 颜色和字体取 `ui/theme` 的 `BB`（`Paper/Ink/Lime/Lavender/Muted/Line`、`display()`、`mono()`、`Title…Caption`），页面和控件取 `ui/common`（`PaperPage`、`Stamp`、`BigButton`、`SectionLabel`、`Stat`、`CircleIconButton`、`ColorCard`、`Hairline`、`BBIcons`）。岩馆、攀爬册、设置都依赖这些名字，改名要一起改。
- **攀爬册只有一个入口** `JournalScreen(onOpenSettings)`。它内部的页面用 `rememberSaveable` 的路由加 `BackHandler` 切换，不往 NavHost 里加路由。导入和回看时底部导航仍然可见；分析中切到岩馆不会停止分析，应用退到后台才会停。
- **存储**都在应用私有目录：攀爬册在 `filesDir/BetaBuddy/`（`archive.json`、原片、缩略图、`staging/`），岩馆只有 `SharedPreferences` 里的两个布尔值。不写外部存储。
- **文字**：导航、岩馆、设置的在 `strings.xml`，攀爬册的在 `strings_journal.xml` 且名字以 `journal_` 开头，两个文件不重名。
- **不加权限，不加联网依赖。** 新增依赖先看它的 Manifest 会合并进来什么权限和组件，该移除的在应用的 Manifest 里移除，并在 `THIRD_PARTY.md` 补一行。
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
adb shell am start -n $A --ez gymNoticeSeen true --es gymQuery "sheet=half&tab=steps"
adb shell am start -n $A --ez gymNoticeSeen true --es gymOpenRoute <线路 id>
adb shell am start -n $A --ez gymNoticeSeen true --es gymOpenWall <岩壁 id>
adb exec-out screencap -p > shot.png
```

参数只在全新启动时读取，换参数前先 `adb shell am force-stop com.betabuddy.android`。`gymQuery` 原样接在入口 URL 的 `hint`、`lang` 之后，页面自己定义其中的参数。岩馆页面可以用 Chrome 的 `chrome://inspect` 调试；日志看 `adb logcat -s gym`。

## 验证状态

做过的和没做过的分开记，没做的不写成做了。

**已经实际跑过**（在一台没有 Android SDK 的 Mac 上）

- Gradle wrapper 启动，下载并校验 Gradle 9.7.1；`./gradlew help` 通过，说明三个插件（AGP 9.3.3、Compose 与序列化编译器插件 2.4.21）能解析、构建脚本能配置。
- `./gradlew :app:dependencies --configuration debugRuntimeClasspath` 通过：版本目录里的每个依赖连同传递依赖都能解析，没有版本冲突告警。
- 全部 Kotlin 源码（应用、单元测试、设备测试）用 Kotlin 2.4.21 的编译器加 Compose 与序列化插件做过类型检查，零错误零警告。依赖用的是 Gradle 解析出来的同一批 jar；Android 框架用的是 AOSP 的 API 37 框架 jar（Robolectric 发布在 Maven Central 的 `android-all`），`R` 和 `BuildConfig` 是按资源文件生成的桩。这验证的是每个 API 的包名、签名和可见性，**不是** Android 工程的构建。
- 单元测试 72 个全部通过（`GymPackTest`、`LanguageTagsTest`、`ShellStringsTest`、`ClimbStoreTest`、`PoseMathTest`、`AnalysisTypesTest`、`PoseModelTest`、`OverlayGeometryTest`、`JournalTextTest`、`JournalStringsTest`），用上面编出来的类在 JDK 17 上直接跑，运行时类路径里没有 Android 框架。
- `scripts/check-resources.py` 通过：资源 XML 格式、五种语言 134 条文字的键 / 占位符 / 换行一致、两个文字文件不重名、代码引用的资源都存在且没有没用到的文字、`locales_config.xml` 与资源目录一致、Manifest 不声明权限、备份规则两节都排除全部数据域。
- `install-gym-pack.sh`：用真实数据包安装、`--remove` 卸载；安装后的目录被 Git 忽略；多出未列文件、文件被改动、用了 APK 不收的文件名三种情况都被拒绝。安装后的清单用 `GymManifest` / `GymPack` 读过一遍，状态为就绪。
- 81 个依赖的 Manifest 用 Gradle 导出后逐个看过：会合并进来的系统权限是 datatransport 的 `INTERNET`、`ACCESS_NETWORK_STATE` 和 Media3 的 `ACCESS_NETWORK_STATE`、`WAKE_LOCK`，都已移除。看过字节码：移除后端发现服务之后 datatransport 的事件在入库和排程前被丢弃；Media3 读网络状态时对没有权限的情况有保护。

**尚未验证（要等 Android SDK 和设备）**

- 应用本身的构建：资源编译（aapt2）、Manifest 合并、Compose 编译产物、dex、lint 都没有实际跑过。
- Release 构建：R8 的 keep 规则是否足够（MediaPipe 与 protobuf 经 JNI 和反射访问），资源压缩有没有误删。
- 任何设备或模拟器上的运行。岩馆：页面加载、WebGL、Blob Worker、消息桥、`openSettings`、离线拦截（没有联网权限加 `blockNetworkLoads` 时本地域名仍由应用应答）、切入口不重载、切语言重载并回到原线路、渲染进程被回收后的恢复、安全区。攀爬册：照片选择器、解码与颜色转换、推理、回看时骨架与画面重合、分享。整体：边到边、预测性返回、应用内语言切换（尤其 Android 12 及以下创建 `WebView` 之后界面语言不被改回系统语言）、设备测试 `VideoAnalyzerInstrumentedTest`。
- 合并后的 Manifest 里确实没有系统权限；去掉 datatransport 的组件之后姿态识别照常工作，没有 `ACCESS_NETWORK_STATE` 时回看照常播放，logcat 里没有异常。
- 应用还没有把第三方许可全文打进安装包（iOS 版有）。发布前要补。

### SDK 就绪后的验证步骤

按顺序做，每一步写着预期结果；哪一步不符就先停在那里。

```sh
cd android
./gradlew :app:assembleDebug
```
预期：`BUILD SUCCESSFUL`，没有 Kotlin 或 AGP 的版本兼容告警。

```sh
./gradlew :app:testDebugUnitTest
```
预期：72 个测试通过。

```sh
./gradlew :app:lintDebug
```
预期：没有 error。重点看 `MissingTranslation`、`ExtraTranslation`、`NewApi`、权限相关的条目。

```sh
"$ANDROID_HOME/build-tools/36.0.0/aapt2" dump permissions app/build/outputs/apk/debug/app-debug.apk
"$ANDROID_HOME/build-tools/36.0.0/aapt2" dump xmltree --file AndroidManifest.xml app/build/outputs/apk/debug/app-debug.apk | grep -iE "permission|datatransport"
```
预期：只有构建自动加的 `com.betabuddy.android.DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION`（签名级、应用自用），没有 `INTERNET`、`ACCESS_NETWORK_STATE`、`WAKE_LOCK`；没有 datatransport 的 service 和 receiver。

```sh
./gradlew :app:assembleRelease
```
预期：R8 通过，没有 `Missing class` 报错。如果报缺类，把报出的类按提示加进 `proguard-rules.pro` 的 `-dontwarn`，不要整体关掉压缩。

```sh
adb install -r app/build/outputs/apk/release/app-release.apk     # 没配 keystore 时是 debug 密钥签的
adb logcat -s gym AndroidRuntime
```
在设备上（至少一台 Android 13 以上、一台 Android 8–12）用 Release 包走一遍：

1. 不装数据包的构建：落在岩馆，显示「这个构建没有内置岩馆数据」；底部齿轮、右上角按钮都能进「设置与关于」。
2. 装了数据包的构建：说明页只出现一次 → 进入后场景出现、能转动缩放、选线 → 切到攀爬册再回来不重载 → 在页面菜单里点「语言与设置」打开原生设置 → 换一种语言，返回后页面是新语言并回到刚才的线路 → 打开飞行模式与否表现一致。
3. 语言：五种语言加「跟随系统」逐个选；杀掉应用重开仍是所选语言；Android 13+ 的系统设置「应用语言」里显示同一个值；系统语言设成五种之外的语言时界面是英语。
4. 攀爬册：导入一段竖拍视频和一段横拍视频 → 分析完成 → 回看时骨架与人重合、0.1 秒步进、0.5× 慢放 → 写笔记、标记完攀、分享文字 → 删除。Release 包上分析能跑完就说明 R8 没有裁掉 MediaPipe。
5. `adb shell bmgr backupnow com.betabuddy.android` 预期被拒绝或不含数据；换机迁移不带走记录。

```sh
./gradlew :app:connectedDebugAndroidTest
```
预期：`VideoAnalyzerInstrumentedTest` 通过。这份测试是没有 SDK 时写的，失败先怀疑测试代码。
