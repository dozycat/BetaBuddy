# BetaBuddy · Android 工程

Kotlin + Jetpack Compose 原生应用，Android 8.0（API 26）及以上，竖屏，只有浅色。和 iOS 版是同一个产品：两个入口（岩馆、攀爬册）加「设置与关于」，五种语言并可在应用内切换，全部离线，不向用户申请任何权限。产品介绍见[仓库首页](../../README.md)，Android 上的设计见 [design.md](design.md)，岩馆的数据包接口见 [../ios/gym-design.md](../ios/gym-design.md)。

应用 ID `com.betabuddy.android`，版本 `1.1.0`（`android/app/build.gradle.kts` 里的 `versionName` / `versionCode`）。

## 构建

### 2026-11-14 · 攀爬册迭代验证（未发布）

文件夹、岩馆／难度／攀爬日期组合筛选及离线目录已接入。`check-resources.py`、77 项 JVM 测试、`assembleDebug`、`lintDebug` 通过；最后的布局和目录别名搜索修改增量构建通过。覆盖旧档兼容、文件夹删除保留媒体、写入失败的原子性和日期边界。

在 Android 12（API 31）只读模拟器上实际走过：创建文件夹→重启后保留→改名→删除；删除后 3 条合成记录全部保留并回到未归档；岩馆＋V 等级＋近 7 天叠加筛选，再叠加完攀状态得到空结果，清除条件恢复全部记录；导入表单展开可选分类，目录搜索并选择岩馆后正确回填。使用合成记录验证分类，没有声称本轮验证了真实视频识别或真机性能。

目录三份资源一致性与 6 项采集测试通过，覆盖范围见[目录说明](../gym-directory.md)。本轮未做最大字号／TalkBack 全流程、真机或逐馆营业核实。

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
      settings/                        设置与关于、隐私政策、开源许可
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
      licenses/                        第三方许可全文（设置里可以查看）
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
- **不加权限，不加联网依赖。** 新增依赖先看它的 Manifest 会合并进来什么权限和组件，该移除的在应用的 Manifest 里移除，并在 `THIRD_PARTY.md` 补一行；依赖有增减时同步 `assets/licenses/components.txt` 里的清单。
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

### 2026-11-13，模拟器

环境：Apple 芯片的 Mac，JDK 17，Android SDK 平台 37、Build-Tools 36.0.0。两台模拟器（Pixel 7 外形，1080×2400，主机 GPU）：Android 15（API 35，google_apis arm64，系统 WebView 124）和 Android 12（API 31，google_apis arm64，系统 WebView 91）。场馆数据包用的是真实的一份，装在本机、不进仓库。

**构建**

| 命令 | 结果 |
| --- | --- |
| `./gradlew :app:assembleDebug` | 通过，没有版本兼容告警 |
| `./gradlew :app:testDebugUnitTest` | 73 个测试全部通过 |
| `./gradlew :app:lintDebug` | 0 error。留下的 warning：有更新的 AGP / Gradle / MediaPipe 版本（版本是有意固定的）；两处「数字 + 名词」建议改成复数资源（线路数为 1 时英语会读成 “1 problems”）；`mipmap-anydpi-v26` 的限定符多余；四处建议改用 KTX 写法 |
| `./gradlew :app:assembleRelease` | R8 与资源压缩通过，没有缺类报错；未配置 keystore，用 debug 密钥签名 |
| `aapt2 dump permissions`（debug 与 release） | 只有 `com.betabuddy.android.DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION`；没有 `INTERNET`、`ACCESS_NETWORK_STATE`、`WAKE_LOCK`；合并后的 Manifest 里没有 datatransport 的组件，也没有 EmojiCompat 的启动项 |
| `./gradlew :app:connectedDebugAndroidTest` | `VideoAnalyzerInstrumentedTest` 7 个测试在 API 35 和 API 31 上都通过 |
| `python3 android/scripts/check-resources.py` | 通过 |

**岩馆（API 35，debug 包）**

- 没有数据包的构建：岩馆显示「这个构建没有内置岩馆数据」，攀爬册和「设置与关于」正常。
- 有数据包的构建：说明页只出现一次 → 进入后 3D 场景出现（WebGL 可用，后台线程启动），页面的 `ready` 在开始加载后约 0.3–0.4 秒到达（API 31 上约 0.6 秒）。
- 点选岩壁后页面选中一条线路并摆出人物；播放、底部抽屉的四个页签都能用；跳过手势提示后 `hintSeen` 记在本机，之后的入口 URL 带 `hint=0`。
- 页面「更多」里的「语言与设置」发出 `openSettings`，原生的「设置与关于」打开；返回后页面没有重载。切到攀爬册再切回也没有重载（页面内的计时连续，已选线路还在）。
- 调试参数 `gymOpenRoute`、`gymQuery` 生效：能直接摆出某条线路的某一步。给一个不存在的线路 id 时页面在 `ready` 之后报错，原生不盖失败页。
- **网络审计**（经 WebView 调试端口用 DevTools 协议抓整次重新加载）：11 个请求，9 个是数据包里的文件，2 个是 `blob:`（后台线程），没有任何其他地址，没有失败的请求。在页面里主动发起 `fetch`（https / http / `file:`）、加载远程图片、打开 WebSocket，全部被拦下；请求数据包目录之外的本地路径和清单里没有的文件得到 403。控制台除了模拟器 GL 的一条 `GL_INVALID_ENUM` 警告之外没有报错。

**语言**

- 在「设置与关于」里选语言：界面立即切换，仍停在设置页；返回岩馆时页面带新的 `lang` 重新加载并回到原来选中的线路。API 35 上系统的「应用语言」是同一个值。
- 五种语言各看了岩馆页面、攀爬册、设置页，没有发现漏翻和截断。
- 「跟随系统」：系统语言是五种之一时用它；系统语言设成法语时界面是英语。`zh-TW`、`zh-HK` 得到繁体，`zh-CN` 得到简体。
- 系统语言列表里第一位不受支持、后面有受支持语言时由 Android 自己决定用哪一种；试过的一次（法语在前、韩语在后）得到的是英语，不是韩语。
- API 31（没有系统级的应用语言）：应用内选的语言在创建 `WebView` 之后仍然有效，杀掉应用重开也还在。

**攀爬册（API 35，debug 包）**

- 照片选择器不需要任何权限就能打开，只列视频。
- 一段没有人的视频：分析结束后给出「没有清楚地找到攀爬者」的提示，不崩溃，不留记录。
- 一段有人的竖屏视频（用 MediaPipe 示例图合成，人在画面里移动）：导入 → 填名称、标记完攀 → 分析 → 回看。骨架与人重合，步进后骨架跟着人走，髋部短轨迹出现；慢放、原片 / 骨架切换可用；写笔记并保存、经系统分享面板分享文字、回到列表、删除记录（文件一并删掉）都走通。
- 一段带 90° 旋转信息的横向编码视频：分析和回看里画面都是正的，骨架重合。
- logcat 里 datatransport 报「没有注册的上报后端」，事件被丢弃，姿态识别不受影响；没有 `ACCESS_NETWORK_STATE` 时回看照常播放。

**Release 包（API 35）**

装上 R8 之后的包重新走了一遍：岩馆加载、`ready` 到达、页面发起的 `openSettings`、换语言后重载；攀爬册导入、分析、回看。全部正常。

**这一轮测出来并修掉的问题**

- Release 包上第一次分析就崩溃：MediaPipe 用的 Flogger 靠类名在调用栈里找调用者，被 R8 改名后抛出异常。`proguard-rules.pro` 里保留了 `com.google.common.flogger.**`。
- 从列表打开一条记录时回看画面是黑的，要步进或播放一次才出现：播放器在画面的 `TextureView` 有 surface 之前就解码了第一帧。改为拿到 surface 之后再开始解码。
- 「精细」档实际采样超过每秒 15 帧（3 秒的 30 帧视频采了 75 帧）：解码器的时间戳是整微秒，采样器的容差太小。修正并补了单元测试。
- lint 的两个 error：Manifest 里引用的启动项类来自只在运行时出现的依赖；一处在可组合函数里读了进程级的默认语言。

### 在 Apple 芯片的 Mac 上跑模拟器

较新的 Apple 芯片支持 SME 指令集，模拟器里的 Android 15 内核会把它报给应用，但实际执行会触发非法指令：MediaPipe 的原生库在第一次推理时以 `SIGILL` 崩溃（崩在读取 SME 向量长度的指令上）。这是模拟器环境的问题。启动模拟器时让内核屏蔽 SME 即可，上面攀爬册的结果都是这样测的；Android 12 的镜像内核不认识 SME，不需要这个参数。

```sh
emulator -avd <名称> -qemu -append "arm64.nosme"
```

### 尚未验证

- **任何真机。** 分析速度、发热与过热中止、内存；不同厂商的硬件解码器和颜色格式；HDR 视频；真实的攀岩视频（上面用的是合成视频，只说明流程和坐标对齐，不说明识别效果）。
- 支持 SME 的真机上 MediaPipe 是否正常（模拟器上的崩溃是环境问题，但没有真机可以对照）。
- Android 8–11、Android 13–14 和 Android 16 以上；折叠屏和平板；手势导航之外的预测性返回动画。
- 渲染进程被系统回收后的恢复、数据包不完整时的提示、视频文件丢失时的回看提示：代码路径在，没有实际触发过。
- 备份与换机迁移确实不带走数据（规则写了，没有做过真实的备份或迁移）。
- 最大字号、TalkBack。
- 安装包体积没有处理：现在带四种 CPU 架构的原生库，Release 包约 61 MB（含数据包）。

### 重新验证的步骤

```sh
cd android
./gradlew :app:assembleDebug :app:testDebugUnitTest :app:lintDebug
./gradlew :app:assembleRelease
"$ANDROID_HOME/build-tools/36.0.0/aapt2" dump permissions app/build/outputs/apk/release/app-release.apk
./gradlew :app:connectedDebugAndroidTest          # 需要连着设备或开着模拟器
adb install -r app/build/outputs/apk/release/app-release.apk
adb logcat -s gym AndroidRuntime
```

预期：单元测试 73 个、设备测试 7 个通过；lint 没有 error；权限里没有 `INTERNET`、`ACCESS_NETWORK_STATE`、`WAKE_LOCK`。然后用 Release 包在设备上走一遍：

1. 不装数据包的构建：落在岩馆，显示「这个构建没有内置岩馆数据」；底部齿轮、右上角按钮都能进「设置与关于」，里面的隐私政策和开源许可能打开。
2. 装了数据包的构建：说明页只出现一次 → 场景出现、能转动缩放、选线 → 切到攀爬册再回来不重载 → 在页面菜单里点「语言与设置」打开原生设置 → 换一种语言，返回后页面是新语言并回到刚才的线路 → 开不开飞行模式表现一致。
3. 语言：五种语言加「跟随系统」逐个选；杀掉应用重开仍是所选语言；系统语言设成五种之外的语言时界面是英语。
4. 攀爬册：导入一段竖拍视频和一段横拍视频 → 分析完成 → 回看时一打开就有画面、骨架与人重合、0.1 秒步进、0.5× 慢放 → 写笔记、标记完攀、分享文字 → 删除。Release 包上分析能跑完，说明 R8 没有裁坏 MediaPipe。

岩馆页面的网络审计只能在 Debug 包上做（只有它打开了 WebView 调试）：`adb forward tcp:9222 localabstract:webview_devtools_remote_<应用的 pid>`，再用 Chrome 的 `chrome://inspect` 或 DevTools 协议看 Network。
