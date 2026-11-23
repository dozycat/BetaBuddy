# BetaBuddy · Android 工程

Kotlin + Jetpack Compose 原生应用，Android 8.0（API 26）及以上，竖屏，只有浅色。和 iOS 版是同一个产品：两个入口（岩馆、攀爬册）加「设置与关于」，五种语言并可在应用内切换，全部离线；后台分析使用本地媒体处理服务，进度通知权限可选；应用内相机用到相机权限，按距离给岩馆排序用到位置权限（大致或精确，由用户选），两项都是用到时才问。产品介绍见[仓库首页](../../README.md)，Android 上的设计见 [design.md](design.md)，岩馆的数据包接口见 [../ios/gym-design.md](../ios/gym-design.md)。

应用 ID `com.betabuddy.android`，版本 `1.2.0`（`android/app/build.gradle.kts` 里的 `versionName` / `versionCode`）。

## 构建

### 2026-11-23 · 应用内相机「拍一趟」（未发布）

攀爬册的主按钮改成「拍一趟」，导入退为次要按钮。拍摄界面取景时由数据包的拍摄页面检查机位、认颜色和岩点，用户选颜色、点一下标起步和结束、选难度和岩馆；停止后先保存，再做姿态识别和「分析这一趟」。相机用 CameraX 1.7.0（预览、不带声音的 1080p 录像、给检查用的小帧）；位置用平台自带的 `LocationManager`，大致和精确两条权限一起申请、由用户选。Manifest 多了 `CAMERA`、`ACCESS_COARSE_LOCATION` 和 `ACCESS_FINE_LOCATION`，没有 `RECORD_AUDIO`。存档的记录多一个可选的 `capture` 字段（与 iOS 同名同义），拍摄计划存成 `<记录 id>.capture.json`；旧存档照常读取。设计见 [design.md](design.md#拍一趟)，验证见下面的[验证状态](#2026-11-23模拟器应用内相机)。

### 2026-11-18 · 攀爬册的分析换成数据包里的分析页面（未发布）

回看页原来的指标（检测比例、肘角）、候选片段、髋部短轨迹和模型信息行下掉，对应的存档字段、计算、文字和测试一并删除；旧存档照常读取。姿态识别多存一份 world landmarks，数值保留 4 位小数。回看页新增「分析这一趟」：原生把画面和姿态交给场馆数据包里的分析页面，保存它交回的结果。数据包没有这个页面（或没有数据包）时不出现入口。设计见 [design.md](design.md#分析这一趟)。

实际做过的：

- `python3 android/scripts/check-resources.py` 通过（168 条文字，五种语言）。
- `./gradlew :app:testDebugUnitTest`：85 项 JVM 测试通过。新增或改写的覆盖：早期存档（含已下掉的字段、没有 world landmarks）照常读取且不丢记录；world landmarks 的读写与 4 位小数；分析结果文件的原样保存、覆盖、随记录删除、原子写入与中断清理；数据包清单里有没有分析页面决定入口；页面消息的解析（结果原样、进度与未知类型忽略、岩馆消息不受影响）；传给页面的调用经 JSON 编码（带引号、反斜杠、换行的字符串）；取帧时间与尺寸；姿态分批。测试夹具都是合成数据。
- `./gradlew :app:assembleDebug :app:lintDebug`：通过，lint 0 error（警告与上一轮相同：固定版本的更新提示、两处复数建议、KTX 写法等）。
- `./gradlew :app:compileDebugAndroidTestKotlin`：设备测试能编译（新增一项取帧测试，原推理测试加了 world landmarks 的断言）。
- `./gradlew :app:assembleRelease`：R8 与资源压缩通过，没有缺类报错（只说明能构建；产物没有安装运行，已删除）。
- 上面的构建都是在装着数据包（清单含分析页面）的工作区里做的。单元测试不读数据包，用合成的清单。
- 模拟器验证之后又改了一处：交给分析页面的画面关键点可见度改用识别器原始的可见度（与 world landmarks 里的同一个值，和 iPhone 版一致）。改后重跑了 `check-resources.py`、85 项 JVM 测试、`assembleDebug` 和 `lintDebug`（数据包 1.4.5），都通过；这一处改动当时没有再上模拟器，后面第二轮模拟器验证用的是含这处改动的构建。
- 之后又把交给分析页面的画面改成按短边 540 取（横屏 1920×1080 的视频取 960×540，原来是 540×304），改了对应的单元测试；`check-resources.py`、85 项 JVM 测试、`assembleDebug`、`lintDebug` 在数据包 1.5.0 下重跑通过。横屏视频和这次改动之后的流程当时没有上模拟器，后面第二轮模拟器验证走了一段横屏视频；真机没有跑过。数据包换到 1.5.1 后只重跑了 85 项 JVM 测试和 `assembleDebug`（通过）；提交前在数据包 1.5.3 下重跑了 `check-resources.py`、85 项 JVM 测试、`assembleDebug`、`lintDebug`（通过）。

**模拟器上实际走过的**（一台已有的模拟器：Android 15 / API 35，google_apis arm64，Pixel 外形 1080×2400，系统 WebView 124；Apple 芯片的 Mac，启动时带 `arm64.nosme`；debug 包，内置数据包清单含分析页面）：

- `./gradlew :app:connectedDebugAndroidTest`：9 项设备测试全部通过，包括新增的取帧测试和推理结果里 world landmarks 的断言。
- 一段 16.6 秒、720×1280、30 帧/秒的真实攀爬视频（放在仓库之外，推到模拟器后经系统照片选择器由应用自己的导入界面导入，默认只保存、不识别姿态）。界面由 adb 驱动，逐步截图看过。走了两遍：简体中文一遍，英语一遍。
- 「分析这一趟」→ 姿态识别经原有队列运行（存档里 166 个采样帧、164 帧保留，每帧 33 个 `joints` 和 33 个 `world`，4 位小数；`archive.json` 约 408 KB）→ 结束后分析页面自动打开 → 顶栏出现「正在准备画面…」→ 页面依次显示自己的几个阶段 → 停在结果上（墙面图、带编号的点、步进控制、结论卡片）。结果文件 `<记录 id>.analysis.json` 约 101 KB，`journal` 偏好里记下了身材参数。
- 全屏对话框：盖住底部导航，页面在状态栏和手势条之内，状态栏图标是深色，「完成」和系统返回都能关掉。
- 页面里走到第 4 步点「看视频里这一刻」：对话框关闭，回看停在 00:07，处于暂停。
- 回到回看页，按钮变成「看分析」和「重新分析」。「看分析」约 3 秒后显示保存的结果，结果文件的修改时间没有变（没有重算）。在页面里把身高加 2 cm：约 14 秒后结果文件被覆盖，偏好里的身高变成 172。
- 「重新分析」：旧结果文件先消失，重新取帧、计算，写入新结果（带着偏好里记下的 172 cm）。之后杀掉应用重开，「看分析」显示的是这份保存的结果。
- 系统的通知权限询问盖在应用上面时（没有点掉），姿态识别、自动打开、取帧和计算照常完成；点「不允许」之后界面正常。
- 删除记录：`filesDir/BetaBuddy/` 里只剩索引，原片、缩略图和结果文件都没有了。
- 岩馆页面（改用共用的 `PackWebView` 之后）：冷启动后 3D 场景出现，`ready` 在开始加载后约 6.5 秒到达；跳过手势提示后 `hintSeen` 记进了偏好（消息桥可用）；英语界面下页面是英语。
- 整个过程 logcat 里没有应用崩溃、渲染进程被回收或分析页面的报错。这一轮没有发现需要改代码的问题。

这台模拟器上的耗时（软件解码，宿主机同时在跑别的任务，只说明量级）：

| 阶段 | 耗时 |
| --- | --- |
| 姿态识别（16.6 秒的视频，每秒 10 帧） | 26–31 秒 |
| 分析页面从打开到就绪 | 约 3 秒 |
| 取 60 帧画面（「正在准备画面…」） | 约 43 秒 |
| 页面计算到出结果 | 约 27 秒 |
| 从分析页面打开到结果保存，合计 | 73–75 秒（三次） |
| 在页面里改身高后的重算 | 约 14 秒 |

取帧是最慢的一段：这段视频是重新编码过的，关键帧间隔很长，按时间逐帧精确定位时每一帧都要从上一个关键帧解起。手机直接拍的视频关键帧密得多，真机上的耗时没有测过。

**第二轮模拟器验证**（同一天稍后，同一台模拟器、同样的启动参数；debug 包重新构建，「设置与关于」里显示的数据版本是 1.5.4；代码是按短边 540 取帧、画面关键点用识别器原始可见度之后的版本）。这一轮只走了攀爬册的分析流程，没有重跑设备测试和岩馆页面。通知权限事先用 adb 授予，没有再出现权限询问。视频都在仓库之外，推到模拟器后经应用自己的导入界面导入（默认只保存），界面由 adb 驱动并逐步截图看过，简体中文界面。

| 视频 | 结果 | 姿态识别 | 取帧 | 页面计算 |
| --- | --- | --- | --- | --- |
| 竖屏 204×360，手持，45.9 秒 | 出结果：一整张墙面图、带编号的点、小人、步进控制、结论卡片；结果约 97 KB | 37 秒（459 采样帧，保留 432） | 约 25 秒 | 约 24 秒 |
| 横屏 1920×1080，镜头跟着人上仰，26.4 秒 | 出结果：墙面图是拼好的完整一张，铺满画面（不是只有左上角有内容的黑图），带编号的点、小人、结论卡片；结果约 246 KB | 30 秒（264 采样帧，保留 212） | 约 69 秒 | 约 34 秒 |
| 横屏 1280×720，转播画面，有剪辑，95.2 秒 | **预期页面会拒绝，实际出了结果**（42 步，墙面图有重影）；结果约 167 KB，入口变成「看分析」。页面的提示写着：有剪辑只分析了最长的一段、上墙 2 次只分析了一趟、镜头晃动已对齐、垫子高度是估的、17 个姿势没参与比较 | 87 秒（953 采样帧，保留 877） | 约 37 秒 | 约 35 秒 |
| 上面竖屏视频里截出的 2 秒（为了触发拒绝另加的） | 页面显示「没认出上墙的那一段（双手在点上、双脚离地）。」，没有保存结果，「完成」关闭后入口仍是「分析这一趟」 | 3 秒（20 帧） | 约 4 秒 | 取完帧随即给出说明 |
| 横屏 1920×1080，转播画面，25 秒（为了触发拒绝另加的） | 页面显示「没能从这段视频里定出墙面。手脚在墙上的画面太少，或者镜头在动。」和一张视频画面，没有保存结果，「完成」关闭后入口仍是「分析这一趟」 | 25 秒（250 采样帧，保留 88） | 约 30 秒 | 取完帧后约 1–2 秒给出说明 |

- 取帧和页面计算的时间是每 2–3 秒看一次界面得到的，只说明量级；姿态识别的时间取自前台服务的起止日志。
- 横屏那一段：页面里走到第 4 步点「看视频里这一刻」，对话框关闭，回看停在 00:19、处于暂停，视频区域回到可见位置；再点「看分析」约 5 秒内显示保存的结果，结果文件的修改时间没有变。
- 两次拒绝时原生都只记了一行日志（页面给的代码），没有写结果文件；页面上只看到说明文字（第二次还有一张画面），没有看到说明下面的灰色小字。
- 第三段没有被拒绝属于页面自己的判断，原生传给页面的数据和其它几段是同一条路径，这一轮没有改代码。
- 前三段导入并识别之后 `archive.json` 约 3.8 MB（主要是 95 秒那一段的 877 帧）。
- logcat 里没有应用崩溃、渲染进程被回收或页面脚本报错。这一轮没有发现原生侧需要改的问题，没有改代码。

尚未验证：

- **任何真机**：取帧、姿态识别和页面计算的真实耗时与发热；硬件解码器下取到的画面；长视频（接近 3 分钟）或高分辨率视频取帧时的内存。
- **Android 15 以外的系统**：包括 API 26 上不带缩放的取帧路径（代码里有，设备测试只在 API 35 上跑过），以及旧版本 WebView 上分析页面是否可用（已有的 API 31 模拟器这一轮没有启动）。
- 页面没加载出来或渲染进程被回收时的说明卡片与「重新加载」、原片缺失时的提示：代码路径在，没有触发过。（页面报告「分析不了」在第二轮触发过两次。）
- 第二轮之后没有重跑的：设备测试（按短边取帧之后新的取帧尺寸只有单元测试和上面这几段视频）、岩馆页面、英语界面、「重新分析」、在页面里改参数后的覆盖保存、删除记录。
- 旧存档（没有 world landmarks 的已识别记录）在设备上点「分析这一趟」：走的是和「从未识别」同一条路径，那条路径走过了；旧存档本身只有单元测试。
- 等待姿态识别期间点「停止」、离开回看页再回来、分析页面开着时 Activity 重建。
- 繁体中文、日语、韩语界面下新文字的排版（简体中文和英语看过）；最大字号、TalkBack。
- 岩馆页面的换语言重载回到原线路、`openSettings`、网络审计在这次重构之后没有重新做（只看了加载、`ready` 和一条消息）。
- 不装数据包的构建：这一轮没有移走已安装的数据包重新构建。代码对数据包文件没有编译期依赖，「清单里没有分析页面就没有入口」有单元测试，但没有数据包时的构建和界面这一轮没有实际看过。
- Release 包（R8 之后）上的这条流程：只确认过能构建，没有安装运行。

### 2026-11-15 · 默认保存、后台分析与拍摄时间（未发布）

`check-resources.py`、80 项 JVM 单元测试、`assembleDebug`、`lintDebug` 均通过（lint 无 error）。Android 15 独立模拟器上的 8 项设备测试通过，包括新增的「保存请求后退到桌面，后台分析完成并保留原片与笔记」。使用仓库既有的 `arm64.nosme` 模拟器参数，避免宿主 SME 指令误报导致 MediaPipe SIGILL；最初未加参数的设备运行失败，不计作通过结果。

默认只保存原片；拍摄时间优先使用所选媒体日期，其次解析视频元数据，支持时区与毫秒；缺失时明确回退当前时间，可编辑日期和时分。分析请求与记录原子保存，后台服务负责执行，进程终止后在下次打开应用重试。JVM 测试覆盖时间格式、无效日期、请求持久化、停止与分析完成保留期间编辑的数据。仅增加本地前台服务和可选通知权限，没有增加网络或相册权限。

尚未验证：真机后台配额、厂商省电限制、耗电／温升，以及本轮 Android 8–14 的服务实机行为；拒绝通知不影响分析，但通知栏不显示进度。

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
| CameraX（camera-core / camera2 / lifecycle / video / view） | 1.7.0 | 应用内相机；连带 media3-muxer（跟随 Media3 的版本） |
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
      gym/                             数据包、WebView 容器（岩馆页面与分析页面共用）、消息桥、岩馆界面
      settings/                        设置与关于、隐私政策、开源许可
      i18n/                            AppLanguage：当前界面语言与应用内语言选择
      capture/                         应用内相机：拍摄页面的消息、画面来源（CameraX / 文件）、取帧、拍摄界面、岩馆建议
      journal/                         攀爬册：列表、导入、文字、岩馆目录与距离、相机开关与分析队列的规则
      pose/                            解码、姿态识别、几何
      review/                          回看：播放与骨架叠加；数据包分析页面的宿主；相机记录的后台分析队列
      data/                            记录的数据结构与本地存储
    src/main/res/
      values*/strings.xml              导航、岩馆、设置的文字（五种语言）
      values*/strings_journal.xml      攀爬册的文字（五种语言，名字都以 journal_ 开头）
      values*/strings_capture.xml      应用内相机的文字（五种语言，名字都以 capture_ 开头，译文取自 iOS 的字符串表）
      raw*/privacy_policy.txt          内置隐私政策（简中、繁中、英）
      xml/locales_config.xml           应用支持的语言
      xml/data_extraction_rules.xml    不进云备份，也不随换机迁移
    src/main/assets/
      pose_landmarker_full.task        姿态模型
      licenses/                        第三方许可全文（设置里可以查看）
      gympack/                         场馆数据包（被忽略，由脚本装入）
    src/test/                          JVM 单元测试（纯 Kotlin，不依赖 Android 类）
    src/androidTest/                   设备测试（解码与推理；拍摄界面，用替身数据包和画出来的墙）
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
| `Core/Models.swift`、`ClimbStore.swift` | `data/Models.kt`、`data/ClimbStore.kt` |
| `Core/PoseMath.swift`、`VideoAnalyzer.swift` | `pose/PoseMath.kt`、`pose/VideoAnalyzer.kt`、`pose/AnalysisTypes.kt` |
| `Views/JournalView.swift`、`ImportView.swift` | `journal/JournalScreen.kt`、`journal/ImportScreen.kt` |
| `Views/ReviewView.swift` | `review/ReviewScreen.kt`、`review/Playback.kt`、`review/SkeletonOverlay.kt` |
| `Views/AnalysisView.swift` | `review/AnalysisScreen.kt`、`review/AnalysisSession.kt`、`review/AnalysisBridge.kt`、`review/AnalysisFrames.kt` |
| `Core/Capture.swift`、`Views/CaptureText.swift` | `capture/Capture.kt`、`capture/CaptureText.kt` |
| `Core/CaptureSource.swift` | `capture/CaptureSource.kt`、`capture/FileCaptureSource.kt`、`capture/CapturePipeline.kt` |
| `Core/CaptureSession.swift`、`Views/CaptureView.swift` | `capture/CaptureSession.kt`、`capture/CaptureScreen.kt`、`capture/CaptureGymSheet.kt` |
| `Core/GymLocator.swift` | `journal/GymDirectory.kt`（目录与距离）、`capture/GymLocator.kt`（取一次位置） |
| `Core/AnalysisQueue.swift` | `journal/CameraGate.kt`、`review/AnalysisQueue.kt` |
| `TestSupport/`（替身页面、合成的墙） | `src/androidTest/assets/*-standin.html`、`src/androidTest/.../capture/StandIns.kt` |
| `Resources/Localizable.xcstrings` | `res/values*/strings.xml`、`strings_journal.xml` |
| `Resources/PrivacyPolicy.<语言>.txt` | `res/raw*/privacy_policy.txt` |
| `ios/GymPack/`（被忽略） | `app/src/main/assets/gympack/`（被忽略） |
| `ios/scripts/install-gym-pack.sh` | `android/scripts/install-gym-pack.sh` |
| `BetaBuddyTests/GymPackTests.swift`、`LocalizationTests.swift` | `src/test/.../gym/GymPackTest.kt`、`i18n/ShellStringsTest.kt`、`journal/JournalStringsTest.kt` |

## 约定

- **界面用同一套组件。** 颜色和字体取 `ui/theme` 的 `BB`（`Paper/Ink/Lime/Lavender/Muted/Line`、`display()`、`mono()`、`Title…Caption`），页面和控件取 `ui/common`（`PaperPage`、`Stamp`、`BigButton`、`SectionLabel`、`Stat`、`CircleIconButton`、`ColorCard`、`Hairline`、`BBIcons`）。岩馆、攀爬册、设置都依赖这些名字，改名要一起改。
- **攀爬册只有一个入口** `JournalScreen(onOpenSettings)`。它内部的页面用 `rememberSaveable` 的路由加 `BackHandler` 切换，不往 NavHost 里加路由。导入和回看时底部导航仍然可见；分析由服务持有，切到岩馆、离开回看页或应用退到后台均不取消。系统结束服务时保留请求，下一次打开应用重试。
- **存储**都在应用私有目录：攀爬册在 `filesDir/BetaBuddy/`（`archive.json`、原片、缩略图、分析页面交回的 `<记录 id>.analysis.json`、拍摄页面交回的 `<记录 id>.capture.json`、`staging/`——导入和录像中的临时文件，下次启动清掉），外加 `SharedPreferences` 里分析页面上一次用的身材参数；岩馆只有 `SharedPreferences` 里的两个布尔值。不写外部存储。
- **文字**：导航、岩馆、设置的在 `strings.xml`，攀爬册的在 `strings_journal.xml` 且名字以 `journal_` 开头，应用内相机的在 `strings_capture.xml` 且名字以 `capture_` 开头，文件之间不重名。
- **不加联网依赖。** 权限只有本地分析服务、可选进度通知、相机和位置（大致与精确，`check-resources.py` 按这个名单检查，并要求移除 `INTERNET`、`ACCESS_NETWORK_STATE`、`WAKE_LOCK`、`RECORD_AUDIO`）；新增依赖先看它的 Manifest 会合并进来什么权限和组件，该移除的在应用的 Manifest 里移除，并在 `THIRD_PARTY.md` 补一行；依赖有增减时同步 `assets/licenses/components.txt` 里的清单。
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

# 应用内相机，在没有像样相机的地方（模拟器）：
adb shell am start -n $A --ei initialTab 1 --ez openCapture true          # 直接打开拍摄界面
adb shell "am start -n $A --ei initialTab 1 --ez openCapture true \
    --es debugCaptureVideo <文件> --es debugCaptureFov 50 \
    --es debugCaptureAttitude '6,0.5,0.01' --es debugLocation '<纬度>,<经度>,<误差米>'"
```

`debugCaptureVideo` 用一个视频文件循环播放代替相机（绝对路径，或者应用 `files/` 目录下的文件名；用 `adb push` 加 `run-as <应用 ID> cp` 放进去）；`debugCaptureFov` 是替它报的水平视场角；`debugCaptureAttitude` 是没有运动传感器时替它报的仰角、侧倾和晃动；`debugLocation` 给一个位置，不出现系统的询问。另有 `--ez debugCaptureOverlay true`（在画面上画出交给页面的数和页面回话的快慢）和 `--es debugCaptureKeepFrame <文件名>`（把最近交给页面的一帧另存到应用的 `files/` 目录）。拍摄界面的日志看 `adb logcat -s capture`，离开时有一行各环节的平均耗时。

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

### 2026-11-23，模拟器：应用内相机

环境：Apple 芯片的 Mac，JDK 17。两台无界面启动的模拟器（`-no-window`，后置相机设为模拟器自带的虚拟场景）：Android 15（API 35，系统 WebView 124，带 `arm64.nosme`）和 Android 12（API 31，系统 WebView 91）。**没有任何真机。**

**构建与自动测试**（仓库里不装数据包）

| 命令 | 结果 |
| --- | --- |
| `python3 android/scripts/check-resources.py` | 通过：269 条文字 × 5 种语言 |
| `./gradlew :app:testDebugUnitTest` | 133 个测试全部通过（其中 48 个是这一轮新增的：拍摄页面的消息与调用、提示文字的选择、岩馆按坐标精度排序、先保存再分析的顺序、垫片写进页面） |
| `./gradlew :app:assembleDebug :app:lintDebug` | 通过；lint 0 error、19 warning（新增的：英语里一句 “carries on on” 被当成重复词；两处「数字 + 名词」；一处建议改用 `StorageManager` 算可用空间；一处 KTX 写法） |
| `./gradlew :app:assembleRelease` | R8 与资源压缩通过，没有缺类报错；不带数据包约 59 MB |
| `aapt2 dump permissions`（debug 与 release） | `POST_NOTIFICATIONS`、三条 `FOREGROUND_SERVICE*`、`CAMERA`、`ACCESS_COARSE_LOCATION`，加 AndroidX Core 的签名级权限；没有 `INTERNET`、`ACCESS_NETWORK_STATE`、`WAKE_LOCK`、`RECORD_AUDIO`、`ACCESS_FINE_LOCATION`。合并后的 Manifest 里 CameraX 带来的只有不导出、默认停用的 `MetadataHolderService` |
| 设备测试，API 35 | 15 个全部通过（`CaptureScreenTest` 6 个，`VideoAnalyzerInstrumentedTest` 9 个） |
| 设备测试，API 31 | 15 个里 14 个通过。没过的是 `theDevicesOwnCameraStartsRecordsAndSaves`：这台模拟器的相机能预览、能出检查用的帧，但 CameraX 的录像一帧都没写进去（编码器以时间戳不在录像区间为由把每一帧都丢了），应用 6 秒后停下并提示。API 35 的模拟器相机没有这个问题；真机上会不会有，不知道 |

`CaptureScreenTest` 从攀爬册的按钮点进去，用替身数据包（两个只会照脚本回话的页面，和 iOS 测试用的是同一份）和一段画出来的墙（文件来源循环播放）走真实的界面：

- 从取景到分析完：拦住的提示（橙色）→ 带怎么挪的提示 → 不认识的代码显示通用的那句 → 300 米内的岩馆只是提议、点了才换 → 机位可以 → 色块（没有颜色名）→ 认出的线路和它的难度、难度被填上并标明 → 点一下标起步手、换工具标结束 → 点在没有岩点的地方 → 「不是这条」换成下一条候选、填上的难度撤掉 → 自己选难度 → 录 → 停 → 记录、拍摄计划、两条请求已经在盘上 → 姿态识别 → 后台的分析页面拿到原样的拍摄计划、结果保存 →「看分析」出现 →「再拍一趟」保留岩馆、难度、颜色。
- 机位被拦住时先问一句再录；画面来源中断时保存已经录到的部分并写明原因；没有分析页面的数据包不留「接着分析」的请求。
- 没有拍摄页面时照常录、照常存；到上限自动停（测试里把上限调成 2 秒）。
- 空间不足不开始；刚开始就停不保存、不留临时文件；录像中空间不够了停下并保存。
- 岩馆表：默认是上一条记录的岩馆；按距离列出知道自己位置的岩馆（门牌级的显示数字，车站级的写「约 1 公里」），区、市级和没有坐标的列在同城里不带距离，署名在；看列表不改变选择。
- 设备自己的相机（模拟器的虚拟场景）：启动、16:9 的直立帧、有视场角、录 2 秒、保存、片子只有画面轨。

**在 API 35 上手动走过的**

- 第一次打开：系统的相机询问 → 拒绝 → 说明卡片（「打开系统设置」「改从相册导入」）。授权后：虚拟场景的预览、录 19.7 秒、保存、卡片。
- Release 包：从攀爬册点「拍一趟」→ 预览 → 开始录 → 按 Home 退到后台 → 回来时已经保存了 7 秒的片子，卡片写着被打断；姿态识别在这段没有人的画面上给出「没有清楚地找到攀爬者」。
- **真实的数据包（1.9.1，装在本机、不进仓库）加一段本机的攀岩视频当文件来源**，Debug 包，做了一趟：拍摄页面在隐藏的 WebView 里就绪并逐帧回话（提示、指示条、色块）→ 用模拟的位置在岩馆表里选了包里的一家岩馆（按清单里的目录 id 对上）→ 选颜色，岩点的描边和画面里的岩点对得上 → 点岩点 → 被拦时的确认 →「仍然开始录」→ 10.5 秒后停 → 记录带着 36.7 KB 的拍摄计划保存 → 姿态识别（104 帧有人）→ 后台的分析页面交回 127 KB 的结果 →「看分析」打开结果。两点要说明：这段视频是手持、镜头一直在动的成片，不是架好的机位，所以分析页面收到了计划、但报告计划和这段录像对不上、按没有计划分析（接口里写明的行为）；拍摄页面在选了岩馆之后有一次超过 5 秒没回话，提示短暂换成「机位检查暂时用不了」，之后自己恢复。这一趟说明的是两边的接口在 Android 的 WebView 里通了，不说明识别效果。这一趟之后界面又有改动（认出线路时的引导和难度填入、车站距离的取整、旧 WebView 的垫片），真实数据包没有重跑。

**每帧的开销**（模拟器、Debug 包，取自离开拍摄界面时的那行日志）

| | 替身页面，540×960 的画 | 真实页面，720×1280 的视频 |
| --- | --- | --- |
| 缩到短边 360、JPEG、base64 | 4–9 ms | 62 ms |
| 姿态识别（每帧都找到人时） | 14–17 ms | 102 ms |
| 交给页面（到调用返回） | 5–7 ms | 135 ms |
| JPEG 大小 | 17 KB | 28 KB |

左列是宿主机空闲时量的；同一个测试在宿主机同时跑着别的模拟器和构建时是 40–70 ms、76–119 ms、35–45 ms。右列只量过一次，就是在这样忙的时候，只能当上限看。真机上的数字没有。

**这一轮测出来并修掉的问题**

- 分析队列和拍摄界面里的分析页面原来用界面的协程作用域，页面的调用可能落到别的线程上；改成各自一个固定在主线程的作用域。
- WebView 91（Android 12）不支持文档开始时注入脚本，页面的第一条消息会丢（三次里丢两次）；垫片改为写进页面。
- 录像一帧都没写进去时原来只会停在 00:00；加了 6 秒的检查和提示。
- lint 的两个 error：一处用了 API 29 才有的 `MediaFormat` 拷贝构造；一处在可组合函数里用 `LocalContext` 取文字。

### 2026-11-23，模拟器：应用内相机，评审之后的修改

一轮独立评审（对 iOS 的实现做的）指出的问题在 Android 上逐条对过并改掉，做法见 [design.md](design.md#拍一趟)：视场角按实际交给页面的那一路画面算；机位提示移到画面上方；保存失败时不能离开界面，没存进去的片子下次启动时补进攀爬册；发出 `lock` 之后不再准备画面和姿态；隐私政策写明拍摄计划里有一张由预览合成的墙面图；取景时画面停了不能开始录；画面停过之后不闪「检查用不了」，页面一直不就绪时不再等；位置的几种状态各说各的；存档里不留界面语言的线路名和难度文字；换岩馆后立刻录不带上一家的计划；默认岩馆按记录建立的时间；没选岩馆时把最近的一家当猜测提出来，或者给一个去问位置的按钮；同一条记录不同时分析两遍；「离你最近」那一条下面也有 OpenStreetMap 的署名。

| 命令 | 结果 |
| --- | --- |
| `python3 android/scripts/check-resources.py` | 通过：275 条文字 × 5 种语言 |
| `./gradlew :app:testDebugUnitTest` | 142 个测试全部通过 |
| `./gradlew :app:assembleDebug :app:lintDebug` | 通过；lint 0 error |
| 设备测试，API 35 | 19 个（`CaptureScreenTest` 10 个，`VideoAnalyzerInstrumentedTest` 9 个）。模拟器刚启动后的第一遍 18 个通过：一个测试等「没有清楚地找到攀爬者」那句提示等了 90 秒没等到；它单独重跑两次、整套再跑两次（第二次是最终的代码）都是全部通过，原因没有查到 |

新增的设备测试（都用替身页面和画出来的墙）：点了岩点紧接着就录，存下来的计划和摘要是页面对 `lock` 的回答、里面有这一下；换了岩馆紧接着就录，这条记录不带计划；发出 `lock` 之后不再交画面；页面一直不就绪时 2 秒（测试里调短）后改口、一帧都没准备；取景时画面来源停了，录像按钮不起作用；保存失败时关不掉界面、片子还在，「再存一次」之后存进去；位置被拒绝、在等、取不到、取到了但附近没有岩馆、从没问过、只有大致位置这六种情况各自的那一句和那一条提议。下次启动时的清理（补进攀爬册 / 删掉）只有用合成文件做的单元测试。

用真实的数据包（1.9.2，装在本机、不进仓库）又做了一趟，这次的文件来源是一段机位固定的本机视频（先取景、选岩馆——没选岩馆时出现的是「可能在：某岩馆」、点「选这家」——选颜色，再录 18 秒）：记录带着 46.5 KB 的计划保存，姿态识别 181 帧有人，后台的分析页面交回结果，并且说它用上了这份计划（`capture.used` 为真；上一轮那段手持视频是按设计被拒的）。应用把页面的这句话原样记在记录里（`capture.planUsed`，没用上时另有 `planUnusedReason`），自己的界面上不就此说任何话；页面说没用上时，应用不会说成用上了。这一趟同样只说明接口通了。

这一轮**没有**做的：Android 12 的模拟器没有重跑；画面停过之后「5 秒从头算」只改了代码、没有测试；用户打开分析页面时队列让路只测了规则，没有在设备上凑出两边同时发生的时刻；系统的位置询问仍然没有点过；视场角的新算法没有在真机上对着画面核对。

### 2026-11-23，模拟器：应用内相机，收尾的一轮

**位置。** 改成 `ACCESS_FINE_LOCATION` 和 `ACCESS_COARSE_LOCATION` 一起声明、一起问。在 API 35 的模拟器上用 `adb emu geo fix` 给了一个紧挨合成目录第一家岩馆的位置，两种授权各跑了一个设备测试，位置是真的从系统的 `LocationManager` 取的：只授予大致位置时，画面上方是「可能在：某岩馆」，岩馆表按远近排、每一行都没有数字；授予精确位置时，是「离你最近：某岩馆 · 200 米」加 OpenStreetMap 的署名，表里门牌级的岩馆带距离、车站级的写「约 1.5 公里」。授权是从外面用 `pm grant` 给的，系统的询问对话框本身仍然没有点过。用真实的目录也看了一次（精确位置，位置设在一家岩馆的车站坐标旁）：表里按距离列出附近的岩馆，那家只知道车站的写的是「约 1 公里」，画面上方把它当猜测提出来。第一次试的时候系统答回来的是上一次的位置（模拟器的卫星定位只在有人问的时候才更新），重发位置之后才对：应用只取一次位置，拿到的可能是系统缓存的那个。

**录像上限。** 改成和 iOS 一样的 20 分钟；相机录下的记录在姿态识别时不再受导入的 3 分钟、500 MB 限制。用文件来源（一段机位固定的本机视频循环播放）和真实数据包 1.9.2 录了 211.9 秒：保存之后姿态识别做完了整段（采样 2118 帧，2094 帧有人），后台的分析页面交回结果并且说用上了计划，停止后 190 秒内全部做完；这一条记录让存档索引长到 5.2 MB。20 分钟的片子没有录过：按这个比例索引会到三十兆字节上下、每次写入都要整个重写，内存和耗时在真机上没有量过，iOS 的存档是同样的结构。

**Android 12 模拟器录不出画面。** 查到了原因，没有修：这台模拟器的相机给每一帧打的时间是从相机启动时从零数起的（日志：`first frame 720x1280: camera clock 0.1 s, uptime 2068.3 s`），而 CameraX 的录像把相机的时间当成系统的开机时间来用（`VideoEncoder: mInputTimebase = UPTIME`，`Start on 00:34:30`），于是每一帧都早于「开始」而被丢掉（`Drop buffer by not in start-stop range`）。和分辨率无关：录像档位已经自己降到了 720p（这台相机只支持到这一档），相机是 LIMITED 级、预览和录像走的是 CameraX 的共用一路。API 35 的模拟器上相机时间和开机时间一致（`camera clock 5873.0 s, uptime 5873.1 s`），录像正常。相机声明自己的时间来源是 UNKNOWN，按 Android 的说法这种相机的时间本来就不保证能和别的时钟比，所以不能断言真机上一定没有同样的相机；CameraX 对已知的这类机型有自己的修正表。应用这边的兜底是 6 秒没写进一帧就停下并提示；`first frame` 这行日志留在了代码里，第一次上真机时看它一眼。

**画面的大小。** 见 [design.md](design.md#拍一趟) 的「布局」。设备测试量出来：1080×2400 上平常状态 81%，认出线路时 73%，并断言不低于 70%；720×1280（`wm size` / `wm density` 调出来的）上没有重叠，画面约为屏宽的一半，受高度限制。

**存档字段。** 分析页面说的「计划用上了没有」改成按 iOS 的名字存：`capture.planUse`，里面是页面给的 `used`、`reason`、`start`、`finish`。上一轮的 `planUsed` / `planUnusedReason` 没有发布过，直接换掉。数据包 1.9.3 的结果里多出的 `shift` 一起存。用 1.9.3 和一段机位固定的本机视频（文件来源）又做了一趟，录 19.3 秒：记录里的 `planUse` 是 `used: true`，`start`、`finish` 都是 `video`，`shift` 是 `dx 0, dy 0, rot 0, scale 1`。

**真实相机这条路。** 之前真实数据包的几趟用的都是文件来源。这一轮在 API 35 上用模拟器相机的虚拟场景做了两趟，数据包是真实的 1.9.2（装在本机）：

- Debug 包，开着调试读数：交给页面的那一帧另存出来看过，是直立的 360×640，和预览里看到的是同一个画面、同一个范围；读数是 `frame 720x1280  fovH 71.5`、`pitch -4.7  roll -0.0  shake 0.000`、`page ready  replies/s 2.09  hand-over 22.8 ms`、`issues no_wall!`（视场角和姿态是模拟器给的值，不说明真机上对不对）。
- Release 包：从攀爬册点进去 → 取景，页面一直在回话（日志里先是 134 帧、`134 replies at 1.99 per second`，再开一次是 46 帧、`1.98 per second`；每帧缩图和编码 7.5–9.4 毫秒、姿态识别 25–34 毫秒、交接 19–38 毫秒）→ 页面说没有攀岩墙（场景是一间屋子），确认后照录 → 21.7 秒后停 → 保存 → 姿态识别在没有人的画面上给出「没有清楚地找到攀爬者」。所以这一趟到姿态识别为止，数据包的分析没有东西可做；带着计划分析完的那一趟仍然是文件来源的。

| 命令 | 结果 |
| --- | --- |
| `python3 android/scripts/check-resources.py` | 通过：276 条文字 × 5 种语言 |
| `./gradlew :app:testDebugUnitTest` | 142 个测试全部通过 |
| `./gradlew :app:assembleDebug :app:lintDebug :app:assembleRelease` | 通过；lint 0 error |
| 设备测试，API 35 | 21 个里 19 个运行并通过，另外两个要真从系统取位置的平时跳过；这两个各自单独跑过（先 `adb emu geo fix`，再 `pm grant` 对应的权限，带 `-e geoFix 1`），都通过 |
| 设备测试，API 31 | 没有重跑整套；只重跑了相机录像那一个（仍然不过，原因见上） |

### 2026-11-23，模拟器：`Recorder` 写不出画面时由应用自己的编码器接手

上一轮查到的事（Android 12 模拟器的相机给帧打的时间从零数起，CameraX 的 `Recorder` 把它当开机时间，每一帧都被丢掉）原来只有「6 秒后停下并提示」这个兜底，那样的相机上这个功能就录不了。现在改成自动换一条路录，做法见 [design.md](design.md#拍一趟) 的「录像」。

| | 之前 | 之后 |
| --- | --- | --- |
| 设备测试，API 31（`Recorder` 写不出画面的那台） | 15 个里 14 个通过，相机录像那一个不过 | 22 个：20 个运行并通过，2 个要真从系统取位置的跳过。两个用设备相机录像的测试都是由应用自己的编码器写的片子（日志：`switching to the app's own recorder: the camera library's recorder wrote nothing in 3 s`） |
| 设备测试，API 35 | 21 个（19 个运行并通过，2 个跳过） | 22 个：20 个运行并通过，2 个跳过。平常那个相机测试仍然是 `Recorder` 写的（`clip written by the camera library's recorder`）；新增的测试先把「这台设备要用自己的编码器」记上再录，由自己的编码器写 |
| `./gradlew :app:testDebugUnitTest` | 142 个 | 147 个全部通过（新增：换路的规则 4 个，按步长抄平面 1 个） |
| `check-resources.py`、`lintDebug` | 通过 | 通过；没有新增文字 |

应用自己的编码器写出来的片子，设备测试里量到的：API 31 上 1280×720、旋转 90°、3.1 秒 85 帧（每秒 27.4 帧）、约 2 MB；API 35 上 3.3 秒 98 帧（每秒 30.1 帧）。

**真实数据包（1.9.3）在 API 31 上用模拟器相机做的一趟**（Debug 包）：取景时页面照常回话（没有攀岩墙）→ 确认后开始录 → 3 秒后自动换成自己的编码器 → 再过约 16 秒停 → 保存 → 姿态识别做完，画面里没有人，给出「没有清楚地找到攀爬者」。把片子取出来用 `ffprobe` 看过：H.264，1280×720，旋转 −90（按它转正后是 720×1280，抽出的一帧确实是直立的），16.5 秒，450 帧（平均每秒 27.2 帧），约 5.6 Mbps，11.5 MB，只有一条画面轨。

**没有验证的：** 任何真机——真机上 `Recorder` 正常的话根本不会走到这条路，走到的那种相机手头没有；自己的编码器在真机上的帧率、发热、长时间录像（这一轮最长 16.5 秒）、各家的 H.264 编码器接不接受这种输入；1280×720 是现在那一路画面的大小，没有为了这条路去要更大的画面；切换前的 3 秒会丢，没有补。

### 第一次上真机时要看的

Debug 包，带上调试读数启动：`adb shell am start -n com.betabuddy.android/.MainActivity --ei initialTab 1 --ez openCapture true --ez debugCaptureOverlay true`，另开一个 `adb logcat -s capture`。

1. `first frame …: camera clock … s, uptime … s`：两个数应该差不多。差得很远，CameraX 的 `Recorder` 会一帧都写不进去，3 秒后应用会自己换成自己的编码器（日志里有 `switching to the app's own recorder`）。每段片子停下时有一行 `recorded … s by the camera library's recorder` 或 `by the app's own recorder`：记下是哪一个。
2. 读数第一行的 `fovH`：竖着拿、主摄，应该在 40–60 之间；`frame` 应该是竖的 16:9（例如 720x1280）。
3. `pitch` / `roll`：手机竖直立着时都接近 0；镜头往上抬，`pitch` 变正；顺时针转（从屏幕这边看），`roll` 变正。`shake` 放稳时接近 0。
4. `replies/s` 在 2 上下；`hand-over` 几十毫秒以内；`pose` 有人入镜时是 `yes`。姿态识别跟不上会自己停掉，日志最后那行会写 `dropped`。
5. 对着墙：`issues` 里的代码和画面上方那句话对得上；色块、岩点描边和预览里的岩点重合（不重合说明给页面的画面和预览不是同一个范围）。
6. 录 30 秒以上再停：顶上的计时在走；停了之后片子能回看、是竖的、没有声音轨；再录一段超过 3 分钟的，看姿态识别做不做得完。
7. 录像中按 Home、来电、锁屏、被别的相机应用抢走：回来时都应该已经保存，卡片上写着被打断。
8. 岩馆：第一次点「按距离找附近的岩馆」，系统的询问里选「大致位置」和「精确位置」各试一次（中间要去系统设置里改）；在馆里看取不取得到位置、要多久。
9. 把手机架好拍完整的一趟，等「看分析」：分析页面里应该写着用上了取景时标的点；架好之后故意挪一下再拍一趟，应该写没用上。

### 在 Apple 芯片的 Mac 上跑模拟器

较新的 Apple 芯片支持 SME 指令集，模拟器里的 Android 15 内核会把它报给应用，但实际执行会触发非法指令：MediaPipe 的原生库在第一次推理时以 `SIGILL` 崩溃（崩在读取 SME 向量长度的指令上）。这是模拟器环境的问题。启动模拟器时让内核屏蔽 SME 即可，上面攀爬册的结果都是这样测的；Android 12 的镜像内核不认识 SME，不需要这个参数。

```sh
emulator -avd <名称> -qemu -append "arm64.nosme"
```

### 尚未验证

- **应用内相机在真机上的一切。** 上面全部是模拟器：真实相机的预览与录像（各厂商的流组合、1080p 在老机型上、3 分钟片子的实际大小、发热）、视场角算得对不对、重力和陀螺仪换出来的仰角、侧倾、晃动在真机上的方向和大小（只有公式的单元测试）、屏幕常亮、来电或别的应用抢相机时的中断（只用文件来源模拟过，和按 Home 退到后台）、真实的存储不足和过热。架好机位拍一趟、再用拍摄计划分析的效果完全没有验证过。
- **位置。** 只在模拟器上从系统取过位置（用 `adb emu geo fix` 设的），两种授权都是从外面 `pm grant` 给的，系统的位置询问对话框没有点过；真机上取不取得到、要多久（尤其在室内、在没有网络定位的机型上）不知道。
- 真实拍摄页面认出线路之后的显示、「不是这条」、难度的填入：只用替身页面走过；用真实数据包的那一趟没有认出线路。
- 拍摄界面只看了简体中文和英语的截图；繁体中文、日语、韩语的文字只经过了资源检查，没有在屏幕上看过。最大字号、TalkBack 没有试。
- Android 12 模拟器上 CameraX 录不出画面：原因查到了（相机的时间从零数起），现在由应用自己的编码器接手，见上面那一节；真机上有没有这样的相机、接手之后的片子在真机上怎么样，没有验证。
- 用户自己打开「分析这一趟」的同时，后台队列正好也在分析同一条记录：现在队列会让路，但只有规则的单元测试，没有在设备上凑出过这个时刻。
- 交给拍摄页面的水平视场角：算法只有单元测试，没有在真机上对着实际画面核对过；拍摄页面靠它对线路。
- 下次启动时把没存进去的片子补进攀爬册：只有用合成文件的单元测试，没有在设备上真的杀掉进程试过。
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

预期：单元测试 147 个、设备测试 22 个通过（其中两个要真从系统取位置的测试平时跳过，跑法见测试里的说明）（Android 12 的模拟器上相机录像那一个不过，见上）；lint 没有 error；权限里没有 `INTERNET`、`ACCESS_NETWORK_STATE`、`WAKE_LOCK`、`RECORD_AUDIO`。设备测试会往应用的外部文件目录写各个状态的截图（`capture-shots/`），用 `adb shell am instrument` 直接跑（不经 Gradle 卸载应用）就能 `adb pull` 出来；截图不进仓库。然后用 Release 包在设备上走一遍：

1. 不装数据包的构建：落在岩馆，显示「这个构建没有内置岩馆数据」；底部齿轮、右上角按钮都能进「设置与关于」，里面的隐私政策和开源许可能打开。
2. 装了数据包的构建：说明页只出现一次 → 场景出现、能转动缩放、选线 → 切到攀爬册再回来不重载 → 在页面菜单里点「语言与设置」打开原生设置 → 换一种语言，返回后页面是新语言并回到刚才的线路 → 开不开飞行模式表现一致。
3. 语言：五种语言加「跟随系统」逐个选；杀掉应用重开仍是所选语言；系统语言设成五种之外的语言时界面是英语。
4. 攀爬册：导入一段竖拍视频和一段横拍视频 → 分析骨架完成 → 回看时一打开就有画面、骨架与人重合、0.1 秒步进、0.5× 慢放 → 写笔记、标记完攀、分享文字 → 删除。Release 包上分析能跑完，说明 R8 没有裁坏 MediaPipe。
5. 分析这一趟（需要清单里带分析页面的数据包）：没识别过姿态的记录点「分析这一趟」→ 先跑姿态识别 → 自动进入全屏分析页面 →「正在准备画面…」之后页面开始计算并显示结果 → 在页面里请求看视频里的某一刻，分析页面关掉、回看停在那一刻 →「完成」后按钮变成「看分析」和「重新分析」→ 杀掉应用重开，「看分析」直接显示保存的结果 →「重新分析」从头再来 → 删除记录后 `filesDir/BetaBuddy/` 里没有这条记录的结果文件。不装数据包的构建里回看页没有这个按钮。
6. 拍一趟：攀爬册 →「拍一趟」→ 第一次出现相机询问（拒绝一次，看说明卡片和「改从相册导入」；再从系统设置里打开）→ 对着墙：提示、色块、岩点描边 → 选颜色、点一下标起步和结束 → 岩馆（默认是上一条记录的；点「按距离找附近的岩馆」看位置询问和列表）→ 录 → 回来停 →「已保存到攀爬册」→ 放着等「看分析」→ 录像中按 Home、来电、锁屏各试一次，回来都应该已经保存。不装数据包的构建：同样能录能存，画面上写着没有机位检查。

岩馆页面的网络审计只能在 Debug 包上做（只有它打开了 WebView 调试）：`adb forward tcp:9222 localabstract:webview_devtools_remote_<应用的 pid>`，再用 Chrome 的 `chrome://inspect` 或 DevTools 协议看 Network。
