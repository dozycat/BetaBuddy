# BetaBuddy · Android 工程

Kotlin + Jetpack Compose 原生应用，Android 8.0（API 26）及以上，竖屏，只有浅色。和 iOS 版是同一个产品：两个入口（岩馆、攀爬册）加「设置与关于」，五种语言并可在应用内切换，全部离线；后台分析使用本地媒体处理服务，进度通知权限可选；应用内相机用到相机权限，按距离给岩馆排序用到大致位置权限，两项都是用到时才问。产品介绍见[仓库首页](../../README.md)，Android 上的设计见 [design.md](design.md)，岩馆的数据包接口见 [../ios/gym-design.md](../ios/gym-design.md)。

应用 ID `com.betabuddy.android`，版本 `1.2.0`（`android/app/build.gradle.kts` 里的 `versionName` / `versionCode`）。

## 构建

### 2026-11-23 · 应用内相机「拍一趟」（未发布）

攀爬册的主按钮改成「拍一趟」，导入退为次要按钮。拍摄界面取景时由数据包的拍摄页面检查机位、认颜色和岩点，用户选颜色、点一下标起步和结束、选难度和岩馆；停止后先保存，再做姿态识别和「分析这一趟」。相机用 CameraX 1.7.0（预览、不带声音的 1080p 录像、给检查用的小帧）；位置用平台自带的 `LocationManager`，只申请大致位置。Manifest 多了 `CAMERA` 和 `ACCESS_COARSE_LOCATION`，没有 `RECORD_AUDIO`。存档的记录多一个可选的 `capture` 字段（与 iOS 同名同义），拍摄计划存成 `<记录 id>.capture.json`；旧存档照常读取。设计见 [design.md](design.md#拍一趟)，验证见下面的[验证状态](#2026-11-23模拟器应用内相机)。

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
- **不加联网依赖。** 权限只有本地分析服务、可选进度通知、相机和大致位置（`check-resources.py` 按这个名单检查，并要求移除 `INTERNET`、`ACCESS_NETWORK_STATE`、`WAKE_LOCK`、`RECORD_AUDIO`）；新增依赖先看它的 Manifest 会合并进来什么权限和组件，该移除的在应用的 Manifest 里移除，并在 `THIRD_PARTY.md` 补一行；依赖有增减时同步 `assets/licenses/components.txt` 里的清单。
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

`debugCaptureVideo` 用一个视频文件循环播放代替相机（绝对路径，或者应用 `files/` 目录下的文件名；用 `adb push` 加 `run-as <应用 ID> cp` 放进去）；`debugCaptureFov` 是替它报的水平视场角；`debugCaptureAttitude` 是没有运动传感器时替它报的仰角、侧倾和晃动；`debugLocation` 给一个位置，不出现系统的询问。拍摄界面的日志看 `adb logcat -s capture`，离开时有一行各环节的平均耗时。

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

@@CAPTURE_VALIDATION@@

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

预期：单元测试 85 个、设备测试 9 个通过；lint 没有 error；权限里没有 `INTERNET`、`ACCESS_NETWORK_STATE`、`WAKE_LOCK`。然后用 Release 包在设备上走一遍：

1. 不装数据包的构建：落在岩馆，显示「这个构建没有内置岩馆数据」；底部齿轮、右上角按钮都能进「设置与关于」，里面的隐私政策和开源许可能打开。
2. 装了数据包的构建：说明页只出现一次 → 场景出现、能转动缩放、选线 → 切到攀爬册再回来不重载 → 在页面菜单里点「语言与设置」打开原生设置 → 换一种语言，返回后页面是新语言并回到刚才的线路 → 开不开飞行模式表现一致。
3. 语言：五种语言加「跟随系统」逐个选；杀掉应用重开仍是所选语言；系统语言设成五种之外的语言时界面是英语。
4. 攀爬册：导入一段竖拍视频和一段横拍视频 → 分析骨架完成 → 回看时一打开就有画面、骨架与人重合、0.1 秒步进、0.5× 慢放 → 写笔记、标记完攀、分享文字 → 删除。Release 包上分析能跑完，说明 R8 没有裁坏 MediaPipe。
5. 分析这一趟（需要清单里带分析页面的数据包）：没识别过姿态的记录点「分析这一趟」→ 先跑姿态识别 → 自动进入全屏分析页面 →「正在准备画面…」之后页面开始计算并显示结果 → 在页面里请求看视频里的某一刻，分析页面关掉、回看停在那一刻 →「完成」后按钮变成「看分析」和「重新分析」→ 杀掉应用重开，「看分析」直接显示保存的结果 →「重新分析」从头再来 → 删除记录后 `filesDir/BetaBuddy/` 里没有这条记录的结果文件。不装数据包的构建里回看页没有这个按钮。

岩馆页面的网络审计只能在 Debug 包上做（只有它打开了 WebView 调试）：`adb forward tcp:9222 localabstract:webview_devtools_remote_<应用的 pid>`，再用 Chrome 的 `chrome://inspect` 或 DevTools 协议看 Network。
