# BetaBuddy Android：导航、岩馆容器、攀爬册与语言

Android 版和 iOS 版是同一个产品：同样的两个入口、同样的五种语言、同样的本地与诚实性原则。产品定位见 [../ios/product-design.md](../ios/product-design.md)，岩馆功能和数据包接口见 [../ios/gym-design.md](../ios/gym-design.md)。这里只写 Android 上怎么落地，以及和 iOS 不一样的地方；接口约定不重复抄。

## 导航

单 Activity（`MainActivity`），Compose 界面，`navigation-compose` 管页面栈。

```
home ─┬─ 岩馆（标签 0，启动落点）
      ├─ 攀爬册（标签 1，JournalScreen 内部自己切 列表 → 导入 → 回看）
      └─ 齿轮 ──▶ settings（设置与关于）─┬─ settings/privacy（隐私政策）
                                        └─ settings/licenses（开源许可）
```

- **底部两个入口**在 `home` 这一页里，不是 NavHost 的路由。岩馆始终留在界面树里，攀爬册盖在它上面；这样切到攀爬册再回来，岩馆页面不重载。攀爬册的界面状态用 `SaveableStateProvider` 保存，它的 ViewModel 挂在 `home` 这一层，切标签不会丢。
- **底部导航**和 iOS 一样：两个入口等宽，最右边是一个齿轮，打开「设置与关于」。齿轮在两个入口下都在，包括岩馆的 3D 页面铺满的时候（页面顶部归数据包，原生不在上面叠按钮）。
- **设置与关于**是一页，结构与 iOS 的 `AboutView` 相同：第一项是语言（「跟随系统」加五种语言，各用自己的文字写），然后是本地说明、分析边界、岩馆说明、记录管理、隐私政策、开源许可、帮助与支持、版本。进入它的地方有四处：底部的齿轮；攀爬册首页右上角的圆形按钮（`JournalScreen(onOpenSettings)`）；岩馆说明页和无数据包页右上角的同一个按钮；数据包页面自己发来的 `openSettings` 消息。
- **返回**：系统返回手势先退设置页，再让攀爬册内部回到列表，再从攀爬册回到岩馆，最后离开应用。Manifest 开了 `enableOnBackInvokedCallback`，页面之间有预测性返回动画。
- **边到边**：`enableEdgeToEdge()`，状态栏和导航栏透明、深色图标，底下是纸白。原生页面用 `PaperPage` 避开状态栏和刘海；底部标签栏自己垫上导航栏/手势条的高度。标签栏以上的内容区已经扣掉了标签栏占掉的底部 inset，里面的页面再用 `imePadding()` 时不会多垫一截。
- **方向**：Manifest 写了竖屏。大屏设备上系统会忽略这个限制，布局按可用宽度排，不依赖固定方向。
- **只有浅色**，和 iOS 一致。

## 视觉

`ui/theme/Theme.kt` 的 `BB` 与 iOS `DesignSystem.swift` 同值：纸白 `#F5F2E8`、墨黑 `#1A1F1A`、荧光黄绿 `#D1F24A`、淡紫 `#D1CCF0`、次要文字 `#616659`、分隔线为墨黑 13%。标题用系统字体的最粗字重，圆角 14–26。Material 3 只是底座（颜色表映射到 `BB`），界面用 `ui/common` 里自己的组件：`PaperPage`、`Stamp`、`BigButton`、`SectionLabel`、`Stat`、`ColorCard`、`CircleIconButton`、`BBIcons`（自绘线条图标和品牌标记，不带图标字体）。岩馆、攀爬册和设置用的是同一套；攀爬册独有的几个小件（描边按钮、进度条、播放控制）在 `journal/JournalStyle.kt`，颜色和字体仍取自 `BB`。

英文小标（`GYM / UNOFFICIAL`、`KEEP IT HONEST` 等）是品牌标记，不翻译。

## 岩馆容器

对应 iOS 的 `GymPack.swift` / `GymView.swift`，代码在 `gym/`：

| 文件 | 内容 |
| --- | --- |
| `GymManifest.kt` | 读 `manifest.json`（字段与宽容写法同 iOS） |
| `GymPack.kt` | 数据包状态（无 / 不完整 / 就绪）、路径安全检查、入口 URL、内容类型表 |
| `GymMessage.kt` | 页面 → 应用的消息、应用 → 页面的调用、垫片脚本 |
| `GymSession.kt` | 唯一的 `WebView`：创建、离线拦截、消息桥、生命周期 |
| `GymScreen.kt` | 说明页、空状态、加载中、失败、页面本身 |

前三个文件不依赖 Android 类型，单元测试直接在 JVM 上跑。

### 数据包在哪

`app/src/main/assets/gympack/`，被 Git 忽略，由 `android/scripts/install-gym-pack.sh` 校验后拷入。启动时读清单、确认格式版本、确认清单里每个文件都能打开；不重算摘要（安装脚本算过，APK 之后有签名）。没有数据包不是错误：入口保留，显示「这个构建没有内置岩馆数据」。

### 加载方式

页面的地址是 `https://appassets.androidplatform.net/gympack/<entry>`。这个域名是 `WebViewAssetLoader` 的保留域名，从不解析，请求在应用里就地回答：

- 只回答 `/gympack/` 之下、**清单里列出的**路径（比 iOS 更严：目录里多出来的文件也不提供）。其余本地域名请求回 404。
- 响应头和 iOS 一样带 `Cache-Control: no-store`、`X-Content-Type-Options: nosniff` 和同一条内容安全策略（去掉了 `gympack:` 协议名）。内容类型按同一张扩展名表给出。
- 这是普通的 https 同源环境，`<script src>`、`fetch`、ES module、Blob Worker 都按常规工作。

入口 URL 的查询参数与 iOS 一致：`hint=0`、`lang=<代码>`，Debug 构建的附加查询串排在后面。

### 离线保证

四层，任何一层单独都够用：

1. **没有 `INTERNET` 权限。** Manifest 不声明，并用 `tools:node="remove"` 去掉依赖库合并进来的声明。没有这个权限，系统不给进程任何网络套接字，WebView 也一样。这一层由系统强制，不依赖应用代码。
2. `WebSettings.blockNetworkLoads = true`。
3. `shouldInterceptRequest` 里除了数据包地址和内存里的 `data:` / `blob:` / `about:` 之外，一律回空的 403；`shouldOverrideUrlLoading` 拦掉数据包之外的导航。
4. 响应头里的内容安全策略不允许任何远程源。

另外关闭了文件访问、内容访问、混合内容、定位、弹窗、缩放，关掉了 Safe Browsing 和 WebView 的指标上报（Manifest 里的两条 `meta-data`）。页面存储在每次启动时清空。

姿态识别用的 MediaPipe Tasks 依赖 Google 的 datatransport 库（它自己的使用统计上报）。除了没有联网权限之外，Manifest 还去掉了这个库用来发现上报后端的服务：没有后端，每条事件在入库和排程之前就被丢弃。

### 消息桥

页面是给 iOS 写的，只会调用 `window.webkit.messageHandlers.gym.postMessage(对象)`。Android 上：

- `addJavascriptInterface` 注入一个对象 `BetaBuddyGymNative`，只有一个方法 `postMessage(字符串)`。
- 一小段垫片脚本在文档开始时运行（`WebViewCompat.addDocumentStartJavaScript`，只对数据包的源生效），把 `window.webkit.messageHandlers.gym.postMessage(obj)` 接到 `BetaBuddyGymNative.postMessage(JSON.stringify(obj))`。WebView 太旧不支持文档开始脚本时，改在 `onPageStarted` / `onPageFinished` 注入；页面每次发消息时才去找这个对象，所以晚一点注入也不丢消息。
- 垫片只在主框架里生效。消息在主线程解析，类型和处理与 iOS 相同：`ready` 收起加载状态；`hintSeen` 记在本机，之后的入口 URL 带 `hint=0`；`routeSelected` 只留 id 和名称，只在内存里；`openSettings` 打开原生的「设置与关于」（只在岩馆页面在前台时响应）；`error` 在 `ready` 之前才盖失败页，之后的只写日志。不认识的类型和多出来的字段忽略。

应用 → 页面仍是 `window.BetaBuddyGym.openWall(id)` / `openRoute(id)`，用 `evaluateJavascript` 调用，参数经 JSON 字符串编码。

### 生命周期

- `WebView` 归 `GymSession` 所有，`GymSession` 放在 Activity 级的 `GymViewModel` 里。切标签、进设置、屏幕配置变化（包括切换语言引起的 Activity 重建）都不销毁它；`WebView` 用 `MutableContextWrapper` 创建，重建后换到新的 Activity 上，不持有旧的。
- 岩馆不在前台（切到攀爬册、应用退到后台）时调用 `WebView.onPause()` 并把视图设为不可见，回来时恢复。
- **切换语言**：页面的语言由 URL 决定，所以语言变了就带新的 `lang` 重新加载，并在页面 `ready` 后对之前选中的线路调用 `openRoute`。
- **渲染进程被回收**（`onRenderProcessGone`）：丢掉这个 `WebView`，显示「岩馆页面被系统回收了」和「重新加载」，应用不崩溃；重新加载时新建一个。
- 页面首次加载完成前盖着纸白底的「正在打开岩馆…」，避免白屏闪烁。

### 安全区

iOS 上页面延伸到状态栏下面，用 `env(safe-area-inset-*)` 自己留白。Android WebView 对安全区的支持随版本不一，所以 Android 上反过来：原生侧把 `WebView` 放在状态栏和刘海**之内**（上方和两侧按系统 inset 留出，露出的是和页面顶部同色的纸白），并且不把 inset 传给 `WebView`，页面拿到的 `env(safe-area-inset-*)` 恒为 0。底部是原生标签栏，标签栏自己垫导航栏的高度，页面的底部抽屉不会被遮住。

### 调试启动参数

只在 Debug 构建里生效，用 `adb shell am start` 的 extra 传入：

| extra | 类型 | 作用 |
| --- | --- | --- |
| `initialTab` | `--ei` 0 / 1 | 启动落在岩馆 / 攀爬册 |
| `gymNoticeSeen` | `--ez true` | 跳过进入前的说明页（会记住） |
| `gymQuery` | `--es` | 原样接到入口 URL 的查询串后面 |
| `gymOpenWall` / `gymOpenRoute` | `--es` | 页面 `ready` 后调用对应接口 |

Debug 构建还可以用 Chrome 的 `chrome://inspect` 连上页面。加载耗时和页面报告的错误写入 logcat（标签 `gym`），不离开设备。

## 语言

- 资源目录：`values/` 是英语（所以不支持的系统语言读到英语），`values-b+zh+Hans`、`values-b+zh+Hant`、`values-ja`、`values-ko`。壳、岩馆、设置的文字在 `strings.xml`，攀爬册的在 `strings_journal.xml`。译文取自 iOS 的 `Localizable.xcstrings`，提到平台的地方（iPhone、相册应用）改成对 Android 成立的说法。
- `res/xml/locales_config.xml` 列出五种语言并在 Manifest 里登记，Android 13+ 的系统设置里会出现「应用语言」。
- **应用内语言设置**（「设置与关于」的第一项）：「跟随系统」加五种语言，各用自己的语言写。用 AppCompat 的按应用语言接口 `AppCompatDelegate.setApplicationLocales`：Android 13+ 和系统设置是同一个值；更低的版本由 AppCompat 自己保存（Manifest 里 `AppLocalesMetadataHolderService` 的 `autoStoreLocales`）。「跟随系统」就是清空应用语言。选了之后 Activity 立即以新语言重建，仍停在「设置与关于」；回到岩馆时页面带着新的 `lang` 重新加载并回到原来的线路。
- `AppLanguage.current(context)` 返回 `zh-Hans` / `zh-Hant` / `en` / `ja` / `ko` 之一。做法是每个 `strings.xml` 里放一条自己的语言代码（`app_language_code`），读到的就是界面实际在用的那套资源，回落到英语时也对。它用于入口 URL 的 `lang=` 和清单里的 `gymNameLocalized`。
- **隐私政策**随应用内置，离线显示：`res/raw*/privacy_policy.txt` 有简体中文、繁体中文、英语三份，日语和韩语界面读英语。文字在 iOS 版的基础上改成了对 Android 成立的说法（没有申请联网权限、系统照片选择器、卸载即删除）。

## 攀爬册

对应 iOS 的 `JournalView` / `ImportView` / `ReviewView` 和 `Core/` 里的存储与分析，功能与文字一致。入口只有 `JournalScreen(onOpenSettings)`，列表 → 导入 → 回看在它内部切换，不占用 NavHost 的路由；`JournalViewModel` 挂在 `home` 这一层。

| 目录 | 内容 |
| --- | --- |
| `data/` | 记录的数据结构（kotlinx-serialization）和 `ClimbStore`：`filesDir/BetaBuddy/` 下的 `archive.json`、原片、缩略图，原子写入；索引读不出来时原样保留、不覆盖 |
| `pose/` | `VideoAnalyzer`（解码与推理）、`PoseMath`（二维几何与候选片段）、采样与导入限制 |
| `journal/` | 列表、筛选与搜索、导入流程、文字与格式化 |
| `review/` | 回看：播放、骨架与髋部短轨迹的叠加、候选片段、笔记、分享 |

- **选视频**用系统照片选择器（`PickVisualMedia`，只选视频），不需要任何存储权限；选中的那一段被复制到应用私有目录的 `staging/`，超过 500 MB 在复制途中就停下。
- **分析**：`MediaExtractor` + `MediaCodec` 顺序解码，不采样的帧在颜色转换之前丢弃；采样帧按旋转信息摆正、缩放后交给 MediaPipe Pose Landmarker Full（视频模式，CPU，33 点），与 iOS 是同一个模型文件，加载前核对 SHA-256。同一时间只跑一个分析，在专用线程上；取消和设备过热在帧与帧之间检查。分析期间保持亮屏，应用退到后台就停止，不留半份结果。
- **回看**：Media3 ExoPlayer 画到 `TextureView`，骨架用 Compose `Canvas` 画在与画面完全重合的矩形上；0.1 秒步进按精确定位，0.5× 慢放。播放器等 `TextureView` 有了 surface 才开始解码，这样暂停状态下打开也能看到第一帧。
- **诚实性**与 iOS 同一套规则：髋部轨迹不叫重心，角度是二维投影角，高抬脚只算候选片段，骨架不跨检测空白延续；`JournalStringsTest` 检查文字里没有这些词。
- 存档里不写界面语言：候选片段按类型保存，显示时取当前语言的文字。

### 文件夹与组合筛选

与 iOS 的[收纳交互](../ios/product-design.md#收好每一趟)一致：文件夹可以新建、改名、删除，删除后保留视频并移回未分类；每条记录最多属于一个文件夹。导入和回看时都能调整岩馆、攀爬日期与归属文件夹。

首页使用横向文件夹入口和紧凑筛选入口，岩馆、难度、时间与原有完攀状态、搜索共同生效。日期以用户本地日历计算，旧记录继续用原日期。全国岩馆目录从 assets 中离线读取，可按城市和馆名搜索，未收录时手填；不会增加网络或定位权限。目录来源与覆盖见[目录说明](../gym-directory.md)。

存档新增字段兼容旧版索引，缺少目录关联或文件夹的记录仍能读取；分类操作只更新 JSON，不搬动原片、不重新分析。

## 隐私与存储

- **不向用户申请任何权限。** Manifest 自己不声明，并移除依赖库会合并进来的 `INTERNET`、`ACCESS_NETWORK_STATE` 和 `WAKE_LOCK`。合并后只剩 AndroidX Core 自动加的一条应用自用的签名级权限。AppCompat 带来的 EmojiCompat 启动项也已移除（它会向系统字体服务要新的表情字体）。
- 岩馆只在本机记两个布尔值（是否读过说明、手势提示是否看过），存在应用私有的 `SharedPreferences` 里。攀爬册的数据全部在 `filesDir/BetaBuddy/`，不写外部存储。
- **不进备份，也不随换机迁移。** `allowBackup="false"`、`fullBackupContent="false"`；`data_extraction_rules.xml` 把 `<cloud-backup>` 和 `<device-transfer>` 两节的全部数据域都排除（Android 12 起两者分开，`allowBackup` 管不到设备间直传）。
- **第三方许可全文**随应用打包在 `assets/licenses/`（组件清单、Apache License 2.0、MediaPipe 原生库的全部声明），在「设置与关于」→「开源许可」里离线查看；许可保持英文原文。
- 「设置与关于」里的「帮助与支持」和邮箱会交给系统的浏览器 / 邮件应用打开；分享复盘文字走系统分享面板，只有文字。应用自己不联网。

## 与 iOS 的差异

| 方面 | iOS | Android |
| --- | --- | --- |
| 页面地址 | 自定义协议 `gympack://local/` | 保留域名 `https://appassets.androidplatform.net/gympack/` |
| 提供的文件 | 包目录里存在的文件 | 只有清单列出的文件 |
| 断网手段 | 内容规则 + 导航限制 | 无联网权限 + `blockNetworkLoads` + 请求拦截 + 导航限制 |
| SDK 的使用统计 | 上报工厂换成空实现 | 无联网权限，并移除上报后端的发现服务 |
| 消息桥 | `WKScriptMessageHandler` | `addJavascriptInterface` + 文档开始时注入的垫片 |
| 安全区 | 页面用 `env()` 自己留白 | 原生侧留白，页面拿到的 inset 为 0 |
| 语言 | 跟随系统，或在应用内选，不重启 | 同左，Activity 以新语言重建；Android 13+ 与系统的「应用语言」同步 |
| 设置与关于 | 从底部弹出的一页；显示已保存的攀爬数量、在线隐私政策链接 | 导航栈里的一页，系统返回退出；不显示数量，隐私政策只有内置的一份；多一项「开源许可」 |
| 备份 | 排除云备份 | 排除云备份和设备间迁移 |
| 视频来源 | 相册或文件 | 系统照片选择器 |
| 调试参数 | 启动参数 `-gymQuery` 等 | intent extra，另有 `gymNoticeSeen` |
