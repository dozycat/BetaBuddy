# 攀爬册 → 壳：需要你处理的事

攀爬册负责人写给壳与岩馆同事的清单。处理完的条目可以直接删掉。

## 现在就需要

1. **`android/THIRD_PARTY.md`（或你定的第三方清单）补两条**，内容与 `ios/THIRD_PARTY.md` 一致：
   - `app/src/main/assets/pose_landmarker_full.task`：MediaPipe Pose Landmarker Full / float16，官方版本 1，SHA-256 `5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1`（从 `ios/BetaBuddy/Resources/` 原样拷贝，`PoseModelTest` 校验）。
   - `app/src/androidTest/assets/pose.jpg`：Google MediaPipe 示例图，只用于设备测试，不进安装包。
2. **Release 构建的 R8 规则**（`proguard-rules.pro`）：`isMinifyEnabled = true` 时请确认 MediaPipe Tasks 和它依赖的 protobuf 没被裁掉。它们经 JNI 和反射访问，官方 AAR 的 consumer rules 不一定够；跑一次 release 构建的分析，如果崩在 `com.google.mediapipe.*` / `com.google.protobuf.*` 找不到类或字段，就加 `-keep class com.google.mediapipe.** { *; }` 和 `-keep class com.google.protobuf.** { *; }`。我没法在这台机器上验证。
3. **`data_extraction_rules.xml` 加 `<device-transfer>`**：现在只排除了 `<cloud-backup>`。Android 12+ 的设备间迁移走另一节，不写的话原片和记录可能被迁到新手机。要不要排除请你定（iOS 那边只承诺「排除云备份」）；如果排除，同样的五个 `<exclude>` 再写一遍。

## 可以删掉的依赖

- `media3-ui`：回看用 `TextureView` + `ExoPlayer`，没用 `PlayerView`（它拿不到与骨架画布严格重合的画面矩形，也没法在代码里选 TextureView）。
- `coil-compose`、`coil-video`：缩略图是本地 JPEG，直接 `BitmapFactory` 解码。你那边不用的话可以去掉。

## 我用到、请不要改名的东西

- `ui/theme`：`BB.Paper/Ink/Lime/Lavender/Muted/Line`、`BB.display()`、`BB.mono()`、`BB.Title/Title3/Headline/Body/Subheadline/Caption`。
- `ui/common`：`PaperPage`、`Stamp`、`BigButton`、`SectionLabel`、`Stat`、`CircleIconButton`、`ColorCard`、`BBIcons.BrandMark/Sliders/Plus/Check/Alert/Lock/ArrowUpRight/ArrowBack`。
- 依赖：`lifecycle-runtime-compose`（`LifecycleEventEffect`）、`lifecycle-viewmodel-compose`（`viewModel()`）、`activity-compose`（Photo Picker、`BackHandler`）、`media3-exoplayer`、`mediapipe tasks-vision`、`kotlinx-serialization-json` 和它的编译器插件、`core-ktx`。
- Manifest：不需要任何权限。Photo Picker 不要求 `READ_MEDIA_VIDEO`；请继续保持没有 `INTERNET`。

## 约定

- 入口只有 `JournalScreen(onOpenSettings)`。列表 → 导入 → 回看都在它内部切换（`rememberSaveable` 的路由 + `BackHandler`），不占用你的 NavHost 路由。`JournalViewModel` 用 `viewModel()` 取，作用域跟着你放 `JournalScreen` 的地方走。
- 页面用你的 `PaperPage`（顶部和两侧 inset 由它处理，底部留给标签栏）。导入和回看时标签栏仍然可见；分析中切到岩馆不会停止分析，退到后台才会停。
- 存储位置 `filesDir/BetaBuddy/`（`archive.json`、原片、缩略图、`staging/`）。不写外部存储。
- 我的字符串全在 `res/values*/strings_journal.xml`，名字都以 `journal_` 开头，不会和 `strings.xml` 撞名。
- 单元测试在 `app/src/test/java/com/betabuddy/android/{data,pose,journal,review}`，读 `src/main/res` 和 `src/main/assets` 用的是相对模块目录的路径（Gradle 默认工作目录）。
