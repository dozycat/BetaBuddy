# Android 第三方内容

Android 版用到的第三方代码和文件。版本固定在 `gradle/libs.versions.toml`；新增依赖时在这里补一行，并确认它不联网。

## 随应用打包的文件

- **Pose Landmarker Full float16，版本 1** — Google MediaPipe 模型包，`app/src/main/assets/pose_landmarker_full.task`，与 `ios/BetaBuddy/Resources/` 里的是同一个文件：https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task 。SHA-256 `5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1`（`PoseModelTest` 校验）。权重未改动。

## 只用于测试、不进安装包

- `app/src/androidTest/assets/pose.jpg` — Google MediaPipe 的示例图，只用于设备测试。

## 依赖库

| 库 | 用途 | 许可 |
| --- | --- | --- |
| AndroidX：core-ktx、appcompat、activity-compose、lifecycle、navigation-compose、webkit | 应用骨架、应用内语言、岩馆的 `WebViewAssetLoader` | Apache 2.0 |
| Jetpack Compose（BOM）：ui、foundation、material3 | 界面 | Apache 2.0 |
| AndroidX Media3 ExoPlayer | 攀爬册回看时播放本地视频 | Apache 2.0 |
| MediaPipe Tasks Vision | 设备上的姿态识别 | Apache 2.0 |
| kotlinx-serialization-json、kotlinx-coroutines | 本地存档、清单解析、后台任务 | Apache 2.0 |
| 测试：JUnit 4、AndroidX Test（runner、ext-junit） | 只在测试里 | EPL 1.0 / Apache 2.0 |

MediaPipe Tasks 依赖 Google 的 datatransport 库（SDK 自己的使用统计上报）、firebase-encoders、Flogger、Guava 和 protobuf-javalite。datatransport 的 Manifest 会带进来 `INTERNET` 和 `ACCESS_NETWORK_STATE` 两个权限和三个组件；应用的 Manifest 把它们全部移除。没有联网权限它发不出去；没有后端发现服务它连一条事件都不会存下来。

Media3 的 Manifest 会带进来 `ACCESS_NETWORK_STATE` 和 `WAKE_LOCK`（码率估计和后台播放用），回看本地文件用不到，也已移除。AppCompat 带来的 EmojiCompat 启动时会向系统字体服务要新的表情字体，应用不用它，初始化项已移除。合并后剩下的唯一一条权限是 AndroidX Core 自动加的 `<应用 ID>.DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION`：签名级，只有应用自己能持有，不涉及用户数据。

MediaPipe 的 AAR 不带 R8 规则，`app/proguard-rules.pro` 整体保留 `com.google.mediapipe.**` 和 `com.google.protobuf.**`（它们经 JNI 和反射访问）。

没有分析、崩溃上报、广告或远程配置 SDK。Gradle wrapper（`gradle/wrapper/gradle-wrapper.jar`）是 Gradle 9.7.1 官方生成的，SHA-256 `7a9ce74cff467ca1bf60a4fcd9f05185acceda4d0f382434d393e17864262c5d`，与 gradle.org 公布的一致。
