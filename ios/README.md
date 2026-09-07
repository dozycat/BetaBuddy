# BetaBuddy for iPhone

原生 SwiftUI 攀岩视频回看与训练手账。iOS 17+，iPhone 优先。

## 构建

1. 安装 Xcode（含 iOS SDK）和 [XcodeGen](https://github.com/yonaskolb/XcodeGen)。
2. 在仓库根目录运行 `./ios/scripts/bootstrap.sh`。脚本只在**开发机**下载固定版本官方 MediaPipe SDK；模型已随源码提供，并校验 SHA-256。
3. 打开 `ios/BetaBuddy.xcodeproj`，选择 BetaBuddy scheme。
4. 真机运行：在 Signing & Capabilities 选择自己的 Development Team，选择连接的 iPhone，Build & Run。
5. 模拟器：需要先安装 iOS runtime。用于界面与 CPU 管线验证，不能代替真机速度、温度和耗电测量。

没有 CocoaPods、Swift 包运行时依赖或后台服务。SDK 指标上报工厂被替换为上游空实现；详见 `LocalMediaPipe/README.md`。SDK 按官方 podspec 的静态 XCFramework 与 graph archive 链接方式集成；下载的 Vendor 目录不进 Git。模型资源在应用的 Copy Bundle Resources 中。

```sh
swift test --package-path ios
xcodebuild -project ios/BetaBuddy.xcodeproj -scheme BetaBuddy \
  -destination 'generic/platform=iOS' -derivedDataPath ios/.derivedData \
  CODE_SIGNING_ALLOWED=NO build
# 将设备名换成已安装的模拟器
xcodebuild -project ios/BetaBuddy.xcodeproj -scheme BetaBuddy \
  -destination 'platform=iOS Simulator,name=BetaBuddy iPhone' test
```

## 模型与手机调优

保留后端原有 **pose_landmarker_full / float16**，33 点；固定为官方版本 `1`，未替换成 Lite，也没有切换为 Apple Vision。权重 SHA-256：

`5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1`

- 单实例串行 `.video` 模式，使用模型自带时间追踪。
- AVAssetReader 顺序解码，跳帧发生在图像变换和推理之前；不将全片转换成 UIImage 数组。
- 原片的方向/镜像变换先烘焙进输入；宽高比与播放画面一致。
- 10fps/720px 或 15fps/960px 最大采样规格；原片保留，不改模型输入张量与权重。
- 推理在独立 actor 上，UI 仅接收低频进度。每帧 autoreleasepool，Core Image 不缓存中间图。
- CPU 为当前可复现基线，避免 GPU 线程亲和与模拟器支持差异。**未宣称这是最快的 delegate**；GPU 对照和实际能耗需要真机基准后再决定。
- critical 热状态终止；用户取消在帧间检查，退出前台停止。第一版不后台长时间占用计算。
- 低置信度/出界点不进入角度和骨架；不跨缺失片段计算位移或绘制轨迹。

这属于工程管线调优，**不是针对攀岩数据重新训练**，也不代表已经证明准确率或性能提升。

## 文件

- `BetaBuddy/Core`：模型结果结构、几何、MediaPipe 推理、原子本地存储。
- `BetaBuddy/Views`：设计系统与三页导航、导入、视频骨架回看。
- `CoreTests`：可在 macOS 直接运行的几何与时序边界测试。
- `BetaBuddyTests`：iOS 本地存储和打包测试。
- `BetaBuddyUITests`：导入与导航测试。
- `../docs/ios/product-design.md`：产品、视觉、交互与状态设计。
- `../docs/ios/validation.md`：验证结果和真机验收计划。

照片选择器可能由系统下载 iCloud 原片；应用自己不下载模型、不上传视频、不调用旧 Python/LLM 接口。
