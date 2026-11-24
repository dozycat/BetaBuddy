# iOS 1.7.0 (19)

状态：2026-11-24 已生成并校验包含相机超时修复的签名归档，App Store IPA 导出等待 Xcode 开发者账户登录；尚未上传或提交审核。以 `feature/camera-capture` 的 `8109608` 为基础，发布分支为 `codex/release-1.7.0`。内置 GymPack 1.9.3，安装时核对清单的 25 个文件。

## 此版本的新增内容（商店语言使用英文）

- Record a climb directly in BetaBuddy. Stop recording to save it to Logbook, then review the on-device analysis.
- Check framing before you start: choose a hold colour and mark the start and finish on the preview. Recognition can be wrong; you can correct it.
- Optionally find nearby gyms using your location, or choose one manually. Location stays on your device and is not stored.
- Camera recordings are video-only, with no sound. Your recordings and analysis stay on your device.

## 宣传文本

Record your next climb in BetaBuddy. Check the framing, mark your holds, then stop to save and review. Video and analysis stay on your iPhone.

## 商店描述增补

RECORD A CLIMB
Open Record in Logbook to use the back camera. Choose a gym and hold colour, check the framing hints, and mark the start and finish on the preview. Stop recording to save the video before pose recognition and attempt analysis run on the device. Recordings have no sound. Camera guidance and recognised holds can be wrong and are not safety or coaching advice.

FIND A GYM
Choose a gym from the bundled offline directory. You can optionally allow location access to order nearby gyms by distance. Location is used on your device only and is not stored. You can still record and choose gyms without location access.

Keep the existing lifetime membership explanation: additional offline gyms and cat climbing remain part of the existing one-time purchase; existing members do not need to purchase again.

## 审核备注

The Logbook's Record button opens the in-app camera (back camera, video only, no microphone or audio track). Allow camera access, choose a gym and a hold colour, and optionally mark start and finish holds on the preview. Framing hints and hold recognition are model estimates and can be corrected. Recording is still available if the framing check is unavailable. Tap Stop to save the clip to Logbook before on-device pose recognition and attempt analysis begin.

While framing, scaled-down preview frames and phone tilt/movement are checked on the device by a page bundled with the app. That page cannot reach the network. The recording and the capture plan, including one still picture of the wall made from the preview, are stored inside the app and deleted with the record. Location is optional, requested after tapping Find gyms near me, and used for a one-time fix to order the offline gym directory by distance. Location is not stored or sent to a server. No account is required.

The existing non-consumable lifetime membership remains unchanged: it unlocks additional offline gyms, cat climbing and the member badge. Purchase and Restore Purchases remain available in Settings & About. Existing verified members keep access.

## 发布核对

- [x] 版本设为 1.7.0 (19)，重新生成 Xcode 工程。
- [x] 安装并校验 GymPack 1.9.3 的 25 个文件。
- [x] Mac 核心测试 21/21 通过。
- [x] 模拟器单元 / 集成复跑：86 项，84 通过、2 项按设计跳过、0 失败；包含此前失败的页面加载超时用例和正常恢复用例。
- [x] 首轮 UI 共 19 项，16 通过、3 项按设计跳过、0 失败；包含猫、会员、单一导入保存和相机。修复后相机中英文流程追加复跑 2/2 通过。
- [x] 签名 Release 归档成功，位于本机忽略目录 `ios/build/1.7.0/BetaBuddy-1.7.0-19.xcarchive`。
- [ ] App Store IPA 导出：命令行和 Organizer 都报 `No Accounts` / 缺少分发证书；Xcode Apple Accounts 页面为空，等待用户登录。
- [x] 归档版本为 1.7.0 (19)；25 个数据文件加清单逐字节匹配，签名校验通过；相机/位置用途说明存在，无麦克风用途说明；隐私清单包含磁盘容量 API 理由，无 StoreKit 测试配置；Release 可执行文件不含相机调试启动参数或诊断文件名。
- [ ] 真机相机预览、录像、保存与播放；拒绝相机/位置、大致位置；后台中断与温度表现。
- [x] 支持站点相机与位置说明已更新源文件并重新生成静态文件。
- [ ] 将支持站点部署到现有线上站点。
- [ ] 上传 App Store Connect、关联版本及提交审核。

本轮修复取景页面加载失败后，已在准备中的帧仍可能进入页面并把状态误恢复为就绪的问题：未完成就绪握手时不再交帧，加载失败会清除当前检查会话。既有 `testAPageThatNeverComesUpIsGivenUpOn` 在首次全套运行中出现 3 个断言失败（属于同一个测试），修复后该用例及全部单元 / 集成通过；不把首次运行记作全通过。模拟器用视频或合成画面替代相机的验证不代表真机相机通过。真实攀爬片段分析与线路识别测试需要指定片段和岩馆；没有输入时按设计跳过，不能计作通过。旧版本的偶发镜头失控未在本轮定位，不列为已修复。

## 本机产物

都在被 Git 忽略的 `ios/build/1.7.0/`：

- `BetaBuddy-1.7.0-19.xcarchive`：包含相机超时修复的最终归档；Organizer 的同名副本已同步。
- `archive-verification.json`：归档资源、权限、签名和可执行文件摘要校验结果。
- `FixedUnits.xcresult`、`FixedCaptureUI.xcresult`：修复后的测试结果。
- `ReleaseQA.xcresult`：修复前的首轮完整结果，UI 全部通过，但一个单元测试有 3 个断言失败，因此整体结果为失败。
- `qa-attachments/`：首轮测试截图（相机画面为合成测试素材，不作为真实识别效果的商店宣传图）。
- `before-camera-timeout-fix/`：仅用于追溯的旧归档，不用于发布。
- `logs/`：核心测试、首轮测试、复跑、归档及导出错误日志。

本轮未 push、未上传 App Store Connect、未提交审核；发布分支留在本机。未在真实 iPhone 上打开相机、录像或测试权限。
