# iOS 1.7.0 (19)

状态：2026-11-24 11:59（中国时间）已提交 App Store 正式审核，Apple 页面确认 1.7.0 (19)「等待审核」。审核通过后自动发布、立即向全部用户提供更新、保留现有评分。以 `feature/camera-capture` 的 `8109608` 为基础，发布分支为 `codex/release-1.7.0`。内置 GymPack 1.9.3，安装时核对清单的 25 个文件。

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
- [x] App Store IPA 已导出，版本 1.7.0 (19)、App Store 分发描述文件、`get-task-allow=false` 及代码签名校验通过。前一次导出因 Xcode 未登录失败；用户登录后重试成功。
- [x] 归档版本为 1.7.0 (19)；25 个数据文件加清单逐字节匹配，签名校验通过；相机/位置用途说明存在，无麦克风用途说明；隐私清单包含磁盘容量 API 理由，无 StoreKit 测试配置；Release 可执行文件不含相机调试启动参数或诊断文件名。
- [x] iPhone 17 Pro Max 真相机录制、停止、保存；文件完整解码；更换 Release 构建并重启后记录仍在。
- [ ] 真正攀爬场景的分析效果、独立核对相机预览层、拒绝相机/位置、大致位置、后台中断、长时间录制与温度表现。
- [x] 支持站点相机与位置说明已更新源文件并重新生成静态文件。
- [x] 将支持站点部署到现有线上站点，正式首页与隐私页浏览器检查通过。
- [x] 上传 App Store Connect、关联 1.7.0 (19) 及提交正式审核。

本轮修复取景页面加载失败后，已在准备中的帧仍可能进入页面并把状态误恢复为就绪的问题：未完成就绪握手时不再交帧，加载失败会清除当前检查会话。既有 `testAPageThatNeverComesUpIsGivenUpOn` 在首次全套运行中出现 3 个断言失败（属于同一个测试），修复后该用例及全部单元 / 集成通过；不把首次运行记作全通过。模拟器用视频或合成画面替代相机的验证不代表真机相机通过。真实攀爬片段分析与线路识别测试需要指定片段和岩馆；没有输入时按设计跳过，不能计作通过。旧版本的偶发镜头失控未在本轮定位，不列为已修复。

## 本机产物

都在被 Git 忽略的 `ios/build/1.7.0/`：

- `BetaBuddy-1.7.0-19.xcarchive`：包含相机超时修复的最终归档；Organizer 的同名副本已同步。
- `AppStore/BetaBuddy.ipa`：App Store 发布包，32,688,050 字节。SHA-256：`449b13a0539a7f1650b888b8b08283c5dd2c9c3acc2034a1626d639b6b5308ee`。
- `archive-verification.json`：归档资源、权限、签名和可执行文件摘要校验结果。
- `FixedUnits.xcresult`、`FixedCaptureUI.xcresult`：修复后的测试结果。
- `ReleaseQA.xcresult`：修复前的首轮完整结果，UI 全部通过，但一个单元测试有 3 个断言失败，因此整体结果为失败。
- `qa-attachments/`：首轮测试截图（相机画面为合成测试素材，不作为真实识别效果的商店宣传图）。
- `before-camera-timeout-fix/`：仅用于追溯的旧归档，不用于发布。
- `logs/`：核心测试、首轮测试、复跑、归档及导出错误日志。

发布分支留在本机，未 push；已上传并提交 App Store 正式审核。已在用户指定的 iPhone 17 Pro Max（iOS 27.2）完成真实相机录像与保存检查，详情见下。

## 2026-11-24 · iPhone 17 Pro Max 真机

Xcode 账户登录后 App Store 导出成功。真实设备运行 1.7.0 (19) Debug 自动测试，不使用视频替身，日志确认 `CameraSource`；初次等待相机授权未录制，用户允许后重跑完成。

- 实际录制 10.13 秒、305 帧、0 丢帧，保存文件 9,707,200 字节；HEVC，1080 × 1920，30 fps，视频轨 10.167 秒，无音轨，完整解码无错误。
- 拍的是室内环境，无攀爬者；姿态识别返回未找到清晰攀爬者，后续分析按设计跳过。不能据此宣称真实攀爬分析通过。
- 测试期间热状态为 nominal，不代表长时间录制或分析的温度测试。
- 看过保存成功的调试截图。预览图来源为 `composite`：原生界面叠加最近一次相机帧，并非独立采集的系统预览层截图，不能据此证明预览层方向完全正确。
- 随后安装同一最终归档中的 Release 应用到该设备并打开攀爬册；读取存档确认刚录制的测试记录仍在。用户可以直接试用 Release，相机自动测试与诊断采集代码不在此构建中。
- 本机证据在 `ios/build/1.7.0/device-camera-authorized/`：测试报告、调试截图、录像副本和只含本次测试记录的持久化校验。全部被 Git 忽略，不上传。

线上支持/隐私页面已于 2026-11-24 部署到原 `betabuddy-support` Worker。应用内政策、商店描述和审核备注均包含相机与可选定位说明。

## 2026-11-24 · 正式送审回执

- 11:55:58 上传日志返回 `Upload succeeded` / `EXPORT SUCCEEDED`，构建随后完成 Apple 处理。
- 11:59 正式提交；页面先确认「已提交 1 个项目」，审核详情显示 `1.7.0 (19)`「等待审核」。
- [审核记录](https://appstoreconnect.apple.com/apps/6809249574/distribution/reviewsubmissions/details/888889af-80fa-4906-a9b4-96338ea6ebe8)，提交 ID `888889af-80fa-4906-a9b4-96338ea6ebe8`。
- 构建 ID `141e5086-2868-49a3-98c6-c4cb039f1f1d`。简体中文和英语（美国）栏目继续使用英文内容，描述、推广文本、新增内容与审核备注均已保存。
- 发布设置：审核通过自动发布、立即向所有用户发布、保留评分。会员内购沿用现有商品，无新商品提交。
- 1.6.0 当时显示「可分发」；1.7.0 尚未通过审核，不能视为已上线。
- 本机凭证：`ios/build/1.7.0/appstore-1.7.0-19-submitted.png`；上传日志 `ios/build/1.7.0/logs/upload.log`。

## 2026-11-24 · 支持站点部署完成

用户切换到站点所属 Cloudflare 账户后，通过控制台上传 `services/support-site/public/` 中的两个静态页面，部署版本 `84ae1477` 显示 `100% Ready`。浏览器打开正式 [支持页](https://betabuddy.dozycat.ai/) 与 [隐私页](https://betabuddy.dozycat.ai/privacy/)，确认相机录制、无音轨、取景数据与画面本地处理、记录删除和可选一次性定位说明已生效，隐私页日期为 November 24, 2026。

Python urllib 的匿名读取返回 403，因此本轮未声明 HTTP 内容哈希一致；最终验证通过正常浏览器完成。隐私页整页凭证保存在本机忽略目录 `ios/build/1.7.0/privacy-1.7.0-live.png`。
