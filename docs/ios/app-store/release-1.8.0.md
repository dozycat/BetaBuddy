# iOS 1.8.0 (21)

状态：2026-11-25 12:55（中国时间）**已正式提审，等待审核**。以最新主分支的 1.8.0 准备工作为基础。构建 21 已于 12:45 上传成功，正式版审核通过后自动向所有用户发布，保留评分。同一构建已提交既有公开 TestFlight 群组的外测审核，通过后自动通知测试员。1.7.1 当前已可分发。

数据包：1.9.8（若换成更新的数据包，改这一行和「更新内容」里对应的描述，再按 [验证记录](../validation.md) 里 1.8.0 那一节重跑）。

## 更新内容

- 分析结果改成一步一步看：先描出画面里这条线的点，每一步只标下一个手点和脚点。
- 一个剪纸小人按视频里认出的姿态连续地走一遍这一趟；认不出来的地方淡出，不编动作。
- 「这一趟的建议」：从这一趟的姿态里量出停太久、双脚离墙、手连着走脚不跟、低脚够太远，每条带步号和视频时间，可以点过去看；是自动量的，可能认错，不是教练意见。
- 岩点用法的名字（如下拉、内侧边）不再显示。
- 本线的点装在别的颜色的点或造型上（并点）时，整块算作这条线。
- 第一屏描出这条线画面里的点（包括没碰过的），不再画外框；挨在一起的两种颜色的点分开算，视频上叠的字和背后另一面墙上的点不描。
- 暂时不显示模型的走法和与模型的对照。
- 「分析这一趟」下面的说明改成上面这些（五种语言）。

## 更新说明草稿

简体中文：

> 「分析这一趟」改成一步一步看：先描出画面里这条线的点，每一步只标下一个手点和脚点，一个剪纸小人跟着你的动作连续走一遍。新增「这一趟的建议」：从这一趟里量出停太久、双脚离墙、手走脚不跟这些情况，点一下就跳到视频里那一刻。并点会整块算进这条线。结果是自动识别的，可能认错，不是教练意见。暂时不再显示模型的走法。

English:

> Review your climb move by move with clearer hold outlines and a paper-cut figure that follows the detected movement. Tap observations about pauses and hand or foot movement to revisit that moment in your video. This update reduces some wrong-colour outlines and false detections from video captions and background walls. Detection can still miss or misidentify holds. The model-generated beta comparison is temporarily hidden. All analysis stays on your device.

## 验证

见 [验证记录](../validation.md)「2026-11-25 · iOS 1.8.0 (21) 发布验证」。核心 21/21；iOS 单元／集成 84 通过、2 项按设计跳过；参考片段分析和两项相机 UI 测试在 iOS 18.5 通过。此前相机测试在新系统上的失败来自 sheet 滚动、popover 取消方式及重复无障碍节点；调整测试后，iOS 27.2 两项完整流程均通过，没有修改应用业务代码。分析第一页、逐步页面和建议卡片已检查截图。

Android 资源检查、147 项单元测试和 Debug 构建通过，未发布 Android 正式版本。签名归档与 Android Debug APK 内的数据包均与源文件逐字节一致；16 段颜色回归复查通过。

用户明确选择本次直接发布，不追加真机验证；本轮没有真机实测，不将模拟器和参考片段检查描述为现场效果验收。

## 发布产物

签名归档、上传日志、测试结果、归档校验及 Apple 回执截图保存在本机忽略目录 `ios/build/1.8.0/`。审核状态只代表提交已被接收，尚未通过审核或上线。公开测试沿用已有邀请链接，当前构建仍需等待外测审核。
