# iOS 1.8.0 (21)

状态：已准备，**未归档、未上传、未提审**。以 1.7.1 发布分支为基础。

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

> “Analyse this attempt” now walks through your climb move by move: it frames the problem, marks only your next hand and foot holds, and a paper-cut figure follows your moves the whole way. New “Tips from this attempt” point out long pauses, both feet cutting loose and hands moving without the feet, each linked to that moment in the video. Holds bolted onto another hold or volume count as part of the problem. Results are auto-detected, can be wrong, and are not coaching advice. The model’s beta is hidden for now.

## 验证

见 [验证记录](../validation.md)「2026-11-25（中午）· 1.8.0 换成数据包 1.9.8」、「2026-11-25 · 1.8.0 换成数据包 1.9.7」（这两次只换了数据包，只重跑了分析界面测试）、「2026-11-24（晚）· 1.8.0 换成数据包 1.9.6」和之前的「2026-11-24 · iOS 1.8.0 发布准备」。摘要：核心 21/21；单元 / 集成 84 通过、2 跳过；UI 中带参考片段的分析端到端通过；相机界面 2 项 UI 测试失败（1.7.1 上同样失败，尚未处理）；未签名真机构建成功（数据包 1.9.6）；Android 单元 147/147、Debug 构建与 lint 通过。

## 发布前还要做

- 处理或确认相机界面那 2 项 UI 测试失败。
- 真机上看一遍分析页：逐步几屏、小人连续播放是否流畅、建议卡片。
- 归档、上传、提审由用户决定；签名团队在命令行传入，不写进仓库。
