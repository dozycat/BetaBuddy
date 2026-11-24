# iOS 1.8.0 (21)

状态：已准备，**未归档、未上传、未提审**。以 1.7.1 发布分支为基础。

数据包：1.9.5（若换成更新的数据包，改这一行和「更新内容」里对应的描述，再按 [验证记录](../validation.md) 里 1.8.0 那一节重跑）。

## 更新内容

- 分析结果改成一步一步看：先框出这条线，每一步只标下一个手点和脚点。
- 从画面里看出岩点怎么用（如下拉、侧拉、内侧边），只在把握够时给名字；是自动识别，可能认错，不是教练意见。
- 本线的点装在别的颜色的点或造型上（并点）时，整块算作这条线。
- 暂时不显示模型的走法和与模型的对照。
- 「分析这一趟」下面的说明改成上面这些（五种语言）。

## 更新说明草稿

简体中文：

> 「分析这一趟」改成一步一步看：先框出线路，每一步只标下一个手点和脚点，并从画面里看出岩点大概怎么用。并点会整块算进这条线。结果是自动识别的，可能认错，不是教练意见。暂时不再显示模型的走法。

English:

> “Analyse this attempt” now walks through your climb move by move: it frames the problem, then marks only your next hand and foot holds, with how each hold looks to be used. Holds bolted onto another hold or volume count as part of the problem. Results are auto-detected, can be wrong, and are not coaching advice. The model’s beta is hidden for now.

## 验证

见 [验证记录](../validation.md)「2026-11-24 · iOS 1.8.0 发布准备」。摘要：核心 21/21；单元 / 集成 84 通过、2 跳过；UI 中带参考片段的分析端到端通过；相机界面 2 项 UI 测试失败（1.7.1 上同样失败，尚未处理）；未签名真机构建成功；Android 单元 147/147、Debug 构建与 lint 通过。

## 发布前还要做

- 处理或确认相机界面那 2 项 UI 测试失败。
- 真机上看一遍分析页的逐步几屏。
- 归档、上传、提审由用户决定；签名团队在命令行传入，不写进仓库。
