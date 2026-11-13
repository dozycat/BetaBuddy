# BetaBuddy 1.3.0 (13) · TestFlight

本次使用 GymPack **1.3.3**，来源为 climber 的 `kinematics-rework` 分支提交 **`8db5933`**（PR #1）。不包含尚未提交的后续修改。数据包仅安装在忽略目录，不提交到应用仓库。分发状态以 [STATUS.md](STATUS.md) 为准。

## What to Test

This build updates the offline climbing simulation and keeps the Logbook folders and filters from build 12.

- Play several routes from start to finish. Try pause, resume, previous and next step, and scrubbing through a move.
- Switch body presets, then try a custom body profile. Check that controls remain responsive while a result is calculated.
- Watch foot placement and transitions from the ground. Report any sudden jumps, detached limbs, or visible intersections.
- Check playback pacing, especially moves that travel farther. Report any freezes or noticeable delays on your iPhone, including the device model and iOS version.
- Reopen the app and confirm your existing Logbook videos, folders and filters still work.

The simulation is an approximation, not climbing instruction. Some motion artifacts remain. Videos and notes stay on your device.

## 验证范围

数据包清单与全部 10 个文件的 SHA-256、文件长度已核对，iOS、Android 已安装副本和来源导出包一致。Swift 核心测试 11 项通过；iOS 单元与集成测试 36 项中 35 项通过、1 项因缺少本地视频按设计跳过。模拟器 Debug 构建与签名 Release 归档成功；归档版本为 1.3.0 (13)，全部数据包文件与固定的 1.3.3 快照逐字节一致。用户明确要求跳过受锁屏影响的模拟器手动播放检查并继续发布，因此本轮没有手动播放、拖动进度、自定义身材计算或画面检查结论。

真机播放流畅度、发热、耗电和实际计算等待时间需要 TestFlight 装机检查；模拟器不能替代真机性能验证。
