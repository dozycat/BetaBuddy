# BetaBuddy 1.3.0 (14) · TestFlight

本次打磨攀爬册、导入和回看页。GymPack 仍为 **1.3.3**，来源提交 **`8db5933`**。分发状态见 [STATUS.md](STATUS.md)。

## What to Test

This build refines the Logbook, import and video review layouts.

- Check that BetaBuddy stays on one line and button labels have comfortable spacing, including on smaller iPhones and with larger text.
- Import a video from Photos or Files. Video tips and analysis settings now expand when needed.
- Search your climbs, switch folders, and combine gym, grade, date and send-status filters. Clear filters and confirm your records return.
- Open a climb, try playback controls, save notes, and use the More menu to organize it. Analysis details are now collapsible.
- Membership details remain available from Settings. Existing videos, notes and folders should be preserved.

Please include your iPhone model, iOS version and text-size setting when reporting layout issues. The offline simulation is unchanged from build 13.

## 验证范围

核心测试 11 项通过；iOS 单元／集成测试 35 项通过、1 项按设计跳过；UI 测试 5 项通过。复核修正后重跑单元／集成测试，最后文案调整后 10 项本地化测试通过。未签名设备构建与最终签名归档通过。SE 窄屏英文空列表和导入页已实际查看；大字体截图误存为岩馆说明页，不作为本轮改动的验收证据。后续因 Mac 锁屏未完成最终文案截图、最大辅助字号、回看手动操作及真机性能验证，详见 [验证记录](../validation.md)。
