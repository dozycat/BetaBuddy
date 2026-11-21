# iOS 1.6.0 (18)

状态：代码与模拟器验证完成、签名归档成功；CLI 上传失败，尚未上传或提交审核。数据包 GymPack 1.8.0（2026-11-21T02:45:33Z）。

## 此版本的新增内容（所有商店语言使用英文）

- Let a cat climb: lifetime members can watch cat climbing replays with two cats and two hold modes. Existing members get access at no extra cost.
- Updated offline gym pack, including 36 problems at DOME Xuhui Vanke.
- Simpler recording: one Save button keeps your video and returns to Logbook. Cancel closes without saving.
- Restored support for climbing videos up to 20 minutes.

## 商店描述（简体中文与英语〔美国〕均填以下英文）

Keep your climbing videos, notes and beta together on your iPhone.

LOG YOUR CLIMBS
Import a video, add a grade and notes, then tap Save. Your recording returns to Logbook, ready to replay. Organize attempts in folders and edit the climb date and time. Optional on-device pose analysis adds skeleton playback and movement review to saved recordings.

EXPLORE OFFLINE GYMS
Browse unofficial 3D gym models, select problems and replay simulated movement with different climber builds. BANANA+ Kerry includes 72 problems and is free to explore. Lifetime membership unlocks additional gyms, including 36 problems at DOME Xuhui Vanke.

LET A CAT CLIMB
Lifetime members can watch cat climbing replays with two cat presets. Choose between using the wall's holds and only the selected problem's holds. These are simulated animations; results depend on the available route data.

ONE-TIME MEMBERSHIP
Human climbing at BANANA+ Kerry and Logbook are free. The optional lifetime membership unlocks more gyms, cat climbing and a permanent member badge. Existing lifetime members receive these features at no extra cost. Restore purchases using the same Apple Account. No subscriptions or ads.

YOUR CLIMB, YOUR DEVICE
The pose model and gym data are bundled with the app. Imported videos, notes and analysis stay on your device. No app account or cloud AI upload is required. Videos can be 0.5 seconds to 20 minutes long and up to 500 MB.

Gym models and problems are unofficial and may differ from the current venue. Pose results can be affected by occlusion, camera movement and multiple people. Simulations and analysis are tools for review, not safety or coaching advice.

## 宣传文本

Let a cat climb. Lifetime members get two cat presets and more offline gyms. Recording is simpler too: one Save button keeps your climb in Logbook.

## 现有终生会员内购元数据

沿用 `com.betabuddy.ios.lifetime`，不新建商品，不改价格或销售地区。五种现有商品本地化继续使用英文：

- Display Name: `BetaBuddy Lifetime Member`
- Description: `More gyms, cat climbing and a member badge.`
- Review Screenshot: 本机 `ios/build/1.6.0/membership.png`，1206 × 2622，来自实际模拟器页面。

## 审核备注

This version adds cat climbing to the existing non-consumable lifetime membership (com.betabuddy.ios.lifetime). No new purchase is required for existing lifetime members. In Gym, select a problem and choose Cat in the climber controls. Non-members see the existing membership offer. A verified purchase or Restore Purchases unlocks the feature. Refund or revocation removes access. Additional gyms remain part of the same membership. Human climbing at BANANA+ Kerry and Logbook remain free.

The cat animation uses precomputed movement plans from the bundled gym pack, rendered in 3D. It is not video analysis or a claim that a real cat can safely climb the route. Two cat presets and two hold modes are included. It runs offline after purchase verification, as do the gym viewer and the on-device video review tools. Imported videos and notes remain on the device.

In Logbook, tap the import button, select a video, then tap Save at the top right. The recording is saved before returning to Logbook. Cancel at the top left discards the import. Optional pose analysis runs after saving. Videos from 0.5 seconds to 20 minutes and up to 500 MB are supported.

## 验证记录

- iOS 单元 / 集成：52 项，51 通过、1 项可选真实视频测试按设计跳过。涵盖真实 1.8.0 数据包的猫入口、StoreKit 购买/恢复/退款、语言重载，以及资源访问控制。
- Mac 核心测试：10/10 通过。
- UI：文件夹、五项导航/语言/会员文案、单一保存后重启读取与取消不保存、两种猫播放均通过。真实攀爬片段端到端分析未提供片段，按设计跳过。
- 首轮保存测试在清理样例时误选删除按钮；改为定位确认弹窗后复跑通过，保存与持久化本身通过。
- 最终代码单元 / 集成复跑：52 项，51 通过、1 项可选片段跳过。
- 签名归档：本机忽略目录 `ios/build/1.6.0/BetaBuddy-1.6.0-18.xcarchive`。已核对版本 1.6.0 (18)、21 个资源哈希与签名，不含 StoreKit 测试配置。清单 SHA-256：`7ff665c2486c6cb64fedac3621e2d8a6553fa690dd7c21e796c894bc615f9051`。同目录保存两只猫、会员页及导入页的实测截图。
- CLI 上传返回 `Failed to Use Accounts`，后续用 Xcode Organizer。Mac 锁屏、Chrome 的 App Store Connect 页面显示登录表单，已请用户解锁并登录。
- 支持站点源文件与静态文件已更新，尚未部署；此工作树未安装 Wrangler。
- 偶发镜头失控：用户确认无闪退、重启恢复；根因尚未确认，不宣称修复。
