# Android 追平 iOS 1.4 与国内商店发布

2026-11-16 起推进。目标是先完整追平 iOS 当前能力（包括终生会员），再向小米和 OPPO 提交正式 Android 包；HarmonyOS 5+ 原生版同步开发。此文件记录可验证状态，不能把源码存在当作已经上架。

## 当前差距

| 能力 | Android 状态 |
| --- | --- |
| 岩馆、72 条线路、离线模拟 | 代码已支持；GymPack 1.3.9 已从私有数据仓库校验并安装到本机忽略目录，发布包须再核验 |
| 攀爬册、文件夹、组合筛选、拍摄时间、后台分析 | 已实现，设备范围与性能仍需实机验收 |
| 20 分钟、500 MB 上限；相册和文件导入 | 本分支已实现；仓库 CI 编译、单测和 lint 已通过 |
| 设置页记录数、在线隐私政策 | 本分支已实现；仓库 CI 编译、单测和 lint 已通过 |
| 终生会员购买、恢复、标识 | 尚未实现；不能声称 Android 与 iOS 完全一致 |

## 上架前必须完成

1. 会员采用各平台分别购买、分别恢复，价格尽量接近 iOS 的商店定价；不承诺跨平台互通。各商店实际本地价格以商品配置和支付页显示为准。
2. 接入适用于小米和 OPPO 的非消耗型商品购买、订单核验、恢复和退款撤销。付款结果不能只信任本地布尔值。小米 IAP 官方流程目前要求商务开通白名单和商品配置；OPPO 的应用联运支付接入需从开发者后台确认合作条件。新增支付后同步修改隐私政策、网络权限和商店资料。
3. 用户确认旧 Android 发布密钥已遗失，且本次国内商店发布无需兼容旧测试包。首次正式发布前生成并妥善备份新密钥；`android/keystore.properties` 与密钥不能进 Git。
4. 在小米、OPPO 真机验证视频导入、20 分钟边界、后台分析、省电限制、温升和购买／恢复／退款。模拟器与 JVM 测试不代替真机验收。
5. 小米、OPPO 开发者账号目前均未注册。账号实名认证后，准备软件版权证明、单机应用承诺函、正式 APK、商店截图和介绍。OPPO 需公开的隐私政策网址；现有 `https://betabuddy.dozycat.ai/privacy/` 可作为基础，支付接入后需更新内容。

## 外部资料

- 小米 IAP 与白名单：https://dev.mi.com/xiaomihyperos/documentation/detail?pId=2232
- 小米单机应用备案选项：https://dev.mi.com/xiaomihyperos/documentation/detail?pId=1775
- 小米基础资质：https://dev.mi.com/xiaomihyperos/documentation/detail?pId=2251
- OPPO 应用上传及单机承诺函：https://open.oppomobile.com/bbs/forum.php?mod=viewthread&tid=6304
