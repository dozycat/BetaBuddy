# App Store 中文资料

简体中文的商店文案源，与应用现在的功能一致。英文见 [metadata-en.md](metadata-en.md)。每个版本的「新增内容」和提审清单在 `release-<版本>.md`；后台实际填了什么、到了哪一步，见 [STATUS.md](STATUS.md)。

- 名称：BetaBuddy
- 副标题：岩馆线路推演与攀岩视频复盘
- 主要类别：体育
- App 价格：免费；可选的一次性终生会员内购，美国售价 $10；无订阅
- 关键词：攀岩,抱石,岩馆,线路,beta,复盘,动作分析,姿态,视频,手账
- 支持邮箱：hi@dozycat.ai
- Support URL：https://betabuddy-support.dozycfs.chatgpt.site
- Privacy Policy URL：https://betabuddy-support.dozycfs.chatgpt.site/privacy
- 隐私标签：根据当前实现，选择“不收集数据”；提交前复核最终打包 SDK 和实际行为
- 登录：无需账号
- 应用内语言：简体中文、繁体中文、英语、日语、韩语；默认跟随系统，可在应用内切换

## 宣传文本

再试一次，这次有点不一样。在「岩馆」里看一条线换个身材可以怎么爬；导入一趟攀爬，回看骨架与动作细节。核心功能免费，可选一次性终生会员，无广告或订阅。

## 描述

每一趟，都有新发现。

BetaBuddy 有两个入口：岩馆和攀爬册。

岩馆
• 内置一座岩馆的非官方 3D 线路图，可按岩壁、难度档和岩点颜色筛选线路，查看起步点和结束点。
• 换个身材看这条线：选择不同身材，查看模型推演出的一种走法，并逐步回放。
• 位置和线路为爱好者根据公开资料整理，可能与现场不符；推演不是教练建议，请以现场和自身判断为准。

攀爬册
• 导入自己的攀爬视频，在手机上识别 33 个身体关键点，对照原片回看骨架。
• 两种分析模式：按需要选择日常回看或动作细看。
• 0.1 秒步进、0.5× 慢放，候选片段一点就到。
• 记下难度、是否完攀和一句下次想试的改变；可按完攀状态筛选、按名字或难度搜索。

你的数据留在手机
• 无需账号，不上传视频，不调用云端 AI；姿态模型和岩馆数据随安装包提供，岩馆页面不联网。
• 岩馆与攀爬册核心功能免费，无广告、无订阅。可一次性购买终生会员，立即显示专属标识；训练建议和新岩馆数据将在后续推出。

适合单人、全身入镜、固定机位的视频。支持 0.5–180 秒、最大 500 MB 的单段视频，建议先在相册中裁剪。

姿态结果是辅助回看信息。遮挡、多人入镜和镜头移动会影响识别；二维角度和髋部轨迹不代表完整的三维身体力学。请结合原片判断。

需要 iOS 17 或更新版本。选择仅保存在 iCloud 的视频时，系统可能需要先下载原片。本地记录不随系统备份恢复，请自行保存需要的复盘文字。

## 审核备注（英文）

BetaBuddy is an iPhone app with two tabs: “岩馆” (Gym) and “攀爬册” (Logbook). The core features are free, with no login, subscriptions or advertising. Version 1.2 offers an optional one-time lifetime membership through Apple's In-App Purchase; it immediately shows a badge in the Logbook. Training recommendations and new gym data are planned for later and are not in this build. The interface is available in Simplified Chinese, Traditional Chinese, English, Japanese and Korean and follows the system language or the user's in-app selection.

Gym (first tab, shown at launch): a bundled, offline, unofficial 3D layout of one climbing gym with its boulder problems, and a model-based what-if simulation of how climbers of different body sizes might climb a selected problem. To test: read the one-time notice and tap the button at the bottom to enter, pick a wall and a problem from the bottom sheet, then press play to step through the simulated sequence; the climber tab in the sheet changes the body parameters. This tab is rendered in an embedded web view from files bundled inside the app. It makes no network requests: all http/https/ws navigation and subresource loads are blocked, and the files are served from a custom local scheme. The layout is a hobbyist reconstruction, labelled as unofficial in the app, and is not affiliated with or endorsed by the gym. The simulation is an illustration, not coaching or safety advice.

Logbook (second tab): tap the import button, select a locally available single-person full-body MOV/MP4 video using the system Photos picker, then start analysis. The MediaPipe Full pose model is bundled and runs on device. Supported duration is 0.5–180 seconds, up to 500 MB. Videos without a detectable body produce an explanatory error. Review results alongside the original video. Saved climbs are listed in this tab. The button at the top right opens About, where the privacy policy is available offline.

The purchase and Restore Purchases buttons are in Settings & About. There is no paywall for Gym or Logbook and no account requirement. Pose output is intended for visual review, not medical diagnosis or a claim of validated climbing biomechanics.

## 需在账号中完成

- 填写个人开发者真实版权信息和审核联系姓名、电话；不要用应用名冒充法律主体。
- 支持页与隐私政策使用以上两个 URL；行为变化后同步线上内容（文本源：[privacy-policy.txt](privacy-policy.txt)、[privacy-policy-en.txt](privacy-policy-en.txt)、[support.md](support.md)）。
- 完成年龄分级问卷、内容权利、加密出口问卷及适用地区信息。
- 发布地区由开发者选择。中国大陆如适用，需提供 Apple 要求的备案资料；不要填造假的编号。
- App 名称和 Bundle ID 的可用性以开发者后台为准。
