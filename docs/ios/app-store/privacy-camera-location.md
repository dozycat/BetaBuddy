# 应用内相机与位置 · App Store 隐私说明

适用于应用内相机「拍一趟」（iOS 1.7.0 起）。提交时按这里填写；实现变了先改这里。

## 用了什么

| 能力 | 用途说明（`InfoPlist.xcstrings`，五种语言） | 实际做的事 |
| --- | --- | --- |
| 相机 `NSCameraUsageDescription` | 拍下你的攀爬，在这台 iPhone 上检查机位并保存。画面和录像不会离开设备。 | 后置相机，只有画面。取景时每秒把两帧缩小的预览画面交给安装包里的拍摄页面（不能联网），画面不逐帧保存；开始录像时页面给出的拍摄计划里有一张由预览画面合成的墙面静态图，它和录像一起保存在应用的本地目录，删除记录时一并删除 |
| 位置 `NSLocationWhenInUseUsageDescription` | 按距离推荐附近的岩馆。位置只在这台 iPhone 上使用，不保存，也不会离开设备。 | 使用期间、一次性定位；只用来在本机给离线岩馆目录排序。不写入记录、偏好或日志。拒绝后功能照常，只是没有按距离的推荐 |

岩馆目录里的坐标来自 OpenStreetMap（ODbL），随安装包离线提供；目录带坐标时，署名「© OpenStreetMap contributors」显示在按距离选岩馆的表、岩馆目录页和「设置与关于」里。

不请求麦克风：录像没有声音轨。不请求相册写入：拍的视频只存在应用里。不使用后台定位、不使用精确位置之外的任何位置能力；用户只给大致位置时照样可用。

## 隐私标签

仍是 **Data Not Collected**。Apple 的定义里「收集」指数据离开设备、开发者或第三方可以取得；相机画面、录像和位置都只在设备上处理，不传给开发者，也不传给任何第三方。应用里没有分析或广告 SDK。

## 隐私清单

`PrivacyInfo.xcprivacy`：不追踪，不收集数据。需要说明理由的 API 增加一项磁盘空间 `NSPrivacyAccessedAPICategoryDiskSpace`，理由 `E174.1`（开始录像前和录像中查看剩余空间，空间不足时不开始或提前停下并保存）。原有的文件时间戳 `C617.1` 和 UserDefaults `CA92.1` 不变。相机、位置和运动传感器不属于需要说明理由的 API。

## 审核备注里要写的话（英文）

> The Logbook's “Record” button opens the in-app camera (back camera, video only, no audio). While framing, scaled-down preview frames are checked on the device by a page bundled in the app, which has no network access. The recording, and one still picture of the wall made from the preview, are stored inside the app only and deleted with the record. Location is optional and requested only after the user taps “Find gyms near me”; one fix is used on the device to order the bundled offline gym list by distance and is not stored. Nothing from the camera or location leaves the device.

## 提交前核对

- 线上隐私政策（`services/support-site`）要与应用内的三份和 `docs/ios/app-store/privacy-policy*.txt` 同步；应用内文本、仓库政策和支持站点源文件已同步，静态页面已重新生成；线上部署尚未完成。
- 用途说明的五种语言与上表一致，没有多出承诺。
- 真机上拒绝相机、拒绝位置、只给大致位置三种情况都走一遍。
