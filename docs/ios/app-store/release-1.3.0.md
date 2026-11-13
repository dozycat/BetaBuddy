# BetaBuddy 1.3.0 · TestFlight

构建 12：攀爬册文件夹、岩馆／难度／攀爬日期组合筛选，以及离线岩馆目录。历史记录兼容读取，删除文件夹保留记录与视频。发布状态以 [STATUS.md](STATUS.md) 为准。

## What to Test

Organize your climbs with folders and combined filters.

- Create and rename a folder, move a climb into it, then delete the folder. Your climbs and videos should remain in Unfiled.
- Combine gym, grade, climbing date and send status filters. Clear filters to see all your climbs again.
- When importing or replaying a climb, edit its gym, climbing date and folder.
- Search the offline gym directory by name or city. It currently includes 342 listings across 52 cities or prefectures in China; coverage is incomplete and listings are not live business-status checks. You can enter an unlisted gym yourself.
- Check that existing videos, analysis and notes remain available after upgrading. Try changing the app language and reopening the app.

Videos and notes stay on your device. Gym and Logbook remain free; lifetime membership is optional.

## 构建核对

`ios/project.yml` 与生成工程使用 1.3.0 (12)，并声明没有非豁免加密。签名归档包含 342 条岩馆目录，目录与仓库快照逐字节一致；3D GymPack 使用本机已安装的数据包，清单和所有文件校验通过，数据包未提交进仓库。

本轮功能验证见 [validation.md](../validation.md)。本次 TestFlight 的重点是真机升级与分类操作；不把模拟器验证描述为真机验证。
