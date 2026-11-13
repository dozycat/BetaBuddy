# 攀爬册离线岩馆名录

这份名录用于给视频标记岩馆、按城市查找岩馆，以及按已标记的岩馆筛选记录；与 3D 线路的 `GymPack` 是独立资源，不含岩壁、线路或推演数据。应用使用随安装包内置的 JSON，不联网抓取，不需要定位权限。

## 本次实际覆盖

2026-11-14 抓取后，共 **342 条岩馆记录、52 个城市或自治州、27 个省级行政区**。这是覆盖多地的第一批离线名录，**不是全国全量，不是实时营业名单**。本批没有收录青海、宁夏、西藏、海南及港澳台的馆；这些地区仍可手填，不代表没有岩馆。源里的拉萨数组为空，香格里拉唯一线索明确没有可核对的 POI，因此没有收录。城市名称按地级区域组织，吉首所在条目归入湘西土家族苗族自治州。

抓取日期不等于核验日期。两站没有为整个目录公开统一更新日期，`sourceUpdatedAt` 为 `null`；不能把本次抓取写成「截至今日已核实营业」。目录里可能存在搬迁、改名、停业、学校或训练馆不对外开放的情况。选择器保留手填入口，现有记录保存馆名和城市快照，不依赖日后目录中还存在该条目。

## 来源与处理

| 来源 | 本次基础条目 | 保留内容 | 限制 |
| --- | ---: | --- | --- |
| [攀岩岩馆大全](https://timeproves.com/) 的公开 `data.js?v=150` | 284 | 品牌、分店、城市、行政区、来源 ID | 源码含 2026-08 的核验备注，部分类型来自 AI 汇总或待核；本项目不导入馆型、风格、面积、评分和估计坐标。无地址时留空。 |
| [Hylonomus 合作岩馆](https://www.hylonomus.com.cn/zh/contact-us-dealership/) | 79 | 岩馆标题、城市、地址、来源 URL | 品牌自己的合作馆列表，覆盖更广的小城市，但不是全国普查；排除海外经销点和白河试鞋点。 |

21 条能根据品牌、城市、分店位置对应的记录合并：`284 + 79 - 21 = 342`。对应关系在 `data/gym-directory/branch-matches.json`，不使用模糊馆名自动合并；同城同名但不同地址的分店保留。合并时保留第一份稳定 ID，记录双方 `sourceIDs` / `sourceURLs` 和标题别名，地址采用合作馆页面的公开值。行政区冲突时采用该地址的行政区，其余未能确定的关系仍保留独立记录，后续复核可能继续发现别名或迁址。

本次也尝试 OpenStreetMap 全国查询，但 Overpass 接口返回 HTTP 406，没有任何 OSM 记录进入安装包；因此这里不声称具有 OSM 的覆盖率或许可证。

未发现上述两站公开的开放数据许可证。这里只整理名称、分店、行政区和地址等基础事实，没有复制文章、图片、馆型评论、营销介绍、个人联系人、电话、营业时间或价格；**第三方源数据不适用本项目源码的 MIT 许可，也没有获得独家或完整性保证**。来源链接、抓取时间和源响应 SHA-256 在 JSON 的 `metadata.sources` 内，支持追溯。

## 文件约定

权威快照：`data/gym-directory/gyms-cn.json`。iOS 的 `ios/BetaBuddy/Resources/gyms-cn.json` 和 Android 的 `android/app/src/main/assets/gyms-cn.json` 必须与它逐字节一致。

```json
{
  "version": 1,
  "updatedAt": "2026-11-14",
  "metadata": { "coverage": "partial", "sources": [] },
  "gyms": [
    {
      "id": "source:stable-branch-id",
      "name": "示例攀岩（北城店）",
      "province": "示例省",
      "city": "示例市",
      "address": "",
      "district": "",
      "sourceURL": "https://example.org/"
    }
  ]
}
```

`id/name/province/city/address/sourceURL` 始终为字符串；未知地址为空串。应用解码器忽略额外元数据字段。ID 与数组位置、排序无关：timeproves 使用上游具名分店 ID，Hylonomus 用省市、原始标题、原始地址的 SHA-256 截断值；上游改址后新 ID 不应让历史记录丢失。坐标没有采集，也没有用城市中心伪造岩馆定位。

## 更新与检查

只在开发机运行；Python 3.9+ 和 curl，无 API key、第三方运行依赖：

```sh
# 从两个公开源重新抓取、合并、校验，再原子替换三份文件
python3 scripts/gym-directory/fetch_directory.py

# 使用保存下来的原始响应离线复现（日期传响应实际抓取日）
python3 scripts/gym-directory/fetch_directory.py \
  --source-file /path/to/data.js \
  --partner-file /path/to/partners.html \
  --retrieved-at 2026-11-14

# 无网络检查：三份内容一致、ID 唯一、必要字段和来源齐全、无伪造坐标
python3 scripts/gym-directory/fetch_directory.py --check
python3 -m unittest discover -s scripts/gym-directory -p 'test_*.py'
```

脚本只解析 JavaScript 的数组、对象、字符串、数字和布尔字面量，不执行下载的 JavaScript。来源结构变化、必需数组消失、跨城市匹配或重复 ID 会中止，不覆盖现有安装包数据。`branch-matches.json` 的上游 ID 不再出现时也中止，需要检查该分店是否改名或迁址后再调整对应关系。实际查询范围和覆盖数量存于 `metadata`，发布说明应报告这些数字，不能把「多地名录」改写成「全国全量」。

本次执行：真实抓取两个源；三份一致性检查通过；解析器安全边界和坏数据回归检查通过。未逐馆电话、现场核实营业状态；未完成全国全量覆盖。
