# 攀爬册离线岩馆名录

这份名录用于给视频标记岩馆、按城市查找岩馆，以及按已标记的岩馆筛选记录；与 3D 线路的 `GymPack` 是独立资源，不含岩壁、线路或推演数据。应用使用随安装包内置的 JSON，不联网抓取；名录本身不取定位。每条记录可以带一个标明精度的坐标（见「坐标」一节），供按距离排序使用。

## 覆盖

当前快照抓取于 2026-11-14，共 **342 条岩馆记录、52 个城市或自治州、27 个省级行政区**。这是覆盖多地的离线名录，**不是全国全量，不是实时营业名单**。没有收录青海、宁夏、西藏、海南及港澳台的馆；这些地区仍可手填，不代表没有岩馆。源里的拉萨数组为空，香格里拉唯一线索明确没有可核对的 POI，因此没有收录。城市名称按地级区域组织，吉首所在条目归入湘西土家族苗族自治州。

抓取日期不等于核验日期。两站没有为整个目录公开统一更新日期，`sourceUpdatedAt` 为 `null`；不能把抓取日期写成「截至今日已核实营业」。目录里可能存在搬迁、改名、停业、学校或训练馆不对外开放的情况。选择器保留手填入口，现有记录保存馆名和城市快照，不依赖日后目录中还存在该条目。

## 来源与处理

| 来源 | 基础条目 | 保留内容 | 限制 |
| --- | ---: | --- | --- |
| [攀岩岩馆大全](https://timeproves.com/) 的公开 `data.js?v=150` | 284 | 品牌、分店、城市、行政区、来源 ID；最近地铁站只用于定位，不进条目 | 源码含 2026-08 的核验备注，部分类型来自 AI 汇总或待核；本项目不导入馆型、风格、面积、评分和源里的估计坐标。无地址时留空。 |
| [Hylonomus 合作岩馆](https://www.hylonomus.com.cn/zh/contact-us-dealership/) | 79 | 岩馆标题、城市、地址、来源 URL | 品牌自己的合作馆列表，覆盖更广的小城市，但不是全国普查；排除海外经销点和白河试鞋点。 |

21 条能根据品牌、城市、分店位置对应的记录合并：`284 + 79 - 21 = 342`。对应关系在 `data/gym-directory/branch-matches.json`，不使用模糊馆名自动合并；同城同名但不同地址的分店保留。合并时保留第一份稳定 ID，记录双方 `sourceIDs` / `sourceURLs` 和标题别名，地址采用合作馆页面的公开值。行政区冲突时采用该地址的行政区，其余未能确定的关系仍保留独立记录，后续复核可能继续发现别名或迁址。

名录条目本身不来自 OpenStreetMap；OpenStreetMap 只用来给已有条目补坐标，许可和署名见「坐标」一节。

未发现上述两站公开的开放数据许可证。这里只整理名称、分店、行政区和地址等基础事实，没有复制文章、图片、馆型评论、营销介绍、个人联系人、电话、营业时间或价格；**第三方源数据不适用本项目源码的 MIT 许可，也没有获得独家或完整性保证**。来源链接、抓取时间和源响应 SHA-256 在 JSON 的 `metadata.sources` 内，支持追溯。

## 坐标

342 条记录都有坐标，全部取自 OpenStreetMap，按能确认到的最细一级标注：

| `coordPrecision` | 条数 | 含义 | 能用来做什么 |
| --- | ---: | --- | --- |
| `poi` | 8 | 岩馆自己在 OpenStreetMap 上的点位 | 排远近 |
| `address` | 64 | 地址或分店名里写明的建筑、商场、园区（跨度不超过 600 米）的位置，不是门牌点 | 排远近，误差按几百米算 |
| `station` | 140 | 名录来源写的「最近地铁站」的位置，不是岩馆位置 | 粗排远近、列出「附近的馆」；误差按一两公里算，个别更大 |
| `district` | 126 | 所在区县的中心点（有驻地点用驻地，否则用边界中心） | 只能判断大致在哪个区 |
| `city` | 4 | 所在城市的中心点 | 只能判断在哪个城市 |
| 无坐标 | 0 | 连城市都对不上时不写坐标字段 | — |

优先级是 `poi` > `address` > `station` > `district` > `city`。六个城市的分布（`poi` / `address` / `station` / `district`）：上海 2 / 15 / 53 / 10，北京 4 / 2 / 35 / 11，深圳 0 / 6 / 11 / 1，广州 0 / 6 / 9 / 8，成都 1 / 4 / 5 / 7，杭州 1 / 5 / 6 / 4。

**`station` 不是「一公里以内」的保证。** 最近地铁站是 timeproves 自己整理的，源文件说明位置是「按所在片区、最近地铁站估的近似值」，不少条目还标着「位置待核」。拿 40 家既有站点、又另外定位到了建筑或岩馆点位的馆来比：站点与实际位置相差的中位数约 1.0 公里，45% 在 1 公里内，72% 在 2 公里内，88% 在 3 公里内，有 4 家相差 6–10 公里。同样这 40 家，区县中心的中位误差约 3.0 公里，最远 22 公里；站点比区县中心更近的有 35 家。所以 `station` 可以用来粗排和列「附近的馆」，不能据此说「最近的是哪一家」。

**`district` 仍有 126 条。** 同一个区的岩馆落在同一个点上，大区的中心离实际岩馆可以有十几到三十公里（重庆、苏州的几个大区都是这样）；`district` 和 `city` 只能用来挑城市。原因是 OpenStreetMap 在中国大陆收录的室内岩馆很少：全国带攀岩标签或名字的要素 317 个，多数是野外岩场和港澳的场地，能对上名录的只有十来个。留在区县一级的 126 条里：39 条来自合作馆页面，那里没有地铁站信息；49 条 timeproves 没写最近地铁站（苏州 13、昆明 7、重庆 7、西安 7 等）；24 条写的站名在该城市的 OpenStreetMap 里找不到同名车站（多是简称或旧名）；4 条同城有两个同名车站；10 条站点落在别的区县。

怎么定的：

- **`poi`**：岩馆的品牌名出现在 OpenStreetMap 要素的名字里，且该点反查出来的区县与名录一致。同城有多家分店的品牌，还要求要素名或所在街道带分店名，或者点位在该馆的建筑附近；否则不用。同一个要素被两家馆对上时都不用。
- **`address`**：从门牌地址里取街道号后面的建筑名，或者取分店名，在所在城市范围内查 OpenStreetMap。只接受名字包含查询词、位于同一区县、自身跨度不超过 600 米的要素；道路、地铁站、片区、公园、路口不算；同区有两处同名且相距超过 600 米时不用。分店名是路名或片区名的（如「某某路」「某镇」）不查。
- **互相矛盾时不用**：岩馆点位与建筑位置相距超过 600 米，又没有地址里的街道名可以裁决，两者都不用，退到站点或区县。
- **`station`**：站名取自 timeproves 每条记录的最近地铁站，存在 `data/gym-directory/station-hints.json`，不写进名录条目。在该城市的行政边界内按站名精确查 OpenStreetMap 的车站（允许多一个「站」字）；查不到不用，同城同名的车站相距超过 1.2 公里（不是同一个换乘站）不用，站点反查出来不在岩馆所属区县的不用，离区县中心超过 25 公里且在区县范围外的也不用。来源没有给线路，所以没有按线路消歧，重名一律放弃。日志里会把同时有建筑位置的馆标出站点与建筑的距离，超过 3 公里的标 `far`。
- **`district` / `city`**：按名字查行政区边界。名录写「浦东」「海淀」这类简称时依次试「区、新区、市、县」；地址里写了区县而 `district` 为空的条目，只在定位时借用，不改 `district` 字段。名录里不是正式区县名的写法按脚本里的显式对照表处理（苏州「工业园区」→ 苏州工业园区，长沙「长沙」→ 长沙县，廊坊「燕郊经济开发区」→ 三河市）；查不到边界的退到城市中心。4 条 `city`：眉山、锦州、牡丹江各 1 条地址里没写区县，中山 1 条写的是街道级的「东区」。

**坐标系**：OpenStreetMap 是 WGS-84，与 CoreLocation、Android 定位一致，没有做转换，也没有用到 GCJ-02 或 BD-09 的来源。timeproves 源里的估计坐标和 Hylonomus 页面上的地图标记都没有导入（前者网站版已删除位置数据，后者只标了品牌自己的一个点）。

**许可与署名**：坐标和 `data/gym-directory/geocode-cache.json`、`coordinate-matches.json` 里的查询结果来自 OpenStreetMap，© OpenStreetMap contributors，按 [Open Database License 1.0](https://www.openstreetmap.org/copyright) 提供，不适用本项目的 MIT 许可。应用里用到这些坐标的界面或许可页需要标注 OpenStreetMap。查询经 Nominatim（每秒不超过 1 次、结果落盘缓存、通用的客户端标识）和 Overpass API，请求里只有公开的馆名、地址里的建筑名和行政区名。

**抽查（2026-11-22）**：`poi` 和 `address` 拿名录外的资料对了 26 条。8 条 `poi` 全部看过：7 条的点位落在公开地址写的那条街上或那个场地里，1 条只找到「某商场东北的园区」这类描述、查不到门牌，无法核实。18 条 `address`：5 条的建筑在 Wikidata 上有坐标，相差 13–370 米；11 条的建筑在合作馆页面给的那条街 300 米以内；2 条只能确认同名、同区，街道很长，算核实不充分。没有发现错误点位。这是抽查，不是逐条核实；`address` 核对的是建筑的位置，岩馆在不在那栋楼里依据的是名录的地址或分店名。

规则排除的典型错配：名字带前缀的另一栋楼、同名的另一个校区、名字里带分店字样的餐馆、同名路口、品牌搬迁前的旧点位。

**`station` 抽查（2026-11-22）**：在 9 个城市挑了 19 家只到站点一级的馆，把站点和公开地址上的建筑（商场、场馆的公开门牌，或合作馆页面的地址）比了距离。14 家在 1 公里内（其中 12 家在 500 米内，有 2 家的建筑不在 OpenStreetMap 上，是按公开资料写的出口和路口判断的）；3 家超过 1 公里（约 1.5、2.8、3 公里：站名与馆所在的路同名但路很长，或来源写的不是真正最近的站）；2 家只知道路名、路上的门牌定不了位，无法核实。和上面 40 家的对比结论一致：多数准，但有一截两三公里的尾巴。

每条坐标对上的 OpenStreetMap 对象和查询词记在 `data/gym-directory/coordinate-matches.json`，发现错点时从这里查起。

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
      "sourceURL": "https://example.org/",
      "lat": 10.12345,
      "lon": 20.12345,
      "coordPrecision": "district"
    }
  ]
}
```

`id/name/province/city/address/sourceURL` 始终为字符串；未知地址为空串。应用解码器忽略额外元数据字段。ID 与数组位置、排序无关：timeproves 使用上游具名分店 ID，Hylonomus 用省市、原始标题、原始地址的 SHA-256 截断值；上游改址后新 ID 不应让历史记录丢失。

`lat`、`lon`、`coordPrecision` 是可选字段，三个要么都有、要么都没有；`lat`、`lon` 是 WGS-84 十进制度的数字，保留 5 位小数，`coordPrecision` 是 `poi`、`address`、`station`、`district`、`city` 之一。地铁站、区县或城市中心点一律按 `station` / `district` / `city` 标注，不冒充岩馆位置；连城市都对不上的条目没有这三个字段。解码时把它们当可选字段，遇到不认识的 `coordPrecision` 值按没有坐标处理。

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

# 从名录快照所用的那份 data.js 重写 station-hints.json（哈希与快照记录不一致就中止），不动名录
python3 scripts/gym-directory/fetch_directory.py --source-file /path/to/data.js --station-hints

# 只重算坐标，不重抓名录：缓存里有的不再查询，缺的才去问 OpenStreetMap
python3 scripts/gym-directory/geocode_directory.py
# 完全离线复现（缓存缺任何一项就中止，不联网）
python3 scripts/gym-directory/geocode_directory.py --offline

# 无网络检查：三份内容一致、ID 唯一、必要字段和来源齐全、坐标都带精度标注且计数一致
python3 scripts/gym-directory/fetch_directory.py --check
# 无网络检查：坐标都在所属省份的范围内，不同城市的馆没有共用同一个 poi 点（站点与区县的核对在定位时做，不合格的不会写进文件）
python3 scripts/gym-directory/geocode_directory.py --check
python3 -m unittest discover -s scripts/gym-directory -p 'test_*.py'
```

`fetch_directory.py` 重抓名录后会接着跑同一套定位（`--coordinates online|offline|skip`，默认只查询缓存里没有的）。定位日志逐条打印每家馆落在哪一级、对上了哪个 OpenStreetMap 对象，以及被放弃的候选和原因。缓存 `data/gym-directory/geocode-cache.json` 进仓库：同一份缓存重跑得到逐字节相同的三份文件；一次完整运行结束后，不再用到的缓存项会被清掉。想刷新某条结果，删掉缓存里对应的那一行再跑。

脚本只解析 JavaScript 的数组、对象、字符串、数字和布尔字面量，不执行下载的 JavaScript。来源结构变化、必需数组消失、跨城市匹配或重复 ID 会中止，不覆盖现有安装包数据。`branch-matches.json` 的上游 ID 不再出现时也中止，需要检查该分店是否改名或迁址后再调整对应关系。实际查询范围和覆盖数量存于 `metadata`，发布说明应报告这些数字，不能把「多地名录」改写成「全国全量」。

## 验证

- 名录：真实抓取过两个源；三份一致性检查、解析器安全边界和坏数据回归检查通过。
- 坐标：离线重跑两次逐字节相同；用保存的源响应走完整的 `fetch_directory.py` 流程，结果除合作馆页面的哈希外一致；`scripts/gym-directory` 的 22 项测试、两个 `--check` 和 `android/scripts/check-resources.py` 通过；`poi`、`address`、`station` 的抽查见「坐标」一节。

尚未验证：逐馆电话或现场核实营业状态和点位；全国全量覆盖。
