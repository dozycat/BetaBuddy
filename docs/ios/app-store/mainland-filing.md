# BetaBuddy 中国大陆 APP 备案资料草稿

核对日期：2026-09-07。尚未提交备案，未取得备案号。

用户已确认：没有云服务商账号、没有已备案域名。用户已确认常住地：上海。用户已确认身份证地址在上海，满足已查到的上海个人备案住所要求。

## 已核对应用资料

- 名称：BetaBuddy
- Bundle ID：com.betabuddy.ios
- 版本：1.0.0 (1)
- App Store Apple ID：6809249574
- 平台：iOS；语言：简体中文；拟分类：体育/运动工具，以实际选项为准。
- 拟沿用个人主体，法定中文姓名及住所省市待确认。
- 支持邮箱：hi@dozycat.ai；备案负责人邮箱待本人确认。
- 不对外提供 SDK；内嵌 MediaPipe Tasks Vision/Common，用于本地姿态识别。其他开源组件见 ios/THIRD_PARTY.md，按表单实际 SDK 分类填报。

## 功能说明

BetaBuddy 是一款攀岩视频回看与训练记录工具。用户通过系统照片选择器导入自己的视频，使用随安装包内置的姿态识别模型在手机本地分析，回看身体关键点和动作细节，并保存训练笔记。无需注册账号，视频与训练记录在本机处理和保存，无广告、内购和订阅，不提供用户公开发布或社交功能。

## 网络情况及受理待核实

核心功能无开发者运营的后台，不上传视频、不调用云端 AI。用户选择 iCloud 视频时，系统可能下载原片。应用包含外部支持网站和在线隐私政策链接，并有离线政策。

- 支持：https://betabuddy-support.dozycfs.chatgpt.site
- 隐私：https://betabuddy-support.dozycfs.chatgpt.site/privacy

以上第三方托管子域名不能当作已实名的自主域名或已备案接入资源。需根据主体省市和接入商确认此架构的备案适用性、受理入口及真实接入资源要求。不要虚构后端、域名或 IP。

## 发布签名

从正式导出的 AppStore-final/BetaBuddy.ipa 提取，仅含公开证书，无私钥。

```text
subject= /UID=PR5A8VMY8S/CN=Apple Distribution: du zeyu (PR5A8VMY8S)/OU=PR5A8VMY8S/O=du zeyu/C=US
notBefore=Sep  6 22:52:40 2026 GMT
notAfter=Sep  6 22:52:39 2027 GMT
SHA1 Fingerprint=99:B2:88:E7:F9:E9:80:AA:CE:CA:AF:D2:AA:5F:0B:C8:5B:0E:E9:79
```

iOS 的“签名 MD5”类字段通常按接入商指引填写证书 SHA-1，不能填写 IPA 文件 MD5。

RSA 公钥 PKCS#1 DER 十六进制（按实际表单/证书查看器要求核对格式）：

```text
3082010A0282010100A40AAF9875BB13B51724876E5483AF82F361DD422B9D1C1FAFEDA04C9B2257147D61D630B614D8B53944B7792B1468F817D022BA56A82354D366800A2BAD849CCABB2ADB91C3E69744342DCE1132859D95934C4DC08EB7C5FF889646FA16919731E9D673A060A6708CA5D1EC3AB531E49F91C60B81B976E57594ED179DB5159249CA730D0C93549F3BFA1AA4DA38E11F200B6936F2D22686DB561788EF3BB17E389B4FA400F17B2B21FB207A495644F5C3DB402D0465E2162C65B4070A48FF24B3FDFB9A3A2A008686A3A7795A6F6A8029D95F3CB62E057527F9CF82037363C30BFCAB26AE821C0AD76853A95290DE9587D0FE3E6BD73404082AE71B9EB192FB0203010001
```

公开证书：ios/build/mainland-filing/distribution-0。
PEM 公钥：ios/build/mainland-filing/public-key.pem。
证书换发导致公钥/指纹变化后，应同步变更备案。

## 后续流程

1. 用户确认主体、住所省市、现有云服务商和已备案域名。
2. 在实际接入商或分发平台核实本地工具的受理方式，建立申请并填技术资料。
3. 用户在官方页面完成身份资料、活体及短信核验。
4. 完成检查并提交后，记录真实受理编号。
5. 获批后在 App Store Connect 填真实 APP 备案号，并在应用显著位置展示及链接查询系统。

工信部规定，省级通信管理局在收到齐全准确材料后二十个工作日内予以备案；不含准备、接入商核验及补正时间。

## 官方参考

- https://www.miit.gov.cn/jgsj/xgj/hlwgl/art/2023/art_564bf0759d7e41d5b4aa8ce4996b9e84.html
- https://help.aliyun.com/zh/icp-filing/basic-icp-service/support/basics-about-icp-filling-for-apps
- https://help.aliyun.com/zh/icp-filing/basic-icp-service/user-guide/fill-in-app-feature-information
- https://cloud.tencent.com/document/product/243/97691
- https://developer.apple.com/help/app-store-connect/reference/app-information/app-information/

## 无现有云资源时的补充核对

西云数据在 AWS 中国官网的 APP 备案指导 FAQ 明确，单机不联网 APP 不需要备案。这是接入商指引，不能直接当作 Apple 对 BetaBuddy 的审核结论；本应用含支持和隐私外链，需如实说明并确认适用性。

- https://d1.awsstatic.com/onedam/marketing-channels/website/www.amazonaws.cn/zh_CN/support/approved/pdf/app-nwcd-new.pdf

腾讯云官方 FAQ 明确：未备案过的域名可直接用于 APP 备案，不要求先完成网站备案；其受理流程要求实名一致的域名和可备案云资源。

下一步先取得主体省市，核实纯本地工具适用性和受理路径，再决定是否需要注册域名及购买真实服务所需的云资源。

## 上海个人主体准备（2026-09-07）

用户已确认常住上海。阿里云公布的上海个人备案规则要求：证件住所为上海；非上海证件需本地暂住证或居住证，居住证有效期建议超过一个月。身份证应为有效原件，且有效期大于三个月。个人主办者与 App 负责人须为同一人。身份信息应由用户在官方页面填写及核验。

参考：https://help.aliyun.com/zh/icp-filing/icp-filing-regulations-for-shanghai

尚未注册云账号、购买域名/资源或提交备案。先核实上海申报资格及本地 App 受理方式，再推进实际申请。

当前操作进度：已打开阿里云中国站官方登录页（由 beian.aliyun.com 跳转）。等待用户本人注册/登录及个人实名认证。未建立备案订单，未购买任何资源。

最新进度：用户已登录阿里云。已进入免费传统表单 /pcContainer/selfEntity，选择 App、个人、居民身份证，App 名称填写 BetaBuddy。地区必须选到区，尚未完成；姓名、证件号码、证件住所待用户在官方页面填写。尚未点击信息校验、保存草稿或提交订单；账号实名认证状态尚未确认。后续应用技术信息和接入资源页面仍未到达。

## 互联网信息服务表单实际进度

已到第三步，订单号 2037834244915；不是已提交管局的受理号。分类已选休闲娱乐/体育、中文简体，已填真实本地功能说明。iOS 平台条目已保存 Bundle ID、公钥、SHA-1 和用户指定域名 betabuddy.dozycat.ai。曾保存草稿并看到成功提示。

阿里云表单实际校验明确提示：根据备案域名后缀要求，域名 betabuddy.dozycat.ai 不支持备案，请更换符合要求的域名。不能使用当前 .ai 域名推进此申请。图标尚未上传，SDK 尚未核对选项，云资源和备案服务码仍未填写。域名当前 DNS 查询未返回地址。尚未完成本页或提交审核。
