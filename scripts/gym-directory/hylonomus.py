#!/usr/bin/env python3
"""Read factual venue listings from Hylonomus's public partner page.

This is a secondary source of catalog entries, not a claim of current operation.
No marketing copy, contact details, photos or coordinates are copied.
"""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

SOURCE_URL = "https://www.hylonomus.com.cn/zh/contact-us-dealership/"


class PartnerParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tag = None
        self.parts = []
        self.section = ""
        self.name = ""
        self.active = False
        self.rows = []

    def handle_starttag(self, tag, attrs):
        if tag in {"h2", "h3", "h4", "p"}:
            self.tag, self.parts = tag, []

    def handle_data(self, data):
        if self.tag:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag != self.tag:
            return
        value = " ".join("".join(self.parts).split())
        if tag == "h2":
            self.active = value == "合作岩馆"
        elif self.active and tag == "h3":
            self.section, self.name = value, ""
        elif self.active and tag == "h4":
            self.name = value
        elif self.active and tag == "p" and self.name:
            self.rows.append((self.section, self.name, value))
            self.name = ""
        self.tag = None


def parse(source):
    parser = PartnerParser()
    parser.feed(source)
    gyms, excluded = [], []
    municipalities = {"北京", "上海", "天津", "重庆"}
    for section, name, address in parser.rows:
        # Retail fitting points and overseas dealers are outside this catalog.
        if section == "马来西亚" or "试鞋点" in name:
            excluded.append({"name": name, "reason": "overseas or retail fitting point"})
            continue
        province = section.removeprefix("中国")
        if province == "Hei Long Jiang, China":
            province = "黑龙江"
        if province == "其他城市":
            match = re.match(r"(.+?)省", address)
            province = match.group(1) if match else ("新疆" if address.startswith("新疆") else "")
        if province in municipalities:
            city = province
        else:
            remainder = re.sub(r"^(?:" + re.escape(province) + r")(?:壮族自治区|回族自治区|维吾尔自治区|自治区|省)?", "", address)
            if remainder.startswith("湘西土家族苗族自治州"):
                city = "湘西土家族苗族自治州"
            elif remainder.startswith("乌鲁木齐"):
                city = "乌鲁木齐"
            else:
                match = re.match(r"(.+?)市", remainder)
                city = match.group(1) if match else ""
        if not province or not city or not address:
            raise ValueError(f"Cannot determine administrative region for {name}: {address}")
        # Address distinguishes branches with the same official heading.
        identity = "|".join([province, city, name, address])
        gyms.append({
            "id": "hylonomus-" + hashlib.sha256(identity.encode()).hexdigest()[:16],
            "name": name, "province": province, "city": city,
            "address": address, "sourceURL": SOURCE_URL,
        })
    if len(gyms) < 50:
        raise ValueError("Unexpectedly short partner directory; inspect the upstream page")
    return {"gyms": gyms, "excluded": excluded}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--html", type=Path, help="Parse a previously fetched HTML snapshot")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.html:
        source = args.html.read_text(encoding="utf-8")
    else:
        request = Request(SOURCE_URL, headers={"User-Agent": "BetaBuddy-GymDirectory/1.0"})
        with urlopen(request, timeout=45) as response:
            source = response.read().decode("utf-8")
    result = parse(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Read {len(result['gyms'])} partner venues; excluded {len(result['excluded'])} out-of-scope entries")


if __name__ == "__main__":
    main()
