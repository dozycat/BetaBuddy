#!/usr/bin/env python3
"""Refresh the offline directory from public pages; no application networking.

Parses only JS data literals, never evaluates downloaded JavaScript. Requires
Python 3.9+ and curl. Use --source-file to reproduce from a saved source response.
"""
import argparse
import ast
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
import hylonomus

ROOT = Path(__file__).resolve().parents[2]
SOURCE_URL = 'https://timeproves.com/data.js?v=150'
PAGE_URL = 'https://timeproves.com/'
# Explicit geographic labels, not inferred from ambiguous gym names.
CITIES = {
    'GYMS': ('北京市', '北京'), 'GYMS_SH': ('上海市', '上海'),
    'GYMS_HZ': ('浙江省', '杭州'), 'GYMS_GZ': ('广东省', '广州'),
    'GYMS_SZ': ('广东省', '深圳'), 'GYMS_CD': ('四川省', '成都'),
    'GYMS_WH': ('湖北省', '武汉'), 'GYMS_XA': ('陕西省', '西安'),
    'GYMS_CS': ('湖南省', '长沙'), 'GYMS_SU': ('江苏省', '苏州'),
    'GYMS_CQ': ('重庆市', '重庆'), 'GYMS_KM': ('云南省', '昆明'),
    'GYMS_LS': ('西藏自治区', '拉萨'), 'GYMS_XG': ('云南省', '香格里拉'),
}
TOKEN = re.compile(r'''\s+|/\*[\s\S]*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[A-Za-z_$][\w$]*|-?\d+(?:\.\d+)?|[\[\]{},:;]''')


def data_array(source, variable):
    """Minimal literal parser rejects expressions and unexpected source changes."""
    match = re.search(r'\bvar\s+' + re.escape(variable) + r'\s*=\s*', source)
    if not match:
        raise ValueError('Missing source array: ' + variable)
    pos = match.end()
    token = None

    def advance():
        nonlocal pos, token
        while True:
            m = TOKEN.match(source, pos)
            if not m:
                raise ValueError('Unexpected token near ' + source[pos:pos + 40])
            pos = m.end()
            token = m.group()
            if token.isspace() or token.startswith(('/*', '//')):
                continue
            return

    def value():
        nonlocal token
        if token == '[':
            result = []
            advance()
            while token != ']':
                result.append(value())
                if token == ',':
                    advance()
                elif token != ']':
                    raise ValueError('Expected array separator')
            advance()
            return result
        if token == '{':
            result = {}
            advance()
            while token != '}':
                key = ast.literal_eval(token) if token[:1] in ('"', "'") else token
                advance()
                if token != ':':
                    raise ValueError('Expected object colon')
                advance()
                result[key] = value()
                if token == ',':
                    advance()
                elif token != '}':
                    raise ValueError('Expected object separator')
            advance()
            return result
        if token[:1] in ('"', "'"):
            result = ast.literal_eval(token)
        elif token in ('null', 'true', 'false'):
            result = {'null': None, 'true': True, 'false': False}[token]
        elif re.fullmatch(r'-?\d+(?:\.\d+)?', token):
            result = float(token) if '.' in token else int(token)
        else:
            raise ValueError('Nonliteral value: ' + token)
        advance()
        return result

    advance()
    return value()


def validate(data):
    assert data['version'] == 1 and data['gyms'], 'Empty or unsupported directory'
    ids = set()
    identities = set()
    for gym in data['gyms']:
        for field in ('id', 'name', 'province', 'city', 'address', 'sourceURL'):
            assert isinstance(gym.get(field), str), (gym.get('id'), field)
        assert gym['name'].strip() and gym['city'].strip() and gym['id'].strip()
        assert gym['id'] not in ids, 'Duplicate ID: ' + gym['id']
        ids.add(gym['id'])
        key = (gym['city'], gym['name'], gym['address'])
        assert key not in identities, 'Duplicate entry: ' + str(key)
        identities.add(key)
        assert gym['sourceURL'].startswith('https://')
        assert 'latitude' not in gym and 'longitude' not in gym, 'No estimated coordinates'
    counts = collections.Counter(g['city'] for g in data['gyms'])
    metadata = data.get('metadata', {})
    if 'gymCount' in metadata:
        assert metadata['gymCount'] == len(data['gyms']), 'Stale gym count'
        assert metadata['cityCount'] == len(counts), 'Stale city count'
        assert metadata['provinceCount'] == len(set(g['province'] for g in data['gyms'])), 'Stale province count'
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-file', type=Path)
    parser.add_argument('--partner-file', type=Path, help='Saved Hylonomus HTML response')
    parser.add_argument('--retrieved-at', default=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).date().isoformat())
    parser.add_argument('--check', action='store_true', help='Validate packaged data without network or writes')
    args = parser.parse_args()
    outputs = [ROOT / 'data/gym-directory/gyms-cn.json', ROOT / 'ios/BetaBuddy/Resources/gyms-cn.json', ROOT / 'android/app/src/main/assets/gyms-cn.json']
    if args.check:
        raw = outputs[0].read_bytes()
        assert all(p.read_bytes() == raw for p in outputs), 'Packaged copies differ'
        cities = validate(json.loads(raw))
        print(json.dumps({'gyms': sum(cities.values()), 'cities': len(cities), 'byCity': cities}, ensure_ascii=False, indent=2))
        return
    raw = args.source_file.read_bytes() if args.source_file else subprocess.check_output(['curl', '--fail', '--location', '--silent', '--show-error', '--max-time', '45', SOURCE_URL])
    source = raw.decode('utf-8')
    gyms = []
    omitted = []
    for variable, (province, city) in CITIES.items():
        for row in data_array(source, variable):
            # Source explicitly says this entry has no corroborating POI; do not ship it.
            if '暂无 POI' in row.get('verify', ''):
                omitted.append(row['id'])
                continue
            name = row['brand'].strip()
            if row.get('branch', '').strip():
                name += '（' + row['branch'].strip() + '）'
            gyms.append({'id': 'timeproves:' + row['id'], 'name': name, 'province': province, 'city': city, 'address': '', 'district': row.get('district', ''), 'sourceURL': PAGE_URL})
    data = {'version': 1, 'updatedAt': args.retrieved_at, 'metadata': {
        'coverage': 'partial', 'scope': '中国岩馆公开名录的已收录条目；非全国全量、非实时营业名单',
        'sources': [{'id': 'timeproves', 'name': '攀岩岩馆大全', 'url': PAGE_URL, 'dataURL': SOURCE_URL, 'retrievedAt': args.retrieved_at, 'sourceUpdatedAt': None, 'sha256': hashlib.sha256(raw).hexdigest(), 'license': '未发现开放数据许可证；源数据不适用本项目 MIT 许可'}],
        'notes': ['timeproves 仅提供馆名、分店和概略行政区；未由 Hylonomus 补充门牌地址的条目，address 留空。', '抓取日期不等于资料核验日期。源数据包含 2026-08 的核验备注，但无全量更新承诺。', '不采集评论、照片、联系人、价格、路线或估计坐标；不使用源中 AI 汇总的馆型与风格。', '未收录的岩馆可以手填；所有条目均未经本项目现场核实。'],
        'omittedSourceIDs': omitted,
    }, 'gyms': sorted(gyms, key=lambda g: (g['province'], g['city'], g['name'], g['id']))}
    partner_raw = args.partner_file.read_bytes() if args.partner_file else subprocess.check_output(['curl', '--fail', '--location', '--silent', '--show-error', '--max-time', '45', hylonomus.SOURCE_URL])
    extra = hylonomus.parse(partner_raw.decode('utf-8'))
    matches = json.loads((ROOT / 'data/gym-directory/branch-matches.json').read_text())
    by_id = {gym['id']: gym for gym in data['gyms']}
    partner_ids = {gym['id'] for gym in extra['gyms']}
    assert set(matches).issubset(partner_ids), 'Source IDs changed; review branch-matches.json'
    province_names = {'北京': '北京市', '上海': '上海市', '天津': '天津市', '重庆': '重庆市', '广西': '广西壮族自治区', '内蒙古': '内蒙古自治区', '新疆': '新疆维吾尔自治区', '宁夏': '宁夏回族自治区', '西藏': '西藏自治区'}
    for gym in extra['gyms']:
        gym['province'] = province_names.get(gym['province'], gym['province'] + '省')
        # Prefer the first-party partner address if sources disagree on district.
        match = re.search(r'(?:市)([^市省区]+区)', gym['address'])
        gym['district'] = match.group(1) if match else ''
        if gym['id'] in matches:
            target = by_id[matches[gym['id']]]
            assert target['city'] == gym['city'], 'Cross-city branch match'
            target['address'] = gym['address']
            target['district'] = gym['district']
            target['sourceURLs'] = [target['sourceURL'], gym['sourceURL']]
            target['sourceURL'] = gym['sourceURL']
            target['aliases'] = [gym['name']]
            target['sourceIDs'] = [target['id'], gym['id']]
        else:
            data['gyms'].append(gym)
    data['metadata']['sources'].append({'id': 'hylonomus', 'name': 'Hylonomus 合作岩馆', 'url': hylonomus.SOURCE_URL, 'retrievedAt': args.retrieved_at, 'sourceUpdatedAt': None, 'sha256': hashlib.sha256(partner_raw).hexdigest(), 'license': '未发现开放数据许可证；源数据不适用本项目 MIT 许可'})
    data['metadata']['mergedBranchCount'] = len(matches)
    data['metadata']['excludedPartnerCount'] = len(extra['excluded'])
    data['gyms'].sort(key=lambda g: (g['province'], g['city'], g['name'], g['id']))
    cities = validate(data)
    data['metadata']['gymCount'] = sum(cities.values())
    data['metadata']['cityCount'] = len(cities)
    data['metadata']['provinceCount'] = len(set(g['province'] for g in data['gyms']))
    data['metadata']['cityCounts'] = dict(sorted(cities.items()))
    payload = (json.dumps(data, ensure_ascii=False, indent=2) + '\n').encode()
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix('.json.tmp')
        temporary.write_bytes(payload)
        temporary.replace(output)
    print('Wrote %d gyms in %d cities to %d copies' % (len(data['gyms']), len(cities), len(outputs)))


if __name__ == '__main__':
    main()
