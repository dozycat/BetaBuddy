#!/usr/bin/env python3
"""Add labelled WGS-84 coordinates to the offline directory from OpenStreetMap.

Every gym gets the finest position this open, key-less source can support and a
label saying how fine it is: the gym's own POI, the building named in its
address or branch label, the centre of its district, or the centre of its city.
Nothing is guessed below that: an unmatched gym keeps the coarser label.

Requests go to Nominatim (at most one per second) and Overpass with a generic
User-Agent and carry only public gym names, address landmarks and administrative
names. Each answer is stored in data/gym-directory/geocode-cache.json, so a
re-run does not query again; --offline fails instead of touching the network.
Requires Python 3.9+ and curl.
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.parse

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'data/gym-directory/gyms-cn.json'
COPIES = [DIRECTORY, ROOT / 'ios/BetaBuddy/Resources/gyms-cn.json', ROOT / 'android/app/src/main/assets/gyms-cn.json']
CACHE = ROOT / 'data/gym-directory/geocode-cache.json'
MATCHES = ROOT / 'data/gym-directory/coordinate-matches.json'
NOMINATIM = 'https://nominatim.openstreetmap.org/search'
OVERPASS = 'https://overpass-api.de/api/interpreter'
# Generic client name on purpose: no project, person or contact details leave the machine.
USER_AGENT = 'gym-directory-geocoder/1.0'
PRECISIONS = ('poi', 'address', 'district', 'city')
OSM_LICENSE = 'ODbL 1.0'
OSM_ATTRIBUTION = '© OpenStreetMap contributors'
# One query for every climbing-tagged or climbing-named feature in the country.
OVERPASS_QUERY = (
    '[out:json][timeout:120];area["ISO3166-1"="CN"][admin_level=2]->.cn;'
    '(nwr["sport"~"climbing"](area.cn);'
    'nwr["leisure"~"sports_centre|fitness_centre|sports_hall"]["name"~"攀岩|抱石|攀石|岩馆|[Cc]limb|[Bb]oulder"](area.cn);'
    'nwr["name"~"攀岩|抱石"](area.cn););out center tags;'
)
# A landmark only places a gym if the gym is inside it, so it has to be compact.
LANDMARK_MAX_SPAN_M = 600


def distance_m(a, b):
    """Great-circle distance between two (lat, lon) pairs in metres."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371008.8 * math.asin(math.sqrt(h))


class Cache:
    """Answers already received, keyed by the exact question that was asked."""

    def __init__(self, path, offline=False, pause=1.1, fetch=None, log=None):
        self.path, self.offline, self.pause = path, offline, pause
        self.fetch = fetch or self._curl
        self.log = log or (lambda message: None)
        self.entries = json.loads(path.read_text(encoding='utf-8'))['entries'] if path and path.exists() else {}
        self.last = 0.0
        self.asked = 0
        self.used = set()

    def _curl(self, url, data=None):
        command = ['curl', '--fail', '--location', '--silent', '--show-error', '--max-time', '180', '--user-agent', USER_AGENT]
        if data is not None:
            command += ['--data-urlencode', 'data=' + data]
        return subprocess.check_output(command + [url]).decode('utf-8')

    def _ask(self, key, url, data, trim):
        self.used.add(key)
        if key in self.entries:
            return self.entries[key]['results']
        if self.offline:
            raise LookupError('Not in the cache and --offline was given: ' + key)
        wait = self.pause - (time.monotonic() - self.last)
        if wait > 0:
            time.sleep(wait)
        try:
            raw = self.fetch(url, data)
        finally:
            self.last = time.monotonic()
        self.asked += 1
        self.entries[key] = {'at': dt.date.today().isoformat(), 'results': trim(json.loads(raw))}
        self.log('  asked ' + key)
        if self.asked % 20 == 0:
            self.save()
        return self.entries[key]['results']

    def search(self, query, box=None):
        """Nominatim free-text search inside mainland China, optionally inside a box (south, north, west, east)."""
        params = {'q': query, 'format': 'jsonv2', 'limit': '10', 'accept-language': 'zh-CN', 'countrycodes': 'cn'}
        key = 'nominatim|' + query
        if box:
            params.update(viewbox='%.4f,%.4f,%.4f,%.4f' % (box[2], box[1], box[3], box[0]), bounded='1')
            key += '|' + params['viewbox']
        return self._ask(key, NOMINATIM + '?' + urllib.parse.urlencode(params), None, trim_places)

    def reverse(self, lat, lon):
        """Street-level administrative context of a point, used to confirm a candidate's district."""
        params = {'lat': '%.5f' % lat, 'lon': '%.5f' % lon, 'format': 'jsonv2', 'zoom': '16', 'accept-language': 'zh-CN'}
        key = 'nominatim-reverse|%s,%s' % (params['lat'], params['lon'])
        return self._ask(key, NOMINATIM.replace('/search', '/reverse') + '?' + urllib.parse.urlencode(params), None, lambda row: row.get('display_name', ''))

    def overpass(self, query):
        key = 'overpass|' + hashlib.sha256(query.encode()).hexdigest()[:16]
        return self._ask(key, OVERPASS, query, trim_elements)

    def save(self, prune=False):
        """Write the cache; after a complete run, prune drops answers the run no longer asked for."""
        if prune:
            self.entries = {key: entry for key, entry in self.entries.items() if key in self.used}
        if not self.path:
            return
        body = ',\n'.join('%s: %s' % (json.dumps(key, ensure_ascii=False), json.dumps(self.entries[key], ensure_ascii=False, separators=(',', ':'))) for key in sorted(self.entries))
        text = '{"version": 1, "license": "%s", "attribution": "%s", "entries": {\n%s\n}}\n' % (OSM_LICENSE, OSM_ATTRIBUTION, body)
        temporary = self.path.with_suffix('.json.tmp')
        temporary.write_text(text, encoding='utf-8')
        temporary.replace(self.path)


def trim_places(rows):
    """Keep only what matching needs; importance scores, licence strings and icons are dropped."""
    kept = []
    for row in rows:
        south, north, west, east = (round(float(v), 5) for v in row['boundingbox'])
        kept.append({
            'osm': '%s/%s' % (row.get('osm_type', ''), row.get('osm_id', '')),
            'lat': round(float(row['lat']), 6), 'lon': round(float(row['lon']), 6),
            'category': row.get('category', ''), 'type': row.get('type', ''), 'addresstype': row.get('addresstype', ''),
            'name': row.get('name', ''), 'display': row.get('display_name', ''), 'box': [south, north, west, east],
        })
    return kept


def trim_elements(document):
    # Overpass reports a timeout inside an HTTP 200 answer; a partial list must not be cached as complete.
    if document.get('remark'):
        raise ValueError('Overpass did not finish: ' + document['remark'])
    kept = []
    for element in document['elements']:
        tags = element.get('tags', {})
        point = element if 'lat' in element else element.get('center')
        if not point or not tags.get('name'):
            continue
        kept.append({
            'osm': '%s/%s' % (element['type'], element['id']), 'lat': round(point['lat'], 6), 'lon': round(point['lon'], 6),
            'name': tags['name'], 'names': sorted({v for k, v in tags.items() if k.startswith(('name:', 'alt_name', 'old_name', 'short_name', 'official_name', 'brand', 'operator'))}),
            'kind': tags.get('leisure') or tags.get('natural') or tags.get('amenity') or tags.get('shop') or '',
            'indoor': tags.get('indoor', ''), 'climbing': tags.get('climbing', ''),
        })
    return kept


# --- Names -----------------------------------------------------------------

MUNICIPALITIES = ('北京', '上海', '天津', '重庆')
# Explicit readings of source labels that are not the official division name.
DISTRICT_NAMES = {
    ('苏州', '工业园区'): ['苏州工业园区'], ('长沙', '长沙'): ['长沙县'],
    ('廊坊', '燕郊经济开发区'): ['三河市'],
}
GENERIC = re.compile(r'攀岩俱乐部|攀岩中心|攀登中心|攀岩空间|攀岩馆|抱石馆|岩馆|攀岩|攀登|抱石|攀石|俱乐部|climbing|bouldering|boulders?|gym|club', re.I)
CLIMBING_WORD = re.compile(r'攀岩|抱石|攀石|岩馆|攀登')
LANDMARK_END = re.compile(r'购物中心|体育中心|文体中心|健身中心|广场|中心|大厦|体育场|体育馆|产业园|创意园|文创园|科技园|创业园|工业园|百货|商场|天地|天街|小镇|学校|小学|中学|宾馆|公园茂|空间|工场|城|园|汇|里|港|谷')
UNIT = re.compile(r'负?[一二三四五六七八九十]+[层楼]|地下|[东南西北]门|[东南西北]座|[A-Za-z]?\d|[A-Z][区座栋幢]|内|后院|一楼|二楼|三楼|四楼|五楼')
NUMBERED_ROAD = re.compile(r'^.*?(?:路|街|大道|道|巷)[^路街道巷\d]{0,3}\d+(?:-\d+)?[号弄]?(?:之\d+)?')
PURE_ROAD = re.compile(r'(?:路|大道|道|巷|(?<!天)街)$')
ROAD = re.compile(r'^.*(?:路|街|大道|道|巷)(?:[一二三四五六七八九十]段)?(?:\d+(?:-\d+)?[号弄]?)?(?:之\d+)?')
# Features a gym can be inside. Roads, stations, neighbourhoods and parks only say "nearby".
NOT_LANDMARK_CATEGORY = {'highway', 'railway', 'place', 'boundary', 'natural', 'waterway', 'public_transport', 'aeroway', 'route', 'man_made', 'barrier', 'junction'}
NOT_LANDMARK_TYPE = {'park', 'bus_station', 'bus_stop', 'parking', 'parking_entrance', 'bicycle_parking', 'subway_entrance', 'station', 'platform', 'toilets', 'atm'}
NOT_GYM_KIND = {'cliff', 'peak', 'rock', 'cave_entrance', 'stone', 'bare_rock', 'park'}


def norm(text):
    return re.sub(r"[\s·・•\-—_,，.。、/&'’()（）+]+", '', text).lower()


def split_name(name):
    """'品牌（分店）' as the sources write it: brand and branch label."""
    match = re.fullmatch(r'(.+?)（(.+)）', name)
    return (match.group(1).strip(), match.group(2).strip()) if match else (name.strip(), '')


def brand_tokens(gym):
    """Distinctive pieces of the gym's name: two or more Chinese characters, or the Latin brand if five letters or longer."""
    tokens = set()
    for name in [gym['name']] + gym.get('aliases', []):
        brand = GENERIC.sub('', split_name(name)[0])
        pieces = re.findall(r'[\u4e00-\u9fff0-9]*[\u4e00-\u9fff][\u4e00-\u9fff0-9]*', brand) + [norm(''.join(re.findall(r'[A-Za-z0-9 ]+', brand)))]
        tokens.update(piece for piece in pieces if len(piece) >= (5 if piece.isascii() else 2))
    return tokens


def stem(district):
    return re.sub(r'(新区|区|县|市|旗)$', '', district) if len(district) > 2 else district


def strip_region(text, gym, district=''):
    """Drop leading province, city and district words from an address or branch label."""
    province = re.sub(r'(壮族自治区|回族自治区|维吾尔自治区|自治区|省|市)$', '', gym['province'])
    words = [gym['province'], province + '省', province, gym['city'] + '市', gym['city']]
    if district:
        words += [district, stem(district) + '区', stem(district)]
    changed = True
    while changed:
        changed = False
        text = text.lstrip(' ,，')
        for word in words:
            # "长沙县…" names a county; taking the city name off the front would leave "县…".
            if word and text.startswith(word) and len(text) > len(word) and text[len(word)] not in '县区市旗':
                text, changed = text[len(word):], True
    return text


def district_label(gym):
    """The directory's district, or the one written in the address when that field is empty."""
    if gym.get('district'):
        return gym['district']
    match = re.match(r'([^\d\s]{2,6}?(?:区|县|旗|市))', strip_region(gym['address'], gym))
    return match.group(1) if match else ''


def address_roads(gym):
    """Street names written in the address, e.g. to confirm that a candidate point is on that street."""
    text = re.sub(r'[\u4e00-\u9fff]{2,4}?(?:街道|镇|开发区)|[与和]', ' ', strip_region(gym['address'], gym, district_label(gym)))
    return [road for road in re.findall(r'[\u4e00-\u9fff]{2,5}?(?:大道|路|街|巷)', text) if not road.endswith('天街')]


def landmark_queries(gym):
    """Named buildings to look for: what follows the street number in the address, then the branch label."""
    brand, branch = split_name(gym['name'])
    own = sorted(brand_tokens(gym), key=len, reverse=True)
    regions = {gym['province'], gym['city'], gym['city'] + '市'}
    found = []

    def add(text):
        text = text.strip(' ,，-')
        compact = text.replace(' ', '')
        for token in own:
            position = compact.lower().find(token)
            if position >= 0:
                text = compact = compact[:position]
        cut = CLIMBING_WORD.search(text)
        if cut:
            text = text[:cut.start()]
        chinese = len(re.findall(r'[\u4e00-\u9fff]', text))
        if chinese < 3 and len(norm(text)) < 5 or chinese < 2 or PURE_ROAD.search(text) or text in regions or text in found:
            return
        found.append(text)
        latin = re.match(r'[A-Za-z0-9 ]+(?=[\u4e00-\u9fff]{4,})', text)
        if latin:
            add(text[latin.end():])

    district = district_label(gym)
    for segment in re.split(r'[\s,，、()（）]+', strip_region(gym['address'], gym, district)):
        for part in (NUMBERED_ROAD.sub('', segment), ROAD.sub('', segment), segment):
            unit = UNIT.search(part)
            part = part[:unit.start()] if unit else part
            add(part)
            end = LANDMARK_END.search(part)
            if end:
                add(part[:end.end()])
    if branch:
        add(branch)
        add(strip_region(branch, gym, district))
    return found[:6]


# --- Matching --------------------------------------------------------------

def span_m(box):
    return distance_m((box[0], box[2]), (box[1], box[3]))


def in_box(point, box, margin=0.0):
    return box[0] - margin <= point[0] <= box[1] + margin and box[2] - margin <= point[1] <= box[3] + margin


def region(cache, names, parent, box=None):
    """An administrative boundary by exact name inside its parent, with its mapped seat when there is one."""
    for name in names:
        rows = [row for row in cache.search(name + (', ' + parent if parent and parent != name else '')) if row['name'] == name and (not parent or parent in row['display'])]
        if box:
            rows = [row for row in rows if in_box((row['lat'], row['lon']), box, 0.05)]
        # A few current districts are tagged as historic boundaries; an administrative one wins when both exist.
        boundary = next((row for kind in ('administrative', 'historic') for row in rows if row['category'] == 'boundary' and row['type'] == kind), None)
        if not boundary:
            continue
        seat = next((row for row in rows if row['category'] == 'place' and in_box((row['lat'], row['lon']), boundary['box'])), None)
        point = seat or boundary
        return {'name': name, 'osm': point['osm'], 'lat': point['lat'], 'lon': point['lon'], 'box': boundary['box'], 'boundary': boundary['osm']}
    return None


def pick_landmark(rows, query, city_name, district_name):
    """The one compact, named feature that fits the query in the right district; None if absent or ambiguous."""
    wanted = norm(query)
    fits = []
    for row in rows:
        # "某店(某广场店)" is a tenant, and a shorter name ("某中心" for "新某中心") may be another building.
        if wanted not in norm(re.sub(r'[(（].*', '', row['name'])):
            continue
        if row['category'] in NOT_LANDMARK_CATEGORY or row['type'] in NOT_LANDMARK_TYPE:
            continue
        if span_m(row['box']) > LANDMARK_MAX_SPAN_M:
            continue
        if city_name not in row['display'] or (district_name and stem(district_name) not in row['display']):
            continue
        fits.append(row)
    if not fits or any(distance_m((a['lat'], a['lon']), (fits[0]['lat'], fits[0]['lon'])) > LANDMARK_MAX_SPAN_M for a in fits):
        return None
    return fits[0]


def poi_candidates(gym, elements, city):
    tokens = brand_tokens(gym)
    for element in elements:
        if element['kind'] in NOT_GYM_KIND or element['climbing'] or not in_box((element['lat'], element['lon']), city['box']):
            continue
        names = [norm(name) for name in [element['name']] + element['names']]
        shared = [token for token in tokens if any(token in name for name in names)]
        if shared:
            yield element, shared


def resolve(data, cache, log=print):
    """Return ({gym id: match}, {province: box}); a match has precision, lat, lon, the OSM object and what was matched."""
    gyms = data['gyms']
    elements = cache.overpass(OVERPASS_QUERY)
    provinces, cities, districts = {}, {}, {}
    for gym in gyms:
        province, city = gym['province'], gym['city']
        if province not in provinces:
            provinces[province] = region(cache, [province], '')
        if (province, city) not in cities:
            names = [province] if city in MUNICIPALITIES else [city if city.endswith(('自治州', '盟', '地区')) else city + '市']
            cities[province, city] = region(cache, names, '' if city in MUNICIPALITIES else province)
        place = cities[province, city]
        label = district_label(gym)
        if place and label and (province, city, label) not in districts:
            names = DISTRICT_NAMES.get((city, label)) or ([label] if label.endswith(('区', '县', '市', '旗')) else [label + '区', label + '新区', label + '市', label + '县'])
            districts[province, city, label] = region(cache, names, place['name'], place['box'])

    matches, landmarks, coarse = {}, {}, {}
    for gym in gyms:
        place = cities[gym['province'], gym['city']]
        if not place:
            continue
        label = district_label(gym)
        district = districts.get((gym['province'], gym['city'], label))
        if district:
            matches[gym['id']] = {'precision': 'district', 'lat': district['lat'], 'lon': district['lon'], 'osm': district['osm'], 'matched': district['name']}
        else:
            matches[gym['id']] = {'precision': 'city', 'lat': place['lat'], 'lon': place['lon'], 'osm': place['osm'], 'matched': place['name']}
        coarse[gym['id']] = matches[gym['id']]
        for query in landmark_queries(gym):
            row = pick_landmark(cache.search(query, place['box']), query, place['name'], district['name'] if district else '')
            if row:
                landmarks[gym['id']] = row
                matches[gym['id']] = {'precision': 'address', 'lat': row['lat'], 'lon': row['lon'], 'osm': row['osm'], 'matched': row['name'], 'query': query}
                break

    # The gym's own POI. A brand with several branches in one city needs evidence for the branch.
    claims = collections.defaultdict(list)
    for gym in gyms:
        place = cities[gym['province'], gym['city']]
        if not place:
            continue
        label = district_label(gym)
        district = districts.get((gym['province'], gym['city'], label))
        branch = norm(split_name(gym['name'])[1])
        for element, shared in poi_candidates(gym, elements, place):
            point = (element['lat'], element['lon'])
            where = cache.reverse(*point)
            if place['name'] not in where or (district and stem(district['name']) not in where):
                log('  skip    %s: %s %s is in another district (%s)' % (gym['name'], element['osm'], element['name'], where))
                continue
            landmark = landmarks.get(gym['id'])
            near = landmark and distance_m(point, (landmark['lat'], landmark['lon'])) <= LANDMARK_MAX_SPAN_M
            on_road = any(road in where for road in address_roads(gym))
            named = branch and any(piece in norm(element['name'] + where) for piece in {branch, norm(strip_region(split_name(gym['name'])[1], gym, label))} if len(piece) >= 2)
            siblings = [other for other in gyms if other is not gym and other['city'] == gym['city'] and brand_tokens(other) & set(shared)]
            if siblings and not (near or named or on_road):
                log('  skip    %s: %s %s could be another branch' % (gym['name'], element['osm'], element['name']))
                continue
            if landmark and not near and not on_road:
                # Two answers for one gym and nothing to choose between them: keep neither.
                log('  skip    %s: %s %s is %.0f m from %s; falling back to the district' % (gym['name'], element['osm'], element['name'], distance_m(point, (landmark['lat'], landmark['lon'])), landmark['name']))
                matches[gym['id']] = coarse[gym['id']]
                continue
            claims[element['osm']].append((gym, element))
    for osm, claimed in claims.items():
        if len(claimed) > 1:
            log('  skip    %s claimed by %s' % (osm, ' / '.join(gym['name'] for gym, _ in claimed)))
            continue
        gym, element = claimed[0]
        matches[gym['id']] = {'precision': 'poi', 'lat': element['lat'], 'lon': element['lon'], 'osm': osm, 'matched': element['name']}

    for gym in gyms:
        match = matches.get(gym['id'])
        if match:
            match['lat'], match['lon'] = round(match['lat'], 5), round(match['lon'], 5)
        log('%-8s %s %s  %s' % (match['precision'] if match else 'none', gym['city'], gym['name'], '<- %s %s' % (match['osm'], match['matched']) if match else ''))
    boxes = {province: place['box'] for province, place in provinces.items() if place}
    return matches, boxes


# --- Checks and output -----------------------------------------------------

def problems(gyms, province_boxes):
    """Sanity checks on finished coordinates; returns human-readable findings, empty when clean."""
    found = []
    seen = {}
    for gym in gyms:
        fields = [field for field in ('lat', 'lon', 'coordPrecision') if field in gym]
        if not fields:
            continue
        if len(fields) != 3 or gym['coordPrecision'] not in PRECISIONS:
            found.append('%s: lat, lon and coordPrecision must come together with a known precision' % gym['id'])
            continue
        box = province_boxes.get(gym['province'])
        if not box:
            found.append('%s: no bounding box for %s' % (gym['id'], gym['province']))
        elif not in_box((gym['lat'], gym['lon']), box):
            found.append('%s: %.5f, %.5f is outside %s' % (gym['id'], gym['lat'], gym['lon'], gym['province']))
        if gym['coordPrecision'] == 'poi':
            other = seen.setdefault((gym['lat'], gym['lon']), gym)
            if other is not gym and other['city'] != gym['city']:
                found.append('%s and %s share a POI coordinate across cities' % (other['id'], gym['id']))
    return found


def counts(gyms):
    tally = collections.Counter(gym.get('coordPrecision', 'none') for gym in gyms)
    return {key: tally.get(key, 0) for key in PRECISIONS + ('none',)}


def apply(data, cache, log=print):
    """Write lat / lon / coordPrecision into the directory in place and describe them in metadata."""
    matches, boxes = resolve(data, cache, log)
    for gym in data['gyms']:
        for field in ('lat', 'lon', 'coordPrecision'):
            gym.pop(field, None)
        match = matches.get(gym['id'])
        if match:
            gym.update(lat=match['lat'], lon=match['lon'], coordPrecision=match['precision'])
    found = problems(data['gyms'], boxes)
    if found:
        raise ValueError('Coordinate checks failed:\n' + '\n'.join(found))
    metadata = data['metadata']
    metadata['notes'] = [NO_SOURCE_ESTIMATES if note.startswith('不采集评论') else note for note in metadata['notes']]
    metadata['coordinates'] = {
        # The date the map answers were fetched, which a cached re-run does not change.
        'datum': 'WGS-84', 'retrievedAt': max(cache.entries[key]['at'] for key in cache.used), 'counts': counts(data['gyms']),
        'precision': {
            'poi': '岩馆本身在 OpenStreetMap 上的点位',
            'address': '地址或分店名里写明的建筑、商场或园区（跨度不超过 %d 米）在 OpenStreetMap 上的位置，不是门牌点' % LANDMARK_MAX_SPAN_M,
            'district': '所在区县的中心点（OpenStreetMap 的驻地或边界中心），不是岩馆位置',
            'city': '所在城市的中心点，不是岩馆位置',
        },
        'sources': [{'id': 'osm', 'name': 'OpenStreetMap', 'via': ['Nominatim', 'Overpass API'], 'url': 'https://www.openstreetmap.org/copyright', 'license': OSM_LICENSE, 'attribution': OSM_ATTRIBUTION}],
        'notes': ['坐标只来自 OpenStreetMap，本身就是 WGS-84，没有做坐标系转换；没有使用 GCJ-02 或 BD-09 的来源。', '多数岩馆在 OpenStreetMap 上没有点位，只能落到区县或城市中心；这类坐标只能用来判断大致在哪个区，不能排同区内的远近。', '点位未经现场核实；OpenStreetMap 在中国大陆的个别要素可能有几百米偏移或已经过时。'],
    }
    return matches


NO_SOURCE_ESTIMATES = '不采集评论、照片、联系人、价格或路线；不导入来源站点的估计坐标，不使用源中 AI 汇总的馆型与风格。坐标另由 OpenStreetMap 补充并标明精度，见 metadata.coordinates。'


def write_matches(matches, data):
    names = {gym['id']: gym['name'] for gym in data['gyms']}
    rows = {gym_id: dict(match, name=names[gym_id]) for gym_id, match in sorted(matches.items())}
    body = ',\n'.join('  %s: %s' % (json.dumps(key, ensure_ascii=False), json.dumps(row, ensure_ascii=False)) for key, row in rows.items())
    MATCHES.write_text('{\n%s\n}\n' % body, encoding='utf-8')


def write_copies(data, outputs=COPIES):
    payload = (json.dumps(data, ensure_ascii=False, indent=2) + '\n').encode()
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix('.json.tmp')
        temporary.write_bytes(payload)
        temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--offline', action='store_true', help='Use only cached answers; fail on anything not yet asked')
    parser.add_argument('--check', action='store_true', help='Check packaged coordinates against cached province boxes; no network, no writes')
    args = parser.parse_args()
    data = json.loads(DIRECTORY.read_text(encoding='utf-8'))
    cache = Cache(CACHE, offline=args.offline or args.check, log=print)
    if args.check:
        boxes = {}
        for province in sorted({gym['province'] for gym in data['gyms']}):
            place = region(cache, [province], '')
            if place:
                boxes[province] = place['box']
        found = problems(data['gyms'], boxes)
        assert data['metadata']['coordinates']['counts'] == counts(data['gyms']), 'Stale precision counts'
        print(json.dumps(counts(data['gyms']), ensure_ascii=False))
        if found:
            sys.exit('\n'.join(found))
        return
    try:
        matches = apply(data, cache, log=print)
    finally:
        cache.save()
    cache.save(prune=True)
    write_matches(matches, data)
    write_copies(data)
    print('Asked %d new questions; precision counts: %s' % (cache.asked, json.dumps(counts(data['gyms']), ensure_ascii=False)))


if __name__ == '__main__':
    main()
