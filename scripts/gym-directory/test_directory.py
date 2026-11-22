"""Regressions for untrusted-source parsing, offline catalog integrity and coordinates.

Everything here is synthetic: no real venue, place name or map answer.
"""
import json
import subprocess
import time
import unittest
import urllib.parse
from fetch_directory import data_array, validate
import geocode_directory as geo


class SourceParsingTests(unittest.TestCase):
    def test_comments_and_strings_do_not_split_records(self):
        source = "var GYMS = [/* ignored { */ {id:'a', name:'literal ]; { text', n:null, flag:true},];"
        self.assertEqual(data_array(source, 'GYMS'), [{'id': 'a', 'name': 'literal ]; { text', 'n': None, 'flag': True}])

    def test_empty_and_independent_arrays(self):
        source = "var GYMS = []; var GYMS_SH = [{id:'sh', nested:[1,2]}];"
        self.assertEqual(data_array(source, 'GYMS'), [])
        self.assertEqual(data_array(source, 'GYMS_SH'), [{'id': 'sh', 'nested': [1, 2]}])

    def test_rejects_executable_expressions(self):
        for value in ["fetch('https://example.org')", "process.exit()", "1 + 2", "function(){return 1}"]:
            with self.assertRaises(ValueError):
                data_array("var GYMS = [{id:" + value + "}];", 'GYMS')

    def test_rejects_missing_source(self):
        with self.assertRaises(ValueError):
            data_array('var NEW_GYMS = [];', 'GYMS')

    def test_rejects_duplicate_ids_and_allows_same_name_branches(self):
        a = dict(id='a', name='Test Gym', province='Province', city='City', address='North', sourceURL='https://example.org/')
        b = dict(a, id='b', address='South')
        self.assertEqual(validate({'version': 1, 'gyms': [a, b]})['City'], 2)
        with self.assertRaises(AssertionError):
            validate({'version': 1, 'gyms': [a, dict(b, id='a')]})

    def test_rejects_unlabelled_coordinates(self):
        gym = dict(id='a', name='Test', province='Province', city='City', address='', sourceURL='https://example.org/')
        validate({'version': 1, 'gyms': [dict(gym, lat=10.5, lon=20.25, coordPrecision='district')]})
        for extra in (dict(latitude=0), dict(lat=10.5, lon=20.25), dict(lat=10.5, lon=20.25, coordPrecision='guess'),
                      dict(lat='10.5', lon=20.25, coordPrecision='poi'), dict(lat=95, lon=20.25, coordPrecision='city')):
            with self.assertRaises(AssertionError):
                validate({'version': 1, 'gyms': [dict(gym, **extra)]})

    def test_rejects_stale_precision_counts(self):
        gym = dict(id='a', name='Test', province='Province', city='City', address='', sourceURL='https://example.org/', lat=1, lon=2, coordPrecision='city')
        counts = {'poi': 0, 'address': 0, 'station': 0, 'district': 0, 'city': 1, 'none': 0}
        validate({'version': 1, 'metadata': {'coordinates': {'counts': counts}}, 'gyms': [gym]})
        with self.assertRaises(AssertionError):
            validate({'version': 1, 'metadata': {'coordinates': {'counts': dict(counts, city=0, poi=1)}}, 'gyms': [gym]})


def place(name, lat, lon, category, kind, display, half=0.0005, osm='way/1'):
    """A search answer in the map service's own shape."""
    return {'osm_type': osm.split('/')[0], 'osm_id': int(osm.split('/')[1]), 'lat': str(lat), 'lon': str(lon), 'category': category, 'type': kind,
            'name': name, 'display_name': display, 'boundingbox': [str(lat - half), str(lat + half), str(lon - half), str(lon + half)]}


class FakeMap:
    """Stands in for the network: an imaginary province with one city, two districts and a few features."""

    def __init__(self):
        self.calls = []
        self.search = {
            'P省': [place('P省', 10, 20, 'boundary', 'administrative', 'P省, 某国', 3, 'relation/1')],
            'C市, P省': [place('C市', 10, 20, 'boundary', 'administrative', 'C市, P省, 某国', 0.5, 'relation/2'), place('C市', 10.01, 20.02, 'place', 'city', 'C市, P省, 某国', osm='node/2')],
            '甲区, C市': [place('甲区', 10.1, 20.1, 'boundary', 'administrative', '甲区, C市, P省, 某国', 0.1, 'relation/3')],
            '乙区, C市': [place('乙区', 9.9, 19.9, 'boundary', 'administrative', '乙区, C市, P省, 某国', 0.1, 'relation/4')],
            '彩虹广场': [place('彩虹广场', 10.12, 20.13, 'shop', 'mall', '彩虹广场, 甲区, C市, P省, 某国', osm='way/5')],
            '环形大楼': [place('环形大楼', 9.91, 19.92, 'building', 'yes', '环形大楼, 乙区, C市, P省, 某国', osm='way/6')],
        }
        self.elements = [
            {'type': 'node', 'id': 7, 'lat': 10.15, 'lon': 20.16, 'tags': {'name': '石头人攀岩馆', 'leisure': 'sports_centre', 'sport': 'climbing'}},
            {'type': 'node', 'id': 8, 'lat': 9.95, 'lon': 19.95, 'tags': {'name': '连锁攀岩', 'leisure': 'sports_centre', 'sport': 'climbing'}},
            {'type': 'node', 'id': 9, 'lat': 10.11, 'lon': 20.11, 'tags': {'name': '石头人崖壁', 'natural': 'cliff', 'sport': 'climbing'}},
        ]
        self.stations = [
            {'type': 'node', 'id': 20, 'lat': 9.93, 'lon': 19.94, 'tags': {'name': '河边', 'railway': 'station'}},
            {'type': 'node', 'id': 21, 'lat': 9.931, 'lon': 19.941, 'tags': {'name': '河边', 'railway': 'station'}},
            {'type': 'node', 'id': 22, 'lat': 10.13, 'lon': 20.14, 'tags': {'name': '双子站', 'railway': 'station'}},
            {'type': 'node', 'id': 23, 'lat': 9.88, 'lon': 19.88, 'tags': {'name': '双子', 'railway': 'station'}},
            {'type': 'node', 'id': 24, 'lat': 10.4, 'lon': 20.4, 'tags': {'name': '远郊', 'railway': 'station'}},
            {'type': 'node', 'id': 25, 'lat': 10.11, 'lon': 20.12, 'tags': {'name': '邻区', 'railway': 'station'}},
        ]
        self.reverse = {('10.15000', '20.16000'): '石头路, 甲区, C市, P省, 某国', ('9.95000', '19.95000'): '某路, 乙区, C市, P省, 某国',
                        ('9.93000', '19.94000'): '河边站, 乙区, C市, P省, 某国', ('10.40000', '20.40000'): '远郊站, 乙区, C市, P省, 某国', ('10.11000', '20.12000'): '邻区站, 甲区, C市, P省, 某国'}

    def __call__(self, url, data=None):
        self.calls.append(url)
        if data is not None:
            return json.dumps({'elements': self.stations if 'railway' in data else self.elements})
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        if '/reverse' in url:
            return json.dumps({'display_name': self.reverse[query['lat'][0], query['lon'][0]]})
        return json.dumps(self.search.get(query['q'][0], []))


def gym(identifier, name, district='', address='', city='C'):
    return {'id': identifier, 'name': name, 'province': 'P省', 'city': city, 'address': address, 'district': district, 'sourceURL': 'https://example.org/'}


class CoordinateTests(unittest.TestCase):
    def resolve(self, gyms, hints=None):
        fake = FakeMap()
        data = {'version': 1, 'metadata': {'notes': []}, 'gyms': gyms}
        geo.apply(data, geo.Cache(None, pause=0, fetch=fake), log=lambda line: None, hints=hints or {})
        return data, fake

    def test_each_gym_gets_the_finest_supported_precision(self):
        data, _ = self.resolve([
            gym('poi', '石头人攀岩', '甲'), gym('mall', '示例攀岩（彩虹广场）', '甲'),
            gym('address', '另一家攀岩馆', '乙区', 'P省C市乙区某某路12号环形大楼3楼另一家攀岩馆'),
            gym('district', '没有线索攀岩', '乙'), gym('city', '区名不存在攀岩', '丙'), gym('none', '城市不存在攀岩', '甲', city='D'),
        ])
        placed = {g['id']: (g.get('coordPrecision'), g.get('lat'), g.get('lon')) for g in data['gyms']}
        self.assertEqual(placed['poi'], ('poi', 10.15, 20.16))
        self.assertEqual(placed['mall'], ('address', 10.12, 20.13))
        self.assertEqual(placed['address'], ('address', 9.91, 19.92))
        self.assertEqual(placed['district'], ('district', 9.9, 19.9))
        self.assertEqual(placed['city'], ('city', 10.01, 20.02))
        self.assertEqual(placed['none'], (None, None, None))
        self.assertNotIn('lat', data['gyms'][-1])
        self.assertEqual(data['metadata']['coordinates']['counts'], {'poi': 1, 'address': 2, 'station': 0, 'district': 1, 'city': 1, 'none': 1})
        self.assertEqual(data['metadata']['coordinates']['datum'], 'WGS-84')

    def test_station_places_a_gym_only_when_it_is_unique_and_near_the_district(self):
        hints = {'a': '河边', 'b': '双子', 'c': '不存在', 'd': '远郊', 'e': '河边', 'f': '河边', 'g': '邻区'}
        data, fake = self.resolve([
            gym('a', '没有线索攀岩', '乙'), gym('b', '同名两站攀岩', '乙'), gym('c', '查无此站攀岩', '乙'), gym('d', '站在远郊攀岩', '乙'),
            gym('e', '示例攀岩（彩虹广场）', '甲'), gym('f', '区名不存在攀岩', '丙'), gym('g', '站在邻区攀岩', '乙'),
        ], hints)
        placed = {g['id']: (g['coordPrecision'], g['lat'], g['lon']) for g in data['gyms']}
        self.assertEqual(placed['a'], ('station', 9.93, 19.94))
        self.assertEqual(placed['b'], ('district', 9.9, 19.9))
        self.assertEqual(placed['c'], ('district', 9.9, 19.9))
        self.assertEqual(placed['d'], ('district', 9.9, 19.9))
        self.assertEqual(placed['e'], ('address', 10.12, 20.13))
        self.assertEqual(placed['f'], ('station', 9.93, 19.94))
        self.assertEqual(placed['g'], ('district', 9.9, 19.9))
        self.assertEqual(data['metadata']['coordinates']['counts'], {'poi': 0, 'address': 1, 'station': 2, 'district': 4, 'city': 0, 'none': 0})
        asked = [call for call in fake.calls if 'interpreter' in call]
        self.assertEqual(len(asked), 2)

    def test_station_query_stays_inside_the_city_and_ignores_odd_names(self):
        query = geo.station_query('relation/2', {'河边', '双子'})
        self.assertIn('area(3600000002)', query)
        self.assertIn('^(双子|河边)站?$', query)
        data, fake = self.resolve([gym('a', '没有线索攀岩', '乙')], {'a': '河边"](area);node'})
        self.assertEqual(data['gyms'][0]['coordPrecision'], 'district')
        self.assertIsNone(geo.station_distance((9.93, 19.94), {'lat': 9.9, 'lon': 19.9, 'box': [9.8, 10.0, 19.8, 20.0]}))
        self.assertIsNone(geo.station_distance((10.25, 19.9), {'lat': 9.9, 'lon': 19.9, 'box': [9.5, 10.3, 19.8, 20.0]}))
        self.assertGreater(geo.station_distance((10.4, 20.4), {'lat': 9.9, 'lon': 19.9, 'box': [9.8, 10.0, 19.8, 20.0]}), 25000)

    def test_chain_branches_do_not_borrow_each_others_poi(self):
        data, _ = self.resolve([gym('a', '连锁攀岩（东门）', '乙'), gym('b', '连锁攀岩（西门）', '乙')])
        self.assertEqual([g['coordPrecision'] for g in data['gyms']], ['district', 'district'])

    def test_poi_in_another_district_is_not_used(self):
        data, _ = self.resolve([gym('a', '石头人攀岩', '乙')])
        self.assertEqual(data['gyms'][0]['coordPrecision'], 'district')

    def test_disagreeing_poi_and_landmark_fall_back_unless_the_street_settles_it(self):
        fake_address = 'P省C市甲区某某路12号彩虹广场3楼'
        data, _ = self.resolve([gym('a', '石头人攀岩', '甲', fake_address)])
        self.assertEqual(data['gyms'][0]['coordPrecision'], 'district')
        data, _ = self.resolve([gym('a', '石头人攀岩', '甲', 'P省C市甲区石头路12号彩虹广场3楼')])
        self.assertEqual((data['gyms'][0]['coordPrecision'], data['gyms'][0]['lat']), ('poi', 10.15))

    def test_district_tagged_as_a_former_boundary_still_counts(self):
        fake = FakeMap()
        fake.search['甲区, C市'][0]['type'] = 'historic'
        hints = None
        data = {'version': 1, 'metadata': {'notes': []}, 'gyms': [gym('a', '没有线索攀岩', '甲')]}
        geo.apply(data, geo.Cache(None, pause=0, fetch=fake), log=lambda line: None, hints=hints or {})
        self.assertEqual(data['gyms'][0]['coordPrecision'], 'district')

    def test_landmark_must_be_compact_unambiguous_and_in_the_district(self):
        mall = geo.trim_places([place('彩虹广场', 10.12, 20.13, 'shop', 'mall', '彩虹广场, 甲区, C市')])
        self.assertEqual(geo.pick_landmark(mall, '彩虹广场', 'C市', '甲区')['name'], '彩虹广场')
        self.assertIsNone(geo.pick_landmark(mall, '彩虹广场', 'C市', '乙区'))
        self.assertIsNone(geo.pick_landmark(mall, '月亮广场', 'C市', '甲区'))
        road = geo.trim_places([place('彩虹广场', 10.12, 20.13, 'highway', 'residential', '彩虹广场, 甲区, C市')])
        self.assertIsNone(geo.pick_landmark(road, '彩虹广场', 'C市', '甲区'))
        campus = geo.trim_places([place('彩虹广场', 10.12, 20.13, 'shop', 'mall', '彩虹广场, 甲区, C市', half=0.01)])
        self.assertIsNone(geo.pick_landmark(campus, '彩虹广场', 'C市', '甲区'))
        tenant = geo.trim_places([place('某餐厅(彩虹广场店)', 10.12, 20.13, 'amenity', 'restaurant', '某餐厅, 甲区, C市')])
        self.assertIsNone(geo.pick_landmark(tenant, '彩虹广场', 'C市', '甲区'))
        self.assertIsNone(geo.pick_landmark(mall, '新彩虹广场', 'C市', '甲区'))
        twins = mall + geo.trim_places([place('彩虹广场', 10.19, 20.19, 'shop', 'mall', '彩虹广场, 甲区, C市')])
        self.assertIsNone(geo.pick_landmark(twins, '彩虹广场', 'C市', '甲区'))

    def test_checks_flag_points_outside_the_province_and_shared_pois(self):
        boxes = {'P省': [7, 13, 17, 23], 'Q省': [30, 40, 30, 40]}
        inside = dict(gym('a', 'A'), lat=10.0, lon=20.0, coordPrecision='poi')
        self.assertEqual(geo.problems([inside, dict(gym('b', 'B'), lat=10.0, lon=20.0, coordPrecision='poi')], boxes), [])
        self.assertEqual(geo.problems([inside, dict(gym('b', 'B', city='D'), lat=10.0, lon=20.0, coordPrecision='district')], boxes), [])
        self.assertEqual(len(geo.problems([inside, dict(gym('b', 'B', city='D'), lat=10.0, lon=20.0, coordPrecision='poi')], boxes)), 1)
        self.assertEqual(len(geo.problems([dict(inside, lat=14.0)], boxes)), 1)
        self.assertEqual(len(geo.problems([dict(inside, province='R省')], boxes)), 1)
        self.assertEqual(len(geo.problems([{k: v for k, v in inside.items() if k != 'coordPrecision'}], boxes)), 1)
        self.assertEqual(geo.counts([inside, gym('b', 'B')]), {'poi': 1, 'address': 0, 'station': 0, 'district': 0, 'city': 0, 'none': 1})

    def test_cache_answers_repeat_questions_and_offline_never_asks(self):
        fake = FakeMap()
        cache = geo.Cache(None, pause=0, fetch=fake)
        first = cache.search('彩虹广场')
        self.assertEqual(cache.search('彩虹广场'), first)
        self.assertEqual(len(fake.calls), 1)
        self.assertNotIn('bounded', fake.calls[0])
        cache.search('彩虹广场', [9, 11, 19, 21])
        self.assertIn('viewbox=19.0000%2C11.0000%2C21.0000%2C9.0000', fake.calls[1])
        offline = geo.Cache(None, offline=True, fetch=fake)
        offline.entries = cache.entries
        self.assertEqual(offline.search('彩虹广场'), first)
        with self.assertRaises(LookupError):
            offline.search('没有问过')
        self.assertEqual(len(fake.calls), 2)

    def test_cache_drops_unused_answers_and_refuses_partial_lists(self):
        fake = FakeMap()
        cache = geo.Cache(None, pause=0, fetch=fake)
        cache.search('彩虹广场')
        cache.entries['nominatim|不再需要'] = {'at': '2000-01-01', 'results': []}
        cache.save(prune=True)
        self.assertEqual(list(cache.entries), ['nominatim|彩虹广场'])
        with self.assertRaises(ValueError):
            geo.trim_elements({'elements': [], 'remark': 'runtime error: Query timed out'})

    def test_requests_are_spaced_and_rerun_asks_nothing(self):
        fake = FakeMap()
        cache = geo.Cache(None, pause=0.05, fetch=fake)
        started = time.monotonic()
        cache.search('彩虹广场')
        cache.search('环形大楼')
        self.assertGreaterEqual(time.monotonic() - started, 0.05)
        gyms = [gym('mall', '示例攀岩（彩虹广场）', '甲')]
        geo.apply({'version': 1, 'metadata': {'notes': []}, 'gyms': gyms}, cache, log=lambda line: None, hints={})
        asked = len(fake.calls)
        geo.apply({'version': 1, 'metadata': {'notes': []}, 'gyms': gyms}, cache, log=lambda line: None, hints={})
        self.assertEqual(len(fake.calls), asked)

    def test_busy_server_is_retried_then_reported(self):
        fake, failures = FakeMap(), [2]

        def flaky(url, data=None):
            if failures[0]:
                failures[0] -= 1
                raise subprocess.CalledProcessError(22, 'curl')
            return fake(url, data)

        cache = geo.Cache(None, pause=0, fetch=flaky, backoff=0)
        self.assertEqual(cache.search('彩虹广场')[0]['name'], '彩虹广场')
        failures[0] = 9
        with self.assertRaises(subprocess.CalledProcessError):
            cache.search('环形大楼')
        self.assertNotIn('nominatim|环形大楼', cache.entries)

    def test_queries_carry_only_names_and_a_generic_client(self):
        self.assertNotRegex(geo.USER_AGENT, r'@|https?:')
        self.assertEqual(geo.landmark_queries(gym('a', '示例攀岩（彩虹广场）', '甲', 'P省C市甲区某某路12号彩虹广场3楼示例攀岩')), ['彩虹广场'])
        self.assertEqual(geo.landmark_queries(gym('a', '示例攀岩（某某路）', '甲')), [])
        self.assertEqual(geo.brand_tokens(gym('a', 'Abcd 石头人攀岩馆（彩虹广场）')), {'石头人'})
        self.assertEqual(geo.brand_tokens(gym('a', 'Abcde Climbing')), {'abcde'})

    def test_distance(self):
        self.assertAlmostEqual(geo.distance_m((0, 0), (0, 1)), 111195, delta=5)


if __name__ == '__main__':
    unittest.main()
