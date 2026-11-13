"""Regressions for untrusted-source parsing and offline catalog integrity."""
import unittest
from fetch_directory import data_array, validate


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

    def test_rejects_estimated_coordinates(self):
        gym = dict(id='a', name='Test', province='Province', city='City', address='', sourceURL='https://example.org/', latitude=0)
        with self.assertRaises(AssertionError):
            validate({'version': 1, 'gyms': [gym]})


if __name__ == '__main__':
    unittest.main()
