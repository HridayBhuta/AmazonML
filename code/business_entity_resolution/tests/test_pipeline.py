"""Unit tests use fabricated records only. They are not challenge performance tests."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pipeline import normalize, Record, Blocker, features, entity_scores, split_queries, parse_ids


def rec(eid, name, address, country='France'):
    import re
    n, a = normalize(name), normalize(address)
    nums = re.findall(r'\d+', a)
    return Record(eid, n, a, normalize(country), frozenset(n.split()),
                  frozenset(a.split()), frozenset(nums), nums[0] if nums else '')


class PipelineTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize('Café & Société'), 'cafe and societe')
        self.assertTrue(normalize('भारत'))

    def test_id_lists(self):
        self.assertEqual(parse_ids(''), [])
        self.assertEqual(parse_ids('S2-A,S3-B'), ['S2-A', 'S3-B'])
        with self.assertRaises(ValueError):
            parse_ids('S2-A,S2-A')
        with self.assertRaises(ValueError):
            parse_ids('S2-A,')

    def test_open_country_and_empty_query(self):
        records = [rec('S2-1', 'Café Soleil', '12 rue verte'), rec('S3-1', 'Pomme Rouge', '30 rue bleue')]
        blocker = Blocker(records, top_k=1, max_postings=128, prefilter=200)
        self.assertEqual(blocker.candidates(rec('S1-1', 'Cafe Soleil', '12 rue verte')), [0])
        self.assertEqual(blocker.candidates(rec('S1-2', '', '')), [])
        self.assertEqual(len(features(records[0], records[1])), 21)

    def test_macro_metric(self):
        # Four entities: correct singleton=1; false merge singleton=0;
        # missing a real match=0; TP2 FP1 with 2 true matches=5/7.
        groups = np.array([1, 3, 3, 3])
        labels = np.array([0, 1, 1, 0])
        probs = np.ones(4)
        sizes = np.array([0, 0, 1, 2])
        score, _, _ = entity_scores(groups, labels, probs, .5, [0, 1, 2, 3], sizes, 4)
        self.assertAlmostEqual(score, (1 + 5/7)/4)

    def test_grouped_split(self):
        queries = [rec(f'S1-{i}', f'Shop {i}', '') for i in range(20)]
        truth = {q.eid: {f'S2-{i}'} for i, q in enumerate(queries)}
        truth['S1-1'] = truth['S1-0']
        valid = split_queries(queries, truth, 42)
        self.assertEqual(0 in valid, 1 in valid)
        self.assertTrue(valid)
        self.assertLess(len(valid), len(queries))


if __name__ == '__main__':
    unittest.main()
