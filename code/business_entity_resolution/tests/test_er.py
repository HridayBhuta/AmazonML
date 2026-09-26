"""Unit tests on fabricated records (not challenge performance tests). Run: python -m unittest discover -s tests"""
import sys
import unittest
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from er import blocking, features  # noqa: E402
from er.text import normalise_frame  # noqa: E402


def frame(rows):
    return pl.DataFrame(rows, schema=["entity_id", "business_name", "business_address", "country"], orient="row")


class TextTests(unittest.TestCase):
    def test_indic_meets_english(self):
        df = normalise_frame(frame([
            ("S2-1", "राम मार्केटिंग प्राइवेट लिमिटेड", "कोलकाता", "India"),
            ("S1-1", "Ram Marketing Private Limited", "Kolkata", "India"),
        ]))
        self.assertEqual(df["name_skel"][0], df["name_skel"][1])
        self.assertEqual(df["addr_skel"][0], df["addr_skel"][1])
        self.assertTrue(df["name_indic"][0])
        self.assertEqual(df["core"][1], "ram marketing")

    def test_legal_forms_and_noise(self):
        df = normalise_frame(frame([
            ("S3-1", "LLC Moncada Léarning Center", "5780 Fawn Ct, Fort Worth, Texas", "US"),
            ("S1-1", "Moncada Learning Center, LLC", "Texas, Fort Worth, 5780 Fawn Court", "US"),
            ("S2-1", "-- wilfordhancock.com", "", "US"),
        ]))
        self.assertEqual(df["core_sorted"][0], df["core_sorted"][1])
        self.assertEqual(df["addr_sorted"][0], df["addr_sorted"][1])
        self.assertEqual(df["house"][0], "5780 faw")
        self.assertEqual(df["core_compact"][2], "wilfordhancock")
        self.assertEqual(df["addr_norm"][2], "")

    def test_open_country_and_france(self):
        df = normalise_frame(frame([
            ("S2-1", "Marina Ecole France Sarl", "63 R. DE DIEPPE, LILLE, Hauts-de-France", "France"),
            ("S1-1", "Marina École France", "63 Rue de Dieppe, Lille", "France"),
            ("S1-2", "Shop", "1 Main St", "Atlantis"),
        ]))
        self.assertIn("rue", df["addr_tok"][0].to_list())
        self.assertEqual(df["core_sorted"][0], df["core_sorted"][1])
        self.assertEqual(df["country_n"][2], "atlantis")


class ReviewRegressionTests(unittest.TestCase):
    def test_indic_names_keep_distinctive_tokens(self):
        df = normalise_frame(frame([
            ("S2-1", "साई ट्रेडर्स", "", "India"),
            ("S2-2", "ओम साई ट्रेडर्स", "", "India"),
            ("S2-3", "मोहन लाल एंड संस", "", "India"),
            ("S2-4", "प्रिया एंटरप्राइजेज प्राइवेट लिमिटेड", "", "India"),
        ]))
        self.assertIn("sai", df["core_tok"][0].to_list())
        self.assertNotEqual(df["core"][0], df["core"][1])
        self.assertIn("lala", df["core_tok"][2].to_list())
        self.assertNotIn("enda", df["core_tok"][2].to_list())
        self.assertIn("priya", df["core_tok"][3].to_list())
        self.assertNotIn("limiteda", df["core_tok"][3].to_list())

    def test_numbered_streets_keep_street_type(self):
        df = normalise_frame(frame([
            ("S1-1", "A", "W 12 St, New York", "US"),
            ("S2-1", "A", "W 12 Street, New York", "US"),
            ("S1-2", "B", "21 st Cross, Bengaluru", "India"),
            ("S1-3", "C", "12 St Marks Road", "India"),
        ]))
        self.assertEqual(df["addr_norm"][0], df["addr_norm"][1])
        self.assertEqual(df["addr_norm"][2], "21 cross bengaluru")
        self.assertEqual(df["house2"][3], "12 marks")

    def test_add_context_numpy_matches_definition(self):
        cut = pl.DataFrame({"idx": [0, 0, 1, 1, 2], "tid": [5, 6, 5, 7, 5], "q": [90.0, 50.0, 80.0, 70.0, 95.0],
                            "qrank": [0, 1, 0, 1, 0]}).with_columns(pl.col("idx", "tid").cast(pl.UInt32),
                                                                    pl.col("q").cast(pl.Float32))
        c = blocking.add_context(cut)
        self.assertEqual(c["t_cnt"].to_list(), [3, 1, 3, 1, 3])
        self.assertEqual(c["t_rank"].to_list(), [2, 1, 3, 1, 1])     # tid 5: 95 (idx2) > 90 > 80
        self.assertEqual(c["t_gap"].to_list(), [5, 0, 15, 0, 0])
        self.assertEqual(c["n_cand"].to_list(), [2, 2, 2, 2, 1])
        self.assertEqual(c["gap_best"].to_list(), [0, 40, 0, 10, 0])
        self.assertEqual(c["gap_2nd"].to_list(), [40, 40, 10, 10, 100])

    def test_conjunctive_keys_exist(self):
        T = normalise_frame(frame([("S2-1", "Prime Money", "12 Elm Street, Austin, TX", "US")])).with_row_index("idx")
        self.assertGreater(blocking.build_keys(T).height, 10)


class BlockingTests(unittest.TestCase):
    def test_retrieves_match_and_handles_empty(self):
        T = normalise_frame(frame([
            ("S2-1", "राम मार्केटिंग प्राइवेट लिमिटेड", "12 MG Road, Kolkata", "India"),
            ("S3-1", "Pomme Rouge", "30 rue bleue", "France"),
            ("S3-2", "Other Business", "99 Elm St", "US"),
        ])).with_row_index("idx")
        Q = normalise_frame(frame([
            ("S1-1", "Ram Marketing Pvt Ltd", "12 M G Rd Kolkata", "India"),
            ("S1-2", "", "", "India"),
        ])).with_row_index("idx")
        idx = blocking.Index(blocking.build_keys(T), T.height, 300)
        qk = idx.query_keys(blocking.build_keys(Q), 1200)
        res = blocking.block_chunk(qk, idx, Q, T, 60, 1)
        self.assertEqual(res.filter(pl.col("idx") == 0)["tid"].to_list()[:1], [0])
        self.assertEqual(res.filter(pl.col("idx") == 1).height, 0)
        C = blocking.add_context(blocking.apply_cut(res, 10, 1e9))
        F = features.compute(C, Q, T, 1)
        self.assertEqual(F.columns[2:], features.FEATURES)
        self.assertFalse(F.select(features.FEATURES).to_numpy().dtype == object)


class MetricTests(unittest.TestCase):
    def test_macro_f05(self):
        # entity 0: singleton, nothing predicted -> 1; entity 1: singleton, false merge -> 0;
        # entity 2: 1 true link missed -> 0; entity 3: TP2 FP1 with 2 true links -> 1.25*2/(0.5+3)=5/7
        idx = np.array([1, 3, 3, 3])
        lab = np.array([0, 1, 1, 0])
        pred = np.ones(4, bool)
        truth = np.array([0, 0, 1, 2], float)
        self.assertAlmostEqual(features.macro_f05(idx, lab, pred, np.arange(4), truth, 4), (1 + 5 / 7) / 4)

    def test_one_to_one(self):
        idx = np.array([0, 1, 1])
        tid = np.array([5, 5, 6])
        p = np.array([0.9, 0.8, 0.7])
        out = features.one_to_one(idx, tid, p, np.ones(3, bool))
        self.assertEqual(out.tolist(), [True, False, True])


if __name__ == "__main__":
    unittest.main()
