"""Pair features for the matching model, plus the official per-entity macro F0.5 metric."""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

CONTEXT = ["bscore", "nk", "q", "qrank", "name_ts", "skel_r", "addr_ts", "n_cand", "t_cnt",
           "t_rank", "gap_best", "gap_2nd", "t_gap"]
STRING_FEATS = ["name_ratio", "name_tsort", "core_partial", "core_ratio", "compact_jw",
                "skel_tset", "addr_ratio", "addr_tsort", "addr_skel_tset", "addr_skel_ratio"]
LIST_FEATS = ["jac_core", "jac_skel", "jac_addr", "jac_addr_skel", "jac_num", "num_conflict",
              "first_num_eq", "house_eq", "post_eq", "country_eq", "is_s3", "a_indic", "b_indic",
              "a_ntok", "b_ntok", "name_len_ratio", "addr_len_ratio", "a_addr_missing",
              "b_addr_missing", "mutual_best"]
FEATURES = CONTEXT + STRING_FEATS + LIST_FEATS


def _jac(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    inter = a.list.set_intersection(b).list.len()
    union = a.list.set_union(b).list.len()
    return pl.when(union > 0).then(inter / union).otherwise(-1.0)


def _tri(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return pl.when(a.is_null() | b.is_null()).then(-1.0).when(a == b).then(1.0).otherwise(0.0)


FEATURE_COLS = ["idx", "entity_id", "country_n", "name_indic", "name_norm", "core", "core_sorted", "core_compact",
                "core_tok", "name_skel", "name_skel_tok", "addr_norm", "addr_tok", "addr_skel", "addr_skel_tok",
                "nums", "first_num", "postcodes", "house"]


def compute(P: pl.DataFrame, A: pl.DataFrame, B: pl.DataFrame, workers: int, b_offset: int = 0) -> pl.DataFrame:
    """P: candidate pairs (idx, tid, context columns). A: S1 table with row position = idx.
    B: a block of the target table whose row position = tid - b_offset."""
    ia = P["idx"].to_numpy()
    ib = P["tid"].to_numpy().astype(np.int64) - b_offset

    def g(col):
        return A[col].gather(ia), B[col].gather(ib)

    def cp(col, scorer, missing=-1.0):
        a, b = g(col)
        v = cpdist(a.to_list(), b.to_list(), scorer=scorer, workers=workers, dtype=np.float32)
        empty = ((a.str.len_chars() == 0) | (b.str.len_chars() == 0)).to_numpy()
        return np.where(empty, missing, v).astype(np.float32)

    sf = {
        "name_ratio": cp("name_norm", fuzz.ratio),
        "name_tsort": cp("name_norm", fuzz.token_sort_ratio),
        "core_partial": cp("core", fuzz.partial_ratio),
        "core_ratio": cp("core_sorted", fuzz.ratio),
        "compact_jw": cp("core_compact", JaroWinkler.normalized_similarity),
        "skel_tset": cp("name_skel", fuzz.token_set_ratio),
        "addr_ratio": cp("addr_norm", fuzz.ratio),
        "addr_tsort": cp("addr_norm", fuzz.token_sort_ratio),
        "addr_skel_tset": cp("addr_skel", fuzz.token_set_ratio),
        "addr_skel_ratio": cp("addr_skel", fuzz.ratio),
    }
    cols = {}
    for c in ("core_tok", "name_skel_tok", "addr_tok", "addr_skel_tok", "nums", "postcodes",
              "first_num", "house", "country_n", "name_indic", "name_norm", "addr_norm"):
        a, b = g(c)
        cols["a_" + c], cols["b_" + c] = a, b
    cols["b_eid"] = B["entity_id"].gather(ib)
    L = pl.DataFrame(cols)
    a, b = pl.col, pl.col
    lf = L.select(
        jac_core=_jac(a("a_core_tok"), b("b_core_tok")),
        jac_skel=_jac(a("a_name_skel_tok"), b("b_name_skel_tok")),
        jac_addr=_jac(a("a_addr_tok"), b("b_addr_tok")),
        jac_addr_skel=_jac(a("a_addr_skel_tok"), b("b_addr_skel_tok")),
        jac_num=_jac(a("a_nums"), b("b_nums")),
        num_conflict=((a("a_nums").list.len() > 0) & (b("b_nums").list.len() > 0)
                      & (a("a_nums").list.set_intersection(b("b_nums")).list.len() == 0)).cast(pl.Float32),
        first_num_eq=_tri(a("a_first_num"), b("b_first_num")),
        house_eq=_tri(a("a_house"), b("b_house")),
        post_eq=pl.when((a("a_postcodes").list.len() == 0) | (b("b_postcodes").list.len() == 0)).then(-1.0)
                  .when(a("a_postcodes").list.set_intersection(b("b_postcodes")).list.len() > 0).then(1.0)
                  .otherwise(0.0),
        country_eq=(a("a_country_n") == b("b_country_n")).cast(pl.Float32),
        is_s3=b("b_eid").str.starts_with("S3-").cast(pl.Float32),
        a_indic=a("a_name_indic").cast(pl.Float32),
        b_indic=b("b_name_indic").cast(pl.Float32),
        a_ntok=a("a_core_tok").list.len().cast(pl.Float32),
        b_ntok=b("b_core_tok").list.len().cast(pl.Float32),
        name_len_ratio=pl.min_horizontal(a("a_name_norm").str.len_chars(), b("b_name_norm").str.len_chars())
                       / pl.max_horizontal(a("a_name_norm").str.len_chars(), b("b_name_norm").str.len_chars(), pl.lit(1)),
        addr_len_ratio=pl.min_horizontal(a("a_addr_norm").str.len_chars(), b("b_addr_norm").str.len_chars())
                       / pl.max_horizontal(a("a_addr_norm").str.len_chars(), b("b_addr_norm").str.len_chars(), pl.lit(1)),
        a_addr_missing=(a("a_addr_norm").str.len_chars() == 0).cast(pl.Float32),
        b_addr_missing=(b("b_addr_norm").str.len_chars() == 0).cast(pl.Float32),
    )
    ctx = P.select(["idx", "tid"] + CONTEXT).with_columns(
        mutual_best=((pl.col("qrank") == 0) & (pl.col("t_rank") == 1)).cast(pl.Float32))
    out = ctx.hstack(pl.DataFrame(sf)).hstack(lf)
    return out.select(["idx", "tid"] + [pl.col(f).cast(pl.Float32) for f in FEATURES])


def macro_f05(idx: np.ndarray, label: np.ndarray, pred: np.ndarray, eval_idx: np.ndarray,
              truth_size: np.ndarray, n_total: int) -> float:
    """Per-entity F0.5 averaged over eval_idx (all evaluated S1 entities, incl. singletons and
    entities with no candidates). truth_size[i] counts ALL true links of S1 i, including links
    that blocking missed. Both-empty scores 1."""
    keep = pred.astype(bool)
    pc = np.bincount(idx[keep], minlength=n_total)
    tp = np.bincount(idx[keep], weights=label[keep].astype(np.float64), minlength=n_total)
    e = np.asarray(eval_idx, dtype=np.int64)
    denom = 0.25 * truth_size[e] + pc[e]
    s = np.ones(len(e))
    act = denom > 0
    s[act] = 1.25 * tp[e][act] / denom[act]
    return float(s.mean()) if len(e) else float("nan")


def one_to_one(idx: np.ndarray, tid: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Keep each target only for the Source 1 entity where it scored highest (S1 is deduplicated)."""
    sel = np.flatnonzero(pred)
    if not len(sel):
        return pred.copy()
    order = sel[np.lexsort((idx[sel], -prob[sel], tid[sel]))]
    t = tid[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = t[1:] != t[:-1]
    out = np.zeros_like(pred, dtype=bool)
    out[order[first]] = True
    return out
