"""Candidate generation (blocking).

Stage A - bounded inverted index. Each record emits hashed keys, always prefixed with its
country label (an open set; no label is special-cased):
  single keys       t  core-name token          k  consonant skeleton of a core-name token
                    c  first 6 chars of the compact core name   e  last 6 chars of it
                    h  house number + first 3 letters of the next word
                    a  address-token skeleton   p  5-6 digit postcode
  conjunctive keys  (rare by construction, so entities whose single keys are all common still
                    get candidates)
                    n2 pair of core-name tokens (first 4 distinct tokens)
                    na first core token + one of the 3 longest address skeleton tokens
                    nh first core token + first address number
                    np first core token + postcode
                    hs house number + first non-street-type word ("12 marks" for "12 St Marks Rd")
Keys held by more than `max_df` Source 2/3 records are ignored entirely (never truncated to an
arbitrary subset). Per Source 1 record the rarest keys are probed first until the summed
posting-list length exceeds `key_budget`; targets are scored by summed IDF of shared keys and
the best `prelim` are kept.

Stage B - cheap string rerank (no ML) of those targets by
    q = 0.6 * max(token_set(core name), ratio(name skeleton)) + 0.4 * token_set(address)
(address term = 50 when either address is empty).

Stage C - final cut: keep candidates with rank < K and q >= best_q - G, where (K, G) are chosen
on training data (train split only) as the smallest candidate budget losing at most
`recall_tol` link recall, with K <= max_k. The Stage-C list is exactly what the matching model
scores and what is written to candidate_pairs.tsv.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from .text import ADDR_STOP_SKEL

KEY_SEED = 7
KEY_COLS = ["idx", "country_n", "core", "core_tok", "name_skel_tok", "core_compact", "house", "house2",
            "addr_skel_tok", "postcodes", "first_num"]


def _kv(d, kind, v):
    return d.select("idx", "country_n", v=v).with_columns(k=pl.lit(kind))


def build_keys(df: pl.DataFrame, chunk: int = 500_000) -> pl.DataFrame:
    """df: normalised records with KEY_COLS. Returns unique (idx:u32, key:u64)."""
    out = []
    stop = list(ADDR_STOP_SKEL)
    for off in range(0, df.height, chunk):
        d = df.slice(off, chunk).with_columns(
            first_tok=pl.col("core").str.split(" ").list.first(),
            ctoks=pl.col("core").str.split(" ").list.unique(maintain_order=True).list.head(4),
            atoks=pl.col("addr_skel_tok").list.eval(
                pl.element().filter((pl.element().str.len_chars() >= 3) & ~pl.element().is_in(stop))),
        )
        d = d.with_columns(
            atop=pl.col("atoks").list.eval(pl.element().sort_by(pl.element().str.len_chars(), descending=True)).list.head(3))
        cc = pl.col("core_compact")
        pairs = (d.select("idx", "country_n", a=pl.col("ctoks"), b=pl.col("ctoks")).explode("a", empty_as_null=True)
                  .explode("b", empty_as_null=True).filter(pl.col("a") < pl.col("b"))
                  .select("idx", "country_n", v=pl.col("a") + " " + pl.col("b")).with_columns(k=pl.lit("n2")))
        parts = [
            d.select("idx", "country_n", v=pl.col("core_tok")).explode("v", empty_as_null=True)
             .filter(pl.col("v").str.len_chars() >= 2).with_columns(k=pl.lit("t")),
            d.select("idx", "country_n", v=pl.col("name_skel_tok")).explode("v", empty_as_null=True)
             .filter(pl.col("v").str.len_chars() >= 3).with_columns(k=pl.lit("k")),
            _kv(d.filter(cc.str.len_chars() >= 4), "c", cc.str.slice(0, 6)),
            _kv(d.filter(cc.str.len_chars() >= 7), "e", cc.str.slice(-6)),
            _kv(d.filter(pl.col("house").is_not_null()), "h", pl.col("house")),
            _kv(d.filter(pl.col("house2").is_not_null()), "hs", pl.col("house2")),
            d.select("idx", "country_n", v=pl.col("atoks")).explode("v", empty_as_null=True)
             .filter(pl.col("v").is_not_null()).with_columns(k=pl.lit("a")),
            d.select("idx", "country_n", v=pl.col("postcodes")).explode("v", empty_as_null=True)
             .filter(pl.col("v").is_not_null()).with_columns(k=pl.lit("p")),
            pairs,
            d.filter(pl.col("first_tok").str.len_chars() >= 2)
             .select("idx", "country_n", f=pl.col("first_tok"), v=pl.col("atop")).explode("v", empty_as_null=True)
             .filter(pl.col("v").is_not_null()).select("idx", "country_n", v=pl.col("f") + " " + pl.col("v"))
             .with_columns(k=pl.lit("na")),
            _kv(d.filter(pl.col("first_num").is_not_null() & (pl.col("first_tok").str.len_chars() >= 2)), "nh",
                pl.col("first_tok") + " " + pl.col("first_num")),
            d.filter(pl.col("first_tok").str.len_chars() >= 2)
             .select("idx", "country_n", f=pl.col("first_tok"), v=pl.col("postcodes")).explode("v", empty_as_null=True)
             .filter(pl.col("v").is_not_null()).select("idx", "country_n", v=pl.col("f") + " " + pl.col("v"))
             .with_columns(k=pl.lit("np")),
        ]
        kf = (pl.concat([p.select("idx", "k", "country_n", "v") for p in parts])
                .select(pl.col("idx").cast(pl.UInt32),
                        key=pl.concat_str(["k", "country_n", "v"], separator="|").hash(KEY_SEED))
                .unique())
        out.append(kf)
    return pl.concat(out) if out else pl.DataFrame(schema={"idx": pl.UInt32, "key": pl.UInt64})


class Index:
    def __init__(self, target_keys: pl.DataFrame, n_targets: int, max_df: int):
        dfs = target_keys.group_by("key").agg(df=pl.len().cast(pl.UInt32))
        self.n = n_targets
        self.keys_total = dfs.height
        self.dfs = dfs.filter(pl.col("df") <= max_df)
        del dfs
        self.keys_kept = self.dfs.height
        self.tk = (target_keys.join(self.dfs.select("key"), on="key", how="semi")
                   .rename({"idx": "tid"}).sort("key"))

    def query_keys(self, s1_keys: pl.DataFrame, budget: int) -> pl.DataFrame:
        q = s1_keys.join(self.dfs, on="key", how="inner")
        return (q.sort(["idx", "df", "key"])
                 .with_columns(cum=pl.col("df").cum_sum().over("idx"))
                 .filter(pl.col("cum") - pl.col("df") < budget)
                 .with_columns(w=(1.0 + self.n / pl.col("df").cast(pl.Float64)).log().cast(pl.Float32))
                 .select("idx", "key", "w"))


def quick_sims(ia, ib, A: pl.DataFrame, B: pl.DataFrame, workers: int):
    def cp(col, scorer):
        return cpdist(A[col].gather(ia).to_list(), B[col].gather(ib).to_list(), scorer=scorer,
                      workers=workers, dtype=np.float32)
    name_ts = cp("core_sorted", fuzz.token_set_ratio)
    skel_r = cp("name_skel", fuzz.ratio)
    addr_ts = cp("addr_sorted", fuzz.token_set_ratio)
    a_ok = (A["addr_norm"].gather(ia).str.len_chars() > 0).to_numpy()
    b_ok = (B["addr_norm"].gather(ib).str.len_chars() > 0).to_numpy()
    both = a_ok & b_ok
    addr_ts = np.where(both, addr_ts, -1.0).astype(np.float32)
    q = 0.6 * np.maximum(name_ts, skel_r) + 0.4 * np.where(both, addr_ts, 50.0)
    return name_ts, skel_r, addr_ts, q.astype(np.float32)


def block_chunk(qk: pl.DataFrame, index: Index, A: pl.DataFrame, B: pl.DataFrame,
                prelim: int, workers: int) -> pl.DataFrame:
    """qk: (idx, key, w) for a chunk of S1 records. A/B are indexed by row position = idx/tid."""
    sub = index.tk.filter(pl.col("key").is_in(qk["key"].unique().implode()))
    pairs = qk.join(sub, on="key", how="inner")
    agg = pairs.group_by(["idx", "tid"]).agg(bscore=pl.col("w").sum(), nk=pl.len().cast(pl.UInt16))
    del pairs, sub
    agg = (agg.sort(["idx", "bscore", "tid"], descending=[False, True, False])
              .with_columns(r=pl.int_range(pl.len()).over("idx"))
              .filter(pl.col("r") < prelim).drop("r"))
    if agg.height == 0:
        return agg.with_columns([pl.lit(None, pl.Float32).alias(c) for c in ("name_ts", "skel_r", "addr_ts", "q")]
                                + [pl.lit(None, pl.UInt16).alias("qrank")])
    ia, ib = agg["idx"].to_numpy(), agg["tid"].to_numpy()
    name_ts, skel_r, addr_ts, q = quick_sims(ia, ib, A, B, workers)
    agg = agg.with_columns(name_ts=pl.Series(name_ts), skel_r=pl.Series(skel_r),
                           addr_ts=pl.Series(addr_ts), q=pl.Series(q))
    return (agg.sort(["idx", "q", "bscore", "tid"], descending=[False, True, True, False])
               .with_columns(qrank=pl.int_range(pl.len()).over("idx").cast(pl.UInt16)))


def choose_cut(prelim: pl.DataFrame, truth_sizes: pl.DataFrame, tol: float, max_k: int, log) -> dict:
    """prelim: idx, qrank, q, label (bool) for a sample of TRAIN-split S1; truth_sizes: idx, n_true
    for the same sample (every sampled S1, including singletons)."""
    p = prelim.with_columns(gap=pl.col("q").max().over("idx") - pl.col("q"))
    total_true = int(truth_sizes["n_true"].sum())
    n_s1 = max(1, truth_sizes.height)
    kmax = min(max_k, int(p["qrank"].max()) + 1 if p.height else 1)
    ks = sorted({k for k in [3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, kmax] if k <= kmax})
    rows = []
    for k in ks:
        for g in [1e9, 40, 30, 25, 20, 15]:
            sel = p.filter((pl.col("qrank") < k) & (pl.col("gap") <= g))
            rows.append({"K": k, "G": g, "link_recall": int(sel["label"].sum()) / total_true if total_true else 1.0,
                         "avg_candidates": sel.height / n_s1})
    grid = pl.DataFrame(rows)
    best = grid["link_recall"].max()
    pick = grid.filter(pl.col("link_recall") >= best - tol).sort(["avg_candidates", "K"]).row(0, named=True)
    log(f"cut grid best recall (K<={kmax}) {best:.4f}; chosen K={pick['K']} G={pick['G']} "
        f"recall={pick['link_recall']:.4f} avg_cand={pick['avg_candidates']:.2f}")
    return {"K": int(pick["K"]), "G": float(pick["G"]), "grid": grid, "max_recall": float(best)}


def apply_cut(prelim: pl.DataFrame, K: int, G: float) -> pl.DataFrame:
    return (prelim.with_columns(gap=pl.col("q").max().over("idx") - pl.col("q"))
                  .filter((pl.col("qrank") < K) & (pl.col("gap") <= G)).drop("gap"))


def add_context(cut: pl.DataFrame) -> pl.DataFrame:
    """Competition features over the whole split's final candidate set, computed with numpy
    (about 30 bytes per pair). Input must be sorted by (idx, qrank)."""
    idx = cut["idx"].to_numpy()
    tid = cut["tid"].to_numpy()
    q = cut["q"].to_numpy().astype(np.float32)
    n = len(q)
    # target side: max q, count, and this pair's rank among all S1 that hold the target
    o = np.lexsort((-q, tid))
    ts = tid[o]
    start = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]]) if n else np.array([], dtype=np.int64)
    grp = np.repeat(start, np.diff(np.r_[start, n])) if n else np.array([], dtype=np.int64)
    t_rank = np.empty(n, np.float32)
    t_rank[o] = (np.arange(n) - grp + 1).astype(np.float32)
    t_max = np.empty(n, np.float32)
    t_max[o] = q[o][grp]
    t_cnt = np.bincount(tid)[tid].astype(np.float32) if n else np.array([], np.float32)
    del o, ts, grp
    # S1 side (rows are grouped by idx with q descending)
    istart = np.flatnonzero(np.r_[True, idx[1:] != idx[:-1]]) if n else np.array([], dtype=np.int64)
    ilen = np.diff(np.r_[istart, n])
    igrp = np.repeat(istart, ilen)
    q_best = q[igrp]
    second = np.where(ilen > 1, q[np.minimum(istart + 1, n - 1)], np.nan).astype(np.float32)
    q_2nd = np.repeat(second, ilen)
    n_cand = np.repeat(ilen, ilen).astype(np.float32)
    return cut.with_columns(
        n_cand=pl.Series(n_cand), t_cnt=pl.Series(t_cnt), t_rank=pl.Series(t_rank),
        gap_best=pl.Series(q_best - q), gap_2nd=pl.Series(np.nan_to_num(q_best - q_2nd, nan=100.0)),
        t_gap=pl.Series(t_max - q))
