"""End-to-end stages. Every stage is resumable. Cache folders are keyed by the relevant settings AND
by a fingerprint of the code that produced them, chained stage to stage, so editing the code or
changing an earlier stage automatically invalidates everything downstream."""
from __future__ import annotations

import gc
import hashlib
import inspect
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from . import blocking, common, features, text
from .common import OFFICIAL_SHA256, Config, Log, Stages, config_hash, read_links, read_source, sha256, total_ram_gb
from .text import normalise_frame

SPLIT_BUCKETS = {"train": (0, 70), "valid": (70, 85), "holdout": (85, 100)}
PART_ROWS = 500_000
SIM_COLS = ["idx", "core_sorted", "name_skel", "addr_sorted", "addr_norm"]


def _fp(*objs) -> str:
    h = hashlib.sha1()
    for o in objs:
        h.update((inspect.getsource(o) if not isinstance(o, str) else o).encode())
    return h.hexdigest()[:10]


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg.resolve()
        c = self.cfg
        self.data = Path(c.data)
        self.run = Path(c.run)
        self.work = self.run / "work"
        self.out = self.run / "output"
        self.cache = Path(c.cache)
        for d in (self.work, self.out, self.run / "artifacts"):
            d.mkdir(parents=True, exist_ok=True)
        self.log = Log(self.run / "pipeline.log")
        d = c.dump()
        (self.run / "artifacts" / "config.json").write_text(json.dumps(d, indent=2))
        data_fp = json.dumps([[r, (self.data / r).stat().st_size] for r in sorted(OFFICIAL_SHA256)
                              if (self.data / r).is_file()])
        self.k_norm = _fp(text, common, Pipeline.prepare, data_fp, f"smoke={c.smoke}")
        self.k_block = config_hash({"n": self.k_norm, "code": _fp(blocking, Pipeline.block), **d},
                                   ["n", "code", "max_df", "key_budget", "prelim", "chunk"])
        self.k_feat = config_hash({"b": self.k_block, "code": _fp(features, Pipeline._cut, Pipeline.featurize,
                                                                  Pipeline._split_table), **d},
                                  ["b", "code", "recall_tol", "max_k", "train_s1", "valid_s1", "holdout_s1",
                                   "tune_s1", "seed", "b_block"])
        self.k_model = config_hash({"f": self.k_feat, "code": _fp(Pipeline.train, Pipeline.predict,
                                                                  Pipeline._load, Pipeline.write_outputs), **d},
                                   ["f", "code", "rounds", "early_stop", "lr", "leaves", "min_leaf",
                                    "feature_fraction", "drop_features", "decision", "refit_all", "seed"])
        self.norm = self.cache / f"norm_{self.k_norm}"
        self.bdir = self.cache / f"block_{self.k_block}"
        self.fdir = self.cache / f"feat_{self.k_feat}"
        self.st_norm = Stages(self.norm, self.k_norm)
        self.st_block = Stages(self.bdir, self.k_block)
        self.st_feat = Stages(self.fdir, self.k_feat)
        self.st_model = Stages(self.work, self.k_model)
        drop = {x.strip() for x in c.drop_features.split(",") if x.strip()}
        unknown = drop - set(features.FEATURES)
        if unknown:
            raise ValueError(f"unknown features in --drop-features: {sorted(unknown)}")
        if c.decision not in ("threshold", "one_to_one"):
            raise ValueError("--decision must be 'threshold' or 'one_to_one'")
        self.feats = [f for f in features.FEATURES if f not in drop]
        self.times = {}
        self.log(f"config: {json.dumps(d)}")
        self.log(f"caches: norm={self.norm.name} block={self.bdir.name} feat={self.fdir.name}")

    # ------------------------------------------------------------------ table helpers
    def _dir(self, split, which):
        return self.norm / f"{split}_{which}"

    def table(self, split, which, columns, lo=None, hi=None) -> pl.DataFrame:
        lf = pl.scan_parquet(self._dir(split, which) / "*.parquet")
        if lo is not None:
            lf = lf.filter(pl.col("idx").is_between(lo, hi - 1))
        df = lf.select(columns).collect().sort("idx")
        start = 0 if lo is None else lo
        if df.height and (df["idx"][0] != start or df["idx"][-1] != start + df.height - 1):
            raise RuntimeError(f"{split}_{which}: idx not contiguous; delete {self.norm} and rerun")
        return df

    def n_rows(self, split, which) -> int:
        return pl.scan_parquet(self._dir(split, which) / "*.parquet").select(pl.len()).collect().item()

    def _timed(self, name, fn, *a):
        t = time.perf_counter()
        r = fn(*a)
        self.times[name] = round(time.perf_counter() - t, 1)
        return r

    # ------------------------------------------------------------------ prepare
    def prepare(self):
        if self.st_norm.done("prepare"):
            self.log("prepare: already done")
            return
        norm = self.norm
        norm.mkdir(parents=True, exist_ok=True)
        if not self.st_norm.done("hashes"):
            files = {}
            for rel, h in OFFICIAL_SHA256.items():
                p = self.data / rel
                got = sha256(p) if p.is_file() else None
                files[rel] = {"sha256": got, "official": got == h}
            bad = [r for r, v in files.items() if not v["official"]]
            if bad and not self.cfg.smoke and not self.cfg.allow_unofficial:
                raise SystemExit(f"These files do not match the official SHA-256: {bad}. The data is probably "
                                 f"truncated or modified: delete it (and the data/ folder) and re-extract the "
                                 f"official zip, or pass --allow-unofficial if this is intentional.")
            (norm / "hashes.json").write_text(json.dumps(files, indent=2))
            self.st_norm.mark("hashes")
        for split in ("train", "test"):
            offsets = {"A": 0, "B": 0}
            for s in (1, 2, 3):
                which = "A" if s == 1 else "B"
                tag = f"src_{split}_{s}"
                meta_f = norm / f"{tag}.json"
                if self.st_norm.done(tag) and meta_f.is_file():
                    meta = json.loads(meta_f.read_text())
                    offsets[which] = meta["end"]
                    continue
                path = self.data / "dataset" / split / f"{split}_source{s}.tsv"
                df = read_source(path, f"S{s}")
                meta = {
                    "rows": df.height, "start": offsets[which],
                    "blank_name": int((df["business_name"].str.strip_chars() == "").sum()),
                    "blank_address": int((df["business_address"].str.strip_chars() == "").sum()),
                    "countries": dict(df.group_by("country").len().sort("len", descending=True).head(12).iter_rows()),
                }
                self.log(f"read {path.name}: {df.height:,} rows")
                d = self._dir(split, which)
                d.mkdir(parents=True, exist_ok=True)
                for old in d.glob(f"part_{s}_*.parquet"):
                    old.unlink()
                for k, off in enumerate(range(0, df.height, PART_ROWS)):
                    part = normalise_frame(df.slice(off, PART_ROWS))
                    base = offsets[which]
                    part = part.with_columns(idx=pl.int_range(base, base + part.height, dtype=pl.UInt32)).select(
                        ["idx"] + [c for c in part.columns if c != "idx"])
                    part.write_parquet(d / f"part_{s}_{k:05d}.parquet")
                    offsets[which] += part.height
                meta["end"] = offsets[which]
                meta_f.write_text(json.dumps(meta))
                self.st_norm.mark(tag)
                del df
                gc.collect()
                self.log(f"normalised {path.name}")
        A = self.table("train", "A", ["idx", "entity_id", "country_n"])
        B = self.table("train", "B", ["idx", "entity_id"])
        gt, links = read_links(self.data / "dataset" / "train" / "train_ground_truth.tsv")
        if gt["s1"].n_unique() != gt.height:
            raise ValueError("train_ground_truth.tsv has duplicate source1_entity_id rows")
        if gt.join(A, left_on="s1", right_on="entity_id", how="anti").height or \
           A.join(gt, left_on="entity_id", right_on="s1", how="anti").height:
            raise ValueError("train_ground_truth.tsv must list every training Source 1 entity exactly once")
        L = (links.join(A.select(s1="entity_id", idx="idx"), on="s1", how="inner")
                  .join(B.select(t="entity_id", tid="idx"), on="t", how="left"))
        missing = int(L["tid"].null_count())
        if missing:
            self.log(f"WARNING: {missing} ground-truth targets are not in train S2/S3 (kept in truth counts)")
        truth = A.select("idx").join(L.group_by("idx").agg(n_true=pl.len().cast(pl.UInt32)), on="idx", how="left") \
                 .fill_null(0)
        L.filter(pl.col("tid").is_not_null()).select(pl.col("idx").cast(pl.UInt32), pl.col("tid").cast(pl.UInt32)) \
         .write_parquet(norm / "train_links.parquet")
        truth.write_parquet(norm / "train_truth.parquet")
        multi = L.group_by("tid").agg(k=pl.col("idx").n_unique()).filter(pl.col("k") > 1).height
        hist = truth.group_by("n_true").len().sort("n_true")
        audit = {"files": json.loads((norm / "hashes.json").read_text())}
        for split in ("train", "test"):
            for s in (1, 2, 3):
                audit[f"{split}_source{s}"] = json.loads((norm / f"src_{split}_{s}.json").read_text())
        audit["ground_truth"] = {
            "s1": gt.height, "links": L.height, "singletons": int((truth["n_true"] == 0).sum()),
            "singleton_fraction": float((truth["n_true"] == 0).mean()),
            "targets_linked_to_multiple_s1": multi,
            "matches_per_s1_hist": {int(k): int(v) for k, v in hist.iter_rows()},
            "singleton_fraction_by_country": dict(truth.join(A, on="idx").group_by("country_n")
                                                  .agg((pl.col("n_true") == 0).mean()).iter_rows()),
        }
        (norm / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
        self.log("audit: " + json.dumps(audit["ground_truth"]))
        self.st_norm.mark("prepare")

    # ------------------------------------------------------------------ blocking
    def block(self, split: str):
        if self.st_block.done(f"block_{split}"):
            self.log(f"block {split}: already done")
            return
        nA, nB = self.n_rows(split, "A"), self.n_rows(split, "B")
        self.log(f"block {split}: {nA:,} S1 vs {nB:,} targets")
        tk = pl.concat([blocking.build_keys(pl.read_parquet(f, columns=blocking.KEY_COLS))
                        for f in sorted(self._dir(split, "B").glob("*.parquet"))])
        index = blocking.Index(tk, nB, self.cfg.max_df)
        del tk
        gc.collect()
        self.log(f"index: {index.keys_total:,} keys, {index.keys_kept:,} with df<={self.cfg.max_df}, "
                 f"{index.tk.height:,} postings")
        sk = pl.concat([blocking.build_keys(pl.read_parquet(f, columns=blocking.KEY_COLS))
                        for f in sorted(self._dir(split, "A").glob("*.parquet"))])
        qk = index.query_keys(sk, self.cfg.key_budget)
        del sk
        A = self.table(split, "A", SIM_COLS)
        B = self.table(split, "B", SIM_COLS)
        pdir = self.bdir / split / "prelim"
        pdir.mkdir(parents=True, exist_ok=True)
        for lo in range(0, nA, self.cfg.chunk):
            hi = min(nA, lo + self.cfg.chunk)
            f = pdir / f"part_{lo:09d}_{hi:09d}.parquet"
            if f.is_file():
                continue
            ck = qk.filter(pl.col("idx").is_between(lo, hi - 1))
            res = blocking.block_chunk(ck, index, A, B, self.cfg.prelim, self.cfg.threads)
            tmp = f.with_suffix(".tmp")
            res.write_parquet(tmp)
            tmp.replace(f)
            self.log(f"block {split}: {hi:,}/{nA:,} S1 (chunk pairs {res.height:,})")
        self.st_block.mark(f"block_{split}")
        del index, qk, A, B
        gc.collect()

    def _prelim_parts(self, split):
        n = self.n_rows(split, "A")
        parts = [self.bdir / split / "prelim" / f"part_{lo:09d}_{min(n, lo + self.cfg.chunk):09d}.parquet"
                 for lo in range(0, n, self.cfg.chunk)]
        missing = [p.name for p in parts if not p.is_file()]
        if missing:
            raise RuntimeError(f"blocking output incomplete for {split}: {missing[:3]}")
        return parts

    # ------------------------------------------------------------------ splits & cut
    def _split_table(self) -> pl.DataFrame:
        truth = pl.read_parquet(self.norm / "train_truth.parquet")
        L = pl.read_parquet(self.norm / "train_links.parquet")
        # S1 entities sharing a labelled target stay in the same group (one hop; S1 is deduplicated,
        # so shared targets are rare - the audit reports how many).
        grp = L.join(L.group_by("tid").agg(g=pl.col("idx").min()), on="tid").group_by("idx").agg(g=pl.col("g").min())
        t = truth.join(grp, on="idx", how="left").with_columns(g=pl.coalesce("g", "idx").cast(pl.UInt64))
        seed = self.cfg.seed
        t = t.with_columns(
            bucket=((pl.col("g") * 2654435761 + seed * 97 + 12345) % 4294967291 % 100),
            order=((pl.col("idx").cast(pl.UInt64) * 40503 + seed) % 1000003))
        caps = {"train": self.cfg.train_s1, "valid": self.cfg.valid_s1, "holdout": self.cfg.holdout_s1}
        parts = []
        for name, (lo, hi) in SPLIT_BUCKETS.items():
            s = t.filter(pl.col("bucket").is_between(lo, hi - 1)).sort("order")
            parts.append(s.with_columns(split=pl.lit(name), used=pl.int_range(pl.len()) < caps[name]))
        return pl.concat(parts).select("idx", "n_true", "split", "used")

    def cut(self):
        self.fdir.mkdir(parents=True, exist_ok=True)
        if not self.st_feat.done("cut"):
            self._cut()
        for f in ("candidates.json", "cut_grid.tsv"):
            shutil.copy2(self.fdir / f, self.run / "artifacts" / f)

    def _cut(self):
        S = self._split_table()
        S.write_parquet(self.fdir / "train_splits.parquet")
        L = pl.read_parquet(self.norm / "train_links.parquet").with_columns(label=pl.lit(True))
        truth = pl.read_parquet(self.norm / "train_truth.parquet")
        # tune the budget on TRAIN-split S1 only (never on valid/holdout labels)
        pool = S.filter(pl.col("split") == "train")
        step = max(1, pool.height // max(1, self.cfg.tune_s1))
        sample = pool.filter((pl.col("idx") % step) == 0).select("idx", "n_true")
        pre = (pl.scan_parquet(self._prelim_parts("train"))
                 .join(sample.lazy().select("idx"), on="idx", how="semi")
                 .select("idx", "tid", "qrank", "q").collect())
        pre = pre.join(L, on=["idx", "tid"], how="left").with_columns(pl.col("label").fill_null(False))
        pick = blocking.choose_cut(pre, sample, self.cfg.recall_tol, self.cfg.max_k, self.log)
        pick["grid"].write_csv(self.fdir / "cut_grid.tsv", separator="\t")
        info = {"K": pick["K"], "G": pick["G"], "max_prelim_recall_at_max_k": pick["max_recall"],
                "tune_sample_s1": sample.height, "tuned_on": "train split only"}
        del pre
        for split in ("train", "test"):
            C = pl.concat([blocking.apply_cut(pl.read_parquet(f), pick["K"], pick["G"])
                           for f in self._prelim_parts(split)])
            if C.select(pl.struct("idx", "tid").n_unique()).item() != C.height:
                raise RuntimeError(f"duplicate (idx, tid) pairs in {split} candidates")
            C = blocking.add_context(C)
            C.write_parquet(self.fdir / f"{split}_cand.parquet")
            A = self.table(split, "A", ["idx", "country_n"])
            nt = self.n_rows(split, "B")
            per = C.group_by("idx").len()
            cov = (A.join(per, on="idx", how="left").group_by("country_n")
                     .agg(s1=pl.len(), no_candidates=pl.col("len").is_null().sum(),
                          avg_candidates=pl.col("len").fill_null(0).mean()))
            stats = {"s1": A.height, "targets": nt, "pairs": C.height, "avg_candidates": C.height / A.height,
                     "max_candidates": int(per["len"].max() or 0), "s1_with_no_candidates": A.height - per.height,
                     "reduction_ratio": 1 - C.height / (A.height * nt),
                     "by_country": {r["country_n"]: {"s1": r["s1"], "no_candidates": r["no_candidates"],
                                                     "no_candidates_share": r["no_candidates"] / r["s1"],
                                                     "avg_candidates": r["avg_candidates"]}
                                    for r in cov.iter_rows(named=True)}}
            for cn, v in stats["by_country"].items():
                if v["no_candidates_share"] > 0.01:
                    self.log(f"WARNING: {split} country '{cn}': {v['no_candidates_share']:.1%} of S1 have NO "
                             f"candidates (they score 0 unless singletons) - consider more key families / max_df")
            if split == "train":
                hits = C.select("idx", "tid").join(L, on=["idx", "tid"], how="inner")
                tot = int(truth["n_true"].sum())
                h = hits.group_by("idx").len().rename({"len": "h"})
                ns = truth.filter(pl.col("n_true") > 0).join(h, on="idx", how="left").fill_null(0)
                stats.update(link_recall=hits.height / tot,
                             macro_recall_non_singletons=float((ns["h"] / ns["n_true"]).mean()),
                             s1_with_all_links_found=float((ns["h"] == ns["n_true"]).mean()))
            info[split] = stats
            self.log(f"cut {split}: {json.dumps(stats)}")
            del C
            gc.collect()
        (self.fdir / "candidates.json").write_text(json.dumps(info, indent=2))
        self.st_feat.mark("cut")

    # ------------------------------------------------------------------ features
    def featurize(self, split: str):
        if self.st_feat.done(f"feat_{split}"):
            self.log(f"features {split}: already done")
            return
        C = pl.read_parquet(self.fdir / f"{split}_cand.parquet")
        if split == "train":
            used = pl.read_parquet(self.fdir / "train_splits.parquet").filter("used").select("idx")
            C = C.join(used, on="idx", how="semi")
        fdir = self.fdir / split / "feat"
        if fdir.exists():
            shutil.rmtree(fdir)
        fdir.mkdir(parents=True)
        A = self.table(split, "A", features.FEATURE_COLS)
        nB = self.n_rows(split, "B")
        step = max(200_000, self.cfg.chunk * 10)
        done = 0
        for bi, lo in enumerate(range(0, nB, self.cfg.b_block)):
            hi = min(nB, lo + self.cfg.b_block)
            P = C.filter(pl.col("tid").is_between(lo, hi - 1)).sort(["idx", "tid"])
            if P.height == 0:
                continue
            Bb = self.table(split, "B", features.FEATURE_COLS, lo, hi)
            for j, s in enumerate(range(0, P.height, step)):
                features.compute(P.slice(s, step), A, Bb, self.cfg.threads, b_offset=lo) \
                        .write_parquet(fdir / f"part_{bi:03d}_{j:04d}.parquet")
            done += P.height
            self.log(f"features {split}: {done:,}/{C.height:,} pairs (targets {hi:,}/{nB:,})")
            del Bb, P
            gc.collect()
        if done != C.height:
            raise RuntimeError(f"featurized {done} pairs but {C.height} candidates")
        self.st_feat.mark(f"feat_{split}")
        del A, C
        gc.collect()

    # ------------------------------------------------------------------ model
    def _load(self, split_name: str):
        S = pl.read_parquet(self.fdir / "train_splits.parquet").filter((pl.col("split") == split_name) & pl.col("used"))
        L = pl.read_parquet(self.norm / "train_links.parquet").with_columns(label=pl.lit(1, pl.UInt8))
        F = (pl.scan_parquet(self.fdir / "train" / "feat" / "*.parquet")
               .join(S.lazy().select("idx"), on="idx", how="semi")
               .join(L.lazy(), on=["idx", "tid"], how="left").with_columns(pl.col("label").fill_null(0))
               .collect())
        X = F.select(self.feats).to_numpy(order="c")
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        return X, F["label"].to_numpy(), F["idx"].to_numpy(), F["tid"].to_numpy(), S

    def train(self):
        if self.st_model.done("train"):
            self.log("train: already done")
            return
        import lightgbm as lgb
        cfg = self.cfg
        tt = pl.read_parquet(self.norm / "train_truth.parquet")
        n_total = tt.height
        truth = np.zeros(n_total)
        truth[tt["idx"].to_numpy()] = tt["n_true"].to_numpy()
        Xt, yt, _, _, St = self._load("train")
        Xv, yv, iv, tv, Sv = self._load("valid")
        self.log(f"train pairs {len(yt):,} (pos {int(yt.sum()):,}) from {St.height:,} S1; "
                 f"valid pairs {len(yv):,} from {Sv.height:,} S1")
        params = dict(objective="binary", metric="binary_logloss", learning_rate=cfg.lr, num_leaves=cfg.leaves,
                      min_data_in_leaf=cfg.min_leaf, feature_fraction=cfg.feature_fraction, bagging_fraction=0.8,
                      bagging_freq=1, lambda_l2=1.0, max_bin=255, num_threads=cfg.threads, seed=cfg.seed,
                      verbose=-1, deterministic=True, force_row_wise=True)
        dt = lgb.Dataset(Xt, label=yt, feature_name=self.feats, free_raw_data=not cfg.refit_all)
        dv = lgb.Dataset(Xv, label=yv, reference=dt, free_raw_data=not cfg.refit_all)
        model = lgb.train(params, dt, num_boost_round=cfg.rounds, valid_sets=[dv], valid_names=["valid"],
                          callbacks=[lgb.early_stopping(cfg.early_stop, verbose=False), lgb.log_evaluation(100)])
        best_iter = model.best_iteration or cfg.rounds
        self.log(f"best iteration {best_iter}")
        pv = model.predict(Xv, num_iteration=best_iter, num_threads=cfg.threads)
        v_idx = Sv["idx"].to_numpy()
        rows = []
        for t in np.round(np.arange(0.05, 0.991, 0.01), 3):
            m = pv >= t
            if cfg.decision == "one_to_one":
                m = features.one_to_one(iv, tv, pv, m)
            rows.append({"threshold": float(t), "variant": cfg.decision,
                         "macro_f05": features.macro_f05(iv, yv, m, v_idx, truth, n_total)})
        grid = pl.DataFrame(rows)
        grid.write_csv(self.run / "artifacts" / "threshold_search.tsv", separator="\t")
        best = grid.sort(["macro_f05", "threshold"], descending=[True, True]).row(0, named=True)
        self.log(f"validation (threshold-tuned) best: {best}")
        if cfg.refit_all:
            rounds = int(best_iter * 1.1) + 1
            self.log(f"refit on train+valid for {rounds} rounds")
            dall = lgb.Dataset(np.vstack([Xt, Xv]), label=np.concatenate([yt, yv]), feature_name=self.feats)
            model = lgb.train(params, dall, num_boost_round=rounds)
            best_iter = rounds
            del dall
        del dt, dv, Xt, yt, Xv
        gc.collect()
        Xh, yh, ih, th, Sh = self._load("holdout")
        ph = model.predict(Xh, num_iteration=best_iter, num_threads=cfg.threads)
        mh = ph >= best["threshold"]
        if cfg.decision == "one_to_one":
            mh = features.one_to_one(ih, th, ph, mh)
        h_idx = Sh["idx"].to_numpy()
        country = self.table("train", "A", ["idx", "country_n"])["country_n"].to_numpy()
        by_country = {}
        for c in sorted(set(country[h_idx])):
            sel = h_idx[country[h_idx] == c]
            by_country[c] = {"s1": int(len(sel)), "macro_f05": features.macro_f05(ih, yh, mh, sel, truth, n_total)}
        sing = h_idx[truth[h_idx] == 0]
        pc = np.bincount(ih[mh], minlength=n_total)
        tp = int(yh[mh].sum())
        fi = {f: i for i, f in enumerate(self.feats)}
        fp, fn = mh & (yh == 0), (~mh) & (yh == 1)

        def share(mask, f, cond):
            if f not in fi or not mask.any():
                return None
            return round(float(cond(Xh[mask, fi[f]]).mean()), 4)
        linked = pl.read_parquet(self.norm / "train_links.parquet").select("tid").unique()
        fp_linked = (pl.DataFrame({"tid": th[fp]}).join(linked, on="tid", how="semi").height / int(fp.sum())
                     if fp.any() else None)
        errors = {
            "holdout_true_links": int(truth[h_idx].sum()), "true_links_in_candidates": int(yh.sum()),
            "missed_by_blocking": int(truth[h_idx].sum() - yh.sum()), "found_but_rejected": int(fn.sum()),
            "false_positive_pairs": int(fp.sum()),
            "fp_share_target_belongs_to_another_s1": None if fp_linked is None else round(fp_linked, 4),
            "fp_share_name_ratio_ge_90": share(fp, "name_ratio", lambda c: c >= 90),
            "fp_share_number_conflict": share(fp, "num_conflict", lambda c: c == 1),
            "fp_share_target_addr_missing": share(fp, "b_addr_missing", lambda c: c == 1),
            "fn_share_indic_script_target": share(fn, "b_indic", lambda c: c == 1),
            "fn_share_name_ratio_lt_60": share(fn, "name_ratio", lambda c: (c >= 0) & (c < 60)),
            "fn_share_addr_ratio_lt_50": share(fn, "addr_ratio", lambda c: (c >= 0) & (c < 50)),
            "fn_share_target_addr_missing": share(fn, "b_addr_missing", lambda c: c == 1),
        }
        metrics = {
            "validation_macro_f05_threshold_tuned": best["macro_f05"],
            "threshold": best["threshold"], "decision": cfg.decision,
            "holdout_macro_f05_untouched": features.macro_f05(ih, yh, mh, h_idx, truth, n_total),
            "holdout_by_country": by_country,
            "error_analysis": errors,
            "holdout_singleton_accuracy": float((pc[sing] == 0).mean()) if len(sing) else None,
            "holdout_pair_precision": tp / max(1, int(mh.sum())),
            "holdout_pair_recall_vs_all_true_links": tp / max(1, int(truth[h_idx].sum())),
            "holdout_s1": int(len(h_idx)), "valid_s1": int(len(v_idx)), "train_s1": int(St.height),
            "best_iteration": int(best_iter), "features": self.feats, "refit_all": cfg.refit_all,
            "note": "Validation score is threshold-tuned on the validation split. Holdout is an untouched grouped "
                    "split of the training data (US/India only); no labelled France data exists."
                    + (" WARNING: one_to_one is evaluated only among the split's own S1, which understates how "
                       "often it fires on the full test set; treat this estimate as optimistic."
                       if cfg.decision == "one_to_one" else ""),
        }
        model.save_model(str(self.run / "artifacts" / "model.txt"), num_iteration=best_iter)
        pl.DataFrame({"feature": self.feats, "gain": model.feature_importance("gain", iteration=best_iter)}) \
          .sort("gain", descending=True).write_csv(self.run / "artifacts" / "feature_importance.tsv", separator="\t")
        (self.run / "artifacts" / "metrics.json").write_text(json.dumps(metrics, indent=2))
        self.log("metrics: " + json.dumps(metrics))
        self.st_model.mark("train")

    # ------------------------------------------------------------------ inference + outputs
    def predict(self):
        if self.st_model.done("predict"):
            self.log("predict: already done")
            return
        import lightgbm as lgb
        m = json.loads((self.run / "artifacts" / "metrics.json").read_text())
        model = lgb.Booster(model_file=str(self.run / "artifacts" / "model.txt"))
        idx, tid, prob = [], [], []
        for f in sorted((self.fdir / "test" / "feat").glob("*.parquet")):
            F = pl.read_parquet(f)
            X = F.select(m["features"]).to_numpy(order="c")
            prob.append(model.predict(X.astype(np.float32, copy=False), num_threads=self.cfg.threads))
            idx.append(F["idx"].to_numpy())
            tid.append(F["tid"].to_numpy())
            del F, X
        idx, tid, prob = np.concatenate(idx), np.concatenate(tid), np.concatenate(prob)
        C = pl.read_parquet(self.fdir / "test_cand.parquet", columns=["idx", "tid"])
        scored = pl.DataFrame({"idx": idx, "tid": tid})
        if C.height != scored.height or C.join(scored, on=["idx", "tid"], how="anti").height:
            raise RuntimeError("Scored pairs differ from the candidate set; refusing to write outputs")
        match = prob >= m["threshold"]
        n_thr = int(match.sum())
        if m["decision"] == "one_to_one":
            match = features.one_to_one(idx, tid, prob, match)
        A = self.table("test", "A", ["idx", "entity_id"])
        Bid = self.table("test", "B", ["idx", "entity_id"])["entity_id"]
        P = pl.DataFrame({"idx": idx, "tid": tid, "p": prob.astype(np.float32), "match": match}) \
              .with_columns(t=Bid.gather(pl.Series(tid)))
        P.write_parquet(self.work / "test_scored_pairs.parquet")
        self.write_outputs(A, P)
        stats = {"test_s1": A.height, "test_candidate_pairs": P.height, "test_matches": int(match.sum()),
                 "test_matches_before_one_to_one": n_thr,
                 "test_s1_with_matches": int(P.filter("match")["idx"].n_unique()),
                 "test_avg_candidates": P.height / A.height}
        (self.run / "artifacts" / "test_stats.json").write_text(json.dumps(stats, indent=2))
        self.log("test: " + json.dumps(stats))
        self.st_model.mark("predict")

    def write_outputs(self, A: pl.DataFrame, P: pl.DataFrame):
        cand = P.group_by("idx").agg(candidate_entity_ids=pl.col("t").sort().str.join(","))
        mat = P.filter("match").group_by("idx").agg(matched_entity_ids=pl.col("t").sort().str.join(","))
        base = A.select("idx", source1_entity_id="entity_id")
        for name, df, col in (("candidate_pairs.tsv", cand, "candidate_entity_ids"),
                              ("matching_results.tsv", mat, "matched_entity_ids")):
            out = base.join(df, on="idx", how="left").sort("idx").select(
                "source1_entity_id", pl.col(col).fill_null(""))
            tmp = self.out / (name + ".tmp")
            out.write_csv(tmp, separator="\t", quote_style="never", line_terminator="\n")
            tmp.replace(self.out / name)
        self.log(f"wrote {self.out / 'matching_results.tsv'} and candidate_pairs.tsv ({A.height:,} rows each)")

    # ------------------------------------------------------------------ validation
    def validate(self) -> dict:
        res = check_outputs(self.out, self.data / "dataset" / "test")
        self.log(f"own validator: {res['own_validator']}")
        gc.collect()
        res.update(official_validator(self.data, self.out, self.run / "artifacts" / "official_validator.log"))
        self.log(f"official validator: {res['official_validator']}")
        (self.run / "artifacts" / "validation.json").write_text(json.dumps(res, indent=2))
        if res["official_validator"] != "PASS":
            raise RuntimeError("Official validator did not PASS; see artifacts/official_validator.log")
        return res

    def all(self):
        self._timed("prepare", self.prepare)
        shutil.copy2(self.norm / "audit.json", self.run / "artifacts" / "audit.json")
        self._timed("block_train", self.block, "train")
        self._timed("block_test", self.block, "test")
        self._timed("cut", self.cut)
        self._timed("features_train", self.featurize, "train")
        self._timed("features_test", self.featurize, "test")
        self._timed("train", self.train)
        self._timed("predict", self.predict)
        res = self._timed("validate", self.validate)
        (self.run / "artifacts" / "stage_seconds.json").write_text(json.dumps(self.times, indent=2))
        return res


def _ids(path: Path) -> pl.DataFrame:
    return (pl.scan_csv(path, separator="\t", quote_char=None, infer_schema=False, truncate_ragged_lines=True,
                        encoding="utf8-lossy").select(pl.col("entity_id").str.strip_chars()).collect())


def check_outputs(out_dir: Path, test_dir: Path) -> dict:
    """Independent vectorised check of both files against the RAW official test TSVs."""
    s1 = _ids(test_dir / "test_source1.tsv").rename({"entity_id": "s1"})
    allowed = pl.concat([_ids(test_dir / "test_source2.tsv"), _ids(test_dir / "test_source3.tsv")]).rename({"entity_id": "t"})
    exploded = {}
    for name, col in (("matching_results.tsv", "matched_entity_ids"), ("candidate_pairs.tsv", "candidate_entity_ids")):
        p = out_dir / name
        with open(p, "rb") as f:
            head = f.readline()
            if head != f"source1_entity_id\t{col}\n".encode():
                raise AssertionError(f"{name}: bad header {head[:80]!r}")
            for block in iter(lambda: f.read(1 << 24), b""):
                if b'"' in block or b"\r" in block:
                    raise AssertionError(f"{name}: contains quotes or CR characters")
        df = pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
        if df.columns != ["source1_entity_id", col]:
            raise AssertionError(f"{name}: columns {df.columns}")
        df = df.rename({"source1_entity_id": "s1", col: "ids"})
        if df.height != s1.height or df["s1"].n_unique() != df.height:
            raise AssertionError(f"{name}: {df.height} rows / {df['s1'].n_unique()} unique, expected {s1.height}")
        if df.join(s1, on="s1", how="anti").height or s1.join(df, on="s1", how="anti").height:
            raise AssertionError(f"{name}: Source 1 ids differ from test_source1.tsv")
        ex = (df.select("s1", t=pl.col("ids").str.split(",")).explode("t", empty_as_null=True)
                .filter(pl.col("t").is_not_null() & (pl.col("t") != "")))
        if ex.group_by("s1", "t").len().filter(pl.col("len") > 1).height:
            raise AssertionError(f"{name}: duplicate ID inside a list")
        if ex.join(allowed, on="t", how="anti").height:
            raise AssertionError(f"{name}: IDs that are not test Source 2/3 records")
        exploded[name] = ex
        del df
    if exploded["matching_results.tsv"].join(exploded["candidate_pairs.tsv"], on=["s1", "t"], how="anti").height:
        raise AssertionError("matches outside the candidate list")
    return {"own_validator": "PASS", "rows": s1.height,
            "matched_pairs": exploded["matching_results.tsv"].height,
            "candidate_pairs": exploded["candidate_pairs.tsv"].height}


def official_validator(sr: Path, out_dir: Path, log_path: Path) -> dict:
    cmd = [sys.executable, "utils/validate_submission.py",
           "--matching", str((out_dir / "matching_results.tsv").resolve()),
           "--candidate", str((out_dir / "candidate_pairs.tsv").resolve()), "--test-dir", "dataset/test"]
    if total_ram_gb() >= 15:
        cmd.append("--check-ids")
    r = subprocess.run(cmd, cwd=sr, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log_path.write_text(r.stdout + r.stderr + f"\nexit={r.returncode}\n", encoding="utf-8")
    passed = r.returncode == 0 and "PASS" in r.stdout
    return {"official_validator": "PASS" if passed else f"FAIL (exit {r.returncode})",
            "official_validator_cmd": " ".join(cmd), "official_validator_data": str(sr)}
