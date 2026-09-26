"""Shared helpers: configuration, logging, memory reporting, TSV IO, stage markers."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]
GT_COLS = ["source1_entity_id", "matched_entity_ids"]
OFFICIAL_SHA256 = {
    "Documentation_template.md": "599b661af92a8d5c239ba283ee9b2ffed050c0bc86360964430a2f9857e8c599",
    "README.md": "be67cd7d9dbc48a58f93e1f61402360655debac64214115bf9d27d3301939285",
    "utils/validate_submission.py": "f96f59934383a15095914f507620c078c474e8f9c60e864b2d051ed173a22dfc",
    "dataset/train/train_ground_truth.tsv": "70bc1d8a16c667e0155c2105d0ab2ebe41d7e7a85d8a529e3ca81c6c3a5af037",
    "dataset/train/train_source1.tsv": "591af0e1dfeb65cab71ea6ee8cb69df00f92d6ba6fa79e05746c938775d14973",
    "dataset/train/train_source2.tsv": "6336c1a055eec79cf8a6d99fdc8d32a2e4d9dc2662e00963cb35d66b89ed09ed",
    "dataset/train/train_source3.tsv": "67da22f5151898ff3006febd836c1a159e97ae95efa7257a5aff4fda685e58e9",
    "dataset/test/test_source1.tsv": "3d4a32c54c2ca9c53fd7c2be105bf26f708f94c4d2f88eb370972a195665c2f5",
    "dataset/test/test_source2.tsv": "79d906c7497af2ace70aa277f6e334a652094909de99bd6c57b53420b6a7b2dd",
    "dataset/test/test_source3.tsv": "850942b11d2a4343486ed0834e28bce9f3b385f3fd497fd60ccf4ea3b8bda035",
}


def total_ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        pass
    try:  # Windows (only used for development smoke tests)
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                (n, ctypes.c_ulonglong) for n in ("tp", "ap", "tpf", "apf", "tv", "av", "aev")]
        m = _MS()
        m.dwLength = ctypes.sizeof(_MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.tp / 2**30
    except Exception:
        return 8.0


def peak_rss_gb() -> float | None:
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 2**30 if sys.platform == "darwin" else r / 2**20
    except Exception:
        return None


@dataclass
class Config:
    data: str = ""                 # student_resource directory (contains dataset/ and utils/)
    run: str = "runs/main"
    threads: int = 0               # 0 = auto (all cores but one)
    seed: int = 42
    max_df: int = 300              # blocking keys shared by more targets than this are ignored
    key_budget: int = 1200         # per S1: add rarest keys until their total postings exceed this
    prelim: int = 60               # IDF-vote shortlist per S1, reranked by string similarity
    recall_tol: float = 0.010      # allowed candidate-recall loss when shrinking K (see sweep)
    max_k: int = 40                # hard cap on candidates per S1 (memory safety and candidate-size review)
    chunk: int = 0                 # S1 records per blocking chunk (0 = auto from RAM)
    b_block: int = 0               # target rows held in memory while computing features (0 = auto)
    train_s1: int = 0              # S1 entities whose pairs train the model (0 = auto)
    valid_s1: int = 0
    holdout_s1: int = 0
    tune_s1: int = 300_000         # S1 sample used to choose K / gap
    rounds: int = 3000
    early_stop: int = 100
    lr: float = 0.05
    leaves: int = 127
    min_leaf: int = 200
    feature_fraction: float = 0.8
    drop_features: str = ""        # comma-separated feature names to exclude (ablations)
    decision: str = "threshold"    # threshold | one_to_one (one_to_one's local estimate is optimistic)
    refit_all: bool = False        # refit on train+valid with the early-stopped round count
    cache: str = "cache"           # shared normalised data + blocking caches (reused across runs)
    allow_unofficial: bool = False # accept data whose SHA-256 differs from the official files
    team: str = ""
    members: str = ""
    smoke: bool = False

    def resolve(self) -> "Config":
        ram = total_ram_gb()
        cores = os.cpu_count() or 4
        if not self.threads:
            self.threads = max(1, cores - 1)
        if not self.chunk:
            self.chunk = 15_000 if ram < 10 else 40_000 if ram < 20 else 80_000
        if not self.train_s1:
            self.train_s1 = 250_000 if ram < 10 else 600_000 if ram < 20 else 1_000_000
        if not self.valid_s1:
            self.valid_s1 = 120_000 if ram < 10 else 200_000
        if not self.holdout_s1:
            self.holdout_s1 = 120_000 if ram < 10 else 200_000
        if not self.b_block:
            self.b_block = 1_000_000 if ram < 10 else 2_500_000 if ram < 20 else 5_000_000
        return self

    def dump(self) -> dict:
        d = asdict(self)
        d["ram_gb"] = round(total_ram_gb(), 1)
        d["cpu_count"] = os.cpu_count()
        d["platform"] = platform.platform()
        d["python"] = platform.python_version()
        return d


class Log:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")
        self.t0 = time.perf_counter()

    def __call__(self, *msg):
        rss = peak_rss_gb()
        line = (f"[{time.strftime('%Y-%m-%d %H:%M:%S')} +{time.perf_counter() - self.t0:7.0f}s"
                + (f" peakRSS {rss:.1f}G" if rss else "") + "] " + " ".join(str(m) for m in msg))
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def read_tsv(path: Path, cols: list[str]) -> pl.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False, has_header=True,
                     truncate_ragged_lines=True, missing_utf8_is_empty_string=True,
                     encoding="utf8-lossy")
    if df.columns[: len(cols)] != cols:
        raise ValueError(f"{path}: expected columns {cols}, found {df.columns}")
    return df.select(cols).with_columns(pl.col(cols[0]).str.strip_chars())


def read_source(path: Path, prefix: str) -> pl.DataFrame:
    df = read_tsv(path, SOURCE_COLS)
    bad = df.filter(~pl.col("entity_id").str.starts_with(prefix + "-") | pl.col("entity_id").str.contains(r"[,\t\s]"))
    if bad.height:
        raise ValueError(f"{path}: {bad.height} entity_ids with wrong prefix/delimiters, e.g. {bad['entity_id'][:3].to_list()}")
    if df["entity_id"].n_unique() != df.height:
        raise ValueError(f"{path}: duplicate entity_id values")
    return df


def read_links(path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Returns (gt rows: s1 eid, links: s1 eid + target eid)."""
    gt = read_tsv(path, GT_COLS)
    links = (gt.select(s1="source1_entity_id", t=pl.col("matched_entity_ids").str.split(","))
               .explode("t").with_columns(pl.col("t").str.strip_chars())
               .filter(pl.col("t").is_not_null() & (pl.col("t") != "")).unique())
    return gt.select(s1="source1_entity_id"), links


class Stages:
    """Resumable stage markers stored under the run's work directory."""

    def __init__(self, work: Path, cfg_hash: str):
        self.dir = work / "_done"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.h = cfg_hash

    def done(self, name: str) -> bool:
        p = self.dir / name
        return p.is_file() and p.read_text() == self.h

    def mark(self, name: str):
        (self.dir / name).write_text(self.h)


def config_hash(d: dict, keys: list[str]) -> str:
    return hashlib.sha1(json.dumps({k: d[k] for k in keys}, sort_keys=True).encode()).hexdigest()[:12]
