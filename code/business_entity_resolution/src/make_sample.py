"""Build a small, structurally faithful sample of the official data for smoke tests.

The sample keeps the first N training Source 1 entities with ALL their labelled Source 2/3
targets plus unrelated distractor records, and the first M test Source 1 entities with the
first test Source 2/3 records. It is only used to prove the pipeline runs end to end."""
from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl


def _scan(p: Path) -> pl.LazyFrame:
    return pl.scan_csv(p, separator="\t", quote_char=None, infer_schema=False, truncate_ragged_lines=True,
                       missing_utf8_is_empty_string=True, encoding="utf8-lossy")


def _write(df: pl.DataFrame, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(p, separator="\t", quote_style="never", line_terminator="\n")


def make_sample(full: Path, dest: Path, n_train: int = 4000, n_test: int = 3000, n_distract: int = 25000) -> Path:
    sr = dest / "student_resource"
    done = sr / ".complete"
    if done.is_file():
        return sr
    if sr.exists():
        shutil.rmtree(sr)
    tr, te = full / "dataset" / "train", full / "dataset" / "test"
    gt = _scan(tr / "train_ground_truth.tsv").head(n_train).collect(engine="streaming")
    s1 = gt["source1_entity_id"].to_list()
    tids = (gt.select(t=pl.col("matched_entity_ids").str.split(",")).explode("t")
              .select(pl.col("t").str.strip_chars()).filter(pl.col("t") != "")["t"].to_list())
    _write(gt, sr / "dataset/train/train_ground_truth.tsv")
    _write(_scan(tr / "train_source1.tsv").filter(pl.col("entity_id").is_in(s1)).collect(engine="streaming"),
           sr / "dataset/train/train_source1.tsv")
    for s in (2, 3):
        f = tr / f"train_source{s}.tsv"
        linked = _scan(f).filter(pl.col("entity_id").is_in(tids)).collect(engine="streaming")
        extra = _scan(f).head(n_distract).collect(engine="streaming")
        _write(pl.concat([linked, extra]).unique("entity_id", keep="first", maintain_order=True),
               sr / f"dataset/train/train_source{s}.tsv")
    _write(_scan(te / "test_source1.tsv").head(n_test).collect(engine="streaming"), sr / "dataset/test/test_source1.tsv")
    for s in (2, 3):
        _write(_scan(te / f"test_source{s}.tsv").head(n_distract).collect(engine="streaming"), sr / f"dataset/test/test_source{s}.tsv")
    (sr / "utils").mkdir(parents=True, exist_ok=True)
    shutil.copy2(full / "utils" / "validate_submission.py", sr / "utils" / "validate_submission.py")
    shutil.copy2(full / "Documentation_template.md", sr / "Documentation_template.md")
    done.write_text("ok")
    return sr
