#!/usr/bin/env python3
"""Memory-capped dataset audit using DuckDB (spills to disk). Reads the 7 official TSVs untouched."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from er_common import connect, read_tsv_sql  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--memory", default="2GB")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    con = connect(a.memory, a.threads)
    rep = {}
    for split in ("train", "test"):
        for s in (1, 2, 3):
            f = a.data / split / f"{split}_source{s}.tsv"
            t = f"{split}_s{s}"
            con.execute(f"CREATE OR REPLACE TEMP VIEW {t} AS SELECT * FROM {read_tsv_sql(f)}")
            r = con.execute(f"""
                SELECT count(*), count(DISTINCT entity_id),
                       sum(CASE WHEN NOT starts_with(entity_id, 'S{s}-') THEN 1 ELSE 0 END),
                       sum(CASE WHEN business_name IS NULL OR trim(business_name)='' THEN 1 ELSE 0 END),
                       sum(CASE WHEN business_address IS NULL OR trim(business_address)='' THEN 1 ELSE 0 END),
                       sum(CASE WHEN country IS NULL OR trim(country)='' THEN 1 ELSE 0 END),
                       avg(length(business_name)), avg(length(business_address))
                FROM {t}""").fetchone()
            countries = dict(con.execute(
                f"SELECT coalesce(country,'<NULL>'), count(*) FROM {t} GROUP BY 1 ORDER BY 2 DESC LIMIT 20").fetchall())
            rep[t] = dict(rows=r[0], distinct_ids=r[1], bad_prefix=r[2], blank_name=r[3], blank_address=r[4],
                          blank_country=r[5], avg_name_len=r[6], avg_addr_len=r[7], countries=countries)
            print(t, json.dumps(rep[t]), flush=True)
    gt = a.data / "train" / "train_ground_truth.tsv"
    con.execute(f"""CREATE OR REPLACE TEMP TABLE gt AS
        SELECT source1_entity_id AS s1, coalesce(matched_entity_ids,'') AS m FROM {read_tsv_sql(gt, ['source1_entity_id','matched_entity_ids'])}""")
    con.execute("""CREATE OR REPLACE TEMP TABLE links AS
        SELECT s1, trim(t) AS tid FROM (SELECT s1, unnest(string_split(m, ',')) AS t FROM gt WHERE m <> '') WHERE trim(t) <> ''""")
    g = {}
    g["gt_rows"], g["gt_distinct_s1"] = con.execute("SELECT count(*), count(DISTINCT s1) FROM gt").fetchone()
    g["gt_s1_not_in_train_s1"] = con.execute("SELECT count(*) FROM gt ANTI JOIN train_s1 ON gt.s1=train_s1.entity_id").fetchone()[0]
    g["train_s1_not_in_gt"] = con.execute("SELECT count(*) FROM train_s1 ANTI JOIN gt ON gt.s1=train_s1.entity_id").fetchone()[0]
    g["links"], g["distinct_targets"] = con.execute("SELECT count(*), count(DISTINCT tid) FROM links").fetchone()
    g["links_by_prefix"] = dict(con.execute("SELECT left(tid,3), count(*) FROM links GROUP BY 1").fetchall())
    g["dup_links"] = g["links"] - con.execute("SELECT count(*) FROM (SELECT DISTINCT s1, tid FROM links)").fetchone()[0]
    g["targets_linked_to_multiple_s1"] = con.execute(
        "SELECT count(*) FROM (SELECT tid FROM links GROUP BY tid HAVING count(DISTINCT s1)>1)").fetchone()[0]
    con.execute("CREATE OR REPLACE TEMP VIEW tgt_ids AS SELECT entity_id FROM train_s2 UNION ALL SELECT entity_id FROM train_s3")
    g["link_targets_missing_from_train_s2_s3"] = con.execute(
        "SELECT count(*) FROM links ANTI JOIN tgt_ids ON links.tid=tgt_ids.entity_id").fetchone()[0]
    g["train_targets_unlinked"] = con.execute(
        "SELECT count(*) FROM tgt_ids ANTI JOIN (SELECT DISTINCT tid FROM links) l ON l.tid=tgt_ids.entity_id").fetchone()[0]
    hist = con.execute("""SELECT n, count(*) FROM (SELECT gt.s1, count(links.tid) n FROM gt LEFT JOIN links USING(s1) GROUP BY gt.s1)
                          GROUP BY n ORDER BY n""").fetchall()
    g["matches_per_s1_hist"] = {int(k): int(v) for k, v in hist}
    g["singleton_fraction"] = g["matches_per_s1_hist"].get(0, 0) / g["gt_rows"]
    g["singletons_by_country"] = dict(con.execute("""
        SELECT s.country, avg(CASE WHEN gt.m='' THEN 1.0 ELSE 0.0 END) FROM gt JOIN train_s1 s ON s.entity_id=gt.s1 GROUP BY 1""").fetchall())
    g["cross_country_links"] = con.execute("""
        SELECT count(*) FROM links l JOIN train_s1 a ON a.entity_id=l.s1
        JOIN (SELECT entity_id, country FROM train_s2 UNION ALL SELECT entity_id, country FROM train_s3) b ON b.entity_id=l.tid
        WHERE coalesce(a.country,'') <> coalesce(b.country,'')""").fetchone()[0]
    rep["ground_truth"] = g
    print("ground_truth", json.dumps(g), flush=True)
    a.out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print("AUDIT DONE", a.out)


if __name__ == "__main__":
    main()
