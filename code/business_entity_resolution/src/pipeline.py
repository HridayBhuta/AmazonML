#!/usr/bin/env python3
"""CPU-first, offline business entity resolution starter. See README before use."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import heapq
import json
import math
import platform
import random
import re
import sys
import time
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

FIELDS = ["entity_id", "business_name", "business_address", "country"]
FEATURES = [
    "name_ratio", "name_sorted", "name_token_set", "name_partial", "name_jaccard",
    "address_ratio", "address_sorted", "address_token_set", "address_jaccard",
    "name_exact", "address_exact", "number_jaccard", "number_conflict",
    "first_number_equal", "country_equal", "country_missing", "name_missing",
    "address_missing", "name_length_ratio", "address_length_ratio", "source3",
]


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold().replace("&", " and "))
    value = "".join(c for c in value if not unicodedata.combining(c))
    return " ".join("".join(c if c.isalnum() else " " for c in value).split())


@dataclass(slots=True)
class Record:
    eid: str
    name: str
    address: str
    country: str
    nt: frozenset
    at: frozenset
    nums: frozenset
    first_num: str


def read_table(path: Path, required: list[str]):
    if not path.is_file():
        raise FileNotFoundError(f"Missing: {path}. Keep the official train/test folder structure.")
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: expected tab-separated columns {required}; got {reader.fieldnames}")
        for line, row in enumerate(reader, 2):
            if None in row or any(row[k] is None for k in required):
                raise ValueError(f"{path}:{line}: malformed TSV row")
            yield {k: row[k] for k in required}


def load_source(path: Path, source: str) -> list[Record]:
    records, seen = [], set()
    for row in read_table(path, FIELDS):
        eid = row["entity_id"].strip()
        if not eid.startswith(source + "-") or eid in seen:
            raise ValueError(f"{path}: invalid/duplicate entity_id {eid!r}")
        if any(c in eid for c in ',\t\n\r"'):
            raise ValueError(f"{path}: entity_id contains an output delimiter: {eid!r}")
        seen.add(eid)
        name, address = normalize(row["business_name"]), normalize(row["business_address"])
        numbers = re.findall(r"\d+", address)
        records.append(Record(eid, name, address, normalize(row["country"]),
                              frozenset(name.split()), frozenset(address.split()),
                              frozenset(numbers), numbers[0] if numbers else ""))
    return records


def load_split(data: Path, split: str):
    sources = [load_source(data / split / f"{split}_source{i}.tsv", f"S{i}") for i in (1, 2, 3)]
    return sources[0], sources[1] + sources[2]


def parse_ids(value: str) -> list[str]:
    if not value.strip():
        return []
    parts = [p.strip() for p in value.split(",")]
    if any(not p for p in parts) or len(parts) != len(set(parts)):
        raise ValueError(f"Empty or duplicate ID inside list: {value[:120]!r}")
    return parts


def load_truth(data: Path, queries: list[Record], targets: list[Record]):
    truth, valid = {}, {r.eid for r in targets}
    path = data / "train" / "train_ground_truth.tsv"
    for row in read_table(path, ["source1_entity_id", "matched_entity_ids"]):
        qid = row["source1_entity_id"].strip()
        if qid in truth:
            raise ValueError(f"Duplicate ground truth row {qid}")
        ids = set(parse_ids(row["matched_entity_ids"]))
        if ids - valid:
            raise ValueError(f"Ground truth references unavailable targets for {qid}: {sorted(ids-valid)[:5]}")
        truth[qid] = ids
    expected = {r.eid for r in queries}
    if set(truth) != expected:
        raise ValueError("Ground truth must contain exactly the training Source 1 IDs.")
    return truth


def audit(data: Path):
    report = {}
    for split in ("train", "test"):
        q, targets = load_split(data, split)
        if not q or not targets:
            raise ValueError(f"{split}: empty Source 1 or target pool")
        report[split] = {}
        for source in ("S1", "S2", "S3"):
            subset = q if source == "S1" else [r for r in targets if r.eid.startswith(source + "-")]
            report[split][source] = {
                "rows": len(subset), "countries_normalized": dict(Counter(r.country for r in subset)),
                "blank_names": sum(not r.name for r in subset),
                "blank_addresses": sum(not r.address for r in subset),
            }
        if split == "train":
            truth = load_truth(data, q, targets)
            counts = [len(truth[r.eid]) for r in q]
            report["ground_truth"] = {
                "source1_entities": len(q), "positive_links": sum(counts),
                "singletons": counts.count(0), "singleton_fraction": counts.count(0) / len(q),
                "max_matches_per_entity": max(counts),
                "mean_matches_per_entity": sum(counts) / len(q),
            }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def block_keys(r: Record):
    compact = r.name.replace(" ", "")
    keys = set()
    if compact:
        keys.add(("exact", compact))
    keys.update(("name", t) for t in r.nt if len(t) > 1)
    keys.update(("gram", compact[i:i + 3]) for i in range(max(0, len(compact) - 2)))
    keys.update(("addr", t) for t in r.at if len(t) > 2 and not t.isdigit())
    keys.update(("num", n) for n in r.nums if len(n) >= 4)
    return sorted(keys)


class Blocker:
    """Bounded inverted-index retrieval, then deterministic fuzzy prefiltering.

    Large posting lists are discarded completely, not truncated to biased first IDs.
    No all-pairs matrix is built. This is an in-memory prototype, not a tested
    billion-record implementation. Frequent/short fields can lose recall: measure it.
    """
    LIMITS = {"exact": 1, "name": 6, "gram": 10, "addr": 4, "num": 2}
    WEIGHTS = {"exact": 12.0, "name": 4.0, "gram": 1.0, "addr": 1.0, "num": 2.0}

    def __init__(self, records: list[Record], top_k: int, max_postings: int, prefilter: int):
        self.records, self.top_k = records, top_k
        self.prefilter = max(prefilter, top_k)
        self.index: dict[tuple[str, str], list[int] | None] = {}
        for j, record in enumerate(records):
            for key in block_keys(record):
                if key not in self.index:
                    self.index[key] = [j]
                elif self.index[key] is not None:
                    self.index[key].append(j)
                    if len(self.index[key]) > max_postings:
                        self.index[key] = None
        print(f"Indexed {len(records):,} target records; {len(self.index):,} keys", flush=True)

    def candidates(self, query: Record) -> list[int]:
        from rapidfuzz import fuzz
        grouped = defaultdict(list)
        for key in block_keys(query):
            postings = self.index.get(key)
            if postings:
                grouped[key[0]].append((len(postings), key, postings))
        votes = defaultdict(float)
        for kind, blocks in grouped.items():
            for size, key, postings in sorted(blocks)[:self.LIMITS[kind]]:
                weight = self.WEIGHTS[kind] * math.log1p(len(self.records) / size)
                for j in postings:
                    votes[j] += weight
        # First bounded retrieval shortlist, then a non-ML string-similarity prefilter.
        initial = heapq.nsmallest(self.prefilter, votes, key=lambda j: (-votes[j], self.records[j].eid))
        def rank(j):
            target = self.records[j]
            name = fuzz.WRatio(query.name, target.name) if query.name and target.name else 0
            addr = fuzz.token_sort_ratio(query.address, target.address) if query.address and target.address else 0
            return -(0.72 * name + 0.28 * addr), -votes[j], target.eid
        return sorted(initial, key=rank)[:self.top_k]


def jac(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def features(a: Record, b: Record):
    from rapidfuzz import fuzz
    def sim(fn, x, y):
        return fn(x, y) / 100.0 if x and y else 0.0
    def lr(x, y):
        return min(len(x), len(y)) / max(len(x), len(y)) if x and y else 0.0
    return [
        sim(fuzz.ratio, a.name, b.name), sim(fuzz.token_sort_ratio, a.name, b.name),
        sim(fuzz.token_set_ratio, a.name, b.name), sim(fuzz.partial_ratio, a.name, b.name), jac(a.nt, b.nt),
        sim(fuzz.ratio, a.address, b.address), sim(fuzz.token_sort_ratio, a.address, b.address),
        sim(fuzz.token_set_ratio, a.address, b.address), jac(a.at, b.at),
        float(bool(a.name) and a.name == b.name), float(bool(a.address) and a.address == b.address),
        jac(a.nums, b.nums), float(bool(a.nums and b.nums) and not (a.nums & b.nums)),
        float(bool(a.first_num) and a.first_num == b.first_num),
        float(bool(a.country) and a.country == b.country), float(not a.country or not b.country),
        float(not a.name or not b.name), float(not a.address or not b.address),
        lr(a.name, b.name), lr(a.address, b.address), float(b.eid.startswith("S3-")),
    ]


def split_queries(queries, truth, seed, fraction=0.2):
    # Keep all pairs of a Source 1 entity together, plus any Source 1 entities
    # sharing a labelled target. Stratify components by country and singleton flag.
    parent = list(range(len(queries)))
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    owner = {}
    for i, q in enumerate(queries):
        for target in sorted(truth[q.eid]):
            if target in owner:
                parent[root(i)] = root(owner[target])
            else:
                owner[target] = i
    components = defaultdict(list)
    for i in range(len(queries)):
        components[root(i)].append(i)
    strata = defaultdict(list)
    for members in components.values():
        countries = tuple(sorted({queries[i].country for i in members}))
        has_match = any(truth[queries[i].eid] for i in members)
        strata[(countries, has_match)].append(members)
    rng, valid = random.Random(seed), set()
    for key in sorted(strata):
        groups = strata[key]
        rng.shuffle(groups)
        n = max(1, min(len(groups)-1, round(len(groups)*fraction))) if len(groups) > 1 else 0
        for members in groups[:n]:
            valid.update(members)
    if not valid or len(valid) == len(queries):
        raise ValueError("Too few independent entities for a grouped validation split.")
    return valid


def pair_arrays(queries, targets, truth, blocker, max_pairs):
    import numpy as np
    capacity = len(queries) * min(blocker.top_k, len(targets))
    if capacity > max_pairs:
        raise MemoryError(
            f"Up to {capacity:,} training pairs, above --max-pairs={max_pairs:,}. "
            "Use a larger-RAM machine and raise --max-pairs, or lower --top-k and measure recall. "
            "Never drop test Source 1 entities."
        )
    X = np.empty((capacity, len(FEATURES)), dtype=np.float32)
    y = np.empty(capacity, dtype=np.uint8)
    groups = np.empty(capacity, dtype=np.int32)
    per_query_count, per_query_hits = [], []
    cursor = 0
    for i, q in enumerate(queries):
        candidates = blocker.candidates(q)
        hits = 0
        for j in candidates:
            label = targets[j].eid in truth[q.eid]
            X[cursor] = features(q, targets[j])
            y[cursor], groups[cursor] = label, i
            cursor += 1
            hits += int(label)
        per_query_count.append(len(candidates))
        per_query_hits.append(hits)
        if (i + 1) % 1000 == 0:
            print(f"Training candidates/features: {i+1:,}/{len(queries):,} entities", flush=True)
    return X[:cursor], y[:cursor], groups[:cursor], per_query_count, per_query_hits


def entity_scores(groups, labels, probabilities, threshold, query_indices, truth_sizes, total_queries):
    import numpy as np
    keep = probabilities >= threshold
    pred = np.bincount(groups[keep], minlength=total_queries)
    tp = np.bincount(groups[keep], weights=labels[keep], minlength=total_queries)
    idx = np.asarray(query_indices, dtype=np.int64)
    denom = 0.25 * truth_sizes[idx] + pred[idx]
    scores = np.ones(len(idx), dtype=np.float64)
    active = denom > 0
    scores[active] = 1.25 * tp[idx][active] / denom[active]
    return float(scores.mean()), pred, tp


def manifest(data: Path):
    hashes = {}
    for path in sorted(data.rglob("*.tsv")):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024*1024), b""):
                h.update(chunk)
        hashes[str(path.relative_to(data))] = h.hexdigest()
    return hashes


def run(args):
    import importlib.metadata
    import numpy as np
    import lightgbm as lgb
    start = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    q, targets = load_split(args.data, "train")
    truth = load_truth(args.data, q, targets)
    if not q or not targets:
        raise ValueError("Training sources must not be empty.")
    valid_set = split_queries(q, truth, args.seed)
    blocker = Blocker(targets, args.top_k, args.max_postings, args.prefilter)
    X, y, group, counts, hits = pair_arrays(q, targets, truth, blocker, args.max_pairs)
    if not len(y):
        raise ValueError("Blocking found no training candidates; inspect normalization/blocking.")
    is_valid_query = np.array([i in valid_set for i in range(len(q))])
    vmask = is_valid_query[group]
    tmask = ~vmask
    if len(np.unique(y[tmask])) < 2:
        raise ValueError("Training candidates need both positive and negative labels. Inspect data and blocking.")
    if not vmask.any():
        raise ValueError("Validation contains no candidates. Inspect data/blocking.")
    params = dict(objective="binary", verbosity=-1, learning_rate=0.05,
                  num_leaves=23, min_data_in_leaf=20, lambda_l2=2.0,
                  feature_fraction=1.0, seed=args.seed, num_threads=args.threads,
                  deterministic=True, force_col_wise=True)
    print(f"Training LightGBM on {int(tmask.sum()):,} pairs; validating {len(valid_set):,} entities", flush=True)
    train_set = lgb.Dataset(X[tmask], label=y[tmask], feature_name=FEATURES)
    model = lgb.train(params, train_set, num_boost_round=args.rounds)
    probabilities = model.predict(X[vmask], num_threads=args.threads)
    vg, vy = group[vmask], y[vmask]
    valid_indices = sorted(valid_set)
    truth_sizes = np.array([len(truth[r.eid]) for r in q])
    thresholds = np.unique(np.concatenate([np.linspace(0.02, 0.98, 97), [0.99, 0.995, 0.999, 1.000001]]))
    trials = []
    for threshold in thresholds:
        score, _, _ = entity_scores(vg, vy, probabilities, threshold, valid_indices, truth_sizes, len(q))
        trials.append((score, float(threshold)))
    best_score, best_threshold = max(trials)  # More conservative threshold breaks exact score ties.
    _, pred, tp = entity_scores(vg, vy, probabilities, best_threshold, valid_indices, truth_sizes, len(q))
    val_truth_count = int(truth_sizes[valid_indices].sum())
    val_hits = sum(hits[i] for i in valid_indices)
    val_counts = [counts[i] for i in valid_indices]
    non_singletons = [i for i in valid_indices if truth_sizes[i] > 0]
    singleton_indices = [i for i in valid_indices if truth_sizes[i] == 0]
    countries = {}
    for country in sorted({q[i].country for i in valid_indices}):
        ids = [i for i in valid_indices if q[i].country == country]
        score, _, _ = entity_scores(vg, vy, probabilities, best_threshold, ids, truth_sizes, len(q))
        countries[country] = {"entities": len(ids), "macro_f0_5": score}
    metrics = {
        "validation_macro_f0_5_tuning_estimate": best_score,
        "threshold": best_threshold,
        "validation_entities": len(valid_indices),
        "candidate_link_recall": val_hits / val_truth_count if val_truth_count else None,
        "candidate_macro_recall_non_singletons": float(np.mean([hits[i]/truth_sizes[i] for i in non_singletons])) if non_singletons else None,
        "average_candidates_validation": float(np.mean(val_counts)),
        "max_candidates_validation": max(val_counts),
        "reduction_ratio_validation": 1-sum(val_counts)/(len(valid_indices)*len(targets)),
        "singleton_accuracy": float(np.mean(pred[singleton_indices] == 0)) if singleton_indices else None,
        "by_country": countries,
        "note": "Threshold chosen on this validation set; score is a tuning estimate, not an untouched holdout or leaderboard score. No France-labelled validation exists in the official training set.",
    }
    print(json.dumps(metrics, indent=2), flush=True)
    (args.out / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.out / "threshold_search.tsv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["threshold", "validation_macro_f0_5"])
        w.writerows((t, s) for s, t in trials)
    # Refit using all labelled training candidate pairs, never test labels.
    print("Refitting on all training candidate pairs", flush=True)
    del model, train_set
    gc.collect()
    full_set = lgb.Dataset(X, label=y, feature_name=FEATURES)
    model = lgb.train(params, full_set, num_boost_round=args.rounds)
    model.save_model(str(args.out / "model.txt"))
    del full_set, X, y, group, blocker, targets, q, truth
    gc.collect()
    test_q, test_targets = load_split(args.data, "test")
    blocker = Blocker(test_targets, args.top_k, args.max_postings, args.prefilter)
    test_counts = []
    with (args.out / "matching_results.tsv").open("w", encoding="utf-8", newline="") as fm, \
         (args.out / "candidate_pairs.tsv").open("w", encoding="utf-8", newline="") as fc:
        mw = csv.writer(fm, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE)
        cw = csv.writer(fc, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE)
        mw.writerow(["source1_entity_id", "matched_entity_ids"])
        cw.writerow(["source1_entity_id", "candidate_entity_ids"])
        # Batch model calls to avoid one expensive inference invocation per query.
        for start_i in range(0, len(test_q), args.batch_size):
            batch = test_q[start_i:start_i + args.batch_size]
            candidate_lists, feat = [], []
            for query in batch:
                candidates = blocker.candidates(query)
                candidate_lists.append(candidates)
                test_counts.append(len(candidates))
                feat.extend(features(query, test_targets[j]) for j in candidates)
            probs = model.predict(np.asarray(feat, dtype=np.float32), num_threads=args.threads) if feat else []
            cursor = 0
            for query, candidates in zip(batch, candidate_lists):
                # This is precisely the candidate list scored by the trained matching model.
                ids = [test_targets[j].eid for j in candidates]
                matches = [eid for eid, p in zip(ids, probs[cursor:cursor+len(ids)]) if p >= best_threshold]
                cursor += len(ids)
                cw.writerow([query.eid, ",".join(sorted(ids))])
                mw.writerow([query.eid, ",".join(sorted(matches))])
            print(f"Test predictions: {min(start_i+args.batch_size, len(test_q)):,}/{len(test_q):,}", flush=True)
    metrics["test_entities"] = len(test_q)
    metrics["average_candidates_test"] = float(np.mean(test_counts)) if test_counts else 0.0
    metrics["elapsed_seconds"] = time.perf_counter() - start
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "func"}
    config["python_version"] = platform.python_version()
    config["platform"] = platform.platform()
    config["package_versions"] = {p: importlib.metadata.version(p) for p in ["numpy", "scipy", "rapidfuzz", "lightgbm"]}
    config["data_sha256"] = manifest(args.data)
    config["feature_names"] = FEATURES
    (args.out / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.out / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    validate(args.data, args.out)
    print(f"Finished. Leaderboard file: {args.out / 'matching_results.tsv'}", flush=True)


def validate(data: Path, output: Path):
    # Independent local format check; run Amazon's official validator as well.
    q, targets = load_split(data, "test")
    expected, allowed = {r.eid for r in q}, {r.eid for r in targets}
    maps = []
    for filename, column in [("matching_results.tsv", "matched_entity_ids"), ("candidate_pairs.tsv", "candidate_entity_ids")]:
        path = output / filename
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            if next(reader, None) != ["source1_entity_id", column]:
                raise ValueError(f"{filename}: headers must match exactly")
        mapping = {}
        for row in read_table(path, ["source1_entity_id", column]):
            eid = row["source1_entity_id"]
            if eid in mapping:
                raise ValueError(f"{filename}: duplicate row {eid}")
            ids = set(parse_ids(row[column]))
            if ids - allowed:
                raise ValueError(f"{filename}: invalid target ID for {eid}")
            mapping[eid] = ids
        if set(mapping) != expected:
            raise ValueError(f"{filename}: missing or extra Source 1 entities")
        maps.append(mapping)
    for eid in expected:
        if not maps[0][eid].issubset(maps[1][eid]):
            raise ValueError(f"Matches outside candidate list for {eid}")
    print(f"LOCAL FORMAT PASS: {len(expected):,} Source 1 entities in both files.", flush=True)


def package(args):
    validate(args.data, args.out)
    if not args.documentation.is_file() or args.documentation.suffix.lower() != ".md":
        raise ValueError("Supply your completed official Documentation_template.md, not the starter notes.")
    root = Path(__file__).resolve().parents[1]
    code_prefix = "code/business_entity_resolution/"
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    cfg_path = args.out / "run_config.json"
    if not cfg_path.is_file():
        raise ValueError("run_config.json is missing; package the complete output of a successful run.")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    command = "python src/pipeline.py run --data /path/to/dataset --out reproduced"
    for name in ("top_k", "max_postings", "prefilter", "rounds", "seed", "threads", "max_pairs", "batch_size"):
        command += f" --{name.replace('_','-')} {cfg[name]}"
    with zipfile.ZipFile(args.destination, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for filename in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(args.out / filename, f"output/{filename}")
        for path in sorted((root / "src").glob("*.py")):
            z.write(path, code_prefix + "src/" + path.name)
        for path in sorted((root / "tests").glob("*.py")):
            z.write(path, code_prefix + "tests/" + path.name)
        for filename in ("README.md", "requirements.txt", "LICENSE"):
            z.write(root / filename, code_prefix + filename)
        for filename in ("methodology_notes.md", "TESTING.md"):
            z.write(root / "docs" / filename, code_prefix + "docs/" + filename)
        for filename in ("run_config.json", "validation_metrics.json", "threshold_search.tsv", "model.txt"):
            z.write(args.out / filename, code_prefix + "artifacts/" + filename)
        z.writestr(code_prefix + "REPRODUCE_THIS_RUN.md", "# Exact reproduction command\n\nRun from this code folder, with the pinned environment. Replace only the data path.\n\n```sh\n" + command + "\n```\n\nThe data SHA-256 hashes are in artifacts/run_config.json. Both outputs are regenerated in reproduced/.\n")
        z.write(args.documentation, "Documentation_template.md")
    print(f"Created final package: {args.destination}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit", help="Check all seven source/label TSVs and print dataset sizes")
    a.add_argument("--data", type=Path, required=True)
    a.set_defaults(func=lambda x: audit(x.data))
    r = sub.add_parser("run", help="Validate locally, train, tune threshold, refit, and predict every test entity")
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True, help="New output directory; refuses to overwrite existing runs")
    r.add_argument("--top-k", type=int, default=30)
    r.add_argument("--max-postings", type=int, default=128)
    r.add_argument("--prefilter", type=int, default=200)
    r.add_argument("--rounds", type=int, default=250)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--max-pairs", type=int, default=1500000)
    r.add_argument("--batch-size", type=int, default=256)
    r.set_defaults(func=run)
    v = sub.add_parser("validate", help="Local format validator, not a leaderboard scorer")
    v.add_argument("--data", type=Path, required=True)
    v.add_argument("--out", type=Path, required=True)
    v.set_defaults(func=lambda x: validate(x.data, x.out))
    k = sub.add_parser("package", help="Build the official final ZIP layout without including the dataset")
    k.add_argument("--data", type=Path, required=True)
    k.add_argument("--out", type=Path, required=True)
    k.add_argument("--documentation", type=Path, required=True)
    k.add_argument("--destination", type=Path, required=True)
    k.set_defaults(func=package)
    return p


def main():
    args = build_parser().parse_args()
    try:
        if args.command == "run":
            for name in ("top_k", "max_postings", "prefilter", "rounds", "threads", "max_pairs", "batch_size"):
                if getattr(args, name) < 1:
                    raise ValueError(f"--{name.replace('_', '-')} must be positive")
        args.func(args)
    except (ValueError, OSError, MemoryError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
