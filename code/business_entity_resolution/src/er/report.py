"""Experiment log, comparison, official documentation fill-in and final ZIP packaging."""
from __future__ import annotations

import datetime as dt
import json
import shutil
import zipfile
from pathlib import Path

import polars as pl

from .common import sha256

ROOT = Path(__file__).resolve().parents[2]          # package root (contains src/, run.sh, ...)
RESULTS = ROOT / "experiments" / "results.tsv"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
# every setting that changes the outputs (RAM-dependent auto values are recorded resolved)
REPRO_KEYS = ("max_df", "key_budget", "prelim", "recall_tol", "max_k", "chunk", "b_block", "train_s1", "valid_s1",
              "holdout_s1", "tune_s1", "rounds", "early_stop", "lr", "leaves", "min_leaf", "feature_fraction",
              "drop_features", "decision", "refit_all", "seed", "threads")


def record(run: Path, name: str, note: str = "") -> dict:
    a = run / "artifacts"
    m = json.loads((a / "metrics.json").read_text())
    c = json.loads((a / "candidates.json").read_text())
    t = json.loads((a / "test_stats.json").read_text()) if (a / "test_stats.json").is_file() else {}
    v = json.loads((a / "validation.json").read_text()) if (a / "validation.json").is_file() else {}
    cfg = json.loads((a / "config.json").read_text())
    out = run / "output" / "matching_results.tsv"
    row = {
        "time_ist": dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M"),
        "run": name,
        "holdout_macro_f05": round(m["holdout_macro_f05_untouched"], 5),
        "valid_macro_f05_tuned": round(m["validation_macro_f05_threshold_tuned"], 5),
        "threshold": m["threshold"], "decision": m["decision"],
        "train_link_recall": round(c["train"].get("link_recall", float("nan")), 5),
        "train_avg_candidates": round(c["train"]["avg_candidates"], 3),
        "test_avg_candidates": round(c["test"]["avg_candidates"], 3),
        "K": c["K"], "G": c["G"],
        "holdout_us": round(m["holdout_by_country"].get("us", {}).get("macro_f05", float("nan")), 5),
        "holdout_india": round(m["holdout_by_country"].get("india", {}).get("macro_f05", float("nan")), 5),
        "official_validator": v.get("official_validator", "NOT RUN"),
        "matching_sha256": sha256(out) if out.is_file() else "",
        "smoke": str(bool(cfg.get("smoke"))),
        "test_rows": str(v.get("rows", "")),
        "config": json.dumps({k: cfg.get(k) for k in REPRO_KEYS}),
        "note": note,
        "public_leaderboard_score": "",
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame([row])
    if RESULTS.is_file():
        old = pl.read_csv(RESULTS, separator="\t", infer_schema=False)
        old = old.filter(pl.col("run") != name)
        df = pl.concat([old, df.cast(pl.String)], how="diagonal")
    df.cast(pl.String).write_csv(RESULTS, separator="\t")
    return row


def compare(select: bool = True) -> dict | None:
    if not RESULTS.is_file():
        print("no experiments recorded yet")
        return None
    df = pl.read_csv(RESULTS, separator="\t", infer_schema=False).with_columns(
        pl.col("holdout_macro_f05").cast(pl.Float64), pl.col("test_avg_candidates").cast(pl.Float64))
    df = df.filter((pl.col("official_validator") == "PASS") & ~pl.col("run").str.starts_with("_"))
    if "smoke" in df.columns:
        df = df.filter(pl.col("smoke").fill_null("False") != "True")
    if df.height == 0:
        print("no run has passed the official validator yet")
        return None
    best_score = df["holdout_macro_f05"].max()
    # Prefer the smallest candidate set among runs within 0.001 of the best untouched holdout score.
    pick = (df.filter(pl.col("holdout_macro_f05") >= best_score - 0.001)
              .sort(["test_avg_candidates", "holdout_macro_f05"], descending=[False, True]).row(0, named=True))
    with pl.Config(tbl_rows=100, tbl_cols=12, fmt_str_lengths=40):
        print(df.select("run", "holdout_macro_f05", "valid_macro_f05_tuned", "train_link_recall",
                        "test_avg_candidates", "threshold", "decision", "note")
                .sort("holdout_macro_f05", descending=True))
    print(f"\nSELECTED: {pick['run']} (holdout {pick['holdout_macro_f05']}, avg candidates {pick['test_avg_candidates']})")
    if select:
        (ROOT / "experiments" / "BEST.json").write_text(json.dumps(pick, indent=2))
    return pick


def _fmt(x, nd=4):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def fill_documentation(template: Path, run: Path, team: str, members: str, dest: Path):
    a = run / "artifacts"
    m = json.loads((a / "metrics.json").read_text())
    c = json.loads((a / "candidates.json").read_text())
    au = json.loads((a / "audit.json").read_text()) if (a / "audit.json").is_file() else {}
    t = json.loads((a / "test_stats.json").read_text())
    cfg = json.loads((a / "config.json").read_text())
    imp = pl.read_csv(a / "feature_importance.tsv", separator="\t").head(12)
    gt = au.get("ground_truth", {})
    tr, te = c["train"], c["test"]
    s = template.read_text(encoding="utf-8")
    today = dt.datetime.now(IST).strftime("%d %B %Y")
    s = s.replace("[Your Team Name]", team or "[Your Team Name]")
    s = s.replace("[List all team members]", members or "[List all team members]")
    s = s.replace("[Date]", today)
    sections = {
        "*Provide a brief 2-3 sentence overview of your approach and key innovations.*":
            "A CPU-only, fully offline pipeline: rule-based normalisation (including romanisation of Indic "
            "scripts and a consonant-skeleton key), a bounded IDF-weighted inverted index with an adaptive, "
            "data-chosen candidate budget, and a LightGBM (MIT) pair classifier over string, address, "
            "number and candidate-competition features, thresholded to maximise per-entity macro F0.5. "
            f"Average final candidates per Source 1 entity on test: {te['avg_candidates']:.2f}.",
        "*Key insights discovered during EDA — noise patterns, address variations, missing fields, etc.*":
            f"- Training: {tr['s1']:,} Source 1 entities against {tr['targets']:,} Source 2/3 records; test: "
            f"{te['s1']:,} against {te['targets']:,}.\n"
            f"- Singleton fraction (training Source 1 with no match): {_fmt(gt.get('singleton_fraction'))}; "
            f"matches per entity histogram: {json.dumps(gt.get('matches_per_s1_hist', {}))}.\n"
            "- Names appear in Latin and Indic scripts (Devanagari, Kannada, ...), with legal forms moved or "
            "abbreviated (Pvt/Private, Ltd/Limited, LLC prefix), junk prefixes ('--', '<<'), website-style "
            "names ('name.com'), typos and injected accents.\n"
            "- Addresses are reordered (state first), abbreviated (ST/RD/AVE, R./BD in France), sometimes the "
            "literal text 'null', or empty; state names alternate with codes (Texas/TX, HR/Haryana).\n"
            "- The test set contains France, absent from training, so every feature is language-agnostic "
            "similarity rather than a country-specific rule learned from labels.",
        "*Outline your high-level approach.*":
            "Normalise -> generate a small candidate list per Source 1 entity -> score every candidate with "
            "a gradient-boosted classifier -> keep candidates above a validation-tuned threshold "
            f"(decision rule: {m['decision']}).",
        "[Blocking + Classifier / End-to-End / Graph-Based / Hybrid, etc]": "Blocking + Classifier",
        "[Brief description of your main technical contribution]":
            "Offline Indic-script romanisation + consonant skeletons that let Devanagari/Kannada names meet "
            "their English spellings, combined with an adaptive candidate budget chosen from the "
            "recall-vs-size curve and target-side competition features.",
        "*Describe how you reduced the comparison space to a manageable candidate set.*":
            "Stage A: every record emits hashed keys, each prefixed with its own country label (open set): "
            "core-name tokens, their consonant skeletons, first/last 6 characters of the compact name, "
            "house number + street prefix, address-token skeletons and 5-6 digit postcodes. Keys shared by "
            f"more than {cfg['max_df']} target records are discarded completely. Each Source 1 record probes "
            f"its rarest keys until {cfg['key_budget']} postings, and targets are ranked by summed IDF; the top "
            f"{cfg['prelim']} are kept. Stage B reranks those with a cheap string score. Stage C keeps rank < "
            f"K={c['K']}" + ("" if c["G"] >= 1e8 else f" and score within G={c['G']} of the best candidate")
            + "; (K, G) was chosen on training data "
            f"as the smallest budget losing at most {cfg['recall_tol']} link recall. candidate_pairs.tsv is "
            "exactly the Stage C list, i.e. every pair the model scores.",
        "[e.g., PIN code, phonetic name encoding, TF-IDF, etc.]":
            "country-prefixed name tokens, consonant-skeleton (phonetic) tokens, compact-name prefix/suffix, "
            "house number + street, address skeleton tokens, postcodes; IDF-weighted voting",
        "[total]": f"{te['pairs']:,} test pairs ({te['avg_candidates']:.2f} per Source 1 on average, max "
                   f"{te['max_candidates']}; reduction ratio {te['reduction_ratio']:.8f})",
        "- **How you ensured true matches were not lost:**":
            f"- **How you ensured true matches were not lost:** measured on all {tr['s1']:,} training Source 1 "
            f"entities: link recall {_fmt(tr.get('link_recall'))}, macro recall on non-singletons "
            f"{_fmt(tr.get('macro_recall_non_singletons'))}; frequent keys are dropped wholesale (not truncated), "
            "several independent key families back each other up, and the budget was picked from the measured "
            "recall curve (artifacts/cut_grid.tsv).",
        "[e.g., Jaccard, Levenshtein, phonetic encoding]":
            "ratio / token-sort / token-set / partial ratio (RapidFuzz), Jaro-Winkler on compact names, "
            "consonant-skeleton similarity, token and skeleton Jaccard",
        "[e.g., token overlap, edit distance, PIN code matching]":
            "ratio / token-sort / token-set on normalised addresses and skeletons, token Jaccard, number-set "
            "Jaccard and conflict flag, first-number, house-number and postcode agreement",
        "- Other: []":
            "- Other: blocking IDF score and shared-key count, rank within the candidate list, gap to the best "
            "and second-best candidate, how many Source 1 entities compete for the same target and this pair's "
            "rank among them, mutual-best flag, script flags, length ratios, missing-field flags, country "
            f"equality, source flag. Top features by gain: {', '.join(imp['feature'].to_list())}.",
        "[e.g., XGBoost, Siamese Network, Transformer, etc.]":
            f"LightGBM gradient-boosted trees (MIT licence, trained from scratch on the provided labels only; "
            f"no pretrained model), {m['best_iteration']} trees, early-stopped on the validation split",
        "[e.g., F_0.5 optimization on validation set]":
            f"threshold {m['threshold']} chosen by maximising per-entity macro F0.5 (singletons included) on a "
            f"grouped validation split; decision rule '{m['decision']}' (one_to_one keeps each target only for "
            "the Source 1 entity where it scores highest)",
        "[your best validation score]":
            f"{m['holdout_macro_f05_untouched']:.4f} on an untouched grouped holdout of {m['holdout_s1']:,} "
            f"training Source 1 entities (validation, threshold-tuned: {m['validation_macro_f05_threshold_tuned']:.4f}); "
            f"by country: {json.dumps({k: round(v['macro_f05'], 4) for k, v in m['holdout_by_country'].items()})}; "
            f"singleton accuracy {_fmt(m.get('holdout_singleton_accuracy'))}. No France labels exist, so France "
            "performance is unmeasured.",
    }
    for k, v in sections.items():
        s = s.replace(k, v, 1)
    e = m.get("error_analysis", {})
    s = s.replace("[brief description]",
                  f"{e.get('false_positive_pairs', 'n/a')} false-positive pairs on the holdout; share whose target "
                  f"truly belongs to a different Source 1 entity (sibling/branch confusion): "
                  f"{_fmt(e.get('fp_share_target_belongs_to_another_s1'))}; share with name ratio >= 90 (same or "
                  f"near-identical names, e.g. chains/common names): {_fmt(e.get('fp_share_name_ratio_ge_90'))}; "
                  f"share with conflicting address numbers: {_fmt(e.get('fp_share_number_conflict'))}; share with an "
                  f"empty target address: {_fmt(e.get('fp_share_target_addr_missing'))}.", 1)
    s = s.replace("[brief description]",
                  f"of {e.get('holdout_true_links', 'n/a')} true holdout links, {e.get('missed_by_blocking', 'n/a')} "
                  f"were never retrieved by blocking and {e.get('found_but_rejected', 'n/a')} were retrieved but "
                  f"rejected. Among rejected ones: Indic-script target names "
                  f"{_fmt(e.get('fn_share_indic_script_target'))}, heavy name rewrites (ratio < 60) "
                  f"{_fmt(e.get('fn_share_name_ratio_lt_60'))}, very different addresses (ratio < 50) "
                  f"{_fmt(e.get('fn_share_addr_ratio_lt_50'))}, empty target address "
                  f"{_fmt(e.get('fn_share_target_addr_missing'))}.", 1)
    s = s.replace("*Summarize your approach, key achievements, and lessons learned in 2-3 sentences.*",
                  "Blocking quality bounds recall, so most effort went into script-robust keys and a measured "
                  "candidate budget; a precision-oriented threshold then maximises F0.5. Everything runs offline "
                  "from the supplied files with a single command.")
    s = s.replace("*Include any additional charts, graphs, or detailed results.*",
                  "Detailed artefacts shipped in code/business_entity_resolution/artifacts/: metrics.json, "
                  "candidates.json, cut_grid.tsv (recall vs candidate budget), threshold_search.tsv, "
                  "feature_importance.tsv, audit.json, config.json, experiments_results.tsv (every experiment).\n\n"
                  f"Test predictions: {t['test_matches']:,} matched pairs over {t['test_s1_with_matches']:,} of "
                  f"{t['test_s1']:,} Source 1 entities.\n\n"
                  "**External data / AI assistance statement:** no external database, API, geocoder, business "
                  "registry, web data or hosted model was used to resolve entities; all learning uses only the "
                  "provided training labels. Code was developed with the help of an AI coding assistant.")
    opts = " ".join(f"--{k.replace('_', '-')} {cfg[k]}" for k in REPRO_KEYS
                    if k not in ("drop_features", "refit_all") and cfg.get(k) is not None)
    opts += f" --drop-features {cfg['drop_features']}" if cfg.get("drop_features") else ""
    opts += " --refit-all" if cfg.get("refit_all") else ""
    code_text = (
        "Structure of `code/business_entity_resolution/`:\n\n"
        "- `src/run_pipeline.py`: single entry point (commands: smoke, run, sweep, compare, finalize).\n"
        "- `src/er/text.py`: normalisation, Indic romanisation, consonant skeletons.\n"
        "- `src/er/blocking.py`: blocking keys, bounded inverted index, rerank, candidate cut.\n"
        "- `src/er/features.py`: pair features, macro F0.5 metric, one-to-one rule.\n"
        "- `src/er/pipeline.py`: resumable stages (prepare -> block -> cut -> features -> train -> predict -> validate).\n"
        "- `src/er/report.py`: experiment log, this document, zip packaging.\n"
        "- `run.sh` / `RUN_ME.command`: environment setup (Python 3.12 + pinned requirements) and run.\n"
        "- `artifacts/`: config, metrics, model, threshold and candidate-budget searches, logs.\n\n"
        "Reproduce both output files from the official data:\n\n```bash\nbash run.sh setup\n"
        f".venv/bin/python src/run_pipeline.py run --data /path/to/student_resource --name {run.name} {opts}\n"
        f"# outputs: runs/{run.name}/output/matching_results.tsv and candidate_pairs.tsv\n```\n\n"
        "Runtime and peak memory per stage are recorded in artifacts/pipeline.log.")
    start = s.find("### A. Code Artefacts")
    end = s.find("### B. Additional Results")
    if start >= 0 and end > start:
        s = s[:start] + "### A. Code Artefacts\n" + code_text + "\n\n" + s[end:]
    else:
        s += "\n\n### A. Code Artefacts\n" + code_text + "\n"
    dest.write_text(s, encoding="utf-8")
    return dest


def build_zip(run: Path, doc: Path, team: str, dest_dir: Path) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (team or "TEAM")).strip("_") or "TEAM"
    dest_dir.mkdir(parents=True, exist_ok=True)
    z = dest_dir / f"{safe}_submission.zip"
    if z.exists():
        z.rename(z.with_suffix(f".{dt.datetime.now().strftime('%H%M%S')}.old.zip"))
    code = "code/business_entity_resolution/"
    with zipfile.ZipFile(z, "x", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            zf.write(run / "output" / f, f"output/{f}")
        for p in sorted((ROOT / "src").rglob("*.py")):
            zf.write(p, code + p.relative_to(ROOT).as_posix())
        for p in sorted((ROOT / "tests").rglob("*.py")):
            zf.write(p, code + p.relative_to(ROOT).as_posix())
        for f in ("README.md", "requirements.txt", "LICENSE", "AGENT_BRIEF.md", "configs/sweep.json",
                  "team_config.json"):
            if (ROOT / f).is_file():
                zf.write(ROOT / f, code + f)
        for f in ("run.sh", "RUN_ME.command"):
            info = zipfile.ZipInfo(code + f)
            info.external_attr = 0o755 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, (ROOT / f).read_bytes())
        for p in sorted((run / "artifacts").glob("*")):
            if p.is_file():
                zf.write(p, code + "artifacts/" + p.name)
        if RESULTS.is_file():
            zf.write(RESULTS, code + "artifacts/experiments_results.tsv")
        zf.write(run / "pipeline.log", code + "artifacts/pipeline.log")
        zf.write(doc, "Documentation_template.md")
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        assert zf.read("output/matching_results.tsv") == (run / "output" / "matching_results.tsv").read_bytes()
        assert zf.read("output/candidate_pairs.tsv") == (run / "output" / "candidate_pairs.tsv").read_bytes()
        assert any(n.startswith(code + "src/") for n in names) and code + "README.md" in names
        assert code + "requirements.txt" in names and "Documentation_template.md" in names
        assert not any(n.endswith((".parquet",)) or "dataset/" in n for n in names)
    return z


def finalize(run: Path, template: Path, team: str, members: str, final_dir: Path, data: Path | None = None) -> dict:
    """data: the REAL official student_resource folder. Both validators are re-run on the selected run's
    outputs against it before anything is packaged (protects against packaging a sample run)."""
    from .pipeline import check_outputs, official_validator
    if data is not None:
        own = check_outputs(run / "output", data / "dataset" / "test")
        off = official_validator(data, run / "output", run / "artifacts" / "official_validator_final.log")
        if off["official_validator"] != "PASS":
            raise SystemExit(f"Refusing to package {run.name}: official validator on the real data says "
                             f"{off['official_validator']} (see artifacts/official_validator_final.log)")
        (run / "artifacts" / "validation_final.json").write_text(json.dumps({**own, **off}, indent=2))
    final_dir.mkdir(parents=True, exist_ok=True)
    doc = fill_documentation(template, run, team, members, final_dir / "Documentation_template.md")
    z = build_zip(run, doc, team, final_dir)
    lb = final_dir / "matching_results.tsv"
    shutil.copy2(run / "output" / "matching_results.tsv", lb)
    info = {"run": run.name, "zip": str(z), "zip_sha256": sha256(z), "leaderboard_file": str(lb),
            "leaderboard_sha256": sha256(lb), "documentation": str(doc),
            "team_name_filled": bool(team), "members_filled": bool(members)}
    (final_dir / "FINAL_INFO.json").write_text(json.dumps(info, indent=2))
    (final_dir / "SUBMIT_THESE.txt").write_text(
        "Submit on Unstop (team leader account):\n"
        f"1. Leaderboard upload: {lb}\n   sha256 {info['leaderboard_sha256']}\n"
        f"2. Final package upload: {z}\n   sha256 {info['zip_sha256']}\n"
        "The matching_results.tsv inside the zip is byte-identical to file 1.\n"
        + ("" if team and members else "WARNING: team name/members were not provided - edit team_config.json and run: bash run.sh finalize\n"))
    return info
