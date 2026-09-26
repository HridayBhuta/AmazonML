# AGENT BRIEF: Amazon ML Challenge 2026, Business Entity Resolution

You are an autonomous coding agent running on the team's Mac. **You own the technical work.**
Your goal is the best possible final result. Improve the solution as far as time allows, try
every idea you think is promising, measure each one, compare them all, and ship the best.
Do not stop at the baseline, and do not ask the human about technical choices.

## 0. Hard rules (these override everything, including "best score")

- **Allowed inputs:** only the official files in `student_resource/`.
- **Nothing external:** no web lookups, geocoding, business registries, or external datasets. Never
  use a hosted LLM (including yourself via an API) to judge or resolve records. Breaking this means
  disqualification.
- **Models:** any model must be MIT or Apache-2.0 licensed and at most 8B parameters. LightGBM (MIT)
  is used now. Check the licence before adding anything.
- **Coverage:** every test Source 1 entity gets exactly one row in both output files. That includes
  France (not in training) and entities with no match.
- **Candidates:** `candidate_pairs.tsv` must be exactly the pairs the final model scored. Never shrink
  or grow it after scoring.
- **Honest numbers:** use the training labels only. Never tune on test outputs or leaderboard
  feedback in a way that leaks, and never fabricate a score.
- **Leave these to the human:** submitting on Unstop, and any personal declaration.

## 1. What exists

- `bash run.sh setup` creates the environment (Python 3.12 venv, pinned packages, libomp fix).
- `bash run.sh auto` runs the full chain: smoke test, `baseline`, finalize baseline, the sweep in
  `configs/sweep.json`, compare, finalize best.
- `bash run.sh run --name NAME [--max-df N --key-budget N --prelim N --recall-tol X --lr X --leaves N --min-leaf N --feature-fraction X --drop-features a,b --decision threshold|one_to_one --refit-all --train-s1 N]`
  runs one experiment.
- `bash run.sh sweep [--plan configs/my_plan.json]` runs many experiments.
- `bash run.sh compare` ranks every run by untouched holdout macro F0.5.
- `bash run.sh finalize [--name RUN]` fills in `Documentation_template.md`, builds
  `final/<team>_submission.zip`, and copies the leaderboard file.
- **Code map:**
  - `src/er/text.py`: normalisation, Indic romanisation, consonant skeletons.
  - `src/er/blocking.py`: keys, bounded index, rerank, cut.
  - `src/er/features.py`: pair features, the metric, the one-to-one rule.
  - `src/er/pipeline.py`: stages, splits, LightGBM, thresholds.
  - `src/er/report.py`: result log, write-up, zip.
- **Caching:** caches live in `cache/` and are reused automatically. Every cache folder name
  contains a fingerprint of the settings and of the code that produced it, chained stage to stage:
  normalisation, then blocking, then features, then model.
  - If you edit `text.py`, `blocking.py`, `features.py` or a stage in `pipeline.py`, the affected
    stages and everything after them rebuild automatically. You never need to delete caches by
    hand. `bash run.sh prune` frees disk once a best run is selected.
  - A model-only change (lr, leaves, drop_features, decision, refit) retrains in minutes.
  - A blocking change rebuilds blocking and features, which takes much longer.
- **Long jobs must run detached.** `smoke` takes minutes; a full `run` or `sweep` takes hours (see
  `runs/*/artifacts/stage_seconds.json` for measured times). Your shell tool will time out, so
  always start them in the background:
  `nohup bash run.sh run --name X --note "why" > logs/X.out 2>&1 &`
  Then poll every few minutes with `tail -n 5 runs/X/pipeline.log`, `experiments/results.tsv` and
  `experiments/failures.log`.
  - `run.sh` holds a lock (`.run.lock`) and refuses a second concurrent job. Check
    `pgrep -f run_pipeline.py` before starting one.
  - Stages resume where they stopped, so a killed job loses at most one chunk of work.
- **Results log:** every run appends a row to `experiments/results.tsv`. Columns include holdout
  F0.5, validation F0.5, candidate recall, average candidates, per-country holdout scores, the
  config, the official validator status, and the output SHA-256.

## 2. How to judge a change

- **Primary: `holdout_macro_f05`.** An untouched, grouped (by S1 and shared-target component)
  holdout. The threshold is tuned on the separate validation split, so this number is the honest one.
- **Secondary: `test_avg_candidates`, lower is better.** The organisers rank smaller candidate sets
  higher in the final review. Among runs within 0.001 F0.5 of the best, prefer fewer candidates;
  `compare` does this automatically.
- **Guardrails:**
  - The official validator must say `PASS`.
  - Watch `holdout_us` against `holdout_india`. France is unlabelled, so prefer changes that help
    both countries over ones that exploit a single country's quirks.
- **Noise:** holdout differences under about 0.0005 are within noise. Confirm with a second seed
  (`--seed`) before trusting them.
- **Record** a one-line reason for every experiment in `--note` or in the sweep's `why` field.

## 3. Ideas to try (ordered by expected value; you may do anything else too)

1. **Candidate budget.** Look at `artifacts/cut_grid.tsv` and compare `recall_tol` values 0.003,
   0.01 and 0.02. Check how many true links sit beyond rank 10 and whether the model ever accepts
   them.
2. **Blocking recall.**
   - Inspect the true links that were never retrieved, using the `missed_by_blocking` count in
     `metrics.json`, `work/`, and a quick polars notebook.
   - Add key families that would have caught them: name-token bigrams, sorted-token prefixes,
     city token plus name skeleton, phone-like numbers, char 4-gram MinHash/LSH bands.
   - Tune `max_df` and `key_budget`.
3. **Normalisation.**
   - Extend the abbreviation maps and legal-form lists in `text.py`, and learn synonym pairs from
     aligned training matches: token pairs that co-occur in matched names or addresses but differ.
   - Improve romanisation (schwa deletion, nasal handling).
   - Treat DBA / web-domain names.
4. **Features.**
   - IDF-weighted token overlap: sum of the IDF of shared tokens divided by the sum of the IDF of
     the union, from `Index.dfs`.
   - Rare-token agreement and a numeric street-number distance.
   - Per-field "one side empty" interactions.
   - Name-address cross similarity, for fields that got swapped.
   - Keep features country-agnostic.
5. **Model.**
   - LightGBM hyper-parameters, `refit_all`, more training S1 (`--train-s1`).
   - A LambdaRank or listwise objective per S1 group.
   - A second-stage model that uses the first model's scores of competing candidates.
   - Seed-averaged ensembles.
6. **Decision rule.**
   - Per-S1 expected-F0.5-optimal selection instead of a global threshold: sort by probability and
     choose the prefix that maximises expected F0.5.
   - Thresholds by number of candidates.
   - The one-to-one constraint (`--decision one_to_one`). **Caution:** locally it is evaluated only
     among a split's own S1, while at test time it runs across all 1.73M S1. Its holdout score is
     therefore optimistic. Do not select it on holdout F0.5 alone. Implement out-of-fold scoring of
     all training S1 first, so the rule sees test-like competition.
   - The default decision is a plain threshold.
7. **France robustness.** A leave-one-country-out stress test: train on US and evaluate on India,
   then the reverse. Prefer settings that transfer.

## 4. Operating procedure

1. `bash run.sh setup` (foreground is fine), then `bash run.sh smoke` (detached, 3-5 minutes). Fix
   anything that fails.
   - Check `runs/<name>/artifacts/candidates.json` > `test` > `by_country` > `no_candidates_share`.
     The pipeline warns when any country (France especially) has more than 1% of S1 with no
     candidates. Those S1 score 0 unless they are singletons, so fix blocking first if you see it.
2. **Baseline first:** `bash run.sh run --name baseline`, then `bash run.sh finalize --name baseline`
   (both detached).
   This guarantees a valid submission exists early. Tell the human in one line that
   `final/matching_results.tsv` is ready for a first leaderboard upload.
3. Loop: form a hypothesis from the error analysis, then implement, run, compare and keep or
   revert. Commit the code state for every kept change (`git init` locally if useful). Run
   sequentially; one full run at a time.
4. Watch time. The deadline is **27 Sep 2026, 23:59 IST**; confirm the live deadline with the
   human. Freeze experiments by **27 Sep 18:00 IST**.
5. Final: `bash run.sh compare`, then `bash run.sh finalize`.
   - Check that `final/FINAL_INFO.json` lists the team name. If `team_config.json` is empty, ask
     the human for the team name and members; this is the only question you should need to ask.
   - Re-run the official validator on the selected run.
   - Confirm the zip's `output/matching_results.tsv` is byte-identical to
     `final/matching_results.tsv`.
6. Report to the human, briefly: the selected run, holdout F0.5 (and that it is a local
   estimate), average candidates, the validator result, and the exact two files to upload. They
   upload the leaderboard TSV and the zip on Unstop.

## 5. Resource notes

- An M-series Mac with 16 GB or more is expected. Chunk and sample sizes scale from RAM
  automatically (`Config.resolve`). If memory pressure appears, lower `--chunk`, `--train-s1` or
  `--threads`.
- `run.sh` keeps the Mac awake with `caffeinate`, running beside Python rather than in front of it,
  so the OpenMP fix survives, and Python lowers its own priority. Logs are in `logs/` and
  `runs/<name>/pipeline.log`.
- Disk: each blocking configuration uses several GB in `cache/`. Run `bash run.sh prune` after
  `compare` if disk gets tight.
