# HANDOFF — Amazon ML Challenge 2026 (execute autonomously)

> **UPDATE, 26 Sep 2026 ~19:30 IST. This supersedes §3 and §4 below.** A complete, tested,
> memory-safe pipeline for a MacBook (M-series) now exists:
>
> - **Code only:** `D:\Amazon_ML\amazon_ml_er_mac.zip` (56 KB).
> - **Code plus the official data zip, one file:** `D:\Amazon_ML\amazon_ml_er_mac_WITH_DATA.zip` (1.09 GB).
> - **Source folder:** `D:\Amazon_ML\mac_package\amazon_ml_er\`.
> - **On the Mac:** unzip into the home folder, then run `bash run.sh`. That performs setup, the
>   smoke test, the baseline, finalize, the sweep, compare, and finalize of the best run.
> - **AI agent on the Mac:** read `amazon_ml_er/AGENT_BRIEF.md`. It gives full autonomy to
>   experiment, with hard rules. Start long jobs detached.
> - **Testing:** 10/10 unit tests pass. The smoke test on real-data samples passes the official
>   validator. An 8-agent adversarial review found 27 real defects, and all were fixed.
> - **Data zip check:** the user-provided `C:\Users\Yuval\Downloads\6ab10eb3b23ba_student_resource.zip`
>   was verified byte-identical (CRC32 and size) to the Drive download.
> - **Still not done:** any full-data run, any Unstop leaderboard upload, and any final-zip upload.
>   Uploads need the team leader signed in to Unstop.

You are a Claude Code agent taking over this project for **Yuval Doshi**. Read this file, then
execute. Do **not** reply with a plan or a tutorial, and do **not** ask the human to run commands.
The human's involvement is limited to the items in §7. Everything else is yours.

Last updated: 2026-09-26 ~17:50 IST by the previous Claude session.

---

## 1. Links

| What | Link / path |
|---|---|
| Competition page (Unstop) | https://unstop.com/hackathons/crp-amazon-ml-challenge-2026-amazon-1743604 |
| Unstop login (candidate) | https://unstop.com/auth/login?login_as=candidate |
| Official data (Google Drive, owner f20230394@goa.bits-pilani.ac.in) | https://drive.google.com/drive/folders/1eQFY8qHPADngQqYf4NCWW0TTHmpJwEz_ |
| Registered account (from user snapshot) | YUVAL DOSHI, yuvaldoshi@gmail.com — team-leader role NOT yet verified |
| Deadline (user snapshot, NOT live-verified) | **27 Sep 2026, 23:59 IST** |

Original brief (read fully): `D:\Amazon_ML\amazon_ml_claude_handoff\CLAUDE_START_HERE.md`
Original problem statement: `D:\Amazon_ML\amazon_ml_claude_handoff\PROBLEM_STATEMENT.md`
Official README (same rules, from the data folder): `D:\Amazon_ML\amazon_ml_starter\student_resource\README.md`

All rules and boundaries in `CLAUDE_START_HERE.md` still apply. The key ones:
- **No external lookups.** No hosted LLM, geocoding, or web search may be used to resolve records.
- **Keep every test Source 1 row**, including France entities and singletons.
- `candidate_pairs.tsv` must be the **exact** set of candidates the model scored.
- **No paid compute.** Do not publish the data anywhere.
- **Never fabricate** a score, validation result, submission, or receipt.

---

## 2. Current state (verified on disk)

- **Workspace:** `D:\Amazon_ML`. **Project root:** `D:\Amazon_ML\amazon_ml_starter` (a copy of the starter; the pristine copy is in `amazon_ml_claude_handoff\`).
- **Machine:**
  - Local Windows 11 laptop with **8 logical CPUs** and **7.7 GB RAM**.
  - No CUDA GPU (Intel UHD only). 773 GB free on D:.
  - Shells: Git Bash and PowerShell, both on the host.
- **Python:** `D:\Amazon_ML\amazon_ml_starter\.venv\Scripts\python.exe` (CPython 3.12.7).
  - Installed: numpy 2.3.5, scipy 1.17.0, RapidFuzz 3.14.3, lightgbm 4.6.0, **polars 1.44.2, duckdb 1.1.1, pyarrow 17.0.0**.
  - The last three were added for memory-efficient processing and are not yet in `requirements.txt`. Pin them when you use them.
- **Unit tests:** 5/5 pass (`logs/unittest_initial.log`).
- **Official data: DOWNLOADED and COMPLETE.**
  - Location: `amazon_ml_starter\student_resource\`.
  - Byte sizes match Drive (`logs/download.log`). SHA-256 values are in `evidence/official_files_sha256.txt`. Drive file IDs are in `evidence/drive_file_ids.txt`.

  | File | Data rows |
  |---|---|
  | train_source1.tsv | 2,206,821 |
  | train_source2.tsv | 5,034,616 |
  | train_source3.tsv | 5,285,603 |
  | train_ground_truth.tsv | 2,206,821 (one row per train S1) |
  | test_source1.tsv | **1,732,544** (every one needs a row in both outputs) |
  | test_source2.tsv | 4,887,273 |
  | test_source3.tsv | 5,082,316 |

- **Official helper files are present:**
  - `student_resource\utils\validate_submission.py`: stdlib only. Run it from `student_resource\`. `--check-ids` is optional and memory-heavy.
  - `student_resource\Documentation_template.md`: the official template. Fill it in; keep its headings.
- **Sample rows:**
  - Columns are `entity_id, business_name, business_address, country`.
  - US addresses look like `1795 Westchester Drive, High Point, NC`. Components are sometimes reordered, e.g. `OH, Columbus, 5559 Orville Avenue`.
  - India addresses are long and full of landmarks, e.g. `H.No.16-11-23/37/A, 2Nd Floor, ..., Hyderabad, Telangana`.
- **Portal: NOT accessed yet.**
  - The Claude built-in browser is not logged in to Unstop.
  - Claude-in-Chrome had no connected browser.
  - So the team name, team-leader role, live deadline, and submission quota are all unverified.
- **Leaderboard submissions:** none. **Final package:** not created.
- **State files:**
  - `amazon_ml_starter\RUN_STATE.json` and `amazon_ml_starter\STATUS.md`. Update both after every milestone.
  - Keep logs in `logs\`, experiments in `experiments\experiment_log.tsv`, and portal evidence in `evidence\`.

- **Draft file:** `src\fast_audit.py` is a DuckDB audit written but **never run**. It imports a helper `er_common` (with `connect()` and `read_tsv_sql()`) that does not exist yet. Write that helper or replace the script.
  - The helper should read the TSVs with `delim='\t'`, `quote=''`, `header=true`, `all_varchar=true`.
  - Set `memory_limit`, `threads`, and `temp_directory='D:/Amazon_ML/tmp'`.
- **The user stopped local compute on 26 Sep, about 17:55 IST.** At that moment the PC had only **0.9 GB of free RAM** (Chrome, Claude, ChatGPT, Edge, and Media Player were open). Before any heavy job, confirm with the user that compute on this PC is OK, or use the machine the user designates.

---

## 3. CRITICAL technical finding: the starter will run out of memory here

The starter `src/pipeline.py` has three blockers on this dataset:
- It holds every record in memory as Python `Record` objects with frozensets.
- It builds a Python dict inverted index over about 10.3M train targets.
- It refuses to run above 1.5M training pairs. 2.2M train S1 × top-k 30 is about 66M pairs, far above that guard.

On a 7.7 GB laptop it **will crash or thrash the PC**. The user explicitly asked that the PC must not hang. Redesign before any full run.

**Required safety limits for every heavy job:**
- Start it as a background process at **BelowNormal** priority. For example, in PowerShell: `Start-Process -PriorityClass BelowNormal`, or set `(Get-Process -Id $pid).PriorityClass='BelowNormal'` inside the script.
- Use at most **6 threads**. Set `POLARS_MAX_THREADS=6` and `OMP_NUM_THREADS=6`, and use LightGBM `num_threads=6`.
- For DuckDB, set `SET memory_limit='4GB'; SET threads=6; SET temp_directory='D:/Amazon_ML/tmp'`.
- Stream and chunk the work. Never materialize all pairs at once.
- Watch the process's working set. Kill it if **total RAM use exceeds about 85%**, then redesign.
- Record the PID in RUN_STATE.json. Before starting a job, check whether one is already running.

**Recommended memory-safe design** (you may adapt it, but measure everything):
1. **Normalize with polars:** NFKD, strip marks, lowercase, `&`→`and`, remove punctuation.
   - Expand common abbreviations: `st→street`, `rd→road`, `ave→avenue`, `pvt→private`, `ltd→limited`, `corp→corporation`, `inc`, `llc`, and so on.
   - Build the abbreviation dictionary by hand or learn it from train pairs only.
   - Strip legal suffixes into a separate "core name".
   - Extract numbers: the house number and the ZIP (5 digits) or PIN (6 digits).
   - Write normalized parquet to `work\`, both train and test.
2. **Blocking: a bounded inverted index in DuckDB or polars, never all-pairs.**
   - **Keys:**
     - rare core-name tokens, with document frequency ≤ a cap;
     - a sorted core-name token signature;
     - (postcode, first name token);
     - (house number, street token);
     - optionally name character 3-gram MinHash/LSH bands.
   - **Order of work:** join S1 keys to target keys **in chunks of about 100k S1 rows**, and only within the same normalized country. Country is an open set, so do not hard-code a list. For missing or unknown countries, fall back to no country filter.
   - **Scoring:** score each (S1, target) pair by summed IDF of shared keys. Then rerank with cheap RapidFuzz (`WRatio` on the name plus `token_set` on the address). Keep an **adaptive top-k**: at most K (try 10–20), and drop candidates far below the best.
   - **Measure on train:** candidate link recall, macro recall on non-singletons, and the average candidate count. Smaller candidate sets are rewarded in the final review.
3. **Features:** the starter's 21 features, plus these:
   - core-name similarity; postcode equal or conflicting; house-number equal or conflicting;
   - token-IDF-weighted Jaccard; rank of the candidate within its S1 list; score gap to the top candidate; number of candidates;
   - whether the candidate is the best match of that S2/S3 record back to any S1 (mutual best).
   - Compute them in chunks and write them to parquet.
4. **Model:** LightGBM (MIT licence).
   - **Split:** a grouped split by S1 entity and by connected component, so an S1 entity and its matches never land on both sides. The starter's `split_queries` logic can be reused.
   - **Training data:** you may subsample the training S1 entities (for example 600k–1M) to fit RAM. That is fine because it is training data, not test. **Never subsample test.**
   - **Threshold:** tune it on the validation split using the exact macro F0.5 (in `entity_scores`), with singletons included.
   - **Honest estimate:** report a separate untouched holdout if possible.
5. **Test inference:**
   - Run in chunks. Write `candidate_pairs.tsv` from exactly the list the model scored, and write `matching_results.tsv` as the thresholded subset.
   - Output rules: sorted IDs, empty field for none, one row per test S1.
   - Then run the starter's `validate` and the **official** validator, and require its `PASS`.

**First deliverable** is a *valid* baseline, even a simple one, run as `runs\baseline_*`. Preserve it before improving anything.

---

## 4. Execution order (do it; don't ask)

1. Read `RUN_STATE.json`, `STATUS.md`, `logs\`, and `runs\`. Check for running python processes before starting anything.
2. **Audit:** use a polars-based audit, not the starter's in-memory `audit`, which is too heavy.
   - Check schema, ID prefix and uniqueness, blanks, and country value counts per file.
   - Check that every GT id exists; count the singleton fraction; build the matches-per-entity histogram.
   - Save the output to `logs\audit.json` and update RUN_STATE (`dataset_downloaded=true`, row counts, hashes).
3. Implement the memory-safe pipeline in `src\` as new modules. Keep the old `pipeline.py` working for the unit tests.
   - Add unit tests for the new metric and the output writer.
   - Pin the new dependencies in `requirements.txt`.
   - Update the `package` step so it includes every new `src\*.py` and asset.
4. Run `runs\baseline_<timestamp>`, then the official validator.
5. **Portal, when a logged-in browser is available** (see §7):
   - Verify the account, the team name, the team-leader role, the live deadline, the rules (including the AI-assistance rule), the quota, and the submission UI.
   - Record everything in RUN_STATE with evidence in `evidence\`.
6. **With the human's explicit per-action OK in chat,** upload the baseline `matching_results.tsv` to the leaderboard.
   - Record the submission ID, the IST time, the SHA-256, the status, and the score. Only a **SCORED** status counts.
7. **Improve,** one experiment at a time: blocking recall, features, threshold, and top-k. Log every run in `experiments\experiment_log.tsv`.
   - Stop experimenting by about **27 Sep 18:00 IST** at the latest.
8. **Finalize:**
   - Fill `student_resource\Documentation_template.md` with real numbers and the real team name. Copy it to a working file; don't edit the official original.
   - Package `<TEAM>_submission.zip` and inspect it. Its `output\matching_results.tsv` must be byte-identical to the final leaderboard upload.
   - With the human's OK: upload the final leaderboard file and the final ZIP. Verify the receipt or final-selection status, and save evidence.
9. Report the facts listed at the end of `CLAUDE_START_HERE.md`.

---

## 5. Output format reminders (official)

- TSV only, UTF-8, with exactly these headers:
  - `source1_entity_id\tmatched_entity_ids`
  - `source1_entity_id\tcandidate_entity_ids`
- One row per test S1 (1,732,544 rows in each file). An empty field means no match. IDs are comma-separated with no spaces or quotes and no duplicates, and must be S2- or S3- IDs only.
- Every match must also appear among that row's candidates.
- **Official validator:**
  - Run from `D:\Amazon_ML\amazon_ml_starter\student_resource`:
    `..\.venv\Scripts\python.exe utils\validate_submission.py --matching <abs>\matching_results.tsv --candidate <abs>\candidate_pairs.tsv --test-dir dataset\test`
  - A pass means exit code 0 and output that starts with `PASS`.
- **ZIP layout:**
  - `output\{matching_results.tsv,candidate_pairs.tsv}`
  - `code\business_entity_resolution\{src\,README.md,requirements.txt,...}`
  - `Documentation_template.md`
  - Exclude the dataset, `.venv`, `work\` parquet, and any secrets.

---

## 6. If this runs on a DIFFERENT computer

The data is only on Yuval's PC (`D:\Amazon_ML`). On another machine:
- Get the Drive folder shared with that person's Google account. Prefer this to "Anyone with the link".
- Then either download it in the browser, or, while the folder is temporarily link-shared, run `bash amazon_ml_starter/scripts/fetch_dataset.sh`. The script checks sizes and SHA-256.
- Re-detect RAM and CPU and adjust the safety limits in §3.

---

## 7. The ONLY things that need a human (Yuval or the team leader)

1. **Drive sharing:** the download is done, so switch the Drive folder back from "Anyone with the link" to **Restricted**. To do it: right-click `student_resource` → Share → General access → Restricted → Done.
2. **Unstop sign-in as TEAM LEADER, once,** in a browser the agent controls. Either:
   - the Claude desktop app's built-in Browser pane (the globe icon at the top right of the session), or
   - Chrome with the **Claude in Chrome** extension connected to this Claude account.

   The agent must never type passwords or OTPs. The human does that part, including any CAPTCHA. If Yuval is not the team leader, the leader must sign in or do the uploads.
3. **Say "yes" in chat** before each upload: the baseline leaderboard upload, the final leaderboard upload, and the final ZIP.
4. **Team member names** for the template header, if the portal doesn't show them.
5. **Any personal declaration or eligibility checkbox** on Unstop. Only the human may confirm it.

Everything else (data, code, training, validation, write-up, packaging) is the agent's job.
