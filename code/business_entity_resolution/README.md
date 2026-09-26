# Amazon ML Challenge 2026: Business Entity Resolution (one-command package)

For every Source 1 business, this finds its copies in Source 2 and Source 3. It runs offline on a
laptop, CPU only: no GPU, no internet lookups, no AI services.

## Run it on a Mac (M-series or Intel)

1. **Put the folder in your home folder.** Unzip `SEND_TO_MAC_AmazonML_code_and_data.zip` into your
   home folder: in Finder, choose Go > Home. That gives `/Users/<you>/AmazonML_Project`. Avoid
   Desktop, Documents and iCloud Drive, because iCloud would upload the multi-GB caches.
2. **Data.** The official data zip, `OFFICIAL_DATA_student_resource.zip`, is already inside the
   folder. Any zip or folder whose name contains `student_resource` also works, placed here or in
   `~/Downloads`; it is found automatically, including a two-part Google Drive download. Don't
   unzip it yourself.
3. **Optional: add your team details.** Write your team name and members into
   `team_config.json`, e.g. `{"team_name": "My Team", "team_members": "A, B, C"}`.
4. **Start it. This is the recommended way.** Double-clicking `run.sh` does nothing on macOS.
   1. Open **Terminal** (Cmd + Space, type "Terminal", Enter).
   2. Run:
      ```bash
      bash ~/AmazonML_Project/run.sh
      ```
   If the folder is somewhere else, type `bash ` and drag `run.sh` into the Terminal window
   instead. A banner reading "AMAZON ML CHALLENGE - STARTED" appears at once, followed by
   progress lines.
   You can also double-click `RUN_ME.command`. If macOS says it "could not verify" the file:
   1. Open **System Settings > Privacy & Security**.
   2. Scroll down and click **Open Anyway** next to RUN_ME.command.
   3. Confirm, then double-click it again.
   (Or run `xattr -dr com.apple.quarantine .` once in this folder.)
5. **Keep the Mac on power with the lid open.** The run keeps the Mac awake, but closing the lid
   pauses it. If it stops for any reason, just run it again: finished steps are skipped.

It then does everything by itself:

1. **Installs Python and packages.** A private Python 3.12 and the pinned packages go inside this
   folder. No admin password is needed, and the libomp/OpenMP fix is applied automatically.
2. **Tests.** Runs the unit tests, then a 3–5 minute smoke test on a small sample of the real data.
3. **Runs the full baseline** and writes a first valid submission to `final/`.
4. **Runs the experiment sweep** (`configs/sweep.json`) and records every result in
   `experiments/results.tsv`.
5. **Picks the best run** by untouched holdout F0.5 (ties go to fewer candidates), re-validates it
   against the real data, and writes the final files to `final/`.

Logs are in `logs/` and `runs/<name>/pipeline.log`. Only one run can be active at a time; a second
start is refused.

## What to upload (team leader, on Unstop)

`final/SUBMIT_THESE.txt` lists the exact files and their SHA-256 hashes:

- **Leaderboard:** `final/matching_results.tsv`
- **Final package:** `final/<team>_submission.zip`. It contains `output/` (both TSVs), `code/`
  (this pipeline) and the filled-in `Documentation_template.md`.

## Commands (for people or AI agents)

```bash
bash run.sh setup                       # environment only
bash run.sh smoke                       # quick end-to-end check on a sample
bash run.sh run --name baseline         # one full experiment (any option below may be added)
bash run.sh run --name big --leaves 255 --min-leaf 100 --lr 0.04 --note "bigger model"
bash run.sh sweep                       # every experiment in configs/sweep.json
bash run.sh compare                     # ranked table + selects the best
bash run.sh finalize [--name RUN]       # re-validate + documentation + zip for the best (or named) run
bash run.sh prune                       # delete caches not used by the best run
```

**Options:** `--max-df --key-budget --prelim --recall-tol --max-k --lr --leaves --min-leaf
--feature-fraction --drop-features a,b --decision threshold|one_to_one --refit-all --train-s1
--valid-s1 --holdout-s1 --seed --threads --chunk --b-block`.

`AGENT_BRIEF.md` is the brief for an autonomous coding agent that continues improving the solution.

## Method in one paragraph

1. **Normalise the text.** Accents are stripped. Indic scripts are romanised and reduced to a
   consonant skeleton, so "राम मार्केटिंग प्राइवेट लिमिटेड" and "Ram Marketing Private Limited"
   produce the same key. Legal forms and address abbreviations are canonicalised.
2. **Build the shortlist** (`candidate_pairs.tsv`) with a bounded, IDF-weighted inverted index.
   - It uses single keys plus rare conjunctive keys: name-token pairs, name + address token,
     name + house number, name + postcode, and house number + street.
   - Every key is country-prefixed; the country field is treated as an open set, and France is
     handled like any other label.
   - Frequent keys are dropped completely.
3. **Rerank and cut.** The shortlist is reranked cheaply by string similarity, then cut to the
   smallest budget that loses at most `recall_tol` of training link recall.
4. **Score and decide.** A LightGBM classifier (MIT licence) scores about 43 pair features, and a
   validation-tuned threshold maximises per-entity macro F0.5. Every test Source 1 entity gets a
   row in both files.

## Reproduce exactly

The filled documentation (Appendix A) and `artifacts/config.json` in the submission zip hold the
exact command and every resolved setting. Dependencies are pinned in `requirements.txt`. Data
integrity is checked against the official SHA-256 values in `src/er/common.py`.
