# Amazon ML Challenge 2026: practical starting kit

A CPU-first baseline for the **Business Entity Resolution Challenge**. This kit is independently prepared, not an official Amazon resource or a winning solution. It does not include the private competition dataset or a real leaderboard score. The repository root contains the official documentation template, and `utils/validate_submission.py` contains the official format validator.

## 1. Get the official resources

Log in to the registered team leader's Unstop account. On the competition page, open the live **ML Challenge | 72-Hour Hackathon** stage and click **Code**, as instructed by the event page you provided. In that challenge area, use the actual **Download Data Set** link and obtain any student-resource archive, `Documentation_template.md`, and `utils/validate_submission.py` offered there. Read any additional portal rules.

The uploaded problem statement contains only the text `Click here to Download`, not a downloadable URL. This kit cannot recover the private link from that text.

Extract your official resources into this project so that these paths exist. The archive name or outer folder may differ; adjust the extraction layout, not the seven TSV filenames.

```text
amazon_ml_starter/
  src/pipeline.py
  requirements.txt
  README.md
  student_resource/
    Documentation_template.md       # official template, obtained from the portal
    utils/validate_submission.py     # official validator, obtained from the portal
    dataset/
      train/
        train_source1.tsv
        train_source2.tsv
        train_source3.tsv
        train_ground_truth.tsv
      test/
        test_source1.tsv
        test_source2.tsv
        test_source3.tsv
```

Keep raw data untouched and private. Do not upload competition data to a public repository. Outputs are TSVs, not CSVs.

## 2. Install the environment

Use **64-bit Python 3.11 or 3.12**. The pinned package versions in this kit were exercised on Python 3.11 in the preparation environment. No GPU is used by this implementation. Actual RAM/time requirements depend on dataset sizes, which are not available here.

### Windows Command Prompt

In File Explorer, open this extracted project folder, click the address bar, type `cmd`, and press Enter. Then:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

With Python 3.12 installed instead, replace `-3.11` with `-3.12`. If the `py` launcher is unavailable, install a compatible 64-bit Python distribution before proceeding.

For PowerShell without activating scripts, invoke the environment executable directly, for example:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/pipeline.py audit --data student_resource/dataset
```

### macOS / Linux terminal

Open a terminal in the project folder:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use `python3.12` when that is your installed compatible version. On macOS, a LightGBM error referring to `libomp.dylib` may require installing OpenMP through your existing Homebrew installation: `brew install libomp`. See the official LightGBM installation guidance linked below. Do not switch to a paid GPU merely because of an installation error.

## 3. Confirm all seven files can be read

From the project root, in the activated environment:

```sh
python src/pipeline.py audit --data student_resource/dataset
```

This checks the schema, ID prefixes, duplicates, ground-truth ID coverage, and target references. It prints row counts, normalized country labels, blank-name/address counts, singleton fraction, and matches per reference entity. Every country label is accepted. Inspect these real numbers before choosing compute resources.

## 4. Run the first baseline

```sh
python src/pipeline.py run --data student_resource/dataset --out runs/baseline
```

The run directory must not already exist. This protects a previous successful submission from accidental overwrite. Use a fresh name for every experiment.

This command:

1. Normalizes the provided strings locally, preserving non-Latin letters and avoiding an explicit country whitelist.
2. Builds an inverted index from Source 2 and Source 3 tokens/name character trigrams.
3. Retrieves a bounded shortlist, then applies a non-learned string-similarity prefilter. The default final candidate cap is **30 total candidates per Source 1 entity**, across both sources; this is an experimental starting point, NOT an organizer requirement.
4. Splits training Source 1 entities into training/validation groups. Pairs for the same Source 1 entity stay together; any Source 1 entities sharing a labelled target are also kept together. The split is stratified by country and match/singleton status where group counts allow.
5. Creates 21 string/address/number/missingness features and trains a LightGBM binary classifier on training candidate pairs, using only the provided labels. No pretrained model, geocoder, business registry, or external business lookup is used.
6. Chooses a match threshold using the official-style **per-entity macro F0.5**, including singleton credit.
7. Refits on all labelled training candidate pairs, then predicts for every test Source 1 entity.
8. Saves both required TSVs and runs this kit's independent local format check.

The saved candidates are exactly those sent to LightGBM at test inference, after the non-ML prefilter. Predictions are always a subset of these candidates. No one-to-one constraint or forced match is imposed.

Expected files:

```text
runs/baseline/
  matching_results.tsv        # upload this one file to the leaderboard
  candidate_pairs.tsv         # required in final ZIP, not leaderboard-scored
  validation_metrics.json     # real local metrics after your run
  threshold_search.tsv
  model.txt
  run_config.json             # parameters, versions, feature names, data hashes
```

**No score is promised.** The threshold is tuned on the same validation set whose score is reported, so `validation_macro_f0_5_tuning_estimate` is a tuning estimate, not an untouched final holdout score. Full-training refitting may change calibration. Add a second untouched split or grouped cross-validation for stronger model selection. The provided training labels do not measure France performance.

## 5. Validate with the official script

Do not rely only on this kit's validator. Run the official validator downloaded with your resources:

```sh
cd student_resource
python utils/validate_submission.py --matching ../runs/baseline/matching_results.tsv --candidate ../runs/baseline/candidate_pairs.tsv --test-dir dataset/test
cd ..
```

The problem statement says `PASS` and exit code 0 indicate safe format, not a good score. Fix every error before uploading. If the official script is absent from your archive, obtain it through the portal resources; this kit does not pretend to include it.

## 6. Make your first leaderboard submission

Use the registered team leader's challenge portal to upload:

```text
runs/baseline/matching_results.tsv
```

Confirm the portal reports **SCORED** and record the score, time, and run name. A local `PASS` is not a leaderboard score. Do not rename this TSV to CSV or save it through Excel. Do not upload the starter archive as a final solution. Check submission limits and final-submission selection in the actual portal; they are not specified in the uploaded statement.

## 7. Improve systematically

Preserve the baseline, then run a smaller-candidate experiment:

```sh
python src/pipeline.py run --data student_resource/dataset --out runs/k15 --top-k 15
```

Compare the tuning estimate, candidate link recall, candidate macro recall on non-singletons, singleton accuracy, per-country scores, and mean candidates. A low candidate recall means the matcher never sees some true matches. A larger cap can help recall, but candidate-set size is explicitly considered in final review. Do not shrink only the reported file: reproduce the actual inference candidate set.

Useful next experiments include locally learned address/name normalization, better rare-token/character-gram retrieval, hard-negative review, and a leave-one-training-country-out stress test. These are recommendations, not organizer-prescribed methods or guaranteed improvements. Check the validation impact of each change and keep complete reproducible configurations.

### Compute and scaling limits

The blocker avoids a full Source1 x Source2/3 Cartesian-product comparison. Posting lists are capped at 128 entries; over-common keys are discarded completely. At most 23 posting lists are probed per query before a 200-record prefilter and final top-k selection. This is bounded retrieval, but the Python index is still in memory and has not been benchmarked at billion-record scale. A production implementation would need sharded/on-disk indexes and distributed processing.

Discarding common keys, truncating at top-k, weak/noisy names, and short names can lose true matches. This kit intentionally reports recall so those failures can be measured. It does not guarantee that the baseline blocker is sufficient for a top rank.

The default guard refuses a run with more than 1,500,000 potential training pairs. Raising `--max-pairs` requires more RAM; lowering `--top-k` affects recall. The in-memory index itself can exceed a small laptop's memory before the pair guard runs. Do not sample or omit test Source 1 entities to work around memory limits. Move the full run to suitable permitted compute or redesign the blocking/index storage.

## 8. Complete the official methodology template

Open the actual `student_resource/Documentation_template.md` and fill its existing sections. `docs/methodology_notes.md` in this kit explains what this baseline does and what real measurements to insert; it is NOT a substitute official template.

Describe your actual preprocessing, candidate-generation stages and caps, feature list, model/threshold, entity-level validation, singleton handling, country generalization, measured candidate statistics, runtime/hardware, dependencies, and external-data policy. Use values from the final selected run. Do not invent France accuracy or leaderboard numbers.

## 9. Build the final ZIP

After completing the official template and selecting a validated run:

```sh
python src/pipeline.py package --data student_resource/dataset --out runs/baseline --documentation student_resource/Documentation_template.md --destination MYTEAM_submission.zip
```

Replace `MYTEAM` with your team name. Change `--out` to your actual final selected run. Existing ZIPs are not overwritten.

The resulting ZIP contains:

```text
output/
  matching_results.tsv
  candidate_pairs.tsv
code/business_entity_resolution/
  src/pipeline.py
  README.md
  requirements.txt
  LICENSE
  REPRODUCE_THIS_RUN.md
  artifacts/
    run_config.json
    validation_metrics.json
    threshold_search.tsv
    model.txt
Documentation_template.md
```

`REPRODUCE_THIS_RUN.md` records the selected run's exact non-default parameters. The code regenerates both TSVs from the provided train/test data. Competition data and your virtual environment are deliberately excluded from the final ZIP.

Upload this final ZIP in the portal's final-package submission area, in addition to making the leaderboard TSV submission. Ensure the final ZIP contains the same final matching file you intend to be evaluated. Follow the portal's actual instructions for choosing the final leaderboard submission; do not assume it automatically selects your best score.

## Testing and provenance

This kit was syntax-checked and exercised using fabricated records only. The unit tests cover country-open normalization, candidate selection, empty lists, macro F0.5, and entity grouping. They do not establish challenge accuracy. Run them with:

```sh
python -m unittest discover -s tests -v
```

Official source material used: the user-uploaded `PROBLEM_STATEMENT.md` and pasted event details. Software references:

- Python virtual environments: https://docs.python.org/3/library/venv.html
- LightGBM 4.6.0 Python API: https://lightgbm.readthedocs.io/en/v4.6.0/Python-API.html
- LightGBM 4.6.0 package/install notes: https://pypi.org/project/lightgbm/4.6.0/
- LightGBM source/license: https://github.com/lightgbm-org/LightGBM

LightGBM is MIT-licensed. This original starter code is also provided under MIT; dependency licenses remain their own. No pretrained weights are included. Review the organizer's full license and assistance rules before final submission; this is not a certification of eligibility or acceptance.
