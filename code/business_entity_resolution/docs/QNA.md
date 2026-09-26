# Questions and answers

Plain-language answers to the questions the team asked while building this. The diagrams are in
[ARCHITECTURE.md](ARCHITECTURE.md).

---

## A. The project

### What is the project?
Amazon gave us business records from **3 sources**. The same real-world business can appear in
several of them, written differently:
- "Ram Marketing Pvt Ltd" and "राम मार्केटिंग प्राइवेट लिमिटेड" (Hindi script) and
  "Ram Marketing Private Limited";
- "105 Elm Street" and "105 ELM ST".

For every business in **Source 1** (the clean master list) we must find all its copies in
Sources 2 and 3, or say it has none. This task is called *entity resolution*.

### Why is it hard?
1. **Scale:** 1.7M test businesses searched against about 10M records.
2. **Messy text:** typos, abbreviations, reordered addresses, missing fields, the literal text
   "null", junk like "--" or "<<", website names like "name.com", and Indic scripts.
3. **A new country:** the test set contains **France**, which never appears in the training data.
4. **Precision-heavy scoring:** a wrong match hurts more than a missed one.
5. **Rules:** no internet lookups, geocoding, registries or hosted AI to resolve records. Any model
   must be MIT or Apache-2.0 licensed and have at most 8B parameters.

### What do we win?
The top teams get cash prizes and pre-placement interviews at Amazon.

---

## B. The files

### What is a `.tsv` file?
It's a table saved as plain text: one row per line, with columns separated by a Tab. Tabs are used
because addresses contain commas. It opens in a text editor. Never re-save it from Excel.

### What are the 7 data files?
Every source file has the columns `entity_id, business_name, business_address, country`.

| File | What it is | Rows |
|---|---|---|
| `train_source1.tsv` | Practice master list | 2,206,821 |
| `train_source2.tsv` | Practice messy list 2 | 5,034,616 |
| `train_source3.tsv` | Practice messy list 3 | 5,285,603 |
| `train_ground_truth.tsv` | **Answer key**: for each practice S1, which S2/S3 IDs are the same business (blank means none) | 2,206,821 |
| `test_source1.tsv` | The exam: every one of these must get an answer | 1,732,544 |
| `test_source2.tsv` | Exam list 2 | 4,887,273 |
| `test_source3.tsv` | Exam list 3 | 5,082,316 |

The ID prefix (`S1-`, `S2-`, `S3-`) tells you the source. The official
`utils/validate_submission.py` (format checker) and `Documentation_template.md` (write-up
template) come with the data.

### What files do we produce?
1. **`matching_results.tsv`**: our answers, one row per test S1:
   `source1_entity_id <TAB> S2-…,S3-…` (blank means no match). **This is the only file scored on
   the leaderboard.**
2. **`candidate_pairs.tsv`**: our shortlist. For each test S1, every look-alike the model examined.
   It isn't scored, but Amazon reviews it, and **smaller shortlists rank higher** in the final
   review.

### What is `candidates.tsv` / `candidate_pairs.tsv`, exactly?
It's the list that goes *into* the matching model. The rules say it must be **exactly** the set
the model scored: not an earlier bigger list, and not a trimmed one. Every match in
`matching_results.tsv` must also appear there. Our code refuses to write the outputs if the scored
pairs differ from the candidate set.

### What goes in the final zip?
```text
<team>_submission.zip
├── output/matching_results.tsv, candidate_pairs.tsv
├── code/business_entity_resolution/  (src/, README.md, requirements.txt, ...)
└── Documentation_template.md          (filled in automatically with real numbers)
```

---

## C. Scoring

### How is it scored?
F0.5 is computed **per Source 1 business**, then averaged over all of them:
`score = 1.25 × correct / (0.25 × true matches + predicted matches)`.
- A business with no copies that we correctly leave blank scores **1.0**.
- Predicting any match for such a business scores **0**.
- A wrong match costs about twice as much as a missed one.

### Is our local score the real leaderboard score?
No, it's an estimate. We hide 15% of the training businesses (the **holdout**), never train or tune
on them, and score them with the official formula. The leaderboard uses the hidden test answers.
France can't be measured locally because the training data has no French labels.

---

## D. How the solution works

### What is the whole procedure, in short?
1. Clean the text.
2. Shortlist about 12 look-alikes per business.
3. A model judges each pair.
4. Keep the confident ones.
5. Validate.
6. Package.

See diagram 1 in ARCHITECTURE.md.

### How does the text cleaning and Hindi romanisation work?
- **Clean-up:** lowercase, remove accents ("École" becomes "ecole"), remove punctuation.
- **Abbreviations:** "Rd" becomes "road", "Pvt" becomes "pvt" (legal forms are canonicalised),
  "Texas" becomes "tx", "HR" becomes "haryana", and French "Bd" becomes "boulevard".
- **Indic scripts:** Devanagari, Kannada, Tamil, Telugu, Bengali, Gujarati, Gurmukhi, Oriya and
  Malayalam are **romanised offline** with a hand-written table ("राम" becomes "rama").
- **Consonant skeleton:** vowels and "h" are dropped and similar sounds merged, so
  "Ram Marketing Private Limited" and "राम मार्केटिंग प्राइवेट लिमिटेड" both become
  `rm mrktng prvt lmtd`.
- **Core name:** legal words (inc, llc, pvt, ltd, sarl…) are removed to get the name's core.
- Nothing here uses the internet or an AI service.

### How does the shortlisting (blocking) step work?
Checking every pair would be about 17 trillion comparisons, so we build a quick shortlist first.
1. **Make keys.** Each business gets short tags:
   - name words (`ram`, `marketing`) and their sound-alikes (`rm`, `mrktng`);
   - the start and end of the squashed name, house number + street (`12 mg`), address sound-alikes
     (`klkt`) and postcodes;
   - combined keys such as `ram marketing`, `ram klkt` and `ram 12`.

   Every key includes the country.
2. **Bounded index.** A lookup table maps each key to the businesses that have it. Keys shared by
   more than 300 businesses ("marketing", "road") are **thrown away completely**, because they
   can't narrow anything down.
3. **Vote and rerank.** For each business we look up its **rarest** keys first, up to about 1,200
   look-alikes. Each look-alike earns points for every shared key, and rarer keys earn more. The
   top 60 then get a quick spelling comparison:
   `0.6 × name similarity + 0.4 × address similarity`.
4. **Adaptive cut.** Keep the best **K**. K is chosen from the practice answers as the smallest
   list that still keeps about 99% of true matches, with a hard cap of 40. On the sample it was 12.
5. The kept list is exactly what the model scores, and it becomes `candidate_pairs.tsv`.

### Why are common keys dropped instead of truncated?
Keeping "the first 300 of 40,000" would be an arbitrary, biased subset. Dropping the key entirely
is honest, and the combined keys (`sai traders mumbai`) cover businesses whose single words are
all common. Without them, about 7% of test businesses (22% of French ones) got no candidates at
all.

### How is the shortlist size K chosen?
On a sample of **train-split** businesses (never the validation or holdout ones) we measure, for
many K and score-gap values, how many true matches survive and how long the lists are. We then pick
the smallest average list within `recall_tol` (1%) of the best recall. The grid is saved as
`artifacts/cut_grid.tsv`.

### What does the matching model look at?
It's a **LightGBM** classifier (MIT licence), trained from scratch on the provided answer key only.
There are **43 features** per pair:
- **Name:** ratio, token-sort, token-set and partial similarity; Jaro-Winkler; skeleton similarity;
  token Jaccard.
- **Address:** similarity, number agreement or conflict, house number, postcode.
- **Context:** blocking score, rank in the list, gap to the best candidate, how many S1 businesses
  compete for the same target, and this pair's rank among them.
- **Flags:** missing fields, Indic script, same country, source.

### How is the match threshold chosen?
The model gives each pair a probability. We try thresholds from 0.05 to 0.99 on the **validation**
split and keep the one with the best per-business macro F0.5. That threshold is then applied,
unchanged, to the holdout (for the honest score) and to the test set.

### What is the "one-to-one" rule and why is it off by default?
It keeps each target only for the S1 that scores it highest, since S1 is deduplicated. Locally it
can only be evaluated among one split's businesses, while on the test set it runs across all
1.7M, so its local score is **optimistic**. The default is therefore a plain threshold; the AI
agent may enable it only after building proper out-of-fold scoring.

### How is France handled without French training data?
- **No country-specific rules:** every key and feature is spelling and sound similarity, never
  "is this the US?". Country is an open set of labels, and nothing is filtered.
- **French abbreviations:** a small hand-written map (R., Bd, Av., St means Saint) plus accent
  stripping.
- **Monitoring:** the pipeline reports, per country, how many businesses got no candidates, and
  warns above 1%.

### How are training, validation and holdout separated?
- **Split:** training businesses are split **70 / 15 / 15** by a deterministic hash.
- **Grouping:** businesses that share a labelled target stay in the same split, so answers can't
  leak across splits.
- **Roles:** the model trains on *train*, early-stops and tunes its threshold on *valid*, and is
  scored on the untouched *holdout*.

---

## E. Running it

### Why not run it on the Windows PC?
It has 7.7 GB of RAM with under 1 GB free, and the full run needs several GB. The run is designed
for a MacBook (M-series, 16 GB or more).

### What exactly happens on the Mac after `bash run.sh`?
See diagram 3 in ARCHITECTURE.md. All times are estimates.
1. **Setup:** private Python 3.12 and pinned packages, with the libomp fix (2–5 min).
2. **Smoke test** on a small sample (3–5 min).
3. **Full baseline** (about 1.5–2 h). The **first valid submission** appears in `final/`.
4. **Sweep** of 11 variations.
5. **Compare** and pick the best.
6. **Re-validate** it on the real data and rebuild `final/`.

If it stops, run it again: finished steps are skipped.

### How are experiments compared and the best chosen?
Every run adds a row to `experiments/results.tsv` with its holdout F0.5, validation F0.5,
candidate recall, average candidates, per-country scores and settings. `compare` picks the highest
holdout F0.5; among runs within 0.001 of it, the **smallest shortlist** wins. Smoke runs and runs
that fail the official checker are ignored.

### What does the AI agent on the Mac do?
It reads `AGENT_BRIEF.md`, which gives it freedom to change keys, features, the model and the
decision rule. It must log every attempt, compare them all, and ship the best, while never
breaking the rules. It starts long jobs in the background and never runs two at once. Only people
upload to Unstop.

### How do we know it works?
- **Unit tests:** 10 of 10 pass.
- **Smoke test:** on a sample of the real data, the whole chain passes Amazon's official checker.
- **Code review:** an 8-agent review found **27 real defects**, all fixed. Examples:
  - a sample run could have been packaged as the final submission;
  - 7% of businesses had no candidates;
  - the libomp setup could break on Macs without Homebrew;
  - Hindi names were losing their key word;
  - stale caches could survive code edits.
- **Not yet run:** the full-data run itself, which happens on the Mac.

---

## F. People and submission

### Where are humans needed?
1. **Drive sharing:** set the Google Drive folder back to "Restricted".
2. **Start the run:** copy the zip to the Mac, unzip it, and start the run (about 10 minutes of
   work).
3. **Team details:** put the team name and members in `team_config.json`.
4. **Team leader uploads:** `final/matching_results.tsv` to the leaderboard and
   `final/<team>_submission.zip` as the final package. The team leader also ticks any
   declarations.
5. **Deadline:** confirm the live deadline on Unstop. It is 27 Sep 2026, 23:59 IST per the
   schedule we copied.

### What should be uploaded, and when?
- **As soon as `final/` first appears (about 2 h):** upload `matching_results.tsv` for an early
  score.
- **Before the deadline:** upload the final best `matching_results.tsv` **and** the zip. The zip's
  `output/matching_results.tsv` is byte-identical to the leaderboard file.

### What if something goes wrong?
See the table in `PLAN.md`, section 7. For anything else, send the newest file in `logs/`.

### What must never go into git?
The dataset (`data/`, `student_resource/`, any `*student_resource*.zip`), `cache/`, `runs/`,
`final/`, `logs/`, `.venv/` and `.tools/`. The `.gitignore` handles this. The data is private
competition data.
