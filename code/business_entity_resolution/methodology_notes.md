# Notes for completing your OFFICIAL methodology template

This is not Amazon's `Documentation_template.md`. Preserve and complete the actual downloaded template. Describe only your final code and actual measured results.

## Baseline method implemented here

- Read the seven supplied TSVs with an explicit tab delimiter; preserve empty ID lists.
- Treat Source 1 as the deduplicated reference, with zero/one/multiple possible Source 2/3 matches.
- Normalize case, punctuation, Unicode marks, and whitespace without a country whitelist. No external normalization service is called. Ampersands map to `and`; this is a hand-written text rule, not a business lookup.
- Build a combined Source 2/3 inverted index on compact exact names, name tokens, compact-name trigrams, address tokens, and address digit sequences of length at least four. Digit sequences are not presumed to be accurate postal codes.
- Discard entire posting lists above the configured cap. Probe bounded numbers of rare lists, add rarity-weighted votes, retain a bounded prefilter shortlist, then rank that shortlist by fixed name/address string similarity.
- The final candidate cap is configurable. The two sources share the cap; there is no guarantee that either source receives a fixed quota. Evaluate recall separately by source when improving the system.
- Pair features include name/address edit and token similarities, Jaccard overlaps, numeric overlap/conflict, first-number equality, missingness, equality/length indicators, country equality, and a source indicator. Source ID sequence numbers are not model inputs.
- A LightGBM binary classifier is trained from scratch. The trained model uses the provided labels, not external identity data or pretrained embeddings.
- Validation splits Source 1 entities, not arbitrary pairs. Any Source 1 entities linked to a common labelled target are grouped together. The target gallery contains all supplied training Source 2/3 records; this is query-entity-held-out validation, not a completely held-out target-gallery or country experiment.
- Threshold selection uses macro F0.5 per Source 1 entity with the specified singleton behavior. The threshold is tuned on the reported validation set. Treat the score as a tuning estimate and disclose that limitation.
- Retrain on all training candidate pairs, then infer all test Source 1 entities with the selected threshold. No forced match and no one-to-one matching rule are applied.
- Export exactly the last candidate set fed into the learned matcher. Final matches are a subset. Both outputs cover every test Source 1 entity.

## Values to take from the final run

Use `run_config.json`, `validation_metrics.json`, and the actual portal result. Include real data counts, parameters, average candidate counts, candidate recall, singleton accuracy, per-country validation scores, selected threshold, hardware, runtime, and public score (clearly identified as public). Do not fabricate any value. No labelled France evaluation is available from the stated training files.

## Limitations and next work to disclose

The prototype index is in-memory Python and is not billion-scale benchmarked. Common-key discarding and top-k truncation may lose true matches. A fixed lexical prefilter can miss transliterations or severe noise. Address numbers are crude features. Threshold calibration may shift after refitting and under the unseen France distribution. The first version has not added a second untouched holdout or out-of-country evaluation. Explain subsequent changes if you make them.

## Licenses and integrity

The matching algorithm uses MIT-licensed LightGBM; the starter source code is MIT-licensed. The final model is trained from scratch, with no pretrained model weights. Keep an accurate dependency/license inventory. No business-registry, geocoding, search-engine enrichment, or entity-resolution API is used. Do not add such services; the problem statement prohibits external business-identity lookup and internet augmentation.
