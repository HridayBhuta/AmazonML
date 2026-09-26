# Testing completed during preparation

- Python source syntax compilation passed.
- Five unit tests passed: Unicode/open-country normalization; ID lists; candidate selection and empty query; per-entity macro F0.5 including singleton cases; grouped entity split.
- A separate synthetic end-to-end test passed the audit, training, prediction, validation, ZIP packaging, and rerun stages. The fabricated data included US and India training records, France test records, one-to-many links, and empty-record singletons.
- Regeneration from the packaged code produced byte-identical matching_results.tsv and candidate_pairs.tsv in that synthetic test.

These are software tests, NOT performance measurements on Amazon data. No official dataset was available. No leaderboard submission was made. No official validator was available; the package includes an independent format check, not Amazon's validator. Real dataset compatibility, quality, runtime, and memory remain to be measured.
