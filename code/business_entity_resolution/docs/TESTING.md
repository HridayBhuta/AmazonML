# Testing record

- Python source syntax compilation passed.
- Five unit tests passed: Unicode/open-country normalization; ID lists; candidate selection and empty query; per-entity macro F0.5 including singleton cases; and grouped entity splitting.
- A synthetic end-to-end test passed the audit, training, prediction, validation, ZIP packaging, and rerun stages. The fabricated data included US and India training records, France test records, one-to-many links, and empty-record singletons.
- Regeneration from the packaged code produced byte-identical `matching_results.tsv` and `candidate_pairs.tsv` files in that synthetic test.
- The repository includes the official format validator at `utils/validate_submission.py`.

These checks verify software behavior and packaging only. They are not Amazon-data performance measurements or a leaderboard result. The full private dataset run and official validation of the final outputs remain outstanding.
