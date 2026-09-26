# Amazon ML Challenge 2026 submission repository

This branch follows the official final-submission layout for the Business Entity Resolution challenge.

```text
.
├── output/
│   ├── matching_results.tsv       # generated final predictions
│   └── candidate_pairs.tsv        # candidates actually scored by the model
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── tests/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

The private competition dataset is deliberately excluded. The two output TSVs are also absent until a complete pipeline run has produced and validated them; header-only or fabricated outputs are not committed as substitutes.

## Current status

- `main` was empty when this branch was created on 26 September 2026.
- The starter implementation, tests, official validator, and official methodology template are present on this branch.
- A full-scale, memory-safe run has not yet produced the final output files.
- `Documentation_template.md` still contains fields that must be filled with the real team name, members, run measurements, and validation results.

## Finalization checklist

1. Complete and run the memory-safe pipeline against the official private dataset.
2. Copy the validated files to `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
3. Fill every placeholder in `Documentation_template.md` with measured values.
4. Run `code/business_entity_resolution/utils/validate_submission.py` against both outputs.
5. Create the final ZIP with this repository's top-level layout and inspect its contents before upload.

See `code/business_entity_resolution/README.md` for the pipeline and reproduction instructions.
