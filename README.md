# Amazon ML Challenge 2026

Business Entity Resolution submission workspace, arranged to match the official final-package format. The repository contains the implementation, validation tooling, methodology template, handoff notes, and a ready-to-transfer macOS package.

## Repository map

| Path | Purpose |
| --- | --- |
| `code/business_entity_resolution/` | Reproducible Python implementation |
| `code/business_entity_resolution/src/` | Pipeline and memory-bounded audit modules |
| `code/business_entity_resolution/tests/` | Unit tests |
| `code/business_entity_resolution/utils/` | Official submission validator |
| `code/business_entity_resolution/docs/` | Testing evidence and methodology notes |
| `output/` | Destination for the two validated submission TSVs |
| `Documentation_template.md` | Official methodology template to complete |
| `HANDOFF.md` | Current project state and execution guidance |
| `executable.zip` | Code-only macOS transfer package; extract and run `bash run.sh` |

## Quick start

The private dataset is intentionally excluded from Git. For the starter implementation:

```powershell
cd code\business_entity_resolution
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The macOS package has the newer memory-safe workflow:

```bash
unzip executable.zip
cd amazon_ml_er
bash run.sh
```

Read `HANDOFF.md` before a full-data run. It documents the current state, resource limits, validation requirements, and work that remains.

## Submission status

- Source code, tests, official validator, and documentation template are present.
- The code-only executable package is tracked at the repository root.
- `output/matching_results.tsv` and `output/candidate_pairs.tsv` will be added only after a complete run passes the official validator.
- `Documentation_template.md` still needs the real team details and measured final-run results.

The final ZIP must preserve the `output/`, `code/business_entity_resolution/`, and root documentation layout shown here. See `code/business_entity_resolution/README.md` for the full reproduction guide.
