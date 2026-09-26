# Amazon ML Challenge 2026

Business Entity Resolution submission workspace, arranged to match the official final-package format. The repository contains the current memory-safe implementation, run instructions, architecture documentation, methodology template, and a code-only macOS package.

## Repository map

| Path | Purpose |
| --- | --- |
| `code/business_entity_resolution/` | Reproducible Python implementation |
| `code/business_entity_resolution/src/` | Resumable normalization, blocking, feature, model, and reporting pipeline |
| `code/business_entity_resolution/tests/` | Ten unit and regression tests |
| `code/business_entity_resolution/configs/` | Experiment sweep configuration |
| `code/business_entity_resolution/docs/` | Architecture, Q&A, and SVG diagrams |
| `output/` | Destination for the two validated submission TSVs |
| `Documentation_template.md` | Official methodology template to complete |
| `HANDOFF.md` | Current project state and execution guidance |
| `START_HERE.txt` | Short MacBook instructions |
| `PLAN.md` | Plain-language MacBook execution and submission plan |
| `executable.zip` | Code-only copy of `CODE_ONLY_AmazonML_no_data.zip` |

## Quick start

The private dataset and the 1.1 GB code-and-data transfer ZIP are intentionally excluded from Git. To check the source implementation:

```powershell
cd code\business_entity_resolution
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`executable.zip` expands to `AmazonML_Project/`. It is code-only, so the official data ZIP must be placed inside that folder before running it:

```bash
unzip executable.zip
mv /path/to/OFFICIAL_DATA_student_resource.zip AmazonML_Project/
bash ~/AmazonML_Project/run.sh
```

For the teammate transfer, use the private `SEND_TO_MAC_AmazonML_code_and_data.zip` kept outside Git. It already contains the data under the exact name expected by `PLAN.md`. Read `START_HERE.txt`, then `PLAN.md`, before the full-data run.

## Submission status

- Current source code, ten tests, architecture docs, and the official documentation template are present.
- The official validator is loaded from the private official resource package during execution.
- The code-only package is tracked as `executable.zip`; its source name outside Git is `CODE_ONLY_AmazonML_no_data.zip`.
- `output/matching_results.tsv` and `output/candidate_pairs.tsv` will be added only after a complete run passes the official validator.
- `Documentation_template.md` still needs the real team details and measured final-run results.

The final ZIP must preserve the `output/`, `code/business_entity_resolution/`, and root documentation layout shown here. See `code/business_entity_resolution/README.md` for the full reproduction guide.
