# Goal: JF-ready “within-paper p-hacking risk” measurement project

## Objective
Turn this repo from a working prototype into a **Journal of Finance–submission-ready** paper + reproducible codebase:

1. A scalable, auditable **within-paper p-hacking risk index** built from finance PDFs (SSRN → published versions).
2. A validation stack that can survive referee scrutiny (measurement validity, reliability, predictive validity).
3. A paper-quality empirical design that connects the index to finance outcomes (knowledge correction, publication, citations, replication, policy shocks).

## Non-goals / scope boundaries
- This project does **not** “prove p-hacking” in a legal sense. It produces **audit-first signals** with provenance (page/table/anchor) and uncertainty.
- The offline index must remain **fully reproducible without LLM calls**; multimodal LLM is for (i) per-paper audits and (ii) validation/labeling support.

## Deliverables (code + data + paper artifacts)
- `scripts/run_offline_pipeline.py`: end-to-end reproducible pipeline with run logs + config capture.
- A **defined sampling frame** and scalable corpus builder (SSRN/Crossref/OpenAlex), with de-duplication and version mapping.
- A cleaned test-level dataset: extracted regression-style test statistics with provenance and a defensible filtering/quality model.
- Validation artifacts:
  - extraction audit tasks + error rates,
  - human audit labels + IRR,
  - publication-map audit + accuracy.
- Analysis outputs producing the paper’s main tables/figures from raw corpora in one command.

