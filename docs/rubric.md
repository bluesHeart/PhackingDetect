# Rubric: “JF-ready” definition of done (engineering + research)

This rubric is intentionally **reject-oriented**: if a criterion fails, we know exactly why we are not ready yet.

## 1) Sampling frame (data)
- **Frame definition**: explicit population definition (what journals/venues, what years, inclusion/exclusion, versioning).
- **Scale**: ≥ 3,000 finance/econ-finance PDFs (order-of-magnitude appropriate for JF stylized facts).
- **De-duplication**: robust title+author+doi+hash de-dup, with a documented rule and a report.
- **Version mapping**: SSRN preprint → likely published version; mapping accuracy supported by manual audit.

## 2) Measurement validity (extraction)
- **Regression-test extraction quality**:
  - cell-level audit dataset exists (snippets + labels) and is reproducible (`annotations/extraction_*`),
  - precision/recall (or error-rate) reported for key fields: table detection, page selection, `abs_t`/p-value reconstruction,
  - explicit handling of “parentheses contain t-stats vs standard errors” (documented + measured accuracy).
- **Noise controls**:
  - reference/bibliography pages excluded from candidate-page selection,
  - citation-like numeric artifacts filtered with measured false-positive rate.

## 3) Reliability (stability)
- Score stability under reasonable perturbations:
  - different page budgets,
  - different extraction settings (table settings / filtering thresholds),
  - repeated runs.
- Human-audit agreement:
  - multi-annotator IRR for paper-level red flags (Cohen’s kappa / rank correlations).

## 4) Predictive validity (finance contribution)
- A pre-specified empirical design that links the risk index to finance outcomes:
  - citations / published venue / revision / retraction / replication outcomes, or
  - “knowledge correction” signatures (effect shrinkage, factor attenuation, etc.).
- Robustness to mechanical confounds:
  - controls for paper length, number of tables, extraction coverage, year fixed effects, etc.

## 5) Reproducibility (engineering hygiene)
- One command reproduces all main exhibits from raw corpus:
  - outputs versioned run logs + config,
  - pinned Python environment (`.venv/`, `requirements.txt`).
- Clear provenance for any LLM-assisted step (prompt + response logs + costs).

