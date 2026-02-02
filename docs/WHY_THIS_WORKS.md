# Why PhackingDetect is (more than) an LLM wrapper

This project is an **end-to-end, evidence-grounded workflow** for *within-paper* selective-reporting / p-hacking **risk screening**.

It is intentionally **not** framed as a definitive detector and it **does not infer intent**. The value is that it helps a human expert triage a paper by surfacing *where* risk signals appear and *why* they matter, with **audit-ready evidence pointers**.

---

## 1) What problem this solves

Method papers in the p-hacking / selective-reporting literature provide diagnostic concepts (threshold bunching, multiplicity, specification search, publication bias, p-curve, etc.). In practice, applying those concepts to a single paper is expensive:

- evidence lives in **tables/figures + notes + appendices**
- a reader must connect **claims ↔ specific results ↔ robustness structure**
- many signals are **suggestive**, so the reader must consider **competing explanations**

PhackingDetect is designed to reduce this friction without claiming statistical certainty.

---

## 2) What “effective” means here (and what it doesn’t)

**Effective** means:

- The report is **falsifiable**: every flagged item points to a *page* and *searchable anchors*.
- The reasoning is **theory-mapped**: each diagnostic claim ties back to specific method references (see `p_hacking_agent_methodology.md`).
- The output is **audit-friendly**: prompts and raw model responses are logged so that reviewers can inspect what the model saw.

**Not effective** (out of scope by design):

- “Proving p-hacking” or “detecting misconduct”
- Corpus-level inference about an entire literature (publication bias estimation, population stylized facts)
- Automatically determining the “true” specification space when it is not described in the paper

---

## 3) The core design principles

### 3.1 Evidence grounding beats eloquence

The most damaging failure mode for LLM-generated analyses is **unverifiable narrative**. This repo enforces grounding:

- each flagged claim must include **page number + anchors**
- artifacts are analyzed at the **table/figure** level
- when possible, the system adds **lightweight numeric cross-checks** (e.g., extracted p-values near 0.05)

If the anchor cannot be found in the PDF, the claim is wrong.

### 3.2 Inference chains (not “magic answers”)

Within-paper signals are rarely definitive. The report therefore uses an explicit chain:

> Observation → Diagnostic → Why it matters (with citation) → Alternative explanations → Checks

This is what makes the report useful even when it ends with uncertainty.

### 3.3 Theory-constrained citations

To prevent “made up” academic-sounding justification, citations are constrained to the method list in:

- `p_hacking_agent_methodology.md`

The system is expected to cite only what is in that file (author–year + reference list).

### 3.4 Auditability: log everything that matters

The agent logs:

- prompts and intermediate JSON outputs
- raw model responses
- final report + machine-readable metrics

This makes it easier to debug mistakes and improves credibility for open-source users.

---

## 4) Diagnostics and methodological basis (high level)

The agent operationalizes a small set of **screening diagnostics**, each tied to method references:

- **Near-threshold clustering** (e.g., “just significant” patterns near 0.05 / |t|≈1.96)
- **Multiplicity risk** (many outcomes/subgroups/specs without correction or higher thresholds)
- **Specification search cues** (fragility / selective robustness emphasis)
- **Alternative explanations** (publication and reporting bias)
- **Downstream p-curve** as a follow-up when the set of primary tests can be defined responsibly

For citations, see `p_hacking_agent_methodology.md`.

---

## 5) How to avoid “AI-generated junk” accusations

Open-source users tend to accept LLM-based tools when the project:

1) **does not oversell** (clear non-goals and limitations)
2) is **auditable** (evidence pointers + logs)
3) is **operational** (one command produces structured outputs)
4) provides **method grounding** (references + explicit logic)

The repository is structured to meet these expectations without shipping copyrighted PDFs or attacking specific authors.

