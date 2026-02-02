# PhackingDetect — within-paper selective-reporting / p-hacking risk screening agent

**LLM agent for p-hacking & selective-reporting risk screening in academic PDFs.**

PhackingDetect is an **end-to-end, evidence-grounded LLM workflow** that takes **one economics/finance paper PDF** and produces a **referee-style** selective-reporting / p-hacking **risk screening** report.

- **Input:** a PDF (`--pdf path/to/paper.pdf`)
- **Output (inside the repo):**
  - `reports/<paper_slug>__<hash>/diagnostic.md` (English, LLM-written)
  - `reports/<paper_slug>__<hash>/diagnostic_metrics.json` (machine-readable metrics)
  - `reports/<paper_slug>__<hash>/diagnostic.json` (full payload + provenance)
  - `reports/<paper_slug>__<hash>/llm_logs/<run_id>/` (prompts + raw responses)

This is a **risk screening tool**, not a misconduct detector: it does **not** infer intent. It tries to surface **where** risk signals appear (table/figure + page + best-effort row/col) and **why** they matter, grounded in the method references listed in `p_hacking_agent_methodology.md`.

---

## What this is / isn’t

**This is**

- A **within-paper** selective-reporting / p-hacking **risk screening** assistant.
- A workflow that produces **audit-ready** outputs (page + anchors + logs), so readers can quickly verify or reject claims.
- An open-source engineering artifact that operationalizes well-known diagnostics from the methods literature.

**This is not**

- A tool that “proves p-hacking” or detects misconduct/intent.
- A corpus-level estimator of publication bias or population stylized facts.
- A substitute for expert judgement or for reading and checking the paper.
- A guarantee of correctness: PDF parsing and LLM reading can be wrong; treat outputs as triage.

If you want to share results publicly, read `docs/RESPONSIBLE_USE.md` first.

## Output format (what you get)

The main machine-readable artifact is `diagnostic_metrics.json`, which is designed to be easy to parse and visualize.

Minimal schema (illustrative):

```json
{
  "generated_at": "YYYY-MM-DD HH:MM:SS",
  "paper": {
    "title": "…",
    "overall_risk_score_0_100": 0,
    "overall_risk_level": "Low | Moderate | High",
    "top_concerns": ["…"],
    "key_artifacts": ["Figure 5", "Table 7"]
  },
  "artifacts": [
    {
      "kind": "figure | table",
      "id": "Figure 5",
      "pages": [47, 48],
      "risk_score_0_100": 0,
      "risk_level": "Low | Moderate | High",
      "has_numeric_evidence": true,
      "evidence_counts": {
        "n_tests": 0,
        "p_005_just_sig": 0,
        "p_005_just_nonsig": 0,
        "t_196_just_below": 0,
        "t_196_just_above": 0,
        "t_exact_2_count": 0,
        "sig_p_le_0_05": 0,
        "sig_p_le_0_10": 0,
        "pcurve_low_half": 0,
        "pcurve_high_half": 0,
        "pcurve_right_skew_z": 0
      },
      "signals": ["…"],
      "flagged_entries": [
        {
          "page": 47,
          "row": "…",
          "col": "…",
          "coef": 0.0,
          "se_paren": 0.0,
          "abs_t": 0.0,
          "p_approx_2s": 0.0
        }
      ]
    }
  ]
}
```

## Quickstart

### 1) Create a local venv (recommended)

```bash
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
```

If `python3 -m venv` fails on Ubuntu/Debian, install the venv package:

```bash
sudo apt-get update && sudo apt-get install -y python3-venv
```

This repo includes wrappers that always use the repo venv:

- `./py` → runs `.venv/bin/python`
- `./pip` → runs `.venv/bin/pip`

### 2) Configure the LLM (strict env vars)

PhackingDetect uses an **OpenAI-compatible chat API** and reads **only** these variables:

- `SKILL_LLM_API_KEY`
- `SKILL_LLM_BASE_URL`
- `SKILL_LLM_MODEL`

Your chosen model must support **image inputs** (the agent reads rendered PDF pages as JPGs). If you only have a text-only model or do not want API calls, use `--offline`.

Example:

```bash
export SKILL_LLM_API_KEY="..."
export SKILL_LLM_BASE_URL="https://your-openai-compatible-endpoint/v1"
export SKILL_LLM_MODEL="your-model-name"
```

You can also copy `.env.example` to `.env` and `source .env` (do not commit `.env`).

If you do not want to call an LLM, run with `--offline` (heuristic mode).

### 3) Run the agent on one PDF

```bash
./py scripts/p_hacking_agent.py --pdf "path/to/paper.pdf"
```

Optional flags:

- `--out-dir reports/p_hacking` (base output directory)
- `--offline` (no LLM calls)
- `--force` (ignore cached outputs)
- `--force-metrics` / `--force-report` (expert mode rebuilds)

---

## How to audit a report (recommended)

PhackingDetect is designed to be **falsifiable**. When you receive `diagnostic.md`:

1) For each flagged item, note the **page number** and the **anchors** (short searchable phrases).
2) Open the PDF and jump to the cited page; use PDF search to find the anchor(s).
3) Confirm the anchor is actually near the cited table/figure and matches the described content.
4) If a claim seems important, check the raw model trace under `llm_logs/<run_id>/` to see what was provided to the model and what it returned.

If the anchors can’t be found, treat the item as a false positive.

## Repo layout (core)

- `scripts/p_hacking_agent.py`: the end-to-end agent (PDF → report)
- `scripts/extract_within_paper_metrics.py`: offline within-paper test extraction used as numeric evidence
- `p_hacking_agent_methodology.md`: **the only allowed** method reference list (citations are constrained to this file)
- `tex/`: a sectioned LaTeX draft (`tex/build/main.tex`) that cites key p-hacking / selective-reporting methods
- `docs/CONFIGURATION.md`: env vars and offline mode
- `docs/TROUBLESHOOTING.md`: common setup/runtime issues
- `docs/FAQ.md`: quick answers for new users
- `docs/WHY_THIS_WORKS.md`: design rationale and why this is more than an LLM wrapper
- `docs/RESPONSIBLE_USE.md`: recommended non-accusatory usage
- `reports/`: generated reports (ignored by git)
- `trash/`: a local “bucket” for moved research artifacts / corpora / drafts (ignored by git)

---

## License

MIT (see `LICENSE`).
