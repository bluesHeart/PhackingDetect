# Docs

This repo is intentionally minimal and centered on the end-to-end agent:

- `scripts/p_hacking_agent.py` (PDF → referee-style risk screening report)
- `scripts/extract_within_paper_metrics.py` (offline numeric evidence extraction)
- `p_hacking_agent_methodology.md` (allowed method references for citations)
- `tex/` (a sectioned LaTeX draft for a paper-style writeup)

Old research pipelines, corpora, and manuscript drafts were moved into `trash/` to keep the repo clean.

## Key docs

- `docs/CONFIGURATION.md`: required environment variables and offline mode
- `docs/TROUBLESHOOTING.md`: common setup/runtime issues and where to look in logs
- `docs/FAQ.md`: quick answers for new users
- `docs/WHY_THIS_WORKS.md`: design rationale, auditability, and why this is more than an LLM wrapper
- `docs/RESPONSIBLE_USE.md`: non-accusatory framing and recommended usage
