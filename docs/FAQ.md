# FAQ

## Is this “detecting p-hacking”?

No. PhackingDetect is a **within-paper risk screening** assistant. It surfaces *where* suggestive signals appear and *why* they matter, with **page numbers + searchable anchors** so a human can verify the evidence. It does **not** infer intent or misconduct.

## Do I need an API key?

Only if you run with LLM enabled (default). If you do not want API calls, run:

```bash
./py scripts/p_hacking_agent.py --pdf path/to/paper.pdf --offline
```

## What kind of LLM endpoint do I need?

An **OpenAI-compatible** endpoint (base URL typically ends with `/v1`) and a model that supports **image inputs** (the agent sends rendered PDF pages as JPGs).

See `docs/CONFIGURATION.md`.

## Where do outputs go?

By default:

- `reports/p_hacking/<paper_slug>__<hash>/`

Change the base output folder with `--out-dir` (e.g., keep everything under the repo to avoid filling system disks).

## How do I verify the report isn’t hallucinating?

Audit the evidence:

1) For each flagged item, note the cited **page** and **anchors** (short phrases).
2) Open the PDF, jump to the page, and search the anchor text.
3) Confirm the anchor matches the described table/figure content.

See `README.md` (“How to audit a report”).

## How do I reduce cost?

Cost mainly scales with pages and artifacts analyzed. You can reduce it by lowering:

- `--max-image-pages`
- `--expert-max-artifacts`
- `--expert-chunk-pages`

Or switch off the full expert workflow with `--no-expert`, or run `--offline`.

## Do you ship paper PDFs or attack specific authors?

No. The repository does not include copyrighted paper PDFs, and the project is not designed to “call out” individual authors. It is intended for internal QA, reviewer assistance, and teaching about diagnostic concepts.

