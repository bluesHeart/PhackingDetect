# Troubleshooting

## 1) “Missing required LLM environment variables…”

You are running without `--offline`, but one or more of these are not set:

- `SKILL_LLM_API_KEY`
- `SKILL_LLM_BASE_URL`
- `SKILL_LLM_MODEL`

Fix: set them (see `docs/CONFIGURATION.md`) or run with `--offline`.

## 2) Image / vision errors

The default workflow sends rendered PDF pages as **JPG images** to the model.

If your provider/model does not support image inputs, you may see errors mentioning:

- `image_url`
- “unsupported content type”
- “multimodal not supported”

Fix: use a vision-capable model, or run with `--offline`.

## 3) 403 / “VALIDATION_REQUIRED” / “request was blocked”

Some OpenAI-compatible gateways intermittently block requests (anti-bot / validation / provider gating).

What PhackingDetect does:

- retries transient blocks
- if system-role requests are blocked, it retries by folding system instructions into the user message (same model)

Fix if it persists:

- use a more stable provider/endpoint
- reduce request size (lower `--expert-max-artifacts`, `--expert-chunk-pages`, `--max-image-pages`)

## 4) Empty or truncated outputs

If the model returns empty content or truncated JSON, the agent will retry and may run a JSON “repair” prompt.

To inspect what happened, check:

- `reports/.../llm_logs/<run_id>/llm_calls.jsonl`
- `reports/.../llm_logs/<run_id>/calls/*_response_raw.json`

## 5) Outputs go to the wrong place / disk fills up

By default, outputs go under:

- `reports/p_hacking/<paper_slug>__<hash>/`

Use `--out-dir` to change the base output folder (still within the repo if you want).

