# Configuration

PhackingDetect is configured via **environment variables only** (no config files are required).

## LLM settings (required unless using `--offline`)

Set the following variables:

- `SKILL_LLM_API_KEY`
- `SKILL_LLM_BASE_URL` (must be **OpenAI-compatible**, e.g. ends with `/v1`)
- `SKILL_LLM_MODEL`

The model must support **image inputs** (the agent reads rendered PDF pages as JPGs).

Example:

```bash
export SKILL_LLM_API_KEY="..."
export SKILL_LLM_BASE_URL="https://your-openai-compatible-endpoint/v1"
export SKILL_LLM_MODEL="your-model-name"
```

Convenience:

- copy `.env.example` → `.env`
- run `source .env`
- never commit `.env`

## Offline mode (no API required)

If you do not want to call an LLM, run with:

```bash
./py scripts/p_hacking_agent.py --pdf path/to/paper.pdf --offline
```
