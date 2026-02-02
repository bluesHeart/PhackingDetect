# Responsible use

PhackingDetect is a **within-paper risk screening** tool for selective reporting / p-hacking signals. It is designed to help a human reader triage a paper, not to convict authors.

## What you may do

- Use the tool as an **internal QA** assistant before submission.
- Use it as a **referee/reviewer aid** to locate potentially fragile results and propose follow-up checks.
- Use it for **teaching** and reproducible discussions about selective reporting diagnostics.

## What you should not do

- Do not frame outputs as proof of misconduct or intent.
- Do not publish “blacklists” of papers/authors based on automated reports.
- Do not treat the report as a substitute for reading the paper and checking the cited evidence.

## Reporting style recommendations

When sharing reports, prefer language like:

- “risk signal”, “worth checking”, “suggestive pattern”, “requires follow-up”

Avoid language like:

- “fraud”, “cheating”, “p-hacking confirmed”, “manipulation”

## Known failure modes

- PDF parsing can be wrong (mis-ordered text, missing lines, scanned tables).
- The model may misread dense tables; always verify with the cited page + anchors.
- Many diagnostics are suggestive; alternative explanations are often plausible.

## Privacy and API keys

- Never commit API keys. Configure LLM access via environment variables only.
- Generated reports may contain paper content; ensure you have the rights to store/share those PDFs and outputs.

