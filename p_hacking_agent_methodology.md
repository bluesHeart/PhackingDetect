# Multimodal within-paper p-hacking risk screening agent: methods and sources

This repository's `scripts/p_hacking_agent.py` is an end-to-end **within-paper** selective-reporting / p-hacking **risk screening** workflow that intentionally uses **no agent framework**.

The core goal is to maximize practical signal-finding at **low cost** (a limited budget of page-image reading + lightweight text heuristics) while keeping outputs **auditable**: evidence must be grounded to **page numbers + anchors** (searchable phrases, table/figure labels, or note sentences).

## What the agent actually does (implementation-level)

1) **Multimodal evidence extraction (core)**  
   - Uses **PyMuPDF** for PDF text extraction (may be noisy) and **page rendering** (often more reliable for tables/figures).  
   - The LLM extracts, per page/artifact: table/figure names, star conventions, borderline results (near thresholds), multiplicity cues, robustness/specification-search cues, and selective-emphasis language, and emits anchors.

2) **Cost-effective page triage (find tables and robustness pages)**  
   - Scores pages using heuristic features (numeric density, Table/Robust/p-value keywords, star density).  
   - Optionally lets the LLM pick a small set of key pages from the page summaries; if that fails, falls back to heuristic selection.

3) **Low-cost within-paper “caliper” approximations near thresholds**  
   - Extracts reported `p=...` / `p<...` patterns from text and summarizes counts near `0.05/0.10/0.01` (a coarse within-paper analog of threshold bunching).  
   - More importantly, asks the LLM to locate borderline cells directly in **table images** (e.g., “t≈1.96 / p≈0.05 / star just appears”) and provide anchors.

4) **Multiplicity risk checks (many outcomes / many specs / many subsamples)**  
   - Detects “many outcomes / many columns / many heterogeneity splits / many mechanism proxies” and checks whether the paper documents corrections (FWER/FDR) or higher thresholds / pre-specification constraints.

5) **Specification search / selective robustness reporting cues**  
   - Flags: heavy specification switching; significance that flips with small control/sample changes; selective emphasis of significant variants; null results moved to appendices without discussion; etc., always with anchors.

6) **Offline scalable evidence layer (reproducible statistics prototype)**  
   - Detects table-like text, reconstructs `(coef, se) → t → p` when possible, and computes:
     - caliper summaries near 0.05/0.10 (including simple z / binomial tests),
     - threshold summaries near |t|≈1.96/1.645,
     - simplified p-curve shape summaries (e.g., “mass near 0.05” within significant region).
   - Entry point: `scripts/extract_within_paper_metrics.py`.

> Note: the agent does not implement heavyweight procedures (full p-curve inference, structural publication-bias estimation). Instead, it treats them as **auditable follow-up recommendations** when within-paper signals justify deeper analysis.

## Method sources (author–year)

The modules above are grounded in the following methods literature and classic references:

- **Threshold bunching / inflated significance, and threshold-based diagnostics:** (Brodeur et al., 2016; Brodeur et al., 2020)  
- **Formal detection of p-hacking (as diagnostic and follow-up):** (Elliott et al., 2022)  
- **p-curve shape diagnostics (reference for downstream checks):** (Simonsohn et al., 2014a; Simonsohn et al., 2014b)  
- **Multiple testing and higher evidentiary thresholds:** (Harvey et al., 2015; Harvey, 2017)  
- **Publication bias identification and correction (alternative explanations):** (Andrews & Kasy, 2019)  
- **Specification search and researcher degrees of freedom:** (Leamer, 1978; Leamer, 1983)  

## References (APA, with DOI)

- Andrews, I., & Kasy, M. (2019). Identification of and correction for publication bias. *American Economic Review, 109*(8), 2766–2794. https://doi.org/10.1257/aer.20180310
- Brodeur, A., Cook, N., & Heyes, A. (2020). Methods matter: p-hacking and publication bias in causal analysis in economics. *American Economic Review, 110*(11), 3634–3660. https://doi.org/10.1257/aer.20190687
- Brodeur, A., Lé, M., Sangnier, M., & Zylberberg, Y. (2016). Star wars: The empirics strike back. *American Economic Journal: Applied Economics, 8*(1), 1–32. https://doi.org/10.1257/app.20150044
- Elliott, G., Kudrin, N., & Wüthrich, K. (2022). Detecting p-hacking. *Econometrica, 90*(2), 887–906. https://doi.org/10.3982/ecta18583
- Harvey, C. R. (2017). Presidential address: The scientific outlook in financial economics. *The Journal of Finance, 72*(4), 1399–1440. https://doi.org/10.1111/jofi.12530
- Harvey, C. R., Liu, Y., & Zhu, H. (2015). … and the cross-section of expected returns. *Review of Financial Studies, 29*(1), 5–68. https://doi.org/10.1093/rfs/hhv059
- Leamer, E. E. (1983). Let's take the con out of econometrics. *The American Economic Review, 73*(1), 31–43. (No DOI found.)
- Leamer, E. E. (1978). *Specification Searches: Ad Hoc Inference with Nonexperimental Data*. John Wiley & Sons. (No DOI; book.)
- Simonsohn, U., Nelson, L. D., & Simmons, J. P. (2014a). P-curve: A key to the file-drawer. *Journal of Experimental Psychology: General, 143*(2), 534–547. https://doi.org/10.1037/a0033242
- Simonsohn, U., Nelson, L. D., & Simmons, J. P. (2014b). *p*-Curve and effect size. *Perspectives on Psychological Science, 9*(6), 666–681. https://doi.org/10.1177/1745691614553988
