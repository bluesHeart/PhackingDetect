#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def main() -> int:
    ap = argparse.ArgumentParser(description="Prototype predictive-validity regressions using OpenAlex citation counts.")
    ap.add_argument("--panel", default="analysis/paper_panel_with_openalex.csv")
    ap.add_argument("--outcome", default="ssrn", choices=["ssrn", "published"], help="Which OpenAlex work to use for outcomes.")
    ap.add_argument("--published-top3-only", action="store_true", help="For outcome=published: restrict to JF/JFE/RFS matches (best_match_is_top3_venue).")
    ap.add_argument("--published-require-confident", action="store_true", help="For outcome=published: restrict to confident title matches.")
    ap.add_argument("--published-min-title-sim", type=float, default=None, help="For outcome=published: restrict to title similarity >= threshold.")
    ap.add_argument("--out-md", default=None, help="Output markdown (default: <panel_dir>/predictive_validity_openalex.md).")
    args = ap.parse_args()

    panel_path = Path(args.panel)
    if not panel_path.exists():
        raise SystemExit(f"Missing: {panel_path}")

    if args.out_md:
        out_md = Path(args.out_md)
    else:
        name = "predictive_validity_openalex.md" if args.outcome == "ssrn" else "predictive_validity_openalex_published.md"
        out_md = panel_path.parent / name
    out_md.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(panel_path)

    if "offline_risk_score" not in df.columns:
        raise SystemExit("panel missing column: offline_risk_score")

    if args.outcome == "ssrn":
        y_col = "openalex_log1p_cites"
        year_col = "openalex_publication_year"
    else:
        y_col = "published_log1p_cites"
        year_col = "published_publication_year"

    for c in [y_col, year_col]:
        if c not in df.columns:
            raise SystemExit(f"panel missing column: {c}")

    # Clean sample
    df["offline_risk_score"] = pd.to_numeric(df["offline_risk_score"], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df["pages"] = pd.to_numeric(df.get("pages"), errors="coerce")
    df["t_pairs_seen"] = pd.to_numeric(df.get("t_pairs_seen"), errors="coerce")
    df["log1p_t_pairs_seen"] = df["t_pairs_seen"].apply(lambda x: (np.log1p(x) if x == x and x is not None else np.nan))

    base = df.dropna(subset=["offline_risk_score", y_col, year_col]).copy()

    if args.outcome == "published":
        if args.published_top3_only:
            if "published_is_top3_venue" not in base.columns:
                raise SystemExit("panel missing column: published_is_top3_venue (rebuild panel_with_openalex with pub map)")
            base = base[base["published_is_top3_venue"].astype(bool)].copy()
        if args.published_require_confident:
            if "published_is_confident_match" not in base.columns:
                raise SystemExit("panel missing column: published_is_confident_match (rebuild panel_with_openalex with pub map)")
            base = base[base["published_is_confident_match"].astype(bool)].copy()
        if args.published_min_title_sim is not None:
            if "published_title_sim_seq" not in base.columns:
                raise SystemExit("panel missing column: published_title_sim_seq (rebuild panel_with_openalex with pub map)")
            base = base[pd.to_numeric(base["published_title_sim_seq"], errors="coerce") >= float(args.published_min_title_sim)].copy()

    base["year_i"] = base[year_col].astype(int)
    if len(base) < 50:
        raise SystemExit("Too few observations with OpenAlex outcomes to run regressions.")

    # Model specs (prototype)
    specs = [
        ("M1", f"{y_col} ~ offline_risk_score + C(year_i)"),
        ("M2", f"{y_col} ~ offline_risk_score + pages + log1p_t_pairs_seen + C(year_i)"),
    ]

    results_rows = []
    for name, formula in specs:
        m = smf.ols(formula=formula, data=base).fit(cov_type="HC1")
        coef = float(m.params.get("offline_risk_score", np.nan))
        se = float(m.bse.get("offline_risk_score", np.nan))
        t = float(m.tvalues.get("offline_risk_score", np.nan))
        p = float(m.pvalues.get("offline_risk_score", np.nan))
        n = int(m.nobs)
        r2 = float(m.rsquared)
        results_rows.append({"model": name, "coef_score": coef, "se": se, "t": t, "p": p, "n": n, "r2": r2, "formula": formula})

    out_csv = out_md.with_suffix(".csv")
    pd.DataFrame(results_rows).to_csv(out_csv, index=False)

    # Simple correlation (nonparametric)
    corr = base[["offline_risk_score", y_col]].corr(method="spearman").iloc[0, 1]
    corr = float(corr) if corr == corr else None

    lines: list[str] = []
    title = "SSRN DOI citations" if args.outcome == "ssrn" else "published-version citations (best OpenAlex title-search match)"
    lines.append(f"# Predictive validity (prototype): OpenAlex citations — {title}")
    lines.append("")
    lines.append(f"> Generated: {_now_iso()}")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append(f"- Input: `{panel_path.name}`")
    lines.append(f"- Outcome: `{y_col}` (year FE: `{year_col}`)")
    lines.append(f"- N (with outcome + year): {len(base)}")

    if args.outcome == "published":
        n_total = int(len(df))
        if "published_has_non_ssrn_candidate" in df.columns:
            n_any = int(df["published_has_non_ssrn_candidate"].fillna(False).astype(bool).sum())
            lines.append(f"- Has any non-SSRN candidate: {n_any}/{n_total} ({n_any/max(1,n_total):.3f})")
        if "published_is_confident_match" in df.columns:
            n_conf = int(df["published_is_confident_match"].fillna(False).astype(bool).sum())
            lines.append(f"- Confident title match: {n_conf}/{n_total} ({n_conf/max(1,n_total):.3f})")
        if "published_is_top3_venue" in df.columns:
            n_top = int(df["published_is_top3_venue"].fillna(False).astype(bool).sum())
            lines.append(f"- Top-3 venue match (JF/JFE/RFS): {n_top}/{n_total} ({n_top/max(1,n_total):.3f})")
    lines.append("")
    if corr is not None:
        lines.append(f"- Spearman(offline_risk_score, log1p(cites)): {corr:.3f}")
        lines.append("")

    lines.append("## Regressions (HC1 robust SE)")
    lines.append("")
    lines.append("| model | coef(score) | se | t | p | n | r2 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in results_rows:
        lines.append(f"| {r['model']} | {r['coef_score']:.4f} | {r['se']:.4f} | {r['t']:.3f} | {r['p']:.4f} | {r['n']} | {r['r2']:.3f} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    if args.outcome == "ssrn":
        lines.append("- This is a *prototype* predictive-validity check; SSRN DOI citations may not capture citations to later published versions.")
        lines.append("- Next JF step: link preprints to published DOIs/venues (OpenAlex related_works / title-author matching) and use citation trajectories.")
    else:
        lines.append("- This uses a best-effort mapping from SSRN works to a likely published version via OpenAlex title search + (optional) author overlap.")
        lines.append("- Next JF step: validate mapping accuracy (manual audit) and use citation trajectories + published-venue outcomes.")
    lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{_now_iso()}] wrote: {out_md}")
    print(f"[{_now_iso()}] wrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
