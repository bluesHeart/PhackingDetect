#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _safe_int(x: Any) -> int | None:
    try:
        if isinstance(x, bool):
            return None
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            if x != x:
                return None
            return int(x)
        if isinstance(x, str) and x.strip():
            return int(float(x.strip()))
    except Exception:
        return None
    return None


def _safe_float(x: Any) -> float | None:
    try:
        if isinstance(x, bool):
            return None
        if isinstance(x, (int, float)):
            v = float(x)
            return None if v != v else v
        if isinstance(x, str) and x.strip():
            v = float(x.strip())
            return None if v != v else v
    except Exception:
        return None
    return None


def _norm_title(s: Any) -> str | None:
    if not isinstance(s, str):
        return None
    t = s.strip().lower()
    if not t or len(t) < 6:
        return None
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[\"'“”‘’`]", "", t)
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) < 6:
        return None
    return t


def _write_duplicates(panel: pd.DataFrame, out_dir: Path) -> tuple[Path, int]:
    df = panel.copy()
    df["norm_title"] = df["title"].apply(_norm_title)
    df = df[df["norm_title"].notna()]
    g = df.groupby("norm_title", dropna=True)

    rows: list[dict[str, Any]] = []
    dup_groups = 0
    for norm, sub in g:
        if len(sub) <= 1:
            continue
        dup_groups += 1
        sub2 = sub.copy()
        sub2["quality_proxy"] = sub2["extracted_text_chars"].fillna(0)
        sub2["pages_proxy"] = sub2["pages"].fillna(0)
        sub2["t_pairs_proxy"] = sub2["t_pairs_seen"].fillna(0)
        sub2 = sub2.sort_values(["quality_proxy", "pages_proxy", "t_pairs_proxy"], ascending=False)
        chosen = sub2.iloc[0]
        ssrn_ids = [str(int(x)) for x in sub["ssrn_id"].dropna().astype(int).tolist()] if "ssrn_id" in sub else []
        years = [str(int(x)) for x in sub["year"].dropna().astype(int).tolist()] if "year" in sub else []
        rows.append(
            {
                "norm_title": norm,
                "count": int(len(sub)),
                "title_example": str(chosen.get("title") or ""),
                "chosen_ssrn_id": _safe_int(chosen.get("ssrn_id")),
                "chosen_pdf_relpath": chosen.get("pdf_relpath"),
                "chosen_extracted_text_chars": _safe_int(chosen.get("extracted_text_chars")),
                "chosen_pages": _safe_int(chosen.get("pages")),
                "chosen_t_pairs_seen": _safe_int(chosen.get("t_pairs_seen")),
                "ssrn_ids": ";".join(ssrn_ids),
                "years": ";".join(years),
            }
        )

    out_csv = out_dir / "title_duplicates.csv"
    if rows:
        pd.DataFrame(rows).sort_values(["count", "norm_title"], ascending=[False, True]).to_csv(out_csv, index=False)
    else:
        pd.DataFrame(columns=["norm_title", "count", "title_example", "chosen_ssrn_id", "ssrn_ids", "years"]).to_csv(
            out_csv, index=False
        )
    return out_csv, dup_groups


def _save_figs(panel: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    df = panel.copy()

    # Histogram of offline risk score
    if "offline_risk_score" in df.columns:
        s = pd.to_numeric(df["offline_risk_score"], errors="coerce").dropna()
        if len(s) > 0:
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            ax.hist(s, bins=24, color="#2E6F9E", edgecolor="white")
            ax.set_title("Offline within-paper p-hacking risk score (distribution)")
            ax.set_xlabel("Score (0–100, higher = more to audit)")
            ax.set_ylabel("Paper count")
            fig.tight_layout()
            p = out_dir / "fig_score_hist.png"
            fig.savefig(p, dpi=180)
            plt.close(fig)
            out["fig_score_hist"] = p

    # Score vs year (mean by year)
    if "year" in df.columns and "offline_risk_score" in df.columns:
        df2 = df.copy()
        df2["year_i"] = pd.to_numeric(df2["year"], errors="coerce").astype("Int64")
        df2["score_f"] = pd.to_numeric(df2["offline_risk_score"], errors="coerce")
        df2 = df2.dropna(subset=["year_i", "score_f"])
        if len(df2) > 3:
            by = df2.groupby("year_i")["score_f"].agg(["mean", "count"]).reset_index()
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            ax.plot(by["year_i"].astype(int), by["mean"], color="#2E6F9E", lw=2)
            ax.scatter(by["year_i"].astype(int), by["mean"], color="#2E6F9E", s=18)
            ax.set_title("Mean offline risk score by year (SSRN Electronic Journal)")
            ax.set_xlabel("Year")
            ax.set_ylabel("Mean score")
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            p = out_dir / "fig_score_by_year.png"
            fig.savefig(p, dpi=180)
            plt.close(fig)
            out["fig_score_by_year"] = p

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate basic stylized facts + figures from analysis/paper_panel.csv.")
    ap.add_argument("--panel", default="analysis/paper_panel.csv", help="Input panel CSV.")
    ap.add_argument("--out-dir", default="analysis", help="Output directory.")
    ap.add_argument("--top-n", type=int, default=20, help="Top-N papers by risk score to list.")
    args = ap.parse_args()

    panel_path = Path(args.panel)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not panel_path.exists():
        raise SystemExit(f"Missing: {panel_path}")

    panel = pd.read_csv(panel_path)
    figs = _save_figs(panel, out_dir)
    dup_csv, dup_groups = _write_duplicates(panel, out_dir)

    # Core summary stats
    n = int(len(panel))
    years = pd.to_numeric(panel.get("year"), errors="coerce")
    score = pd.to_numeric(panel.get("offline_risk_score"), errors="coerce")
    t_pairs = pd.to_numeric(panel.get("t_pairs_seen"), errors="coerce")
    keep_rate = pd.to_numeric(panel.get("t_pairs_keep_rate"), errors="coerce") if "t_pairs_keep_rate" in panel.columns else None

    year_min = int(years.min()) if years.notna().any() else None
    year_max = int(years.max()) if years.notna().any() else None

    score_desc = None
    if score.notna().any():
        score_desc = score.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()

    keep_desc = None
    if keep_rate is not None and keep_rate.notna().any():
        keep_desc = keep_rate.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()

    top = panel.copy()
    if "offline_risk_score" in top.columns:
        top["offline_risk_score"] = pd.to_numeric(top["offline_risk_score"], errors="coerce")
        top = top.sort_values("offline_risk_score", ascending=False)
    top = top.head(int(args.top_n))

    out_md = out_dir / "stylized_facts.md"
    lines: list[str] = []
    lines.append("# Stylized facts (prototype): within-paper p-hacking risk on SSRN PDFs")
    lines.append("")
    lines.append(f"> Generated: {_now_iso()}")
    lines.append("")
    lines.append("## Snapshot")
    lines.append("")
    lines.append(f"- Papers: {n}")
    if year_min is not None and year_max is not None:
        lines.append(f"- Years: {year_min}-{year_max}")
    if score_desc:
        keys = ["min", "10%", "25%", "50%", "75%", "90%", "max", "mean"]
        parts = [f"{k}={float(score_desc[k]):.2f}" for k in keys if k in score_desc and score_desc[k] == score_desc[k]]
        lines.append(f"- Offline risk score: {', '.join(parts)}")
    lines.append(f"- Duplicate-title groups (heuristic): {dup_groups} (see `{dup_csv.name}`)")
    if "skipped_reason" in panel.columns:
        n_skipped = int(panel["skipped_reason"].fillna("").astype(str).str.strip().ne("").sum())
        if n_skipped > 0:
            lines.append(f"- Skipped PDFs (e.g., too many pages): {n_skipped}")
    if t_pairs.notna().any():
        n_pairs0 = int((t_pairs.fillna(0) <= 0).sum())
        lines.append(f"- Papers with t_pairs_seen==0: {n_pairs0} ({(n_pairs0/max(1,n)):.3f})")
    if keep_desc:
        keys = ["10%", "25%", "50%", "75%", "90%", "mean"]
        parts = [f"{k}={float(keep_desc[k]):.3f}" for k in keys if k in keep_desc and keep_desc[k] == keep_desc[k]]
        lines.append(f"- Extraction keep-rate (kept/raw): {', '.join(parts)}")
    if "paren_mode" in panel.columns:
        vc = panel["paren_mode"].fillna("").astype(str).str.strip().replace({"": "unknown"}).value_counts()
        if len(vc) > 0:
            parts = [f"{k}={int(v)}" for k, v in vc.head(6).items()]
            lines.append(f"- Parentheses mode: {', '.join(parts)}")
    lines.append("")

    if "fig_score_hist" in figs:
        lines.append("## Figures")
        lines.append("")
        lines.append(f"![]({figs['fig_score_hist'].name})")
        if "fig_score_by_year" in figs:
            lines.append("")
            lines.append(f"![]({figs['fig_score_by_year'].name})")
        lines.append("")

    lines.append("## Top papers by offline score (audit-first list)")
    lines.append("")
    lines.append("| rank | ssrn_id | year | score | title | ssrn_url |")
    lines.append("|---:|---:|---:|---:|---|---|")
    for idx, row in enumerate(top.itertuples(index=False), start=1):
        ssrn_id = getattr(row, "ssrn_id", "")
        year = getattr(row, "year", "")
        sc = getattr(row, "offline_risk_score", "")
        title = getattr(row, "title", "")
        url = getattr(row, "ssrn_url", "")
        title_s = str(title).replace("|", "\\|") if isinstance(title, str) else ""
        url_s = str(url).replace("|", "\\|") if isinstance(url, str) else ""
        lines.append(f"| {idx} | {ssrn_id} | {year} | {sc} | {title_s} | {url_s} |")
    lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{_now_iso()}] wrote: {out_md}")
    print(f"[{_now_iso()}] wrote: {dup_csv}")
    for k, p in figs.items():
        print(f"[{_now_iso()}] wrote: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
