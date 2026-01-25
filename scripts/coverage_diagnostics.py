#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def main() -> int:
    ap = argparse.ArgumentParser(description="Coverage diagnostics for table-based extraction (JF hygiene).")
    ap.add_argument("--panel", default="analysis/paper_panel.csv", help="Paper-level panel CSV.")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: panel's parent dir).")
    args = ap.parse_args()

    panel_path = Path(args.panel)
    if not panel_path.exists():
        raise SystemExit(f"Missing: {panel_path}")
    out_dir = Path(args.out_dir) if args.out_dir else panel_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(panel_path)
    if "offline_risk_score" not in df.columns:
        raise SystemExit("panel missing offline_risk_score")
    if "t_pairs_seen" not in df.columns:
        raise SystemExit("panel missing t_pairs_seen")

    df["score"] = pd.to_numeric(df["offline_risk_score"], errors="coerce")
    df["t_pairs"] = pd.to_numeric(df["t_pairs_seen"], errors="coerce").fillna(0.0)
    df["log_t_pairs"] = df["t_pairs"].apply(lambda x: math.log1p(max(0.0, float(x))))

    keep_rate_available = False
    if "t_pairs_keep_rate" in df.columns:
        df["keep_rate"] = pd.to_numeric(df["t_pairs_keep_rate"], errors="coerce")
        keep_rate_available = True
    elif "t_pairs_seen_raw" in df.columns:
        raw = pd.to_numeric(df["t_pairs_seen_raw"], errors="coerce")
        df["keep_rate"] = df["t_pairs"] / raw.where(raw > 0)
        keep_rate_available = True

    n = int(len(df))
    n_score = int(df["score"].notna().sum())
    n_pairs0 = int((df["t_pairs"] <= 0).sum())
    frac_pairs0 = float(n_pairs0 / max(1, n))
    n_keep = int(df["keep_rate"].notna().sum()) if keep_rate_available else 0
    frac_keep_lt_090 = (
        float((df["keep_rate"] < 0.90).sum() / max(1, n_keep)) if keep_rate_available and n_keep > 0 else None
    )

    # Coverage quantiles
    df2 = df[df["score"].notna()].copy()
    if len(df2) > 0:
        try:
            df2["coverage_bin"] = pd.qcut(df2["t_pairs"].rank(method="first"), q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        except Exception:
            df2["coverage_bin"] = "all"
        by = df2.groupby("coverage_bin", dropna=False, observed=False).agg(
            n=("paper_id", "count"),
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            mean_t_pairs=("t_pairs", "mean"),
            median_t_pairs=("t_pairs", "median"),
        )
    else:
        by = pd.DataFrame()

    # Figures
    fig_hist = out_dir / "fig_t_pairs_hist.png"
    fig_scatter = out_dir / "fig_score_vs_t_pairs.png"
    fig_keep = out_dir / "fig_score_vs_keep_rate.png"

    # Histogram of t_pairs
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(df["log_t_pairs"].dropna(), bins=30, color="#2E6F9E", edgecolor="white")
    ax.set_title("Coverage proxy: extracted (coef,se) pairs per paper")
    ax.set_xlabel("log(1 + t_pairs_seen)")
    ax.set_ylabel("Paper count")
    fig.tight_layout()
    fig.savefig(fig_hist, dpi=180)
    plt.close(fig)

    # Scatter score vs log coverage
    df_sc = df[df["score"].notna()].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.scatter(df_sc["log_t_pairs"], df_sc["score"], s=10, alpha=0.45, color="#2E6F9E", edgecolor="none")
    ax.set_title("Offline risk score vs extraction coverage")
    ax.set_xlabel("log(1 + t_pairs_seen)")
    ax.set_ylabel("offline_risk_score")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_scatter, dpi=180)
    plt.close(fig)

    if keep_rate_available and df["keep_rate"].notna().any():
        df_sc2 = df[df["score"].notna() & df["keep_rate"].notna()].copy()
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.scatter(df_sc2["keep_rate"], df_sc2["score"], s=10, alpha=0.45, color="#2E6F9E", edgecolor="none")
        ax.set_title("Offline risk score vs extraction keep-rate")
        ax.set_xlabel("t_pairs_keep_rate (kept/raw)")
        ax.set_ylabel("offline_risk_score")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(fig_keep, dpi=180)
        plt.close(fig)

    # Markdown
    out_md = out_dir / "coverage_diagnostics.md"
    lines: list[str] = []
    lines.append("# Coverage diagnostics (prototype)")
    lines.append("")
    lines.append(f"> Generated: {_now_iso()}")
    lines.append("")
    lines.append("## Snapshot")
    lines.append("")
    lines.append(f"- Papers: {n}")
    lines.append(f"- Papers with score: {n_score}")
    lines.append(f"- Papers with t_pairs_seen==0: {n_pairs0} ({frac_pairs0:.3f})")
    if keep_rate_available:
        lines.append(f"- Papers with keep-rate: {n_keep}")
        if frac_keep_lt_090 is not None:
            lines.append(f"- Share keep-rate < 0.90: {frac_keep_lt_090:.3f}")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append(f"![]({fig_hist.name})")
    lines.append("")
    lines.append(f"![]({fig_scatter.name})")
    if keep_rate_available and fig_keep.exists():
        lines.append("")
        lines.append(f"![]({fig_keep.name})")
    lines.append("")

    if not by.empty:
        lines.append("## Score by coverage quintile")
        lines.append("")
        lines.append("| bin | n | mean_score | median_score | mean_t_pairs | median_t_pairs |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for idx, row in by.reset_index().iterrows():
            b = str(row["coverage_bin"])
            lines.append(
                f"| {b} | {int(row['n'])} | {float(row['mean_score']):.2f} | {float(row['median_score']):.2f} | {float(row['mean_t_pairs']):.1f} | {float(row['median_t_pairs']):.1f} |"
            )
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{_now_iso()}] wrote: {out_md}")
    print(f"[{_now_iso()}] wrote: {fig_hist}")
    print(f"[{_now_iso()}] wrote: {fig_scatter}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
