#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


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


def _bin5(y: int) -> str:
    start = (y // 5) * 5
    return f"{start}-{start+4}"


def _bootstrap_ci(values: list[float], *, seed: int, B: int = 2000, alpha: float = 0.05) -> tuple[float, float] | None:
    if not values:
        return None
    rng = random.Random(int(seed))
    n = len(values)
    if n <= 1:
        return None
    means = []
    for _ in range(int(B)):
        s = 0.0
        for _i in range(n):
            s += values[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * B)]
    hi = means[int((1 - alpha / 2) * B) - 1]
    return float(lo), float(hi)


def _bootstrap_ci_pooled(
    left: list[int], right: list[int], total: list[int], *, direction: str, seed: int, B: int = 2000, alpha: float = 0.05
) -> tuple[float, float] | None:
    if not left or not right or not total:
        return None
    if not (len(left) == len(right) == len(total)):
        return None
    rng = random.Random(int(seed))
    n = len(total)
    if n <= 1:
        return None
    stats = []
    for _ in range(int(B)):
        L = 0
        R = 0
        T = 0
        for _i in range(n):
            j = rng.randrange(n)
            L += int(left[j])
            R += int(right[j])
            T += int(total[j])
        if T <= 0:
            continue
        if direction == "right":
            stats.append(float((R - L) / T))
        else:
            stats.append(float((L - R) / T))
    if not stats:
        return None
    stats.sort()
    lo = stats[int((alpha / 2) * len(stats))]
    hi = stats[int((1 - alpha / 2) * len(stats)) - 1]
    return float(lo), float(hi)


def _ttest_mean(values: list[float]) -> tuple[float, float] | None:
    n = len(values)
    if n <= 1:
        return None
    m = sum(values) / n
    s2 = sum((x - m) ** 2 for x in values) / (n - 1)
    if s2 <= 0:
        return None
    se = math.sqrt(s2 / n)
    t = m / se
    # normal approx (n typically large enough here)
    p = float(math.erfc(abs(t) / math.sqrt(2.0)))
    return float(t), float(p)


def _compute_metric(df: pd.DataFrame, *, left_col: str, right_col: str, total_col: str, direction: str) -> pd.DataFrame:
    """
    direction:
      - "left":  effect = (left-right)/total
      - "right": effect = (right-left)/total
    """
    out = df.copy()
    out["L"] = pd.to_numeric(out[left_col], errors="coerce").fillna(0).astype(int)
    out["R"] = pd.to_numeric(out[right_col], errors="coerce").fillna(0).astype(int)
    out["T"] = pd.to_numeric(out[total_col], errors="coerce").fillna(0).astype(int)
    out = out[out["T"] > 0].copy()
    if direction == "right":
        out["effect"] = (out["R"] - out["L"]) / out["T"]
    else:
        out["effect"] = (out["L"] - out["R"]) / out["T"]
    out["z"] = (out["effect"] * (out["T"] ** 0.5)).astype(float)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Cluster-aware inference for caliper bunching metrics (paper as cluster).")
    ap.add_argument("--panel", default="analysis/paper_panel.csv", help="Panel CSV (must include *_left/right/total columns).")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: panel parent).")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--bootstrap-B", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--year-bin", type=int, default=5, help="Bin width in years (default=5).")
    args = ap.parse_args()

    panel_path = Path(args.panel)
    if not panel_path.exists():
        raise SystemExit(f"Missing: {panel_path}")
    out_dir = Path(args.out_dir) if args.out_dir else panel_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(panel_path)
    if "paper_id" not in df.columns:
        raise SystemExit("panel missing paper_id")

    metrics = [
        {
            "name": "p_from_t_caliper_0_05",
            "left": "p_from_t_near_005_left",
            "right": "p_from_t_near_005_right",
            "total": "p_from_t_near_005_total",
            "direction": "left",
            "title": "p≈0.05 caliper (from (coef,se)→t→p)",
        },
        {
            "name": "p_from_t_caliper_0_10",
            "left": "p_from_t_near_010_left",
            "right": "p_from_t_near_010_right",
            "total": "p_from_t_near_010_total",
            "direction": "left",
            "title": "p≈0.10 caliper (from (coef,se)→t→p)",
        },
        {
            "name": "t_caliper_1_96",
            "left": "t_near_196_left",
            "right": "t_near_196_right",
            "total": "t_near_196_total",
            "direction": "right",
            "title": "|t|≈1.96 caliper (more mass just above threshold)",
        },
        {
            "name": "t_caliper_1_645",
            "left": "t_near_1645_left",
            "right": "t_near_1645_right",
            "total": "t_near_1645_total",
            "direction": "right",
            "title": "|t|≈1.645 caliper (more mass just above threshold)",
        },
    ]

    rows_summary: list[dict[str, Any]] = []
    for m in metrics:
        for c in [m["left"], m["right"], m["total"]]:
            if c not in df.columns:
                raise SystemExit(f"panel missing required column: {c}")

        sub = _compute_metric(df, left_col=m["left"], right_col=m["right"], total_col=m["total"], direction=m["direction"])
        effects = sub["effect"].astype(float).tolist()
        Ls = sub["L"].astype(int).tolist()
        Rs = sub["R"].astype(int).tolist()
        Ts = sub["T"].astype(int).tolist()

        mean_effect = float(sum(effects) / len(effects)) if effects else None
        pooled_effect = None
        if Ts and sum(Ts) > 0:
            if m["direction"] == "right":
                pooled_effect = float((sum(Rs) - sum(Ls)) / sum(Ts))
            else:
                pooled_effect = float((sum(Ls) - sum(Rs)) / sum(Ts))

        t_p = _ttest_mean(effects)
        ci_mean = _bootstrap_ci(effects, seed=int(args.seed), B=int(args.bootstrap_B), alpha=float(args.alpha))
        ci_pooled = _bootstrap_ci_pooled(
            Ls, Rs, Ts, direction=m["direction"], seed=int(args.seed), B=int(args.bootstrap_B), alpha=float(args.alpha)
        )

        rows_summary.append(
            {
                "metric": m["name"],
                "n_papers": int(len(sub)),
                "mean_effect": mean_effect,
                "mean_t": (t_p[0] if t_p else None),
                "mean_p_normal": (t_p[1] if t_p else None),
                "mean_ci_low": (ci_mean[0] if ci_mean else None),
                "mean_ci_high": (ci_mean[1] if ci_mean else None),
                "pooled_effect": pooled_effect,
                "pooled_ci_low": (ci_pooled[0] if ci_pooled else None),
                "pooled_ci_high": (ci_pooled[1] if ci_pooled else None),
                "sum_left": int(sum(Ls)),
                "sum_right": int(sum(Rs)),
                "sum_total": int(sum(Ts)),
            }
        )

        # Hist figure
        fig_path = out_dir / f"fig_{m['name']}_effect_hist.png"
        if effects:
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            ax.hist(effects, bins=30, color="#2E6F9E", edgecolor="white")
            ax.axvline(0.0, color="black", lw=1, alpha=0.7)
            ax.set_title(f"Per-paper effect distribution: {m['title']}")
            ax.set_xlabel("effect (directional left-right / total)")
            ax.set_ylabel("Paper count (with window mass)")
            fig.tight_layout()
            fig.savefig(fig_path, dpi=180)
            plt.close(fig)

        # Year-bin time series (mean effect)
        if "year" in df.columns:
            years = df[["paper_id", "year"]].copy()
            years["year_i"] = years["year"].apply(_safe_int)
            sub2 = sub.merge(years[["paper_id", "year_i"]], on="paper_id", how="left")
            sub2 = sub2[sub2["year_i"].notna()].copy()
            if len(sub2) >= 15:
                # Bin by width
                width = max(1, int(args.year_bin))
                sub2["year_bin"] = sub2["year_i"].astype(int).apply(lambda y: f"{(y//width)*width}-{(y//width)*width+width-1}")
                by = sub2.groupby("year_bin").agg(n=("paper_id", "count"), mean_effect=("effect", "mean")).reset_index()
                # Simple CI via normal approx on per-paper effects in bin
                cis = []
                for b in by["year_bin"].tolist():
                    vals = sub2[sub2["year_bin"] == b]["effect"].astype(float).tolist()
                    ci = _bootstrap_ci(vals, seed=int(args.seed), B=max(400, int(args.bootstrap_B // 5)), alpha=float(args.alpha))
                    cis.append(ci)
                by["ci_low"] = [(c[0] if c else None) for c in cis]
                by["ci_high"] = [(c[1] if c else None) for c in cis]

                fig_ts = out_dir / f"fig_{m['name']}_by_yearbin.png"
                fig, ax = plt.subplots(figsize=(9.0, 4.2))
                xs = list(range(len(by)))
                ax.plot(xs, by["mean_effect"], color="#2E6F9E", lw=2)
                ax.scatter(xs, by["mean_effect"], color="#2E6F9E", s=18)
                # Error bars (if available)
                yerr_lo = []
                yerr_hi = []
                for _, r in by.iterrows():
                    lo = _safe_float(r.get("ci_low"))
                    hi = _safe_float(r.get("ci_high"))
                    mu = _safe_float(r.get("mean_effect"))
                    if lo is None or hi is None or mu is None:
                        yerr_lo.append(0.0)
                        yerr_hi.append(0.0)
                    else:
                        yerr_lo.append(max(0.0, mu - lo))
                        yerr_hi.append(max(0.0, hi - mu))
                ax.errorbar(xs, by["mean_effect"], yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="black", alpha=0.35, lw=1)
                ax.set_xticks(xs)
                ax.set_xticklabels(by["year_bin"].tolist(), rotation=45, ha="right")
                ax.set_title(f"Mean per-paper effect by year bin: {m['title']}")
                ax.set_xlabel("Year bin")
                ax.set_ylabel("Mean effect (with bootstrap CI)")
                ax.grid(True, alpha=0.2)
                fig.tight_layout()
                fig.savefig(fig_ts, dpi=180)
                plt.close(fig)

    out_csv = out_dir / "bunching_inference_summary.csv"
    pd.DataFrame(rows_summary).to_csv(out_csv, index=False)

    out_md = out_dir / "bunching_inference.md"
    lines: list[str] = []
    lines.append("# Cluster-aware inference (paper as cluster): caliper diagnostics")
    lines.append("")
    lines.append(f"> Generated: {_now_iso()}")
    lines.append("")
    lines.append("## Summary table")
    lines.append("")
    lines.append("| metric | n_papers | mean_effect | mean_t | mean_p | pooled_effect | sum_left | sum_right | sum_total |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows_summary:
        lines.append(
            "| {metric} | {n_papers} | {mean_effect:.4f} | {mean_t:.3f} | {mean_p:.4f} | {pooled_effect:.4f} | {sum_left} | {sum_right} | {sum_total} |".format(
                metric=r["metric"],
                n_papers=int(r["n_papers"]),
                mean_effect=float(r["mean_effect"] or 0.0),
                mean_t=float(r["mean_t"] or 0.0),
                mean_p=float(r["mean_p_normal"] or 1.0),
                pooled_effect=float(r["pooled_effect"] or 0.0),
                sum_left=int(r["sum_left"]),
                sum_right=int(r["sum_right"]),
                sum_total=int(r["sum_total"]),
            )
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `mean_effect` is the *unweighted* mean across papers with at least one test in the caliper window.")
    lines.append("- `pooled_effect` is the *pooled* (sum across papers) effect; CI is best read via the bootstrap outputs in the CSV.")
    lines.append("- These are still *best-effort* because extraction coverage varies by PDF structure; see `coverage_diagnostics.md`.")
    lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{_now_iso()}] wrote: {out_md}")
    print(f"[{_now_iso()}] wrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

