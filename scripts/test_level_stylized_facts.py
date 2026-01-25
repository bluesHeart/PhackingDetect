#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


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


def _two_sided_binom_pvalue(k: int, n: int, *, max_exact_n: int = 400) -> float | None:
    if n <= 0:
        return None
    if k < 0 or k > n:
        return None
    if n > max_exact_n:
        z = (2.0 * float(k) - float(n)) / math.sqrt(float(n))
        return float(math.erfc(abs(z) / math.sqrt(2.0)))
    center = float(n) / 2.0
    dist = abs(float(k) - center)
    base = 0.5**n
    p = 0.0
    for x in range(n + 1):
        if abs(float(x) - center) + 1e-12 < dist:
            continue
        p += math.comb(n, x) * base
    return float(min(1.0, max(0.0, p)))


def _caliper_z(left: int, right: int, *, direction: str) -> float:
    n = int(left) + int(right)
    if n <= 0:
        return 0.0
    if direction == "right":
        return float((int(right) - int(left)) / math.sqrt(float(n)))
    return float((int(left) - int(right)) / math.sqrt(float(n)))


def _caliper_p(left: int, right: int) -> float | None:
    n = int(left) + int(right)
    if n <= 0:
        return None
    return _two_sided_binom_pvalue(int(left), n)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute test-level stylized facts from corpus/tests/*.jsonl.")
    ap.add_argument("--corpus-dir", default="corpus", help="Corpus directory with tests/ and manifest.csv.")
    ap.add_argument("--out-dir", default="analysis", help="Output directory for md + figures.")
    ap.add_argument("--max-tests", type=int, default=None, help="Optional cap on number of extracted tests (for speed).")
    ap.add_argument("--p-hist-max", type=float, default=0.10, help="Max p-value for histogram range.")
    ap.add_argument("--p-hist-bins", type=int, default=100, help="Number of bins for p histogram (0..p-hist-max).")
    args = ap.parse_args()

    corpus_dir = Path(args.corpus_dir)
    tests_dir = corpus_dir / "tests"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_files = sorted(tests_dir.glob("*.jsonl"))
    if not test_files:
        raise SystemExit(f"No test jsonl files found under: {tests_dir}")

    p_max = float(args.p_hist_max)
    bins = int(args.p_hist_bins)
    if p_max <= 0 or bins <= 0:
        raise SystemExit("Invalid histogram args.")

    hist = [0] * bins
    hist_edges = [p_max * i / bins for i in range(bins + 1)]

    n_tests = 0
    n_p_valid = 0
    n_p_le_005 = 0
    n_p_le_010 = 0

    # Calipers
    p05_left = 0
    p05_right = 0
    p10_left = 0
    p10_right = 0
    t196_left = 0
    t196_right = 0
    t1645_left = 0
    t1645_right = 0

    for tf in test_files:
        for line in tf.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue

            at = _safe_float(rec.get("abs_t"))
            if at is not None:
                # |t| calipers (delta=0.20 to match extract script)
                if 1.96 - 0.20 <= at <= 1.96:
                    t196_left += 1
                elif 1.96 < at <= 1.96 + 0.20:
                    t196_right += 1
                if 1.645 - 0.20 <= at <= 1.645:
                    t1645_left += 1
                elif 1.645 < at <= 1.645 + 0.20:
                    t1645_right += 1

            p2 = _safe_float(rec.get("p_approx_2s"))
            if p2 is None and at is not None:
                p2 = float(math.erfc(float(at) / math.sqrt(2.0)))
            if p2 is None:
                continue
            if not (0.0 <= p2 <= 1.0):
                continue

            n_tests += 1
            n_p_valid += 1
            if p2 <= 0.05:
                n_p_le_005 += 1
            if p2 <= 0.10:
                n_p_le_010 += 1

            # Histogram
            if 0.0 <= p2 <= p_max:
                bi = min(bins - 1, int((p2 / p_max) * bins))
                hist[bi] += 1

            # p calipers (delta=0.005)
            if 0.045 <= p2 <= 0.05:
                p05_left += 1
            elif 0.05 < p2 <= 0.055:
                p05_right += 1
            if 0.095 <= p2 <= 0.10:
                p10_left += 1
            elif 0.10 < p2 <= 0.105:
                p10_right += 1

            if args.max_tests is not None and n_tests >= int(args.max_tests):
                break
        if args.max_tests is not None and n_tests >= int(args.max_tests):
            break

    # Figures
    fig_hist = out_dir / "fig_test_p_hist.png"
    fig_zoom = out_dir / "fig_test_p_zoom_0_04_0_06.png"

    if sum(hist) > 0:
        xs = [(hist_edges[i] + hist_edges[i + 1]) / 2.0 for i in range(bins)]
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.bar(xs, hist, width=p_max / bins, color="#2E6F9E", edgecolor="white")
        ax.set_title(f"Test-level p-value histogram (0–{p_max:.2f}) from extracted (coef,se)")
        ax.set_xlabel("Two-sided p-value (approx from |t|)")
        ax.set_ylabel("Extracted test count")
        fig.tight_layout()
        fig.savefig(fig_hist, dpi=180)
        plt.close(fig)

    # Zoom around 0.05
    zoom_lo, zoom_hi = 0.04, 0.06
    zoom_bins = int((zoom_hi - zoom_lo) / 0.0005)
    zoom_bins = max(20, min(200, zoom_bins))
    zoom_hist = [0] * zoom_bins
    zoom_edges = [zoom_lo + (zoom_hi - zoom_lo) * i / zoom_bins for i in range(zoom_bins + 1)]

    # Re-scan (still fast) for zoom histogram only
    n_zoom = 0
    for tf in test_files:
        for line in tf.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            at = _safe_float(rec.get("abs_t"))
            p2 = _safe_float(rec.get("p_approx_2s"))
            if p2 is None and at is not None:
                p2 = float(math.erfc(float(at) / math.sqrt(2.0)))
            if p2 is None or not (zoom_lo <= p2 <= zoom_hi):
                continue
            bi = min(zoom_bins - 1, int(((p2 - zoom_lo) / (zoom_hi - zoom_lo)) * zoom_bins))
            zoom_hist[bi] += 1
            n_zoom += 1
            if args.max_tests is not None and n_zoom >= int(args.max_tests):
                break
        if args.max_tests is not None and n_zoom >= int(args.max_tests):
            break

    if sum(zoom_hist) > 0:
        xs = [(zoom_edges[i] + zoom_edges[i + 1]) / 2.0 for i in range(zoom_bins)]
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.bar(xs, zoom_hist, width=(zoom_hi - zoom_lo) / zoom_bins, color="#2E6F9E", edgecolor="white")
        ax.axvline(0.05, color="black", lw=1, alpha=0.7)
        ax.set_title("Zoom: p-values around 0.05")
        ax.set_xlabel("Two-sided p-value (approx)")
        ax.set_ylabel("Extracted test count")
        fig.tight_layout()
        fig.savefig(fig_zoom, dpi=180)
        plt.close(fig)

    # Stats
    p05_z = _caliper_z(p05_left, p05_right, direction="left")
    p05_p = _caliper_p(p05_left, p05_right)
    p10_z = _caliper_z(p10_left, p10_right, direction="left")
    p10_p = _caliper_p(p10_left, p10_right)
    t196_z = _caliper_z(t196_left, t196_right, direction="right")
    t196_p = _caliper_p(t196_left, t196_right)
    t1645_z = _caliper_z(t1645_left, t1645_right, direction="right")
    t1645_p = _caliper_p(t1645_left, t1645_right)

    out_md = out_dir / "test_level_stylized_facts.md"
    lines: list[str] = []
    lines.append("# Test-level stylized facts (prototype)")
    lines.append("")
    lines.append(f"> Generated: {_now_iso()}")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(f"- Test files: {len(test_files)}")
    lines.append(f"- Extracted tests (with p): {n_p_valid}")
    lines.append(f"- Share p<=0.05: {n_p_le_005}/{n_p_valid} ({(n_p_le_005/max(1,n_p_valid)):.3f})")
    lines.append(f"- Share p<=0.10: {n_p_le_010}/{n_p_valid} ({(n_p_le_010/max(1,n_p_valid)):.3f})")
    lines.append("")
    lines.append("## Caliper diagnostics (test-level)")
    lines.append("")
    lines.append(f"- p≈0.05 caliper: left={p05_left} right={p05_right} z={p05_z:.3f} p2={p05_p:.4f}" if p05_p is not None else f"- p≈0.05 caliper: left={p05_left} right={p05_right} z={p05_z:.3f} p2=n/a")
    lines.append(f"- p≈0.10 caliper: left={p10_left} right={p10_right} z={p10_z:.3f} p2={p10_p:.4f}" if p10_p is not None else f"- p≈0.10 caliper: left={p10_left} right={p10_right} z={p10_z:.3f} p2=n/a")
    lines.append(f"- |t|≈1.96 caliper: left={t196_left} right={t196_right} z={t196_z:.3f} p2={t196_p:.4f}" if t196_p is not None else f"- |t|≈1.96 caliper: left={t196_left} right={t196_right} z={t196_z:.3f} p2=n/a")
    lines.append(f"- |t|≈1.645 caliper: left={t1645_left} right={t1645_right} z={t1645_z:.3f} p2={t1645_p:.4f}" if t1645_p is not None else f"- |t|≈1.645 caliper: left={t1645_left} right={t1645_right} z={t1645_z:.3f} p2=n/a")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    if fig_hist.exists():
        lines.append(f"![]({fig_hist.name})")
        lines.append("")
    if fig_zoom.exists():
        lines.append(f"![]({fig_zoom.name})")
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{_now_iso()}] wrote: {out_md}")
    if fig_hist.exists():
        print(f"[{_now_iso()}] wrote: {fig_hist}")
    if fig_zoom.exists():
        print(f"[{_now_iso()}] wrote: {fig_zoom}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

