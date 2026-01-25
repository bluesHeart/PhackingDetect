#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _safe_float(x: Any) -> float | None:
    try:
        if isinstance(x, bool):
            return None
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str) and x.strip():
            return float(x.strip())
    except Exception:
        return None
    return None


def _in_window(x: float, *, center: float, delta: float) -> int:
    if (center - delta) <= x <= center:
        return -1
    if center < x <= (center + delta):
        return 1
    return 0


def _normalize_ratio(counts: dict[str, int]) -> float:
    tot = int(counts.get("total") or 0)
    if tot <= 0:
        return 0.0
    left = int(counts.get("left") or 0)
    right = int(counts.get("right") or 0)
    return (left - right) / max(1, tot)


def _caliper_z(left: int, right: int) -> float:
    n = int(left) + int(right)
    if n <= 0:
        return 0.0
    return float((int(left) - int(right)) / math.sqrt(float(n)))


def _cap01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v != v:
        return 0.0
    return max(0.0, min(1.0, v))


def _risk_score_from_features(features: dict[str, Any]) -> tuple[int, dict[str, float]]:
    comp: dict[str, float] = {}
    b05 = features.get("p_from_t_near_0_05") or features.get("near_0_05_p") or {}
    b10 = features.get("p_from_t_near_0_10") or features.get("near_0_10_p") or {}
    l05 = int(b05.get("left") or 0) if isinstance(b05, dict) else 0
    r05 = int(b05.get("right") or 0) if isinstance(b05, dict) else 0
    l10 = int(b10.get("left") or 0) if isinstance(b10, dict) else 0
    r10 = int(b10.get("right") or 0) if isinstance(b10, dict) else 0
    comp["caliper_z_0_05"] = _caliper_z(l05, r05)
    comp["caliper_z_0_10"] = _caliper_z(l10, r10)

    robust = float(features.get("robust_mentions_fulltext") or 0)
    spec = float(features.get("spec_search_terms_fulltext") or 0)
    multi = float(features.get("multiple_testing_terms_fulltext") or 0)
    comp["robust_term_intensity"] = min(1.0, robust / 10.0)
    comp["spec_term_intensity"] = min(1.0, spec / 8.0)
    comp["multiple_testing_exposure"] = min(1.0, multi / 4.0)

    has_correction = 1.0 if bool(features.get("has_multiple_testing_correction")) else 0.0
    comp["has_correction"] = has_correction

    score = 20.0
    score += 12.0 * _cap01(max(0.0, comp["caliper_z_0_05"]) / 3.0)
    score += 6.0 * _cap01(max(0.0, comp["caliper_z_0_10"]) / 3.0)
    score += 12.0 * comp["robust_term_intensity"]
    score += 10.0 * comp["spec_term_intensity"]
    score += 8.0 * comp["multiple_testing_exposure"]
    score -= 12.0 * has_correction
    score = max(0.0, min(100.0, score))
    return int(round(score)), comp


def _auc(scores: list[float], labels: list[int]) -> float | None:
    if not scores or len(scores) != len(labels):
        return None
    pairs = list(zip(scores, labels))
    if not any(l == 0 for _, l in pairs) or not any(l == 1 for _, l in pairs):
        return None
    # Rank-based AUC (handles ties).
    pairs_sorted = sorted(pairs, key=lambda x: x[0])
    ranks: list[float] = [0.0] * len(pairs_sorted)
    i = 0
    while i < len(pairs_sorted):
        j = i
        while j < len(pairs_sorted) and pairs_sorted[j][0] == pairs_sorted[i][0]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)  # 1-based ranks
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    n_pos = sum(1 for _, l in pairs_sorted if l == 1)
    n_neg = len(pairs_sorted) - n_pos
    rank_sum_pos = sum(ranks[k] for k, (_, l) in enumerate(pairs_sorted) if l == 1)
    # Mann–Whitney U
    u = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u / (n_pos * n_neg))


def _read_features_csv(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            pid = (row.get("paper_id") or "").strip()
            if not pid:
                continue
            # Keep fields as strings; we'll only read a small subset.
            out[pid] = dict(row)
    return out


def _parse_counts(s: Any) -> dict[str, int] | None:
    if isinstance(s, dict):
        try:
            return {"left": int(s.get("left") or 0), "right": int(s.get("right") or 0), "total": int(s.get("total") or 0)}
        except Exception:
            return None
    if isinstance(s, str) and s.strip().startswith("{"):
        try:
            d = json.loads(s)
        except Exception:
            return None
        if isinstance(d, dict):
            try:
                return {"left": int(d.get("left") or 0), "right": int(d.get("right") or 0), "total": int(d.get("total") or 0)}
            except Exception:
                return None
    return None


def _caliper_counts(values: list[float], *, center: float, delta: float) -> dict[str, int]:
    out = {"left": 0, "right": 0, "total": 0}
    for v in values:
        w = _in_window(float(v), center=center, delta=delta)
        if w == 0:
            continue
        out["total"] += 1
        if w < 0:
            out["left"] += 1
        else:
            out["right"] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Half-synthetic validation: inject threshold bunching and test score sensitivity.")
    ap.add_argument("--corpus-dir", default="corpus", help="Corpus directory with tests/ and features.csv.")
    ap.add_argument("--out-dir", default="analysis", help="Output directory.")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument(
        "--manipulation-rate",
        type=float,
        default=0.5,
        help="Share of tests in (0.055,0.10] to move into [0.045,0.05].",
    )
    ap.add_argument("--max-papers", type=int, default=400, help="Cap number of papers for runtime.")
    args = ap.parse_args()

    corpus_dir = Path(args.corpus_dir)
    tests_dir = corpus_dir / "tests"
    features_csv = corpus_dir / "features.csv"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats_by_id = _read_features_csv(features_csv)
    rng = random.Random(int(args.seed))

    rows_out: list[dict[str, Any]] = []
    scores: list[float] = []
    labels: list[int] = []

    test_files = sorted(tests_dir.glob("*.jsonl"))[: int(args.max_papers)]
    for tf in test_files:
        paper_id = tf.stem
        feat = feats_by_id.get(paper_id)
        if not feat:
            continue

        # Parse per-test p-values from extracted (coef,se) pairs.
        p2s: list[float] = []
        for line in tf.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            p2 = _safe_float(rec.get("p_approx_2s"))
            if p2 is None:
                at = _safe_float(rec.get("abs_t"))
                if at is not None:
                    p2 = float(math.erfc(float(at) / math.sqrt(2.0)))
            if p2 is None:
                continue
            if 0.0 <= p2 <= 1.0:
                p2s.append(float(p2))
        if not p2s:
            continue

        b05 = _caliper_counts(p2s, center=0.05, delta=0.005)
        b10 = _caliper_counts(p2s, center=0.10, delta=0.005)

        base_feat = {
            "p_from_t_near_0_05": b05,
            "p_from_t_near_0_10": b10,
            "robust_mentions_fulltext": _safe_float(feat.get("robust_mentions_fulltext")) or 0.0,
            "spec_search_terms_fulltext": _safe_float(feat.get("spec_search_terms_fulltext")) or 0.0,
            "multiple_testing_terms_fulltext": _safe_float(feat.get("multiple_testing_terms_fulltext")) or 0.0,
            "has_multiple_testing_correction": str(feat.get("has_multiple_testing_correction") or "").strip().lower() in {"true", "1", "yes"},
        }
        base_score, _ = _risk_score_from_features(base_feat)

        # Now manipulate p-values within the 0.05 caliper window.
        # Half-synthetic mechanism: turn marginally insignificant tests into marginally significant ones.
        # Move p in (0.055, 0.10] into [0.045, 0.05], increasing bunching below 0.05.
        pool = sum(1 for p in p2s if 0.055 < p <= 0.10)
        moved = 0
        if pool > 0 and float(args.manipulation_rate) > 0:
            moved = int(math.ceil(pool * float(args.manipulation_rate)))
            moved = max(1, min(pool, moved))

        hacked_b05 = {
            "left": int(b05.get("left") or 0) + moved,
            "right": int(b05.get("right") or 0),
            "total": int(b05.get("total") or 0) + moved,
        }
        hacked_feat = dict(base_feat)
        hacked_feat["p_from_t_near_0_05"] = hacked_b05
        hacked_score, _ = _risk_score_from_features(hacked_feat)

        rows_out.append(
            {
                "paper_id": paper_id,
                "base_score": base_score,
                "hacked_score": hacked_score,
                "delta_score": hacked_score - base_score,
                "b05_left": int(b05.get("left") or 0),
                "b05_right": int(b05.get("right") or 0),
                "moved_into_left_bin": moved,
                "eligible_pool_p_in_0_055_0_10": pool,
            }
        )

        scores.extend([float(base_score), float(hacked_score)])
        labels.extend([0, 1])

    # Write CSV
    out_csv = out_dir / "half_synthetic_experiment.csv"
    if rows_out:
        fields = list(rows_out[0].keys())
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows_out:
                w.writerow(r)

    auc = _auc(scores, labels)
    deltas = [float(r["delta_score"]) for r in rows_out if isinstance(r.get("delta_score"), (int, float))]
    mean_delta = sum(deltas) / len(deltas) if deltas else None

    out_md = out_dir / "half_synthetic_experiment.md"
    lines: list[str] = []
    lines.append("# Half-synthetic validation: injected threshold bunching")
    lines.append("")
    lines.append(f"- Generated: {_now_iso()}")
    lines.append(f"- Papers: {len(rows_out)}")
    lines.append(f"- Manipulation rate (from (0.055,0.10] → [0.045,0.05]): {float(args.manipulation_rate):.2f}")
    lines.append(f"- AUC (base vs hacked): {auc:.3f}" if auc is not None else "- AUC: n/a")
    lines.append(f"- Mean Δ score: {mean_delta:.2f}" if mean_delta is not None else "- Mean Δ score: n/a")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "This is a *sanity-check* experiment: it mechanically injects caliper-window bunching and verifies that the "
        "paper-level score responds in the correct direction. It is not a substitute for ground-truth or replication-based validation."
    )
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"[{_now_iso()}] wrote: {out_csv}")
    print(f"[{_now_iso()}] wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
