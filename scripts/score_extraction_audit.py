#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
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


def _safe_int(x: Any) -> int | None:
    try:
        if isinstance(x, bool):
            return None
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            return int(x)
        if isinstance(x, str) and x.strip():
            return int(float(x.strip()))
    except Exception:
        return None
    return None


def _norm_paren_mode(x: Any) -> str | None:
    if not isinstance(x, str):
        return None
    v = x.strip().lower()
    if v in {"se", "stderr", "standard_error", "standard errors", "standard error"}:
        return "se"
    if v in {"t", "tstat", "t_stat", "t-stat", "t-statistic", "t statistics", "t-statistics"}:
        return "t"
    return None


def _summ(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    values = [float(v) for v in values]
    values_sorted = sorted(values)
    return {
        "n": float(len(values_sorted)),
        "mean": float(statistics.mean(values_sorted)),
        "median": float(statistics.median(values_sorted)),
        "p90": float(values_sorted[int(math.floor(0.9 * (len(values_sorted) - 1)))]),
        "max": float(values_sorted[-1]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Score extraction audit labels against extracted values.")
    ap.add_argument("--tasks-dir", default="annotations/extraction_tasks", help="Directory with task JSON files.")
    ap.add_argument("--labels-dir", default="annotations/extraction_labels", help="Directory with completed label JSON files.")
    ap.add_argument("--out", default="annotations/extraction_audit_report.json", help="Output JSON report path.")
    args = ap.parse_args()

    tasks_dir = Path(args.tasks_dir)
    labels_dir = Path(args.labels_dir)
    out_path = Path(args.out)

    if not tasks_dir.exists():
        raise SystemExit(f"Missing tasks dir: {tasks_dir}")
    if not labels_dir.exists():
        raise SystemExit(f"Missing labels dir: {labels_dir}")

    abs_err_coef: list[float] = []
    abs_err_paren: list[float] = []
    abs_err_se: list[float] = []
    abs_err_t: list[float] = []
    abs_err_abs_t: list[float] = []
    abs_err_p2: list[float] = []
    stars_match = 0
    stars_total = 0
    items_labeled = 0
    paren_mode_match = 0
    paren_mode_total = 0

    per_paper: dict[str, Any] = {}

    for task_path in sorted(tasks_dir.glob("*.json")):
        paper_id = task_path.stem
        label_path = labels_dir / f"{paper_id}.json"
        if not label_path.exists():
            continue
        try:
            task = json.loads(task_path.read_text(encoding="utf-8"))
            label = json.loads(label_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        task_items = {str(it.get("item_id")): it for it in (task.get("items") or []) if isinstance(it, dict)}
        label_items = [it for it in (label.get("items") or []) if isinstance(it, dict)]
        paper_coef_err: list[float] = []
        paper_paren_err: list[float] = []
        paper_se_err: list[float] = []
        paper_t_err: list[float] = []
        paper_abs_t_err: list[float] = []
        paper_p2_err: list[float] = []
        paper_items = 0
        paper_stars_match = 0
        paper_stars_total = 0
        paper_paren_mode_match = 0
        paper_paren_mode_total = 0

        for li in label_items:
            item_id = str(li.get("item_id") or "")
            if item_id not in task_items:
                continue
            ti = task_items[item_id]
            obs_coef = _safe_float(li.get("observed_coef"))
            obs_se = _safe_float(li.get("observed_se"))
            obs_t = _safe_float(li.get("observed_t"))
            obs_paren = _safe_float(li.get("observed_paren"))
            obs_paren_mode = _norm_paren_mode(li.get("observed_paren_mode"))

            if obs_paren is None:
                if obs_se is not None:
                    obs_paren = obs_se
                    obs_paren_mode = obs_paren_mode or "se"
                elif obs_t is not None:
                    obs_paren = obs_t
                    obs_paren_mode = obs_paren_mode or "t"

            if obs_paren_mode is None:
                if obs_se is not None:
                    obs_paren_mode = "se"
                elif obs_t is not None:
                    obs_paren_mode = "t"

            if obs_coef is None and obs_paren is None:
                continue

            ext_coef = _safe_float(ti.get("coef_extracted"))
            ext_paren = _safe_float(ti.get("paren_extracted"))
            if ext_paren is None:
                ext_paren = _safe_float(ti.get("se_extracted"))
            if ext_paren is None:
                ext_paren = _safe_float(ti.get("t_extracted"))

            ext_abs_t = _safe_float(ti.get("abs_t_extracted"))
            ext_t = _safe_float(ti.get("t_extracted"))
            if ext_abs_t is None and ext_t is not None:
                ext_abs_t = abs(ext_t)

            ext_p2 = _safe_float(ti.get("p_approx_2s_extracted"))
            if ext_p2 is None and ext_abs_t is not None:
                ext_p2 = float(math.erfc(float(ext_abs_t) / math.sqrt(2.0)))

            ext_paren_mode_assumed = _norm_paren_mode(ti.get("paren_mode_assumed"))

            if obs_paren_mode is not None and ext_paren_mode_assumed is not None:
                paren_mode_total += 1
                paper_paren_mode_total += 1
                if obs_paren_mode == ext_paren_mode_assumed:
                    paren_mode_match += 1
                    paper_paren_mode_match += 1

            if obs_coef is not None and ext_coef is not None:
                e = abs(obs_coef - ext_coef)
                abs_err_coef.append(e)
                paper_coef_err.append(e)

            if obs_paren is not None and ext_paren is not None:
                e = abs(obs_paren - ext_paren)
                abs_err_paren.append(e)
                paper_paren_err.append(e)
                if obs_paren_mode == "se":
                    abs_err_se.append(e)
                    paper_se_err.append(e)
                if obs_paren_mode == "t":
                    abs_err_t.append(e)
                    paper_t_err.append(e)

            # Derived |t| and p from observed values (when possible)
            obs_abs_t = None
            if obs_paren_mode == "t" and obs_paren is not None:
                obs_abs_t = abs(float(obs_paren))
            elif obs_paren_mode == "se" and obs_coef is not None and obs_paren not in (None, 0.0):
                try:
                    obs_abs_t = abs(float(obs_coef) / float(obs_paren))
                except Exception:
                    obs_abs_t = None
            if obs_abs_t is not None and ext_abs_t is not None:
                e = abs(float(obs_abs_t) - float(ext_abs_t))
                abs_err_abs_t.append(e)
                paper_abs_t_err.append(e)
            if obs_abs_t is not None and ext_p2 is not None:
                obs_p2 = float(math.erfc(float(obs_abs_t) / math.sqrt(2.0)))
                e = abs(float(obs_p2) - float(ext_p2))
                abs_err_p2.append(e)
                paper_p2_err.append(e)

            ext_stars = _safe_int(ti.get("stars_extracted"))
            obs_stars = _safe_int(li.get("observed_stars"))
            if ext_stars is not None and obs_stars is not None:
                paper_stars_total += 1
                stars_total += 1
                if ext_stars == obs_stars:
                    paper_stars_match += 1
                    stars_match += 1

            paper_items += 1
            items_labeled += 1

        if paper_items:
            per_paper[paper_id] = {
                "items_labeled": paper_items,
                "coef_abs_error": _summ(paper_coef_err),
                "paren_abs_error": _summ(paper_paren_err),
                "se_abs_error": _summ(paper_se_err),
                "t_abs_error": _summ(paper_t_err),
                "abs_t_abs_error": _summ(paper_abs_t_err),
                "p_approx_2s_abs_error": _summ(paper_p2_err),
                "paren_mode_match_rate": (paper_paren_mode_match / paper_paren_mode_total) if paper_paren_mode_total else None,
                "stars_match_rate": (paper_stars_match / paper_stars_total) if paper_stars_total else None,
            }

    report = {
        "generated_at": _now_iso(),
        "items_labeled": items_labeled,
        "coef_abs_error": _summ(abs_err_coef),
        "paren_abs_error": _summ(abs_err_paren),
        "se_abs_error": _summ(abs_err_se),
        "t_abs_error": _summ(abs_err_t),
        "abs_t_abs_error": _summ(abs_err_abs_t),
        "p_approx_2s_abs_error": _summ(abs_err_p2),
        "paren_mode_match_rate": (paren_mode_match / paren_mode_total) if paren_mode_total else None,
        "stars_match_rate": (stars_match / stars_total) if stars_total else None,
        "per_paper": per_paper,
    }

    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{_now_iso()}] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
