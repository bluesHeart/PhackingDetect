#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd


LABEL_FIELDS = [
    "outcomes_exposure",
    "heterogeneity_exposure",
    "robustness_spec_search_exposure",
    "multiple_testing_correction_reported",
    "pre_registration_or_analysis_plan",
    "borderline_significance_emphasis",
    "selective_reporting_language",
]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _rankdata(values: list[float]) -> list[float]:
    # Average ranks for ties. Returns 1-based ranks.
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def _pearson(x: list[float], y: list[float]) -> float | None:
    if not x or not y or len(x) != len(y):
        return None
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    denx = math.sqrt(sum((x[i] - mx) ** 2 for i in range(n)))
    deny = math.sqrt(sum((y[i] - my) ** 2 for i in range(n)))
    if denx <= 0 or deny <= 0:
        return None
    return float(num / (denx * deny))


def _spearman(x: list[float], y: list[float]) -> float | None:
    if not x or not y or len(x) != len(y):
        return None
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _pearson(rx, ry)


def _cohen_kappa(a: list[str], b: list[str]) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    n = len(a)
    if n == 0:
        return None
    cats = sorted(set(a) | set(b))
    if len(cats) <= 1:
        return None
    po = sum(1 for i in range(n) if a[i] == b[i]) / n
    pa = {c: 0 for c in cats}
    pb = {c: 0 for c in cats}
    for x in a:
        pa[x] += 1
    for x in b:
        pb[x] += 1
    pe = sum((pa[c] / n) * (pb[c] / n) for c in cats)
    if 1.0 - pe <= 1e-12:
        return None
    return float((po - pe) / (1.0 - pe))


def _safe_str(x: Any) -> str | None:
    if isinstance(x, str):
        s = x.strip()
        return s or None
    return None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Score human-audit labels and produce a validation summary report.")
    ap.add_argument("--labels-dir", default="annotations/labels", help="Directory with completed label JSON files.")
    ap.add_argument("--panel", default="analysis/paper_panel.csv", help="Panel CSV with offline scores to join.")
    ap.add_argument("--out-md", default="annotations/audit_report.md", help="Output markdown report path.")
    ap.add_argument("--out-json", default="annotations/audit_report.json", help="Output JSON report path.")
    args = ap.parse_args()

    labels_dir = Path(args.labels_dir)
    panel_path = Path(args.panel)
    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(panel_path) if panel_path.exists() else pd.DataFrame()
    score_by_pid: dict[str, float] = {}
    if not panel.empty and "paper_id" in panel.columns and "offline_risk_score" in panel.columns:
        for r in panel.itertuples(index=False):
            pid = getattr(r, "paper_id", None)
            sc = getattr(r, "offline_risk_score", None)
            if isinstance(pid, str) and isinstance(sc, (int, float)) and sc == sc:
                score_by_pid[pid] = float(sc)

    rows: list[dict[str, Any]] = []
    if labels_dir.exists():
        for p in sorted(labels_dir.glob("*.json")):
            obj = _read_json(p)
            if not isinstance(obj, dict):
                continue
            pid = _safe_str(obj.get("paper_id")) or p.stem
            annotator = _safe_str(obj.get("annotator")) or p.stem
            labels = obj.get("labels") if isinstance(obj.get("labels"), dict) else {}
            overall = obj.get("overall_audit_assessment") if isinstance(obj.get("overall_audit_assessment"), dict) else {}

            rec: dict[str, Any] = {
                "paper_id": pid,
                "annotator": annotator,
                "completed_at": _safe_str(obj.get("completed_at")),
                "overall": _safe_str(overall.get("value")),
                "offline_risk_score": score_by_pid.get(pid),
            }
            for f in LABEL_FIELDS:
                box = labels.get(f) if isinstance(labels, dict) else None
                val = box.get("value") if isinstance(box, dict) else None
                rec[f] = _safe_str(val)
            rows.append(rec)

    df = pd.DataFrame(rows)
    n_files = int(len(df))
    if n_files == 0:
        report = {
            "generated_at": _now_iso(),
            "n_label_files": 0,
            "n_completed": 0,
            "message": f"No label files found in {labels_dir}.",
        }
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        out_md.write_text(
            "\n".join(
                [
                    "# Human-audit report (no labels found)",
                    "",
                    f"> Generated: {_now_iso()}",
                    "",
                    f"- labels-dir: `{labels_dir}`",
                    "",
                    "No label JSON files were found; generate tasks with `scripts/make_audit_tasks.py` and fill in `annotations/labels/`.",
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[{_now_iso()}] wrote: {out_md}")
        print(f"[{_now_iso()}] wrote: {out_json}")
        return 0

    def is_completed_row(r: pd.Series) -> bool:
        if isinstance(r.get("completed_at"), str) and r.get("completed_at").strip():
            return True
        if isinstance(r.get("overall"), str) and r.get("overall").strip():
            return True
        for f in LABEL_FIELDS:
            v = r.get(f)
            if isinstance(v, str) and v.strip():
                return True
        return False

    df["is_completed"] = df.apply(is_completed_row, axis=1)
    n_completed = int(df["is_completed"].sum())
    n_unique_papers = int(df["paper_id"].nunique())
    n_multi_annotated_papers = int((df.groupby("paper_id").size() >= 2).sum())

    # Distributions
    field_counts: dict[str, dict[str, int]] = {}
    for f in ["overall"] + LABEL_FIELDS:
        counts = df.loc[df["is_completed"], f].value_counts(dropna=False).to_dict()
        out: dict[str, int] = {}
        for k, v in counts.items():
            key = str(k) if k == k else "null"
            out[key] = int(v)
        field_counts[f] = out

    # Convergent validity vs offline score (ordinal coding)
    ord_map = {"low": 0.0, "moderate": 1.0, "high": 2.0}
    conv: dict[str, Any] = {}
    completed = df[df["is_completed"]].copy()
    if not completed.empty and "offline_risk_score" in completed.columns:
        # Overall
        xs: list[float] = []
        ys: list[float] = []
        for r in completed.itertuples(index=False):
            sc = getattr(r, "offline_risk_score", None)
            ov = getattr(r, "overall", None)
            if not isinstance(sc, (int, float)) or sc != sc:
                continue
            if not isinstance(ov, str) or ov.strip().lower() not in ord_map:
                continue
            xs.append(float(sc))
            ys.append(float(ord_map[ov.strip().lower()]))
        conv["overall_spearman"] = _spearman(xs, ys) if len(xs) >= 6 else None

        # Means by category
        means: dict[str, float] = {}
        for cat in ["low", "moderate", "high"]:
            vals = [xs[i] for i in range(len(xs)) if ys[i] == ord_map[cat]]
            if vals:
                means[cat] = float(sum(vals) / len(vals))
        conv["overall_mean_score_by_label"] = means
        conv["n_overall_with_score"] = int(len(xs))

    # Inter-rater reliability (pairwise Cohen's kappa per field if possible)
    irr: dict[str, Any] = {"pairwise_kappa": {}}
    if n_multi_annotated_papers > 0:
        annotators = sorted(set(df["annotator"].astype(str).tolist()))
        # Build per-annotator dict: field -> {paper_id: value}
        per_ann: dict[str, dict[str, dict[str, str]]] = {f: {} for f in ["overall"] + LABEL_FIELDS}
        for ann in annotators:
            sub = df[(df["annotator"] == ann) & (df["is_completed"])].copy()
            for f in ["overall"] + LABEL_FIELDS:
                m: dict[str, str] = {}
                for r in sub.itertuples(index=False):
                    pid = getattr(r, "paper_id", None)
                    val = getattr(r, f, None)
                    if isinstance(pid, str) and isinstance(val, str) and val.strip():
                        m[pid] = val.strip()
                per_ann[f][ann] = m

        for f in ["overall"] + LABEL_FIELDS:
            pair_rows: list[dict[str, Any]] = []
            for a1, a2 in itertools.combinations(annotators, 2):
                m1 = per_ann[f].get(a1, {})
                m2 = per_ann[f].get(a2, {})
                common = sorted(set(m1.keys()) & set(m2.keys()))
                if len(common) < 8:
                    continue
                v1 = [m1[pid] for pid in common]
                v2 = [m2[pid] for pid in common]
                kappa = _cohen_kappa(v1, v2)
                pair_rows.append({"annotator_1": a1, "annotator_2": a2, "n_common": len(common), "kappa": kappa})
            irr["pairwise_kappa"][f] = pair_rows

    report = {
        "generated_at": _now_iso(),
        "labels_dir": str(labels_dir).replace("\\", "/"),
        "panel": str(panel_path).replace("\\", "/"),
        "n_label_files": n_files,
        "n_unique_papers": n_unique_papers,
        "n_completed": n_completed,
        "n_multi_annotated_papers": n_multi_annotated_papers,
        "field_counts_completed_only": field_counts,
        "convergent_validity": conv,
        "inter_rater_reliability": irr,
    }
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown report
    lines: list[str] = []
    lines.append("# Human-audit report (prototype)")
    lines.append("")
    lines.append(f"> Generated: {_now_iso()}")
    lines.append("")
    lines.append("## Snapshot")
    lines.append("")
    lines.append(f"- Label files: {n_files}")
    lines.append(f"- Unique papers: {n_unique_papers}")
    lines.append(f"- Completed: {n_completed}")
    lines.append(f"- Multi-annotated papers: {n_multi_annotated_papers}")
    lines.append("")

    if conv:
        lines.append("## Convergent validity (toward JF)")
        lines.append("")
        if conv.get("n_overall_with_score") is not None:
            lines.append(f"- n (overall with offline score): {conv['n_overall_with_score']}")
        if conv.get("overall_spearman") is not None:
            lines.append(f"- Spearman(offline score, overall label): {float(conv['overall_spearman']):.3f}")
        means = conv.get("overall_mean_score_by_label") or {}
        if isinstance(means, dict) and means:
            parts = [f"{k}={float(v):.2f}" for k, v in means.items()]
            lines.append(f"- Mean offline score by overall label: {', '.join(parts)}")
        lines.append("")

    lines.append("## Label distributions (completed only)")
    lines.append("")
    for f in ["overall"] + LABEL_FIELDS:
        lines.append(f"### {f}")
        counts = field_counts.get(f) or {}
        if not counts:
            lines.append("- (no completed labels)")
            lines.append("")
            continue
        # Sort with null last
        items = [(k, int(v)) for k, v in counts.items()]
        items.sort(key=lambda kv: (kv[0] == "null", kv[0]))
        for k, v in items:
            lines.append(f"- {k}: {v}")
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{_now_iso()}] wrote: {out_md}")
    print(f"[{_now_iso()}] wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

