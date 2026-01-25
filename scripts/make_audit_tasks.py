#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import pandas as pd


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _parse_json_cell(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return v
    return v


def _empty_labels() -> dict[str, Any]:
    def box() -> dict[str, Any]:
        return {"value": None, "evidence": []}

    return {
        "outcomes_exposure": box(),
        "heterogeneity_exposure": box(),
        "robustness_spec_search_exposure": box(),
        "multiple_testing_correction_reported": box(),
        "pre_registration_or_analysis_plan": box(),
        "borderline_significance_emphasis": box(),
        "selective_reporting_language": box(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Create human-audit annotation tasks from a corpus.")
    ap.add_argument("--corpus-dir", default="corpus")
    ap.add_argument("--n", type=int, default=200, help="Number of papers to sample for human audit.")
    ap.add_argument("--seed", type=int, default=123, help="Random seed.")
    ap.add_argument(
        "--selection",
        choices=["random", "top_risk", "stratified"],
        default="random",
        help="Sampling strategy. 'stratified' bins by offline_risk_score quantiles.",
    )
    ap.add_argument("--tasks-dir", default="annotations/tasks", help="Output tasks directory.")
    ap.add_argument("--labels-dir", default="annotations/labels", help="Output labels directory (empty templates).")
    args = ap.parse_args()

    corpus_dir = Path(args.corpus_dir)
    manifest_csv = corpus_dir / "manifest.csv"
    features_csv = corpus_dir / "features.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(manifest_csv)
    if not features_csv.exists():
        raise FileNotFoundError(features_csv)

    m = pd.read_csv(manifest_csv)
    f = pd.read_csv(features_csv)
    f["ssrn_id"] = f["paper_id"].astype(str).str.replace("ssrn_", "", regex=False)
    if "ssrn_id" in m.columns:
        m["ssrn_id"] = m["ssrn_id"].astype(str)
    df = f.merge(m, on="ssrn_id", how="left", suffixes=("", "_m"))

    # Prefer downloaded/exists
    if "status" in df.columns:
        st = df["status"].astype(str)
        df = df[st.str.startswith("downloaded") | st.str.startswith("exists")].copy()

    # Parse candidate pages JSON
    df["candidate_pages_parsed"] = df["candidate_pages"].apply(_parse_json_cell) if "candidate_pages" in df.columns else [[]] * len(df)

    # Sample
    random.seed(int(args.seed))
    if df.empty:
        raise RuntimeError("No papers available to sample.")
    n = min(int(args.n), int(len(df)))
    if n <= 0:
        raise RuntimeError("n must be positive.")

    if str(args.selection) == "top_risk" and "offline_risk_score" in df.columns:
        df2 = df.copy()
        df2["offline_risk_score"] = pd.to_numeric(df2["offline_risk_score"], errors="coerce")
        df2 = df2.sort_values("offline_risk_score", ascending=False)
        sampled = df2["paper_id"].astype(str).head(n).tolist()
    elif str(args.selection) == "stratified" and "offline_risk_score" in df.columns:
        df2 = df.copy()
        df2["offline_risk_score"] = pd.to_numeric(df2["offline_risk_score"], errors="coerce")
        df2 = df2.dropna(subset=["offline_risk_score"]).copy()
        if len(df2) < n:
            sampled = df2["paper_id"].astype(str).tolist()
            random.shuffle(sampled)
            sampled = sampled[:n]
        else:
            # 3 quantile bins
            try:
                df2["risk_bin"] = pd.qcut(df2["offline_risk_score"], q=3, labels=["low", "mid", "high"], duplicates="drop")
            except Exception:
                df2["risk_bin"] = "all"
            bins = sorted(df2["risk_bin"].dropna().unique().tolist())
            per_bin = max(1, int(math.ceil(n / max(1, len(bins)))))
            sampled = []
            for b in bins:
                pool = df2[df2["risk_bin"] == b]["paper_id"].astype(str).tolist()
                random.shuffle(pool)
                sampled.extend(pool[:per_bin])
            sampled = sampled[:n]
    else:
        paper_ids = df["paper_id"].astype(str).tolist()
        sampled = random.sample(paper_ids, n)

    tasks_dir = Path(args.tasks_dir)
    labels_dir = Path(args.labels_dir)
    tasks_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    by_id = {str(r.paper_id): r for r in df.itertuples(index=False)}
    wrote = 0

    for pid in sampled:
        row = by_id.get(pid)
        if row is None:
            continue
        ssrn_id = getattr(row, "ssrn_id", None)
        title = getattr(row, "title", None)
        year = getattr(row, "year", None)
        pdf_relpath = getattr(row, "pdf_relpath", None)
        ssrn_url = getattr(row, "ssrn_url", None)
        pages_to_review = getattr(row, "candidate_pages_parsed", []) or []
        if isinstance(pages_to_review, str):
            pages_to_review = _parse_json_cell(pages_to_review)
        if not isinstance(pages_to_review, list):
            pages_to_review = []
        pages_to_review = [int(x) for x in pages_to_review if isinstance(x, (int, float)) and int(x) > 0][:12]

        task = {
            "paper_id": pid,
            "ssrn_id": ssrn_id,
            "title": title if isinstance(title, str) else None,
            "year": int(year) if isinstance(year, (int, float)) and year == year else None,
            "pdf_relpath": str(pdf_relpath) if isinstance(pdf_relpath, str) else None,
            "ssrn_url": str(ssrn_url) if isinstance(ssrn_url, str) else None,
            "pages_to_review": pages_to_review,
            "labels": _empty_labels(),
            "overall_audit_assessment": {"value": None, "notes": None},
            "annotator": None,
            "completed_at": None,
            "generated_at": _now_iso(),
        }

        task_path = tasks_dir / f"{pid}.json"
        label_path = labels_dir / f"{pid}.json"
        task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
        if not label_path.exists():
            label_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
        wrote += 1

    print(f"[{_now_iso()}] wrote tasks: {wrote} -> {tasks_dir}")
    print(f"[{_now_iso()}] wrote empty labels: {wrote} -> {labels_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
