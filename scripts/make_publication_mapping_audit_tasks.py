#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _paper_id_from_filename(path: Path) -> str:
    stem = path.stem
    if "__" in stem:
        return stem.split("__", 1)[0]
    return stem


def _ssrn_id_from_paper_id(paper_id: str) -> int | None:
    m = re.search(r"(\d+)$", paper_id or "")
    return int(m.group(1)) if m else None


def _safe_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return bool(x)
    if isinstance(x, (int, float)):
        return bool(int(x))
    if isinstance(x, str):
        s = x.strip().lower()
        return s in {"true", "1", "yes", "y"}
    return False


def _box() -> dict[str, Any]:
    return {"value": None, "evidence": []}


def _empty_labels() -> dict[str, Any]:
    return {
        "is_same_work": _box(),  # yes/no/unsure
        "correct_openalex_id": _box(),  # optional: if not same, choose correct candidate openalex_id if visible
        "notes": {"value": None},
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Create human-audit tasks to validate SSRN→published-version mapping.")
    ap.add_argument(
        "--search-jsonl",
        default="analysis/openalex_search_publication_map.jsonl",
        help="JSONL from map_published_versions_openalex_search.py.",
    )
    ap.add_argument("--panel", default=None, help="Optional paper_panel.csv to enrich tasks with ssrn_url/pdf/year/offline score.")
    ap.add_argument("--n", type=int, default=150, help="Number of papers to sample for mapping audit.")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--selection",
        choices=["random", "stratified", "top_confident", "top_top3"],
        default="stratified",
        help="Sampling strategy over mapping predictions.",
    )
    ap.add_argument("--tasks-dir", default=None, help="Output tasks directory (default: <dir>/publication_map_tasks).")
    ap.add_argument("--labels-dir", default=None, help="Output labels directory (default: <dir>/publication_map_labels).")
    ap.add_argument("--force-label-templates", action="store_true", help="Overwrite existing label templates.")
    args = ap.parse_args()

    search_path = Path(args.search_jsonl)
    if not search_path.exists():
        raise SystemExit(f"Missing: {search_path}")
    out_dir = search_path.parent
    tasks_dir = Path(args.tasks_dir) if args.tasks_dir else (out_dir / "publication_map_tasks")
    labels_dir = Path(args.labels_dir) if args.labels_dir else (out_dir / "publication_map_labels")
    tasks_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(search_path)
    if not rows:
        raise SystemExit(f"No records found in: {search_path}")

    # Build sampling frame
    frame: list[dict[str, Any]] = []
    for r in rows:
        if str(r.get("status") or "").strip() != "ok":
            continue
        pid = str(r.get("paper_id") or "").strip()
        if not pid:
            continue
        best = r.get("best") if isinstance(r.get("best"), dict) else {}
        frame.append(
            {
                "paper_id": pid,
                "best_is_confident": _safe_bool(r.get("best_is_confident")),
                "best_is_top3_venue": _safe_bool(best.get("is_top3_venue")),
                "best_score": float(best.get("score") or 0.0) if isinstance(best.get("score"), (int, float)) else 0.0,
                "best_title_sim_token_f1": float(best.get("title_sim_token_f1") or 0.0)
                if isinstance(best.get("title_sim_token_f1"), (int, float))
                else 0.0,
            }
        )

    df = pd.DataFrame(frame)
    if df.empty:
        raise SystemExit("No ok mapping records to sample from.")

    rng = random.Random(int(args.seed))
    n = min(int(args.n), int(len(df)))
    if n <= 0:
        raise SystemExit("n must be positive.")

    sampled: list[str] = []
    if args.selection == "top_confident":
        df2 = df[df["best_is_confident"].astype(bool)].copy()
        df2 = df2.sort_values(["best_score", "best_title_sim_token_f1"], ascending=False)
        sampled = df2["paper_id"].head(n).tolist()
    elif args.selection == "top_top3":
        df2 = df[df["best_is_top3_venue"].astype(bool)].copy()
        df2 = df2.sort_values(["best_score", "best_title_sim_token_f1"], ascending=False)
        sampled = df2["paper_id"].head(n).tolist()
    elif args.selection == "stratified":
        bins: list[pd.DataFrame] = []
        for conf in [True, False]:
            for top3 in [True, False]:
                sub = df[(df["best_is_confident"].astype(bool) == conf) & (df["best_is_top3_venue"].astype(bool) == top3)].copy()
                if not sub.empty:
                    bins.append(sub)
        if not bins:
            paper_ids = df["paper_id"].tolist()
            sampled = rng.sample(paper_ids, n)
        else:
            per_bin = int(math.ceil(n / len(bins)))
            picked: set[str] = set()
            for sub in bins:
                pool = sub["paper_id"].tolist()
                rng.shuffle(pool)
                for pid in pool[:per_bin]:
                    if pid in picked:
                        continue
                    sampled.append(pid)
                    picked.add(pid)
            if len(sampled) < n:
                remaining = [pid for pid in df["paper_id"].tolist() if pid not in picked]
                rng.shuffle(remaining)
                sampled.extend(remaining[: (n - len(sampled))])
            sampled = sampled[:n]
    else:
        paper_ids = df["paper_id"].tolist()
        sampled = rng.sample(paper_ids, n)

    by_pid = {str(r.get("paper_id")): r for r in rows if isinstance(r, dict) and isinstance(r.get("paper_id"), str)}

    panel_by_pid: dict[str, dict[str, Any]] = {}
    if args.panel:
        panel_path = Path(args.panel)
        if not panel_path.exists():
            raise SystemExit(f"Missing: {panel_path}")
        p = pd.read_csv(panel_path)
        if "paper_id" in p.columns:
            for rec in p.to_dict(orient="records"):
                pid = str(rec.get("paper_id") or "").strip()
                if pid:
                    panel_by_pid[pid] = rec

    wrote = 0
    for pid in sampled:
        rec = by_pid.get(pid)
        if not isinstance(rec, dict):
            continue

        best = rec.get("best") if isinstance(rec.get("best"), dict) else None
        cand = rec.get("candidates_top10") if isinstance(rec.get("candidates_top10"), list) else []
        panel = panel_by_pid.get(pid, {})

        task = {
            "paper_id": pid,
            "ssrn_id": _ssrn_id_from_paper_id(pid),
            "preprint": {
                "title": rec.get("preprint_title"),
                "authors": rec.get("preprint_authors"),
                "doi": rec.get("preprint_doi"),
                "ssrn_url": panel.get("ssrn_url") if isinstance(panel, dict) else None,
                "pdf_relpath": panel.get("pdf_relpath") if isinstance(panel, dict) else None,
                "year": panel.get("year") if isinstance(panel, dict) else None,
            },
            "predicted_best": best,
            "candidates_top10": cand,
            "model_flags": {
                "best_is_confident": _safe_bool(rec.get("best_is_confident")),
                "n_candidates_non_ssrn": int(rec.get("n_candidates_non_ssrn") or 0),
            },
            "offline": {
                "offline_risk_score": panel.get("offline_risk_score") if isinstance(panel, dict) else None,
                "t_pairs_seen": panel.get("t_pairs_seen") if isinstance(panel, dict) else None,
                "pages": panel.get("pages") if isinstance(panel, dict) else None,
            },
            "labels": _empty_labels(),
            "annotator": None,
            "completed_at": None,
            "generated_at": _now_iso(),
            "instructions": [
                "Goal: verify whether the predicted published-version match refers to the same work as the SSRN preprint.",
                "Use title + author + venue/year + DOI to decide. If unsure, choose 'unsure'.",
                "If 'no' but a correct match is visible in candidates_top10, fill correct_openalex_id with that candidate's openalex_id.",
                "Multi-annotator IRR: duplicate the label JSON as '<paper_id>__<annotator>.json' and fill 'annotator'.",
            ],
        }

        task_path = tasks_dir / f"{pid}.json"
        label_path = labels_dir / f"{pid}.json"
        task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.force_label_templates or not label_path.exists():
            label_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
        wrote += 1

    print(f"[{_now_iso()}] wrote tasks: {wrote} -> {tasks_dir}")
    print(f"[{_now_iso()}] wrote label templates: {wrote} -> {labels_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
