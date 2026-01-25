#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _safe_str(x: Any) -> str | None:
    if isinstance(x, str):
        s = x.strip()
        return s or None
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


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _ensure_box(labels: dict[str, Any], name: str) -> dict[str, Any]:
    box = labels.get(name)
    if not isinstance(box, dict):
        box = {"value": None, "evidence": []}
        labels[name] = box
    if "value" not in box:
        box["value"] = None
    if "evidence" not in box or not isinstance(box.get("evidence"), list):
        box["evidence"] = []
    return box


def _preprint_has_authors(task: dict[str, Any]) -> bool:
    pre = task.get("preprint") if isinstance(task.get("preprint"), dict) else {}
    a = pre.get("authors")
    return isinstance(a, str) and bool(a.strip())


def _best_candidate(task: dict[str, Any]) -> dict[str, Any] | None:
    v = task.get("predicted_best")
    return v if isinstance(v, dict) else None


def _candidates(task: dict[str, Any]) -> list[dict[str, Any]]:
    v = task.get("candidates_top10")
    if isinstance(v, list):
        return [x for x in v if isinstance(x, dict)]
    return []


def _rule_label_candidate(
    *,
    token_f1: float | None,
    author_overlap: float | None,
    has_preprint_authors: bool,
    yes_token_f1: float,
    yes_author_overlap: float,
    unsure_token_f1: float,
) -> tuple[str, str]:
    """
    Return (label, reason) for whether the candidate is the same work as the preprint.
    label in {"yes","no","unsure"}.
    """
    tf1 = float(token_f1) if isinstance(token_f1, (int, float)) and token_f1 == token_f1 else 0.0
    aov = float(author_overlap) if isinstance(author_overlap, (int, float)) and author_overlap == author_overlap else 0.0

    if tf1 >= float(yes_token_f1):
        if has_preprint_authors and aov < float(yes_author_overlap):
            # Perfect title match but author overlap missing is a red flag; mark unsure.
            return "unsure", f"title_token_f1={tf1:.3f} but author_overlap={aov:.3f} < {yes_author_overlap:.2f}"
        return "yes", f"title_token_f1={tf1:.3f} (>= {yes_token_f1:.2f}) and author_overlap={aov:.3f}"

    if tf1 >= float(unsure_token_f1):
        if has_preprint_authors and aov >= float(yes_author_overlap):
            return "yes", f"title_token_f1={tf1:.3f} and author_overlap={aov:.3f} (>= {yes_author_overlap:.2f})"
        return "unsure", f"title_token_f1={tf1:.3f} (>= {unsure_token_f1:.2f}) but author_overlap={aov:.3f}"

    return "no", f"title_token_f1={tf1:.3f} (< {unsure_token_f1:.2f})"


def _pick_correct_candidate(
    *,
    candidates: list[dict[str, Any]],
    has_preprint_authors: bool,
    yes_token_f1: float,
    yes_author_overlap: float,
    unsure_token_f1: float,
) -> dict[str, Any] | None:
    """
    Pick the best-looking candidate (for correction) based on title token-F1 then author overlap.
    """
    scored: list[tuple[float, float, dict[str, Any]]] = []
    for c in candidates:
        tf1 = _safe_float(c.get("title_sim_token_f1")) or 0.0
        aov = _safe_float(c.get("author_overlap")) or 0.0
        scored.append((float(tf1), float(aov), c))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    if not scored:
        return None
    # Only suggest a correction if it clears at least the "unsure" bar.
    tf1, aov, cand = scored[0]
    lab, _ = _rule_label_candidate(
        token_f1=tf1,
        author_overlap=aov,
        has_preprint_authors=has_preprint_authors,
        yes_token_f1=yes_token_f1,
        yes_author_overlap=yes_author_overlap,
        unsure_token_f1=unsure_token_f1,
    )
    if lab == "no":
        return None
    return cand


def main() -> int:
    ap = argparse.ArgumentParser(description="Auto-fill publication-map audit labels (AI-assisted substitute for human RAs).")
    ap.add_argument("--tasks-dir", default="analysis/publication_map_tasks", help="Directory with mapping audit task JSONs.")
    ap.add_argument("--labels-dir", default="analysis/publication_map_labels", help="Directory to write label JSONs.")
    ap.add_argument("--annotator", default="ai_auto", help="Annotator name to write into JSON.")
    ap.add_argument("--yes-token-f1", type=float, default=0.90)
    ap.add_argument("--yes-author-overlap", type=float, default=0.25)
    ap.add_argument("--unsure-token-f1", type=float, default=0.70)
    ap.add_argument("--fill-correct-id", action="store_true", help="If predicted_best is labeled no, try to suggest a correct_openalex_id from candidates_top10.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing label files.")
    args = ap.parse_args()

    tasks_dir = Path(args.tasks_dir)
    labels_dir = Path(args.labels_dir)
    if not tasks_dir.exists():
        raise SystemExit(f"Missing: {tasks_dir}")
    labels_dir.mkdir(parents=True, exist_ok=True)

    wrote = 0
    skipped = 0
    for task_path in sorted(tasks_dir.glob("*.json")):
        task = _read_json(task_path)
        if not isinstance(task, dict):
            continue
        pid = _safe_str(task.get("paper_id")) or task_path.stem
        out_path = labels_dir / f"{pid}.json"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        labels = task.get("labels") if isinstance(task.get("labels"), dict) else {}
        task["labels"] = labels
        is_same = _ensure_box(labels, "is_same_work")
        correct = _ensure_box(labels, "correct_openalex_id")
        # notes is a plain box in our task schema; keep it compatible.
        if not isinstance(labels.get("notes"), dict):
            labels["notes"] = {"value": None}

        best = _best_candidate(task) or {}
        has_auth = _preprint_has_authors(task)
        tf1 = _safe_float(best.get("title_sim_token_f1"))
        aov = _safe_float(best.get("author_overlap"))
        lab, reason = _rule_label_candidate(
            token_f1=tf1,
            author_overlap=aov,
            has_preprint_authors=has_auth,
            yes_token_f1=float(args.yes_token_f1),
            yes_author_overlap=float(args.yes_author_overlap),
            unsure_token_f1=float(args.unsure_token_f1),
        )

        is_same["value"] = lab
        is_same["evidence"] = [
            {
                "type": "ai_auto_rule",
                "reason": reason,
                "pred_best_openalex_id": best.get("openalex_id"),
                "pred_best_doi_url": best.get("doi_url"),
                "pred_best_primary_source": best.get("primary_source"),
                "pred_best_title_sim_token_f1": tf1,
                "pred_best_author_overlap": aov,
            }
        ]

        if lab == "no" and args.fill_correct_id:
            cand = _pick_correct_candidate(
                candidates=_candidates(task),
                has_preprint_authors=has_auth,
                yes_token_f1=float(args.yes_token_f1),
                yes_author_overlap=float(args.yes_author_overlap),
                unsure_token_f1=float(args.unsure_token_f1),
            )
            if cand and isinstance(cand.get("openalex_id"), str):
                correct["value"] = cand.get("openalex_id")
                correct["evidence"] = [
                    {
                        "type": "ai_auto_suggestion",
                        "reason": "best candidate by title token-F1 / author overlap among candidates_top10",
                        "openalex_id": cand.get("openalex_id"),
                        "doi_url": cand.get("doi_url"),
                        "primary_source": cand.get("primary_source"),
                        "title_sim_token_f1": cand.get("title_sim_token_f1"),
                        "author_overlap": cand.get("author_overlap"),
                    }
                ]
            else:
                correct["value"] = None
                correct["evidence"] = []
        else:
            correct["value"] = None
            correct["evidence"] = []

        task["annotator"] = str(args.annotator)
        task["completed_at"] = _now_iso()
        task["generated_at"] = task.get("generated_at") or _now_iso()

        _write_json(out_path, task)
        wrote += 1

    print(f"[{_now_iso()}] wrote: {wrote} labels -> {labels_dir}")
    print(f"[{_now_iso()}] skipped: {skipped} (exists; use --overwrite)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

