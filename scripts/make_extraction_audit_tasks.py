#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


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


def _read_features(features_csv: Path) -> list[dict[str, Any]]:
    if not features_csv.exists():
        return []
    rows: list[dict[str, Any]] = []
    with features_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(dict(row))
    return rows


def _pick_papers(
    rows: list[dict[str, Any]],
    *,
    n_papers: int,
    seed: int,
    selection: str,
    min_t_pairs_seen: int = 1,
) -> list[str]:
    eligible: list[str] = []
    for row in rows:
        pid = (row.get("paper_id") or "").strip()
        if not pid:
            continue
        t_pairs_seen = _safe_int(row.get("t_pairs_seen")) or 0
        if int(t_pairs_seen) >= int(min_t_pairs_seen):
            eligible.append(pid)
    paper_ids = sorted(set(eligible))
    if not paper_ids:
        return []

    rng = random.Random(seed)
    if selection == "top_risk":
        scored: list[tuple[float, str]] = []
        for row in rows:
            pid = (row.get("paper_id") or "").strip()
            s = _safe_float(row.get("offline_risk_score"))
            t_pairs_seen = _safe_int(row.get("t_pairs_seen")) or 0
            if pid and s is not None and int(t_pairs_seen) >= int(min_t_pairs_seen):
                scored.append((float(s), pid))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        out: list[str] = []
        for _, pid in scored:
            if pid not in out:
                out.append(pid)
            if len(out) >= n_papers:
                break
        return out

    if selection == "stratified":
        by_level: dict[str, list[str]] = {"Low": [], "Moderate": [], "High": []}
        for row in rows:
            pid = (row.get("paper_id") or "").strip()
            lvl = (row.get("offline_risk_level") or "").strip()
            t_pairs_seen = _safe_int(row.get("t_pairs_seen")) or 0
            if pid and lvl in by_level and int(t_pairs_seen) >= int(min_t_pairs_seen) and pid not in by_level[lvl]:
                by_level[lvl].append(pid)
        for lvl in by_level:
            rng.shuffle(by_level[lvl])
        per = max(1, int(math.ceil(n_papers / 3)))
        out: list[str] = []
        for lvl in ["High", "Moderate", "Low"]:
            for pid in by_level[lvl][:per]:
                if pid not in out:
                    out.append(pid)
                if len(out) >= n_papers:
                    return out
        # Fill remaining randomly
        remaining = [pid for pid in paper_ids if pid not in out]
        rng.shuffle(remaining)
        out.extend(remaining[: max(0, n_papers - len(out))])
        return out[:n_papers]

    # random
    rng.shuffle(paper_ids)
    return paper_ids[:n_papers]


def _union_bbox(a: list[float] | None, b: list[float] | None) -> list[float] | None:
    if not a and not b:
        return None
    if a and not b:
        return list(a)
    if b and not a:
        return list(b)
    if not a or not b:
        return None
    try:
        x0 = min(float(a[0]), float(b[0]))
        y0 = min(float(a[1]), float(b[1]))
        x1 = max(float(a[2]), float(b[2]))
        y1 = max(float(a[3]), float(b[3]))
        return [x0, y0, x1, y1]
    except Exception:
        return None


def _clip_rect(page_rect: fitz.Rect, bbox: list[float], pad: float) -> fitz.Rect:
    r = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
    r = fitz.Rect(r.x0 - pad, r.y0 - pad, r.x1 + pad, r.y1 + pad)
    # Clamp
    r = fitz.Rect(
        max(page_rect.x0, r.x0),
        max(page_rect.y0, r.y0),
        min(page_rect.x1, r.x1),
        min(page_rect.y1, r.y1),
    )
    # If degenerate, fall back to full page.
    if r.is_empty or r.x1 <= r.x0 or r.y1 <= r.y0:
        return fitz.Rect(page_rect)
    return r


def _is_near_threshold(rec: dict[str, Any]) -> bool:
    at = _safe_float(rec.get("abs_t"))
    if at is not None:
        if 1.445 <= at <= 2.165:
            return True
    p2 = _safe_float(rec.get("p_approx_2s"))
    if p2 is not None:
        if 0.045 <= p2 <= 0.055 or 0.095 <= p2 <= 0.105:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Create cell-level extraction audit tasks + evidence snippets.")
    ap.add_argument("--corpus-dir", default="corpus", help="Corpus directory (with pdfs/ and tests/).")
    ap.add_argument("--paper-id", action="append", default=[], help="Paper id (repeatable). Default: sample from features.csv.")
    ap.add_argument("--n-papers", type=int, default=12, help="How many papers to sample if --paper-id not provided.")
    ap.add_argument("--per-paper", type=int, default=12, help="How many extracted pairs per paper to sample.")
    ap.add_argument("--seed", type=int, default=7, help="Random seed.")
    ap.add_argument(
        "--min-t-pairs",
        type=int,
        default=1,
        help="Only sample papers with at least this many extracted pairs (t_pairs_seen) when using features.csv.",
    )
    ap.add_argument(
        "--selection",
        choices=["random", "top_risk", "stratified"],
        default="stratified",
        help="Paper sampling strategy when using features.csv.",
    )
    ap.add_argument("--out-dir", default="annotations/extraction_tasks", help="Task JSON output directory.")
    ap.add_argument("--labels-dir", default="annotations/extraction_labels", help="Empty label JSON directory.")
    ap.add_argument("--snippets-dir", default="annotations/extraction_snippets", help="PNG evidence output directory.")
    ap.add_argument("--dpi", type=int, default=220, help="Render DPI for snippets.")
    ap.add_argument("--pad", type=float, default=10.0, help="Padding (points) around bbox when clipping.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing tasks/snippets.")
    args = ap.parse_args()

    corpus_dir = Path(args.corpus_dir)
    pdf_dir = corpus_dir / "pdfs"
    tests_dir = corpus_dir / "tests"
    features_csv = corpus_dir / "features.csv"

    out_dir = Path(args.out_dir)
    labels_dir = Path(args.labels_dir)
    snippets_dir = Path(args.snippets_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    snippets_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))

    paper_ids: list[str] = [str(x).strip() for x in args.paper_id if str(x).strip()]
    if not paper_ids:
        feats = _read_features(features_csv)
        paper_ids = _pick_papers(
            feats,
            n_papers=int(args.n_papers),
            seed=int(args.seed),
            selection=str(args.selection),
            min_t_pairs_seen=int(args.min_t_pairs),
        )

    if not paper_ids:
        raise SystemExit(f"No papers found. Expected {features_csv} or explicit --paper-id.")

    made = 0
    for paper_id in paper_ids:
        tests_path = tests_dir / f"{paper_id}.jsonl"
        pdf_path = pdf_dir / f"{paper_id}.pdf"
        if not tests_path.exists() or not pdf_path.exists():
            continue

        task_path = out_dir / f"{paper_id}.json"
        label_path = labels_dir / f"{paper_id}.json"
        if task_path.exists() and label_path.exists() and not args.force:
            continue

        # Load extracted pairs
        recs: list[dict[str, Any]] = []
        for line in tests_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                recs.append(obj)
        if not recs:
            continue

        near = [r for r in recs if _is_near_threshold(r)]
        rest = [r for r in recs if r not in near]
        rng.shuffle(near)
        rng.shuffle(rest)
        k = max(1, int(args.per_paper))
        k_near = min(len(near), max(0, k // 2))
        picked = near[:k_near] + rest[: max(0, k - k_near)]
        if not picked:
            continue

        # Render snippets
        paper_snip_dir = snippets_dir / paper_id
        paper_snip_dir.mkdir(parents=True, exist_ok=True)
        doc = fitz.open(pdf_path)

        items: list[dict[str, Any]] = []
        for idx, rec in enumerate(picked, start=1):
            page_1based = _safe_int(rec.get("page"))
            if page_1based is None or page_1based < 1 or page_1based > doc.page_count:
                continue
            page = doc.load_page(page_1based - 1)
            page_rect = page.rect

            coef_bbox = rec.get("coef_cell_bbox")
            se_bbox = rec.get("se_cell_bbox")
            table_bbox = rec.get("table_bbox")

            bbox = _union_bbox(coef_bbox if isinstance(coef_bbox, list) else None, se_bbox if isinstance(se_bbox, list) else None)
            if bbox is None and isinstance(table_bbox, list):
                bbox = list(table_bbox)
            if bbox is None:
                continue

            clip = _clip_rect(page_rect, bbox, pad=float(args.pad))
            pix = page.get_pixmap(dpi=int(args.dpi), clip=clip)

            item_id = f"{idx:03d}"
            out_png = paper_snip_dir / f"{paper_id}__item_{item_id}_p{page_1based}_t{_safe_int(rec.get('table_index')) or 0}.png"
            if out_png.exists() and not args.force:
                pass
            else:
                pix.save(out_png)

            coef = _safe_float(rec.get("coef"))
            paren = _safe_float(rec.get("paren"))
            if paren is None:
                # Back-compat: older extractors stored the parentheses number in `se`.
                paren = _safe_float(rec.get("se"))
            se = _safe_float(rec.get("se"))
            t = _safe_float(rec.get("t"))
            abs_t = _safe_float(rec.get("abs_t"))
            p2 = _safe_float(rec.get("p_approx_2s"))
            paren_mode_assumed = (rec.get("paren_mode_assumed") or "").strip() if isinstance(rec.get("paren_mode_assumed"), str) else None
            if paren_mode_assumed is None and "paren" not in rec and _safe_float(rec.get("se")) is not None:
                paren_mode_assumed = "se"
            t_mode_extracted = (rec.get("t_mode") or "").strip() if isinstance(rec.get("t_mode"), str) else None
            paren_source = (rec.get("paren_source") or "").strip() if isinstance(rec.get("paren_source"), str) else None

            items.append(
                {
                    "item_id": item_id,
                    "page": page_1based,
                    "table_index": _safe_int(rec.get("table_index")),
                    "row_index": _safe_int(rec.get("row_index")),
                    "col_index": _safe_int(rec.get("col_index")),
                    "coef_extracted": coef,
                    "paren_extracted": paren,
                    "paren_mode_assumed": paren_mode_assumed,
                    "se_extracted": se,
                    "t_extracted": t,
                    "abs_t_extracted": abs_t,
                    "p_approx_2s_extracted": p2,
                    "t_mode_extracted": t_mode_extracted,
                    "paren_source": paren_source,
                    "stars_extracted": _safe_int(rec.get("stars")),
                    "snippet_relpath": str(out_png).replace("\\", "/"),
                    "bbox_used": bbox,
                }
            )

        doc.close()
        if not items:
            continue

        task_obj = {
            "version": "0.2",
            "created_at": _now_iso(),
            "paper_id": paper_id,
            "pdf_relpath": str(pdf_path.relative_to(corpus_dir)).replace("\\", "/"),
            "tests_relpath": str(tests_path.relative_to(corpus_dir)).replace("\\", "/"),
            "instructions": (
                "For each item, open the snippet image and transcribe the coefficient and the number in parentheses exactly as printed "
                "(including sign and decimal places). Then indicate whether the parentheses number is a standard error or a t-statistic "
                "(choose: se / t / unknown). Optionally record the number of significance stars."
            ),
            "items": items,
        }
        label_obj = {
            "version": "0.2",
            "paper_id": paper_id,
            "annotator": None,
            "completed_at": None,
            "items": [
                {
                    "item_id": it["item_id"],
                    "snippet_relpath": it["snippet_relpath"],
                    "coef_extracted": it["coef_extracted"],
                    "paren_extracted": it.get("paren_extracted"),
                    "paren_mode_assumed": it.get("paren_mode_assumed"),
                    "se_extracted": it["se_extracted"],
                    "t_extracted": it.get("t_extracted"),
                    "stars_extracted": it["stars_extracted"],
                    "observed_coef": None,
                    "observed_paren": None,
                    "observed_paren_mode": None,
                    "observed_se": None,
                    "observed_t": None,
                    "observed_stars": None,
                    "notes": None,
                }
                for it in items
            ],
        }

        task_path.write_text(json.dumps(task_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        label_path.write_text(json.dumps(label_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        made += 1

    print(f"[{_now_iso()}] wrote {made} task files to {out_dir}")
    print(f"[{_now_iso()}] wrote {made} label templates to {labels_dir}")
    print(f"[{_now_iso()}] snippets under {snippets_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
