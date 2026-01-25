#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


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


def _safe_str(x: Any) -> str | None:
    if isinstance(x, str):
        v = x.strip()
        return v or None
    return None


def _read_features_map(features_csv: Path) -> dict[str, dict[str, Any]]:
    if not features_csv.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with features_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            paper_id = (row.get("paper_id") or "").strip()
            if not paper_id:
                continue
            out[paper_id] = {
                "offline_risk_score": _safe_int(row.get("offline_risk_score")),
                "offline_risk_level": _safe_str(row.get("offline_risk_level")),
                "t_pairs_seen": _safe_int(row.get("t_pairs_seen")),
                "t_pairs_seen_raw": _safe_int(row.get("t_pairs_seen_raw")),
                "tables_seen": _safe_int(row.get("tables_seen")),
                "tables_seen_raw": _safe_int(row.get("tables_seen_raw")),
            }
    return out


def _top_k_counts(d: dict[str, int], k: int = 5) -> list[tuple[str, int]]:
    items = [(str(a), int(b)) for a, b in (d or {}).items() if isinstance(a, str)]
    items.sort(key=lambda x: (-x[1], x[0]))
    return items[: max(0, int(k))]


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize extraction meta diagnostics from corpus/tests/*.meta.json.")
    ap.add_argument("--corpus-dir", default="corpus", help="Corpus directory (contains tests/ and optionally features.csv).")
    ap.add_argument("--out-dir", default="analysis", help="Output directory for report + CSV.")
    ap.add_argument("--out-csv", default=None, help="Output CSV path (default: <out-dir>/extraction_quality.csv).")
    ap.add_argument(
        "--out-md",
        default=None,
        help="Output markdown path (default: <out-dir>/extraction_quality_report.md).",
    )
    ap.add_argument("--top-n", type=int, default=20, help="Top-N papers to list in the markdown report.")
    args = ap.parse_args()

    corpus_dir = Path(args.corpus_dir)
    tests_dir = corpus_dir / "tests"
    if not tests_dir.exists():
        raise SystemExit(f"Missing tests dir: {tests_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv) if args.out_csv else (out_dir / "extraction_quality.csv")
    out_md = Path(args.out_md) if args.out_md else (out_dir / "extraction_quality_report.md")

    features_map = _read_features_map(corpus_dir / "features.csv")

    tests_files = sorted(tests_dir.glob("*.jsonl"))
    if not tests_files:
        raise SystemExit(f"No tests jsonl files under: {tests_dir}")

    rows: list[dict[str, Any]] = []
    missing_meta: list[str] = []

    paren_mode_counts: Counter[str] = Counter()
    paren_mode_source_counts: Counter[str] = Counter()
    extractor_version_counts: Counter[str] = Counter()
    global_filter_counts: Counter[str] = Counter()

    total_pairs_raw = 0
    total_pairs_kept = 0

    for tf in tests_files:
        paper_id = tf.stem
        meta_path = tf.with_suffix(".meta.json")
        meta: dict[str, Any] | None = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                meta = None
        else:
            missing_meta.append(paper_id)

        extractor_version = _safe_str(meta.get("extractor_version")) if isinstance(meta, dict) else None
        paren_mode = _safe_str(meta.get("paren_mode")) if isinstance(meta, dict) else None
        paren_mode_source = _safe_str(meta.get("paren_mode_source")) if isinstance(meta, dict) else None

        candidate_pages = meta.get("candidate_pages") if isinstance(meta, dict) else None
        ref_pages = meta.get("reference_pages_detected") if isinstance(meta, dict) else None
        pairs_raw = _safe_int(meta.get("pairs_raw")) if isinstance(meta, dict) else None
        pairs_kept = _safe_int(meta.get("pairs_kept")) if isinstance(meta, dict) else None
        tables_raw = meta.get("tables_raw") if isinstance(meta, dict) else None
        tables_kept = meta.get("tables_kept") if isinstance(meta, dict) else None
        filter_counts = meta.get("filter_counts") if isinstance(meta, dict) else None

        pairs_raw_i = int(pairs_raw or 0)
        pairs_kept_i = int(pairs_kept or 0)
        total_pairs_raw += pairs_raw_i
        total_pairs_kept += pairs_kept_i

        if extractor_version:
            extractor_version_counts[extractor_version] += 1
        if paren_mode:
            paren_mode_counts[paren_mode] += 1
        if paren_mode_source:
            paren_mode_source_counts[paren_mode_source] += 1

        if isinstance(filter_counts, dict):
            for k, v in filter_counts.items():
                if not isinstance(k, str):
                    continue
                vv = _safe_int(v)
                if vv is None:
                    continue
                global_filter_counts[k] += int(vv)

        keep_rate = (pairs_kept_i / pairs_raw_i) if pairs_raw_i else None
        tables_raw_n = len(tables_raw) if isinstance(tables_raw, list) else None
        tables_kept_n = len(tables_kept) if isinstance(tables_kept, list) else None
        table_keep_rate = (float(tables_kept_n) / float(tables_raw_n)) if tables_raw_n else None

        fm = features_map.get(paper_id, {})
        rows.append(
            {
                "paper_id": paper_id,
                "tests_relpath": str(tf.relative_to(corpus_dir)).replace("\\", "/"),
                "meta_relpath": str(meta_path.relative_to(corpus_dir)).replace("\\", "/") if meta_path.exists() else None,
                "extractor_version": extractor_version,
                "paren_mode": paren_mode,
                "paren_mode_source": paren_mode_source,
                "candidate_pages_n": len(candidate_pages) if isinstance(candidate_pages, list) else None,
                "reference_pages_detected_n": len(ref_pages) if isinstance(ref_pages, list) else None,
                "pairs_raw": pairs_raw_i if meta_path.exists() else None,
                "pairs_kept": pairs_kept_i if meta_path.exists() else None,
                "keep_rate": keep_rate,
                "tables_raw_n": tables_raw_n,
                "tables_kept_n": tables_kept_n,
                "table_keep_rate": table_keep_rate,
                "filter_counts_json": json.dumps(filter_counts, ensure_ascii=False) if isinstance(filter_counts, dict) else None,
                "filter_top5": "; ".join([f"{k}:{v}" for k, v in _top_k_counts(filter_counts or {}, 5)]) if isinstance(filter_counts, dict) else None,
                "offline_risk_score": fm.get("offline_risk_score"),
                "offline_risk_level": fm.get("offline_risk_level"),
                "features_t_pairs_seen": fm.get("t_pairs_seen"),
                "features_t_pairs_seen_raw": fm.get("t_pairs_seen_raw"),
                "features_tables_seen": fm.get("tables_seen"),
                "features_tables_seen_raw": fm.get("tables_seen_raw"),
            }
        )

    # Write CSV
    fields: list[str] = []
    for r0 in rows:
        for k in r0.keys():
            if k not in fields:
                fields.append(k)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r0 in rows:
            w.writerow({k: r0.get(k) for k in fields})

    # Markdown report
    meta_covered = len(rows) - len(missing_meta)
    keep_rate_overall = (float(total_pairs_kept) / float(total_pairs_raw)) if total_pairs_raw else None
    top_n = max(1, int(args.top_n))
    rows_with_keep = [r for r in rows if isinstance(r.get("keep_rate"), float)]
    rows_with_keep.sort(key=lambda r: (-(r.get("pairs_raw") or 0), (r.get("keep_rate") or 0.0)))
    worst_keep = sorted(rows_with_keep, key=lambda r: (r.get("keep_rate") or 0.0, -(r.get("pairs_raw") or 0)))[:top_n]

    lines: list[str] = []
    lines.append("# Extraction quality report (tests/*.meta.json)")
    lines.append("")
    lines.append(f"> Generated: {_now_iso()}")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Test files: {len(tests_files)}")
    lines.append(f"- Meta files: {meta_covered} (missing: {len(missing_meta)})")
    lines.append(f"- Total pairs raw: {total_pairs_raw}")
    lines.append(f"- Total pairs kept: {total_pairs_kept}")
    lines.append(f"- Overall keep rate: {keep_rate_overall:.3f}" if keep_rate_overall is not None else "- Overall keep rate: n/a")
    lines.append("")

    def _fmt_counts(c: Counter[str]) -> str:
        items = sorted(c.items(), key=lambda x: (-x[1], x[0]))
        return ", ".join([f"{k}={v}" for k, v in items[:12]]) if items else "n/a"

    lines.append("## Paren mode (paper-level inference)")
    lines.append("")
    lines.append(f"- `paren_mode`: {_fmt_counts(paren_mode_counts)}")
    lines.append(f"- `paren_mode_source`: {_fmt_counts(paren_mode_source_counts)}")
    lines.append("")
    lines.append("## Extractor versions")
    lines.append("")
    lines.append(f"- `extractor_version`: {_fmt_counts(extractor_version_counts)}")
    lines.append("")
    lines.append("## Global filter reasons (sum over papers)")
    lines.append("")
    lines.append(f"- Top reasons: {_fmt_counts(global_filter_counts)}")
    lines.append("")

    if worst_keep:
        lines.append(f"## Lowest keep-rate papers (top {top_n})")
        lines.append("")
        lines.append("| paper_id | pairs_raw | pairs_kept | keep_rate | paren_mode | top_filters |")
        lines.append("|---|---:|---:|---:|---|---|")
        for r0 in worst_keep:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(r0.get("paper_id") or ""),
                        str(r0.get("pairs_raw") or 0),
                        str(r0.get("pairs_kept") or 0),
                        f"{float(r0.get('keep_rate') or 0.0):.3f}",
                        str(r0.get("paren_mode") or ""),
                        str(r0.get("filter_top5") or "").replace("|", "\\|"),
                    ]
                )
                + " |"
            )
        lines.append("")

    if missing_meta:
        lines.append("## Missing meta files (first 50)")
        lines.append("")
        for pid in missing_meta[:50]:
            lines.append(f"- {pid}")
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[{_now_iso()}] wrote: {out_csv}")
    print(f"[{_now_iso()}] wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

