#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import time
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


def _parse_json_cell(x: Any) -> Any:
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, str) and x.strip().startswith("{"):
        try:
            return json.loads(x)
        except Exception:
            return None
    if isinstance(x, str) and x.strip().startswith("["):
        try:
            return json.loads(x)
        except Exception:
            return None
    return None


def _flatten_counts(prefix: str, d: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not isinstance(d, dict):
        return out
    for k in ["left", "right", "total"]:
        out[f"{prefix}_{k}"] = _safe_int(d.get(k))
    return out


def _sum_dict_ints(d: Any) -> int | None:
    if not isinstance(d, dict):
        return None
    s = 0
    for v in d.values():
        vi = _safe_int(v)
        if vi is None:
            continue
        s += int(vi)
    return int(s)


def _ssrn_id_from_paper_id(paper_id: str) -> int | None:
    m = re.search(r"(\d+)$", paper_id or "")
    return int(m.group(1)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a paper-level panel dataset by joining manifest + features.")
    ap.add_argument("--corpus-dir", default="corpus", help="Corpus directory (manifest.csv + features.csv).")
    ap.add_argument("--out", default="analysis/paper_panel.csv", help="Output CSV path.")
    args = ap.parse_args()

    corpus_dir = Path(args.corpus_dir)
    manifest_path = corpus_dir / "manifest.csv"
    features_path = corpus_dir / "features.csv"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise SystemExit(f"Missing: {manifest_path}")
    if not features_path.exists():
        raise SystemExit(f"Missing: {features_path}")

    manifest_by_id: dict[str, dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ssrn_id = (row.get("ssrn_id") or "").strip()
            if not ssrn_id:
                continue
            manifest_by_id[ssrn_id] = dict(row)

    panel_rows: list[dict[str, Any]] = []
    with features_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            paper_id = (row.get("paper_id") or "").strip()
            if not paper_id:
                continue
            ssrn_id = _ssrn_id_from_paper_id(paper_id)
            m = manifest_by_id.get(str(ssrn_id)) if ssrn_id is not None else None

            out: dict[str, Any] = {
                "paper_id": paper_id,
                "ssrn_id": ssrn_id,
                "year": _safe_int(m.get("year")) if m else _safe_int(row.get("year")),
                "title": (m.get("title") if m else None),
                "authors": (m.get("authors") if m else None),
                "doi": (m.get("doi") if m else None),
                "ssrn_url": (m.get("ssrn_url") if m else None),
                "pdf_relpath": (m.get("pdf_relpath") if m else row.get("pdf_relpath")),
                "extractor_version": (row.get("extractor_version") or "").strip() or None,
                "tests_relpath": (row.get("tests_relpath") or "").strip() or None,
                "tests_meta_relpath": (row.get("tests_meta_relpath") or "").strip() or None,
                "paren_mode": (row.get("paren_mode") or "").strip() or None,
                "paren_mode_source": (row.get("paren_mode_source") or "").strip() or None,
                "reference_pages_detected_n": _safe_int(row.get("reference_pages_detected_n")),
                "offline_risk_score": _safe_int(row.get("offline_risk_score")),
                "offline_risk_level": (row.get("offline_risk_level") or "").strip() or None,
                "pages": _safe_int(row.get("pages")),
                "skipped_reason": (row.get("skipped_reason") or "").strip() or None,
                "extracted_text_chars": _safe_int(row.get("extracted_text_chars")),
                "p_values_found": _safe_int(row.get("p_values_found")),
                "tables_seen": _safe_int(row.get("tables_seen")),
                "tables_seen_raw": _safe_int(row.get("tables_seen_raw")),
                "t_pairs_seen": _safe_int(row.get("t_pairs_seen")),
                "t_pairs_seen_raw": _safe_int(row.get("t_pairs_seen_raw")),
                "t_pairs_keep_rate": _safe_float(row.get("t_pairs_keep_rate")),
                "p_from_t_significant_0_05": _safe_int(row.get("p_from_t_significant_0_05")),
                "p_from_t_significant_0_10": _safe_int(row.get("p_from_t_significant_0_10")),
                "robust_mentions_fulltext": _safe_int(row.get("robust_mentions_fulltext")),
                "spec_search_terms_fulltext": _safe_int(row.get("spec_search_terms_fulltext")),
                "multiple_testing_terms_fulltext": _safe_int(row.get("multiple_testing_terms_fulltext")),
                "has_multiple_testing_correction": (row.get("has_multiple_testing_correction") or "").strip(),
                "p_from_t_caliper_0_05_z": _safe_float(row.get("p_from_t_caliper_0_05_z")),
                "p_from_t_caliper_0_05_p": _safe_float(row.get("p_from_t_caliper_0_05_p")),
                "p_from_t_caliper_0_10_z": _safe_float(row.get("p_from_t_caliper_0_10_z")),
                "p_from_t_caliper_0_10_p": _safe_float(row.get("p_from_t_caliper_0_10_p")),
                "p_regex_caliper_0_05_z": _safe_float(row.get("p_regex_caliper_0_05_z")),
                "p_regex_caliper_0_05_p": _safe_float(row.get("p_regex_caliper_0_05_p")),
                "p_regex_caliper_0_10_z": _safe_float(row.get("p_regex_caliper_0_10_z")),
                "p_regex_caliper_0_10_p": _safe_float(row.get("p_regex_caliper_0_10_p")),
                "t_caliper_1_96_z": _safe_float(row.get("t_caliper_1_96_z")),
                "t_caliper_1_96_p": _safe_float(row.get("t_caliper_1_96_p")),
                "t_caliper_1_645_z": _safe_float(row.get("t_caliper_1_645_z")),
                "t_caliper_1_645_p": _safe_float(row.get("t_caliper_1_645_p")),
                "pcurve_significant_n": _safe_int(row.get("pcurve_significant_n")),
                "pcurve_right_skew_z": _safe_float(row.get("pcurve_right_skew_z")),
                "pcurve_right_skew_p": _safe_float(row.get("pcurve_right_skew_p")),
            }

            out.update(_flatten_counts("p_near_005", _parse_json_cell(row.get("near_0_05_p"))))
            out.update(_flatten_counts("p_near_010", _parse_json_cell(row.get("near_0_10_p"))))
            out.update(_flatten_counts("p_from_t_near_005", _parse_json_cell(row.get("p_from_t_near_0_05"))))
            out.update(_flatten_counts("p_from_t_near_010", _parse_json_cell(row.get("p_from_t_near_0_10"))))
            out.update(_flatten_counts("t_near_196", _parse_json_cell(row.get("t_near_1_96"))))
            out.update(_flatten_counts("t_near_1645", _parse_json_cell(row.get("t_near_1_645"))))

            filter_counts = _parse_json_cell(row.get("filter_counts"))
            out["filter_dropped_total"] = _sum_dict_ints(filter_counts)
            out["filter_counts_json"] = json.dumps(filter_counts, ensure_ascii=False) if isinstance(filter_counts, dict) else None

            out["generated_at"] = _now_iso()
            panel_rows.append(out)

    if not panel_rows:
        raise SystemExit("No rows produced.")

    fields: list[str] = []
    for r0 in panel_rows:
        for k in r0.keys():
            if k not in fields:
                fields.append(k)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r0 in panel_rows:
            w.writerow({k: r0.get(k) for k in fields})

    print(f"[{_now_iso()}] wrote: {out_path} (rows={len(panel_rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
