#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
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


def _annotator_from_filename(path: Path) -> str | None:
    stem = path.stem
    if "__" in stem:
        tail = stem.split("__", 1)[1].strip()
        return tail or None
    return None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


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


def _safe_str(x: Any) -> str | None:
    if isinstance(x, str):
        s = x.strip()
        return s or None
    return None


def _norm_label(x: Any) -> str | None:
    s = _safe_str(x)
    if not s:
        return None
    s2 = s.strip().lower()
    if s2 in {"yes", "y", "true", "1"}:
        return "yes"
    if s2 in {"no", "n", "false", "0"}:
        return "no"
    if s2 in {"unsure", "unknown", "unclear", "maybe"}:
        return "unsure"
    return s2


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


def _bin(x: float | None) -> str:
    if x is None or x != x:
        return "null"
    if x < 0.5:
        return "[0,0.5)"
    if x < 0.85:
        return "[0.5,0.85)"
    return "[0.85,1.0]"


def _looks_like_ai_annotator(name: Any) -> bool:
    if not isinstance(name, str):
        return False
    s = name.strip().lower()
    if not s:
        return False
    needles = ["ai", "auto", "gpt", "llm", "assistant", "model"]
    return any(n in s for n in needles)


def main() -> int:
    ap = argparse.ArgumentParser(description="Score publication-mapping audit labels and summarize mapping accuracy.")
    ap.add_argument("--labels-dir", default="analysis/publication_map_labels", help="Directory with completed mapping label JSON files.")
    ap.add_argument("--search-jsonl", default="analysis/openalex_search_publication_map.jsonl", help="JSONL from map_published_versions_openalex_search.py.")
    ap.add_argument("--panel", default=None, help="Optional paper_panel.csv to join offline_risk_score.")
    ap.add_argument("--out-md", default=None, help="Output markdown (default: <dir>/publication_map_audit_report.md).")
    ap.add_argument("--out-json", default=None, help="Output JSON (default: <dir>/publication_map_audit_report.json).")
    args = ap.parse_args()

    labels_dir = Path(args.labels_dir)
    search_path = Path(args.search_jsonl)
    if not labels_dir.exists():
        raise SystemExit(f"Missing: {labels_dir}")
    if not search_path.exists():
        raise SystemExit(f"Missing: {search_path}")

    out_dir = search_path.parent
    out_md = Path(args.out_md) if args.out_md else (out_dir / "publication_map_audit_report.md")
    out_json = Path(args.out_json) if args.out_json else (out_dir / "publication_map_audit_report.json")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    search_rows = _read_jsonl(search_path)
    by_pid: dict[str, dict[str, Any]] = {}
    for r in search_rows:
        pid = _safe_str(r.get("paper_id"))
        if pid:
            by_pid[pid] = r

    # Population mapping stats
    pop = pd.DataFrame(
        [
            {
                "paper_id": _safe_str(r.get("paper_id")),
                "status": _safe_str(r.get("status")),
                "best_is_confident": bool(r.get("best_is_confident")),
                "n_candidates_non_ssrn": int(r.get("n_candidates_non_ssrn") or 0),
                "best_is_top3_venue": bool((r.get("best") or {}).get("is_top3_venue")) if isinstance(r.get("best"), dict) else False,
                "best_title_sim_token_f1": _safe_float((r.get("best") or {}).get("title_sim_token_f1")) if isinstance(r.get("best"), dict) else None,
            }
            for r in search_rows
            if isinstance(r, dict)
        ]
    )
    pop_ok = pop[pop["status"].astype(str).eq("ok")].copy() if not pop.empty and "status" in pop.columns else pop.copy()

    # Read label files
    label_rows: list[dict[str, Any]] = []
    for p in sorted(labels_dir.glob("*.json")):
        obj = _read_json(p)
        if not isinstance(obj, dict):
            continue
        pid = _safe_str(obj.get("paper_id")) or _paper_id_from_filename(p)
        annotator = _safe_str(obj.get("annotator")) or _annotator_from_filename(p) or p.stem
        labels = obj.get("labels") if isinstance(obj.get("labels"), dict) else {}
        is_same = None
        if isinstance(labels.get("is_same_work"), dict):
            is_same = labels["is_same_work"].get("value")
        correct_id = None
        if isinstance(labels.get("correct_openalex_id"), dict):
            correct_id = labels["correct_openalex_id"].get("value")

        rec = {
            "paper_id": pid,
            "annotator": annotator,
            "completed_at": _safe_str(obj.get("completed_at")),
            "is_same_work": _norm_label(is_same),
            "correct_openalex_id": _safe_str(correct_id),
        }
        # Join predicted fields (from search jsonl)
        srec = by_pid.get(pid)
        best = srec.get("best") if isinstance(srec, dict) and isinstance(srec.get("best"), dict) else {}
        rec.update(
            {
                "pred_best_is_confident": bool(srec.get("best_is_confident")) if isinstance(srec, dict) else False,
                "pred_best_is_top3_venue": bool(best.get("is_top3_venue")) if isinstance(best, dict) else False,
                "pred_best_title_sim_token_f1": _safe_float(best.get("title_sim_token_f1")) if isinstance(best, dict) else None,
                "pred_best_author_overlap": _safe_float(best.get("author_overlap")) if isinstance(best, dict) else None,
                "pred_best_openalex_id": _safe_str(best.get("openalex_id")) if isinstance(best, dict) else None,
            }
        )

        # Top-10 recall check if correct id provided
        top10 = srec.get("candidates_top10") if isinstance(srec, dict) and isinstance(srec.get("candidates_top10"), list) else []
        top10_ids = [str(x.get("openalex_id")) for x in top10 if isinstance(x, dict) and isinstance(x.get("openalex_id"), str)]
        rec["correct_in_top10"] = bool(rec["correct_openalex_id"] and rec["correct_openalex_id"] in top10_ids)
        label_rows.append(rec)

    df = pd.DataFrame(label_rows)
    if df.empty:
        out_md.write_text(
            "\n".join(
                [
                    "# Publication mapping audit report (no labels found)",
                    "",
                    f"> Generated: {_now_iso()}",
                    "",
                    f"- labels-dir: `{labels_dir}`",
                    f"- search-jsonl: `{search_path}`",
                    "",
                    "No label JSON files were found. Generate tasks with `scripts/make_publication_mapping_audit_tasks.py` and fill `labels.is_same_work.value` in the label JSONs.",
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        out_json.write_text(
            json.dumps({"generated_at": _now_iso(), "n_labels": 0, "message": "no labels found"}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[{_now_iso()}] wrote: {out_md}")
        print(f"[{_now_iso()}] wrote: {out_json}")
        return 0

    def is_completed(r: pd.Series) -> bool:
        return isinstance(r.get("is_same_work"), str) and r.get("is_same_work").strip() != ""

    df["is_completed"] = df.apply(is_completed, axis=1)
    completed = df[df["is_completed"]].copy()

    annotator_counts = completed["annotator"].astype(str).value_counts().to_dict() if not completed.empty else {}
    annotator_counts = {str(k): int(v) for k, v in annotator_counts.items()}
    n_unique_annotators_completed = int(len(annotator_counts))
    any_ai_annotator = any(_looks_like_ai_annotator(k) for k in annotator_counts.keys())

    # Accuracy (excluding unsure)
    def acc(sub: pd.DataFrame) -> dict[str, Any]:
        if sub.empty:
            return {"n": 0, "n_yes": 0, "n_no": 0, "n_unsure": 0, "accuracy_yes_no": None}
        vals = sub["is_same_work"].astype(str).str.lower()
        n_yes = int((vals == "yes").sum())
        n_no = int((vals == "no").sum())
        n_unsure = int((vals == "unsure").sum())
        denom = n_yes + n_no
        return {
            "n": int(len(sub)),
            "n_yes": n_yes,
            "n_no": n_no,
            "n_unsure": n_unsure,
            "accuracy_yes_no": (float(n_yes / denom) if denom > 0 else None),
        }

    overall = acc(completed)

    # Subgroup accuracy
    by_conf = {
        "confident_true": acc(completed[completed["pred_best_is_confident"].astype(bool)]),
        "confident_false": acc(completed[~completed["pred_best_is_confident"].astype(bool)]),
    }
    by_top3 = {
        "top3_true": acc(completed[completed["pred_best_is_top3_venue"].astype(bool)]),
        "top3_false": acc(completed[~completed["pred_best_is_top3_venue"].astype(bool)]),
    }

    completed["token_f1_bin"] = completed["pred_best_title_sim_token_f1"].apply(lambda x: _bin(_safe_float(x)))
    by_bin = {str(k): acc(v) for k, v in completed.groupby("token_f1_bin", dropna=False)}

    # IRR (pairwise Cohen's kappa) if multi-annotated
    irr = {"pairwise_kappa": []}
    if int(completed["paper_id"].nunique()) > 0:
        annotators = sorted(set(completed["annotator"].astype(str).tolist()))
        per_ann: dict[str, dict[str, str]] = {}
        for ann in annotators:
            sub = completed[completed["annotator"] == ann]
            m: dict[str, str] = {}
            for r in sub.itertuples(index=False):
                pid = getattr(r, "paper_id", None)
                lab = getattr(r, "is_same_work", None)
                if isinstance(pid, str) and isinstance(lab, str) and lab.strip():
                    m[pid] = lab.strip().lower()
            per_ann[ann] = m

        for a1, a2 in itertools.combinations(annotators, 2):
            m1 = per_ann.get(a1, {})
            m2 = per_ann.get(a2, {})
            common = sorted(set(m1.keys()) & set(m2.keys()))
            if len(common) < 8:
                continue
            v1 = [m1[pid] for pid in common]
            v2 = [m2[pid] for pid in common]
            k = _cohen_kappa(v1, v2)
            irr["pairwise_kappa"].append({"a1": a1, "a2": a2, "n_common": int(len(common)), "kappa": k})

    # Optional join with offline risk score
    if args.panel:
        panel_path = Path(args.panel)
        if panel_path.exists():
            p = pd.read_csv(panel_path)
            if "paper_id" in p.columns and "offline_risk_score" in p.columns:
                s = p[["paper_id", "offline_risk_score"]].copy()
                s["offline_risk_score"] = pd.to_numeric(s["offline_risk_score"], errors="coerce")
                completed = completed.merge(s, on="paper_id", how="left")

    # Report
    pop_stats = {
        "n_total": int(len(pop)) if not pop.empty else 0,
        "n_ok": int(len(pop_ok)) if not pop_ok.empty else 0,
        "has_any_candidate_share": float((pop_ok["n_candidates_non_ssrn"] > 0).mean()) if "n_candidates_non_ssrn" in pop_ok.columns and not pop_ok.empty else None,
        "confident_share": float(pop_ok["best_is_confident"].astype(bool).mean()) if "best_is_confident" in pop_ok.columns and not pop_ok.empty else None,
        "top3_share": float(pop_ok["best_is_top3_venue"].astype(bool).mean()) if "best_is_top3_venue" in pop_ok.columns and not pop_ok.empty else None,
    }

    report = {
        "generated_at": _now_iso(),
        "inputs": {"labels_dir": str(labels_dir), "search_jsonl": str(search_path), "panel": args.panel},
        "population": pop_stats,
        "labels": {
            "n_files": int(len(df)),
            "n_completed": int(len(completed)),
            "n_unique_papers_completed": int(completed["paper_id"].nunique()) if not completed.empty else 0,
            "n_unique_annotators_completed": n_unique_annotators_completed,
            "annotator_counts_completed": annotator_counts,
            "warning_labels_may_be_ai_generated": bool(any_ai_annotator),
        },
        "accuracy": {"overall": overall, "by_confidence": by_conf, "by_top3": by_top3, "by_title_token_f1_bin": by_bin},
        "irr": irr,
    }

    # Markdown
    lines: list[str] = []
    lines.append("# Publication mapping audit report (prototype)")
    lines.append("")
    lines.append(f"> Generated: {_now_iso()}")
    lines.append("")
    lines.append("## Population (search-based map)")
    lines.append("")
    lines.append(f"- search-jsonl: `{search_path.name}`")
    lines.append(f"- N total records: {pop_stats['n_total']}")
    lines.append(f"- N status==ok: {pop_stats['n_ok']}")
    if pop_stats["has_any_candidate_share"] is not None:
        lines.append(f"- Has any non-SSRN candidate (share): {pop_stats['has_any_candidate_share']:.3f}")
    if pop_stats["confident_share"] is not None:
        lines.append(f"- Confident best match (share): {pop_stats['confident_share']:.3f}")
    if pop_stats["top3_share"] is not None:
        lines.append(f"- Top-3 venue best match (share): {pop_stats['top3_share']:.3f}")
    lines.append("")
    lines.append("## Labeled sample")
    lines.append("")
    lines.append(f"- labels-dir: `{labels_dir}`")
    lines.append(f"- N label files: {report['labels']['n_files']}")
    lines.append(f"- N completed: {report['labels']['n_completed']}")
    lines.append(f"- N unique papers (completed): {report['labels']['n_unique_papers_completed']}")
    lines.append(f"- N unique annotators (completed): {report['labels']['n_unique_annotators_completed']}")
    if report["labels"].get("warning_labels_may_be_ai_generated"):
        lines.append("- Warning: annotator name(s) suggest AI/auto-generated labels; treat as *draft* and not a substitute for independent human audit.")
    lines.append("")

    if annotator_counts:
        lines.append("### Annotators (completed labels)")
        lines.append("")
        lines.append("| annotator | n |")
        lines.append("|---|---:|")
        for k in sorted(annotator_counts.keys(), key=lambda x: (-annotator_counts[x], x)):
            lines.append(f"| {k} | {annotator_counts[k]} |")
        lines.append("")
    lines.append("")

    def _acc_block(name: str, a: dict[str, Any]) -> None:
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- N: {a.get('n')}")
        lines.append(f"- yes/no/unsure: {a.get('n_yes')}/{a.get('n_no')}/{a.get('n_unsure')}")
        v = a.get("accuracy_yes_no")
        if isinstance(v, (int, float)):
            lines.append(f"- Accuracy (yes among yes/no): {float(v):.3f}")
        else:
            lines.append("- Accuracy (yes among yes/no): null")
        lines.append("")

    lines.append("## Accuracy")
    lines.append("")
    _acc_block("Overall", overall)
    _acc_block("Predicted confident=True", by_conf["confident_true"])
    _acc_block("Predicted confident=False", by_conf["confident_false"])
    _acc_block("Predicted top3=True", by_top3["top3_true"])
    _acc_block("Predicted top3=False", by_top3["top3_false"])

    lines.append("### By title token-F1 bin")
    lines.append("")
    lines.append("| bin | n | yes | no | unsure | acc_yes_no |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for b in sorted(by_bin.keys()):
        a = by_bin[b]
        av = a.get("accuracy_yes_no")
        lines.append(
            f"| {b} | {a.get('n',0)} | {a.get('n_yes',0)} | {a.get('n_no',0)} | {a.get('n_unsure',0)} | {(f'{float(av):.3f}' if isinstance(av,(int,float)) else 'null')} |"
        )
    lines.append("")

    if irr["pairwise_kappa"]:
        lines.append("## Inter-rater reliability (pairwise Cohen's kappa)")
        lines.append("")
        lines.append("| a1 | a2 | n_common | kappa |")
        lines.append("|---|---|---:|---:|")
        for r in irr["pairwise_kappa"]:
            k = r.get("kappa")
            ks = f"{float(k):.3f}" if isinstance(k, (int, float)) else "null"
            lines.append(f"| {r.get('a1')} | {r.get('a2')} | {int(r.get('n_common') or 0)} | {ks} |")
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[{_now_iso()}] wrote: {out_md}")
    print(f"[{_now_iso()}] wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
