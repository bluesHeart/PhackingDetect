#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _safe_int(x: Any) -> int | None:
    try:
        if isinstance(x, bool):
            return None
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            if x != x:
                return None
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
            v = float(x)
            return None if v != v else v
        if isinstance(x, str) and x.strip():
            v = float(x.strip())
            return None if v != v else v
    except Exception:
        return None
    return None


def _norm_title(s: Any) -> str | None:
    if not isinstance(s, str):
        return None
    t = s.strip().lower()
    if not t or len(t) < 6:
        return None
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[\"'“”‘’`]", "", t)
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) < 6:
        return None
    return t


def _first_author_family(authors: Any) -> str | None:
    if not isinstance(authors, str):
        return None
    s = authors.strip()
    if not s:
        return None
    # Common SSRN meta formats:
    #   "Chen, Yong; Zhang, Hanjiang"
    #   "Yong Chen; Hanjiang Zhang"
    first = s.split(";")[0].strip()
    if not first:
        return None
    # If "Family, Given"
    if "," in first:
        fam = first.split(",")[0].strip()
    else:
        fam = first.split()[-1].strip()
    fam = re.sub(r"[^a-zA-Z\- ]+", "", fam).strip().lower()
    fam = re.sub(r"\s+", " ", fam)
    return fam or None


def _hash_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="replace")).hexdigest()[:10]


def _pdf_bytes(corpus_dir: Path, pdf_relpath: Any) -> int | None:
    if not isinstance(pdf_relpath, str) or not pdf_relpath.strip():
        return None
    p = corpus_dir / pdf_relpath
    try:
        if p.exists() and p.is_file():
            return int(p.stat().st_size)
    except Exception:
        return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="De-duplicate a paper panel by normalized title (with ambiguity guard).")
    ap.add_argument("--panel", default="analysis/paper_panel.csv", help="Input panel CSV.")
    ap.add_argument("--corpus-dir", default="corpus", help="Corpus directory (used to read PDF file sizes).")
    ap.add_argument("--out-panel", default="analysis/paper_panel_dedup.csv", help="Output deduped panel CSV.")
    ap.add_argument("--out-map", default="analysis/paper_dedup_map.csv", help="Output mapping CSV.")
    ap.add_argument("--out-report", default="analysis/dedupe_report.md", help="Output report markdown.")
    args = ap.parse_args()

    panel_path = Path(args.panel)
    corpus_dir = Path(args.corpus_dir)
    out_panel = Path(args.out_panel)
    out_map = Path(args.out_map)
    out_report = Path(args.out_report)
    out_panel.parent.mkdir(parents=True, exist_ok=True)
    out_map.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    if not panel_path.exists():
        raise SystemExit(f"Missing: {panel_path}")

    df = pd.read_csv(panel_path)
    if "paper_id" not in df.columns:
        raise SystemExit("panel missing column: paper_id")

    df["norm_title"] = df["title"].apply(_norm_title) if "title" in df.columns else None
    df["first_author_family"] = df["authors"].apply(_first_author_family) if "authors" in df.columns else None
    df["pdf_bytes"] = [
        _pdf_bytes(corpus_dir, rp) for rp in (df["pdf_relpath"].tolist() if "pdf_relpath" in df.columns else [None] * len(df))
    ]

    # Only dedupe where we actually have a title.
    has_title = df["norm_title"].notna()
    groups = df[has_title].groupby("norm_title", dropna=True)

    keep_flags = pd.Series(False, index=df.index)
    group_ids: dict[str, str] = {}
    ambiguous_titles: list[str] = []

    for norm, sub in groups:
        if len(sub) <= 1:
            idx = int(sub.index[0])
            gid = f"title_{_hash_id(str(norm))}"
            group_ids[str(df.loc[idx, "paper_id"])] = gid
            keep_flags.loc[idx] = True
            continue

        fams = sorted({str(x) for x in sub["first_author_family"].dropna().tolist() if str(x).strip()})
        if len(fams) >= 2:
            # Ambiguous: same title appears with multiple distinct first authors.
            # Do NOT dedupe; keep all, but still assign deterministic group ids per paper.
            ambiguous_titles.append(str(norm))
            for idx in sub.index.tolist():
                pid = str(df.loc[idx, "paper_id"])
                group_ids[pid] = f"ambig_{_hash_id(str(norm) + '|' + pid)}"
                keep_flags.loc[idx] = True
            continue

        gid = f"title_{_hash_id(str(norm))}"
        # Choose best representative by extraction quality / evidence richness.
        sub2 = sub.copy()
        sub2["quality_text"] = pd.to_numeric(sub2.get("extracted_text_chars"), errors="coerce").fillna(0)
        sub2["quality_pairs"] = pd.to_numeric(sub2.get("t_pairs_seen"), errors="coerce").fillna(0)
        sub2["quality_pages"] = pd.to_numeric(sub2.get("pages"), errors="coerce").fillna(0)
        sub2["quality_bytes"] = pd.to_numeric(sub2.get("pdf_bytes"), errors="coerce").fillna(0)
        sub2 = sub2.sort_values(["quality_text", "quality_pairs", "quality_pages", "quality_bytes"], ascending=False)
        best_idx = int(sub2.index[0])

        for idx in sub.index.tolist():
            pid = str(df.loc[idx, "paper_id"])
            group_ids[pid] = gid
            if int(idx) == best_idx:
                keep_flags.loc[idx] = True

    # Rows without titles: keep (cannot dedupe safely).
    for idx in df[~has_title].index.tolist():
        pid = str(df.loc[idx, "paper_id"])
        group_ids[pid] = f"no_title_{_hash_id(pid)}"
        keep_flags.loc[idx] = True

    # Mapping
    map_rows: list[dict[str, Any]] = []
    kept_by_gid: dict[str, str] = {}
    for idx, row in df[keep_flags].iterrows():
        pid = str(row.get("paper_id"))
        kept_by_gid[group_ids.get(pid, "")] = pid

    for idx, row in df.iterrows():
        pid = str(row.get("paper_id"))
        gid = group_ids.get(pid) or f"unk_{_hash_id(pid)}"
        kept_pid = kept_by_gid.get(gid)
        map_rows.append(
            {
                "paper_id": pid,
                "dedup_group_id": gid,
                "is_kept": bool(keep_flags.loc[idx]),
                "kept_paper_id": kept_pid,
                "norm_title": row.get("norm_title"),
                "first_author_family": row.get("first_author_family"),
                "year": _safe_int(row.get("year")),
                "title": row.get("title"),
                "authors": row.get("authors"),
                "offline_risk_score": _safe_int(row.get("offline_risk_score")),
                "t_pairs_seen": _safe_int(row.get("t_pairs_seen")),
                "extracted_text_chars": _safe_int(row.get("extracted_text_chars")),
                "pdf_relpath": row.get("pdf_relpath"),
                "pdf_bytes": _safe_int(row.get("pdf_bytes")),
            }
        )

    map_df = pd.DataFrame(map_rows)
    map_df.to_csv(out_map, index=False)

    dedup_df = df[keep_flags].copy()
    dedup_df.to_csv(out_panel, index=False)

    # Report
    total = int(len(df))
    kept = int(keep_flags.sum())
    groups_total = int(len(set(group_ids.values())))
    n_ambig = int(len(ambiguous_titles))

    dup_groups = int((groups.size() > 1).sum())
    dup_papers = int((groups.size() - 1).clip(lower=0).sum())

    lines: list[str] = []
    lines.append("# De-duplication report (prototype)")
    lines.append("")
    lines.append(f"> Generated: {_now_iso()}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Input papers: {total}")
    lines.append(f"- Kept after dedupe: {kept}")
    lines.append(f"- Dedup groups (ids): {groups_total}")
    lines.append(f"- Duplicate-title groups (raw): {dup_groups}")
    lines.append(f"- Duplicate papers (raw, count-1): {dup_papers}")
    lines.append(f"- Ambiguous title groups (kept-all): {n_ambig}")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- Deduped panel: `{out_panel.name}`")
    lines.append(f"- Mapping: `{out_map.name}`")
    lines.append("")
    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[{_now_iso()}] wrote: {out_panel} (rows={len(dedup_df)})")
    print(f"[{_now_iso()}] wrote: {out_map} (rows={len(map_df)})")
    print(f"[{_now_iso()}] wrote: {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

