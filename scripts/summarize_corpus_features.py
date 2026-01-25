#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize corpus manifest + extracted within-paper metrics.")
    ap.add_argument("--corpus-dir", default="corpus")
    ap.add_argument("--out-md", default=None, help="Output markdown path (default: <corpus-dir>/summary.md).")
    args = ap.parse_args()

    corpus_dir = Path(args.corpus_dir)
    manifest_csv = corpus_dir / "manifest.csv"
    features_csv = corpus_dir / "features.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(manifest_csv)
    if not features_csv.exists():
        raise FileNotFoundError(features_csv)

    out_md = Path(args.out_md) if args.out_md else (corpus_dir / "summary.md")

    m = pd.read_csv(manifest_csv)
    f = pd.read_csv(features_csv)

    # Join on ssrn_id derived from paper_id
    f["ssrn_id"] = f["paper_id"].astype(str).str.replace("ssrn_", "", regex=False)
    if "ssrn_id" in m.columns:
        m["ssrn_id"] = m["ssrn_id"].astype(str)
    joined = f.merge(m, on="ssrn_id", how="left", suffixes=("", "_m"))

    downloaded = m["status"].astype(str).str.startswith("downloaded").sum() if "status" in m.columns else None
    exists = m["status"].astype(str).str.startswith("exists").sum() if "status" in m.columns else None
    download_errors = m["status"].astype(str).str.startswith("download_error").sum() if "status" in m.columns else None
    meta_errors = m["status"].astype(str).str.contains("meta_error").sum() if "status" in m.columns else None
    meta_skipped = m["status"].astype(str).str.contains("meta_skipped").sum() if "status" in m.columns else None

    top = joined.sort_values("offline_risk_score", ascending=False).head(20).copy()

    def j(v: Any) -> str:
        if isinstance(v, str) and (v.startswith("{") or v.startswith("[")):
            try:
                return json.dumps(json.loads(v), ensure_ascii=False)
            except Exception:
                return v
        return str(v)

    lines: list[str] = []
    lines.append(f"# Corpus Summary ({corpus_dir.name})")
    lines.append("")
    lines.append(f"> Generated: {_now_iso()}")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(f"- Manifest rows: {len(m)}")
    if downloaded is not None:
        lines.append(f"- Downloaded: {int(downloaded)}")
    if exists is not None:
        lines.append(f"- Already existed: {int(exists)}")
    if download_errors is not None:
        lines.append(f"- Download errors: {int(download_errors)}")
    if meta_errors is not None:
        lines.append(f"- Meta errors: {int(meta_errors)}")
    if meta_skipped is not None:
        lines.append(f"- Meta skipped: {int(meta_skipped)}")
    lines.append(f"- Features rows: {len(f)}")
    lines.append("")

    lines.append("## Risk Score Distribution (offline baseline)")
    lines.append("")
    if "offline_risk_score" in f.columns:
        desc = f["offline_risk_score"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
        keys = ["min", "10%", "25%", "50%", "75%", "90%", "max", "mean"]
        parts = [f"{k}={float(desc[k]):.2f}" for k in keys if k in desc and desc[k] == desc[k]]
        lines.append("- " + ", ".join(parts))
    lines.append("")

    lines.append("## Top Papers by Offline Risk Score")
    lines.append("")
    lines.append("| rank | ssrn_id | year | score | level | title | ssrn_url |")
    lines.append("|---:|---:|---:|---:|---|---|---|")
    for idx, row in enumerate(top.itertuples(index=False), start=1):
        ssrn_id = getattr(row, "ssrn_id", "")
        year = getattr(row, "year", "")
        score = getattr(row, "offline_risk_score", "")
        level = getattr(row, "offline_risk_level", "")
        title = getattr(row, "title", "") or getattr(row, "crossref_title", "")
        ssrn_url = getattr(row, "ssrn_url", "")
        title_s = str(title).replace("|", "\\|") if isinstance(title, str) else ""
        ssrn_url_s = str(ssrn_url).replace("|", "\\|") if isinstance(ssrn_url, str) else ""
        lines.append(f"| {idx} | {ssrn_id} | {year} | {score} | {level} | {title_s} | {ssrn_url_s} |")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- This is an *offline*, scalable baseline: it does **not** claim true p-hacking; it flags where to audit.")
    lines.append("- Extraction is best-effort; use `t_pairs_seen`, `tables_seen`, and provenance samples to assess coverage.")
    lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{_now_iso()}] wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
