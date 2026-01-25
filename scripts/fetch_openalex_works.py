#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


OPENALEX_BASE = "https://api.openalex.org/works/"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        }
    )
    return s


def _sleep_polite(seconds: float) -> None:
    if seconds and seconds > 0:
        time.sleep(float(seconds))


def _read_done(jsonl_path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not jsonl_path.exists():
        return out
    for line in jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        pid = obj.get("paper_id")
        if isinstance(pid, str) and pid.strip():
            out[pid.strip()] = obj
    return out


def _select_fields(work: dict[str, Any]) -> dict[str, Any]:
    primary_location = work.get("primary_location") if isinstance(work.get("primary_location"), dict) else {}
    primary_source = primary_location.get("source") if isinstance(primary_location.get("source"), dict) else {}
    return {
        "openalex_id": work.get("id"),
        "openalex_doi_url": work.get("doi"),
        "title": work.get("title"),
        "publication_year": work.get("publication_year"),
        "publication_date": work.get("publication_date"),
        "type": work.get("type"),
        "cited_by_count": work.get("cited_by_count"),
        "counts_by_year": work.get("counts_by_year"),
        "related_works": work.get("related_works"),
        "primary_source": primary_source.get("display_name"),
        "primary_source_type": primary_source.get("type"),
        "is_retracted": work.get("is_retracted"),
        "is_paratext": work.get("is_paratext"),
        "updated_date": work.get("updated_date"),
        "created_date": work.get("created_date"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch OpenAlex work metadata for a panel of SSRN DOIs.")
    ap.add_argument("--panel", default="analysis/paper_panel.csv", help="Panel CSV with paper_id and doi.")
    ap.add_argument("--out-jsonl", default=None, help="Output JSONL path (default: <panel_dir>/openalex_works.jsonl).")
    ap.add_argument("--out-csv", default=None, help="Output CSV path (default: <panel_dir>/openalex_works.csv).")
    ap.add_argument("--sleep-s", type=float, default=0.2, help="Polite sleep between requests.")
    ap.add_argument("--max-papers", type=int, default=None, help="Optional cap (for debugging).")
    ap.add_argument("--force", action="store_true", help="Re-fetch even if output already contains paper_id.")
    args = ap.parse_args()

    panel_path = Path(args.panel)
    if not panel_path.exists():
        raise SystemExit(f"Missing: {panel_path}")
    out_dir = panel_path.parent
    out_jsonl = Path(args.out_jsonl) if args.out_jsonl else (out_dir / "openalex_works.jsonl")
    out_csv = Path(args.out_csv) if args.out_csv else (out_dir / "openalex_works.csv")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    mailto = os.environ.get("OPENALEX_MAILTO") or os.environ.get("CROSSREF_MAILTO")
    session = _make_session()

    panel = pd.read_csv(panel_path)
    if "paper_id" not in panel.columns or "doi" not in panel.columns:
        raise SystemExit("panel must include columns: paper_id, doi")
    panel = panel[panel["paper_id"].notna() & panel["doi"].notna()].copy()
    if args.max_papers is not None:
        panel = panel.head(int(args.max_papers)).copy()

    done = _read_done(out_jsonl) if (out_jsonl.exists() and not args.force) else {}
    rows_out: list[dict[str, Any]] = []

    mode = "a" if out_jsonl.exists() and not args.force else "w"
    if mode == "w":
        done = {}

    n_total = int(len(panel))
    fetched = 0
    skipped = 0
    errors = 0

    with out_jsonl.open(mode, encoding="utf-8") as f:
        for i, row in enumerate(panel.itertuples(index=False), start=1):
            pid = str(getattr(row, "paper_id"))
            doi = str(getattr(row, "doi"))
            if not pid or not doi:
                continue
            if pid in done and not args.force:
                skipped += 1
                # also collect for csv
                obj = done[pid]
                sel = obj.get("openalex") if isinstance(obj.get("openalex"), dict) else {}
                rows_out.append({"paper_id": pid, "doi": doi, "status": obj.get("status"), **(sel if isinstance(sel, dict) else {})})
                continue

            url = OPENALEX_BASE + "https://doi.org/" + doi
            params = {
                "mailto": mailto,
                "select": (
                    "id,doi,title,publication_year,publication_date,type,cited_by_count,counts_by_year,"
                    "primary_location,related_works,is_retracted,is_paratext,updated_date,created_date"
                ),
            }
            if not mailto:
                params.pop("mailto", None)
            try:
                r = session.get(url, params=params, timeout=60)
                if r.status_code == 404:
                    rec = {"paper_id": pid, "doi": doi, "status": "not_found", "openalex": None, "error": "404"}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    rows_out.append({"paper_id": pid, "doi": doi, "status": "not_found"})
                    errors += 1
                else:
                    r.raise_for_status()
                    work = r.json()
                    sel = _select_fields(work if isinstance(work, dict) else {})
                    rec = {
                        "paper_id": pid,
                        "doi": doi,
                        "status": "ok",
                        "openalex": sel,
                        "fetched_at": _now_iso(),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    rows_out.append({"paper_id": pid, "doi": doi, "status": "ok", **sel})
                    fetched += 1
            except Exception as e:
                rec = {
                    "paper_id": pid,
                    "doi": doi,
                    "status": "error",
                    "openalex": None,
                    "error": f"{type(e).__name__}: {e}",
                    "fetched_at": _now_iso(),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rows_out.append({"paper_id": pid, "doi": doi, "status": "error", "error": rec["error"]})
                errors += 1

            if i % 25 == 0:
                print(f"[{_now_iso()}] ({i}/{n_total}) fetched={fetched} skipped={skipped} errors={errors}")
            _sleep_polite(float(args.sleep_s))

    pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    print(f"[{_now_iso()}] wrote: {out_jsonl}")
    print(f"[{_now_iso()}] wrote: {out_csv}")
    print(f"[{_now_iso()}] done: fetched={fetched} skipped={skipped} errors={errors} total={n_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
