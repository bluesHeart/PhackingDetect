#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import difflib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


OPENALEX_WORKS = "https://api.openalex.org/works/"


TOP_VENUES = {
    "the journal of finance",
    "journal of financial economics",
    "the review of financial studies",
    "review of financial studies",
}


STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


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


def _parse_list_cell(x: Any) -> list[str]:
    if isinstance(x, list):
        return [str(v) for v in x if isinstance(v, str) and v.strip()]
    if isinstance(x, str) and x.strip().startswith("["):
        try:
            v = ast.literal_eval(x)
        except Exception:
            try:
                v = json.loads(x)
            except Exception:
                return []
        if isinstance(v, list):
            return [str(z) for z in v if isinstance(z, str) and str(z).strip()]
    return []


def _norm_title(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    t = s.strip().lower()
    if not t:
        return ""
    t = re.sub(r"[^0-9a-z]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _title_tokens(s: Any) -> set[str]:
    t = _norm_title(s)
    if not t:
        return set()
    toks = [x for x in t.split(" ") if x and x not in STOPWORDS]
    return set(toks)


def _title_similarity(a: Any, b: Any) -> dict[str, float]:
    na = _norm_title(a)
    nb = _norm_title(b)
    if not na or not nb:
        return {"title_sim_seq": 0.0, "title_sim_jaccard": 0.0, "title_sim_token_f1": 0.0}
    seq = difflib.SequenceMatcher(None, na, nb).ratio()
    ta = _title_tokens(na)
    tb = _title_tokens(nb)
    if not ta or not tb:
        return {"title_sim_seq": float(seq), "title_sim_jaccard": 0.0, "title_sim_token_f1": 0.0}
    inter = len(ta & tb)
    union = len(ta | tb)
    j = (inter / union) if union else 0.0
    f1 = (2.0 * inter / (len(ta) + len(tb))) if (len(ta) + len(tb)) else 0.0
    return {"title_sim_seq": float(seq), "title_sim_jaccard": float(j), "title_sim_token_f1": float(f1)}


def _openalex_id_short(url: str) -> str:
    s = (url or "").strip()
    if not s:
        return ""
    if s.startswith("https://openalex.org/"):
        return s.rsplit("/", 1)[-1]
    return s


def _select_related_fields(work: dict[str, Any]) -> dict[str, Any]:
    primary_location = work.get("primary_location") if isinstance(work.get("primary_location"), dict) else {}
    source = primary_location.get("source") if isinstance(primary_location.get("source"), dict) else {}
    return {
        "openalex_id": work.get("id"),
        "doi_url": work.get("doi"),
        "title": work.get("title"),
        "publication_year": work.get("publication_year"),
        "type": work.get("type"),
        "cited_by_count": work.get("cited_by_count"),
        "primary_source": source.get("display_name"),
        "primary_source_type": source.get("type"),
        "updated_date": work.get("updated_date"),
    }


def _is_top_venue(name: Any) -> bool:
    if not isinstance(name, str):
        return False
    s = name.strip().lower()
    if not s:
        return False
    return s in TOP_VENUES


def _is_ssrn_source(name: Any) -> bool:
    if not isinstance(name, str):
        return False
    s = name.strip().lower()
    return s == "ssrn electronic journal"


def _safe_int(x: Any) -> int:
    try:
        if isinstance(x, bool):
            return 0
        if isinstance(x, int):
            return int(x)
        if isinstance(x, float):
            return 0 if x != x else int(x)
        if isinstance(x, str) and x.strip():
            return int(float(x.strip()))
    except Exception:
        return 0
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch OpenAlex metadata for related_works and map preprints to likely published versions.")
    ap.add_argument("--openalex-works", default="analysis/openalex_works.csv", help="CSV from fetch_openalex_works.py.")
    ap.add_argument("--out-related", default=None, help="Output CSV of fetched related works (default: <dir>/openalex_related_works.csv).")
    ap.add_argument("--out-map", default=None, help="Output per-paper mapping CSV (default: <dir>/paper_openalex_publication_map.csv).")
    ap.add_argument("--sleep-s", type=float, default=0.15)
    ap.add_argument("--max-related-per-paper", type=int, default=10)
    ap.add_argument("--max-related-total", type=int, default=None, help="Optional cap on unique related works to fetch.")
    ap.add_argument("--min-title-sim", type=float, default=0.90, help="Min title similarity (SequenceMatcher) for a confident match.")
    ap.add_argument("--min-token-f1", type=float, default=0.85, help="Min token-F1 title similarity for a confident match.")
    ap.add_argument("--force", action="store_true", help="Ignore any existing related-work cache and refetch.")
    args = ap.parse_args()

    works_path = Path(args.openalex_works)
    if not works_path.exists():
        raise SystemExit(f"Missing: {works_path}")
    out_dir = works_path.parent
    out_related = Path(args.out_related) if args.out_related else (out_dir / "openalex_related_works.csv")
    out_map = Path(args.out_map) if args.out_map else (out_dir / "paper_openalex_publication_map.csv")
    out_related.parent.mkdir(parents=True, exist_ok=True)
    out_map.parent.mkdir(parents=True, exist_ok=True)

    mailto = os.environ.get("OPENALEX_MAILTO") or os.environ.get("CROSSREF_MAILTO")
    session = _make_session()

    df = pd.read_csv(works_path)
    if "paper_id" not in df.columns or "related_works" not in df.columns:
        raise SystemExit("openalex works csv must include columns: paper_id, related_works")
    df = df[df.get("status").astype(str).eq("ok")].copy() if "status" in df.columns else df.copy()

    per_paper_related: dict[str, list[str]] = {}
    per_paper_title: dict[str, str] = {}
    all_related: list[str] = []
    for r in df.itertuples(index=False):
        pid = getattr(r, "paper_id", None)
        rel = getattr(r, "related_works", None)
        if not isinstance(pid, str) or not pid.strip():
            continue
        t = getattr(r, "title", None)
        if isinstance(t, str) and t.strip():
            per_paper_title[pid] = t.strip()
        rel_list = _parse_list_cell(rel)[: int(args.max_related_per_paper)]
        rel_ids = [_openalex_id_short(x) for x in rel_list if _openalex_id_short(x)]
        per_paper_related[pid] = rel_ids
        all_related.extend(rel_ids)

    uniq_related = sorted(set(all_related))
    if args.max_related_total is not None:
        uniq_related = uniq_related[: int(args.max_related_total)]

    # Fetch related works (with caching in-memory)
    selection_keys = [
        "openalex_id",
        "doi_url",
        "title",
        "publication_year",
        "type",
        "cited_by_count",
        "primary_source",
        "primary_source_type",
        "updated_date",
    ]
    cache: dict[str, dict[str, Any]] = {}
    rows_by_work_id: dict[str, dict[str, Any]] = {}

    if out_related.exists() and not args.force:
        try:
            existing = pd.read_csv(out_related)
            for rec in existing.to_dict(orient="records"):
                wid = str(rec.get("work_id") or "").strip()
                if not wid:
                    continue
                rows_by_work_id[wid] = dict(rec)
                if str(rec.get("status") or "").strip() == "ok":
                    cache[wid] = {k: rec.get(k) for k in selection_keys}
        except Exception:
            # If cache read fails, proceed without it.
            cache = {}
            rows_by_work_id = {}

    fetched = 0
    errors = 0
    to_fetch = [wid for wid in uniq_related if wid not in cache]
    for i, wid in enumerate(to_fetch, start=1):
        url = OPENALEX_WORKS + wid
        params = {
            "mailto": mailto,
            "select": "id,doi,title,publication_year,type,cited_by_count,primary_location,updated_date",
        }
        if not mailto:
            params.pop("mailto", None)
        try:
            r = session.get(url, params=params, timeout=60)
            r.raise_for_status()
            w = r.json()
            sel = _select_related_fields(w if isinstance(w, dict) else {})
            cache[wid] = sel
            rows_by_work_id[wid] = {"work_id": wid, "status": "ok", **sel}
            fetched += 1
        except Exception as e:
            rows_by_work_id[wid] = {"work_id": wid, "status": "error", "error": f"{type(e).__name__}: {e}"}
            errors += 1
        if i % 200 == 0:
            print(f"[{_now_iso()}] ({i}/{len(to_fetch)}) fetched={fetched} errors={errors}")
        _sleep_polite(float(args.sleep_s))

    df_related = pd.DataFrame(list(rows_by_work_id.values()))
    if not df_related.empty and "work_id" in df_related.columns:
        df_related = df_related.sort_values("work_id")
    df_related.to_csv(out_related, index=False)

    # Per-paper mapping
    map_rows: list[dict[str, Any]] = []
    for pid, rel_ids in per_paper_related.items():
        pre_title = per_paper_title.get(pid, "")
        candidates: list[dict[str, Any]] = []
        for wid in rel_ids:
            rec = cache.get(wid)
            if not isinstance(rec, dict):
                continue
            src = rec.get("primary_source")
            if _is_ssrn_source(src):
                continue
            sim = _title_similarity(pre_title, rec.get("title"))
            candidates.append({"work_id": wid, **rec, **sim})

        # Determine best match: prioritize high title similarity, then venue/citations.
        min_seq = float(args.min_title_sim)
        min_f1 = float(args.min_token_f1)
        confident = [c for c in candidates if float(c.get("title_sim_seq") or 0) >= min_seq and float(c.get("title_sim_token_f1") or 0) >= min_f1]
        top_conf = [c for c in confident if _is_top_venue(c.get("primary_source"))]
        best: dict[str, Any] | None = None
        reason = None

        def score_key(c: dict[str, Any]) -> tuple[float, float, int, int]:
            seq = float(c.get("title_sim_seq") or 0.0)
            f1 = float(c.get("title_sim_token_f1") or 0.0)
            cites_i = _safe_int(c.get("cited_by_count"))
            year_i = _safe_int(c.get("publication_year"))
            return (seq, f1, cites_i, year_i)

        if top_conf:
            best = sorted(top_conf, key=score_key, reverse=True)[0]
            reason = "top_venue_confident"
        elif confident:
            best = sorted(confident, key=score_key, reverse=True)[0]
            reason = "non_ssrn_confident"
        elif candidates:
            best = sorted(candidates, key=score_key, reverse=True)[0]
            reason = "non_ssrn_best_effort"

        map_rows.append(
            {
                "paper_id": pid,
                "preprint_title": pre_title or None,
                "n_related_considered": int(len(rel_ids)),
                "n_non_ssrn_candidates": int(len(candidates)),
                "has_non_ssrn_candidate": bool(len(candidates) > 0),
                "n_confident_candidates": int(len(confident)),
                "has_top3_candidate": bool(any(_is_top_venue(c.get("primary_source")) for c in candidates)),
                "best_match_reason": reason,
                "best_match_is_confident": bool(
                    best
                    and float(best.get("title_sim_seq") or 0) >= min_seq
                    and float(best.get("title_sim_token_f1") or 0) >= min_f1
                ),
                "best_match_is_top3_venue": bool(best and _is_top_venue(best.get("primary_source"))),
                "best_related_work_id": (best.get("work_id") if isinstance(best, dict) else None),
                "best_related_title_sim_seq": (best.get("title_sim_seq") if isinstance(best, dict) else None),
                "best_related_title_sim_token_f1": (best.get("title_sim_token_f1") if isinstance(best, dict) else None),
                "best_related_title_sim_jaccard": (best.get("title_sim_jaccard") if isinstance(best, dict) else None),
                "best_related_openalex_id": (best.get("openalex_id") if isinstance(best, dict) else None),
                "best_related_doi_url": (best.get("doi_url") if isinstance(best, dict) else None),
                "best_related_title": (best.get("title") if isinstance(best, dict) else None),
                "best_related_publication_year": (best.get("publication_year") if isinstance(best, dict) else None),
                "best_related_type": (best.get("type") if isinstance(best, dict) else None),
                "best_related_cited_by_count": (best.get("cited_by_count") if isinstance(best, dict) else None),
                "best_related_primary_source": (best.get("primary_source") if isinstance(best, dict) else None),
            }
        )

    pd.DataFrame(map_rows).to_csv(out_map, index=False)

    print(f"[{_now_iso()}] wrote: {out_related}")
    print(f"[{_now_iso()}] wrote: {out_map}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
