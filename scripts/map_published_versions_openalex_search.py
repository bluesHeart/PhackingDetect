#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


OPENALEX_WORKS = "https://api.openalex.org/works"


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


def _sleep_polite(seconds: float) -> None:
    if seconds and seconds > 0:
        time.sleep(float(seconds))


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


def _is_top_venue(name: Any) -> bool:
    if not isinstance(name, str):
        return False
    s = name.strip().lower()
    return s in TOP_VENUES


def _is_ssrn_record(*, doi_url: Any, primary_source: Any) -> bool:
    if isinstance(primary_source, str) and primary_source.strip().lower() == "ssrn electronic journal":
        return True
    if isinstance(doi_url, str) and doi_url.strip().lower().startswith("https://doi.org/10.2139/ssrn"):
        return True
    return False


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


def _token_f1(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    denom = len(a) + len(b)
    return float((2.0 * inter / denom) if denom else 0.0)


def _title_similarity(a: Any, b: Any) -> dict[str, float]:
    na = _norm_title(a)
    nb = _norm_title(b)
    if not na or not nb:
        return {"title_sim_token_f1": 0.0, "title_sim_jaccard": 0.0}
    ta = _title_tokens(na)
    tb = _title_tokens(nb)
    if not ta or not tb:
        return {"title_sim_token_f1": 0.0, "title_sim_jaccard": 0.0}
    inter = len(ta & tb)
    union = len(ta | tb)
    j = (inter / union) if union else 0.0
    return {"title_sim_token_f1": _token_f1(ta, tb), "title_sim_jaccard": float(j)}


def _parse_preprint_lastnames(authors: Any) -> set[str]:
    if not isinstance(authors, str):
        return set()
    out: set[str] = set()
    for part in authors.split(";"):
        p = part.strip()
        if not p:
            continue
        last = p.split(",", 1)[0].strip()
        last = re.sub(r"[^a-zA-Z\- ]+", "", last).strip().lower()
        if last:
            out.add(last)
    return out


def _lastname_from_display_name(name: Any) -> str:
    if not isinstance(name, str):
        return ""
    s = name.strip()
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    parts = s.split(" ")
    if not parts:
        return ""
    last = parts[-1].strip()
    last = re.sub(r"[^a-zA-Z\-]+", "", last).strip().lower()
    return last


def _candidate_lastnames(authorships: Any) -> set[str]:
    if not isinstance(authorships, list):
        return set()
    out: set[str] = set()
    for a in authorships:
        if not isinstance(a, dict):
            continue
        author = a.get("author")
        if not isinstance(author, dict):
            continue
        dn = author.get("display_name")
        last = _lastname_from_display_name(dn)
        if last:
            out.add(last)
    return out


def _author_overlap(pre: set[str], cand: set[str]) -> float:
    if not pre or not cand:
        return 0.0
    inter = len(pre & cand)
    denom = min(len(pre), len(cand))
    return float(inter / denom) if denom else 0.0


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


def _get_json_with_retries(session: requests.Session, url: str, *, params: dict[str, Any], max_tries: int = 4) -> dict[str, Any]:
    last_err: Exception | None = None
    for attempt in range(int(max_tries)):
        try:
            r = session.get(url, params=params, timeout=60)
            if r.status_code in {429, 500, 502, 503, 504}:
                _sleep_polite(0.8 * (2**attempt))
                continue
            r.raise_for_status()
            obj = r.json()
            return obj if isinstance(obj, dict) else {}
        except Exception as e:
            last_err = e
            _sleep_polite(0.8 * (2**attempt))
    raise RuntimeError(str(last_err) if last_err else "unknown error")


def _select_candidate_fields(work: dict[str, Any]) -> dict[str, Any]:
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
        "authorships": work.get("authorships"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Map SSRN preprints to likely published versions using OpenAlex title search.")
    ap.add_argument("--panel", default="analysis/paper_panel.csv", help="Paper-level panel CSV (paper_id, title, authors, doi).")
    ap.add_argument(
        "--out-map",
        default=None,
        help="Output CSV (default: <panel_dir>/paper_openalex_publication_map_search.csv).",
    )
    ap.add_argument(
        "--out-jsonl",
        default=None,
        help="Output JSONL cache (default: <panel_dir>/openalex_search_publication_map.jsonl).",
    )
    ap.add_argument("--per-page", type=int, default=25, help="OpenAlex search results per paper.")
    ap.add_argument("--sleep-s", type=float, default=0.12, help="Polite sleep between requests.")
    ap.add_argument("--max-papers", type=int, default=None, help="Optional cap for debugging.")
    ap.add_argument("--min-title-sim-token-f1", type=float, default=0.85, help="Confidence threshold on token-F1 title similarity.")
    ap.add_argument("--min-author-overlap", type=float, default=0.25, help="Confidence threshold on last-name overlap.")
    ap.add_argument("--force", action="store_true", help="Refetch even if JSONL already contains paper_id.")
    args = ap.parse_args()

    panel_path = Path(args.panel)
    if not panel_path.exists():
        raise SystemExit(f"Missing: {panel_path}")
    out_dir = panel_path.parent
    out_map = Path(args.out_map) if args.out_map else (out_dir / "paper_openalex_publication_map_search.csv")
    out_jsonl = Path(args.out_jsonl) if args.out_jsonl else (out_dir / "openalex_search_publication_map.jsonl")
    out_map.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(panel_path)
    for c in ["paper_id", "title"]:
        if c not in df.columns:
            raise SystemExit(f"panel missing column: {c}")
    if args.max_papers is not None:
        df = df.head(int(args.max_papers)).copy()

    done: set[str] = set()
    if out_jsonl.exists() and not args.force:
        for line in out_jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("paper_id"), str):
                done.add(obj["paper_id"])

    mailto = os.environ.get("OPENALEX_MAILTO") or os.environ.get("CROSSREF_MAILTO")
    session = _make_session()

    n = int(len(df))
    fetched = 0
    skipped = 0
    errors = 0

    mode = "a" if out_jsonl.exists() and not args.force else "w"
    if mode == "w":
        done = set()

    with out_jsonl.open(mode, encoding="utf-8") as f:
        for i, r in enumerate(df.itertuples(index=False), start=1):
            paper_id = str(getattr(r, "paper_id"))
            title = getattr(r, "title", None)
            authors = getattr(r, "authors", None) if "authors" in df.columns else None
            doi = getattr(r, "doi", None) if "doi" in df.columns else None

            if not paper_id or not isinstance(title, str) or not title.strip():
                continue
            if paper_id in done and not args.force:
                skipped += 1
                continue

            params: dict[str, Any] = {
                "search": title,
                "per_page": int(args.per_page),
                "select": "id,doi,title,publication_year,type,cited_by_count,primary_location,authorships",
            }
            if mailto:
                params["mailto"] = mailto

            try:
                js = _get_json_with_retries(session, OPENALEX_WORKS, params=params)
                results = js.get("results") if isinstance(js.get("results"), list) else []

                pre_last = _parse_preprint_lastnames(authors)
                candidates: list[dict[str, Any]] = []
                for w in results:
                    if not isinstance(w, dict):
                        continue
                    cand = _select_candidate_fields(w)
                    if _is_ssrn_record(doi_url=cand.get("doi_url"), primary_source=cand.get("primary_source")):
                        continue
                    sim = _title_similarity(title, cand.get("title"))
                    cand_last = _candidate_lastnames(cand.get("authorships"))
                    aov = _author_overlap(pre_last, cand_last)
                    cites = _safe_int(cand.get("cited_by_count"))
                    is_top3 = _is_top_venue(cand.get("primary_source"))
                    score = 100.0 * float(sim.get("title_sim_token_f1") or 0.0)
                    score += 40.0 * float(aov)
                    score += 6.0 * math.log1p(max(0, cites))
                    score += 15.0 * (1.0 if is_top3 else 0.0)
                    candidates.append(
                        {
                            **{k: cand.get(k) for k in ["openalex_id", "doi_url", "title", "publication_year", "type", "cited_by_count", "primary_source", "primary_source_type"]},
                            "title_sim_token_f1": float(sim.get("title_sim_token_f1") or 0.0),
                            "title_sim_jaccard": float(sim.get("title_sim_jaccard") or 0.0),
                            "author_overlap": float(aov),
                            "is_top3_venue": bool(is_top3),
                            "score": float(score),
                        }
                    )

                candidates.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
                best = candidates[0] if candidates else None
                is_conf = bool(
                    best
                    and float(best.get("title_sim_token_f1") or 0.0) >= float(args.min_title_sim_token_f1)
                    and (not pre_last or float(best.get("author_overlap") or 0.0) >= float(args.min_author_overlap))
                )

                rec = {
                    "paper_id": paper_id,
                    "preprint_title": title,
                    "preprint_authors": (authors if isinstance(authors, str) and authors.strip() else None),
                    "preprint_doi": (doi if isinstance(doi, str) and doi.strip() else None),
                    "status": "ok",
                    "n_results_returned": int(len(results)),
                    "n_candidates_non_ssrn": int(len(candidates)),
                    "best": best,
                    "best_is_confident": bool(is_conf),
                    "fetched_at": _now_iso(),
                }
                # Keep only a small candidate list for auditability
                rec["candidates_top10"] = candidates[:10]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fetched += 1
            except Exception as e:
                rec = {
                    "paper_id": paper_id,
                    "preprint_title": title,
                    "preprint_authors": (authors if isinstance(authors, str) and authors.strip() else None),
                    "preprint_doi": (doi if isinstance(doi, str) and doi.strip() else None),
                    "status": "error",
                    "error": f"{type(e).__name__}: {e}",
                    "fetched_at": _now_iso(),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                errors += 1

            if i % 25 == 0:
                print(f"[{_now_iso()}] ({i}/{n}) fetched={fetched} skipped={skipped} errors={errors}")
            _sleep_polite(float(args.sleep_s))

    # Build CSV from JSONL
    rows: list[dict[str, Any]] = []
    for line in out_jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        best = obj.get("best") if isinstance(obj.get("best"), dict) else {}
        rows.append(
            {
                "paper_id": obj.get("paper_id"),
                "preprint_title": obj.get("preprint_title"),
                "preprint_authors": obj.get("preprint_authors"),
                "preprint_doi": obj.get("preprint_doi"),
                "status": obj.get("status"),
                "n_results_returned": obj.get("n_results_returned"),
                "n_candidates_non_ssrn": obj.get("n_candidates_non_ssrn"),
                "best_is_confident": obj.get("best_is_confident"),
                "best_related_openalex_id": best.get("openalex_id") if isinstance(best, dict) else None,
                "best_related_doi_url": best.get("doi_url") if isinstance(best, dict) else None,
                "best_related_title": best.get("title") if isinstance(best, dict) else None,
                "best_related_publication_year": best.get("publication_year") if isinstance(best, dict) else None,
                "best_related_type": best.get("type") if isinstance(best, dict) else None,
                "best_related_cited_by_count": best.get("cited_by_count") if isinstance(best, dict) else None,
                "best_related_primary_source": best.get("primary_source") if isinstance(best, dict) else None,
                "best_related_title_sim_token_f1": best.get("title_sim_token_f1") if isinstance(best, dict) else None,
                "best_related_author_overlap": best.get("author_overlap") if isinstance(best, dict) else None,
                "best_related_is_top3_venue": best.get("is_top3_venue") if isinstance(best, dict) else None,
                "best_related_score": best.get("score") if isinstance(best, dict) else None,
            }
        )

    pd.DataFrame(rows).to_csv(out_map, index=False)
    print(f"[{_now_iso()}] wrote: {out_jsonl}")
    print(f"[{_now_iso()}] wrote: {out_map}")
    print(f"[{_now_iso()}] done: fetched={fetched} skipped={skipped} errors={errors} total={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
