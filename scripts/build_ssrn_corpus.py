#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests
from bs4 import BeautifulSoup


SSRN_PAPER_URL = "https://papers.ssrn.com/sol3/papers.cfm"
SSRN_DELIVERY_URL = "https://papers.ssrn.com/sol3/Delivery.cfm"
CROSSREF_WORKS_URL = "https://api.crossref.org/works"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _slugify(s: str, *, max_len: int = 120) -> str:
    s = (s or "").strip()
    if not s:
        return "paper"
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\-\. ]+", "_", s, flags=re.UNICODE)
    s = s.replace(" ", "_")
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        return "paper"
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s


def _sha256_file(path: Path, *, max_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        if max_bytes is None:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        else:
            remaining = int(max_bytes)
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                h.update(chunk)
                remaining -= len(chunk)
    return h.hexdigest()


def _stable_sample(records: list[dict[str, Any]], *, n: int, seed: int) -> list[dict[str, Any]]:
    """
    Deterministic "pseudo-random" sampling that is stable across runs given the same input set.

    We rank each ssrn_id by sha1(seed:ssrn_id) and take the smallest n.
    """
    if n <= 0 or n >= len(records):
        return records
    scored: list[tuple[str, dict[str, Any]]] = []
    for rec in records:
        sid = str(rec.get("ssrn_id") or "").strip()
        if not sid:
            continue
        h = hashlib.sha1(f"{seed}:{sid}".encode("utf-8", errors="replace")).hexdigest()
        scored.append((h, rec))
    scored.sort(key=lambda x: x[0])
    return [r for _, r in scored[:n]]


def _extract_ssrn_id(s: str) -> str | None:
    s = (s or "").strip()
    if not s:
        return None
    # DOI: 10.2139/ssrn.2694998
    m = re.search(r"10\.2139/ssrn\.(\d+)", s, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    # URL: ...abstract_id=2694998 or abstractid=2694998
    m = re.search(r"(?:abstract_id|abstractid)=(\d+)", s, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    # bare id
    if s.isdigit():
        return s
    return None


def _read_ids_file(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sid = _extract_ssrn_id(line)
        if sid:
            ids.append(sid)
    # de-dupe preserve order
    out: list[str] = []
    seen: set[str] = set()
    for x in ids:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _read_queries_file(path: Path) -> list[str]:
    queries: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        q = line.strip()
        if not q or q.startswith("#"):
            continue
        queries.append(q)
    # de-dupe preserve order
    out: list[str] = []
    seen: set[str] = set()
    for q in queries:
        if q in seen:
            continue
        seen.add(q)
        out.append(q)
    return out


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return s


def _sleep_polite(base_s: float) -> None:
    if base_s <= 0:
        return
    time.sleep(base_s)


def _crossref_iter_ssrn_ids(
    session: requests.Session,
    *,
    query: str,
    from_year: int,
    until_year: int,
    max_results: int,
    rows: int = 200,
    mailto: str | None = None,
    polite_sleep_s: float = 0.2,
) -> Iterable[dict[str, Any]]:
    filter_str = (
        "container-title:SSRN Electronic Journal,"
        f"from-pub-date:{from_year}-01-01,"
        f"until-pub-date:{until_year}-12-31"
    )
    cursor = "*"
    got = 0
    while got < max_results:
        params = {
            "query.bibliographic": query,
            "filter": filter_str,
            "rows": min(rows, max_results - got),
            "cursor": cursor,
        }
        if mailto:
            params["mailto"] = mailto
        r = session.get(CROSSREF_WORKS_URL, params=params, timeout=60)
        r.raise_for_status()
        msg = r.json().get("message") or {}
        items = msg.get("items") or []
        if not items:
            break
        for it in items:
            doi = (it.get("DOI") or "").strip()
            sid = _extract_ssrn_id(doi)
            if not sid:
                continue
            authors: list[str] = []
            for a in (it.get("author") or []) if isinstance(it.get("author"), list) else []:
                if not isinstance(a, dict):
                    continue
                family = (a.get("family") or "").strip()
                given = (a.get("given") or "").strip()
                if family and given:
                    authors.append(f"{family}, {given}")
                elif family:
                    authors.append(family)
                elif given:
                    authors.append(given)
            year = None
            try:
                parts = (it.get("issued") or {}).get("date-parts")
                if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
                    y = parts[0][0]
                    if isinstance(y, int):
                        year = y
            except Exception:
                year = None
            out = {
                "ssrn_id": sid,
                "source": "crossref",
                "crossref_doi": doi,
                "crossref_title": (it.get("title") or [None])[0],
                "crossref_issued": (it.get("issued") or {}).get("date-parts"),
                "crossref_year": year,
                "crossref_authors": authors,
                "crossref_score": it.get("score"),
                "crossref_url": it.get("URL"),
            }
            yield out
            got += 1
            if got >= max_results:
                break
        cursor = msg.get("next-cursor") or cursor
        _sleep_polite(polite_sleep_s)


def _fetch_ssrn_citation_meta(session: requests.Session, ssrn_id: str, *, polite_sleep_s: float = 0.2) -> dict[str, Any]:
    params = {"abstract_id": ssrn_id}
    r = session.get(SSRN_PAPER_URL, params=params, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    meta: dict[str, Any] = {}
    authors: list[str] = []
    for tag in soup.find_all("meta"):
        name = (tag.get("name") or tag.get("property") or "").strip()
        if not name:
            continue
        content = (tag.get("content") or "").strip()
        if not content:
            continue
        if name.lower() == "citation_author":
            authors.append(content)
            continue
        if name.lower().startswith("citation_"):
            meta[name.lower()] = content
    if authors:
        meta["citation_author"] = authors
    # Fallbacks
    if "citation_title" not in meta:
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            meta["citation_title"] = h1.get_text(" ", strip=True)
    meta["ssrn_id"] = ssrn_id
    _sleep_polite(polite_sleep_s)
    return meta


def _download_ssrn_pdf(
    session: requests.Session,
    *,
    ssrn_id: str,
    out_path: Path,
    referer_url: str,
    timeout_s: int = 180,
) -> tuple[int, str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    params = {"abstractid": ssrn_id, "download": "1"}
    headers = {"Referer": referer_url}
    with session.get(SSRN_DELIVERY_URL, params=params, headers=headers, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        ctype = (r.headers.get("content-type") or "").lower()
        if "pdf" not in ctype and not r.content.startswith(b"%PDF"):
            raise RuntimeError(f"Unexpected content-type: {ctype}")
        n = 0
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                f.write(chunk)
                n += len(chunk)
    return n, r.url


@dataclass
class ManifestRow:
    ssrn_id: str
    status: str
    pdf_relpath: str | None
    bytes: int | None
    sha256: str | None
    title: str | None
    authors: str | None
    year: int | None
    doi: str | None
    ssrn_url: str
    download_url: str | None
    error: str | None
    source: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ssrn_id": self.ssrn_id,
            "status": self.status,
            "pdf_relpath": self.pdf_relpath,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "doi": self.doi,
            "ssrn_url": self.ssrn_url,
            "download_url": self.download_url,
            "error": self.error,
            "source": self.source,
        }


def _year_from_meta(meta: dict[str, Any]) -> int | None:
    for k in ["citation_publication_date", "citation_online_date", "citation_date"]:
        v = meta.get(k)
        if isinstance(v, str):
            m = re.search(r"(19|20)\d{2}", v)
            if m:
                return int(m.group(0))
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a local SSRN PDF corpus (download + manifest).")
    ap.add_argument("--out-dir", default="corpus", help="Corpus output directory.")
    ap.add_argument("--ids-file", default=None, help="Text file containing SSRN ids/URLs/DOIs (one per line).")
    ap.add_argument("--ssrn-id", action="append", default=[], help="SSRN abstract id (repeatable).")
    ap.add_argument("--query", action="append", default=[], help="Crossref query (repeatable).")
    ap.add_argument("--queries-file", action="append", default=[], help="Text file with one Crossref query per line.")
    ap.add_argument("--from-year", type=int, default=2010)
    ap.add_argument("--until-year", type=int, default=int(time.strftime("%Y")))
    ap.add_argument("--max-results", type=int, default=200, help="Max SSRN ids to collect from Crossref (per query).")
    ap.add_argument("--sample-n", type=int, default=None, help="Optional down-sampling after de-duplication (for big pulls).")
    ap.add_argument("--sample-seed", type=int, default=123, help="Seed for deterministic down-sampling.")
    ap.add_argument("--crossref-mailto", default=os.environ.get("CROSSREF_MAILTO"), help="Optional mailto for Crossref.")
    ap.add_argument(
        "--skip-ssrn-meta",
        action="store_true",
        help="Do not fetch SSRN abstract pages; use Crossref metadata only (much faster for large id lists).",
    )
    ap.add_argument("--download", action="store_true", help="Download PDFs (otherwise only writes id list/manifest stubs).")
    ap.add_argument("--force", action="store_true", help="Force refresh metadata and PDFs (equivalent to --force-meta --force-pdf).")
    ap.add_argument("--force-meta", action="store_true", help="Refresh SSRN metadata even if cached meta JSON exists.")
    ap.add_argument("--force-pdf", action="store_true", help="Redownload PDFs even if they already exist.")
    ap.add_argument("--sleep-s", type=float, default=0.5, help="Polite sleep seconds between SSRN requests.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    pdf_dir = out_dir / "pdfs"
    meta_dir = out_dir / "meta"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    session = _make_session()
    force_meta = bool(args.force or args.force_meta)
    force_pdf = bool(args.force or args.force_pdf)
    skip_ssrn_meta = bool(args.skip_ssrn_meta)

    ids: list[dict[str, Any]] = []

    # From file / explicit ids
    if args.ids_file:
        for sid in _read_ids_file(Path(args.ids_file)):
            ids.append({"ssrn_id": sid, "source": "ids_file"})
    for sid in args.ssrn_id:
        sid2 = _extract_ssrn_id(sid)
        if sid2:
            ids.append({"ssrn_id": sid2, "source": "cli"})

    # From Crossref
    queries: list[str] = []
    if isinstance(args.query, list):
        queries.extend([str(q).strip() for q in args.query if str(q).strip()])
    if isinstance(args.queries_file, list):
        for qp in args.queries_file:
            if qp:
                queries.extend(_read_queries_file(Path(qp)))
    # De-dupe while preserving order
    q_seen: set[str] = set()
    q_uniq: list[str] = []
    for q in queries:
        if q in q_seen:
            continue
        q_seen.add(q)
        q_uniq.append(q)

    for q in q_uniq:
        for rec in _crossref_iter_ssrn_ids(
            session,
            query=q,
            from_year=int(args.from_year),
            until_year=int(args.until_year),
            max_results=int(args.max_results),
            mailto=args.crossref_mailto,
            polite_sleep_s=float(args.sleep_s),
        ):
            ids.append(rec)

    # De-dupe by ssrn_id (preserve first occurrence)
    uniq: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rec in ids:
        sid = str(rec.get("ssrn_id") or "").strip()
        if not sid or sid in seen:
            continue
        seen.add(sid)
        uniq.append(rec)

    if args.sample_n is not None:
        n = int(args.sample_n)
        if n > 0 and n < len(uniq):
            uniq = _stable_sample(uniq, n=n, seed=int(args.sample_seed))
            print(f"[{_now_iso()}] sampled: {len(uniq)} (seed={int(args.sample_seed)})")

    id_list_path = out_dir / "ssrn_ids.txt"
    id_list_path.write_text("\n".join([r["ssrn_id"] for r in uniq]) + ("\n" if uniq else ""), encoding="utf-8")
    print(f"[{_now_iso()}] ids: {len(uniq)} -> {id_list_path}")

    manifest_rows: list[ManifestRow] = []

    for i, rec in enumerate(uniq, start=1):
        ssrn_id = str(rec.get("ssrn_id"))
        ssrn_url = f"{SSRN_PAPER_URL}?abstract_id={ssrn_id}"
        meta_path = meta_dir / f"ssrn_{ssrn_id}.json"
        pdf_path = pdf_dir / f"ssrn_{ssrn_id}.pdf"

        meta_error: str | None = None
        meta: dict[str, Any] = {}
        try:
            if skip_ssrn_meta:
                meta_error = "skipped_ssrn_meta"
                meta = {
                    "ssrn_id": ssrn_id,
                    "citation_title": rec.get("crossref_title"),
                    "citation_author": rec.get("crossref_authors"),
                    "citation_doi": rec.get("crossref_doi"),
                    "citation_publication_date": (f"{rec.get('crossref_year')}-01-01" if rec.get("crossref_year") else None),
                    "_meta_error": meta_error,
                    "_source": rec.get("source"),
                }
                if (not meta_path.exists()) or force_meta:
                    try:
                        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                    except Exception:
                        pass
            elif meta_path.exists() and not force_meta:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            else:
                meta = _fetch_ssrn_citation_meta(session, ssrn_id, polite_sleep_s=float(args.sleep_s))
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            meta_error = f"{type(e).__name__}: {e}"
            # Fall back to Crossref-provided metadata (still try download).
            meta = {
                "ssrn_id": ssrn_id,
                "citation_title": rec.get("crossref_title"),
                "citation_author": rec.get("crossref_authors"),
                "citation_doi": rec.get("crossref_doi"),
                "citation_publication_date": (f"{rec.get('crossref_year')}-01-01" if rec.get("crossref_year") else None),
                "_meta_error": meta_error,
                "_source": rec.get("source"),
            }
            try:
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

        title = meta.get("citation_title") or rec.get("crossref_title")
        authors = meta.get("citation_author")
        if isinstance(authors, list):
            authors_s = "; ".join(authors)
        else:
            authors_s = str(authors) if isinstance(authors, str) else None
        year = _year_from_meta(meta) or (int(rec.get("crossref_year")) if isinstance(rec.get("crossref_year"), int) else None)
        doi = meta.get("citation_doi") or rec.get("crossref_doi")

        if not args.download:
            manifest_rows.append(
                ManifestRow(
                    ssrn_id=ssrn_id,
                    status=("collected" if not meta_error else ("collected_meta_error" if not skip_ssrn_meta else "collected_meta_skipped")),
                    pdf_relpath=str(pdf_path.relative_to(out_dir)).replace("\\", "/") if pdf_path.exists() else None,
                    bytes=pdf_path.stat().st_size if pdf_path.exists() else None,
                    sha256=_sha256_file(pdf_path) if pdf_path.exists() else None,
                    title=str(title) if isinstance(title, str) else None,
                    authors=authors_s,
                    year=year,
                    doi=str(doi) if isinstance(doi, str) else None,
                    ssrn_url=ssrn_url,
                    download_url=None,
                    error=meta_error,
                    source=str(rec.get("source") or "unknown"),
                )
            )
            continue

        # Download
        try:
            if pdf_path.exists() and not force_pdf:
                nbytes = pdf_path.stat().st_size
                download_url = None
            else:
                nbytes, download_url = _download_ssrn_pdf(
                    session,
                    ssrn_id=ssrn_id,
                    out_path=pdf_path,
                    referer_url=ssrn_url,
                )
                _sleep_polite(float(args.sleep_s))

            sha = _sha256_file(pdf_path)
            manifest_rows.append(
                ManifestRow(
                    ssrn_id=ssrn_id,
                    status=(
                        ("downloaded" if download_url else "exists")
                        if not meta_error
                        else (("downloaded_meta_error" if download_url else "exists_meta_error") if not skip_ssrn_meta else ("downloaded_meta_skipped" if download_url else "exists_meta_skipped"))
                    ),
                    pdf_relpath=str(pdf_path.relative_to(out_dir)).replace("\\", "/"),
                    bytes=int(nbytes),
                    sha256=sha,
                    title=str(title) if isinstance(title, str) else None,
                    authors=authors_s,
                    year=year,
                    doi=str(doi) if isinstance(doi, str) else None,
                    ssrn_url=ssrn_url,
                    download_url=download_url,
                    error=meta_error,
                    source=str(rec.get("source") or "unknown"),
                )
            )
        except Exception as e:
            manifest_rows.append(
                ManifestRow(
                    ssrn_id=ssrn_id,
                    status="download_error" if not meta_error else "download_error_meta_error",
                    pdf_relpath=str(pdf_path.relative_to(out_dir)).replace("\\", "/") if pdf_path.exists() else None,
                    bytes=pdf_path.stat().st_size if pdf_path.exists() else None,
                    sha256=_sha256_file(pdf_path) if pdf_path.exists() else None,
                    title=str(title) if isinstance(title, str) else None,
                    authors=authors_s,
                    year=year,
                    doi=str(doi) if isinstance(doi, str) else None,
                    ssrn_url=ssrn_url,
                    download_url=None,
                    error=(meta_error + " | " if meta_error else "") + f"{type(e).__name__}: {e}",
                    source=str(rec.get("source") or "unknown"),
                )
            )

        print(f"[{_now_iso()}] ({i}/{len(uniq)}) ssrn_id={ssrn_id} status={manifest_rows[-1].status}")

    # Write manifest
    manifest_jsonl = out_dir / "manifest.jsonl"
    with manifest_jsonl.open("w", encoding="utf-8") as f:
        for r in manifest_rows:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    manifest_csv = out_dir / "manifest.csv"
    fieldnames = list(ManifestRow.__annotations__.keys())
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in manifest_rows:
            w.writerow({k: getattr(r, k) for k in fieldnames})

    print(f"[{_now_iso()}] wrote: {manifest_jsonl}")
    print(f"[{_now_iso()}] wrote: {manifest_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
