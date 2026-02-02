#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pdfplumber


EXTRACTOR_VERSION = "v2_20260125_3"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _sha256_file(path: Path, *, max_bytes: int | None = 5_000_000) -> str:
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

def _read_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        ids.add(s)
    return ids


def _cache_key(*, max_pages_per_paper: int, max_pdf_pages: int | None) -> str:
    mp = max(1, int(max_pages_per_paper))
    mpp = f"pdfp{int(max_pdf_pages)}" if max_pdf_pages is not None else "pdfpNone"
    return f"mp{mp}_{mpp}"


def _p_value_regex_hits(text: str) -> list[float]:
    if not text:
        return []
    hits: list[float] = []
    patterns = [
        r"(?i)\bp\s*(?:=|==)\s*(0\.\d+)\b",
        r"(?i)\bp\s*(?:<|≤)\s*(0\.\d+)\b",
        r"(?i)\bp[- ]?value\s*(?:=|==)\s*(0\.\d+)\b",
        r"(?i)\bp[- ]?value\s*(?:<|≤)\s*(0\.\d+)\b",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            try:
                v = float(m.group(1))
            except Exception:
                continue
            if 0.0 <= v <= 1.0:
                hits.append(v)
    return hits


def _bin_counts(values: list[float], lo: float, mid: float, hi: float) -> dict[str, int]:
    a = sum(1 for v in values if lo <= v <= mid)
    b = sum(1 for v in values if mid < v <= hi)
    return {"left": a, "right": b, "total": a + b}


def _looks_like_reference_page(text: str) -> bool:
    """
    Best-effort detection of bibliography/reference pages.

    Why we need this:
    - Reference pages have extremely high digit density (years, volume/issue/pages),
      so the naive page scorer tends to over-select them, which severely pollutes
      table-based numeric extraction.
    """
    t = (text or "").strip()
    if not t:
        return False

    head = t[:800].lower()
    if re.search(r"(?m)^\s*(references|bibliography)\b", head):
        return True

    # Heuristic: very high citation/year density + "references" anywhere early.
    year_paren = len(re.findall(r"\((?:19|20)\d{2}\)", t))
    doi_hits = len(re.findall(r"(?i)\bdoi\b|https?://|www\.", t))
    ref_word = bool(re.search(r"(?i)\breferences\b", head))
    if ref_word and (year_paren >= 12 or doi_hits >= 3):
        return True
    # Pure citation blocks without explicit header (common in appendices).
    if year_paren >= 25 and doi_hits >= 2:
        return True
    return False


def _numeric_ratio(s: str) -> float:
    s2 = re.sub(r"\s+", "", s or "")
    if not s2:
        return 0.0
    numeric = sum(ch.isdigit() or ch in ".-+*(),%" for ch in s2)
    return float(numeric / max(1, len(s2)))


def _normalize_table_cell_text(s: str) -> str:
    t = (s or "").strip()
    if not t:
        return ""
    # Normalize common unicode punctuation seen in PDF text extraction.
    t = t.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-").replace("\u2011", "-")
    t = t.replace("\u00b7", ".").replace("\u2219", ".").replace("\u2022", ".")
    # Remove thousands separators inside numbers.
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    # Tighten whitespace around numeric punctuation.
    t = re.sub(r"(?<=\d)\s+(?=[\.,])", "", t)
    t = re.sub(r"(?<=[\.,])\s+(?=\d)", "", t)
    t = re.sub(r"(?<=[+-])\s+(?=\d)", "", t)
    t = re.sub(r"(?<=\d)\s+(?=\*)", "", t)
    t = re.sub(r"(?<=\*)\s+(?=\*)", "", t)
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"\s+\)", ")", t)
    # Some PDFs split digits with spaces inside a token: "- 2 . 1 0" -> "-2.10".
    if _numeric_ratio(t) >= 0.9:
        t = re.sub(r"(?<=\d)\s+(?=\d)", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _looks_like_year(x: float) -> bool:
    try:
        xi = int(round(float(x)))
    except Exception:
        return False
    return 1800 <= xi <= 2100


def _looks_like_citation_cell(text: str) -> bool:
    """
    Detect common non-table numeric artifacts (references, journal citations).
    """
    t = (text or "").strip()
    if not t:
        return False
    tl = t.lower()
    if re.search(r"\((?:19|20)\d{2}\)", tl):
        return True
    if re.search(r"(?i)\bdoi\b|https?://|www\.", tl):
        return True
    # Volume(issue): pages
    if re.search(r"\b\d+\s*\(\s*\d+\s*\)\s*[:：]\s*\d+", tl):
        return True
    # Volume(issue), pages (common citation style): 2(4), 264-272
    if re.search(r"\b\d+\s*\(\s*\d+\s*\)\s*,\s*\d{2,4}\s*[-–]\s*\d{2,4}\b", tl):
        return True
    if re.search(r"(?i)\bpp\.?\b|\bp\.?\s*\d+", tl):
        return True
    if re.search(r"(?i)\bjournal\b|\breview\b|\bworking paper\b|\bvol\.?\b|\bno\.?\b", tl):
        return True
    return False


def _is_small_int_like(x: float, *, max_abs: int = 30, tol: float = 1e-9) -> bool:
    try:
        xf = float(x)
    except Exception:
        return False
    if xf != xf:
        return False
    if abs(xf) < tol:
        return False
    xi = int(round(xf))
    if abs(xf - float(xi)) > tol:
        return False
    return 1 <= abs(xi) <= int(max_abs)


def _looks_like_column_label_pair(rec: dict[str, Any]) -> bool:
    """
    Detect a common false-positive pattern: column labels like "8) (9)" being
    mistaken for (coef, paren) pairs.
    """
    try:
        stars = int(rec.get("stars") or 0)
    except Exception:
        stars = 0
    if stars != 0:
        return False

    cell = str(rec.get("cell_text_snippet") or "").strip()
    se_cell = str(rec.get("se_cell_text_snippet") or "").strip()
    coef_raw = str(rec.get("coef_raw") or "").strip()

    # Same-cell patterns: "8) (9)" or "73) (-4.12)" or "8) (9) (10)".
    if re.fullmatch(r"-?\d+\)\s*(?:\(\s*-?\d+(?:\.\d+)?\s*\)\s*)+", cell):
        return True
    # Next-row patterns: coef is a bare integer label, next row is "(9)".
    if coef_raw.isdigit() and re.search(rf"\b{re.escape(coef_raw)}\)", cell) and re.fullmatch(r"\(\s*-?\d+(?:\.\d+)?\s*\)", se_cell):
        return True
    return False


def _first_paren_number(s: str) -> float | None:
    vals = _extract_se_tokens(s)
    if not vals:
        return None
    v0 = float(vals[0])
    if _looks_like_year(v0):
        return None
    return v0


def _infer_paren_mode_from_text(full_text: str) -> tuple[str | None, str | None]:
    """
    Infer whether parentheses likely contain standard errors or t-stats.
    Returns: (mode, reason) where mode in {"se","t"} or None if ambiguous.
    """
    t = full_text or ""
    pats_t = [
        r"(?i)t[- ]?stat(?:istic)?s?\s+(?:are\s+)?(?:in|reported in)\s+parentheses",
        r"(?i)t[- ]?statistics\s+in\s+parentheses",
        r"(?i)\(t[- ]?stat(?:istic)?s?\)",
    ]
    pats_se = [
        r"(?i)standard\s+errors?\s+(?:are\s+)?(?:in|reported in)\s+parentheses",
        r"(?i)robust\s+standard\s+errors?\s+(?:are\s+)?(?:in|reported in)\s+parentheses",
        r"(?i)\(standard\s+errors?\)",
    ]
    t_hits = sum(len(re.findall(p, t)) for p in pats_t)
    se_hits = sum(len(re.findall(p, t)) for p in pats_se)
    if t_hits > se_hits and t_hits >= 1:
        return "t", f"text:t_hits={t_hits}>se_hits={se_hits}"
    if se_hits > t_hits and se_hits >= 1:
        return "se", f"text:se_hits={se_hits}>t_hits={t_hits}"
    return None, ("text:ambiguous" if (t_hits or se_hits) else "text:no_signal")


def _infer_paren_mode_from_values(paren_values: list[float]) -> tuple[str | None, str]:
    vals = [abs(float(v)) for v in paren_values if isinstance(v, (int, float)) and v == v and v != 0]
    # Remove obvious garbage years/outliers.
    vals = [v for v in vals if (not _looks_like_year(v)) and v <= 50]
    if len(vals) < 8:
        return None, f"values:too_few(n={len(vals)})"
    vals.sort()
    med = vals[len(vals) // 2]
    # Heuristic: t-stats are commonly ~[1,3]; SEs often <1 in many finance regressions.
    if len(vals) >= 15:
        if med >= 1.5:
            return "t", f"values:n={len(vals)} median_abs_paren={med:.3f}>=1.5"
        return "se", f"values:n={len(vals)} median_abs_paren={med:.3f}<1.5"
    if med >= 2.5:
        return "t", f"values:n={len(vals)} median_abs_paren={med:.3f}>=2.5"
    if med <= 0.8:
        return "se", f"values:n={len(vals)} median_abs_paren={med:.3f}<=0.8"
    return None, f"values:n={len(vals)} median_abs_paren={med:.3f} ambiguous"


def _find_terms(text: str, patterns: list[str]) -> int:
    if not text:
        return 0
    n = 0
    for pat in patterns:
        n += len(re.findall(pat, text, flags=re.IGNORECASE))
    return n


@dataclass(frozen=True)
class PageScore:
    page_1based: int
    score: float
    digit_count: int
    star_count: int
    table_mentions: int
    robust_mentions: int
    p_mentions: int


def _page_score(page_1based: int, text: str) -> PageScore:
    t = text or ""
    digits = sum(ch.isdigit() for ch in t)
    stars = t.count("*")
    table_mentions = len(re.findall(r"(?i)\btable\b", t))
    robust_mentions = len(re.findall(r"(?i)\brobust(?:ness)?\b", t))
    p_mentions = len(re.findall(r"(?i)\bp[- ]?value\b|\bp\s*(?:=|<|≤)", t))
    ref_penalty = 1.5 if _looks_like_reference_page(t) else 0.0
    # Empirical weighting: digits matter most, then tables/stars/robust cues.
    score = (digits / 2000.0) + 0.8 * table_mentions + 0.25 * robust_mentions + 0.02 * stars + 0.5 * p_mentions - ref_penalty
    return PageScore(page_1based, score, digits, stars, table_mentions, robust_mentions, p_mentions)


def _pick_pages(scores: list[PageScore], *, max_pages: int, exclude_pages: set[int] | None = None) -> list[int]:
    if not scores:
        return [1]
    ranked = sorted(scores, key=lambda s: s.score, reverse=True)
    picks: list[int] = []
    for s in ranked:
        if exclude_pages and s.page_1based in exclude_pages and s.page_1based != 1:
            continue
        if s.page_1based not in picks:
            picks.append(s.page_1based)
        if len(picks) >= max_pages:
            break
    if 1 not in picks:
        picks.insert(0, 1)
        picks = picks[:max_pages]
    return picks


def _extract_numeric_tokens(s: str) -> list[tuple[str, float, int]]:
    """
    Extract numeric tokens like 0.123*** or -1.45* and return (raw, value, stars).
    """
    out: list[tuple[str, float, int]] = []
    if not s:
        return out
    s2 = _normalize_table_cell_text(s)
    for m in re.finditer(r"(?<![\w.])([+-]?(?:\d+(?:\.\d+)?|\.\d+))(\*{1,3})?(?![\w.])", s2):
        raw = m.group(0)
        try:
            v = float(m.group(1))
        except Exception:
            continue
        stars = len(m.group(2) or "")
        out.append((raw, v, stars))
    return out


def _extract_se_tokens(s: str) -> list[float]:
    """
    Extract numeric tokens in parentheses: (0.069), (1.23), etc.

    Note: depending on the paper/table, parentheses may contain standard errors OR t-stats.
    We infer the meaning at paper-level later. Here we only extract numbers.
    """
    out: list[float] = []
    if not s:
        return out
    s2 = _normalize_table_cell_text(s)
    for m in re.finditer(r"([+-])?\s*\(\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*\)", s2):
        try:
            num_str = m.group(2)
            v = float(num_str)
            prefix = m.group(1)
            if prefix == "-" and not (num_str.startswith("-") or num_str.startswith("+")):
                v = -v
        except Exception:
            continue
        # Filter obvious bibliography years to reduce false positives.
        if _looks_like_year(v):
            continue
        out.append(v)
    return out


def _extract_first_coef_token(s: str) -> tuple[str, float, int] | None:
    """
    Return the most plausible coefficient token from a cell (raw, value, stars).

    Heuristic:
    - Ignore anything inside parentheses (to avoid standard errors / column labels like (1)).
    - Prefer tokens with stars; otherwise take the first token.
    """
    if not s:
        return None
    s2 = _normalize_table_cell_text(s)
    # Avoid pure column-label cells like "(1) (2) (3)".
    if re.fullmatch(r"(?:\(\s*[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*\)\s*)+", s2.strip()):
        return None

    s_no_parens = re.sub(r"\([^)]*\)", " ", s2)
    # Remove a trailing unfinished parenthetical token like "(5" to avoid false coef=5.
    s_no_parens = re.sub(r"\(\s*[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*$", " ", s_no_parens)
    nums = _extract_numeric_tokens(s_no_parens)
    if not nums:
        return None
    # Prefer starred coefficients (common in tables).
    starred = [x for x in nums if x[2] > 0]
    return starred[0] if starred else nums[0]


def _first_se(s: str) -> float | None:
    ses = _extract_se_tokens(s or "")
    return ses[0] if ses else None


def _table_settings() -> dict[str, Any]:
    # Tuned for SSRN / academic PDFs: text-based grid inference.
    return {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
        "min_words_vertical": 2,
        "min_words_horizontal": 1,
        "intersection_tolerance": 5,
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "text_x_tolerance": 2,
        "text_y_tolerance": 2,
    }


def _in_window(x: float, *, center: float, delta: float) -> int:
    """
    Returns:
    -1 if x in [center-delta, center]
     1 if x in (center, center+delta]
     0 otherwise
    """
    if (center - delta) <= x <= center:
        return -1
    if center < x <= (center + delta):
        return 1
    return 0


def _iter_t_pairs_from_tables(pdf_path: Path, *, candidate_pages_1based: list[int]) -> Any:
    """
    Yield raw extracted pairs from detected tables.

    The parentheses number may be a standard error OR a t-stat depending on the paper;
    we postpone interpretation until we infer the parentheses mode.
    """
    with pdfplumber.open(str(pdf_path)) as pdf:
        for p1 in candidate_pages_1based:
            if p1 < 1 or p1 > len(pdf.pages):
                continue
            page = pdf.pages[p1 - 1]
            try:
                tables = page.find_tables(table_settings=_table_settings())
            except Exception:
                tables = []
            if not tables:
                continue
            for ti, tb in enumerate(tables):
                grid = tb.extract()
                if not isinstance(grid, list) or not grid:
                    continue
                rows = getattr(tb, "rows", None)
                for ri, row in enumerate(grid):
                    if not isinstance(row, list) or not row:
                        continue
                    for ci, cell in enumerate(row):
                        cell_text = _normalize_table_cell_text((cell or ""))
                        # Avoid pulling numeric-looking tokens out of bibliography-like text cells.
                        if _numeric_ratio(cell_text) < 0.60:
                            continue
                        if _looks_like_citation_cell(cell_text):
                            continue
                        coef_tok = _extract_first_coef_token(cell_text)
                        if coef_tok is None:
                            continue
                        raw, coef, stars = coef_tok

                        # Parentheses number may be in the same cell, or in the next row same column.
                        paren = _first_paren_number(cell_text)
                        paren_row_index = ri
                        paren_cell_text = cell_text
                        if paren is None and (ri + 1) < len(grid):
                            row2 = grid[ri + 1]
                            if isinstance(row2, list) and ci < len(row2):
                                paren_cell_text = _normalize_table_cell_text((row2[ci] or ""))
                                if _numeric_ratio(paren_cell_text) >= 0.60 and not _looks_like_citation_cell(paren_cell_text):
                                    paren = _first_paren_number(paren_cell_text)
                                    paren_row_index = ri + 1
                        if paren is None or paren == 0:
                            continue

                        coef_cell_bbox = None
                        paren_cell_bbox = None
                        try:
                            if rows is not None and ri < len(rows):
                                coef_cell_bbox = list(rows[ri].cells[ci]) if ci < len(rows[ri].cells) else None
                            if rows is not None and paren_row_index < len(rows):
                                paren_cell_bbox = (
                                    list(rows[paren_row_index].cells[ci]) if ci < len(rows[paren_row_index].cells) else None
                                )
                        except Exception:
                            coef_cell_bbox = None
                            paren_cell_bbox = None

                        yield {
                            "page": p1,
                            "table_index": ti,
                            "table_bbox": list(tb.bbox),
                            "row_index": ri,
                            "col_index": ci,
                            "coef_raw": raw,
                            "coef": coef,
                            "paren": paren,
                            "paren_source": ("same_cell" if paren_row_index == ri else "next_row_same_col"),
                            "stars": stars,
                            "coef_cell_bbox": coef_cell_bbox,
                            "se_cell_bbox": paren_cell_bbox,
                            "cell_text_snippet": (cell_text[:120] if cell_text else None),
                            "se_cell_text_snippet": (paren_cell_text[:120] if paren_cell_text else None),
                        }


def _extract_t_stats_from_tables(pdf_path: Path, *, candidate_pages_1based: list[int]) -> dict[str, Any]:
    """
    Best-effort extraction of coefficient/SE pairs from detected tables.
    Returns summary counts + a small sample of extracted t-stats with provenance.
    """
    extracted: list[dict[str, Any]] = []
    n_tables = 0
    n_pairs = 0

    with pdfplumber.open(str(pdf_path)) as pdf:
        for p1 in candidate_pages_1based:
            if p1 < 1 or p1 > len(pdf.pages):
                continue
            page = pdf.pages[p1 - 1]
            try:
                tables = page.find_tables(table_settings=_table_settings())
            except Exception:
                tables = []
            if not tables:
                continue
            for ti, tb in enumerate(tables):
                n_tables += 1
                grid = tb.extract()
                if not isinstance(grid, list) or not grid:
                    continue
                # Iterate row pairs: coef row then SE row.
                for ri in range(len(grid) - 1):
                    row = grid[ri]
                    row2 = grid[ri + 1]
                    if not isinstance(row, list) or not isinstance(row2, list):
                        continue
                    s1 = " ".join([c or "" for c in row])
                    s2 = " ".join([c or "" for c in row2])
                    nums = _extract_numeric_tokens(s1)
                    ses = _extract_se_tokens(s2)
                    if len(nums) < 2 or len(ses) < 2:
                        continue
                    k = min(len(nums), len(ses), 8)
                    for j in range(k):
                        raw, coef, stars = nums[j]
                        se = ses[j]
                        if se == 0:
                            continue
                        t = coef / se
                        n_pairs += 1
                        if len(extracted) < 200:
                            extracted.append(
                                {
                                    "page": p1,
                                    "table_index": ti,
                                    "table_bbox": list(tb.bbox),
                                    "row_index": ri,
                                    "coef_raw": raw,
                                    "coef": coef,
                                    "se": se,
                                    "t": t,
                                    "abs_t": abs(t),
                                    "stars": stars,
                                }
                            )

    abs_ts = [float(x["abs_t"]) for x in extracted if isinstance(x.get("abs_t"), (int, float))]
    abs_ts_sorted = sorted(abs_ts)
    return {
        "tables_seen": n_tables,
        "t_pairs_seen": n_pairs,
        "t_pairs_sample_n": len(extracted),
        "abs_t_sample": abs_ts_sorted[:50],
        "extracted_pairs_sample": extracted[:50],
    }


def _t_bunching_counts(abs_ts: list[float], *, center: float, delta: float) -> dict[str, int]:
    left = sum(1 for t in abs_ts if (center - delta) <= t <= center)
    right = sum(1 for t in abs_ts if center < t <= (center + delta))
    return {"left": left, "right": right, "total": left + right}


def _normalize_ratio(counts: dict[str, int]) -> float:
    tot = int(counts.get("total") or 0)
    if tot <= 0:
        return 0.0
    left = int(counts.get("left") or 0)
    right = int(counts.get("right") or 0)
    return (left - right) / max(1, tot)


def _two_sided_binom_pvalue(k: int, n: int, *, max_exact_n: int = 400) -> float | None:
    """
    Two-sided p-value for X~Bin(n, 0.5) at observed k.

    For small n we compute an exact probability by summing pmf over outcomes at
    least as extreme as k around the center. For larger n we use a normal
    approximation with z=(2k-n)/sqrt(n).
    """
    if n <= 0:
        return None
    if k < 0 or k > n:
        return None
    if n > max_exact_n:
        z = (2.0 * float(k) - float(n)) / math.sqrt(float(n))
        return float(math.erfc(abs(z) / math.sqrt(2.0)))

    center = float(n) / 2.0
    dist = abs(float(k) - center)
    base = 0.5**n
    p = 0.0
    for x in range(n + 1):
        if abs(float(x) - center) + 1e-12 < dist:
            continue
        try:
            p += math.comb(n, x) * base
        except Exception:
            z = (2.0 * float(k) - float(n)) / math.sqrt(float(n))
            return float(math.erfc(abs(z) / math.sqrt(2.0)))
    return float(min(1.0, max(0.0, p)))


def _caliper_z(left: int, right: int, *, direction: str) -> float:
    """
    direction:
      - "left":  (left-right)/sqrt(n)  (e.g., p just below threshold vs above)
      - "right": (right-left)/sqrt(n)  (e.g., |t| just above threshold vs below)
    """
    n = int(left) + int(right)
    if n <= 0:
        return 0.0
    if direction == "right":
        return float((int(right) - int(left)) / math.sqrt(float(n)))
    return float((int(left) - int(right)) / math.sqrt(float(n)))


def _caliper_p_two_sided(left: int, right: int, *, max_exact_n: int = 400) -> float | None:
    n = int(left) + int(right)
    if n <= 0:
        return None
    # symmetric under p=0.5; choose k=left
    return _two_sided_binom_pvalue(int(left), n, max_exact_n=max_exact_n)


def _cap01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v != v:
        return 0.0
    return max(0.0, min(1.0, v))


def _compute_offline_risk_score(features: dict[str, Any]) -> tuple[int, dict[str, float]]:
    """
    A conservative, deterministic score designed for scaling. It is NOT a claim of true p-hacking.
    Returns (0..100 score, components).
    """
    comp: dict[str, float] = {}

    # Threshold bunching around conventional cutoffs (caliper on p-values).
    # Prefer p-values reconstructed from extracted (coef,se) pairs; fall back to regex p-values.
    b05z = features.get("p_from_t_caliper_0_05_z")
    b10z = features.get("p_from_t_caliper_0_10_z")
    if not isinstance(b05z, (int, float)):
        b05z = features.get("p_regex_caliper_0_05_z")
    if not isinstance(b10z, (int, float)):
        b10z = features.get("p_regex_caliper_0_10_z")
    comp["caliper_z_0_05"] = float(b05z) if isinstance(b05z, (int, float)) else 0.0
    comp["caliper_z_0_10"] = float(b10z) if isinstance(b10z, (int, float)) else 0.0

    # t-stat bunching around conventional cutoffs (more mass just above threshold).
    t196z = features.get("t_caliper_1_96_z")
    t1645z = features.get("t_caliper_1_645_z")
    comp["t_caliper_z_1_96"] = float(t196z) if isinstance(t196z, (int, float)) else 0.0
    comp["t_caliper_z_1_645"] = float(t1645z) if isinstance(t1645z, (int, float)) else 0.0

    # p-curve shape (among significant p-values): suspicious if "left-skewed" toward 0.05.
    pcurve_z = features.get("pcurve_right_skew_z")
    comp["pcurve_right_skew_z"] = float(pcurve_z) if isinstance(pcurve_z, (int, float)) else 0.0

    # Researcher degrees of freedom proxy: robustness/spec/search term intensity
    robust = float(features.get("robust_mentions_fulltext") or 0)
    spec = float(features.get("spec_search_terms_fulltext") or 0)
    multi = float(features.get("multiple_testing_terms_fulltext") or 0)
    comp["robust_term_intensity"] = min(1.0, robust / 10.0)
    comp["spec_term_intensity"] = min(1.0, spec / 8.0)
    comp["multiple_testing_exposure"] = min(1.0, multi / 4.0)

    # Correction disclosure reduces risk
    has_correction = 1.0 if bool(features.get("has_multiple_testing_correction")) else 0.0
    comp["has_correction"] = has_correction

    # Aggregate (weights chosen to be conservative; recalibrate in paper)
    score = 20.0
    score += 12.0 * _cap01(max(0.0, comp["caliper_z_0_05"]) / 3.0)
    score += 6.0 * _cap01(max(0.0, comp["caliper_z_0_10"]) / 3.0)
    score += 6.0 * _cap01(max(0.0, comp["t_caliper_z_1_96"]) / 3.0)
    score += 3.0 * _cap01(max(0.0, comp["t_caliper_z_1_645"]) / 3.0)
    score += 6.0 * _cap01(max(0.0, -comp["pcurve_right_skew_z"]) / 3.0)
    score += 12.0 * comp["robust_term_intensity"]
    score += 10.0 * comp["spec_term_intensity"]
    score += 8.0 * comp["multiple_testing_exposure"]
    score -= 12.0 * has_correction
    score = max(0.0, min(100.0, score))
    return int(round(score)), comp


def _risk_level(score: int) -> str:
    if score >= 70:
        return "High"
    if score >= 40:
        return "Moderate"
    return "Low"


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract scalable within-paper p-hacking proxy metrics from a PDF corpus.")
    ap.add_argument("--corpus-dir", default="corpus", help="Corpus directory (from build_ssrn_corpus.py).")
    ap.add_argument("--out-csv", default=None, help="Output CSV path (default: <corpus-dir>/features.csv).")
    ap.add_argument(
        "--tests-dir",
        default=None,
        help="Write per-paper extracted test pairs as JSONL under this dir (default: <corpus-dir>/tests).",
    )
    ap.add_argument(
        "--max-pdf-pages",
        type=int,
        default=None,
        help="Skip PDFs with more than this many pages (guards against books/dissertations in large SSRN pulls).",
    )
    ap.add_argument("--max-pages-per-paper", type=int, default=12, help="Max candidate pages for table parsing.")
    ap.add_argument("--paper-ids-file", default=None, help="Optional newline-delimited paper_id list to process.")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of PDFs processed (for smoke tests).")
    ap.add_argument("--only-regex", default=None, help="Only process PDFs whose stem matches this regex.")
    ap.add_argument("--force", action="store_true", help="Recompute even if cache exists.")
    args = ap.parse_args()

    corpus_dir = Path(args.corpus_dir)
    pdf_dir = corpus_dir / "pdfs"
    if not pdf_dir.exists():
        raise FileNotFoundError(pdf_dir)
    out_csv = Path(args.out_csv) if args.out_csv else (corpus_dir / "features.csv")
    tests_dir = Path(args.tests_dir) if args.tests_dir else (corpus_dir / "tests")
    tests_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = corpus_dir / "_cache_features"
    cache_dir.mkdir(parents=True, exist_ok=True)

    filter_ids: set[str] | None = None
    if args.paper_ids_file:
        filter_ids = _read_ids(Path(args.paper_ids_file))
        if not filter_ids:
            raise SystemExit(f"--paper-ids-file had no ids: {args.paper_ids_file}")

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if filter_ids is not None:
        pdfs = [p for p in pdfs if p.stem in filter_ids]
    if args.only_regex:
        try:
            pat = re.compile(str(args.only_regex))
        except re.error as e:
            raise SystemExit(f"Invalid --only-regex: {e}")
        pdfs = [p for p in pdfs if pat.search(p.stem)]
    if args.limit is not None:
        pdfs = pdfs[: max(0, int(args.limit))]
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    rows: list[dict[str, Any]] = []
    cache_key = _cache_key(max_pages_per_paper=int(args.max_pages_per_paper), max_pdf_pages=(int(args.max_pdf_pages) if args.max_pdf_pages is not None else None))

    for i, pdf_path in enumerate(pdfs, start=1):
        sha = _sha256_file(pdf_path)
        cache_path = cache_dir / f"{pdf_path.stem}__{sha[:12]}__{cache_key}.json"
        legacy_cache_path = cache_dir / f"{pdf_path.stem}__{sha[:12]}.json"
        if cache_path.exists() and not args.force:
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                cached = None
            if (
                isinstance(cached, dict)
                and cached.get("extractor_version") == EXTRACTOR_VERSION
                and cached.get("cache_key") == cache_key
                and int(cached.get("max_pages_per_paper") or 0) == int(args.max_pages_per_paper)
            ):
                rows.append(cached)
                continue
        if (not cache_path.exists()) and legacy_cache_path.exists() and not args.force:
            # Backward-compatible cache: older versions used <paper_id>__<sha>.json without the cache_key suffix.
            try:
                cached = json.loads(legacy_cache_path.read_text(encoding="utf-8"))
            except Exception:
                cached = None
            if isinstance(cached, dict) and cached.get("extractor_version") == EXTRACTOR_VERSION:
                cached.setdefault("cache_key", cache_key)
                cached.setdefault("max_pages_per_paper", int(args.max_pages_per_paper))
                cached.setdefault("max_pdf_pages", int(args.max_pdf_pages) if args.max_pdf_pages is not None else None)
                rows.append(cached)
                try:
                    cache_path.write_text(json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass
                continue

        doc = fitz.open(pdf_path)
        try:
            page_count = int(doc.page_count)
            if args.max_pdf_pages is not None and page_count > int(args.max_pdf_pages):
                feats: dict[str, Any] = {
                    "extractor_version": EXTRACTOR_VERSION,
                    "cache_key": cache_key,
                    "max_pages_per_paper": int(args.max_pages_per_paper),
                    "max_pdf_pages": int(args.max_pdf_pages) if args.max_pdf_pages is not None else None,
                    "paper_id": pdf_path.stem,
                    "pdf_relpath": str(pdf_path.relative_to(corpus_dir)).replace("\\", "/"),
                    "sha256": sha,
                    "pages": page_count,
                    "skipped_reason": f"page_count>{int(args.max_pdf_pages)}",
                    "generated_at": _now_iso(),
                    "offline_risk_score": None,
                    "offline_risk_level": None,
                    "offline_risk_components": None,
                }
                cache_path.write_text(json.dumps(feats, ensure_ascii=False, indent=2), encoding="utf-8")
                rows.append(feats)
                print(
                    f"[{_now_iso()}] ({i}/{len(pdfs)}) {pdf_path.name} pages={page_count} skipped={feats['skipped_reason']}"
                )
                continue

            page_texts: list[str] = []
            scores: list[PageScore] = []
            for pi in range(page_count):
                t = doc.load_page(pi).get_text("text") or ""
                page_texts.append(t)
                scores.append(_page_score(pi + 1, t))
            full_text = "\n\n".join(page_texts)
            pvals = _p_value_regex_hits(full_text)
        finally:
            doc.close()

        ref_pages = {pi + 1 for pi, txt in enumerate(page_texts) if _looks_like_reference_page(txt)}
        candidate_pages = _pick_pages(
            scores,
            max_pages=max(1, int(args.max_pages_per_paper)),
            exclude_pages=ref_pages,
        )
        tests_path = tests_dir / f"{pdf_path.stem}.jsonl"
        tests_meta_path = tests_path.with_suffix(".meta.json")

        t_near_196 = {"left": 0, "right": 0, "total": 0}
        t_near_1645 = {"left": 0, "right": 0, "total": 0}
        p_from_t_near_005 = {"left": 0, "right": 0, "total": 0}
        p_from_t_near_010 = {"left": 0, "right": 0, "total": 0}
        p_from_t_sig_005 = 0
        p_from_t_sig_010 = 0
        pcurve_sig_n = 0
        pcurve_low_half = 0
        pcurve_high_half = 0
        tables_seen_raw_set: set[tuple[int, int]] = set()
        tables_seen_set: set[tuple[int, int]] = set()
        pairs_seen_raw = 0
        pairs_seen = 0  # kept pairs
        sample_pairs: list[dict[str, Any]] = []
        filter_counts: Counter[str] = Counter()
        dropped_examples: list[dict[str, Any]] = []
        paren_mode: str | None = None
        paren_mode_source: str | None = None
        paren_mode_reason: str | None = None

        reuse_tests = False
        tests_meta: dict[str, Any] = {}
        if tests_path.exists() and tests_meta_path.exists() and not args.force:
            try:
                tests_meta = json.loads(tests_meta_path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                tests_meta = {}
            if isinstance(tests_meta, dict) and tests_meta.get("extractor_version") == EXTRACTOR_VERSION:
                reuse_tests = True

        if reuse_tests:
            paren_mode = tests_meta.get("paren_mode") if isinstance(tests_meta.get("paren_mode"), str) else None
            paren_mode_source = tests_meta.get("paren_mode_source") if isinstance(tests_meta.get("paren_mode_source"), str) else None
            paren_mode_reason = tests_meta.get("paren_mode_reason") if isinstance(tests_meta.get("paren_mode_reason"), str) else None
            pairs_seen_raw = int(tests_meta.get("pairs_raw") or 0)
            tables_seen_raw_set = set()
            try:
                for p_t in tests_meta.get("tables_raw") or []:
                    if isinstance(p_t, list) and len(p_t) == 2 and all(isinstance(x, int) for x in p_t):
                        tables_seen_raw_set.add((int(p_t[0]), int(p_t[1])))
            except Exception:
                tables_seen_raw_set = set()
            filter_counts = Counter((tests_meta.get("filter_counts") or {}))

            for line in tests_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue
                at = float(rec.get("abs_t") or 0.0)
                w1 = _in_window(at, center=1.96, delta=0.20)
                if w1 != 0:
                    t_near_196["total"] += 1
                    if w1 < 0:
                        t_near_196["left"] += 1
                    else:
                        t_near_196["right"] += 1
                w2 = _in_window(at, center=1.645, delta=0.20)
                if w2 != 0:
                    t_near_1645["total"] += 1
                    if w2 < 0:
                        t_near_1645["left"] += 1
                    else:
                        t_near_1645["right"] += 1
                p = rec.get("page")
                ti = rec.get("table_index")
                if isinstance(p, int) and isinstance(ti, int):
                    tables_seen_set.add((p, ti))
                pairs_seen += 1
                if len(sample_pairs) < 50:
                    sample_pairs.append(rec)

                p2 = rec.get("p_approx_2s")
                if not isinstance(p2, (int, float)):
                    p2 = float(math.erfc(at / math.sqrt(2.0)))
                p2 = float(p2)
                if not (0.0 <= p2 <= 1.0):
                    continue
                if p2 <= 0.05:
                    p_from_t_sig_005 += 1
                    pcurve_sig_n += 1
                    if p2 <= 0.025:
                        pcurve_low_half += 1
                    else:
                        pcurve_high_half += 1
                if p2 <= 0.10:
                    p_from_t_sig_010 += 1
                w3 = _in_window(p2, center=0.05, delta=0.005)
                if w3 != 0:
                    p_from_t_near_005["total"] += 1
                    if w3 < 0:
                        p_from_t_near_005["left"] += 1
                    else:
                        p_from_t_near_005["right"] += 1
                w4 = _in_window(p2, center=0.10, delta=0.005)
                if w4 != 0:
                    p_from_t_near_010["total"] += 1
                    if w4 < 0:
                        p_from_t_near_010["left"] += 1
                    else:
                        p_from_t_near_010["right"] += 1
        else:
            raw_pairs: list[dict[str, Any]] = []
            raw_parens: list[float] = []
            try:
                for rec in _iter_t_pairs_from_tables(pdf_path, candidate_pages_1based=candidate_pages):
                    rec2 = {"paper_id": pdf_path.stem, **rec}
                    raw_pairs.append(rec2)
                    pairs_seen_raw += 1
                    p = rec2.get("page")
                    ti = rec2.get("table_index")
                    if isinstance(p, int) and isinstance(ti, int):
                        tables_seen_raw_set.add((p, ti))
                    pv = rec2.get("paren")
                    if isinstance(pv, (int, float)) and not _looks_like_column_label_pair(rec2):
                        raw_parens.append(float(pv))
            except Exception:
                raw_pairs = []
                raw_parens = []
                pairs_seen_raw = 0
                tables_seen_raw_set = set()

            mode_text, reason_text = _infer_paren_mode_from_text(full_text)
            mode_val, reason_val = _infer_paren_mode_from_values(raw_parens)
            paren_mode = mode_text or mode_val or "se"
            paren_mode_source = "text" if mode_text else ("values" if mode_val else "default")
            paren_mode_reason = "; ".join([x for x in [reason_text, reason_val, f"chosen={paren_mode_source}:{paren_mode}"] if x])

            def compute_abs_t_and_fields(coef: float, paren: float) -> tuple[float, float, float | None, str]:
                if paren_mode == "t":
                    t_val = float(paren)
                    abs_t = abs(t_val)
                    return abs_t, t_val, None, "paren_t"
                se = abs(float(paren))
                if se == 0:
                    return 0.0, 0.0, None, "invalid_se_zero"
                t_val = float(coef / se)
                abs_t = abs(t_val)
                return abs_t, t_val, float(se), "coef_over_se"

            def drop_reasons(rec: dict[str, Any], *, abs_t: float, se_val: float | None) -> list[str]:
                reasons: list[str] = []
                cell = str(rec.get("cell_text_snippet") or "")
                se_cell = str(rec.get("se_cell_text_snippet") or "")
                coef_val = float(rec.get("coef") or 0.0)
                paren_val = float(rec.get("paren") or 0.0)
                if _looks_like_column_label_pair(rec):
                    reasons.append("column_label_like")
                if _looks_like_citation_cell(cell) or _looks_like_citation_cell(se_cell):
                    reasons.append("citation_like_text")
                if _looks_like_year(paren_val):
                    reasons.append("paren_looks_like_year")
                if _looks_like_year(coef_val) and str(rec.get("coef_raw") or "").strip().isdigit():
                    reasons.append("coef_looks_like_year")
                if paren_mode == "t" and abs_t > 30:
                    reasons.append("abs_t_gt_30(paren_t)")
                if paren_mode != "t":
                    if se_val is None:
                        reasons.append("missing_se")
                    else:
                        if se_val > 50:
                            reasons.append("se_gt_50")
                        if abs_t > 80:
                            reasons.append("abs_t_gt_80(coef_over_se)")
                return reasons

            try:
                with tests_path.open("w", encoding="utf-8") as out:
                    for rec2 in raw_pairs:
                        coef = float(rec2.get("coef") or 0.0)
                        paren = float(rec2.get("paren") or 0.0)
                        abs_t, t_val, se_val, t_mode = compute_abs_t_and_fields(coef, paren)
                        reasons = drop_reasons(rec2, abs_t=abs_t, se_val=se_val)
                        if reasons:
                            for r in reasons:
                                filter_counts[r] += 1
                            if len(dropped_examples) < 20:
                                dropped_examples.append(
                                    {
                                        "page": rec2.get("page"),
                                        "table_index": rec2.get("table_index"),
                                        "coef": coef,
                                        "paren": paren,
                                        "cell_text_snippet": rec2.get("cell_text_snippet"),
                                        "se_cell_text_snippet": rec2.get("se_cell_text_snippet"),
                                        "reasons": reasons,
                                    }
                                )
                            continue

                        p2 = float(math.erfc(abs_t / math.sqrt(2.0))) if abs_t >= 0 else None
                        if p2 is None or not (0.0 <= p2 <= 1.0):
                            filter_counts["invalid_p"] += 1
                            continue

                        rec2_out = dict(rec2)
                        rec2_out.update(
                            {
                                "extractor_version": EXTRACTOR_VERSION,
                                "paren_mode_assumed": paren_mode,
                                "t_mode": t_mode,
                                "se": se_val,
                                "t": t_val,
                                "abs_t": abs_t,
                                "p_approx_2s": p2,
                            }
                        )
                        out.write(json.dumps(rec2_out, ensure_ascii=False) + "\n")

                        w1 = _in_window(abs_t, center=1.96, delta=0.20)
                        if w1 != 0:
                            t_near_196["total"] += 1
                            if w1 < 0:
                                t_near_196["left"] += 1
                            else:
                                t_near_196["right"] += 1
                        w2 = _in_window(abs_t, center=1.645, delta=0.20)
                        if w2 != 0:
                            t_near_1645["total"] += 1
                            if w2 < 0:
                                t_near_1645["left"] += 1
                            else:
                                t_near_1645["right"] += 1
                        w3 = _in_window(p2, center=0.05, delta=0.005)
                        if w3 != 0:
                            p_from_t_near_005["total"] += 1
                            if w3 < 0:
                                p_from_t_near_005["left"] += 1
                            else:
                                p_from_t_near_005["right"] += 1
                        w4 = _in_window(p2, center=0.10, delta=0.005)
                        if w4 != 0:
                            p_from_t_near_010["total"] += 1
                            if w4 < 0:
                                p_from_t_near_010["left"] += 1
                            else:
                                p_from_t_near_010["right"] += 1
                        if p2 <= 0.05:
                            p_from_t_sig_005 += 1
                            pcurve_sig_n += 1
                            if p2 <= 0.025:
                                pcurve_low_half += 1
                            else:
                                pcurve_high_half += 1
                        if p2 <= 0.10:
                            p_from_t_sig_010 += 1

                        p = rec2_out.get("page")
                        ti = rec2_out.get("table_index")
                        if isinstance(p, int) and isinstance(ti, int):
                            tables_seen_set.add((p, ti))
                        pairs_seen += 1
                        if len(sample_pairs) < 50:
                            sample_pairs.append(rec2_out)
            except Exception:
                pairs_seen = 0
                tables_seen_set = set()
                sample_pairs = []
                p_from_t_near_005 = {"left": 0, "right": 0, "total": 0}
                p_from_t_near_010 = {"left": 0, "right": 0, "total": 0}
                p_from_t_sig_005 = 0
                p_from_t_sig_010 = 0
                pcurve_sig_n = 0
                pcurve_low_half = 0
                pcurve_high_half = 0
                filter_counts = Counter()
                dropped_examples = []

            # Write a sidecar meta file so we can reuse tests safely across extractor versions.
            try:
                meta_obj = {
                    "extractor_version": EXTRACTOR_VERSION,
                    "cache_key": cache_key,
                    "max_pages_per_paper": int(args.max_pages_per_paper),
                    "max_pdf_pages": int(args.max_pdf_pages) if args.max_pdf_pages is not None else None,
                    "paper_id": pdf_path.stem,
                    "pdf_relpath": str(pdf_path.relative_to(corpus_dir)).replace("\\", "/"),
                    "generated_at": _now_iso(),
                    "candidate_pages": candidate_pages,
                    "reference_pages_detected": sorted(ref_pages)[:200],
                    "paren_mode": paren_mode,
                    "paren_mode_source": paren_mode_source,
                    "paren_mode_reason": paren_mode_reason,
                    "pairs_raw": int(pairs_seen_raw),
                    "pairs_kept": int(pairs_seen),
                    "tables_raw": [list(x) for x in sorted(tables_seen_raw_set)],
                    "tables_kept": [list(x) for x in sorted(tables_seen_set)],
                    "filter_counts": dict(filter_counts),
                    "dropped_examples": dropped_examples,
                }
                tests_meta_path.write_text(json.dumps(meta_obj, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

        multiple_terms = _find_terms(full_text, [r"\bbonferroni\b", r"\bfdr\b", r"\bfwer\b", r"\bmultiple\s+testing\b", r"\bbenjamini\b", r"\bholm\b"])
        has_correction = bool(multiple_terms > 0)
        spec_terms = _find_terms(full_text, [r"\bspecification\b", r"\bmultiverse\b", r"\bsearch\b", r"\bdata\s+snoop", r"\bmodel\s+selection\b"])
        p_near_005 = _bin_counts(pvals, 0.045, 0.05, 0.055)
        p_near_010 = _bin_counts(pvals, 0.095, 0.10, 0.105)
        pcurve_z = _caliper_z(pcurve_low_half, pcurve_high_half, direction="left")
        pcurve_p2s = _caliper_p_two_sided(pcurve_low_half, pcurve_high_half)

        feats: dict[str, Any] = {
            "extractor_version": EXTRACTOR_VERSION,
            "cache_key": cache_key,
            "max_pages_per_paper": int(args.max_pages_per_paper),
            "max_pdf_pages": int(args.max_pdf_pages) if args.max_pdf_pages is not None else None,
            "paper_id": pdf_path.stem,
            "pdf_relpath": str(pdf_path.relative_to(corpus_dir)).replace("\\", "/"),
            "sha256": sha,
            "pages": page_count,
            "tests_relpath": str(tests_path.relative_to(corpus_dir)).replace("\\", "/"),
            "tests_meta_relpath": str(tests_meta_path.relative_to(corpus_dir)).replace("\\", "/"),
            "paren_mode": paren_mode,
            "paren_mode_source": paren_mode_source,
            "paren_mode_reason": paren_mode_reason,
            "reference_pages_detected_n": int(len(ref_pages)),
            "tables_seen_raw": int(len(tables_seen_raw_set)),
            "t_pairs_seen_raw": int(pairs_seen_raw),
            "t_pairs_keep_rate": (float(pairs_seen) / float(pairs_seen_raw)) if pairs_seen_raw else None,
            "filter_counts": dict(filter_counts),
            "extracted_text_chars": len(full_text),
            "p_values_found": len(pvals),
            "near_0_05_p": p_near_005,
            "near_0_10_p": p_near_010,
            "star_count_fulltext": full_text.count("*"),
            "table_mentions_fulltext": len(re.findall(r"(?i)\btable\b", full_text)),
            "robust_mentions_fulltext": len(re.findall(r"(?i)\brobust(?:ness)?\b", full_text)),
            "multiple_testing_terms_fulltext": multiple_terms,
            "spec_search_terms_fulltext": spec_terms,
            "has_multiple_testing_correction": has_correction,
            "candidate_pages": candidate_pages,
            "tables_seen": int(len(tables_seen_set)),
            "t_pairs_seen": int(pairs_seen),
            "t_pairs_sample_n": int(len(sample_pairs)),
            "t_pairs_sample": sample_pairs,
            "t_near_1_96": t_near_196,
            "t_near_1_645": t_near_1645,
            "p_from_t_near_0_05": p_from_t_near_005,
            "p_from_t_near_0_10": p_from_t_near_010,
            "p_from_t_significant_0_05": int(p_from_t_sig_005),
            "p_from_t_significant_0_10": int(p_from_t_sig_010),
            "p_from_t_caliper_0_05_z": _caliper_z(
                int(p_from_t_near_005.get("left") or 0), int(p_from_t_near_005.get("right") or 0), direction="left"
            ),
            "p_from_t_caliper_0_05_p": _caliper_p_two_sided(
                int(p_from_t_near_005.get("left") or 0), int(p_from_t_near_005.get("right") or 0)
            ),
            "p_from_t_caliper_0_10_z": _caliper_z(
                int(p_from_t_near_010.get("left") or 0), int(p_from_t_near_010.get("right") or 0), direction="left"
            ),
            "p_from_t_caliper_0_10_p": _caliper_p_two_sided(
                int(p_from_t_near_010.get("left") or 0), int(p_from_t_near_010.get("right") or 0)
            ),
            "p_regex_caliper_0_05_z": _caliper_z(
                int(p_near_005.get("left") or 0), int(p_near_005.get("right") or 0), direction="left"
            ),
            "p_regex_caliper_0_05_p": _caliper_p_two_sided(
                int(p_near_005.get("left") or 0), int(p_near_005.get("right") or 0)
            ),
            "p_regex_caliper_0_10_z": _caliper_z(
                int(p_near_010.get("left") or 0), int(p_near_010.get("right") or 0), direction="left"
            ),
            "p_regex_caliper_0_10_p": _caliper_p_two_sided(
                int(p_near_010.get("left") or 0), int(p_near_010.get("right") or 0)
            ),
            "t_caliper_1_96_z": _caliper_z(
                int(t_near_196.get("left") or 0), int(t_near_196.get("right") or 0), direction="right"
            ),
            "t_caliper_1_96_p": _caliper_p_two_sided(int(t_near_196.get("left") or 0), int(t_near_196.get("right") or 0)),
            "t_caliper_1_645_z": _caliper_z(
                int(t_near_1645.get("left") or 0), int(t_near_1645.get("right") or 0), direction="right"
            ),
            "t_caliper_1_645_p": _caliper_p_two_sided(
                int(t_near_1645.get("left") or 0), int(t_near_1645.get("right") or 0)
            ),
            "pcurve_significant_n": int(pcurve_sig_n),
            "pcurve_low_half": int(pcurve_low_half),
            "pcurve_high_half": int(pcurve_high_half),
            "pcurve_right_skew_z": float(pcurve_z),
            "pcurve_right_skew_p": pcurve_p2s,
        }

        score, components = _compute_offline_risk_score(feats)
        feats["offline_risk_score"] = score
        feats["offline_risk_level"] = _risk_level(score)
        feats["offline_risk_components"] = components
        feats["generated_at"] = _now_iso()

        cache_path.write_text(json.dumps(feats, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(feats)

        print(f"[{_now_iso()}] ({i}/{len(pdfs)}) {pdf_path.name} pages={page_count} score={score}")

    # Write flat CSV (serialize dict fields as JSON)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    def cell(v: Any) -> Any:
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False)
        return v

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: cell(r.get(k)) for k in fieldnames})

    print(f"[{_now_iso()}] wrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
