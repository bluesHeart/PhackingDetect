#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import dataclasses
import hashlib
import io
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from PIL import Image


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _slugify(s: str, *, max_len: int = 80) -> str:
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


def _sha256_file(path: Path, *, max_bytes: int = 10_000_000) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        remaining = max_bytes
        while remaining > 0:
            chunk = f.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def _extract_json_obj(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty response")

    # Strip markdown fences if present
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    # Fast path
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try extracting the first JSON object inside
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    candidate = text[start : end + 1]
    obj = json.loads(candidate)
    if not isinstance(obj, dict):
        raise ValueError("Top-level JSON is not an object")
    return obj


def _safe_json(obj: Any) -> Any:
    try:
        if obj is None:
            return None
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if isinstance(obj, (dict, list, str, int, float, bool)):
            return obj
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return {"_repr": repr(obj)}


def _load_offline_risk_score_from_features(*, corpus_dir: Path, paper_id: str) -> float | None:
    """
    Best-effort: load offline_risk_score from <corpus_dir>/features.csv.
    Avoids pandas dependency inside the agent.
    """
    try:
        features_csv = corpus_dir / "features.csv"
        if not features_csv.exists() or features_csv.stat().st_size <= 0:
            return None
        with features_csv.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                pid = (row.get("paper_id") or "").strip()
                if pid != paper_id:
                    continue
                v = row.get("offline_risk_score")
                if v is None:
                    return None
                return float(v)
    except Exception:
        return None
    return None


def _match_corpus_features_row(
    *,
    corpus_dir: Path,
    paper_id: str | None = None,
    sha256: str | None = None,
) -> dict[str, str] | None:
    """
    Best-effort: find the features.csv row that corresponds to the PDF.
    Prefer matching by sha256 (robust to renames), fall back to paper_id.
    """
    try:
        features_csv = corpus_dir / "features.csv"
        if not features_csv.exists() or features_csv.stat().st_size <= 0:
            return None
        sha256 = (sha256 or "").strip().lower()
        paper_id = (paper_id or "").strip()
        with features_csv.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                if sha256:
                    if (row.get("sha256") or "").strip().lower() == sha256:
                        return {k: str(v) if v is not None else "" for k, v in row.items()}
                if paper_id:
                    if (row.get("paper_id") or "").strip() == paper_id:
                        return {k: str(v) if v is not None else "" for k, v in row.items()}
    except Exception:
        return None
    return None


def _extract_table_captions_by_page(page_texts: list[str]) -> dict[int, str]:
    """
    Best-effort table caption mapping:
    - captures caption lines like "Table 3. Determinants of Disclosure"
    - propagates across "Panel A/B/C" continuation pages.
    """
    out: dict[int, str] = {}
    last_caption = ""
    for i, t in enumerate(page_texts, start=1):
        text = t or ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        head = lines[:40]

        caption = ""
        for ln in head:
            ln2 = re.sub(r"\s+", " ", ln.strip())
            m = re.match(r"(?i)^table\s+([a-z]{0,3}\d+)[\.:]\s*(.+)$", ln2)
            if not m:
                continue
            caption = f"Table {m.group(1)}. {m.group(2)}".strip()
            break

        if not caption and last_caption:
            head_txt = " ".join(head[:10]).lower()
            if "panel a" in head_txt or "panel b" in head_txt or "panel c" in head_txt:
                caption = last_caption

        if caption:
            out[int(i)] = caption
            last_caption = caption
    return out


def _extract_figure_captions_by_page(page_texts: list[str]) -> dict[int, str]:
    """
    Best-effort figure caption mapping:
    - captures caption lines like "Figure 2. Event study"
    - propagates across "Panel A/B/C" continuation pages.
    """
    out: dict[int, str] = {}
    last_caption = ""
    for i, t in enumerate(page_texts, start=1):
        text = t or ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        head = lines[:50]

        caption = ""
        for ln in head:
            ln2 = re.sub(r"\s+", " ", ln.strip())
            m = re.match(r"(?i)^figure\s+([a-z]{0,3}\d+)[\.:]\s*(.+)$", ln2)
            if not m:
                continue
            caption = f"Figure {m.group(1)}. {m.group(2)}".strip()
            break

        if not caption and last_caption:
            head_txt = " ".join(head[:12]).lower()
            if "panel a" in head_txt or "panel b" in head_txt or "panel c" in head_txt:
                caption = last_caption

        if caption:
            out[int(i)] = caption
            last_caption = caption
    return out


def _md_escape_cell(s: str) -> str:
    s = (s or "").replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    # Markdown table cell escape
    return s.replace("|", "\\|")


def _bbox4(v: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(v, list) or len(v) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
    except Exception:
        return None
    if not (x0 == x0 and y0 == y0 and x1 == x1 and y1 == y1):
        return None
    return (x0, y0, x1, y1)


def _bbox_overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    lo = max(min(a0, a1), min(b0, b1))
    hi = min(max(a0, a1), max(b0, b1))
    return max(0.0, hi - lo)


def _page_lines_from_words(page: fitz.Page) -> list[dict[str, Any]]:
    """
    Reconstruct text lines with bounding boxes from PyMuPDF `words`.
    Each line: {text: str, bbox: (x0,y0,x1,y1)}
    """
    words = page.get_text("words")  # type: ignore
    groups: dict[tuple[int, int], list[tuple[int, float, float, float, float, str]]] = {}
    for w in words or []:
        if not isinstance(w, (list, tuple)) or len(w) < 8:
            continue
        x0, y0, x1, y1, text, block_no, line_no, word_no = w[:8]
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            key = (int(block_no), int(line_no))
            item = (int(word_no), float(x0), float(y0), float(x1), float(y1), str(text))
        except Exception:
            continue
        groups.setdefault(key, []).append(item)

    lines: list[dict[str, Any]] = []
    for (_b, _l), items in groups.items():
        items.sort(key=lambda t: t[0])
        texts: list[str] = []
        x0s: list[float] = []
        y0s: list[float] = []
        x1s: list[float] = []
        y1s: list[float] = []
        for _wn, x0, y0, x1, y1, txt in items:
            texts.append(txt)
            x0s.append(x0)
            y0s.append(y0)
            x1s.append(x1)
            y1s.append(y1)
        line_text = re.sub(r"\s+", " ", " ".join(texts)).strip()
        if not line_text:
            continue
        lines.append({"text": line_text, "bbox": (min(x0s), min(y0s), max(x1s), max(y1s))})
    lines.sort(key=lambda d: (float(d["bbox"][1]), float(d["bbox"][0])))
    return lines


def _is_model_number_label(s: str) -> bool:
    t = re.sub(r"\s+", "", (s or "")).strip()
    if not t:
        return False
    t2 = t
    if t2.startswith("(") and t2.endswith(")"):
        t2 = t2[1:-1]
    return t2.isdigit()


def _infer_row_label(*, lines: list[dict[str, Any]], table_bbox: tuple[float, float, float, float], cell_bbox: tuple[float, float, float, float]) -> str:
    tx0, ty0, tx1, ty1 = table_bbox
    cx0, cy0, cx1, cy1 = cell_bbox
    y_mid = (cy0 + cy1) / 2.0
    band = max(5.0, (cy1 - cy0) * 0.9)
    best: tuple[float, str] | None = None

    for ln in lines:
        bbox = ln.get("bbox")
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        # constrain to table region
        if y1 < ty0 - 2 or y0 > ty1 + 2:
            continue
        if x1 > cx0 - 2:
            continue
        # y band overlap
        if _bbox_overlap_1d(y0, y1, y_mid - band, y_mid + band) <= 0:
            continue
        text = str(ln.get("text") or "").strip()
        if not text:
            continue
        # Prefer labels with letters.
        has_letters = re.search(r"[A-Za-z]", text) is not None
        # Score: closer in y is better; closer to coef column (x1 near cx0) is better.
        y_line_mid = (y0 + y1) / 2.0
        dy = abs(y_line_mid - y_mid)
        dx = max(0.0, cx0 - x1)
        score = dy + 0.02 * dx + (0.0 if has_letters else 8.0)
        if best is None or score < best[0]:
            best = (score, text)
    return _md_escape_cell(best[1]) if best else ""


def _infer_col_label(
    *,
    lines: list[dict[str, Any]],
    table_bbox: tuple[float, float, float, float],
    cell_bbox: tuple[float, float, float, float],
    header_bottom_y: float | None = None,
) -> str:
    tx0, ty0, tx1, ty1 = table_bbox
    cx0, cy0, cx1, cy1 = cell_bbox
    candidates: list[tuple[float, float, str]] = []
    for ln in lines:
        bbox = ln.get("bbox")
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        if y1 >= cy0 - 1:
            continue
        if y1 < ty0 - 2 or y0 > ty1 + 2:
            continue
        if header_bottom_y is not None and y1 > float(header_bottom_y) - 1.0:
            continue
        overlap = _bbox_overlap_1d(x0, x1, cx0, cx1)
        if overlap <= 0:
            continue
        text = str(ln.get("text") or "").strip()
        if not text:
            continue
        dy = max(0.0, cy0 - y1)
        candidates.append((dy, -overlap, text))
    candidates.sort(key=lambda t: (t[0], t[1]))
    if not candidates:
        return ""
    best_text = candidates[0][2]
    best = _md_escape_cell(best_text)

    # If the closest header is just a model number "(1)", try to prepend a more descriptive line above.
    if _is_model_number_label(best_text):
        # Find the nearest line above (within ~40pt) that overlaps x and has letters.
        # Re-run with extra info (need bbox), so do a second pass.
        best_extra: tuple[float, str] | None = None
        for ln in lines:
            bbox = ln.get("bbox")
            if not isinstance(bbox, tuple) or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            if y1 >= cy0 - 1:
                continue
            if _bbox_overlap_1d(x0, x1, cx0, cx1) <= 0:
                continue
            text = str(ln.get("text") or "").strip()
            if not text or re.search(r"[A-Za-z]", text) is None:
                continue
            dy = max(0.0, cy0 - y1)
            if dy > 80:
                continue
            if best_extra is None or dy < best_extra[0]:
                best_extra = (dy, text)
        if best_extra:
            return f"{_md_escape_cell(best_extra[1])} {best}".strip()
    return best


def _annotate_test_examples_with_row_col_labels(
    doc: fitz.Document,
    examples: list[dict[str, Any]],
    *,
    table_min_coef_y0_by_table: dict[str, Any] | None = None,
) -> None:
    if not examples:
        return
    lines_cache: dict[int, list[dict[str, Any]]] = {}

    def lines_for_page(p1: int) -> list[dict[str, Any]]:
        if p1 in lines_cache:
            return lines_cache[p1]
        try:
            page = doc.load_page(int(p1) - 1)
        except Exception:
            lines_cache[p1] = []
            return []
        lines_cache[p1] = _page_lines_from_words(page)
        return lines_cache[p1]

    for ex in examples:
        if not isinstance(ex, dict):
            continue
        try:
            page = int(ex.get("page") or 0)
        except Exception:
            continue
        if page <= 0:
            continue
        tb = _bbox4(ex.get("table_bbox"))
        cb = _bbox4(ex.get("coef_cell_bbox"))
        if tb is None or cb is None:
            continue
        lines = lines_for_page(page)
        if not lines:
            continue
        header_bottom_y = None
        try:
            if isinstance(table_min_coef_y0_by_table, dict):
                tidx = ex.get("table_index")
                if isinstance(tidx, int):
                    key = f"{page}:{tidx}"
                    v = table_min_coef_y0_by_table.get(key)
                    if v is not None:
                        header_bottom_y = float(v)
        except Exception:
            header_bottom_y = None
        try:
            row = _infer_row_label(lines=lines, table_bbox=tb, cell_bbox=cb)
            col = _infer_col_label(lines=lines, table_bbox=tb, cell_bbox=cb, header_bottom_y=header_bottom_y)
        except Exception:
            continue
        if row:
            ex["row_label"] = row
        if col:
            ex["col_label"] = col


class LLMRunLogger:
    def __init__(self, logs_dir: Path, *, model: str, base_url: str) -> None:
        self.logs_dir = logs_dir
        self.calls_dir = logs_dir / "calls"
        self.calls_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.base_url = base_url
        self._call_n = 0
        self.jsonl_path = logs_dir / "llm_calls.jsonl"

    def _next_id(self) -> str:
        self._call_n += 1
        return f"{self._call_n:04d}"

    def _write_text(self, path: Path, text: str) -> None:
        path.write_text(text or "", encoding="utf-8", errors="replace")

    def log_call(
        self,
        *,
        kind: str,
        system_prompt: str,
        user_prompt: str,
        image_path: str | None,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
        response_content: str | None,
        response_raw: Any | None,
        error: str | None,
        duration_ms: int,
        notes: str | None = None,
    ) -> None:
        call_id = self._next_id()
        sys_path = self.calls_dir / f"call_{call_id}_system.txt"
        user_path = self.calls_dir / f"call_{call_id}_user.txt"
        resp_path = self.calls_dir / f"call_{call_id}_response.txt"
        raw_path = self.calls_dir / f"call_{call_id}_response_raw.json"

        self._write_text(sys_path, system_prompt)
        self._write_text(user_path, user_prompt)
        self._write_text(resp_path, response_content or "")
        raw_path.write_text(json.dumps(_safe_json(response_raw), ensure_ascii=False, indent=2), encoding="utf-8")

        record = {
            "call_id": call_id,
            "ts": _now_iso(),
            "kind": kind,
            "model": self.model,
            "base_url": self.base_url,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "json_mode": bool(json_mode),
            "image_path": image_path,
            "system_prompt_path": str(sys_path.relative_to(self.logs_dir)).replace("\\", "/"),
            "user_prompt_path": str(user_path.relative_to(self.logs_dir)).replace("\\", "/"),
            "response_path": str(resp_path.relative_to(self.logs_dir)).replace("\\", "/"),
            "response_raw_path": str(raw_path.relative_to(self.logs_dir)).replace("\\", "/"),
            "response_chars": len(response_content or ""),
            "duration_ms": duration_ms,
            "error": error,
            "notes": notes,
        }
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _merge_unique(dst: list[str], src: list[str]) -> list[str]:
    seen = {x.strip() for x in dst if isinstance(x, str) and x.strip()}
    for x in src:
        if not isinstance(x, str):
            continue
        v = x.strip()
        if not v or v in seen:
            continue
        dst.append(v)
        seen.add(v)
    return dst


def _extract_apa_references(method_md_path: Path) -> list[str]:
    if not method_md_path.exists():
        return []
    text = method_md_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(
        r"^##\s+(?:参考文献（APA）|References\b.*APA.*)\s*$",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    if not m:
        return []
    tail = text[m.end() :].strip("\n")
    lines = tail.splitlines()
    refs: list[str] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("## "):
            break
        if not line.strip():
            continue
        if line.lstrip().startswith("- "):
            if current:
                refs.append(" ".join(current).strip())
                current = []
            current.append(line.lstrip()[2:].strip())
        else:
            if current:
                current.append(line.strip())
    if current:
        refs.append(" ".join(current).strip())
    # De-dupe while preserving order
    out: list[str] = []
    seen = set()
    for r in refs:
        rr = re.sub(r"\s+", " ", r).strip()
        if not rr or rr in seen:
            continue
        out.append(rr)
        seen.add(rr)
    return out


def _normalize_for_contains(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _ensure_method_references(report_md: str, apa_refs: list[str]) -> str:
    if not report_md or not apa_refs:
        return report_md
    hay = _normalize_for_contains(report_md)
    missing = [r for r in apa_refs if _normalize_for_contains(r) not in hay]
    if not missing:
        return report_md
    lines = []
    lines.append(report_md.rstrip())
    lines.append("")
    lines.append("## References (APA, with DOI) — Method Sources (Auto-appended)")
    lines.append("")
    for r in apa_refs:
        lines.append(f"- {r}")
    lines.append("")
    return "\n".join(lines)


def _p_value_regex_hits(text: str) -> list[float]:
    if not text:
        return []
    hits: list[float] = []
    # Numeric p-values
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


def _in_window(x: float, *, center: float, delta: float) -> int:
    """
    Returns:
      -1 if x is in [center-delta, center] (left)
      +1 if x is in (center, center+delta] (right)
       0 otherwise
    """
    try:
        v = float(x)
    except Exception:
        return 0
    if v != v:
        return 0
    lo = float(center) - float(delta)
    hi = float(center) + float(delta)
    if lo <= v <= float(center):
        return -1
    if float(center) < v <= hi:
        return 1
    return 0


def _fmt_test_anchor(rec: dict[str, Any]) -> str:
    try:
        page = rec.get("page")
        table_index = rec.get("table_index")
        coef_raw = (rec.get("coef_raw") or rec.get("cell_text_snippet") or "").strip()
        se_snip = (rec.get("se_cell_text_snippet") or "").strip()
        t = rec.get("t")
        p2 = rec.get("p_approx_2s")
        bits: list[str] = []
        if page is not None:
            bits.append(f"p.{page}")
        if isinstance(table_index, int):
            bits.append(f"Table#{table_index + 1}")
        if coef_raw:
            bits.append(f"coef={coef_raw}")
        if se_snip:
            bits.append(f"paren={se_snip}")
        if isinstance(t, (int, float)) and t == t:
            bits.append(f"|t|≈{abs(float(t)):.2f}")
        if isinstance(p2, (int, float)) and p2 == p2:
            bits.append(f"p≈{float(p2):.3g}")
        return " · ".join(bits) if bits else "(unavailable)"
    except Exception:
        return "(unavailable)"


def _load_tests_summary(tests_path: Path) -> tuple[dict[str, Any], dict[int, list[dict[str, str]]]]:
    """
    Returns:
      (summary_dict, borderline_by_page)

    borderline_by_page[page] contains up to a few formatted anchors suitable for
    page_evidence.borderline_results.
    """
    t_near_196 = {"left": 0, "right": 0, "total": 0}
    t_near_1645 = {"left": 0, "right": 0, "total": 0}
    p_from_t_near_005 = {"left": 0, "right": 0, "total": 0}
    p_from_t_near_010 = {"left": 0, "right": 0, "total": 0}
    p_sig_005 = 0
    p_sig_010 = 0
    pcurve_sig_n = 0
    pcurve_low_half = 0
    pcurve_high_half = 0

    borderline_by_page: dict[int, list[tuple[float, dict[str, str]]]] = {}
    p005_examples: list[tuple[float, dict[str, Any]]] = []
    t196_examples: list[tuple[float, dict[str, Any]]] = []
    p005_page_counts: dict[int, int] = {}
    t196_page_counts: dict[int, int] = {}
    t_exact_2_in_t196 = 0
    table_min_coef_y0_by_table: dict[str, float] = {}

    n_pairs = 0
    with tests_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue

            n_pairs += 1
            page = rec.get("page")
            if not isinstance(page, int):
                page = None
            table_index = rec.get("table_index")
            if not isinstance(table_index, int):
                table_index = None

            # Track the top of the *data* region for each table: the minimum y0 of coefficient cells.
            # This helps separate column headers from data rows when inferring col labels.
            if page is not None and table_index is not None:
                cb = _bbox4(rec.get("coef_cell_bbox"))
                if cb is not None:
                    key = f"{page}:{table_index}"
                    y0 = float(cb[1])
                    prev = table_min_coef_y0_by_table.get(key)
                    if prev is None or y0 < float(prev):
                        table_min_coef_y0_by_table[key] = y0

            at = rec.get("abs_t")
            if not isinstance(at, (int, float)) or at != at:
                t = rec.get("t")
                at = abs(float(t)) if isinstance(t, (int, float)) and t == t else None
            if isinstance(at, (int, float)) and at == at:
                w1 = _in_window(float(at), center=1.96, delta=0.20)
                if w1 != 0:
                    t_near_196["total"] += 1
                    if w1 < 0:
                        t_near_196["left"] += 1
                    else:
                        t_near_196["right"] += 1
                w2 = _in_window(float(at), center=1.645, delta=0.20)
                if w2 != 0:
                    t_near_1645["total"] += 1
                    if w2 < 0:
                        t_near_1645["left"] += 1
                    else:
                        t_near_1645["right"] += 1

            p2 = rec.get("p_approx_2s")
            if isinstance(p2, (int, float)) and p2 == p2 and 0.0 <= float(p2) <= 1.0:
                p2f = float(p2)
                if p2f <= 0.05:
                    p_sig_005 += 1
                    pcurve_sig_n += 1
                    if p2f <= 0.025:
                        pcurve_low_half += 1
                    else:
                        pcurve_high_half += 1
                if p2f <= 0.10:
                    p_sig_010 += 1
                w3 = _in_window(p2f, center=0.05, delta=0.005)
                if w3 != 0:
                    p_from_t_near_005["total"] += 1
                    if w3 < 0:
                        p_from_t_near_005["left"] += 1
                    else:
                        p_from_t_near_005["right"] += 1
                    if page is not None:
                        p005_page_counts[page] = int(p005_page_counts.get(page, 0)) + 1
                    # keep example rows in-window
                    try:
                        d = abs(p2f - 0.05)
                        item = {
                            "page": page,
                            "table_index": rec.get("table_index"),
                            "row_index": rec.get("row_index"),
                            "col_index": rec.get("col_index"),
                            "table_bbox": rec.get("table_bbox"),
                            "coef_cell_bbox": rec.get("coef_cell_bbox"),
                            "coef_raw": rec.get("coef_raw") or rec.get("cell_text_snippet"),
                            "se_text": rec.get("se_cell_text_snippet"),
                            "abs_t": float(at) if isinstance(at, (int, float)) else None,
                            "p_approx_2s": p2f,
                            "stars": rec.get("stars"),
                            "anchor": _fmt_test_anchor(rec),
                        }
                        p005_examples.append((float(d), item))
                    except Exception:
                        pass
                w4 = _in_window(p2f, center=0.10, delta=0.005)
                if w4 != 0:
                    p_from_t_near_010["total"] += 1
                    if w4 < 0:
                        p_from_t_near_010["left"] += 1
                    else:
                        p_from_t_near_010["right"] += 1

            if page is not None:
                examples = borderline_by_page.setdefault(page, [])
                best_score = None
                why = None
                if isinstance(at, (int, float)) and at == at:
                    d196 = abs(float(at) - 1.96)
                    if d196 <= 0.20:
                        best_score = d196
                        why = "abs(t)≈1.96"
                        if abs(float(at) - 2.0) < 1e-9:
                            t_exact_2_in_t196 += 1
                        t196_page_counts[page] = int(t196_page_counts.get(page, 0)) + 1
                        try:
                            item = {
                                "page": page,
                                "table_index": rec.get("table_index"),
                                "row_index": rec.get("row_index"),
                                "col_index": rec.get("col_index"),
                                "table_bbox": rec.get("table_bbox"),
                                "coef_cell_bbox": rec.get("coef_cell_bbox"),
                                "coef_raw": rec.get("coef_raw") or rec.get("cell_text_snippet"),
                                "se_text": rec.get("se_cell_text_snippet"),
                                "abs_t": float(at),
                                "p_approx_2s": float(p2) if isinstance(p2, (int, float)) else None,
                                "stars": rec.get("stars"),
                                "anchor": _fmt_test_anchor(rec),
                            }
                            t196_examples.append((float(d196), item))
                        except Exception:
                            pass
                if isinstance(p2, (int, float)) and p2 == p2:
                    d005 = abs(float(p2) - 0.05)
                    if d005 <= 0.005 and (best_score is None or d005 < best_score):
                        best_score = d005
                        why = "p≈0.05"
                if best_score is not None and why:
                    examples.append((float(best_score), {"anchor": _fmt_test_anchor(rec), "why_borderline": str(why)}))
                    examples.sort(key=lambda x: x[0])
                    if len(examples) > 3:
                        del examples[3:]

    borderline_simple: dict[int, list[dict[str, str]]] = {k: [x[1] for x in v] for k, v in borderline_by_page.items()}

    # Sort and cap example lists
    p005_examples.sort(key=lambda x: x[0])
    t196_examples.sort(key=lambda x: x[0])
    p005_cap = 60
    t196_cap = 80
    p005_truncated = len(p005_examples) > p005_cap
    t196_truncated = len(t196_examples) > t196_cap
    p005_examples_out = [x[1] for x in p005_examples[:p005_cap]]
    t196_examples_out = [x[1] for x in t196_examples[:t196_cap]]

    pcurve_z = 0.0
    try:
        n_sig = int(pcurve_low_half) + int(pcurve_high_half)
        if n_sig > 0:
            pcurve_z = float((int(pcurve_low_half) - int(pcurve_high_half)) / math.sqrt(float(n_sig)))
    except Exception:
        pcurve_z = 0.0

    summary = {
        "t_pairs_seen": int(n_pairs),
        "t_near_1_96": t_near_196,
        "t_near_1_645": t_near_1645,
        "p_from_t_near_0_05": p_from_t_near_005,
        "p_from_t_near_0_10": p_from_t_near_010,
        "p_from_t_significant_0_05": int(p_sig_005),
        "p_from_t_significant_0_10": int(p_sig_010),
        "pcurve_significant_n": int(pcurve_sig_n),
        "pcurve_low_half": int(pcurve_low_half),
        "pcurve_high_half": int(pcurve_high_half),
        "pcurve_right_skew_z": float(pcurve_z),
        "p_from_t_near_0_05_examples": p005_examples_out,
        "p_from_t_near_0_05_examples_truncated": bool(p005_truncated),
        "t_near_1_96_examples": t196_examples_out,
        "t_near_1_96_examples_truncated": bool(t196_truncated),
        "p_from_t_near_0_05_page_counts": p005_page_counts,
        "t_near_1_96_page_counts": t196_page_counts,
        "t_near_1_96_exact_2_count": int(t_exact_2_in_t196),
        "table_min_coef_y0_by_table": table_min_coef_y0_by_table,
    }
    return summary, borderline_simple


def _clean_snippet(s: str, *, max_len: int = 220) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    if len(s) <= max_len:
        return s
    return s[: max(0, max_len - 1)].rstrip() + "…"


def _find_snippets(text: str, patterns: list[str], *, max_hits: int = 6, ctx: int = 80) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            start = max(0, m.start() - ctx)
            end = min(len(text), m.end() + ctx)
            snip = _clean_snippet(text[start:end])
            key = snip.lower()
            if not snip or key in seen:
                continue
            out.append(snip)
            seen.add(key)
            if len(out) >= max_hits:
                return out
    return out


def _offline_page_evidence(*, page_1based: int, extracted_text: str) -> dict[str, Any]:
    t = extracted_text or ""
    table_names = sorted(set(re.findall(r"(?i)\btable\s+[a-z]{0,3}\d+[a-z]?\b", t)))[:6]
    fig_names = sorted(set(re.findall(r"(?i)\bfigure\s+[a-z]?\d+[a-z]?\b", t)))[:6]

    sig_conv = _find_snippets(
        t,
        [
            r"\*+\s*p\s*[<≤]\s*0\.\d+",
            r"p\s*[<≤]\s*0\.\d+\s*\*+",
            r"significance\s+level",
            r"two[- ]sided",
        ],
        max_hits=4,
        ctx=60,
    )

    borderline_results: list[dict[str, str]] = []
    for m in re.finditer(r"(?i)\bp\s*(?:=|==|<|≤)\s*(0\.\d+)\b", t):
        try:
            pv = float(m.group(1))
        except Exception:
            continue
        if 0.045 <= pv <= 0.055:
            borderline_results.append({"anchor": _clean_snippet(t[max(0, m.start() - 60) : min(len(t), m.end() + 60)]), "why_borderline": "p≈0.05"})
        elif 0.095 <= pv <= 0.105:
            borderline_results.append({"anchor": _clean_snippet(t[max(0, m.start() - 60) : min(len(t), m.end() + 60)]), "why_borderline": "p≈0.10"})
        if len(borderline_results) >= 3:
            break

    signals: list[dict[str, Any]] = []

    def add_signal(signal: str, evidence: str, anchors: list[str]) -> None:
        if len(signals) >= 4:
            return
        anchors2 = [_clean_snippet(a, max_len=180) for a in anchors if a.strip()]
        signals.append({"signal": signal, "evidence": _clean_snippet(evidence, max_len=220), "anchors": anchors2[:3]})

    robust_anchors = _find_snippets(t, [r"\brobust(?:ness)?\b", r"\balternative\b", r"\bspecification\b", r"\bcontrols?\b"], max_hits=3)
    if robust_anchors:
        add_signal(
            "Specification/robustness variation mentioned",
            "Text indicates robustness/specification/controls variation on this page (possible researcher degrees of freedom).",
            robust_anchors,
        )

    mult_anchors = _find_snippets(
        t,
        [r"\bmultiple\s+(?:testing|hypothesis|comparisons?)\b", r"\bbonferroni\b", r"\bfdr\b", r"\bbenjamini\b", r"\bfwer\b"],
        max_hits=3,
    )
    if mult_anchors:
        add_signal(
            "Multiple testing / correction mentioned",
            "Page mentions multiple-testing adjustments or related terms (affects false-positive control).",
            mult_anchors,
        )

    placebo_anchors = _find_snippets(t, [r"\bplacebo\b", r"\bfalsification\b", r"\bpre[- ]trend\b"], max_hits=3)
    if placebo_anchors:
        add_signal("Placebo/falsification checks mentioned", "Page references placebo/falsification/pre-trend checks.", placebo_anchors)

    iv_anchors = _find_snippets(t, [r"\binstrument(?:al)?\b", r"\bfirst[- ]stage\b", r"\bf[- ]stat(?:istic)?\b", r"\bweak\s+instrument\b"], max_hits=3)
    if iv_anchors:
        add_signal("IV / weak-instrument cues", "Page references IV/first-stage/F-statistics (weak-IV risk check).", iv_anchors)

    notes: list[str] = []
    if not (sig_conv or borderline_results or signals):
        notes.append("Offline fallback: no high-signal patterns found in extracted text; page image saved for manual review.")
    if not t.strip():
        notes.append("Extracted text is empty on this page (PDF extraction limitation).")

    return {
        "page": page_1based,
        "page_role": None,
        "table_or_figure_names": (table_names + fig_names)[:6],
        "significance_conventions": sig_conv[:4],
        "borderline_results": borderline_results,
        "signals": signals,
        "notes": notes,
        "confidence_0_1": 0.25,
    }


def _offline_paper_meta(doc: fitz.Document, pdf_path: Path, *, first_page_text: str) -> dict[str, Any]:
    meta = doc.metadata or {}
    title = (meta.get("title") or "").strip() or pdf_path.stem
    author = (meta.get("author") or "").strip()
    subject = (meta.get("subject") or "").strip()

    year = None
    m = re.search(r"(19|20)\d{2}", subject)
    if m:
        year = int(m.group(0))
    doi_or_url = None
    m2 = re.search(r"(?i)\bdoi:\s*([0-9]{2}\.[0-9A-Za-z./_-]+)", subject)
    if m2:
        doi_or_url = f"https://doi.org/{m2.group(1)}"
    else:
        m3 = re.search(r"(?i)\b(10\.[0-9A-Za-z./_-]+)", subject)
        if m3:
            doi_or_url = f"https://doi.org/{m3.group(1)}"

    authors: list[str] = []
    if author:
        authors = [author]

    venue = None
    if subject:
        venue = subject.split(".")[0].strip() or None

    # Try extracting DOI/URL from first page text if metadata is empty.
    if not doi_or_url:
        m4 = re.search(r"(?i)\bdoi[:\s]*([0-9]{2}\.[0-9A-Za-z./_-]+)", first_page_text or "")
        if m4:
            doi_or_url = f"https://doi.org/{m4.group(1)}"

    return {
        "title": title or None,
        "authors": authors,
        "year": year,
        "venue_or_series": venue,
        "doi_or_url": doi_or_url,
    }


def _risk_level(score: int) -> str:
    if score >= 70:
        return "High"
    if score >= 40:
        return "Moderate"
    return "Low"


def _compose_offline_report(
    *,
    paper_title: str,
    paper_meta: dict[str, Any],
    heuristics: dict[str, Any],
    selected_pages: list[dict[str, Any]],
    page_evidence: list[dict[str, Any]],
    apa_refs: list[str],
    llm_disabled_reason: str | None,
) -> str:
    # Use offline_risk_score if available to avoid confusing mismatched scores.
    score = None
    try:
        s0 = heuristics.get("offline_risk_score")
        if isinstance(s0, (int, float)) and s0 == s0:
            score = int(round(float(s0)))
    except Exception:
        score = None
    if score is None:
        # Fallback: heuristic score (conservative; offline evidence is weaker).
        score_i = 20
        try:
            t_near = heuristics.get("t_near_1_96") or {}
            if isinstance(t_near, dict) and int(t_near.get("total") or 0) >= 30:
                left = int(t_near.get("left") or 0)
                right = int(t_near.get("right") or 0)
                if right >= left + 6:
                    score_i += 14
                elif right >= left + 2:
                    score_i += 8

            p_from_t = heuristics.get("p_from_t_near_0_05") or {}
            if isinstance(p_from_t, dict) and int(p_from_t.get("total") or 0) >= 10:
                left = int(p_from_t.get("left") or 0)
                right = int(p_from_t.get("right") or 0)
                if left >= right + 4:
                    score_i += 12
                elif left >= right + 1:
                    score_i += 6

            pcurve_z = heuristics.get("pcurve_right_skew_z")
            if isinstance(pcurve_z, (int, float)) and float(pcurve_z) < -1.0:
                score_i += 6
        except Exception:
            pass
        score = int(max(0, min(100, score_i)))

    level = _risk_level(score)

    refs_block = "\n".join(f"- {r}" for r in apa_refs)

    captions_by_page = heuristics.get("table_captions_by_page") or {}
    if not isinstance(captions_by_page, dict):
        captions_by_page = {}

    def _caption(p: int) -> str:
        v = captions_by_page.get(str(p)) if isinstance(captions_by_page, dict) else None
        if v is None and isinstance(captions_by_page, dict):
            v = captions_by_page.get(p)
        if isinstance(v, str) and v.strip():
            return v.strip()
        # fall back to page_evidence detected table names
        for ev in page_evidence:
            if not isinstance(ev, dict):
                continue
            if int(ev.get("page") or 0) != int(p):
                continue
            names = ev.get("table_or_figure_names")
            if isinstance(names, list) and names:
                return str(names[0])
        return ""

    # Pull concrete borderline examples from extracted tests (preferred).
    p005_examples = heuristics.get("p_from_t_near_0_05_examples") or []
    t196_examples = heuristics.get("t_near_1_96_examples") or []
    if not isinstance(p005_examples, list):
        p005_examples = []
    if not isinstance(t196_examples, list):
        t196_examples = []

    def _fmt_float(x: Any, *, digits: int = 3) -> str:
        try:
            v = float(x)
        except Exception:
            return ""
        if v != v:
            return ""
        return f"{v:.{digits}f}"

    def _fmt_p(x: Any) -> str:
        try:
            v = float(x)
        except Exception:
            return ""
        if v != v:
            return ""
        return f"{v:.4f}"

    def _as_int(x: Any) -> int:
        try:
            return int(x)
        except Exception:
            return 0

    def _table_title_for_page(page: int) -> str:
        cap = _caption(page)
        return cap if cap else "(unknown)"

    lines: list[str] = []
    lines.append(f"# Within-paper p-hacking risk screening report: {paper_title}")
    lines.append("")
    if paper_meta:
        title = paper_meta.get("title") or paper_title
        year = paper_meta.get("year")
        venue = paper_meta.get("venue_or_series")
        doi = paper_meta.get("doi_or_url")
        who = title
        if year:
            who = f"{who} ({year})"
        if venue:
            who = f"{who}. {venue}"
        lines.append(f"**Target:** {who}")
        if doi:
            lines.append(f"**DOI/URL：** {doi}")
        lines.append("")
    if llm_disabled_reason:
        lines.append(
            f"> Note: LLM unavailable in this run; falling back to heuristic/text extraction mode. Reason: "
            f"{_clean_snippet(llm_disabled_reason, max_len=240)}"
        )
        lines.append("")

    lines.append("## Bottom line")
    lines.append("")
    lines.append(f"- **Risk level: {level}** (offline mode; risk signals are not proof)")
    lines.append(f"- **Risk score: {score}/100**")
    if heuristics.get("tests_relpath"):
        lines.append(f"- Extracted table-test records: `{heuristics.get('tests_relpath')}`")
    lines.append("")

    # Key numeric evidence
    p_from_t = heuristics.get("p_from_t_near_0_05") or {}
    t_near = heuristics.get("t_near_1_96") or {}
    lines.append("## What was checked (interpretable summary)")
    lines.append("")
    lines.append(
        "We extract `(coef, parenthesized value)` pairs from regression tables. Treating the parenthesized value as a standard error (SE), "
        "we reconstruct `|t| = |coef/SE|` and approximate a two-sided `p`-value. "
        "We then summarize two near-threshold windows—`p≈0.05` and `|t|≈1.96`—to check for asymmetry consistent with "
        "``just significant'' clustering. This is a within-paper screening signal, not a judgement of intent."
    )
    lines.append("")

    lines.append("## Findings (numbers + locations)")
    lines.append("")
    if isinstance(p_from_t, dict) and int(p_from_t.get("total") or 0) > 0:
        left = int(p_from_t.get("left") or 0)
        right = int(p_from_t.get("right") or 0)
        total = int(p_from_t.get("total") or 0)
        lines.append("### 1) Near p≈0.05 window (p inferred from reconstructed t)")
        lines.append("")
        lines.append(f"- Window: `[0.045,0.05]` vs `(0.05,0.055]` (width 0.005)")
        lines.append(f"- Counts: left (more significant)={left}, right (less significant)={right}, total={total}")
        if right > 0:
            lines.append(f"- Ratio: left/right ≈ {left / right:.1f}x")
        lines.append(
            "- Interpretation: if researchers explore many specifications and selectively report results crossing the 0.05 cutoff, "
            "the left side can become inflated relative to the right. (Brodeur et al., 2016; Brodeur et al., 2020)"
        )
        lines.append("")

        if p005_examples:
            by_table: dict[str, dict[str, int]] = {}
            for ex in p005_examples:
                if not isinstance(ex, dict):
                    continue
                pn = _as_int(ex.get("page"))
                title = _table_title_for_page(pn)
                st = by_table.setdefault(title, {"p_left": 0, "p_right": 0})
                p2 = ex.get("p_approx_2s")
                try:
                    p2f = float(p2)
                except Exception:
                    continue
                if 0.045 <= p2f <= 0.05:
                    st["p_left"] += 1
                elif 0.05 < p2f <= 0.055:
                    st["p_right"] += 1

            lines.append("These borderline entries concentrate in:")
            lines.append("")
            lines.append("| table | p∈[0.045,0.05] | p∈(0.05,0.055] |")
            lines.append("|---|---:|---:|")
            for tbl, st in sorted(by_table.items(), key=lambda x: -(x[1]["p_left"] + x[1]["p_right"]))[:12]:
                if (st["p_left"] + st["p_right"]) == 0:
                    continue
                lines.append(f"| {tbl} | {st['p_left']} | {st['p_right']} |")
            lines.append("")

            lines.append("Specific near-0.05 entries (verify by page):")
            lines.append("")
            lines.append("| page | table/title | row | col | coef | paren | |t| | p≈ | stars |")
            lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|")
            for ex in p005_examples:
                if not isinstance(ex, dict):
                    continue
                pn = _as_int(ex.get("page"))
                row = _md_escape_cell(str(ex.get("row_label") or ""))
                col = _md_escape_cell(str(ex.get("col_label") or ""))
                lines.append(
                    f"| {pn} | {_table_title_for_page(pn)} | {row} | {col} | `{(ex.get('coef_raw') or '').strip()}` | `{(ex.get('se_text') or '').strip()}` | {_fmt_float(ex.get('abs_t'), digits=3)} | {_fmt_p(ex.get('p_approx_2s'))} | {_as_int(ex.get('stars'))} |"
                )
            if bool(heuristics.get("p_from_t_near_0_05_examples_truncated")):
                lines.append("")
                lines.append("> Note: list truncated (kept only the closest-to-0.05 entries); full list is in the tests JSONL.")
            lines.append("")

    if isinstance(t_near, dict) and int(t_near.get("total") or 0) > 0:
        left = int(t_near.get("left") or 0)
        right = int(t_near.get("right") or 0)
        total = int(t_near.get("total") or 0)
        lines.append("### 2) Near |t|≈1.96 window (from reconstructed t)")
        lines.append("")
        lines.append(f"- Window: `[1.76,1.96]` vs `(1.96,2.16]` (width 0.20)")
        lines.append(f"- Counts: left={left}, right={right}, total={total}")
        lines.append("")
        if t196_examples:
            by_table: dict[str, dict[str, int]] = {}
            for ex in t196_examples:
                if not isinstance(ex, dict):
                    continue
                pn = _as_int(ex.get("page"))
                title = _table_title_for_page(pn)
                st = by_table.setdefault(title, {"t_left": 0, "t_right": 0})
                at = ex.get("abs_t")
                try:
                    atf = float(at)
                except Exception:
                    continue
                if 1.76 <= atf <= 1.96:
                    st["t_left"] += 1
                elif 1.96 < atf <= 2.16:
                    st["t_right"] += 1

            lines.append("These near-|t|≈1.96 entries concentrate in:")
            lines.append("")
            lines.append("| table | |t|∈[1.76,1.96] | |t|∈(1.96,2.16] |")
            lines.append("|---|---:|---:|")
            for tbl, st in sorted(by_table.items(), key=lambda x: -(x[1]["t_left"] + x[1]["t_right"]))[:12]:
                if (st["t_left"] + st["t_right"]) == 0:
                    continue
                lines.append(f"| {tbl} | {st['t_left']} | {st['t_right']} |")
            lines.append("")

            lines.append("Entries closest to 1.96 (for quick verification):")
            lines.append("")
            lines.append("| page | table/title | row | col | coef | paren | |t| | p≈ | stars |")
            lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|")
            for ex in t196_examples[: min(25, len(t196_examples))]:
                if not isinstance(ex, dict):
                    continue
                pn = _as_int(ex.get("page"))
                row = _md_escape_cell(str(ex.get("row_label") or ""))
                col = _md_escape_cell(str(ex.get("col_label") or ""))
                lines.append(
                    f"| {pn} | {_table_title_for_page(pn)} | {row} | {col} | `{(ex.get('coef_raw') or '').strip()}` | `{(ex.get('se_text') or '').strip()}` | {_fmt_float(ex.get('abs_t'), digits=3)} | {_fmt_p(ex.get('p_approx_2s'))} | {_as_int(ex.get('stars'))} |"
                )
            if bool(heuristics.get("t_near_1_96_examples_truncated")):
                lines.append("")
                lines.append("> Note: list truncated (kept only the closest-to-1.96 entries); full list is in the tests JSONL.")
            lines.append("")

    # Quick “where” summary by page counts if available
    try:
        p_counts = heuristics.get("p_from_t_near_0_05_page_counts") or {}
        t_counts = heuristics.get("t_near_1_96_page_counts") or {}
        if isinstance(p_counts, dict) and p_counts:
            lines.append("### 3) Where do these signals concentrate?")
            lines.append("")
            lines.append("| page | table/title | p≈0.05 count | |t|≈1.96 count |")
            lines.append("|---:|---|---:|---:|")
            pages = set()
            for k in p_counts.keys():
                try:
                    pages.add(int(k))
                except Exception:
                    pass
            for k in t_counts.keys() if isinstance(t_counts, dict) else []:
                try:
                    pages.add(int(k))
                except Exception:
                    pass
            for pn in sorted(pages, key=lambda x: -(int(p_counts.get(str(x), p_counts.get(x, 0))) + int(t_counts.get(str(x), t_counts.get(x, 0))))):
                pc = int(p_counts.get(str(pn), p_counts.get(pn, 0)) or 0)
                tc = int(t_counts.get(str(pn), t_counts.get(pn, 0)) or 0)
                if pc == 0 and tc == 0:
                    continue
                lines.append(f"| {pn} | {_caption(pn)} | {pc} | {tc} |")
            lines.append("")
    except Exception:
        pass

    # Alternative explanations / artifacts
    lines.append("## Possible false positives (must consider)")
    lines.append("")
    t_exact2 = int(heuristics.get("t_near_1_96_exact_2_count") or 0)
    t_total = int((heuristics.get("t_near_1_96") or {}).get("total") or 0) if isinstance(heuristics.get("t_near_1_96"), dict) else 0
    if t_total > 0 and t_exact2 > 0:
        lines.append(
            f"- Within the `|t|≈1.96` window, **{t_exact2}/{t_total}** entries have `|t|` almost exactly **2.00**. "
            "This often indicates mechanical rounding/coarsening (e.g., few decimals reported), which can create artificial bunching."
        )
    lines.append(
        "- Star conventions (*, **, ***) and table formatting can amplify the visibility of near-threshold results; this does not imply intent."
    )
    lines.append("")

    lines.append("## Minimal-cost follow-up checks")
    lines.append("")
    lines.append(
        "- Open the cited pages (especially those with the highest near-threshold counts) and verify each entry using the anchors: "
        "confirm `coef` and `(SE)` were captured correctly."
    )
    lines.append(
        "- If higher-precision t-stats/p-values (or code/supplement) are available, recompute near-threshold summaries using unrounded values."
    )
    lines.append(
        "- If the appendix contains many specifications/subsamples/outcomes, consider multiple-testing control or full disclosure (FDR/FWER). "
        "(Harvey et al., 2015; Harvey, 2017)"
    )
    lines.append("")

    lines.append("## Machine summary (debug)")
    lines.append("")
    lines.append(f"- PDF pages: {heuristics.get('pdf_pages')}")
    lines.append(f"- Extracted text chars: {heuristics.get('extracted_text_chars')}")
    lines.append(f"- Extracted t-pairs (tables): {heuristics.get('t_pairs_seen')}")
    lines.append(f"- p-values found (regex): {heuristics.get('p_values_found')}")
    lines.append(f"- table_mentions_fulltext: {heuristics.get('table_mentions_fulltext')}")
    lines.append(f"- robust_mentions_fulltext: {heuristics.get('robust_mentions_fulltext')}")
    lines.append("")

    lines.append("## References (method, APA)")
    lines.append("")
    lines.append(refs_block if refs_block else "(no method references provided)")
    lines.append("")
    return "\n".join(lines)


@dataclasses.dataclass(frozen=True)
class PageFeatures:
    page_1based: int
    text_head: str
    text_len: int
    digit_count: int
    star_count: int
    p_mentions: int
    table_mentions: int
    robust_mentions: int
    score: float


def _page_features(page_1based: int, text: str) -> PageFeatures:
    t = text or ""
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    head = "\n".join(lines[:6])[:600]
    digit_count = sum(ch.isdigit() for ch in t)
    star_count = t.count("*")
    p_mentions = len(re.findall(r"(?i)\bp[- ]?value\b|\bp\s*[=<≤]\s*0\.", t))
    table_mentions = len(re.findall(r"(?i)\btable\b", t))
    robust_mentions = len(
        re.findall(r"(?i)\brobust(?:ness)?\b|\balternative specification\b|\bappendix\b|\bsensitivity\b", t)
    )
    # Cheap table-ish score
    score = (
        3.0 * table_mentions
        + 2.0 * robust_mentions
        + 1.5 * p_mentions
        + 0.004 * digit_count
        + 0.02 * star_count
        + (1.0 if re.search(r"(?i)\bobservations\b|\br-squared\b|standard errors?", t) else 0.0)
    )
    return PageFeatures(
        page_1based=page_1based,
        text_head=head,
        text_len=len(t),
        digit_count=digit_count,
        star_count=star_count,
        p_mentions=p_mentions,
        table_mentions=table_mentions,
        robust_mentions=robust_mentions,
        score=score,
    )


def _heuristic_select_pages(features: list[PageFeatures], *, max_pages: int) -> list[dict[str, Any]]:
    max_pages = max(1, int(max_pages))
    picks: list[dict[str, Any]] = []

    def looks_like_references(pf: PageFeatures) -> bool:
        head = (pf.text_head or "").lower()
        if "references" in head or "bibliography" in head:
            # If it also has many tables/p mentions, keep; otherwise skip
            return pf.table_mentions == 0 and pf.p_mentions == 0 and pf.star_count < 10
        return False

    # Always include first page for context
    picks.append({"page": 1, "role": "title/abstract", "reason": "Heuristic: always include first page for context."})

    candidates = sorted(features, key=lambda x: x.score, reverse=True)
    for pf in candidates:
        if pf.page_1based == 1:
            continue
        if looks_like_references(pf):
            continue
        picks.append(
            {
                "page": pf.page_1based,
                "role": "auto_pick",
                "reason": f"Heuristic score={pf.score:.3f} tables={pf.table_mentions} p_mentions={pf.p_mentions} stars={pf.star_count} robust={pf.robust_mentions}",
            }
        )
        if len(picks) >= max_pages:
            break

    # De-dupe while preserving order
    seen: set[int] = set()
    out: list[dict[str, Any]] = []
    for it in picks:
        p = it.get("page")
        if isinstance(p, int) and p not in seen:
            seen.add(p)
            out.append(it)
    return out[:max_pages]


def _render_page_image_jpg(
    doc: fitz.Document,
    page_index0: int,
    *,
    max_side_px: int = 1600,
    zoom: float = 2.0,
    crop_table_region: bool = True,
    jpeg_quality: int = 75,
) -> bytes:
    page = doc.load_page(page_index0)
    rect = page.rect

    clip: fitz.Rect | None = None
    if crop_table_region:
        # Try to crop to a numeric-dense region (often tables).
        try:
            blocks = page.get_text("blocks")  # type: ignore
            rects: list[fitz.Rect] = []
            for b in blocks:
                if len(b) < 5:
                    continue
                bx0, by0, bx1, by1, btxt = b[0], b[1], b[2], b[3], b[4]
                if not isinstance(btxt, str) or not btxt:
                    continue
                digits = sum(ch.isdigit() for ch in btxt)
                if digits < 30:
                    continue
                ratio = digits / max(1, len(btxt))
                if ratio < 0.08 and "***" not in btxt and "Observations" not in btxt:
                    continue
                rects.append(fitz.Rect(bx0, by0, bx1, by1))
            if rects:
                u = rects[0]
                for r in rects[1:]:
                    u |= r
                # Add margin
                margin = 12
                u = fitz.Rect(u.x0 - margin, u.y0 - margin, u.x1 + margin, u.y1 + margin)
                u = u & rect
                # Only use if meaningfully smaller than full page
                if (u.get_area() / rect.get_area()) < 0.75:
                    clip = u
        except Exception:
            clip = None

    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False, clip=clip)

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img.thumbnail((max_side_px, max_side_px), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    return buf.getvalue()


def _load_llm_client() -> Any:
    """
    Load a self-contained OpenAI-compatible chat client configured strictly via:
      - SKILL_LLM_API_KEY
      - SKILL_LLM_BASE_URL
      - SKILL_LLM_MODEL
    """

    @dataclasses.dataclass
    class _ChatResult:
        content: str
        raw: Any | None = None

    class AdaptiveLLMClient:
        """
        Wrap a base OpenAI-compatible client with a provider-compat fallback:
        some OpenAI-compatible gateways intermittently block requests that include a `system` role.
        If we detect those failures, retry once by folding the system prompt into the user prompt
        and sending a user-only message via the OpenAI SDK.
        """

        def __init__(self, cfg: Any, base_client: Any) -> None:
            self.cfg = cfg
            self.base = base_client
            self._force_user_only = False
            try:
                from openai import OpenAI  # type: ignore
            except Exception:
                self.raw_client = None
            else:
                self.raw_client = OpenAI(
                    api_key=str(getattr(cfg, "api_key", "")),
                    base_url=str(getattr(cfg, "base_url", "")),
                    timeout=float(getattr(cfg, "timeout_s", 60.0) or 60.0),
                    default_headers={"User-Agent": "Mozilla/5.0"},
                )

        def _should_fallback(self, err: BaseException) -> bool:
            msg = str(err).lower()
            if "empty response content" in msg:
                return True
            if "your request was blocked" in msg:
                return True
            if "error code: 403" in msg or " status=403" in msg:
                if any(s in msg for s in ["validation_required", "verify your account", "permission_denied"]):
                    return True
            return False

        def _raw_chat_text_user_only(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            temperature: float,
            max_tokens: int,
            json_mode: bool,
        ) -> _ChatResult:
            if self.raw_client is None:
                raise RuntimeError("OpenAI SDK not available for fallback call")
            merged = (
                "SYSTEM INSTRUCTIONS (treat as highest priority):\n"
                + (system_prompt or "").strip()
                + "\n\nUSER TASK:\n"
                + (user_prompt or "").strip()
            ).strip()
            kwargs: dict[str, Any] = {}
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                resp = self.raw_client.chat.completions.create(
                    model=str(getattr(self.cfg, "model", "")),
                    messages=[{"role": "user", "content": merged}],
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                    **kwargs,
                )
            except Exception as e:
                # JSON mode unsupported: retry once without json_mode.
                msg = str(e).lower()
                if json_mode and any(s in msg for s in ["response_format", "unknown parameter", "unrecognized", "invalid request"]):
                    resp = self.raw_client.chat.completions.create(
                        model=str(getattr(self.cfg, "model", "")),
                        messages=[{"role": "user", "content": merged}],
                        temperature=float(temperature),
                        max_tokens=int(max_tokens),
                    )
                else:
                    raise
            content = ""
            try:
                content = (resp.choices[0].message.content or "").strip()
            except Exception:
                content = ""
            if not content:
                raise RuntimeError("Empty response content (fallback user-only)")
            return _ChatResult(content=content, raw=resp)

        def _raw_chat_image_user_only(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            image_bytes: bytes,
            image_mime: str,
            temperature: float,
            max_tokens: int,
            json_mode: bool,
        ) -> _ChatResult:
            if self.raw_client is None:
                raise RuntimeError("OpenAI SDK not available for fallback call")
            merged = (
                "SYSTEM INSTRUCTIONS (treat as highest priority):\n"
                + (system_prompt or "").strip()
                + "\n\nUSER TASK:\n"
                + (user_prompt or "").strip()
            ).strip()
            data_url = "data:" + (image_mime or "image/jpeg") + ";base64," + base64.b64encode(image_bytes).decode("ascii")
            msg_content: list[dict[str, Any]] = [
                {"type": "text", "text": merged},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
            kwargs: dict[str, Any] = {}
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                resp = self.raw_client.chat.completions.create(
                    model=str(getattr(self.cfg, "model", "")),
                    messages=[{"role": "user", "content": msg_content}],
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                    **kwargs,
                )
            except Exception as e:
                msg = str(e).lower()
                if json_mode and any(s in msg for s in ["response_format", "unknown parameter", "unrecognized", "invalid request"]):
                    resp = self.raw_client.chat.completions.create(
                        model=str(getattr(self.cfg, "model", "")),
                        messages=[{"role": "user", "content": msg_content}],
                        temperature=float(temperature),
                        max_tokens=int(max_tokens),
                    )
                else:
                    raise
            content = ""
            try:
                content = (resp.choices[0].message.content or "").strip()
            except Exception:
                content = ""
            if not content:
                raise RuntimeError("Empty response content (fallback user-only)")
            return _ChatResult(content=content, raw=resp)

        def chat_text(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            temperature: float = 0.0,
            max_tokens: int = 4096,
            json_mode: bool = False,
        ) -> Any:
            if self._force_user_only:
                try:
                    return self._raw_chat_text_user_only(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        json_mode=json_mode,
                    )
                except Exception as e:
                    # If the user-only fallback is blocked, try the standard system+user route again.
                    if self._should_fallback(e):
                        return self.base.chat_text(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            json_mode=json_mode,
                        )
                    raise
            try:
                return self.base.chat_text(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )
            except Exception as e:
                if self._should_fallback(e):
                    # Persist this choice within the run to avoid repeated provider gating.
                    self._force_user_only = True
                    return self._raw_chat_text_user_only(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        json_mode=json_mode,
                    )
                raise

        def chat_with_image(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            image_bytes: bytes,
            image_mime: str = "image/png",
            temperature: float = 0.0,
            max_tokens: int = 1200,
            json_mode: bool = False,
        ) -> Any:
            if self._force_user_only:
                try:
                    return self._raw_chat_image_user_only(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        image_bytes=image_bytes,
                        image_mime=image_mime,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        json_mode=json_mode,
                    )
                except Exception as e:
                    if self._should_fallback(e):
                        return self.base.chat_with_image(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            image_bytes=image_bytes,
                            image_mime=image_mime,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            json_mode=json_mode,
                        )
                    raise
            try:
                return self.base.chat_with_image(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_bytes=image_bytes,
                    image_mime=image_mime,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )
            except Exception as e:
                if self._should_fallback(e):
                    self._force_user_only = True
                    return self._raw_chat_image_user_only(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        image_bytes=image_bytes,
                        image_mime=image_mime,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        json_mode=json_mode,
                    )
                raise

    # Strict env-only config: do NOT fallback or auto-select models.
    required = ["SKILL_LLM_API_KEY", "SKILL_LLM_BASE_URL", "SKILL_LLM_MODEL"]
    missing = [k for k in required if not (os.environ.get(k) or "").strip()]
    if missing:
        raise ValueError(
            "Missing required LLM environment variables: "
            + ", ".join(missing)
            + ". Set SKILL_LLM_API_KEY, SKILL_LLM_BASE_URL, and SKILL_LLM_MODEL."
        )

    api_key = (os.environ.get("SKILL_LLM_API_KEY") or "").strip()
    base_url = (os.environ.get("SKILL_LLM_BASE_URL") or "").strip()
    model = (os.environ.get("SKILL_LLM_MODEL") or "").strip()

    # Open-source fallback: self-contained OpenAI-compatible client (no external skill needed).
    @dataclasses.dataclass
    class _FallbackConfig:
        api_key: str
        base_url: str
        model: str
        timeout_s: float = 180.0
        max_retries: int = 6

    class _OpenAIBaseClient:
        def __init__(self, cfg: _FallbackConfig) -> None:
            from openai import OpenAI  # type: ignore

            self.cfg = cfg
            self.client = OpenAI(
                api_key=cfg.api_key,
                base_url=cfg.base_url,
                timeout=float(cfg.timeout_s),
                max_retries=int(cfg.max_retries),
                default_headers={"User-Agent": "Mozilla/5.0"},
            )

        def chat_text(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            temperature: float = 0.0,
            max_tokens: int = 1200,
            json_mode: bool = False,
        ) -> _ChatResult:
            kwargs: dict[str, Any] = {}
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            messages = []
            if (system_prompt or "").strip():
                messages.append({"role": "system", "content": (system_prompt or "").strip()})
            messages.append({"role": "user", "content": (user_prompt or "").strip()})
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                    **kwargs,
                )
            except Exception as e:
                msg = str(e).lower()
                if json_mode and any(
                    s in msg for s in ["response_format", "unknown parameter", "unrecognized", "invalid request"]
                ):
                    resp = self.client.chat.completions.create(
                        model=self.cfg.model,
                        messages=messages,
                        temperature=float(temperature),
                        max_tokens=int(max_tokens),
                    )
                else:
                    raise
            content = ""
            try:
                content = (resp.choices[0].message.content or "").strip()
            except Exception:
                content = ""
            if not content:
                raise RuntimeError("Empty response content (system+user)")
            return _ChatResult(content=content, raw=resp)

        def chat_with_image(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
            image_bytes: bytes,
            image_mime: str = "image/jpeg",
            temperature: float = 0.0,
            max_tokens: int = 1200,
            json_mode: bool = False,
        ) -> _ChatResult:
            kwargs: dict[str, Any] = {}
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            img_b64 = base64.b64encode(image_bytes).decode("ascii")
            messages = []
            if (system_prompt or "").strip():
                messages.append({"role": "system", "content": (system_prompt or "").strip()})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (user_prompt or "").strip()},
                        {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{img_b64}"}},
                    ],
                }
            )
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                    **kwargs,
                )
            except Exception as e:
                msg = str(e).lower()
                if json_mode and any(
                    s in msg for s in ["response_format", "unknown parameter", "unrecognized", "invalid request"]
                ):
                    resp = self.client.chat.completions.create(
                        model=self.cfg.model,
                        messages=messages,
                        temperature=float(temperature),
                        max_tokens=int(max_tokens),
                    )
                else:
                    raise
            content = ""
            try:
                content = (resp.choices[0].message.content or "").strip()
            except Exception:
                content = ""
            if not content:
                raise RuntimeError("Empty response content (system+user image)")
            return _ChatResult(content=content, raw=resp)

    cfg = _FallbackConfig(api_key=api_key, base_url=base_url, model=model, timeout_s=180.0, max_retries=6)
    base_client = _OpenAIBaseClient(cfg)
    client = AdaptiveLLMClient(cfg, base_client)
    return cfg, client


def _chat_text_logged(
    llm: Any,
    logger: LLMRunLogger,
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
    notes: str | None = None,
) -> Any:
    start = time.time()
    res = None
    err: str | None = None
    try:
        res = llm.chat_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
        return res
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        raise
    finally:
        duration_ms = int((time.time() - start) * 1000)
        try:
            logger.log_call(
                kind="chat_text",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_path=None,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=json_mode,
                response_content=(getattr(res, "content", None) if res is not None else None),
                response_raw=(getattr(res, "raw", None) if res is not None else None),
                error=err,
                duration_ms=duration_ms,
                notes=notes,
            )
        except Exception:
            # Never break the main flow due to logging.
            pass


def _looks_like_transient_gateway_block(err: BaseException) -> bool:
    msg = str(err).lower()
    # Some OpenAI-compatible gateways intermittently return 403s that are effectively transient
    # (anti-bot, validation-required, or provider-side gating). Treat these as retryable.
    if "error code: 403" in msg or " status=403" in msg:
        if any(s in msg for s in ["validation_required", "verify your account", "permission_denied", "request was blocked"]):
            return True
    if "your request was blocked" in msg:
        return True
    # Occasional provider bug: HTTP 200 but empty assistant message.
    if "empty response content" in msg:
        return True
    return False


def _retry_delay_s(attempt_1based: int) -> float:
    # 1, 2, 4, 8... capped + small jitter
    base = min(20.0, float(2 ** max(0, attempt_1based - 1)))
    jitter = float(secrets.randbelow(250)) / 1000.0
    return base + jitter


def _chat_text_logged_retry(
    llm: Any,
    logger: LLMRunLogger,
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
    notes: str | None = None,
    max_attempts: int = 10,
) -> Any:
    last_err: Exception | None = None
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        try:
            note2 = notes if attempt == 1 else f"{notes or 'call'}_retry{attempt}"
            return _chat_text_logged(
                llm,
                logger,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=json_mode,
                notes=note2,
            )
        except Exception as e:
            last_err = e
            if attempt >= max_attempts or not _looks_like_transient_gateway_block(e):
                raise
            time.sleep(_retry_delay_s(attempt))
    raise last_err or RuntimeError("LLM call failed (unknown)")


def _chat_image_logged(
    llm: Any,
    logger: LLMRunLogger,
    *,
    system_prompt: str,
    user_prompt: str,
    image_bytes: bytes,
    image_path: str,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
    notes: str | None = None,
) -> Any:
    start = time.time()
    res = None
    err: str | None = None
    try:
        res = llm.chat_with_image(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_bytes=image_bytes,
            image_mime="image/jpeg",
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
        return res
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        raise
    finally:
        duration_ms = int((time.time() - start) * 1000)
        try:
            logger.log_call(
                kind="chat_with_image",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_path=image_path,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=json_mode,
                response_content=(getattr(res, "content", None) if res is not None else None),
                response_raw=(getattr(res, "raw", None) if res is not None else None),
                error=err,
                duration_ms=duration_ms,
                notes=notes,
            )
        except Exception:
            pass


def _chat_image_logged_retry(
    llm: Any,
    logger: LLMRunLogger,
    *,
    system_prompt: str,
    user_prompt: str,
    image_bytes: bytes,
    image_path: str,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
    notes: str | None = None,
    max_attempts: int = 10,
) -> Any:
    last_err: Exception | None = None
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        try:
            note2 = notes if attempt == 1 else f"{notes or 'call'}_retry{attempt}"
            return _chat_image_logged(
                llm,
                logger,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_bytes=image_bytes,
                image_path=image_path,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=json_mode,
                notes=note2,
            )
        except Exception as e:
            last_err = e
            if attempt >= max_attempts or not _looks_like_transient_gateway_block(e):
                raise
            time.sleep(_retry_delay_s(attempt))
    raise last_err or RuntimeError("LLM call failed (unknown)")


def _call_llm_json_text(logger: LLMRunLogger, llm: Any, *, system: str, user: str, max_tokens: int) -> dict[str, Any]:
    res = _chat_text_logged_retry(
        llm,
        logger,
        system_prompt=system,
        user_prompt=user,
        temperature=0.0,
        max_tokens=max_tokens,
        json_mode=True,
        notes="json_call",
    )
    content = (res.content or "").strip()
    last_err: str | None = None
    for attempt in range(4):
        try:
            return _extract_json_obj(content)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            repair_user = (
                "The following text was supposed to be a single strict JSON object, but parsing failed.\n"
                f"Parsing error: {last_err}\n\n"
                "Fix it and output ONLY a strict JSON object (no markdown fences, no commentary). If something is truncated, complete it.\n\n"
                + content
            )
            res2 = _chat_text_logged_retry(
                llm,
                logger,
                system_prompt=system,
                user_prompt=repair_user,
                temperature=0.0,
                max_tokens=max_tokens,
                json_mode=True,
                notes=f"repair_json_{attempt+1}",
            )
            content = (res2.content or "").strip()
    raise ValueError(f"Failed to obtain valid JSON after repairs. Last error: {last_err}")


def _call_llm_json_image(
    logger: LLMRunLogger,
    llm: Any,
    *,
    system: str,
    user: str,
    image_bytes: bytes,
    image_path: str,
    max_tokens: int,
) -> dict[str, Any]:
    res = _chat_image_logged_retry(
        llm,
        logger,
        system_prompt=system,
        user_prompt=user,
        image_bytes=image_bytes,
        image_path=image_path,
        temperature=0.0,
        max_tokens=max_tokens,
        json_mode=True,
        notes="json_image_call",
    )
    content = (res.content or "").strip()
    last_err: str | None = None
    for attempt in range(4):
        try:
            return _extract_json_obj(content)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            repair_user = (
                "The following text was supposed to be a single strict JSON object, but parsing failed.\n"
                f"Parsing error: {last_err}\n\n"
                "Fix it and output ONLY a strict JSON object (no markdown fences, no commentary). If something is truncated, complete it.\n\n"
                + content
            )
            res2 = _chat_text_logged_retry(
                llm,
                logger,
                system_prompt=system,
                user_prompt=repair_user,
                temperature=0.0,
                max_tokens=max_tokens,
                json_mode=True,
                notes=f"repair_json_from_image_{attempt+1}",
            )
            content = (res2.content or "").strip()
    raise ValueError(f"Failed to obtain valid JSON after repairs. Last error: {last_err}")


def _build_method_citation_guide(apa_refs: list[str]) -> str:
    if not apa_refs:
        return ""
    # Provide a short mapping for stable in-text citations
    guide_lines = [
        "Allowed method references (for author–year citations; do not invent new references not in the list):",
    ]
    for r in apa_refs:
        guide_lines.append(f"- {r}")
    guide_lines.append("")
    guide_lines.append("Suggested citation map (use only if relevant):")
    guide_lines.append("- Threshold bunching / caliper diagnostics: (Brodeur et al., 2016; Brodeur et al., 2020)")
    guide_lines.append("- p-curve–style shape diagnostics: (Elliott et al., 2022; Simonsohn et al., 2014a; Simonsohn et al., 2014b)")
    guide_lines.append("- Multiple testing / higher t-thresholds / FDR/FWER: (Harvey et al., 2015; Harvey, 2017)")
    guide_lines.append("- Publication bias identification/correction: (Andrews & Kasy, 2019)")
    guide_lines.append("- Specification search / robustness selection: (Leamer, 1978; Leamer, 1983)")
    guide_lines.append("")
    guide_lines.append("Diagnostic theory cheat-sheet (use for 'why it matters' and for referee-style inference chains; keep tight, evidence-first):")
    guide_lines.append(
        "- Threshold bunching near p=0.05 or |t|≈1.96: If researchers explore many specifications and selectively report conventional significance, "
        "the distribution of reported statistics can show local spikes/deficits around the cutoff. This is a risk signal, not proof. "
        "(Brodeur et al., 2016; Brodeur et al., 2020)"
    )
    guide_lines.append(
        "- p-curve logic (within significant results): With genuine evidential value, the distribution of significant p-values should be right-skewed "
        "(more very small p's than p's just below 0.05). A flat or left-skewed shape can be consistent with selective reporting or p-hacking, "
        "subject to selection assumptions. (Simonsohn et al., 2014a; Simonsohn et al., 2014b; Elliott et al., 2022)"
    )
    guide_lines.append(
        "- Multiple testing / multiple outcomes / many subgroups: When many hypotheses are tested, naïve p<0.05 thresholds inflate false discoveries; "
        "referees expect FWER/FDR control, higher thresholds, or pre-specification to maintain credibility. (Harvey et al., 2015; Harvey, 2017)"
    )
    guide_lines.append(
        "- Specification search / researcher degrees of freedom: If results are sensitive to modelling choices (controls, functional forms, cutoffs, samples), "
        "the risk is that reported significance reflects search rather than stable signal. Referees ask for transparent robustness and design justification. "
        "(Leamer, 1978; Leamer, 1983)"
    )
    guide_lines.append(
        "- Publication selection vs within-paper selection: Cross-paper publication bias is distinct from within-paper selective reporting; selection models can help "
        "in meta settings, but within-paper transparency still matters for credibility. (Andrews & Kasy, 2019)"
    )
    return "\n".join(guide_lines)


def _system_prompt_agent() -> str:
    return (
        "You are a rigorous econometrics/statistics auditor. Your task is within-paper selective-reporting / p-hacking risk screening for a single paper.\n"
        "You must combine: (1) PDF text extraction (may be noisy/incomplete) and (2) rendered page images (often more reliable for tables/figures).\n"
        "Do NOT invent results that are not present in the paper. For every claim, provide page-grounded evidence (page number + identifiable anchors such as a table title, column header, or note sentence).\n"
        "Output a single strict JSON object only (no markdown fences like ```json).\n"
        "Be token-efficient: keep lists short and prioritize high-value, verifiable evidence."
    )


def _build_page_selection_prompt(*, title_hint: str, page_features: list[PageFeatures], max_pages: int) -> str:
    rows = []
    for pf in page_features:
        rows.append(
            {
                "page": pf.page_1based,
                "score": round(pf.score, 3),
                "text_len": pf.text_len,
                "digits": pf.digit_count,
                "stars": pf.star_count,
                "p_mentions": pf.p_mentions,
                "table_mentions": pf.table_mentions,
                "robust_mentions": pf.robust_mentions,
                "head": pf.text_head,
            }
        )

    schema = {
        "selected_pages": [
            {"page": "number", "role": "string", "reason": "string"},
        ],
        "notes": "string|null",
    }

    return (
        f"Paper title (may be imperfect): {title_hint}\n"
        f"Select at most {max_pages} pages from 1..{len(page_features)} to view as page images, maximizing the chance of finding within-paper p-hacking/selective-reporting risk signals.\n"
        "Coverage requirement: include at least 1 main-results page (main regression table) and at least 1 robustness/appendix page (if present).\n"
        "Exclude: reference-only pages are usually uninformative.\n\n"
        "Per-page summary (heuristic scores + first lines of extracted text):\n"
        + json.dumps(rows, ensure_ascii=False, indent=2)
        + "\n\nOutput strict JSON with this schema:\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
    )


def _build_page_evidence_prompt(
    *,
    paper_title: str,
    page_1based: int,
    extracted_text: str,
) -> str:
    extracted_text = (extracted_text or "")[:2500]
    schema = {
        "page": "number",
        "page_role": "string|null",
        "table_or_figure_names": ["string"],
        "significance_conventions": ["string"],
        "borderline_results": [
            {
                "anchor": "string",
                "why_borderline": "string",
            }
        ],
        "signals": [{"signal": "string", "evidence": "string", "anchors": ["string"]}],
        "notes": ["string"],
        "confidence_0_1": "number",
    }

    return (
        f"Paper: {paper_title}\n"
        f"Page: {page_1based}\n\n"
        + "Task: you will see a page image plus (possibly noisy) extracted text. Prioritize the image.\n"
        "Extract high-value evidence related to p-hacking / selective reporting / multiple testing / specification search.\n"
        "Requirements:\n"
        "- Output must be a single strict JSON object (no ```json fences). Keep fields short.\n"
        "- At most 4 `signals`; each signal has at most 3 `anchors`; keep each `evidence` short (<= 2 sentences).\n"
        "- At most 3 `borderline_results`.\n"
        "- `anchors` must be short identifiable phrases on the page (table/figure label, column name, note sentence, etc.).\n\n"
        "Extracted text on this page (auxiliary only; may be missing/misaligned):\n"
        + extracted_text
        + "\n\nOutput strict JSON with this schema:\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
    )


def _build_page_evidence_prompt_minimal(*, paper_title: str, page_1based: int) -> str:
    schema = {
        "page": "number",
        "signals": [{"signal": "string", "evidence": "string"}],
        "anchors": ["string"],
        "confidence_0_1": "number",
    }
    return (
        f"Paper: {paper_title}\n"
        f"Page: {page_1based}\n\n"
        "The previous JSON output failed to parse. Re-output the shortest strict JSON (no ```json fences) and only fill schema fields.\n"
        "At most 4 `signals`; at most 6 `anchors`; keep each entry short.\n\n"
        "JSON schema:\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
    )


def _build_final_report_prompt(
    *,
    paper_title: str,
    paper_meta: dict[str, Any],
    heuristics: dict[str, Any],
    selected_pages: list[dict[str, Any]],
    page_evidence: list[dict[str, Any]],
    apa_refs: list[str],
) -> str:
    return (
        f"Based on the materials below, write a within-paper p-hacking/selective-reporting risk screening report for this paper (output Markdown, not JSON). Paper title: {paper_title}\n\n"
        "Materials (JSON is for reading; do not paste it verbatim):\n"
        "1) Paper metadata (from PDF metadata / first page):\n"
        + json.dumps(paper_meta, ensure_ascii=False, indent=2)
        + "\n\n"
        "2) Heuristic statistics (from full-text extraction; may underestimate numbers inside tables):\n"
        + json.dumps(heuristics, ensure_ascii=False, indent=2)
        + "\n\n"
        "3) Selected pages (and selection reasons):\n"
        + json.dumps(selected_pages, ensure_ascii=False, indent=2)
        + "\n\n"
        "4) Per-page extracted evidence (each item has anchors + page):\n"
        + json.dumps(page_evidence, ensure_ascii=False, indent=2)
        + "\n\n"
        "Output requirements (Markdown):\n"
        "- Must include `Risk score (0–100)` and `Risk level`.\n"
        "- Must include 5–12 `Evidence cards`: each card states category, claim, why it matters, page, anchors, and a minimal follow-up check; end the card with an author–year citation.\n"
        "- Must include limitations: this is risk screening, not proof; explain key within-paper identification weaknesses.\n"
        "- End with a `References (APA, with DOI)` list: you must use only the provided list (copy/paste allowed); do not add new references.\n"
        "- Cover all modules: multiple testing, near-threshold clustering, specification search, p-curve, publication bias (even if finding is 'not detected' or 'insufficient evidence').\n"
        "- In-text author–year citations must come from the provided list.\n"
        "- Every evidentiary claim must cite page + anchors (from page_evidence).\n\n"
        "Allowed method references (APA; for citations and final reference list):\n"
        + "\n".join(f"- {r}" for r in apa_refs)
        + "\n"
    )


def _artifact_id_from_caption(*, kind: str, caption: str) -> str:
    caption = (caption or "").strip()
    if kind == "table":
        m = re.match(r"(?i)^table\s+([a-z]{0,3}\d+[a-z]?)\b", caption)
        if m:
            return f"Table {m.group(1)}"
    if kind == "figure":
        m = re.match(r"(?i)^figure\s+([a-z]{0,3}\d+[a-z]?)\b", caption)
        if m:
            return f"Figure {m.group(1)}"
    # Fallback: first token(s)
    return caption.split(".")[0].strip() or caption[:32] or f"{kind.title()} (unknown)"


def _build_artifact_inventory(
    *,
    table_captions_by_page: dict[int, str],
    figure_captions_by_page: dict[int, str],
) -> list[dict[str, Any]]:
    tables: dict[str, dict[str, Any]] = {}
    for p, cap in sorted(table_captions_by_page.items(), key=lambda x: int(x[0])):
        if not cap:
            continue
        tid = _artifact_id_from_caption(kind="table", caption=cap)
        obj = tables.setdefault(tid, {"kind": "table", "id": tid, "caption": cap, "pages": []})
        obj["caption"] = obj.get("caption") or cap
        obj["pages"].append(int(p))

    figures: dict[str, dict[str, Any]] = {}
    for p, cap in sorted(figure_captions_by_page.items(), key=lambda x: int(x[0])):
        if not cap:
            continue
        fid = _artifact_id_from_caption(kind="figure", caption=cap)
        obj = figures.setdefault(fid, {"kind": "figure", "id": fid, "caption": cap, "pages": []})
        obj["caption"] = obj.get("caption") or cap
        obj["pages"].append(int(p))

    items = list(tables.values()) + list(figures.values())
    items.sort(key=lambda d: min(d.get("pages") or [10**9]))
    # De-dupe pages and keep stable order
    for it in items:
        pages = it.get("pages") or []
        if isinstance(pages, list):
            seen: set[int] = set()
            pp: list[int] = []
            for p in pages:
                try:
                    pi = int(p)
                except Exception:
                    continue
                if pi not in seen:
                    seen.add(pi)
                    pp.append(pi)
            it["pages"] = pp
    return items


def _page_excerpt(text: str, *, max_chars: int) -> str:
    t = (text or "").strip()
    t = re.sub(r"\n{3,}", "\n\n", t)
    if len(t) <= max_chars:
        return t
    return t[: max(0, max_chars - 1)].rstrip() + "…"


def _collect_mentions_for_artifact(page_texts: list[str], artifact_id: str, *, max_hits_total: int = 8) -> list[dict[str, Any]]:
    pat = rf"(?i)\b{re.escape(artifact_id)}\b"
    out: list[dict[str, Any]] = []
    for i, t in enumerate(page_texts, start=1):
        if len(out) >= max_hits_total:
            break
        if not re.search(pat, t or ""):
            continue
        snips = _find_snippets(t or "", [pat], max_hits=2, ctx=140)
        for s in snips:
            out.append({"page": int(i), "snippet": _clean_snippet(s, max_len=260)})
            if len(out) >= max_hits_total:
                break
    return out


def _summarize_tables_from_tests(
    *,
    tests_path: Path,
    table_captions_by_page: dict[int, str],
    max_flagged_per_table: int = 25,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    def tid_for_page(page: int) -> tuple[str, str]:
        cap = table_captions_by_page.get(int(page)) or ""
        tid = _artifact_id_from_caption(kind="table", caption=cap) if cap else "Table (unknown)"
        return tid, cap

    def ensure(tid: str, cap: str) -> dict[str, Any]:
        obj = out.get(tid)
        if obj is None:
            obj = {
                "table_id": tid,
                "caption": cap or "",
                "pages": set(),
                "n_tests": 0,
                "p_005_left": 0,
                "p_005_right": 0,
                "t_196_left": 0,
                "t_196_right": 0,
                "sig_p_le_0_05": 0,
                "sig_p_le_0_10": 0,
                "pcurve_sig_n": 0,
                "pcurve_low_half": 0,
                "pcurve_high_half": 0,
                "t_exact_2_count": 0,
                "flagged_p_0_05": [],  # list[(dist, entry)]
                "flagged_t_1_96": [],
            }
            out[tid] = obj
        if cap and not obj.get("caption"):
            obj["caption"] = cap
        return obj

    with tests_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            page = rec.get("page")
            if not isinstance(page, int) or page <= 0:
                continue
            tid, cap = tid_for_page(page)
            obj = ensure(tid, cap)
            obj["pages"].add(int(page))
            obj["n_tests"] += 1

            abs_t = rec.get("abs_t")
            try:
                at = float(abs_t)
            except Exception:
                at = None
            p2 = rec.get("p_approx_2s")
            try:
                p2f = float(p2)
            except Exception:
                p2f = None

            if p2f is not None and 0.0 <= p2f <= 1.0:
                if p2f <= 0.05:
                    obj["sig_p_le_0_05"] += 1
                    obj["pcurve_sig_n"] += 1
                    if p2f <= 0.025:
                        obj["pcurve_low_half"] += 1
                    else:
                        obj["pcurve_high_half"] += 1
                if p2f <= 0.10:
                    obj["sig_p_le_0_10"] += 1
                if 0.045 <= p2f <= 0.055:
                    if 0.045 <= p2f <= 0.05:
                        obj["p_005_left"] += 1
                    else:
                        obj["p_005_right"] += 1
                    d = abs(p2f - 0.05)
                    entry = {
                        "page": int(page),
                        "coef_raw": rec.get("coef_raw") or rec.get("cell_text_snippet") or "",
                        "se_text": rec.get("se_cell_text_snippet") or "",
                        "abs_t": at,
                        "p_approx_2s": p2f,
                        "stars": rec.get("stars"),
                        "row_index": rec.get("row_index"),
                        "col_index": rec.get("col_index"),
                        "table_index": rec.get("table_index"),
                        "table_bbox": rec.get("table_bbox"),
                        "coef_cell_bbox": rec.get("coef_cell_bbox"),
                    }
                    obj["flagged_p_0_05"].append((float(d), entry))
                    obj["flagged_p_0_05"].sort(key=lambda x: x[0])
                    if len(obj["flagged_p_0_05"]) > max_flagged_per_table:
                        del obj["flagged_p_0_05"][max_flagged_per_table:]

            if at is not None and at == at:
                if 1.76 <= float(at) <= 2.16:
                    if 1.76 <= float(at) <= 1.96:
                        obj["t_196_left"] += 1
                    else:
                        obj["t_196_right"] += 1
                    d = abs(float(at) - 1.96)
                    if abs(float(at) - 2.0) < 1e-9:
                        obj["t_exact_2_count"] += 1
                    entry = {
                        "page": int(page),
                        "coef_raw": rec.get("coef_raw") or rec.get("cell_text_snippet") or "",
                        "se_text": rec.get("se_cell_text_snippet") or "",
                        "abs_t": float(at),
                        "p_approx_2s": p2f,
                        "stars": rec.get("stars"),
                        "row_index": rec.get("row_index"),
                        "col_index": rec.get("col_index"),
                        "table_index": rec.get("table_index"),
                        "table_bbox": rec.get("table_bbox"),
                        "coef_cell_bbox": rec.get("coef_cell_bbox"),
                    }
                    obj["flagged_t_1_96"].append((float(d), entry))
                    obj["flagged_t_1_96"].sort(key=lambda x: x[0])
                    if len(obj["flagged_t_1_96"]) > max_flagged_per_table:
                        del obj["flagged_t_1_96"][max_flagged_per_table:]

    # finalize: convert sets and drop distances
    for tid, obj in out.items():
        obj["pages"] = sorted(int(p) for p in (obj.get("pages") or set()))
        obj["flagged_p_0_05"] = [x[1] for x in (obj.get("flagged_p_0_05") or [])]
        obj["flagged_t_1_96"] = [x[1] for x in (obj.get("flagged_t_1_96") or [])]
        # clearer aliases (avoid left/right confusion)
        obj["p_005_just_sig"] = int(obj.get("p_005_left") or 0)
        obj["p_005_just_nonsig"] = int(obj.get("p_005_right") or 0)
        obj["t_196_just_below"] = int(obj.get("t_196_left") or 0)
        obj["t_196_just_above"] = int(obj.get("t_196_right") or 0)
        # convenience ratios (avoid divide-by-zero)
        try:
            obj["p_005_left_over_right"] = float(obj.get("p_005_left") or 0) / float(max(1, int(obj.get("p_005_right") or 0)))
        except Exception:
            obj["p_005_left_over_right"] = None
        try:
            obj["t_196_right_over_left"] = float(obj.get("t_196_right") or 0) / float(max(1, int(obj.get("t_196_left") or 0)))
        except Exception:
            obj["t_196_right_over_left"] = None
        # pcurve z
        try:
            n_sig = int(obj.get("pcurve_low_half") or 0) + int(obj.get("pcurve_high_half") or 0)
            if n_sig > 0:
                obj["pcurve_right_skew_z"] = float(
                    (int(obj.get("pcurve_low_half") or 0) - int(obj.get("pcurve_high_half") or 0)) / math.sqrt(float(n_sig))
                )
            else:
                obj["pcurve_right_skew_z"] = 0.0
        except Exception:
            obj["pcurve_right_skew_z"] = 0.0
        try:
            obj["pcurve_low_over_high"] = float(obj.get("pcurve_low_half") or 0) / float(max(1, int(obj.get("pcurve_high_half") or 0)))
        except Exception:
            obj["pcurve_low_over_high"] = None
        try:
            t_total = int(obj.get("t_196_left") or 0) + int(obj.get("t_196_right") or 0)
            obj["t_exact_2_share_in_t196"] = float(obj.get("t_exact_2_count") or 0) / float(t_total) if t_total else 0.0
        except Exception:
            obj["t_exact_2_share_in_t196"] = None
    return out


def _auto_extract_tests_for_pdf(
    *,
    out_dir: Path,
    pdf_path: Path,
    max_pages_per_paper: int,
    max_pdf_pages: int | None,
    force: bool,
) -> tuple[Path, Path | None]:
    """
    Create a tiny self-contained corpus under <out_dir>/_auto_corpus and run
    extract_within_paper_metrics.py to produce tests JSONL for this PDF.

    Returns:
      (auto_corpus_dir, tests_path_or_none)
    """
    auto_corpus_dir = out_dir / "_auto_corpus"
    pdfs_dir = auto_corpus_dir / "pdfs"
    pdfs_dir.mkdir(parents=True, exist_ok=True)

    dest_pdf = pdfs_dir / pdf_path.name
    try:
        # Avoid needless copies if already present and identical.
        if not dest_pdf.exists() or _sha256_file(dest_pdf) != _sha256_file(pdf_path):
            shutil.copy2(pdf_path, dest_pdf)
    except Exception:
        shutil.copy2(pdf_path, dest_pdf)

    script_path = Path(__file__).resolve().parent / "extract_within_paper_metrics.py"
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    cmd = [
        sys.executable,
        str(script_path),
        "--corpus-dir",
        str(auto_corpus_dir),
        "--limit",
        "1",
        "--max-pages-per-paper",
        str(int(max(1, max_pages_per_paper))),
    ]
    if max_pdf_pages is not None:
        cmd += ["--max-pdf-pages", str(int(max_pdf_pages))]
    if force:
        cmd.append("--force")

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    tests_path = auto_corpus_dir / "tests" / f"{dest_pdf.stem}.jsonl"
    if tests_path.exists() and tests_path.stat().st_size > 0:
        return auto_corpus_dir, tests_path
    return auto_corpus_dir, None


def _system_prompt_expert_json() -> str:
    return (
        "You are a senior economics/econometrics referee and research-integrity auditor.\n"
        "Your task: produce an end-to-end, table-by-table (and figure-by-figure) selective-reporting / researcher-degrees-of-freedom risk assessment for ONE paper.\n"
        "You MUST be evidence-disciplined: do not invent results that are not supported by the provided inputs.\n"
        "You MUST NOT accuse the authors of misconduct or speculate about intent; focus on statistical risk signals and transparency/robustness recommendations.\n"
        "You MAY form hypotheses (like a real referee) about mechanisms that could generate patterns, but label them explicitly and avoid attributing intent.\n"
        "Do NOT quote long passages from the paper (max 20 words per quote). Prefer paraphrase.\n"
        "When you explain why a diagnostic matters, ground it in the provided method literature and use author–year citations.\n"
        "Output MUST be strict JSON only (a single JSON object), with no markdown fences and no extra commentary.\n"
        "Write in English."
    )


def _system_prompt_expert_markdown() -> str:
    return (
        "You are a senior economics/econometrics referee and research-integrity auditor.\n"
        "Write a rigorous, detailed referee-style selective-reporting / p-hacking risk screening report for ONE paper.\n"
        "Use ONLY the provided evidence; do not fabricate. Clearly separate: (a) observations, (b) inferences, (c) hypotheses.\n"
        "Do NOT accuse authors of misconduct or speculate about intent.\n"
        "Do NOT quote long passages from the paper (max 20 words per quote). Prefer paraphrase.\n"
        "Ground each diagnostic claim in the provided method literature and use author–year citations.\n"
        "Write in English. Output MUST be Markdown text only (no JSON, no code fences)."
    )


def _build_expert_chunk_prompt(
    *,
    paper_title_hint: str,
    page_start: int,
    page_end: int,
    chunk_text: str,
    method_guide: str,
) -> str:
    schema = {
        "pages": [page_start, page_end],
        "section_guess": "string|null",
        "key_points": ["string"],
        "claims": [{"claim": "string", "pages": ["number"]}],
        "tables_figures_mentioned": [{"id": "string", "pages": ["number"], "note": "string"}],
        "integrity_relevant_notes": [{"type": "string", "pages": ["number"], "evidence_paraphrase": "string"}],
    }
    return (
        f"Paper title (hint): {paper_title_hint}\n"
        f"Task: Summarize pages {page_start}–{page_end} for a research-integrity / p-hacking oriented referee review.\n"
        "Focus on: what is being claimed, what identification/statistical choices are described, and where tables/figures are used.\n"
        "If you see mentions of multiple testing, robustness/specification search, selective reporting, p-values/t-stats/stars, note them.\n\n"
        "Pages text (noisy PDF extraction):\n"
        + chunk_text
        + "\n\n"
        + method_guide
        + "\n\nReturn strict JSON matching this schema:\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
    )


def _build_expert_digest_prompt(
    *,
    paper_title_hint: str,
    paper_meta: dict[str, Any],
    chunk_summaries: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    method_guide: str,
) -> str:
    schema = {
        "paper_title": "string",
        "one_paragraph_summary": "string",
        "research_question": "string",
        "data_and_sample": "string",
        "identification_and_empirical_strategy": "string",
        "main_outcomes_and_main_claims": ["string"],
        "reported_significance_conventions": ["string"],
        "where_key_results_live": [{"id": "string", "role": "string", "pages": ["number"]}],
        "degrees_of_freedom_map": [{"dimension": "string", "what_varies": "string", "why_it_matters": "string"}],
        "audit_focus_priorities": ["string"],
    }
    return (
        f"Paper title (hint): {paper_title_hint}\n"
        "Task: Build an auditor-grade paper understanding from the chunk summaries and the artifact inventory.\n"
        "Be concrete: name the key outcomes, identification, and where the key results appear (tables/figures).\n\n"
        "PDF metadata (may be incomplete):\n"
        + json.dumps(paper_meta, ensure_ascii=False, indent=2)
        + "\n\nChunk summaries:\n"
        + json.dumps(chunk_summaries, ensure_ascii=False, indent=2)
        + "\n\nArtifact inventory (tables/figures):\n"
        + json.dumps(artifacts, ensure_ascii=False, indent=2)
        + "\n\n"
        + method_guide
        + "\n\nReturn strict JSON matching this schema:\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
    )


def _build_expert_artifact_audit_prompt(
    *,
    artifact: dict[str, Any],
    paper_meta: dict[str, Any],
    paper_digest: dict[str, Any],
    method_guide: str,
) -> str:
    kind = str(artifact.get("kind") or "")
    schema = {
        "kind": kind,
        "id": artifact.get("id"),
        "caption": artifact.get("caption"),
        "pages": artifact.get("pages"),
        "role_in_paper": "string|null",
        "p_hacking_risk_score_0_100": "number",
        "risk_level": "Low|Moderate|High",
        "key_concerns": ["string"],
        "evidence_items": [
            {
                "signal": "string",
                "strength": "weak|moderate|strong",
                "why_it_matters": "string",
                "pages": ["number"],
                "flagged_entries": [
                    {
                        "page": "number",
                        "row_guess": "string|null",
                        "col_guess": "string|null",
                        "coef": "string|null",
                        "se_paren": "string|null",
                        "abs_t": "number|null",
                        "p_approx_2s": "number|null",
                    }
                ],
                "notes": "string|null",
            }
        ],
        "alternative_explanations": ["string"],
        "recommended_checks": ["string"],
    }
    return (
        "Task: Audit this SINGLE table/figure like a meticulous economics referee focusing on p-hacking / researcher degrees of freedom.\n"
        "Non-negotiables:\n"
        "- Evidence discipline: do not invent paper content not present in the provided packet.\n"
        "- If the packet contains `extracted_table_metrics`, you MUST use them explicitly (counts/ratios/flagged entries) and cite page numbers.\n"
        "- Metric meaning matters: `p_005_left/right` are the *0.05-threshold bunching window*; `pcurve_low_half/high_half` are a *p-curve split within p<=0.05*. Do NOT conflate them.\n"
        "- If there is NO numeric evidence for a table (e.g., extracted_table_metrics.n_tests==0), you must say \"insufficient numeric evidence\" and avoid numerical claims.\n"
        "- Treat `row_label` / `col_label` inside flagged entries as best-effort machine guesses; if you rely on them, label them as such.\n"
        "- Incorporate paper-wide context from `paper_digest` when interpreting what this artifact is supposed to show.\n"
        "- Every `evidence_items[].why_it_matters` MUST include at least one author–year citation from the Allowed references (e.g., '(Brodeur et al., 2016)').\n"
        "- Your reasoning must follow a clear chain: Observation -> diagnostic mapping -> why it matters (with citation) -> risk implication -> alternative explanations -> recommended checks.\n\n"
        "Paper meta:\n"
        + json.dumps(paper_meta, ensure_ascii=False, indent=2)
        + "\n\nPaper digest:\n"
        + json.dumps(paper_digest, ensure_ascii=False, indent=2)
        + "\n\nArtifact packet:\n"
        + json.dumps(artifact, ensure_ascii=False, indent=2)
        + "\n\n"
        + method_guide
        + "\n\nReturn strict JSON matching this schema:\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
    )


def _build_expert_metrics_prompt(
    *,
    paper_meta: dict[str, Any],
    paper_digest: dict[str, Any],
    artifact_inventory: list[dict[str, Any]],
    artifact_audits: list[dict[str, Any]],
) -> str:
    schema = {
        "paper": {
            "title": "string",
            "overall_risk_score_0_100": "number",
            "overall_risk_level": "Low|Moderate|High",
            "top_concerns": ["string"],
            "key_artifacts": ["string"],
        },
        "artifacts": [
            {
                "kind": "table|figure",
                "id": "string",
                "pages": ["number"],
                "risk_score_0_100": "number",
                "risk_level": "Low|Moderate|High",
                "has_numeric_evidence": "boolean",
                "evidence_counts": {
                    "n_tests": "number|null",
                    "p_005_just_sig": "number|null",
                    "p_005_just_nonsig": "number|null",
                    "t_196_just_below": "number|null",
                    "t_196_just_above": "number|null",
                    "t_exact_2_count": "number|null",
                    "sig_p_le_0_05": "number|null",
                    "sig_p_le_0_10": "number|null",
                    "pcurve_low_half": "number|null",
                    "pcurve_high_half": "number|null",
                    "pcurve_right_skew_z": "number|null",
                },
                "signals": ["string"],
                "flagged_entries": [
                    {
                        "page": "number",
                        "row": "string|null",
                        "col": "string|null",
                        "coef": "string|null",
                        "se_paren": "string|null",
                        "abs_t": "number|null",
                        "p_approx_2s": "number|null",
                    }
                ],
            }
        ],
        "generated_at": "string",
    }
    return (
        "Task: Produce a machine-readable metrics JSON for this paper-level and artifact-level p-hacking audit.\n"
        "You MUST cover EVERY artifact in the provided inventory.\n"
        "Use the artifact_audits as the source of truth for risk signals, but keep the (kind,id,pages) from the inventory.\n"
        "For tables, if `extracted_table_metrics` exists in the inventory, set has_numeric_evidence=true and copy the evidence_counts from those fields.\n"
        "If numeric evidence does not exist, set has_numeric_evidence=false and evidence_counts fields to null.\n"
        "Do not invent new artifacts.\n"
        "Output MUST match the provided schema EXACTLY (no extra top-level keys, no extra fields).\n"
        "Do NOT include narrative text, citations, references, or DOIs in this metrics JSON.\n\n"
        "paper_meta:\n"
        + json.dumps(paper_meta, ensure_ascii=False, indent=2)
        + "\n\npaper_digest:\n"
        + json.dumps(paper_digest, ensure_ascii=False, indent=2)
        + "\n\nartifact_inventory (authoritative list of artifacts to cover):\n"
        + json.dumps(artifact_inventory, ensure_ascii=False, indent=2)
        + "\n\nartifact_audits:\n"
        + json.dumps(artifact_audits, ensure_ascii=False, indent=2)
        + "\n\nReturn strict JSON matching this schema:\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
    )


def _risk_level_from_score(score_0_100: float | int | None) -> str:
    try:
        s = float(score_0_100) if score_0_100 is not None else 0.0
    except Exception:
        s = 0.0
    if s >= 67.0:
        return "High"
    if s >= 34.0:
        return "Moderate"
    return "Low"


def _validate_expert_metrics_schema(obj: Any, *, artifact_inventory: list[dict[str, Any]]) -> tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "top-level is not an object"
    allowed_top = {"paper", "artifacts", "generated_at"}
    extra_top = set(obj.keys()) - allowed_top
    if extra_top:
        return False, f"unexpected top-level keys: {sorted(extra_top)}"
    if not isinstance(obj.get("paper"), dict):
        return False, "missing/invalid `paper`"
    if not isinstance(obj.get("artifacts"), list):
        return False, "missing/invalid `artifacts` (must be a list)"

    inv_ids: list[str] = []
    inv_kinds: dict[str, str] = {}
    inv_pages: dict[str, list[int]] = {}
    for a in artifact_inventory:
        if not isinstance(a, dict) or a.get("id") is None:
            continue
        aid = str(a.get("id"))
        inv_ids.append(aid)
        inv_kinds[aid] = str(a.get("kind") or "")
        pp = a.get("pages")
        inv_pages[aid] = [int(x) for x in pp] if isinstance(pp, list) else []

    seen: set[str] = set()
    for it in obj.get("artifacts") or []:
        if not isinstance(it, dict):
            return False, "an `artifacts[]` entry is not an object"
        aid = str(it.get("id") or "")
        if not aid or aid not in inv_kinds:
            return False, f"artifact id not in inventory: {aid!r}"
        if aid in seen:
            return False, f"duplicate artifact id: {aid}"
        seen.add(aid)
        kind = str(it.get("kind") or "")
        if kind not in {"table", "figure"}:
            return False, f"invalid kind for {aid}: {kind!r}"
        if kind and inv_kinds.get(aid) and kind != inv_kinds.get(aid):
            return False, f"kind mismatch for {aid}: got {kind!r}, expected {inv_kinds.get(aid)!r}"
        if not isinstance(it.get("pages"), list):
            return False, f"invalid pages for {aid} (must be list)"

    if len(seen) != len(inv_ids):
        missing = [x for x in inv_ids if x not in seen]
        return False, f"missing artifacts: {missing[:8]}{'...' if len(missing) > 8 else ''}"

    paper = obj.get("paper") or {}
    for key in ["title", "overall_risk_score_0_100", "overall_risk_level", "top_concerns", "key_artifacts"]:
        if key not in paper:
            return False, f"missing paper.{key}"
    return True, "ok"


def _fallback_expert_metrics_from_audits(
    *,
    paper_title: str,
    artifact_inventory: list[dict[str, Any]],
    artifact_audits: list[dict[str, Any]],
) -> dict[str, Any]:
    audit_by_id: dict[str, dict[str, Any]] = {}
    for a in artifact_audits:
        if isinstance(a, dict) and a.get("id") is not None:
            audit_by_id[str(a.get("id"))] = a

    artifacts_out: list[dict[str, Any]] = []
    scores: list[float] = []
    top_concerns: list[str] = []

    for inv in artifact_inventory:
        if not isinstance(inv, dict) or inv.get("id") is None:
            continue
        aid = str(inv.get("id"))
        kind = str(inv.get("kind") or "")
        pages = inv.get("pages") if isinstance(inv.get("pages"), list) else []
        audit = audit_by_id.get(aid, {})
        score = audit.get("p_hacking_risk_score_0_100")
        try:
            score_f = float(score) if score is not None else None
        except Exception:
            score_f = None
        if score_f is not None:
            scores.append(score_f)
        key_conc = audit.get("key_concerns")
        if isinstance(key_conc, list):
            for c in key_conc:
                if isinstance(c, str) and c.strip():
                    _merge_unique(top_concerns, [c.strip()])

        tm = inv.get("extracted_table_metrics") if isinstance(inv.get("extracted_table_metrics"), dict) else {}
        has_numeric = bool(kind == "table" and int(tm.get("n_tests") or 0) > 0)
        evidence_counts = {k: tm.get(k) for k in [
            "n_tests",
            "p_005_just_sig",
            "p_005_just_nonsig",
            "t_196_just_below",
            "t_196_just_above",
            "t_exact_2_count",
            "sig_p_le_0_05",
            "sig_p_le_0_10",
            "pcurve_low_half",
            "pcurve_high_half",
            "pcurve_right_skew_z",
        ]} if has_numeric else {k: None for k in [
            "n_tests",
            "p_005_just_sig",
            "p_005_just_nonsig",
            "t_196_just_below",
            "t_196_just_above",
            "t_exact_2_count",
            "sig_p_le_0_05",
            "sig_p_le_0_10",
            "pcurve_low_half",
            "pcurve_high_half",
            "pcurve_right_skew_z",
        ]}

        flagged_entries: list[dict[str, Any]] = []
        eis = audit.get("evidence_items") if isinstance(audit.get("evidence_items"), list) else []
        for ei in eis:
            if not isinstance(ei, dict):
                continue
            fe = ei.get("flagged_entries") if isinstance(ei.get("flagged_entries"), list) else []
            for row in fe:
                if not isinstance(row, dict):
                    continue
                flagged_entries.append(
                    {
                        "page": row.get("page"),
                        "row": row.get("row_guess") or row.get("row"),
                        "col": row.get("col_guess") or row.get("col"),
                        "coef": row.get("coef"),
                        "se_paren": row.get("se_paren"),
                        "abs_t": row.get("abs_t"),
                        "p_approx_2s": row.get("p_approx_2s"),
                    }
                )
                if len(flagged_entries) >= 8:
                    break
            if len(flagged_entries) >= 8:
                break

        artifacts_out.append(
            {
                "kind": "table" if kind == "table" else "figure",
                "id": aid,
                "pages": [int(x) for x in pages if isinstance(x, (int, float, str)) and str(x).strip().isdigit()],
                "risk_score_0_100": score_f if score_f is not None else 0.0,
                "risk_level": str(audit.get("risk_level") or _risk_level_from_score(score_f)),
                "has_numeric_evidence": bool(has_numeric),
                "evidence_counts": evidence_counts,
                "signals": [c for c in (audit.get("key_concerns") or []) if isinstance(c, str)][:6],
                "flagged_entries": flagged_entries[:8],
            }
        )

    overall = float(sum(scores) / len(scores)) if scores else 0.0
    key_artifacts = [a["id"] for a in sorted(artifacts_out, key=lambda x: float(x.get("risk_score_0_100") or 0.0), reverse=True)[:8]]
    return {
        "paper": {
            "title": paper_title,
            "overall_risk_score_0_100": round(overall, 2),
            "overall_risk_level": _risk_level_from_score(overall),
            "top_concerns": top_concerns[:10],
            "key_artifacts": key_artifacts,
        },
        "artifacts": artifacts_out,
        "generated_at": _now_iso(),
    }


def _build_expert_report_prompt(
    *,
    paper_meta: dict[str, Any],
    paper_digest: dict[str, Any],
    expert_metrics: dict[str, Any],
    artifact_evidence: list[dict[str, Any]],
    artifact_audits: list[dict[str, Any]],
    apa_refs: list[str],
) -> str:
    refs_block = "\n".join(f"- {r}" for r in apa_refs)
    return (
        "Write an end-to-end, referee-style selective-reporting / p-hacking *risk screening* report (Markdown) for this paper.\n"
        "You must write as if you actually audited the entire paper context and every table/figure in the inventory.\n"
        "Hard requirements:\n"
        "- English only.\n"
        "- Use the risk scores from `expert_metrics` consistently.\n"
        "- Provide a clear overall verdict, an overall risk score (0–100), and a risk level (Low/Moderate/High).\n"
        "- Include a short section \"What I actually checked\" that explains the workflow (context digestion + artifact audits + numeric signals).\n"
        "- Audit EVERY artifact (each table and figure) in the provided inventory.\n"
        "- For TABLES with numeric evidence (extracted_table_metrics), you must explicitly report:\n"
        "  * n_tests, p≈0.05 just-sig vs just-non-sig counts (p_005_just_sig / p_005_just_nonsig), |t|≈1.96 just-below vs just-above counts (t_196_just_below / t_196_just_above), any rounding artifact indicator (t_exact_2_count), and a short list of flagged entries.\n"
        "- Metric meaning matters: `p_005_left/right` are the 0.05 bunching window; `pcurve_low_half/high_half` are the p-curve split within p<=0.05.\n"
        "- Do NOT use placeholders like '?', '≈', or 'n/a' when exact numbers exist in `artifact_evidence` / `expert_metrics`.\n"
        "- For each table with has_numeric_evidence=true, include a small Markdown evidence table that lists the exact evidence_counts and the top flagged entries.\n"
        "- When flagging, always include concrete location info: artifact id + page number + (row/col if available).\n"
        "- Separate clearly: (1) Observations, (2) Inferences, (3) Hypotheses (explicitly labeled).\n"
        "- Hypotheses MUST be non-attributional (do not claim intent or misconduct); frame as mechanisms that could produce the patterns.\n"
        "- No long quotes from the paper (max 20 words per quote).\n"
        "- End with `References (APA, with DOI)` and ONLY use the provided references list.\n"
        "- Do NOT paste raw JSON blobs. Summarize.\n\n"
        "Required outline (use these headings):\n"
        "# Selective-reporting / p-hacking risk screening: <paper title>\n"
        "## Executive summary\n"
        "## Paper context (claims & empirical setup)\n"
        "## What I actually checked (workflow)\n"
        "## Artifact-by-artifact audit (tables & figures)\n"
        "## Overall assessment & suggested author responses\n"
        "## Limitations\n"
        "## References (APA, with DOI)\n\n"
        "paper_meta:\n"
        + json.dumps(paper_meta, ensure_ascii=False, indent=2)
        + "\n\npaper_digest:\n"
        + json.dumps(paper_digest, ensure_ascii=False, indent=2)
        + "\n\nexpert_metrics (JSON; use these scores consistently):\n"
        + json.dumps(expert_metrics, ensure_ascii=False, indent=2)
        + "\n\nartifact_evidence (authoritative numeric evidence, when available):\n"
        + json.dumps(artifact_evidence, ensure_ascii=False, indent=2)
        + "\n\nartifact_audits:\n"
        + json.dumps(artifact_audits, ensure_ascii=False, indent=2)
        + "\n\nAllowed references (copy into References section; do not add others):\n"
        + refs_block
        + "\n"
    )


def _build_expert_report_header_prompt(
    *,
    paper_meta: dict[str, Any],
    paper_digest: dict[str, Any],
    expert_metrics: dict[str, Any],
    artifact_inventory_compact: list[dict[str, Any]],
    method_guide: str,
) -> str:
    return (
        "Write the FIRST PART of a referee-style selective-reporting / p-hacking risk screening report (Markdown).\n"
        "Hard requirements:\n"
        "- English only.\n"
        "- Use ONLY the provided inputs; do not fabricate.\n"
        "- Use the risk scores from `expert_metrics` consistently.\n"
        "- Do NOT include any artifact subsections yet (no '### Table ...' / '### Figure ...').\n"
        "- End your output with the heading line exactly: '## Artifact-by-artifact audit (tables & figures)'.\n"
        "- Keep this part concise so it fits in one response: target <= 900 words total.\n\n"
        "Required headings to include in this part (in order):\n"
        "# Selective-reporting / p-hacking risk screening: <paper title>\n"
        "## Executive summary\n"
        "## Paper context (claims & empirical setup)\n"
        "## What I actually checked (workflow)\n"
        "## Methodological basis (diagnostics & theory)\n"
        "## Artifact-by-artifact audit (tables & figures)\n\n"
        "In 'Methodological basis', give a tight mapping from diagnostics -> interpretation -> citations.\n"
        "Use author–year in-text citations (e.g., (Brodeur et al., 2016)).\n\n"
        "paper_meta:\n"
        + json.dumps(paper_meta, ensure_ascii=False, indent=2)
        + "\n\npaper_digest:\n"
        + json.dumps(paper_digest, ensure_ascii=False, indent=2)
        + "\n\nexpert_metrics:\n"
        + json.dumps(expert_metrics, ensure_ascii=False, indent=2)
        + "\n\nartifact_inventory_compact (authoritative list; do not add others):\n"
        + json.dumps(artifact_inventory_compact, ensure_ascii=False, indent=2)
        + "\n\n"
        + method_guide
        + "\n"
    )


def _build_expert_report_artifact_batch_prompt(
    *,
    paper_title: str,
    artifact_evidence_batch: list[dict[str, Any]],
    artifact_audits_batch: list[dict[str, Any]],
    method_guide: str,
) -> str:
    return (
        "Write ONLY the artifact audit subsections (Markdown) for the provided batch.\n"
        "Hard requirements:\n"
        "- English only.\n"
        "- Output ONLY Markdown subsections starting with '### <id>: <caption> (Pages ...)' for EACH artifact in the batch.\n"
        "- Do NOT write any top-level headings (no '#', no '##').\n"
        "- Do NOT add artifacts not present in `artifact_evidence_batch`.\n"
        "- For each artifact subsection, include these labeled blocks in order:\n"
        "  1) Role in paper\n"
        "  2) Observations (evidence, with page numbers)\n"
        "  3) Diagnostics & theory (each diagnostic must include an author–year citation from Allowed references)\n"
        "  4) Inferences (risk implications, must be logically tied to diagnostics)\n"
        "  5) Hypotheses (explicitly labeled; non-attributional; no intent)\n"
        "  6) Recommended author checks / disclosures\n"
        "- If `has_numeric_evidence=true`, include a small Markdown evidence table with:\n"
        "  n_tests, p_005_just_sig, p_005_just_nonsig, t_196_just_below, t_196_just_above, t_exact_2_count, sig_p_le_0_05, sig_p_le_0_10, pcurve_low_half, pcurve_high_half, pcurve_right_skew_z.\n"
        "  Then list up to 8 flagged entries with location (page + row/col if available) and values.\n"
        "- Do NOT use placeholders when exact numbers exist.\n"
        "- No long quotes from the paper (max 20 words per quote).\n\n"
        f"Paper title: {paper_title}\n\n"
        "artifact_evidence_batch:\n"
        + json.dumps(artifact_evidence_batch, ensure_ascii=False, indent=2)
        + "\n\nartifact_audits_batch:\n"
        + json.dumps(artifact_audits_batch, ensure_ascii=False, indent=2)
        + "\n\n"
        + method_guide
        + "\n"
    )


def _build_expert_report_tail_prompt(
    *,
    paper_title: str,
    expert_metrics: dict[str, Any],
    artifact_inventory_compact: list[dict[str, Any]],
    apa_refs: list[str],
) -> str:
    refs_block = "\n".join(f"- {r}" for r in apa_refs)
    return (
        "Write the FINAL PART of the referee-style selective-reporting / p-hacking risk screening report (Markdown).\n"
        "Hard requirements:\n"
        "- English only.\n"
        "- Do NOT repeat earlier sections; start with '## Overall assessment & suggested author responses'.\n"
        "- Use the risk score/level from `expert_metrics` consistently.\n"
        "- Do NOT add artifacts not in the inventory.\n"
        "- End with a complete '## References (APA, with DOI)' section, using ONLY the provided references list.\n\n"
        "Required headings in this part (in order):\n"
        "## Overall assessment & suggested author responses\n"
        "## Limitations\n"
        "## References (APA, with DOI)\n\n"
        f"Paper title: {paper_title}\n\n"
        "expert_metrics:\n"
        + json.dumps(expert_metrics, ensure_ascii=False, indent=2)
        + "\n\nartifact_inventory_compact:\n"
        + json.dumps(artifact_inventory_compact, ensure_ascii=False, indent=2)
        + "\n\nAllowed references (copy into References section; do not add others):\n"
        + refs_block
        + "\n"
    )


def _run_expert_workflow(
    *,
    llm: Any,
    logger: LLMRunLogger,
    doc: fitz.Document,
    pdf_path: Path,
    out_dir: Path,
    cache_dir: Path,
    page_texts: list[str],
    paper_meta: dict[str, Any],
    paper_title_hint: str,
    heuristics: dict[str, Any],
    tests_path: Path | None,
    apa_refs: list[str],
    method_guide: str,
    force: bool,
    force_metrics: bool,
    force_report: bool,
    expert_chunk_pages: int,
    expert_max_page_chars: int,
    expert_max_pages_per_artifact: int,
    expert_max_artifacts: int,
    expert_max_flagged_per_table: int,
) -> tuple[str, dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Returns:
      (report_md, expert_metrics, paper_digest, artifacts, artifact_audits)
    """
    expert_dir = out_dir / "expert"
    expert_dir.mkdir(parents=True, exist_ok=True)

    # Inventory
    table_caps = heuristics.get("table_captions_by_page") or {}
    if not isinstance(table_caps, dict):
        table_caps = {}
    # Normalize keys to int
    table_caps_i: dict[int, str] = {}
    for k, v in table_caps.items():
        try:
            table_caps_i[int(k)] = str(v)
        except Exception:
            continue
    fig_caps_i = _extract_figure_captions_by_page(page_texts)
    heuristics["figure_captions_by_page"] = fig_caps_i

    artifacts = _build_artifact_inventory(table_captions_by_page=table_caps_i, figure_captions_by_page=fig_caps_i)
    if expert_max_artifacts and len(artifacts) > int(expert_max_artifacts):
        artifacts = artifacts[: int(expert_max_artifacts)]

    table_metrics: dict[str, dict[str, Any]] = {}
    if tests_path is not None and tests_path.exists() and tests_path.stat().st_size > 0:
        try:
            table_metrics = _summarize_tables_from_tests(
                tests_path=tests_path,
                table_captions_by_page=table_caps_i,
                max_flagged_per_table=int(expert_max_flagged_per_table),
            )
            # Add best-effort row/col label guesses for flagged entries, so the LLM can cite concrete locations.
            tbl_min = heuristics.get("table_min_coef_y0_by_table") if isinstance(heuristics.get("table_min_coef_y0_by_table"), dict) else None
            for tm in table_metrics.values():
                if not isinstance(tm, dict):
                    continue
                for k in ["flagged_p_0_05", "flagged_t_1_96"]:
                    ex = tm.get(k)
                    if isinstance(ex, list) and ex:
                        _annotate_test_examples_with_row_col_labels(doc, ex, table_min_coef_y0_by_table=tbl_min)
        except Exception:
            table_metrics = {}

    # Enrich artifacts with mentions + excerpts + extracted metrics
    for a in artifacts:
        aid = str(a.get("id") or "")
        a["mentions"] = _collect_mentions_for_artifact(page_texts, aid, max_hits_total=8)
        pages = a.get("pages") or []
        pp: list[int] = []
        for p in pages if isinstance(pages, list) else []:
            try:
                pp.append(int(p))
            except Exception:
                continue
        pp = pp[: max(1, int(expert_max_pages_per_artifact))]
        a["page_text_excerpts"] = {str(p): _page_excerpt(page_texts[p - 1], max_chars=int(expert_max_page_chars)) for p in pp if 1 <= p <= len(page_texts)}
        if a.get("kind") == "table":
            a["extracted_table_metrics"] = table_metrics.get(aid) or {}
            a["extracted_table_metrics_definitions"] = {
                "n_tests": "Number of extracted (coef, SE-paren) pairs mapped to this table (approx; extraction may miss cells).",
                "p_005_left": "Count of approximate two-sided p in [0.045, 0.050] (just-significant).",
                "p_005_right": "Count of approximate two-sided p in (0.050, 0.055] (just-non-significant).",
                "p_005_just_sig": "Alias of p_005_left (just-significant).",
                "p_005_just_nonsig": "Alias of p_005_right (just-non-significant).",
                "t_196_left": "Count of |t| in [1.76, 1.96] (just-below 1.96).",
                "t_196_right": "Count of |t| in (1.96, 2.16] (just-above 1.96).",
                "t_196_just_below": "Alias of t_196_left (just-below 1.96).",
                "t_196_just_above": "Alias of t_196_right (just-above 1.96).",
                "t_exact_2_count": "How many flagged |t|≈1.96 entries are exactly |t|==2.00 (suggests rounding/precision artifacts).",
                "sig_p_le_0_05": "Count of approximate p <= 0.05 (significant).",
                "sig_p_le_0_10": "Count of approximate p <= 0.10.",
                "pcurve_low_half": "Among p<=0.05, count with p<=0.025 (p-curve 'low half').",
                "pcurve_high_half": "Among p<=0.05, count with 0.025<p<=0.05 (p-curve 'high half').",
                "pcurve_right_skew_z": "Simple z-score: (low_half - high_half)/sqrt(n_sig).",
                "flagged_p_0_05": "List of closest-to-0.05 entries with page/row/col guesses, coef/se, |t|, p.",
                "flagged_t_1_96": "List of closest-to-1.96 entries with page/row/col guesses, coef/se, |t|, p.",
                "IMPORTANT": "All p-values are approximations reconstructed from extracted table numbers; treat as screening evidence, not ground truth.",
            }
    (expert_dir / "artifact_inventory.json").write_text(json.dumps(artifacts, ensure_ascii=False, indent=2), encoding="utf-8")

    system_json = _system_prompt_expert_json()

    # Chunk summaries
    chunk_path = expert_dir / "paper_chunk_summaries.json"
    if chunk_path.exists() and not force:
        chunk_summaries = json.loads(chunk_path.read_text(encoding="utf-8"))
        if not isinstance(chunk_summaries, list):
            chunk_summaries = []
    else:
        chunk_summaries: list[dict[str, Any]] = []
        n_pages = len(page_texts)
        step = max(1, int(expert_chunk_pages))
        for start in range(1, n_pages + 1, step):
            end = min(n_pages, start + step - 1)
            parts: list[str] = []
            for p in range(start, end + 1):
                parts.append(f"[PAGE {p}]\n" + _page_excerpt(page_texts[p - 1], max_chars=int(expert_max_page_chars)))
            chunk_text = "\n\n".join(parts)
            user = _build_expert_chunk_prompt(
                paper_title_hint=paper_title_hint,
                page_start=start,
                page_end=end,
                chunk_text=chunk_text,
                method_guide=method_guide,
            )
            obj = _call_llm_json_text(logger, llm, system=system_json, user=user, max_tokens=3500)
            if isinstance(obj, dict):
                chunk_summaries.append(obj)
            time.sleep(0.2)
        chunk_path.write_text(json.dumps(chunk_summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    # Paper digest
    digest_path = expert_dir / "paper_digest.json"
    if digest_path.exists() and not force:
        paper_digest = json.loads(digest_path.read_text(encoding="utf-8"))
        if not isinstance(paper_digest, dict):
            paper_digest = {}
    else:
        user = _build_expert_digest_prompt(
            paper_title_hint=paper_title_hint,
            paper_meta=paper_meta,
            chunk_summaries=chunk_summaries if isinstance(chunk_summaries, list) else [],
            artifacts=artifacts,
            method_guide=method_guide,
        )
        paper_digest = _call_llm_json_text(logger, llm, system=system_json, user=user, max_tokens=3500)
        digest_path.write_text(json.dumps(paper_digest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Artifact audits
    audits_path = expert_dir / "artifact_audits.json"
    if audits_path.exists() and not force:
        artifact_audits = json.loads(audits_path.read_text(encoding="utf-8"))
        if not isinstance(artifact_audits, list):
            artifact_audits = []
        # Normalize identity fields to avoid missing ids in downstream steps.
        if isinstance(artifact_audits, list) and isinstance(artifacts, list) and len(artifact_audits) == len(artifacts):
            for i, obj in enumerate(artifact_audits):
                if not isinstance(obj, dict):
                    continue
                a = artifacts[i]
                if isinstance(a, dict):
                    obj["kind"] = a.get("kind") or obj.get("kind")
                    obj["id"] = a.get("id") or obj.get("id")
                    obj["caption"] = a.get("caption") or obj.get("caption")
                    obj["pages"] = a.get("pages") or obj.get("pages")
    else:
        artifact_audits = []
        for a in artifacts:
            user = _build_expert_artifact_audit_prompt(
                artifact=a,
                paper_meta=paper_meta,
                paper_digest=paper_digest if isinstance(paper_digest, dict) else {},
                method_guide=method_guide,
            )
            obj = _call_llm_json_text(logger, llm, system=system_json, user=user, max_tokens=3500)
            if isinstance(obj, dict):
                # Enforce identity fields to avoid downstream omissions.
                obj["kind"] = a.get("kind") or obj.get("kind")
                obj["id"] = a.get("id") or obj.get("id")
                obj["caption"] = a.get("caption") or obj.get("caption")
                obj["pages"] = a.get("pages") or obj.get("pages")
                artifact_audits.append(obj)
            time.sleep(0.2)
        audits_path.write_text(json.dumps(artifact_audits, ensure_ascii=False, indent=2), encoding="utf-8")

    # Expert metrics JSON (LLM-produced)
    metrics_path = out_dir / "diagnostic_metrics.json"
    expert_metrics: dict[str, Any] = {}
    need_metrics = bool(force or force_metrics) or (not metrics_path.exists())
    if not need_metrics:
        try:
            expert_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            expert_metrics = {}
        ok0, _why0 = _validate_expert_metrics_schema(expert_metrics, artifact_inventory=artifacts)
        if not ok0:
            need_metrics = True

    if need_metrics:
        user = _build_expert_metrics_prompt(
            paper_meta=paper_meta,
            paper_digest=paper_digest if isinstance(paper_digest, dict) else {},
            artifact_inventory=artifacts,
            artifact_audits=artifact_audits,
        )
        expert_metrics = _call_llm_json_text(logger, llm, system=system_json, user=user, max_tokens=3500)
        expert_metrics["generated_at"] = _now_iso()

        ok, why = _validate_expert_metrics_schema(expert_metrics, artifact_inventory=artifacts)
        if not ok:
            schema_hint = json.dumps(
                {
                    "paper": {
                        "title": "string",
                        "overall_risk_score_0_100": "number",
                        "overall_risk_level": "Low|Moderate|High",
                        "top_concerns": ["string"],
                        "key_artifacts": ["string"],
                    },
                    "artifacts": [
                        {
                            "kind": "table|figure",
                            "id": "string",
                            "pages": ["number"],
                            "risk_score_0_100": "number",
                            "risk_level": "Low|Moderate|High",
                            "has_numeric_evidence": "boolean",
                            "evidence_counts": {
                                "n_tests": "number|null",
                                "p_005_just_sig": "number|null",
                                "p_005_just_nonsig": "number|null",
                                "t_196_just_below": "number|null",
                                "t_196_just_above": "number|null",
                                "t_exact_2_count": "number|null",
                                "sig_p_le_0_05": "number|null",
                                "sig_p_le_0_10": "number|null",
                                "pcurve_low_half": "number|null",
                                "pcurve_high_half": "number|null",
                                "pcurve_right_skew_z": "number|null",
                            },
                            "signals": ["string"],
                            "flagged_entries": [
                                {
                                    "page": "number",
                                    "row": "string|null",
                                    "col": "string|null",
                                    "coef": "string|null",
                                    "se_paren": "string|null",
                                    "abs_t": "number|null",
                                    "p_approx_2s": "number|null",
                                }
                            ],
                        }
                    ],
                    "generated_at": "string",
                },
                ensure_ascii=False,
                indent=2,
            )
            for attempt in range(2):
                repair_user = (
                    "Your previous output did NOT match the required metrics schema.\n"
                    f"Schema validation error: {why}\n\n"
                    "Rewrite the ENTIRE JSON object to match the schema EXACTLY.\n"
                    "- Only top-level keys: paper, artifacts, generated_at.\n"
                    "- Cover EVERY artifact in artifact_inventory (exactly once).\n"
                    "- Do NOT add any extra fields, citations, references, or DOIs.\n"
                    "- Output ONLY strict JSON.\n\n"
                    "paper_meta:\n"
                    + json.dumps(paper_meta, ensure_ascii=False, indent=2)
                    + "\n\nartifact_inventory:\n"
                    + json.dumps(artifacts, ensure_ascii=False, indent=2)
                    + "\n\nartifact_audits:\n"
                    + json.dumps(artifact_audits, ensure_ascii=False, indent=2)
                    + "\n\nSchema:\n"
                    + schema_hint
                )
                expert_metrics = _call_llm_json_text(logger, llm, system=system_json, user=repair_user, max_tokens=3500)
                expert_metrics["generated_at"] = _now_iso()
                ok, why = _validate_expert_metrics_schema(expert_metrics, artifact_inventory=artifacts)
                if ok:
                    break

        ok2, _why2 = _validate_expert_metrics_schema(expert_metrics, artifact_inventory=artifacts)
        if not ok2:
            # Last resort: derive a schema-compliant metrics JSON from artifact audits.
            expert_metrics = _fallback_expert_metrics_from_audits(
                paper_title=str(paper_meta.get("title") or paper_title_hint),
                artifact_inventory=artifacts,
                artifact_audits=artifact_audits,
            )

        metrics_path.write_text(json.dumps(expert_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # Final report (Markdown; LLM-produced)
    report_path = out_dir / "diagnostic.md"
    if report_path.exists() and not (force or force_report or force_metrics):
        report_md = report_path.read_text(encoding="utf-8", errors="replace")
    else:
        # Compact numeric evidence pack for the final report (avoid sending long page excerpts again).
        artifact_evidence: list[dict[str, Any]] = []
        for a in artifacts:
            if not isinstance(a, dict):
                continue
            item = {
                "kind": a.get("kind"),
                "id": a.get("id"),
                "caption": a.get("caption"),
                "pages": a.get("pages"),
            }
            if a.get("kind") == "table":
                tm = a.get("extracted_table_metrics") if isinstance(a.get("extracted_table_metrics"), dict) else {}
                item["has_numeric_evidence"] = bool(int(tm.get("n_tests") or 0) > 0)
                if tm:
                    item["evidence_counts"] = {k: tm.get(k) for k in [
                        "n_tests",
                        "p_005_just_sig",
                        "p_005_just_nonsig",
                        "t_196_just_below",
                        "t_196_just_above",
                        "t_exact_2_count",
                        "sig_p_le_0_05",
                        "sig_p_le_0_10",
                        "pcurve_low_half",
                        "pcurve_high_half",
                        "pcurve_right_skew_z",
                    ]}
                    item["flagged_p_0_05"] = (tm.get("flagged_p_0_05") or [])[:10]
                    item["flagged_t_1_96"] = (tm.get("flagged_t_1_96") or [])[:10]
            else:
                item["has_numeric_evidence"] = False
            artifact_evidence.append(item)

        system_md = _system_prompt_expert_markdown()

        audits_by_id: dict[str, dict[str, Any]] = {}
        for obj in artifact_audits:
            if isinstance(obj, dict) and (obj.get("id") is not None):
                audits_by_id[str(obj.get("id"))] = obj

        metrics_by_id: dict[str, dict[str, Any]] = {}
        mets = expert_metrics.get("artifacts") if isinstance(expert_metrics, dict) else None
        if isinstance(mets, list):
            for m in mets:
                if isinstance(m, dict) and (m.get("id") is not None):
                    metrics_by_id[str(m.get("id"))] = m

        artifact_inventory_compact: list[dict[str, Any]] = []
        for a in artifact_evidence:
            aid = str(a.get("id") or "")
            m = metrics_by_id.get(aid, {})
            ec = m.get("evidence_counts") if isinstance(m.get("evidence_counts"), dict) else {}
            artifact_inventory_compact.append(
                {
                    "kind": a.get("kind"),
                    "id": a.get("id"),
                    "caption": a.get("caption"),
                    "pages": a.get("pages"),
                    "risk_score_0_100": m.get("risk_score_0_100"),
                    "risk_level": m.get("risk_level"),
                    "has_numeric_evidence": m.get("has_numeric_evidence"),
                    "n_tests": (ec.get("n_tests") if isinstance(ec, dict) else None),
                }
            )

        paper_title = str(
            paper_meta.get("title")
            or (paper_digest.get("paper_title") if isinstance(paper_digest, dict) else None)
            or "Paper"
        ).strip() or "Paper"

        # Generate the report in parts to avoid truncation (provider-dependent output limits).
        expert_dir = out_dir / "expert"
        sections_dir = expert_dir / "report_sections"
        sections_dir.mkdir(parents=True, exist_ok=True)

        header_user = _build_expert_report_header_prompt(
            paper_meta=paper_meta,
            paper_digest=paper_digest if isinstance(paper_digest, dict) else {},
            expert_metrics=expert_metrics,
            artifact_inventory_compact=artifact_inventory_compact,
            method_guide=method_guide,
        )
        header_res = _chat_text_logged_retry(
            llm,
            logger,
            system_prompt=system_md,
            user_prompt=header_user,
            temperature=0.0,
            max_tokens=4200,
            json_mode=False,
            notes="expert_report_header",
        )
        header_md = (header_res.content or "").strip()
        required_headings = [
            "# Selective-reporting / p-hacking risk screening:",
            "## Executive summary",
            "## Paper context (claims & empirical setup)",
            "## What I actually checked (workflow)",
            "## Methodological basis (diagnostics & theory)",
            "## Artifact-by-artifact audit (tables & figures)",
        ]
        missing = [h for h in required_headings if h not in header_md]
        if missing or ("### " in header_md):
            for attempt in range(2):
                repair_user = (
                    "Your previous header output violated hard requirements.\n"
                    f"- Missing headings: {missing}\n"
                    f"- Contains artifact subsections ('###'): {('### ' in header_md)}\n\n"
                    "Rewrite the header part from scratch.\n"
                    "- Include ALL required headings in order.\n"
                    "- Do NOT include any '###' subsections.\n"
                    "- End EXACTLY with the heading line: '## Artifact-by-artifact audit (tables & figures)'.\n"
                    "- Keep it concise (<= 900 words).\n"
                    "- Output ONLY Markdown.\n\n"
                    + header_user
                )
                header_res = _chat_text_logged_retry(
                    llm,
                    logger,
                    system_prompt=system_md,
                    user_prompt=repair_user,
                    temperature=0.0,
                    max_tokens=4200,
                    json_mode=False,
                    notes=f"expert_report_header_repair_{attempt+1}",
                )
                header_md = (header_res.content or "").strip()
                missing = [h for h in required_headings if h not in header_md]
                if not missing and ("### " not in header_md):
                    break
        (sections_dir / "00_header.md").write_text(header_md + "\n", encoding="utf-8")

        batch_size = 4
        artifact_sections: list[str] = []
        for i in range(0, len(artifact_evidence), batch_size):
            batch = artifact_evidence[i : i + batch_size]
            audits_batch = [audits_by_id.get(str(b.get("id") or ""), {}) for b in batch]
            batch_user = _build_expert_report_artifact_batch_prompt(
                paper_title=paper_title,
                artifact_evidence_batch=batch,
                artifact_audits_batch=audits_batch,
                method_guide=method_guide,
            )
            batch_res = _chat_text_logged_retry(
                llm,
                logger,
                system_prompt=system_md,
                user_prompt=batch_user,
                temperature=0.0,
                max_tokens=4200,
                json_mode=False,
                notes=f"expert_report_artifacts_batch_{(i // batch_size) + 1}",
            )
            batch_md = (batch_res.content or "").strip()
            artifact_sections.append(batch_md)
            (sections_dir / f"10_artifacts_batch_{(i // batch_size) + 1:02d}.md").write_text(batch_md + "\n", encoding="utf-8")
            time.sleep(0.2)

        # Ensure every artifact appears at least once; regenerate missing ones (rare provider noncompliance).
        combined_artifacts_md = "\n\n".join(artifact_sections)
        missing_ids = []
        for a in artifact_evidence:
            aid = str(a.get("id") or "")
            if not aid:
                continue
            if f"### {aid}:" not in combined_artifacts_md:
                missing_ids.append(aid)
        if missing_ids:
            for aid in missing_ids[:12]:
                a = next((x for x in artifact_evidence if str(x.get("id") or "") == aid), None)
                if not isinstance(a, dict):
                    continue
                audits_one = [audits_by_id.get(aid, {})]
                one_user = _build_expert_report_artifact_batch_prompt(
                    paper_title=paper_title,
                    artifact_evidence_batch=[a],
                    artifact_audits_batch=audits_one,
                    method_guide=method_guide,
                )
                one_res = _chat_text_logged_retry(
                    llm,
                    logger,
                    system_prompt=system_md,
                    user_prompt=one_user,
                    temperature=0.0,
                    max_tokens=2600,
                    json_mode=False,
                    notes=f"expert_report_artifact_missing_{aid}",
                )
                one_md = (one_res.content or "").strip()
                artifact_sections.append(one_md)
                (sections_dir / f"11_artifact_missing_{_slugify(aid)}.md").write_text(one_md + "\n", encoding="utf-8")
                time.sleep(0.2)

        tail_user = _build_expert_report_tail_prompt(
            paper_title=paper_title,
            expert_metrics=expert_metrics,
            artifact_inventory_compact=artifact_inventory_compact,
            apa_refs=apa_refs,
        )
        tail_res = _chat_text_logged_retry(
            llm,
            logger,
            system_prompt=system_md,
            user_prompt=tail_user,
            temperature=0.0,
            max_tokens=3200,
            json_mode=False,
            notes="expert_report_tail",
        )
        tail_md = (tail_res.content or "").strip()
        tail_required = [
            "## Overall assessment & suggested author responses",
            "## Limitations",
            "## References (APA, with DOI)",
        ]
        missing_tail = [h for h in tail_required if h not in tail_md]
        if missing_tail:
            for attempt in range(2):
                repair_user = (
                    "Your previous tail output is missing required headings.\n"
                    f"Missing: {missing_tail}\n\n"
                    "Rewrite the tail part from scratch. Start with '## Overall assessment & suggested author responses' and end with '## References (APA, with DOI)'.\n"
                    "Output ONLY Markdown.\n\n"
                    + tail_user
                )
                tail_res = _chat_text_logged_retry(
                    llm,
                    logger,
                    system_prompt=system_md,
                    user_prompt=repair_user,
                    temperature=0.0,
                    max_tokens=3200,
                    json_mode=False,
                    notes=f"expert_report_tail_repair_{attempt+1}",
                )
                tail_md = (tail_res.content or "").strip()
                missing_tail = [h for h in tail_required if h not in tail_md]
                if not missing_tail:
                    break
        (sections_dir / "99_tail.md").write_text(tail_md + "\n", encoding="utf-8")

        report_md = "\n\n".join([header_md, *artifact_sections, tail_md]).strip()
        report_path.write_text(report_md + "\n", encoding="utf-8")

    return report_md, expert_metrics, paper_digest if isinstance(paper_digest, dict) else {}, artifacts, artifact_audits


def main() -> int:
    ap = argparse.ArgumentParser(description="Multimodal within-paper p-hacking risk diagnostic agent (no framework).")
    ap.add_argument("--pdf", required=True, help="PDF path to diagnose.")
    ap.add_argument("--out-dir", default="reports/p_hacking", help="Base output directory.")
    ap.add_argument(
        "--corpus-dir",
        default=None,
        help="Optional corpus directory to attach extracted tests from <corpus>/tests/<paper_id>.jsonl (improves offline evidence).",
    )
    ap.add_argument(
        "--auto-extract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If no usable tests are found, auto-run within-paper extraction into <out_dir>/_auto_corpus (expert mode).",
    )
    ap.add_argument("--auto-extract-max-pages-per-paper", type=int, default=24, help="Max candidate pages for auto extraction.")
    ap.add_argument("--auto-extract-max-pdf-pages", type=int, default=None, help="Skip PDFs over this page count in auto extraction.")
    ap.add_argument("--method-md", default="p_hacking_agent_methodology.md", help="Method summary markdown for citations.")
    ap.add_argument("--max-image-pages", type=int, default=8, help="Max pages to analyze with page images.")
    ap.add_argument(
        "--expert",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use end-to-end LLM expert workflow (full-context + artifact-by-artifact audit). Use --no-expert for legacy light mode.",
    )
    ap.add_argument("--expert-chunk-pages", type=int, default=8, help="Pages per LLM context chunk (expert mode).")
    ap.add_argument("--expert-max-page-chars", type=int, default=3200, help="Max chars per page fed to LLM (expert mode).")
    ap.add_argument("--expert-max-pages-per-artifact", type=int, default=4, help="Max pages excerpted per table/figure (expert mode).")
    ap.add_argument("--expert-max-artifacts", type=int, default=40, help="Max number of tables/figures to audit (expert mode).")
    ap.add_argument("--expert-max-flagged-per-table", type=int, default=25, help="Max borderline entries kept per table (expert mode).")
    ap.add_argument("--force", action="store_true", help="Re-run even if cached outputs exist.")
    ap.add_argument("--force-metrics", action="store_true", help="(expert mode) Rebuild diagnostic_metrics.json even if cached.")
    ap.add_argument("--force-report", action="store_true", help="(expert mode) Rebuild diagnostic.md/report_sections even if cached.")
    ap.add_argument("--no-crop", action="store_true", help="Disable table-region cropping when rendering page images.")
    ap.add_argument("--offline", action="store_true", help="Disable LLM calls (fallback to heuristic/text-only mode).")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    method_md_path = Path(args.method_md)
    apa_refs = _extract_apa_references(method_md_path)
    method_guide = _build_method_citation_guide(apa_refs)

    # Output folder
    pdf_hash = _sha256_file(pdf_path, max_bytes=5_000_000)[:12]
    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    out_dir = out_base / f"{_slugify(pdf_path.stem)}__{pdf_hash}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(2)}"
    logs_dir = out_dir / "llm_logs" / run_id
    logs_dir.mkdir(parents=True, exist_ok=True)

    system = _system_prompt_agent()

    cfg = None
    llm = None
    llm_disabled_reason: str | None = None
    if args.offline:
        llm_disabled_reason = "offline mode (--offline)"
    else:
        cfg, llm = _load_llm_client()

    base_url = str(getattr(cfg, "base_url", "") or "")
    model = str(getattr(cfg, "model", "") or "")
    api_key = str(getattr(cfg, "api_key", "") or "")
    api_key_masked = re.sub(r".(?=.{4})", "*", api_key) if api_key else ""
    if args.offline:
        print(f"[{_now_iso()}] LLM disabled: {llm_disabled_reason or 'offline'}")
    else:
        print(f"[{_now_iso()}] LLM: base_url={base_url} model={model} api_key={api_key_masked}")
    print(f"[{_now_iso()}] PDF: {pdf_path} -> {out_dir}")

    logger = LLMRunLogger(logs_dir, model=model, base_url=base_url)
    run_meta_path = logs_dir / "run.json"
    run_meta = {
        "run_id": run_id,
        "started_at": _now_iso(),
        "pdf_path": str(pdf_path),
        "pdf_sha256_prefix": pdf_hash,
        "out_dir": str(out_dir),
        "logs_dir": str(logs_dir),
        "llm_base_url": base_url or None,
        "llm_model": model or None,
        "llm_api_key_masked": api_key_masked or None,
        "llm_disabled_reason": llm_disabled_reason,
        "args": vars(args),
    }
    run_meta_path.write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    doc = fitz.open(pdf_path)
    page_count = doc.page_count

    # Extract per-page text + features
    page_texts: list[str] = []
    features: list[PageFeatures] = []
    for i in range(page_count):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        page_texts.append(text)
        features.append(_page_features(i + 1, text))

    paper_title_hint = doc.metadata.get("title") or pdf_path.stem
    # Heuristics over whole text
    full_text = "\n\n".join(page_texts)
    pvals = _p_value_regex_hits(full_text)
    heuristics: dict[str, Any] = {
        "pdf_pages": page_count,
        "extracted_text_chars": len(full_text),
        "p_values_found": len(pvals),
        "p_values_sample": sorted(pvals)[:30],
        "near_0_05": _bin_counts(pvals, 0.045, 0.05, 0.055),
        "near_0_10": _bin_counts(pvals, 0.095, 0.10, 0.105),
        "near_0_01": _bin_counts(pvals, 0.009, 0.01, 0.011),
        "star_count_fulltext": full_text.count("*"),
        "table_mentions_fulltext": len(re.findall(r"(?i)\btable\b", full_text)),
        "robust_mentions_fulltext": len(re.findall(r"(?i)\brobust(?:ness)?\b", full_text)),
    }
    heuristics["table_captions_by_page"] = _extract_table_captions_by_page(page_texts)
    heuristics["figure_captions_by_page"] = _extract_figure_captions_by_page(page_texts)

    tests_borderline_by_page: dict[int, list[dict[str, str]]] = {}
    tests_path_resolved: Path | None = None
    if args.corpus_dir:
        try:
            corpus_dir = Path(args.corpus_dir)
            pdf_sha256_full = _sha256_file(pdf_path, max_bytes=int(pdf_path.stat().st_size) + 1)

            # Prefer matching by sha256 so renamed PDFs still map to the same extracted tests.
            row = _match_corpus_features_row(corpus_dir=corpus_dir, sha256=pdf_sha256_full, paper_id=pdf_path.stem)
            paper_id = (row.get("paper_id") or "").strip() if isinstance(row, dict) else ""
            tests_rel = (row.get("tests_relpath") or "").strip() if isinstance(row, dict) else ""

            tests_path = None
            if tests_rel:
                tests_path = corpus_dir / tests_rel
            if tests_path is None or not tests_path.exists():
                pid = paper_id or pdf_path.stem
                tests_path = corpus_dir / "tests" / f"{pid}.jsonl"

            heuristics["paper_id"] = paper_id or pdf_path.stem
            score = None
            try:
                if isinstance(row, dict):
                    score = float(row.get("offline_risk_score") or "")
            except Exception:
                score = None
            if score is None:
                score = _load_offline_risk_score_from_features(corpus_dir=corpus_dir, paper_id=paper_id or pdf_path.stem)
            if isinstance(score, (int, float)) and score == score:
                heuristics["offline_risk_score"] = float(score)

            if tests_path.exists() and tests_path.stat().st_size > 0:
                tests_path_resolved = tests_path
                tests_summary, tests_borderline_by_page = _load_tests_summary(tests_path)
                heuristics["tests_relpath"] = str(tests_path.relative_to(corpus_dir)).replace("\\", "/")
                heuristics.update(tests_summary)
                try:
                    tbl_min = heuristics.get("table_min_coef_y0_by_table")
                    ex1 = heuristics.get("p_from_t_near_0_05_examples")
                    if isinstance(ex1, list) and ex1:
                        _annotate_test_examples_with_row_col_labels(doc, ex1, table_min_coef_y0_by_table=tbl_min)
                    ex2 = heuristics.get("t_near_1_96_examples")
                    if isinstance(ex2, list) and ex2:
                        _annotate_test_examples_with_row_col_labels(doc, ex2, table_min_coef_y0_by_table=tbl_min)
                except Exception:
                    pass
        except Exception:
            pass

    # If expert mode is requested but we don't have extracted tests yet, auto-run extraction into the output folder.
    if (
        tests_path_resolved is None
        and bool(args.auto_extract)
        and (llm is not None)
        and (not args.offline)
        and bool(args.expert)
    ):
        try:
            auto_corpus_dir, auto_tests = _auto_extract_tests_for_pdf(
                out_dir=out_dir,
                pdf_path=pdf_path,
                max_pages_per_paper=int(args.auto_extract_max_pages_per_paper),
                max_pdf_pages=args.auto_extract_max_pdf_pages,
                force=bool(args.force),
            )
            heuristics["auto_corpus_dir"] = str(auto_corpus_dir)
            if auto_tests is not None:
                tests_path_resolved = auto_tests
                heuristics["tests_relpath"] = str(auto_tests.relative_to(auto_corpus_dir)).replace("\\", "/")
        except Exception as e:
            print(f"[{_now_iso()}] auto-extract failed: {type(e).__name__}: {e}")

    # Expert end-to-end workflow (LLM-only report)
    if llm is not None and (not args.offline) and bool(args.expert):
        try:
            paper_meta = _offline_paper_meta(doc, pdf_path, first_page_text=page_texts[0] if page_texts else "")
        except Exception:
            paper_meta = {"title": pdf_path.stem, "authors": [], "year": None, "venue_or_series": None, "doi_or_url": None}

        report_md, expert_metrics, paper_digest, artifacts, artifact_audits = _run_expert_workflow(
            llm=llm,
            logger=logger,
            doc=doc,
            pdf_path=pdf_path,
            out_dir=out_dir,
            cache_dir=cache_dir,
            page_texts=page_texts,
            paper_meta=paper_meta,
            paper_title_hint=paper_title_hint,
            heuristics=heuristics,
            tests_path=tests_path_resolved,
            apa_refs=apa_refs,
            method_guide=method_guide,
            force=bool(args.force),
            force_metrics=bool(args.force_metrics),
            force_report=bool(args.force_report),
            expert_chunk_pages=int(args.expert_chunk_pages),
            expert_max_page_chars=int(args.expert_max_page_chars),
            expert_max_pages_per_artifact=int(args.expert_max_pages_per_artifact),
            expert_max_artifacts=int(args.expert_max_artifacts),
            expert_max_flagged_per_table=int(args.expert_max_flagged_per_table),
        )

        final_json_path = out_dir / "diagnostic.json"
        final_payload = {
            "paper_title": paper_meta.get("title") or paper_title_hint,
            "paper_meta": paper_meta,
            "heuristics": heuristics,
            "expert_paper_digest": paper_digest,
            "expert_artifacts": artifacts,
            "expert_artifact_audits": artifact_audits,
            "expert_metrics": expert_metrics,
            "report_md": report_md,
            "generated_at": _now_iso(),
            "model": model or None,
            "base_url": base_url or None,
            "llm_logs_dir": str(logs_dir),
        }
        final_json_path.write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{_now_iso()}] Wrote: {out_dir / 'diagnostic.md'}")
        print(f"[{_now_iso()}] Wrote: {out_dir / 'diagnostic_metrics.json'}")
        print(f"[{_now_iso()}] Wrote: {final_json_path}")
        return 0

    # Page selection (cached)
    selected_pages_path = cache_dir / "selected_pages.json"
    if selected_pages_path.exists() and not args.force:
        selected_obj = json.loads(selected_pages_path.read_text(encoding="utf-8"))
    else:
        if llm is None:
            selected_obj = {
                "selected_pages": _heuristic_select_pages(features, max_pages=int(args.max_image_pages)),
                "notes": f"LLM disabled; used heuristic selection instead. reason={llm_disabled_reason}",
            }
        else:
            sel_user = _build_page_selection_prompt(
                title_hint=paper_title_hint,
                page_features=features,
                max_pages=max(1, int(args.max_image_pages)),
            )
            try:
                selected_obj = _call_llm_json_text(logger, llm, system=system, user=sel_user, max_tokens=2500)
            except Exception as e:
                # Fall back to deterministic heuristics to avoid being blocked by JSON formatting issues.
                selected_obj = {
                    "selected_pages": _heuristic_select_pages(features, max_pages=int(args.max_image_pages)),
                    "notes": f"LLM page selection failed; used heuristic selection instead. error={type(e).__name__}: {e}",
                }
                if type(e).__name__.endswith("LLMError"):
                    llm_disabled_reason = llm_disabled_reason or f"LLM unavailable: {type(e).__name__}: {e}"
                    llm = None
        selected_pages_path.write_text(json.dumps(selected_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    selected_pages = selected_obj.get("selected_pages", [])
    if not isinstance(selected_pages, list):
        selected_pages = []

    # Normalize page list, enforce bounds and cap
    picked: list[dict[str, Any]] = []
    seen_pages: set[int] = set()
    for item in selected_pages:
        if not isinstance(item, dict):
            continue
        p = item.get("page")
        if isinstance(p, str) and p.strip().isdigit():
            p = int(p.strip())
        if not isinstance(p, int):
            continue
        if p < 1 or p > page_count:
            continue
        if p in seen_pages:
            continue
        seen_pages.add(p)
        picked.append({"page": p, "role": item.get("role"), "reason": item.get("reason")})
        if len(picked) >= int(args.max_image_pages):
            break

    # Always include first page (title/abstract) for context if not present
    if 1 not in seen_pages and len(picked) < int(args.max_image_pages):
        picked.insert(0, {"page": 1, "role": "title/abstract", "reason": "Always include first page for context."})

    # Render and analyze selected pages with images
    page_evidence: list[dict[str, Any]] = []
    for item in picked:
        p1 = int(item["page"])
        cache_path = cache_dir / f"page_{p1:03d}_evidence.json"
        if cache_path.exists() and not args.force:
            obj = json.loads(cache_path.read_text(encoding="utf-8"))
            page_evidence.append(obj)
            continue

        img_bytes = _render_page_image_jpg(
            doc,
            p1 - 1,
            crop_table_region=not args.no_crop,
        )
        img_path = pages_dir / f"page_{p1:03d}.jpg"
        img_path.write_bytes(img_bytes)

        if llm is None:
            obj0 = _offline_page_evidence(page_1based=p1, extracted_text=page_texts[p1 - 1])
            try:
                extra = tests_borderline_by_page.get(int(p1)) if tests_borderline_by_page else None
                if isinstance(extra, list) and extra:
                    br = obj0.get("borderline_results")
                    if not isinstance(br, list):
                        br = []
                    for item2 in extra:
                        if len(br) >= 3:
                            break
                        if isinstance(item2, dict):
                            br.append(item2)
                    obj0["borderline_results"] = br[:3]
                    obj0["confidence_0_1"] = max(float(obj0.get("confidence_0_1") or 0.0), 0.35)
            except Exception:
                pass
        else:
            user = _build_page_evidence_prompt(
                paper_title=paper_title_hint,
                page_1based=p1,
                extracted_text=page_texts[p1 - 1],
            )
            obj0 = None
            err0: Exception | None = None
            try:
                obj0 = _call_llm_json_image(
                    logger,
                    llm,
                    system=system,
                    user=user,
                    image_bytes=img_bytes,
                    image_path=str(img_path),
                    max_tokens=4096,
                )
            except Exception as e:
                err0 = e
                # Fallback: much shorter prompt, no extracted text.
                user2 = _build_page_evidence_prompt_minimal(paper_title=paper_title_hint, page_1based=p1)
                try:
                    obj0 = _call_llm_json_image(
                        logger,
                        llm,
                        system=system,
                        user=user2,
                        image_bytes=img_bytes,
                        image_path=str(img_path),
                        max_tokens=2500,
                    )
                except Exception as e2:
                    err0 = e2
            if not isinstance(obj0, dict):
                obj0 = _offline_page_evidence(page_1based=p1, extracted_text=page_texts[p1 - 1])
                try:
                    extra = tests_borderline_by_page.get(int(p1)) if tests_borderline_by_page else None
                    if isinstance(extra, list) and extra:
                        br = obj0.get("borderline_results")
                        if not isinstance(br, list):
                            br = []
                        for item2 in extra:
                            if len(br) >= 3:
                                break
                            if isinstance(item2, dict):
                                br.append(item2)
                        obj0["borderline_results"] = br[:3]
                        obj0["confidence_0_1"] = max(float(obj0.get("confidence_0_1") or 0.0), 0.35)
                except Exception:
                    pass
                if err0 is not None and type(err0).__name__.endswith("LLMError"):
                    llm_disabled_reason = llm_disabled_reason or f"LLM unavailable: {type(err0).__name__}: {err0}"
                    llm = None

        if isinstance(obj0, dict):
            obj0["page"] = p1
        cache_path.write_text(json.dumps(obj0, ensure_ascii=False, indent=2), encoding="utf-8")
        page_evidence.append(obj0)
        time.sleep(0.2)

    # Build paper meta using a cheap vision call on first page (cached)
    meta_path = cache_dir / "paper_meta.json"
    if meta_path.exists() and not args.force:
        paper_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        img_bytes = _render_page_image_jpg(doc, 0, crop_table_region=False, max_side_px=1600)
        img_path = pages_dir / "page_001_full.jpg"
        img_path.write_bytes(img_bytes)
        if llm is None:
            paper_meta = _offline_paper_meta(doc, pdf_path, first_page_text=page_texts[0])
        else:
            user = (
                "Extract bibliographic metadata from this paper's first-page image as accurately as possible. Output strict JSON:\n"
                "{\n"
                '  \"title\": string|null,\n'
                '  \"authors\": [string],\n'
                '  \"year\": number|null,\n'
                '  \"venue_or_series\": string|null,\n'
                '  \"doi_or_url\": string|null\n'
                "}\n"
            )
            try:
                meta0 = _call_llm_json_image(
                    logger,
                    llm,
                    system=system,
                    user=user,
                    image_bytes=img_bytes,
                    image_path=str(img_path),
                    max_tokens=2000,
                )
                paper_meta = meta0 if isinstance(meta0, dict) else {}
            except Exception as e:
                paper_meta = _offline_paper_meta(doc, pdf_path, first_page_text=page_texts[0])
                if type(e).__name__.endswith("LLMError"):
                    llm_disabled_reason = llm_disabled_reason or f"LLM unavailable: {type(e).__name__}: {e}"
                    llm = None
        meta_path.write_text(json.dumps(paper_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Final synthesis (cached)
    final_json_path = out_dir / "diagnostic.json"
    final_md_path = out_dir / "diagnostic.md"
    if final_json_path.exists() and final_md_path.exists() and not args.force:
        print(f"[{_now_iso()}] Using cached outputs: {final_md_path}")
        return 0

    final_user = _build_final_report_prompt(
        paper_title=paper_meta.get("title") or paper_title_hint,
        paper_meta=paper_meta,
        heuristics=heuristics,
        selected_pages=picked,
        page_evidence=page_evidence,
        apa_refs=apa_refs,
    )
    final_system = system + "\n\nFinal output must be Markdown text (not JSON; no code fences)."
    report_md = ""
    if llm is None:
        report_md = _compose_offline_report(
            paper_title=paper_meta.get("title") or paper_title_hint,
            paper_meta=paper_meta,
            heuristics=heuristics,
            selected_pages=picked,
            page_evidence=page_evidence,
            apa_refs=apa_refs,
            llm_disabled_reason=llm_disabled_reason,
        )
    else:
        try:
            res = _chat_text_logged_retry(
                llm,
                logger,
                system_prompt=final_system,
                user_prompt=final_user,
                temperature=0.0,
                max_tokens=3500,
                json_mode=False,
                notes="final_report_md",
            )
            report_md = (res.content or "").strip()
        except Exception as e:
            if type(e).__name__.endswith("LLMError"):
                llm_disabled_reason = llm_disabled_reason or f"LLM unavailable: {type(e).__name__}: {e}"
                llm = None
            report_md = _compose_offline_report(
                paper_title=paper_meta.get("title") or paper_title_hint,
                paper_meta=paper_meta,
                heuristics=heuristics,
                selected_pages=picked,
                page_evidence=page_evidence,
                apa_refs=apa_refs,
                llm_disabled_reason=llm_disabled_reason,
            )
    report_md = _ensure_method_references(report_md, apa_refs)

    # Write outputs (JSON assembled by us to avoid model JSON instability).
    final_payload = {
        "paper_title": paper_meta.get("title") or paper_title_hint,
        "paper_meta": paper_meta,
        "heuristics": heuristics,
        "selected_pages": picked,
        "page_evidence": page_evidence,
        "method_references_apa": apa_refs,
        "report_md": report_md,
        "generated_at": _now_iso(),
        "model": model or None,
        "base_url": base_url or None,
        "llm_disabled_reason": llm_disabled_reason,
        "llm_logs_dir": str(logs_dir),
    }
    final_json_path.write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if report_md:
        final_md_path.write_text(report_md + "\n", encoding="utf-8")
    else:
        final_md_path.write_text("# p-hacking risk screening report\n\n(LLM returned empty content)\n", encoding="utf-8")

    print(f"[{_now_iso()}] Wrote: {final_md_path}")
    print(f"[{_now_iso()}] Wrote: {final_json_path}")

    # Keep a per-run copy next to logs for reproducibility.
    try:
        (logs_dir / "diagnostic.md").write_text(report_md + "\n", encoding="utf-8")
        (logs_dir / "diagnostic.json").write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    try:
        run_meta["finished_at"] = _now_iso()
        run_meta["diagnostic_md"] = str(final_md_path)
        run_meta["diagnostic_json"] = str(final_json_path)
        run_meta["llm_disabled_reason"] = llm_disabled_reason
        run_meta_path.write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
