#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import io
import json
import os
import re
import secrets
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from PIL import Image


def _default_llm_api_client_scripts_dir() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home).expanduser() / "skills" / "llm-api-client" / "scripts"
    return Path.home() / ".codex" / "skills" / "llm-api-client" / "scripts"


SKILL_LLM_SCRIPTS_DIR = _default_llm_api_client_scripts_dir()


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
    table_names = sorted(set(re.findall(r"(?i)\btable\s+[a-z]?\d+[a-z]?\b", t)))[:6]
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
    # Basic heuristic scoring (conservative; offline evidence is weaker).
    score = 20
    try:
        near = heuristics.get("near_0_05") or {}
        if isinstance(near, dict) and int(near.get("total") or 0) >= 10:
            left = int(near.get("left") or 0)
            right = int(near.get("right") or 0)
            if left >= right + 3:
                score += 12
            elif left >= right + 1:
                score += 6
        if int(heuristics.get("table_mentions_fulltext") or 0) >= 15:
            score += 6
        if int(heuristics.get("robust_mentions_fulltext") or 0) >= 5:
            score += 6
        full_text_has_correction = bool(
            re.search(
                r"(?i)\b(bonferroni|benjamini|hochberg|holm|fdr|fwer|multiple\s+(testing|hypothesis))\b",
                "\n".join(
                    [
                        json.dumps(heuristics, ensure_ascii=False),
                        json.dumps(page_evidence, ensure_ascii=False),
                    ]
                ),
            )
        )
        if full_text_has_correction:
            score -= 8
    except Exception:
        pass
    score = max(0, min(100, int(score)))
    level = _risk_level(score)

    refs_block = "\n".join(f"- {r}" for r in apa_refs)

    # Pull some anchors from selected pages as “evidence handles”
    anchor_lines: list[str] = []
    for ev in page_evidence:
        if not isinstance(ev, dict):
            continue
        p = ev.get("page")
        sigs = ev.get("signals")
        if isinstance(sigs, list):
            for s in sigs:
                if not isinstance(s, dict):
                    continue
                anchors = s.get("anchors")
                if isinstance(anchors, list) and anchors:
                    anchor_lines.append(f"p.{p}: {anchors[0]}")
        if len(anchor_lines) >= 6:
            break

    def first_anchor_or_na() -> str:
        return anchor_lines[0] if anchor_lines else "(no text anchors; see saved page images)"

    lines: list[str] = []
    lines.append(f"# 篇内 p-hacking 风险诊断报告：{paper_title}")
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
        lines.append(f"**诊断对象：** {who}")
        if doi:
            lines.append(f"**DOI/URL：** {doi}")
        lines.append("")
    if llm_disabled_reason:
        lines.append(f"> 注：本次运行 LLM 不可用，已降级为纯启发式/文本抽取模式。原因：{_clean_snippet(llm_disabled_reason, max_len=240)}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("### 一、总体评估结果")
    lines.append("")
    lines.append(f"**Risk Score: {score}/100**")
    lines.append(f"**Risk Level: {level} (离线启发式)**")
    lines.append("")
    lines.append("**诊断综述：**")
    lines.append(
        "本报告在缺少 LLM 逐页读图的情况下，仅基于 PDF 文本抽取与少量启发式统计给出保守诊断。"
        "篇内证据天然较弱：无法证明 p-hacking，只能提示“哪里值得复核”。"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("### 二、证据卡片 (Evidence Cards)")
    lines.append("")

    # Always cover key modules; cite method sources.
    cards = [
        (
            "阈值附近异常 (Threshold Bunching)",
            "文本中若存在大量临界显著结果（p≈0.05 或 t≈1.96），需要警惕阈值附近堆积/缺口。",
            "在许多领域，0.05 阈值附近的异常堆积常被用作显著性通胀/规格搜索的信号。",
            "下一步：从表格中系统抽取 t/z/p 并对 0.05 附近做 caliper test / 分布检验。",
            "(Brodeur et al., 2016; Brodeur et al., 2020)",
        ),
        (
            "规格搜索与研究者自由度 (Specification Searching)",
            "若论文存在大量可选规格（控制变量、样本、变量定义、窗口、子样本等），且只突出最有利结果，风险上升。",
            "规格搜索会使经典显著性推断失效，容易把“挑出来的显著”当作稳健发现。",
            "下一步：要求作者提供完整规格网格/所有尝试过的定义与回归；做稳健性全披露。",
            "(Leamer, 1978; Leamer, 1983)",
        ),
        (
            "多重检验与选择性强调 (Multiple Testing)",
            "当 outcome/机制/异质性指标很多时，若未做 FDR/FWER 校正或更高门槛控制，假阳性风险上升。",
            "大量检验会让“名义 5% 显著”不再可靠，需要校准或提高显著性门槛。",
            "下一步：统计检验数量 K，报告 FDR/FWER/更高 t 门槛下的稳健性。",
            "(Harvey et al., 2015; Harvey, 2017)",
        ),
        (
            "p-curve / 形状约束诊断 (p-hacking Tests)",
            "若能抽取足够多的 p 值，可用 p-curve/形状约束检验识别 p-hacking 或选择性报告。",
            "形状约束提供了可检验限制（testable implications），可将“异常的显著性分布”形式化。",
            "下一步：抽取完整 p 值集合，进行 p-curve 形状检验/上界诊断。",
            "(Elliott et al., 2022)",
        ),
        (
            "发表/报告选择偏倚 (Selection/Publication Bias)",
            "篇内也可能存在选择性报告（正文/附录取舍、安慰剂结果叙事偏置）。",
            "选择性发表/报告会导致估计膨胀与过度自信的置信区间，需要显式建模或校正。",
            "下一步：检查注册/预分析计划、附录与线上补充材料，核对是否存在未披露结果。",
            "(Andrews & Kasy, 2019)",
        ),
    ]

    # Inject a lightweight “anchors” line where possible
    anchors_note = first_anchor_or_na()
    for i, (cat, claim, why, next_steps, cite) in enumerate(cards, start=1):
        lines.append(f"#### {i}. {cat}")
        lines.append(f"*   **Category:** {cat}")
        lines.append(f"*   **Claim:** {claim}")
        lines.append(f"*   **Why it matters:** {why} {cite}")
        lines.append(f"*   **Page & Anchors:** {anchors_note}")
        lines.append(f"*   **下一步复核建议:** {next_steps}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("### 三、机器启发式摘要")
    lines.append("")
    lines.append(f"- PDF pages: {heuristics.get('pdf_pages')}")
    lines.append(f"- Extracted text chars: {heuristics.get('extracted_text_chars')}")
    lines.append(f"- p-values found (regex): {heuristics.get('p_values_found')}")
    lines.append(f"- Near 0.05 counts (0.045–0.05 vs 0.05–0.055): {heuristics.get('near_0_05')}")
    lines.append(f"- Table mentions (full text): {heuristics.get('table_mentions_fulltext')}")
    lines.append(f"- Robust mentions (full text): {heuristics.get('robust_mentions_fulltext')}")
    lines.append("")
    if selected_pages:
        lines.append("**抽样查看的页码（已导出图片到 `pages/`）：**")
        for it in selected_pages:
            try:
                lines.append(f"- p.{it.get('page')}: {it.get('reason')}")
            except Exception:
                continue
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("### 四、局限性说明")
    lines.append("")
    lines.append("1. 本报告为“篇内风险诊断”，不是定罪；只能提示复核方向。")
    lines.append("2. 离线模式无法逐页读图抽取表格数值，p 值/临界显著等信号可能被低估。")
    lines.append("3. 若论文主要以显著性星号呈现结果，单纯文本抽取难以恢复精确 p 值分布。")
    lines.append("")

    lines.append("## References (APA, with DOI)")
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
    if not SKILL_LLM_SCRIPTS_DIR.exists():
        raise FileNotFoundError(
            f"llm-api-client skill scripts not found: {SKILL_LLM_SCRIPTS_DIR} "
            "(expected under $CODEX_HOME/skills or ~/.codex/skills)."
        )
    if str(SKILL_LLM_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SKILL_LLM_SCRIPTS_DIR))
    from config_llm import LLMConfig  # type: ignore
    from client import LLMClient  # type: ignore

    cfg = LLMConfig.resolve(timeout_s=180.0, max_retries=6)
    client = LLMClient(cfg)
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


def _call_llm_json_text(logger: LLMRunLogger, llm: Any, *, system: str, user: str, max_tokens: int) -> dict[str, Any]:
    res = _chat_text_logged(
        llm,
        logger,
        system_prompt=system,
        user_prompt=user,
        temperature=0.0,
        max_tokens=max_tokens,
        json_mode=True,
    )
    try:
        return _extract_json_obj(res.content)
    except Exception:
        # Repair once
        repair_user = (
            "下面这段文本本应是一个 JSON 对象，但格式错误或被截断。请修复为严格 JSON（仅输出 JSON，不要解释）：\n\n"
            + res.content
        )
        res2 = _chat_text_logged(
            llm,
            logger,
            system_prompt=system,
            user_prompt=repair_user,
            temperature=0.0,
            max_tokens=max_tokens,
            json_mode=True,
            notes="repair_json",
        )
        return _extract_json_obj(res2.content)


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
    res = _chat_image_logged(
        llm,
        logger,
        system_prompt=system,
        user_prompt=user,
        image_bytes=image_bytes,
        image_path=image_path,
        temperature=0.0,
        max_tokens=max_tokens,
        json_mode=True,
    )
    try:
        return _extract_json_obj(res.content)
    except Exception:
        # Repair once via text
        repair_user = (
            "下面这段文本本应是一个 JSON 对象，但格式错误或被截断。请修复为严格 JSON（仅输出 JSON，不要解释）：\n\n"
            + (res.content or "")
        )
        res2 = _chat_text_logged(
            llm,
            logger,
            system_prompt=system,
            user_prompt=repair_user,
            temperature=0.0,
            max_tokens=max_tokens,
            json_mode=True,
            notes="repair_json_from_image",
        )
        return _extract_json_obj(res2.content)


def _build_method_citation_guide(apa_refs: list[str]) -> str:
    if not apa_refs:
        return ""
    # Provide a short mapping for stable in-text citations
    guide_lines = [
        "可用的方法参考文献（用于作者-年份引用；不要发明不在列表里的新参考文献）：",
    ]
    for r in apa_refs:
        guide_lines.append(f"- {r}")
    guide_lines.append("")
    guide_lines.append("引用规则建议（按常见用法）：")
    guide_lines.append("- 阈值附近堆积/缺口（caliper/阈值诊断）：(Brodeur et al., 2016; Brodeur et al., 2020)")
    guide_lines.append("- p-curve/形状约束与上界：(Elliott et al., 2022)")
    guide_lines.append("- 多重检验/FDR/FWER/更高 t 门槛：(Harvey et al., 2015; Harvey, 2017)")
    guide_lines.append("- 发表偏倚识别与校正：(Andrews & Kasy, 2019)")
    guide_lines.append("- 规格搜索/稳健性选择与推断有效性：(Leamer, 1978; Leamer, 1983)")
    return "\n".join(guide_lines)


def _system_prompt_agent() -> str:
    return (
        "你是一位非常严谨的计量经济学/统计学审计专家，任务是对单篇论文做“篇内 p-hacking 风险诊断”。\n"
        "你必须结合：1) PDF 文本抽取（可能有错/缺失）与 2) 单页 PDF 图像（更可靠）。\n"
        "你不能臆测论文未出现的实证结果；对每条可疑点要给出页面证据（页码+可识别的短语/表名）。\n"
        "输出严格 JSON（一个对象），不要输出 Markdown code fence（不要 ```json）。\n"
        "请尽量节省 token：列表不要过长，优先输出高价值证据。"
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
        f"论文标题（可能不准）：{title_hint}\n"
        f"你要从 1..{len(page_features)} 页里选出最多 {max_pages} 页去看“单页图片”，以最大性价比发现 p-hacking 风险信号。\n"
        "必须覆盖：至少 1 页主结果（主回归表），至少 1 页稳健性/附录（若存在）。\n"
        "排除：纯参考文献页（全是引用列表）通常不选。\n\n"
        "每页摘要（含启发式分数与开头几行文本）：\n"
        + json.dumps(rows, ensure_ascii=False, indent=2)
        + "\n\n输出严格 JSON，schema：\n"
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
        f"论文：{paper_title}\n"
        f"页码：{page_1based}\n\n"
        + "任务：你将看到该页的图片 +（可能有噪声的）文本抽取。请优先以图片为准。\n"
        "从该页提取与 p-hacking/选择性报告/多重检验/规格搜索相关的“高价值证据”。\n"
        "要求：\n"
        "- 输出必须是严格 JSON（不要 ```json），字段尽量短。\n"
        "- signals 最多 4 条；每条 anchors 最多 3 条；每条 evidence 尽量短（<= 25 个英文词或 2 句中文）。\n"
        "- borderline_results 最多 3 条。\n"
        "- anchors 必须是页面上可指认的短语/表名/列名/注释句。\n\n"
        "该页文本抽取（仅供辅助，可能缺失/错位）：\n"
        + extracted_text
        + "\n\n输出严格 JSON，schema：\n"
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
        f"论文：{paper_title}\n"
        f"页码：{page_1based}\n\n"
        "上一轮输出 JSON 失败。请你重新输出“最短”的严格 JSON（不要 ```json），只填 schema 字段。\n"
        "signals 最多 4 条；anchors 最多 6 条；每条尽量短。\n\n"
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
        f"你要基于以下材料，为论文做一份“篇内 p-hacking 风险诊断报告”（输出 Markdown，不要 JSON）。论文标题：{paper_title}\n\n"
        "材料（JSON 只是给你阅读，不要原样粘贴全部）：\n"
        "1) 论文元信息（来自 PDF metadata/第一页）：\n"
        + json.dumps(paper_meta, ensure_ascii=False, indent=2)
        + "\n\n"
        "2) 机器启发式统计（来自全文文本抽取；可能低估表格里的数字）：\n"
        + json.dumps(heuristics, ensure_ascii=False, indent=2)
        + "\n\n"
        "3) 你选择查看的页码（以及选择理由）：\n"
        + json.dumps(selected_pages, ensure_ascii=False, indent=2)
        + "\n\n"
        "4) 逐页证据抽取（每条都带 anchors/页码）：\n"
        + json.dumps(page_evidence, ensure_ascii=False, indent=2)
        + "\n\n"
        "输出要求（Markdown）：\n"
        "- 必须给出 `Risk score (0–100)` 与 `Risk level`。\n"
        "- 必须包含 5–12 张 `Evidence cards`：每张卡片写清楚 category、claim、why it matters、页码、anchors、下一步复核建议；并在句末给出作者-年份引用。\n"
        "- 必须明确局限性：这是“风险诊断”不是定罪；特别说明篇内证据的弱点。\n"
        "- 最后给出 `References (APA, with DOI)` 列表：必须只使用给定列表（可以原样复制）；禁止添加未在列表里的参考文献。\n"
        "- 把“多重检验/阈值附近堆积/规格搜索/p-curve/发表偏倚”等模块都覆盖到（即使结论是未发现/数据不足）。\n"
        "- 文内作者-年份引用也必须来自给定列表；如果需要通用的方法性引用，选列表里最接近的那篇。\n"
        "- 证据必须引用页面与 anchors（来自 page_evidence）。\n\n"
        "可用方法参考文献（APA，供引用/列表）：\n"
        + "\n".join(f"- {r}" for r in apa_refs)
        + "\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Multimodal within-paper p-hacking risk diagnostic agent (no framework).")
    ap.add_argument("--pdf", required=True, help="PDF path to diagnose.")
    ap.add_argument("--out-dir", default="reports/p_hacking", help="Base output directory.")
    ap.add_argument("--method-md", default="p_hacking_agent_methodology.md", help="Method summary markdown for citations.")
    ap.add_argument("--max-image-pages", type=int, default=8, help="Max pages to analyze with page images.")
    ap.add_argument("--force", action="store_true", help="Re-run even if cached outputs exist.")
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
        try:
            cfg, llm = _load_llm_client()
        except Exception as e:
            llm_disabled_reason = f"LLM init failed: {type(e).__name__}: {e}"

    base_url = str(getattr(cfg, "base_url", "") or os.environ.get("SKILL_LLM_BASE_URL") or "")
    model = str(getattr(cfg, "model", "") or os.environ.get("SKILL_LLM_MODEL") or "")
    api_key = str(getattr(cfg, "api_key", "") or os.environ.get("SKILL_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "")
    api_key_masked = re.sub(r".(?=.{4})", "*", api_key) if api_key else ""
    if llm is not None:
        print(f"[{_now_iso()}] LLM: base_url={base_url} model={model} api_key={api_key_masked}")
    else:
        print(f"[{_now_iso()}] LLM disabled: {llm_disabled_reason or 'unknown'}")
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
                "请从这张论文第一页图片中抽取尽可能准确的书目信息。输出严格 JSON：\n"
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
    final_system = system + "\n\n最终输出必须是 Markdown 文本，不要 JSON，不要代码块围栏。"
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
            res = _chat_text_logged(
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

    # Write outputs (JSON 由我们组装，避免模型 JSON 格式不稳定)
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
        final_md_path.write_text("# p-hacking 风险诊断报告\n\n（LLM 返回为空）\n", encoding="utf-8")

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
