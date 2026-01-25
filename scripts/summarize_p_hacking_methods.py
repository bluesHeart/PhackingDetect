#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _default_llm_api_client_scripts_dir() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home).expanduser() / "skills" / "llm-api-client" / "scripts"
    return Path.home() / ".codex" / "skills" / "llm-api-client" / "scripts"


SKILL_LLM_SCRIPTS_DIR = _default_llm_api_client_scripts_dir()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _read_zip_markdown(zip_path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(zip_path, "r") as z:
        md_files = [n for n in z.namelist() if n.lower().endswith(".md")]
        if not md_files:
            raise FileNotFoundError(f"No .md found in {zip_path}")
        md_name = md_files[0]
        raw = z.read(md_name)
    return md_name, raw.decode("utf-8", errors="replace")


def _paper_id_from_zip_name(name: str) -> str:
    # Strip trailing timestamp like _2026-01-23-19_51_02.zip
    m = re.match(r"^(?P<id>.+)_\d{4}-\d{2}-\d{2}-\d{2}_\d{2}_\d{2}\.zip$", name)
    return (m.group("id") if m else name.rsplit(".", 1)[0]).strip()


def _chunk_text(text: str, *, max_chars: int, overlap: int) -> list[str]:
    if max_chars <= 0:
        return [text]
    if overlap < 0:
        overlap = 0
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def _extract_json_obj(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty response")
    # Fast path
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Attempt to locate first JSON object in free-form text
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in response")
    candidate = text[start : end + 1]
    obj = json.loads(candidate)
    if not isinstance(obj, dict):
        raise ValueError("Top-level JSON is not an object")
    return obj


def _norm_method_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def _merge_unique_list(dst: list[str], src: list[str]) -> list[str]:
    seen = {x.strip() for x in dst if isinstance(x, str) and x.strip()}
    for item in src:
        if not isinstance(item, str):
            continue
        v = item.strip()
        if not v or v in seen:
            continue
        dst.append(v)
        seen.add(v)
    return dst


def _merge_methods(methods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for m in methods:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name") or "").strip()
        if not name:
            continue
        key = _norm_method_name(name)
        if key not in merged:
            merged[key] = dict(m)
            merged[key]["name"] = name
            continue
        dst = merged[key]
        # Merge string fields by appending distinct snippets
        for field in ["category", "what_it_detects", "core_statistic_or_test", "paper_specific_notes"]:
            v = m.get(field)
            if not isinstance(v, str) or not v.strip():
                continue
            if not isinstance(dst.get(field), str) or not str(dst.get(field)).strip():
                dst[field] = v.strip()
            elif v.strip() not in str(dst[field]):
                dst[field] = str(dst[field]).rstrip() + "\n" + v.strip()

        # Merge list-ish fields
        for field in [
            "step_by_step",
            "data_requirements",
            "assumptions",
            "limitations",
            "implementation_notes",
            "citations",
            "evidence_anchors",
        ]:
            src_list = m.get(field)
            if not isinstance(src_list, list):
                continue
            if not isinstance(dst.get(field), list):
                dst[field] = []
            _merge_unique_list(dst[field], [str(x) for x in src_list if x is not None])
    return list(merged.values())


def _first_nonempty_str(*vals: Any) -> str | None:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _mask(s: str) -> str:
    if not s:
        return ""
    if len(s) <= 8:
        return "*" * len(s)
    return "*" * (len(s) - 4) + s[-4:]


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


def _build_system_prompt() -> str:
    return (
        "你是一位非常严谨的计量经济学/统计学研究者，专注于识别 p-hacking、选择性报告、"
        "发表偏倚与多重检验/规格搜索带来的虚假显著性。\n"
        "只能基于我提供的论文 Markdown 文本（以及同一提示内的 header 摘录）提炼方法；"
        "不要编造论文未出现的结论或参考文献信息。\n"
        "输出必须是严格 JSON（一个对象），不要包含任何额外文字或 Markdown 代码块（不要输出 ```json）。\n"
        "为避免输出被截断：methods 最多 8 个；每个数组字段最多 6 条；单条字符串尽量短（<= 2 句）。"
    )


def _build_user_prompt(*, paper_id: str, md_name: str, header: str, chunk: str, chunk_i: int, chunk_n: int) -> str:
    schema = {
        "paper": {
            "title": "string|null",
            "authors": ["string"],
            "year": "number|null",
            "venue_or_series": "string|null",
            "working_paper_number": "string|null",
            "source_url": "string|null",
            "notes": "string|null",
        },
        "in_text_citation": "string|null",
        "apa_reference": "string|null",
        "methods": [
            {
                "name": "string",
                "category": "string|null",
                "what_it_detects": "string|null",
                "core_statistic_or_test": "string|null",
                "step_by_step": ["string"],
                "data_requirements": ["string"],
                "assumptions": ["string"],
                "limitations": ["string"],
                "implementation_notes": ["string"],
                "paper_specific_notes": "string|null",
                "citations": ["string"],
                "evidence_anchors": ["string"],
            }
        ],
        "key_takeaways": ["string"],
        "keywords": ["string"],
    }

    return (
        f"论文ID: {paper_id}\n"
        f"来源文件: {md_name}\n"
        f"分块: {chunk_i}/{chunk_n}\n\n"
        "任务：\n"
        "1) 从 header + 当前 chunk 中提取与“识别 p-hacking/选择性报告/发表偏倚/多重检验/规格搜索导致的假阳性”直接相关的、可操作的统计检测方法/诊断思路。\n"
        "2) 每个方法尽量给出可实现的步骤、需要的数据、关键统计量/检验、典型阈值或对照、局限性。\n"
        "3) 尽量抽取该论文的作者、年份、标题，并给出作者-年份的 in-text citation（如：\"Elliott, Kudrin, & Wüthrich (2021)\" 或 \"Brodeur et al. (2013)\"），以及 APA 参考文献条目（若文本里缺失必要字段，写 null 并在 notes 说明缺失）。\n"
        "4) citations 字段里放作者-年份引用（例如 \"(Brodeur et al., 2013)\"）。\n"
        "5) 严禁输出 Markdown code fence；只输出 JSON。\n"
        "6) 为避免截断：methods 最多 8 个；每个数组字段最多 6 条；每条尽量短。\n\n"
        "必须严格按下列 JSON schema 输出（字段可为 null 或空数组，但必须存在顶层键）：\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
        + "\n\n---\nHEADER 摘录（来自文档开头，可能包含题名/作者/日期/工作论文号/链接）：\n"
        + header
        + "\n\n---\nCHUNK 文本：\n"
        + chunk
    )


def _build_user_prompt_compact(*, paper_id: str, md_name: str, header: str, chunk: str, chunk_i: int, chunk_n: int) -> str:
    compact_schema = {
        "paper": {
            "title": "string|null",
            "authors": ["string"],
            "year": "number|null",
            "venue_or_series": "string|null",
            "source_url": "string|null",
        },
        "in_text_citation": "string|null",
        "apa_reference": "string|null",
        "methods": [
            {
                "name": "string",
                "category": "string|null",
                "what_it_detects": "string|null",
                "core_statistic_or_test": "string|null",
            }
        ],
        "key_takeaways": ["string"],
        "keywords": ["string"],
    }

    return (
        f"论文ID: {paper_id}\n"
        f"来源文件: {md_name}\n"
        f"分块: {chunk_i}/{chunk_n}\n\n"
        "上一轮输出可能不是合法 JSON。请重新输出：只输出严格 JSON（不要 ```json），并严格匹配 schema。\n"
        "只保留最关键的方法点，methods 最多 6 个；每条字段尽量短。\n\n"
        "JSON schema:\n"
        + json.dumps(compact_schema, ensure_ascii=False, indent=2)
        + "\n\n---\nHEADER:\n"
        + header
        + "\n\n---\nCHUNK:\n"
        + chunk
    )


def _build_user_prompt_ultra_compact(
    *, paper_id: str, md_name: str, header: str, chunk: str, chunk_i: int, chunk_n: int
) -> str:
    ultra_schema = {
        "paper": {"title": "string|null", "authors": ["string"], "year": "number|null"},
        "in_text_citation": "string|null",
        "apa_reference": "string|null",
        "methods": ["string"],
        "key_takeaways": ["string"],
        "keywords": ["string"],
    }

    return (
        f"论文ID: {paper_id}\n"
        f"来源文件: {md_name}\n"
        f"分块: {chunk_i}/{chunk_n}\n\n"
        "请输出最短、最稳健的 JSON（不要 ```json），严格匹配 schema。\n"
        "methods 只列方法/检验名称（字符串），最多 8 条；key_takeaways 最多 5 条。\n\n"
        "JSON schema:\n"
        + json.dumps(ultra_schema, ensure_ascii=False, indent=2)
        + "\n\n---\nHEADER:\n"
        + header
        + "\n\n---\nCHUNK:\n"
        + chunk
    )


def _build_user_prompt_minimal(*, paper_id: str, md_name: str, header: str, chunk: str, chunk_i: int, chunk_n: int) -> str:
    minimal_schema = {
        "paper": {"title": "string|null", "authors": ["string"], "year": "number|null"},
        "methods": ["string"],
    }
    return (
        f"论文ID: {paper_id}\n"
        f"来源文件: {md_name}\n"
        f"分块: {chunk_i}/{chunk_n}\n\n"
        "请输出最短 JSON（不要 ```json），严格匹配 schema。不要输出任何其他文字。\n"
        "methods 只列方法/检验名称（字符串），最多 8 条。\n\n"
        "JSON schema:\n"
        + json.dumps(minimal_schema, ensure_ascii=False, indent=2)
        + "\n\n---\nHEADER:\n"
        + header
        + "\n\n---\nCHUNK:\n"
        + chunk
    )


@dataclass
class PaperSummary:
    paper_id: str
    paper: dict[str, Any]
    in_text_citation: str | None
    apa_reference: str | None
    methods: list[dict[str, Any]]
    key_takeaways: list[str]
    keywords: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "paper": self.paper,
            "in_text_citation": self.in_text_citation,
            "apa_reference": self.apa_reference,
            "methods": self.methods,
            "key_takeaways": self.key_takeaways,
            "keywords": self.keywords,
        }


def _coerce_chunk_obj(obj: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize model outputs into our expected schema. Some providers may return
    a flatter structure or methods as strings.
    """
    out: dict[str, Any] = {
        "paper": {
            "title": None,
            "authors": [],
            "year": None,
            "venue_or_series": None,
            "working_paper_number": None,
            "source_url": None,
            "notes": None,
        },
        "in_text_citation": None,
        "apa_reference": None,
        "methods": [],
        "key_takeaways": [],
        "keywords": [],
    }

    # Paper meta may be nested or flat
    p = obj.get("paper")
    if isinstance(p, str) and p.strip():
        out["paper"]["title"] = p.strip()
    elif isinstance(p, dict):
        for k in out["paper"].keys():
            if k == "authors":
                continue
            if isinstance(p.get(k), str) and p.get(k).strip():
                out["paper"][k] = p.get(k).strip()
            elif k == "year" and isinstance(p.get("year"), (int, float)):
                out["paper"]["year"] = int(p["year"])
        authors = p.get("authors")
        if isinstance(authors, list):
            _merge_unique_list(out["paper"]["authors"], [str(a) for a in authors if str(a).strip()])
        elif isinstance(authors, str) and authors.strip():
            _merge_unique_list(out["paper"]["authors"], [a.strip() for a in re.split(r"[;,/]| and | & ", authors) if a.strip()])

    # Flat fallbacks
    for k in ["title", "venue_or_series", "working_paper_number", "source_url", "notes"]:
        if out["paper"].get(k) in (None, "", []):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                out["paper"][k] = v.strip()
    if out["paper"].get("year") is None:
        y = obj.get("year")
        if isinstance(y, (int, float)):
            out["paper"]["year"] = int(y)
        elif isinstance(y, str):
            m = re.search(r"(19|20)\d{2}", y)
            if m:
                out["paper"]["year"] = int(m.group(0))

    if not out["paper"]["authors"]:
        a = obj.get("authors")
        if isinstance(a, list):
            _merge_unique_list(out["paper"]["authors"], [str(x) for x in a if str(x).strip()])
        elif isinstance(a, str) and a.strip():
            _merge_unique_list(out["paper"]["authors"], [x.strip() for x in re.split(r"[;,/]| and | & ", a) if x.strip()])

    # Citations + references
    for k in ["in_text_citation", "citation", "cite"]:
        v = obj.get(k)
        if isinstance(v, str) and v.strip() and not out["in_text_citation"]:
            out["in_text_citation"] = v.strip()
    for k in ["apa_reference", "apa", "reference_apa", "reference"]:
        v = obj.get(k)
        if isinstance(v, str) and v.strip() and not out["apa_reference"]:
            out["apa_reference"] = v.strip()

    # Methods: allow list[str] or list[dict]
    methods = obj.get("methods")
    if methods is None:
        for alt in ["p_hacking_detection_methods", "detection_methods", "p_hacking_methods"]:
            if isinstance(obj.get(alt), list):
                methods = obj[alt]
                break
    if isinstance(methods, list):
        for m in methods:
            if isinstance(m, str) and m.strip():
                out["methods"].append(
                    {
                        "name": m.strip(),
                        "category": None,
                        "what_it_detects": None,
                        "core_statistic_or_test": None,
                        "step_by_step": [],
                        "data_requirements": [],
                        "assumptions": [],
                        "limitations": [],
                        "implementation_notes": [],
                        "paper_specific_notes": None,
                        "citations": [],
                        "evidence_anchors": [],
                    }
                )
            elif isinstance(m, dict):
                mm = {
                    "name": str(m.get("name") or "").strip(),
                    "category": m.get("category"),
                    "what_it_detects": m.get("what_it_detects"),
                    "core_statistic_or_test": m.get("core_statistic_or_test"),
                    "step_by_step": m.get("step_by_step") if isinstance(m.get("step_by_step"), list) else [],
                    "data_requirements": m.get("data_requirements") if isinstance(m.get("data_requirements"), list) else [],
                    "assumptions": m.get("assumptions") if isinstance(m.get("assumptions"), list) else [],
                    "limitations": m.get("limitations") if isinstance(m.get("limitations"), list) else [],
                    "implementation_notes": m.get("implementation_notes")
                    if isinstance(m.get("implementation_notes"), list)
                    else [],
                    "paper_specific_notes": m.get("paper_specific_notes"),
                    "citations": m.get("citations") if isinstance(m.get("citations"), list) else [],
                    "evidence_anchors": m.get("evidence_anchors") if isinstance(m.get("evidence_anchors"), list) else [],
                }
                if mm["name"]:
                    out["methods"].append(mm)

    # Takeaways / keywords
    for k in ["key_takeaways", "takeaways", "summary_points"]:
        v = obj.get(k)
        if isinstance(v, list):
            _merge_unique_list(out["key_takeaways"], [str(x) for x in v if str(x).strip()])
            break
    for k in ["keywords", "tags"]:
        v = obj.get(k)
        if isinstance(v, list):
            _merge_unique_list(out["keywords"], [str(x) for x in v if str(x).strip()])
            break

    return out


def _derive_citation_paren(*, paper_id: str, paper_meta: dict[str, Any], in_text_citation: str | None) -> str:
    year = paper_meta.get("year")
    authors = paper_meta.get("authors") or []
    if isinstance(year, int) and isinstance(authors, list) and authors:
        last_names: list[str] = []
        for a in authors:
            s = str(a).strip()
            if not s:
                continue
            parts = re.split(r"\s+", s)
            last = parts[-1].strip(",.")
            if last and last not in last_names:
                last_names.append(last)
        if last_names:
            if len(last_names) == 1:
                return f"({last_names[0]}, {year})"
            if len(last_names) == 2:
                return f"({last_names[0]} & {last_names[1]}, {year})"
            return f"({last_names[0]} et al., {year})"

    # Fallback: attempt to extract "(YYYY)" from in_text_citation, otherwise use paper_id
    if isinstance(in_text_citation, str) and in_text_citation.strip():
        m = re.search(r"(19|20)\d{2}", in_text_citation)
        if m:
            # Try to keep something like "X et al., YYYY"
            left = in_text_citation.strip()
            left = re.sub(r"\([^)]*\)", "", left).strip()
            if left:
                return f"({left}, {m.group(0)})"
    return f"({paper_id})"


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize p-hacking detection methods from papers_md zips via llm-api-client.")
    ap.add_argument("--papers-md-dir", default="papers_md", help="Directory containing *.zip with a single .md per zip.")
    ap.add_argument("--out-md", default="p_hacking_detection_methods.md", help="Output markdown path.")
    ap.add_argument("--out-json", default="papers_md/paper_summaries.json", help="Output JSON with per-paper summaries.")
    ap.add_argument("--cache-dir", default="papers_md/_llm_cache", help="Cache dir for per-chunk LLM JSON.")
    ap.add_argument("--logs-dir", default=None, help="LLM logs directory (default: <papers_md_dir>/llm_logs).")
    ap.add_argument("--max-chars-per-chunk", type=int, default=60000)
    ap.add_argument("--overlap", type=int, default=1500)
    ap.add_argument("--max-tokens", type=int, default=2500, help="Max completion tokens per chunk call.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--force", action="store_true", help="Recompute even if cache exists.")
    args = ap.parse_args()

    papers_md_dir = Path(args.papers_md_dir)
    if not papers_md_dir.exists():
        raise FileNotFoundError(papers_md_dir)

    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Import skill client
    if not SKILL_LLM_SCRIPTS_DIR.exists():
        raise FileNotFoundError(
            f"llm-api-client skill scripts not found: {SKILL_LLM_SCRIPTS_DIR} "
            "(expected under $CODEX_HOME/skills or ~/.codex/skills)."
        )
    if str(SKILL_LLM_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SKILL_LLM_SCRIPTS_DIR))
    from config_llm import LLMConfig  # type: ignore
    from client import LLMClient  # type: ignore

    cfg = LLMConfig.resolve()
    api_key_masked = _mask(cfg.api_key)
    print(f"[{_now_iso()}] LLM config: base_url={cfg.base_url} model={cfg.model} api_key={api_key_masked}")
    llm = LLMClient(cfg)

    logs_base = Path(args.logs_dir) if args.logs_dir else (papers_md_dir / "llm_logs")
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(2)}"
    logs_dir = logs_base / run_id
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = LLMRunLogger(logs_dir, model=str(getattr(cfg, "model", "")), base_url=str(getattr(cfg, "base_url", "")))
    (logs_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "started_at": _now_iso(),
                "papers_md_dir": str(papers_md_dir),
                "out_md": str(out_md),
                "out_json": str(out_json),
                "cache_dir": str(cache_dir),
                "logs_dir": str(logs_dir),
                "llm_base_url": getattr(cfg, "base_url", None),
                "llm_model": getattr(cfg, "model", None),
                "llm_api_key_masked": api_key_masked,
                "args": vars(args),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[{_now_iso()}] LLM logs: {logs_dir}")

    system_prompt = _build_system_prompt()

    zip_files = sorted(papers_md_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No .zip files found in {papers_md_dir}")

    paper_summaries: list[PaperSummary] = []

    for zi, zip_path in enumerate(zip_files, start=1):
        paper_id = _paper_id_from_zip_name(zip_path.name)
        md_name, text = _read_zip_markdown(zip_path)
        header = text[:8000]
        chunks = _chunk_text(text, max_chars=args.max_chars_per_chunk, overlap=args.overlap)

        print(f"[{_now_iso()}] ({zi}/{len(zip_files)}) {paper_id}: {len(text):,} chars -> {len(chunks)} chunks")

        chunk_objs: list[dict[str, Any]] = []
        for ci, chunk in enumerate(chunks, start=1):
            cache_path = cache_dir / f"{paper_id}__chunk{ci:03d}_of_{len(chunks):03d}.json"
            if cache_path.exists() and not args.force:
                try:
                    chunk_objs.append(json.loads(cache_path.read_text(encoding="utf-8")))
                    continue
                except Exception:
                    pass

            user_prompt = _build_user_prompt_compact(
                paper_id=paper_id,
                md_name=md_name,
                header=header,
                chunk=chunk,
                chunk_i=ci,
                chunk_n=len(chunks),
            )

            res = _chat_text_logged(
                llm,
                logger,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                json_mode=True,
                notes=f"{paper_id} chunk {ci}/{len(chunks)} prompt=compact",
            )

            raw = res.content
            try:
                obj0 = _extract_json_obj(raw)
            except Exception:
                raw_path = cache_path.with_suffix(".raw.txt")
                raw_path.write_text(raw, encoding="utf-8")
                # Retry once with an ultra-compact prompt to reduce truncation/format drift.
                user_prompt2 = _build_user_prompt_ultra_compact(
                    paper_id=paper_id,
                    md_name=md_name,
                    header=header,
                    chunk=chunk,
                    chunk_i=ci,
                    chunk_n=len(chunks),
                )
                res2 = _chat_text_logged(
                    llm,
                    logger,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt2,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    json_mode=True,
                    notes=f"{paper_id} chunk {ci}/{len(chunks)} prompt=ultra_compact",
                )
                raw2 = res2.content
                raw_path2 = cache_path.with_suffix(".raw2.txt")
                raw_path2.write_text(raw2, encoding="utf-8")
                try:
                    obj0 = _extract_json_obj(raw2)
                except Exception:
                    # Last resort: minimal schema (names only)
                    user_prompt3 = _build_user_prompt_minimal(
                        paper_id=paper_id,
                        md_name=md_name,
                        header=header,
                        chunk=chunk,
                        chunk_i=ci,
                        chunk_n=len(chunks),
                    )
                    res3 = _chat_text_logged(
                        llm,
                        logger,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt3,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        json_mode=True,
                        notes=f"{paper_id} chunk {ci}/{len(chunks)} prompt=minimal",
                    )
                    raw3 = res3.content
                    raw_path3 = cache_path.with_suffix(".raw3.txt")
                    raw_path3.write_text(raw3, encoding="utf-8")
                    obj0 = _extract_json_obj(raw3)

            obj = _coerce_chunk_obj(obj0)
            cache_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            chunk_objs.append(obj)
            time.sleep(0.2)  # gentle pacing

        # Merge chunk objects
        paper_meta: dict[str, Any] = {
            "title": None,
            "authors": [],
            "year": None,
            "venue_or_series": None,
            "working_paper_number": None,
            "source_url": None,
            "notes": None,
        }
        in_text_citation: str | None = None
        apa_reference: str | None = None
        methods_all: list[dict[str, Any]] = []
        takeaways: list[str] = []
        keywords: list[str] = []

        for obj in chunk_objs:
            if not isinstance(obj, dict):
                continue
            p = obj.get("paper")
            if isinstance(p, dict):
                paper_meta["title"] = _first_nonempty_str(paper_meta.get("title"), p.get("title"))
                paper_meta["venue_or_series"] = _first_nonempty_str(paper_meta.get("venue_or_series"), p.get("venue_or_series"))
                paper_meta["working_paper_number"] = _first_nonempty_str(
                    paper_meta.get("working_paper_number"), p.get("working_paper_number")
                )
                paper_meta["source_url"] = _first_nonempty_str(paper_meta.get("source_url"), p.get("source_url"))
                paper_meta["notes"] = _first_nonempty_str(paper_meta.get("notes"), p.get("notes"))
                if paper_meta.get("year") is None and isinstance(p.get("year"), (int, float)):
                    paper_meta["year"] = int(p["year"])
                authors = p.get("authors")
                if isinstance(authors, list):
                    paper_meta["authors"] = _merge_unique_list(
                        paper_meta.get("authors", []),
                        [str(a) for a in authors if isinstance(a, (str, int, float))],
                    )

            in_text_citation = _first_nonempty_str(in_text_citation, obj.get("in_text_citation"))
            apa_reference = _first_nonempty_str(apa_reference, obj.get("apa_reference"))

            m = obj.get("methods")
            if isinstance(m, list):
                for mi in m:
                    if isinstance(mi, dict):
                        methods_all.append(mi)
                    elif isinstance(mi, str) and mi.strip():
                        methods_all.append({"name": mi.strip()})

            kt = obj.get("key_takeaways")
            if isinstance(kt, list):
                _merge_unique_list(takeaways, [str(x) for x in kt if x is not None])

            kw = obj.get("keywords")
            if isinstance(kw, list):
                _merge_unique_list(keywords, [str(x) for x in kw if x is not None])

        methods_merged = _merge_methods(methods_all)
        cite_paren = _derive_citation_paren(paper_id=paper_id, paper_meta=paper_meta, in_text_citation=in_text_citation)
        for m in methods_merged:
            if not isinstance(m, dict):
                continue
            m.setdefault("citations", [])
            if isinstance(m["citations"], list):
                _merge_unique_list(m["citations"], [cite_paren])

        paper_summaries.append(
            PaperSummary(
                paper_id=paper_id,
                paper=paper_meta,
                in_text_citation=in_text_citation,
                apa_reference=apa_reference,
                methods=methods_merged,
                key_takeaways=takeaways,
                keywords=keywords,
            )
        )

    # Write per-paper JSON
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps([p.to_dict() for p in paper_summaries], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Build global method index by category
    by_category: dict[str, list[dict[str, Any]]] = {}
    for p in paper_summaries:
        for m in p.methods:
            if not isinstance(m, dict):
                continue
            name = str(m.get("name") or "").strip()
            if not name:
                continue
            cat = str(m.get("category") or "未分类").strip() or "未分类"
            m2 = dict(m)
            by_category.setdefault(cat, []).append(m2)

    # Merge methods within each category by name
    by_category_merged: dict[str, list[dict[str, Any]]] = {}
    for cat, ms in by_category.items():
        by_category_merged[cat] = _merge_methods(ms)

    # Collect APA references
    apa_refs: list[str] = []
    for p in paper_summaries:
        if p.apa_reference and p.apa_reference.strip():
            if p.apa_reference.strip() not in apa_refs:
                apa_refs.append(p.apa_reference.strip())

    # Render markdown
    lines: list[str] = []
    lines.append("# 识别 p-hacking 的方法：基于 papers_md 的逐篇精华总结")
    lines.append("")
    lines.append(f"> 生成时间：{_now_iso()}")
    lines.append("> 说明：本总结由脚本使用 OpenAI-compatible LLM API 对 `papers_md/` 内每篇论文的 Markdown 转写内容逐块阅读后提炼。仅基于文本可见信息；若文内缺失书目信息则在对应条目中标注缺失。")
    lines.append("")

    lines.append("## 方法总览（按类别）")
    lines.append("")
    for cat in sorted(by_category_merged.keys(), key=lambda s: (s == "未分类", s.lower())):
        lines.append(f"### {cat}")
        lines.append("")
        for m in sorted(by_category_merged[cat], key=lambda x: _norm_method_name(str(x.get('name') or ''))):
            name = str(m.get("name") or "").strip()
            if not name:
                continue
            cites = m.get("citations")
            cite_txt = ""
            if isinstance(cites, list) and cites:
                cite_txt = " " + "; ".join(sorted({str(c).strip() for c in cites if str(c).strip()}))
            lines.append(f"#### {name}{cite_txt}")
            lines.append("")
            for field, label in [
                ("what_it_detects", "检测目标"),
                ("core_statistic_or_test", "核心统计量/检验"),
                ("paper_specific_notes", "论文内要点"),
            ]:
                v = m.get(field)
                if isinstance(v, str) and v.strip():
                    lines.append(f"- {label}：{v.strip()}")
            for field, label in [
                ("data_requirements", "数据需求"),
                ("step_by_step", "实现步骤"),
                ("assumptions", "关键假设"),
                ("limitations", "局限性/误判风险"),
                ("implementation_notes", "工程实现提示"),
                ("evidence_anchors", "文本锚点"),
            ]:
                v = m.get(field)
                if isinstance(v, list) and [x for x in v if isinstance(x, str) and x.strip()]:
                    lines.append(f"- {label}：")
                    for item in v:
                        if isinstance(item, str) and item.strip():
                            lines.append(f"  - {item.strip()}")
            lines.append("")

    lines.append("## 逐篇精华（按文献）")
    lines.append("")
    for p in paper_summaries:
        title = _first_nonempty_str(p.paper.get("title"), p.paper_id) or p.paper_id
        cite = p.in_text_citation or p.paper_id
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"- 引用：{cite}")
        authors = p.paper.get("authors")
        if isinstance(authors, list) and authors:
            lines.append(f"- 作者：{', '.join([str(a).strip() for a in authors if str(a).strip()])}")
        year = p.paper.get("year")
        if isinstance(year, int):
            lines.append(f"- 年份：{year}")
        venue = p.paper.get("venue_or_series")
        if isinstance(venue, str) and venue.strip():
            lines.append(f"- 载体/系列：{venue.strip()}")
        wp = p.paper.get("working_paper_number")
        if isinstance(wp, str) and wp.strip():
            lines.append(f"- 编号：{wp.strip()}")
        url = p.paper.get("source_url")
        if isinstance(url, str) and url.strip():
            lines.append(f"- 链接：{url.strip()}")
        notes = p.paper.get("notes")
        if isinstance(notes, str) and notes.strip():
            lines.append(f"- 备注：{notes.strip()}")
        if p.keywords:
            lines.append(f"- 关键词：{', '.join(p.keywords)}")
        lines.append("")
        if p.key_takeaways:
            lines.append("**核心要点**")
            for t in p.key_takeaways:
                lines.append(f"- {t}")
            lines.append("")
        if p.methods:
            lines.append("**该文献提供/强调的方法**")
            for m in sorted(p.methods, key=lambda x: _norm_method_name(str(x.get('name') or ''))):
                mn = str(m.get("name") or "").strip()
                if mn:
                    cat = str(m.get("category") or "").strip()
                    if cat:
                        lines.append(f"- {mn}（{cat}）")
                    else:
                        lines.append(f"- {mn}")
            lines.append("")

    lines.append("## 参考文献（APA）")
    lines.append("")
    if apa_refs:
        for r in apa_refs:
            lines.append(f"- {r}")
    else:
        lines.append("- （未能从文本中抽取到完整 APA 条目）")
    lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[{_now_iso()}] Wrote: {out_md}")
    print(f"[{_now_iso()}] Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
