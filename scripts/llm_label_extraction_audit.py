#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _resolve_llm_api_client_dir() -> Path:
    override = os.environ.get("LLM_API_CLIENT_DIR")
    if override and override.strip():
        p = Path(override).expanduser().resolve()
        if p.exists():
            return p

    codex_home = os.environ.get("CODEX_HOME")
    if codex_home and codex_home.strip():
        base = Path(codex_home).expanduser().resolve()
    else:
        base = Path.home() / ".codex"

    p = base / "skills" / "llm-api-client" / "scripts"
    if p.exists():
        return p

    raise SystemExit(
        "Missing llm-api-client skill scripts. Expected under "
        f"{p} (or set env LLM_API_CLIENT_DIR / CODEX_HOME)."
    )


def _extract_first_json_obj(s: str) -> dict[str, Any] | None:
    if not s:
        return None
    s2 = s.strip()

    # Strip common markdown fences.
    if s2.startswith("```"):
        s2 = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s2)
        s2 = re.sub(r"\s*```$", "", s2).strip()
    try:
        obj = json.loads(s2)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return obj[0]
        return None
    except Exception:
        pass

    # Fallback: extract first {...} block.
    i = s2.find("{")
    j = s2.rfind("}")
    if i < 0 or j < 0 or j <= i:
        return None
    try:
        obj = json.loads(s2[i : j + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _norm_paren_mode(x: Any) -> str | None:
    if not isinstance(x, str):
        return None
    v = x.strip().lower()
    if v in {"se", "stderr", "standard_error", "standard errors", "standard error"}:
        return "se"
    if v in {"t", "tstat", "t_stat", "t-stat", "t-statistic", "t statistics", "t-statistics"}:
        return "t"
    if v in {"unknown", "unk", "na", "n/a", "null"}:
        return "unknown"
    return None


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return None
        if isinstance(x, (int, float)):
            v = float(x)
            return None if v != v else v
        if isinstance(x, str) and x.strip():
            t = x.strip()
            # Remove thousands separators.
            t = t.replace(",", "")
            # Remove wrapping parentheses: "(-1.23)" -> "-1.23"
            if t.startswith("(") and t.endswith(")"):
                t = t[1:-1].strip()
            # Drop percent sign and trailing stars.
            t = t.rstrip("%").strip()
            t = re.sub(r"\*+$", "", t).strip()
            v = float(t)
            return None if v != v else v
    except Exception:
        return None
    return None


def _safe_int(x: Any) -> int | None:
    try:
        if x is None:
            return None
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


def _build_prompt() -> tuple[str, str]:
    system = (
        "You are a meticulous human annotator extracting numbers from an academic regression-table snippet image.\n"
        "Rules:\n"
        "- Do NOT guess. If you cannot read a value with high confidence, output null.\n"
        "- Output MUST be a single JSON object and nothing else.\n"
        "- Keep signs and decimal places as printed.\n"
        "- If the coefficient has significance stars (e.g., 0.12***), return coef=0.12 and stars=3.\n"
    )
    user = (
        "Task: From the image snippet, transcribe:\n"
        "1) coef: the coefficient value\n"
        "2) paren: the numeric value shown in parentheses near/below the coefficient\n"
        "3) paren_mode: \"se\" if the parentheses clearly indicate standard errors, \"t\" if they clearly indicate t-statistics, "
        "otherwise \"unknown\".\n"
        "4) stars: integer 0..3 if stars are present next to the coefficient, else 0. If unclear, null.\n"
        "5) notes: brief string if anything is ambiguous, else null.\n\n"
        "Return a SINGLE JSON OBJECT (not an array) with exactly these keys:\n"
        "{\n"
        "  \"coef\": number|null,\n"
        "  \"paren\": number|null,\n"
        "  \"paren_mode\": \"se\"|\"t\"|\"unknown\",\n"
        "  \"stars\": 0|1|2|3|null,\n"
        "  \"notes\": string|null\n"
        "}\n"
    )
    return system, user


def _label_one_image(*, client: Any, image_path: Path, max_attempts: int = 2) -> tuple[dict[str, Any] | None, str]:
    system, user = _build_prompt()

    raw_text = ""
    for attempt in range(1, max_attempts + 1):
        result = client.chat_with_image(
            system_prompt=system,
            user_prompt=user,
            image_bytes=image_path.read_bytes(),
            image_mime="image/png",
            temperature=0.0,
            max_tokens=2048,
            json_mode=True,
        )
        raw_text = (result.content or "").strip()
        obj = _extract_first_json_obj(raw_text)
        if isinstance(obj, dict):
            return obj, raw_text

        # Second attempt: ask to output only JSON.
        user = (
            "Your previous output was not valid JSON.\n"
            "Return ONLY a single JSON object, no markdown, no explanations.\n\n"
            "Schema:\n"
            "{\n"
            "  \"coef\": number|null,\n"
            "  \"paren\": number|null,\n"
            "  \"paren_mode\": \"se\"|\"t\"|\"unknown\",\n"
            "  \"stars\": 0|1|2|3|null,\n"
            "  \"notes\": string|null\n"
            "}\n"
        )

    return None, raw_text


def main() -> int:
    ap = argparse.ArgumentParser(description="Fill extraction audit labels using a vision-capable LLM (auto-annotation).")
    ap.add_argument("--tasks-dir", default="analysis_800_v2/extraction_tasks", help="Directory with task JSON files.")
    ap.add_argument("--out-labels-dir", default="analysis_800_v2/extraction_labels_llm", help="Output labels directory.")
    ap.add_argument("--log-jsonl", default="analysis_800_v2/extraction_labels_llm_log.jsonl", help="Write per-item logs as JSONL.")
    ap.add_argument("--max-items", type=int, default=None, help="Optional cap on number of items labeled (for smoke tests).")
    ap.add_argument("--max-papers", type=int, default=None, help="Optional cap on number of task files processed.")
    ap.add_argument("--sleep-s", type=float, default=0.0, help="Optional sleep between LLM calls.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing observed_* fields.")
    args = ap.parse_args()

    tasks_dir = Path(args.tasks_dir)
    out_labels_dir = Path(args.out_labels_dir)
    log_path = Path(args.log_jsonl)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    llm_dir = _resolve_llm_api_client_dir()
    sys.path.insert(0, str(llm_dir))
    from client import LLMClient  # type: ignore
    from config_llm import LLMConfig  # type: ignore

    client = LLMClient(LLMConfig.resolve())
    annotator = f"llm:{os.environ.get('SKILL_LLM_MODEL') or 'unknown_model'}"

    task_files = sorted(tasks_dir.glob("*.json"))
    if args.max_papers is not None:
        task_files = task_files[: max(0, int(args.max_papers))]
    if not task_files:
        raise SystemExit(f"No task files under: {tasks_dir}")

    done_items = 0
    total_items = 0
    with log_path.open("a", encoding="utf-8") as log:
        for ti, task_path in enumerate(task_files, start=1):
            try:
                task = json.loads(task_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(task, dict):
                continue
            paper_id = str(task.get("paper_id") or task_path.stem).strip()
            items = [it for it in (task.get("items") or []) if isinstance(it, dict)]
            if not paper_id or not items:
                continue

            out_label_path = out_labels_dir / f"{paper_id}.json"
            if out_label_path.exists():
                try:
                    label_obj = json.loads(out_label_path.read_text(encoding="utf-8"))
                except Exception:
                    label_obj = None
            else:
                label_obj = None

            if not isinstance(label_obj, dict):
                label_obj = {
                    "version": str(task.get("version") or "0.2"),
                    "paper_id": paper_id,
                    "annotator": annotator,
                    "completed_at": None,
                    "items": [],
                }

            label_items_by_id: dict[str, dict[str, Any]] = {}
            for li in label_obj.get("items") or []:
                if isinstance(li, dict) and str(li.get("item_id") or "").strip():
                    label_items_by_id[str(li["item_id"])] = li

            updated = False
            for it in items:
                item_id = str(it.get("item_id") or "").strip()
                if not item_id:
                    continue
                total_items += 1

                out_item = label_items_by_id.get(item_id)
                if not isinstance(out_item, dict):
                    out_item = {
                        "item_id": item_id,
                        "snippet_relpath": it.get("snippet_relpath"),
                        "coef_extracted": it.get("coef_extracted"),
                        "paren_extracted": it.get("paren_extracted"),
                        "paren_mode_assumed": it.get("paren_mode_assumed"),
                        "se_extracted": it.get("se_extracted"),
                        "t_extracted": it.get("t_extracted"),
                        "stars_extracted": it.get("stars_extracted"),
                        "observed_coef": None,
                        "observed_paren": None,
                        "observed_paren_mode": None,
                        "observed_se": None,
                        "observed_t": None,
                        "observed_stars": None,
                        "notes": None,
                    }
                    label_obj["items"].append(out_item)
                    label_items_by_id[item_id] = out_item
                    updated = True

                if not args.force:
                    if out_item.get("observed_coef") is not None or out_item.get("observed_paren") is not None:
                        done_items += 1
                        continue

                snippet_rel = str(out_item.get("snippet_relpath") or "").strip()
                if not snippet_rel:
                    continue
                image_path = Path(snippet_rel)
                if not image_path.exists():
                    # Try relative to repo root.
                    image_path = (Path.cwd() / snippet_rel).resolve()
                if not image_path.exists():
                    out_item["notes"] = f"missing_snippet:{snippet_rel}"
                    updated = True
                    continue

                parsed, raw_text = _label_one_image(client=client, image_path=image_path)
                if parsed is None:
                    out_item["notes"] = f"llm_invalid_json:{raw_text[:160]}"
                    updated = True
                else:
                    coef = _safe_float(parsed.get("coef"))
                    paren = _safe_float(parsed.get("paren"))
                    paren_mode = _norm_paren_mode(parsed.get("paren_mode")) or "unknown"
                    stars = _safe_int(parsed.get("stars"))
                    if stars is not None and stars not in (0, 1, 2, 3):
                        stars = None
                    notes = parsed.get("notes")
                    if isinstance(notes, str):
                        notes = notes.strip() or None
                    else:
                        notes = None

                    out_item["observed_coef"] = coef
                    out_item["observed_paren"] = paren
                    out_item["observed_paren_mode"] = paren_mode
                    out_item["observed_stars"] = stars
                    out_item["notes"] = notes
                    out_item["observed_se"] = paren if (paren is not None and paren_mode == "se") else None
                    out_item["observed_t"] = paren if (paren is not None and paren_mode == "t") else None
                    updated = True
                    done_items += 1

                log.write(
                    json.dumps(
                        {
                            "ts": _now_iso(),
                            "paper_id": paper_id,
                            "item_id": item_id,
                            "snippet_relpath": snippet_rel,
                            "llm_raw": raw_text,
                            "parsed": parsed,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                log.flush()

                if args.sleep_s and args.sleep_s > 0:
                    time.sleep(float(args.sleep_s))

                if args.max_items is not None and done_items >= int(args.max_items):
                    break

            if updated:
                # Mark completion timestamp if every item has at least one observed numeric.
                all_done = True
                for li in label_obj.get("items") or []:
                    if not isinstance(li, dict):
                        continue
                    if li.get("observed_coef") is None and li.get("observed_paren") is None:
                        all_done = False
                        break
                label_obj["annotator"] = annotator
                label_obj["completed_at"] = _now_iso() if all_done else None

                # Stable ordering.
                label_obj["items"] = sorted(
                    [x for x in (label_obj.get("items") or []) if isinstance(x, dict)],
                    key=lambda x: str(x.get("item_id") or ""),
                )
                out_label_path.write_text(json.dumps(label_obj, ensure_ascii=False, indent=2), encoding="utf-8")

            print(f"[{_now_iso()}] ({ti}/{len(task_files)}) {paper_id} labeled_items={done_items}")

            if args.max_items is not None and done_items >= int(args.max_items):
                break

    print(f"[{_now_iso()}] done. labeled_items={done_items} total_task_items_seen={total_items}")
    print(f"[{_now_iso()}] labels dir: {out_labels_dir}")
    print(f"[{_now_iso()}] log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
