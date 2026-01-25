#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _safe_float(x: Any) -> float | None:
    try:
        if isinstance(x, bool):
            return None
        if isinstance(x, (int, float)):
            v = float(x)
            return None if v != v else v
        if isinstance(x, str) and x.strip():
            v = float(x.strip())
            return None if v != v else v
    except Exception:
        return None
    return None


def _col_or_na(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(np.nan, index=df.index)


def _col_or_false(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(False, index=df.index)


def _first_col(df: pd.DataFrame, names: list[str], default: pd.Series) -> pd.Series:
    for n in names:
        if n in df.columns:
            return df[n]
    return default


def main() -> int:
    ap = argparse.ArgumentParser(description="Join panel CSV with OpenAlex works metadata.")
    ap.add_argument("--panel", default="analysis/paper_panel.csv")
    ap.add_argument("--openalex", default=None, help="OpenAlex CSV (default: <panel_dir>/openalex_works.csv).")
    ap.add_argument(
        "--pub-map",
        default=None,
        help="Optional publication-map CSV from fetch_openalex_related_works.py (default: <panel_dir>/paper_openalex_publication_map.csv if present).",
    )
    ap.add_argument("--out", default=None, help="Output CSV (default: <panel_dir>/paper_panel_with_openalex.csv).")
    args = ap.parse_args()

    panel_path = Path(args.panel)
    if not panel_path.exists():
        raise SystemExit(f"Missing: {panel_path}")
    panel_dir = panel_path.parent
    openalex_path = Path(args.openalex) if args.openalex else (panel_dir / "openalex_works.csv")
    if not openalex_path.exists():
        raise SystemExit(f"Missing: {openalex_path}")
    if args.pub_map:
        pub_map_path = Path(args.pub_map)
    else:
        # Prefer the search-based map (usually much higher quality than OpenAlex related_works).
        cand_search = panel_dir / "paper_openalex_publication_map_search.csv"
        cand_related = panel_dir / "paper_openalex_publication_map.csv"
        pub_map_path = cand_search if cand_search.exists() else cand_related
    out_path = Path(args.out) if args.out else (panel_dir / "paper_panel_with_openalex.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = pd.read_csv(panel_path)
    o = pd.read_csv(openalex_path)
    if "paper_id" not in p.columns or "paper_id" not in o.columns:
        raise SystemExit("Both inputs must include paper_id.")

    merged = p.merge(o, on=["paper_id"], how="left", suffixes=("", "_openalex"))

    if pub_map_path.exists():
        m = pd.read_csv(pub_map_path)
        if "paper_id" in m.columns:
            merged = merged.merge(m, on=["paper_id"], how="left", suffixes=("", "_pubmap"))

    # Derived outcomes
    merged["openalex_cited_by_count"] = pd.to_numeric(merged.get("cited_by_count"), errors="coerce")
    merged["openalex_log1p_cites"] = merged["openalex_cited_by_count"].apply(lambda x: (None if x != x else float(np.log1p(x))))
    merged["openalex_publication_year"] = pd.to_numeric(merged.get("publication_year"), errors="coerce").astype("Int64")
    merged["openalex_is_found"] = merged.get("status").astype(str).eq("ok") if "status" in merged.columns else False

    # Published-version (best related) outcomes, if publication map exists
    merged["published_openalex_id"] = _col_or_na(merged, "best_related_openalex_id")
    merged["published_doi_url"] = _col_or_na(merged, "best_related_doi_url")
    merged["published_title"] = _col_or_na(merged, "best_related_title")
    merged["published_primary_source"] = _col_or_na(merged, "best_related_primary_source")
    merged["published_publication_year"] = pd.to_numeric(_col_or_na(merged, "best_related_publication_year"), errors="coerce").astype("Int64")
    merged["published_cited_by_count"] = pd.to_numeric(_col_or_na(merged, "best_related_cited_by_count"), errors="coerce")
    merged["published_log1p_cites"] = merged["published_cited_by_count"].apply(lambda x: (None if x != x else float(np.log1p(x))))
    merged["published_title_sim_seq"] = pd.to_numeric(_col_or_na(merged, "best_related_title_sim_seq"), errors="coerce")
    merged["published_title_sim_token_f1"] = pd.to_numeric(_col_or_na(merged, "best_related_title_sim_token_f1"), errors="coerce")

    # Map quality flags (bool-ish)
    conf_src = _first_col(merged, ["best_match_is_confident", "best_is_confident"], pd.Series(False, index=merged.index))
    merged["published_is_confident_match"] = conf_src.astype(str).str.lower().isin(["true", "1", "yes"])

    top_src = _first_col(merged, ["best_match_is_top3_venue", "best_related_is_top3_venue"], pd.Series(False, index=merged.index))
    merged["published_is_top3_venue"] = top_src.astype(str).str.lower().isin(["true", "1", "yes"])

    if "has_non_ssrn_candidate" in merged.columns:
        has_any = merged["has_non_ssrn_candidate"]
        merged["published_has_non_ssrn_candidate"] = has_any.astype(str).str.lower().isin(["true", "1", "yes"])
    elif "n_candidates_non_ssrn" in merged.columns:
        merged["published_has_non_ssrn_candidate"] = pd.to_numeric(merged["n_candidates_non_ssrn"], errors="coerce").fillna(0).astype(int) > 0
    else:
        merged["published_has_non_ssrn_candidate"] = False

    merged["generated_at"] = _now_iso()
    merged.to_csv(out_path, index=False)
    print(f"[{_now_iso()}] wrote: {out_path} (rows={len(merged)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
