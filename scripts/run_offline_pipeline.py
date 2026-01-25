#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _now_compact() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _tee_run(cmd: list[str], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_now_iso()}] $ {' '.join(cmd)}\n")
        log.flush()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            log.write(line)
        rc = p.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the offline SSRN→metrics→stylized-facts pipeline end-to-end.")
    ap.add_argument("--corpus-dir", default="corpus", help="Corpus directory (manifest/pdfs/features/tests).")
    ap.add_argument("--analysis-dir", default="analysis", help="Analysis output directory (panel, figs, reports).")
    ap.add_argument("--queries-file", action="append", default=[], help="Text file with one Crossref query per line.")
    ap.add_argument("--from-year", type=int, default=2010)
    ap.add_argument("--until-year", type=int, default=int(time.strftime("%Y")))
    ap.add_argument("--max-results", type=int, default=200, help="Max SSRN ids per query (Crossref).")
    ap.add_argument("--sample-n", type=int, default=None, help="Optional down-sampling after de-duplication.")
    ap.add_argument("--sample-seed", type=int, default=123, help="Seed for deterministic down-sampling.")
    ap.add_argument("--download", action="store_true", help="Download PDFs from SSRN.")
    ap.add_argument("--skip-ssrn-meta", action="store_true", help="Skip SSRN HTML metadata fetch (use Crossref only; faster).")
    ap.add_argument("--sleep-s", type=float, default=0.5, help="Polite sleep seconds between SSRN requests.")
    ap.add_argument("--max-pages-per-paper", type=int, default=12, help="Max candidate pages for table parsing.")
    ap.add_argument("--max-pdf-pages", type=int, default=None, help="Skip PDFs with more than this many pages in extraction.")
    ap.add_argument("--force-extract", action="store_true", help="Force recomputation of features even if cached.")
    ap.add_argument("--top-n", type=int, default=20, help="Top-N to list in stylized facts.")
    ap.add_argument("--half-synth-rate", type=float, default=0.5, help="Half-synthetic manipulation rate.")
    ap.add_argument("--make-audit-tasks", action="store_true", help="Create human-audit task templates.")
    ap.add_argument("--audit-n", type=int, default=200)
    ap.add_argument("--audit-seed", type=int, default=123)
    ap.add_argument("--audit-selection", default="stratified", choices=["random", "top_risk", "stratified"])
    ap.add_argument("--pub-map-audit-n", type=int, default=180, help="Number of papers to sample for publication-map validation.")
    ap.add_argument("--pub-map-audit-seed", type=int, default=123)
    ap.add_argument(
        "--pub-map-audit-selection",
        default="stratified",
        choices=["random", "stratified", "top_confident", "top_top3"],
        help="Sampling strategy for publication-map validation tasks.",
    )
    ap.add_argument("--make-extraction-audit", action="store_true", help="Create extraction audit snippets/tasks.")
    ap.add_argument("--extraction-audit-n-papers", type=int, default=40)
    ap.add_argument("--extraction-audit-per-paper", type=int, default=12)
    ap.add_argument(
        "--steps",
        default="build,extract,summarize,panel,stylized,test_level,dedupe,half_synth",
        help=(
            "Comma-separated steps: build,extract,summarize,panel,stylized,test_level,dedupe,half_synth,"
            "extraction_quality,coverage,bunching,openalex_works,pub_map,openalex_panel,predictive_validity,"
            "pub_map_audit_tasks,audit_tasks,extraction_audit"
        ),
    )
    args = ap.parse_args()

    steps = {s.strip() for s in str(args.steps).split(",") if s.strip()}
    analysis_dir = Path(args.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    run_dir = analysis_dir / "pipeline_runs" / _now_compact()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    cfg_path = run_dir / "config.json"

    cfg = {
        "generated_at": _now_iso(),
        "cwd": str(Path.cwd()),
        "python": sys.executable,
        "env": {
            "SKILL_LLM_BASE_URL": os.environ.get("SKILL_LLM_BASE_URL"),
            "SKILL_LLM_MODEL": os.environ.get("SKILL_LLM_MODEL"),
            "CROSSREF_MAILTO": os.environ.get("CROSSREF_MAILTO"),
            "OPENALEX_MAILTO": os.environ.get("OPENALEX_MAILTO"),
        },
        "args": vars(args),
        "steps": sorted(steps),
    }
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    py = sys.executable
    corpus_dir = str(args.corpus_dir)
    panel_csv = str((analysis_dir / "paper_panel.csv").as_posix())

    if "build" in steps:
        cmd = [
            py,
            "scripts/build_ssrn_corpus.py",
            "--out-dir",
            corpus_dir,
            "--from-year",
            str(int(args.from_year)),
            "--until-year",
            str(int(args.until_year)),
            "--max-results",
            str(int(args.max_results)),
            "--sample-seed",
            str(int(args.sample_seed)),
            "--sleep-s",
            str(float(args.sleep_s)),
        ]
        for qf in args.queries_file or []:
            cmd.extend(["--queries-file", str(qf)])
        if args.sample_n is not None:
            cmd.extend(["--sample-n", str(int(args.sample_n))])
        if bool(args.skip_ssrn_meta):
            cmd.append("--skip-ssrn-meta")
        if bool(args.download):
            cmd.append("--download")
        _tee_run(cmd, log_path=log_path)

    if "extract" in steps:
        cmd = [
            py,
            "scripts/extract_within_paper_metrics.py",
            "--corpus-dir",
            corpus_dir,
            "--max-pages-per-paper",
            str(int(args.max_pages_per_paper)),
        ]
        if args.max_pdf_pages is not None:
            cmd.extend(["--max-pdf-pages", str(int(args.max_pdf_pages))])
        if bool(args.force_extract):
            cmd.append("--force")
        _tee_run(cmd, log_path=log_path)

    if "extraction_quality" in steps:
        _tee_run(
            [py, "scripts/extraction_quality_report.py", "--corpus-dir", corpus_dir, "--out-dir", str(analysis_dir)],
            log_path=log_path,
        )

    if "summarize" in steps:
        _tee_run([py, "scripts/summarize_corpus_features.py", "--corpus-dir", corpus_dir], log_path=log_path)

    if "panel" in steps:
        _tee_run(
            [py, "scripts/build_panel_dataset.py", "--corpus-dir", corpus_dir, "--out", panel_csv],
            log_path=log_path,
        )

    if "stylized" in steps:
        _tee_run(
            [py, "scripts/generate_stylized_facts.py", "--panel", panel_csv, "--out-dir", str(analysis_dir), "--top-n", str(int(args.top_n))],
            log_path=log_path,
        )

    if "coverage" in steps:
        _tee_run([py, "scripts/coverage_diagnostics.py", "--panel", panel_csv, "--out-dir", str(analysis_dir)], log_path=log_path)

    if "bunching" in steps:
        _tee_run([py, "scripts/bunching_inference.py", "--panel", panel_csv, "--out-dir", str(analysis_dir)], log_path=log_path)

    if "test_level" in steps:
        _tee_run(
            [py, "scripts/test_level_stylized_facts.py", "--corpus-dir", corpus_dir, "--out-dir", str(analysis_dir)],
            log_path=log_path,
        )

    if "dedupe" in steps:
        _tee_run(
            [
                py,
                "scripts/dedupe_panel_by_title.py",
                "--panel",
                panel_csv,
                "--corpus-dir",
                corpus_dir,
                "--out-panel",
                str((analysis_dir / "paper_panel_dedup.csv").as_posix()),
                "--out-map",
                str((analysis_dir / "paper_dedup_map.csv").as_posix()),
                "--out-report",
                str((analysis_dir / "dedupe_report.md").as_posix()),
            ],
            log_path=log_path,
        )

    if "half_synth" in steps:
        _tee_run(
            [
                py,
                "scripts/half_synthetic_experiment.py",
                "--corpus-dir",
                corpus_dir,
                "--out-dir",
                str(analysis_dir),
                "--manipulation-rate",
                str(float(args.half_synth_rate)),
            ],
            log_path=log_path,
        )

    if "openalex_works" in steps:
        _tee_run([py, "scripts/fetch_openalex_works.py", "--panel", panel_csv], log_path=log_path)

    if "pub_map" in steps:
        _tee_run([py, "scripts/map_published_versions_openalex_search.py", "--panel", panel_csv], log_path=log_path)

    if "pub_map_audit_tasks" in steps:
        _tee_run(
            [
                py,
                "scripts/make_publication_mapping_audit_tasks.py",
                "--search-jsonl",
                str((analysis_dir / "openalex_search_publication_map.jsonl").as_posix()),
                "--panel",
                panel_csv,
                "--n",
                str(int(args.pub_map_audit_n)),
                "--seed",
                str(int(args.pub_map_audit_seed)),
                "--selection",
                str(args.pub_map_audit_selection),
                "--tasks-dir",
                str((analysis_dir / "publication_map_tasks").as_posix()),
                "--labels-dir",
                str((analysis_dir / "publication_map_labels").as_posix()),
            ],
            log_path=log_path,
        )

    if "openalex_panel" in steps:
        _tee_run([py, "scripts/build_panel_with_openalex.py", "--panel", panel_csv], log_path=log_path)

    if "predictive_validity" in steps:
        panel_openalex = str((analysis_dir / "paper_panel_with_openalex.csv").as_posix())
        _tee_run([py, "scripts/predictive_validity_openalex.py", "--panel", panel_openalex, "--outcome", "ssrn"], log_path=log_path)
        _tee_run([py, "scripts/predictive_validity_openalex.py", "--panel", panel_openalex, "--outcome", "published"], log_path=log_path)

    if "audit_tasks" in steps or bool(args.make_audit_tasks):
        _tee_run(
            [
                py,
                "scripts/make_audit_tasks.py",
                "--corpus-dir",
                corpus_dir,
                "--n",
                str(int(args.audit_n)),
                "--seed",
                str(int(args.audit_seed)),
                "--selection",
                str(args.audit_selection),
                "--tasks-dir",
                str((analysis_dir / "audit_tasks").as_posix()),
                "--labels-dir",
                str((analysis_dir / "audit_labels").as_posix()),
            ],
            log_path=log_path,
        )

    if "extraction_audit" in steps or bool(args.make_extraction_audit):
        _tee_run(
            [
                py,
                "scripts/make_extraction_audit_tasks.py",
                "--corpus-dir",
                corpus_dir,
                "--n-papers",
                str(int(args.extraction_audit_n_papers)),
                "--per-paper",
                str(int(args.extraction_audit_per_paper)),
                "--out-dir",
                str((analysis_dir / "extraction_tasks").as_posix()),
                "--labels-dir",
                str((analysis_dir / "extraction_labels").as_posix()),
                "--snippets-dir",
                str((analysis_dir / "extraction_snippets").as_posix()),
            ],
            log_path=log_path,
        )

    print(f"[{_now_iso()}] run dir: {run_dir}")
    print(f"[{_now_iso()}] config: {cfg_path}")
    print(f"[{_now_iso()}] log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
