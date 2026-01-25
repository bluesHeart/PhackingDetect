# PhackingDetect：多模态 within-paper p-hacking 风险诊断（Finance）

本仓库是一个**论文原型项目**：把“p-hacking/选择性报告/多重检验/规格搜索”等问题，落到可规模化的 PDF 证据抽取与可复现的风险测度上。

你可以把它理解为两条线并行推进：

1. **离线可规模化（可复现统计层）**：对 SSRN PDF 语料做表格抽取，重建 `(coef,paren)→t→p`（括号可能是标准误或 t-stat），计算阈值附近 caliper/bunching、简化 p-curve、关键词暴露度等，输出 paper-level 的 `offline_risk_score` 与一组可审计的中间产物。
2. **单篇多模态审计（证据链）**：对某一篇论文，用“少量关键页的图像阅读 + anchors（页码+可指认短语/表名）”生成可复核的风险诊断报告（可选接入 OpenAI-compatible LLM；无 LLM 时自动降级为启发式/文本抽取模式）。

研究写作与定位建议见：`jf_p_hacking_agent_paper_plan.md`。

---

## 目录结构（重要产物）

- `scripts/`：所有可运行脚本（入口都在这里）
  - `run_offline_pipeline.py`：一键跑完 SSRN→metrics→stylized-facts（含日志与 config 记录）
  - `build_ssrn_corpus.py`：构建 SSRN PDF 语料（Crossref 查询/SSRN id 列表→下载+manifest）
  - `extract_within_paper_metrics.py`：从 PDF 表格抽取 `(coef,paren)` 并重建 `t/p`（推断括号语义），输出 `features.csv`、逐条 test JSONL 与 `tests/*.meta.json`
  - `generate_stylized_facts.py` / `test_level_stylized_facts.py`：生成描述性事实与图
  - `fetch_openalex_works.py` / `map_published_versions_openalex_search.py` / `build_panel_with_openalex.py`：OpenAlex 元数据与“预印本→已发表版本”映射
  - `p_hacking_agent.py`：单篇多模态审计（LLM 可选）
  - `make_*_audit_tasks.py` / `score_*_audit*.py`：人工审计任务与评分（measurement validity / mapping validity）
- `corpus/`：语料与抽取结果（可重建）
  - `manifest.csv` / `manifest.jsonl`：SSRN id→PDF/元信息清单
  - `pdfs/`：下载的 SSRN PDF
  - `meta/`：SSRN 抽象页元信息缓存
  - `features.csv`：每篇论文的离线特征与 `offline_risk_score`
  - `tests/*.jsonl`：逐条抽取的 test（含 page/table/cell bbox、t/p 近似等）
- `analysis/`：分析输出（可重建）
  - `paper_panel.csv`：paper-level 面板（manifest+features join）
  - `stylized_facts.md` / `test_level_stylized_facts.md`：原型风格化事实
  - `pipeline_runs/<timestamp>/config.json` + `run.log`：每次 pipeline 的可复现记录
- `reports/`：单篇审计 agent 的输出（默认 `reports/p_hacking/<paper>__<hash>/`）
- `annotations/`：人工审计标注与报告（validation 用）

仓库里还包含不同样本规模的“快照输出”（例如 `corpus_300/`、`analysis_300/`、`corpus_800/`、`analysis_800/`、`corpus_large/` 等），用于对比与调试。

---

## Python 环境（建议）

本项目统一使用仓库根目录下的 venv：`.venv/`。

- Python：`3.10+`
- 依赖：见 `requirements.txt`（或使用 `environment.yml` 创建 conda 环境）

使用 venv：

```bash
python3 -m venv --without-pip .venv
python3 - <<'PY'
import urllib.request
urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', '/tmp/get-pip.py')
print('downloaded /tmp/get-pip.py')
PY
.venv/bin/python /tmp/get-pip.py
.venv/bin/pip install -r requirements.txt
```

> 说明：`PyMuPDF`/`pdfplumber` 用于 PDF 文本与表格抽取；`pandas/matplotlib/statsmodels` 用于面板、图形与回归。

使用 conda（可选）：

```bash
conda env create -f environment.yml
conda activate phackingdetect
```

不想 activate 的话，可以直接用仓库自带 wrapper：

```bash
./py scripts/run_offline_pipeline.py --help
./pip list
```

---

## 一键离线 Pipeline（推荐入口）

最常用入口：`scripts/run_offline_pipeline.py`。它会把每次运行的命令、stdout/stderr、参数与关键环境变量写入 `analysis/pipeline_runs/`。

### 1) 基于已有语料，重算特征与分析

```bash
./py scripts/run_offline_pipeline.py \
  --corpus-dir corpus \
  --analysis-dir analysis \
  --steps extract,extraction_quality,summarize,panel,stylized,test_level,dedupe,half_synth
```

### 2) 从 Crossref/SSRN 拉取语料并下载 PDF（需要网络）

准备一个 queries 文件（每行一个 Crossref query，例如 `asset pricing`、`expected returns`）。可参考 `configs/queries.example.txt`，复制为 `configs/queries.txt`，然后：

```bash
export CROSSREF_MAILTO="you@example.com"   # 建议填写，避免被限流
./py scripts/run_offline_pipeline.py \
  --corpus-dir corpus \
  --analysis-dir analysis \
  --queries-file configs/queries.txt \
  --download \
  --steps build,extract,extraction_quality,summarize,panel,stylized,test_level,dedupe
```

也可以直接给 SSRN id 列表（见 `scripts/build_ssrn_corpus.py --help`）。

---

## OpenAlex 扩展（可选：外部效度/版本映射）

OpenAlex 相关脚本会读取 `OPENALEX_MAILTO`（或回退到 `CROSSREF_MAILTO`）：

```bash
export OPENALEX_MAILTO="you@example.com"
./py scripts/run_offline_pipeline.py \
  --corpus-dir corpus \
  --analysis-dir analysis \
  --steps openalex_works,pub_map,openalex_panel,predictive_validity,pub_map_audit_tasks
```

主要产物：

- `analysis/openalex_works.csv`
- `analysis/paper_openalex_publication_map_search.csv`
- `analysis/paper_panel_with_openalex.csv`
- `analysis/predictive_validity_openalex*.md`
- `analysis/publication_map_tasks/` + `analysis/publication_map_labels/` + `analysis/publication_map_audit_report.md`

---

## 单篇多模态审计 Agent（可选）

入口：`scripts/p_hacking_agent.py`，默认输出到 `reports/p_hacking/`，并保存完整的 LLM 调用日志（prompt/response/raw/error）。

```bash
./py scripts/p_hacking_agent.py --pdf "path/to/paper.pdf"
```

LLM 通过 OpenAI-compatible API（可选）：

- `SKILL_LLM_API_KEY`（或 `OPENAI_API_KEY`）
- `SKILL_LLM_BASE_URL`
- `SKILL_LLM_MODEL`

不配置 LLM 也能运行：会自动降级为离线启发式模式（或显式 `--offline`），并输出保守的“建议复核点”。

---

## 复核与人工标注（可选：validation）

- 人工审计任务（paper-level 红旗标注）：`scripts/make_audit_tasks.py` → `annotations/tasks/` + `annotations/labels/`
- 标注评分/一致性报告：`scripts/score_human_audit_labels.py` → `annotations/audit_report.md`
- 抽取准确性验证（cell-level snippets，含括号语义 se/t）：`scripts/make_extraction_audit_tasks.py` + `scripts/score_extraction_audit.py`
- （可选）用视觉 LLM 做一轮快速“代标注/triage”（不替代真人）：`scripts/llm_label_extraction_audit.py`
- 版本映射验证：`scripts/make_publication_mapping_audit_tasks.py` + `scripts/score_publication_mapping_audit.py`

---

## 关键方法与资料

- 多模态审计方法与引用来源：`p_hacking_agent_methodology.md`
- “方法论文”逐篇精华总结（LLM 生成）：`p_hacking_detection_methods.md`（生成脚本：`scripts/summarize_p_hacking_methods.py`）
- 论文链接与资料：`论文链接.md`
