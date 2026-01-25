# PROGRESS（JF 目标推进记录）

> 目的：把 `docs/goal.md` / `docs/rubric.md` 的“目标—差距—行动—验收”落成可追溯的迭代记录，避免只在对话里推进而不可复盘。

## 当前样本规模（repo 内快照）
- `corpus/`：138 篇 PDF（原型）
- `corpus_300/`：280 篇 PDF
- `corpus_800/`：725 篇 PDF
- `corpus_800_v2/`：725 篇 PDF（与 `corpus_800/` 同一批 PDF 的 v2 抽取快照；仅复用 pdf/manifest，tests/features 独立）
- `corpus_800_v3/`：725 篇 PDF（v2 抽取器迭代版，主要强化噪声过滤与 paren_mode 推断）
- `corpus_800_v4/`：725 篇 PDF（v2 抽取器再迭代：修复 unicode minus / 断裂数字 / 小样本 paren_mode 推断）

JF 级别（`docs/rubric.md`）期望：≥ 3,000 篇（并且有清晰 sampling frame + 去重 + 版本映射审计）。

## 已完成（本轮迭代）
- 统一 venv：仓库根 `.venv/`，并提供 `./py` / `./pip` wrapper 作为项目唯一 Python 入口。
- README：补齐项目定位、目录结构、环境、pipeline 入口与审计流程说明（`README.md`）。
- 抽取器 v2（`scripts/extract_within_paper_metrics.py`）：
  - 参考文献页识别与候选页惩罚/排除（减少被 references “高数字密度”污染）。
  - `()` 语义（SE vs t-stat）推断框架 + 版本化 `tests/*.meta.json` sidecar（支持可审计复用）。
  - 新增“列编号/引文式噪声”过滤（典型误抽取 `8) (9)`），并把过滤原因写入 meta/特征中。
  - 新增 `--limit` / `--only-regex` 便于 smoke/debug，不必每次跑全量。
- 人工审计任务升级（`scripts/make_extraction_audit_tasks.py` + `scripts/score_extraction_audit.py`）：
  - 任务 v0.2：要求标注“括号里的数值”以及它是 SE 还是 t-stat（直接验证关键假设）。
  - 评分报告扩展：支持 paren 数值误差、paren 语义匹配率、由观测值推导的 |t|/p 误差等。
- 新增抽取质量总览（`scripts/extraction_quality_report.py`）并集成到 pipeline：
  - `run_offline_pipeline.py` 新增 step：`extraction_quality`，输出 `analysis/extraction_quality_report.md` + `analysis/extraction_quality.csv`。

## 最新结果：`corpus_800_v2`（725 篇）v2 抽取诊断
- 全量抽取耗时：约 73 分钟（`--max-pages-per-paper 12 --max-pdf-pages 200`，跳过 3 篇超长 PDF）
- 抽取质量总览：`analysis_800_v2/extraction_quality_report.md`
  - tests/meta 覆盖：722/722（跳过的 3 篇无 tests）
  - pairs raw/kept：19854 / 18995（overall keep rate ≈ 0.957）
  - paren_mode：`se=602`、`t=120`
  - 主要过滤原因（总计）：`abs_t_gt_80(coef_over_se)=374`、`abs_t_gt_30(paren_t)=284`、`se_gt_50=162`、`column_label_like=31`
- stylized facts：`analysis_800_v2/stylized_facts.md`
  - `t_pairs_seen==0`：352/725（≈0.486）
  - keep-rate 分位数（paper-level）：p10≈0.50, p25≈0.875, p50=1.0
- coverage confound check：`analysis_800_v2/coverage_diagnostics.md`（含 `fig_score_vs_keep_rate.png`）
- extraction audit 任务（v0.2，待人工标注）：
  - tasks：`analysis_800_v2/extraction_tasks/`（30 papers，合计 242 items；目标 per-paper=10，但部分记录因 bbox/snippet 生成失败被跳过）
  - labels（模板，需要填写）：`analysis_800_v2/extraction_labels/`
  - snippets：`analysis_800_v2/extraction_snippets/`
  - 标注完成后评分：`./py scripts/score_extraction_audit.py --tasks-dir analysis_800_v2/extraction_tasks --labels-dir analysis_800_v2/extraction_labels --out analysis_800_v2/extraction_audit_report.json`

## 进展：用视觉 LLM 做一轮“代标注”（用于 triage/调参，不替代真人）
- 自动标注脚本：`scripts/llm_label_extraction_audit.py`
- LLM labels：`analysis_800_v2/extraction_labels_llm/`
- 逐条调用日志：`analysis_800_v2/extraction_labels_llm_log.jsonl`
- 评分报告（基于 LLM 读图）：`analysis_800_v2/extraction_audit_report_llm.json`

> 说明：这一步的价值主要是快速发现系统性误抽取模式（例如列编号/脚注数字被当成 coef），用来迭代抽取器与筛选规则；JF 投稿需要最终用“真人多标注者 + IRR + 误差率报告”落地。

## 最新结果：`corpus_800_v3`（725 篇）抽取器 v2_20260125_2 复跑（对照）
- 目的：验证“列编号/引文式噪声过滤 + paren_mode 推断”修复是否能压低极端误抽。
- 抽取质量总览：`analysis_800_v3/extraction_quality_report.md`
  - pairs raw/kept：19729 / 18810（overall keep rate ≈ 0.953）
  - `column_label_like=109`（更强过滤能捕捉 `62) (-1.70)` 这类误抽模式）
- v3 extraction audit（LLM 代标注，用于 triage）：`analysis_800_v3/extraction_audit_report_llm.json`
  - `paren_mode_match_rate≈0.772`，`abs_t_abs_error.p90≈2.951`（提示 paren_mode 在“少量样本/无文本信号”场景仍会错判）

## 最新结果：`corpus_800_v4`（725 篇）抽取器 v2_20260125_3（本轮修复验证）
- 修复点（`scripts/extract_within_paper_metrics.py`）：
  - 小样本 paren 值也参与 paren_mode 推断（避免 n<20 时一律 default=se 的系统性错判）。
  - cell 文本规范化：unicode minus/断裂数字/星号分隔/不完整括号，显著降低“符号/小数点被吞”的可修复误差。
- 抽取质量总览：`analysis_800_v4/extraction_quality_report.md`
- v4 extraction audit（LLM 代标注，用于 triage）：`analysis_800_v4/extraction_audit_report_llm.json`
  - `paren_mode_match_rate≈0.899`（vs v3≈0.772）
  - `abs_t_abs_error.p90≈0.945`（vs v3≈2.951）
  - `paren_abs_error.max≈6.983`（vs v3≈21.079）
  - 仍存在少量“PDF 文本层缺失小数点导致 coef 缩放错误”的极端 case（例如 `11207` vs `11.207`）；但 t-mode 表格不受影响（t 直接来自括号值）。

## 关键差距（离 JF 还远在哪里）
1. **规模与 sampling frame**：当前仍是 ≤ 800 的快照样本；需要定义清晰的“金融/经金”范围与年份窗口，并扩到 3k+。
2. **测度有效性（measurement validity）**：已具备审计框架，但还缺“真实标注数据 + 误差率/准确率报告 + 对主结果的误差鲁棒性”。
3. **外部效度/版本映射**：版本映射需要更系统的策略与人工审计准确率（当前脚本链已在，但要规模化跑通+报告）。
4. **实证贡献（predictive validity）**：需要一套能说服 JF referee 的识别设计，把 risk index 连到 finance outcome（并排除机械混淆）。
5. **可复现与可移植**：当前 `requirements.txt` 仍是 `>=`（非 pinned）；JF 级别建议补 `pip freeze` lock 或 `uv`/`pip-tools` 锁定依赖版本。

## 下一步建议（按影响力排序）
- 把 extraction audit 真的做起来（关键）：
  - 在 `corpus_800_v2/` 上运行 `make_extraction_audit_tasks.py` 生成任务；
  - 至少 2 位标注者完成 labels；
  - 用 `score_extraction_audit.py` 输出：paren 数值误差、paren 语义匹配率、由观测推导的 |t|/p 误差；
  - 把误差率/准确率写入 paper 的 measurement validity，并做“误抽取剔除/加权”的稳健性。
- 若调整抽取阈值/规则：在 `corpus_800_v2/` 重新跑 `extract,extraction_quality,test_level`，对比 `analysis_800_v2/` 的诊断与主结果变化。
- 扩样本到 3k+ 并形成“可写进 paper 的 sampling frame”。
- 设计并固化 predictive validity 的主回归与 robustness（并把代码产出对齐 paper exhibits）。
