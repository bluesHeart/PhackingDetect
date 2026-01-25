# Roadmap（向 JF-ready 推进的下一步清单）

> 目标：把 `docs/rubric.md` 的“拒稿点”变成可执行的里程碑（每一条都有产出物与验收）。

## 0) 当前状态（工程/测度）
- 最新抽取器：`scripts/extract_within_paper_metrics.py`（`EXTRACTOR_VERSION=v2_20260125_3`）
- 最新快照：`corpus_800_v4/` + `analysis_800_v4/`
- v4 代标注（triage，不替代真人）：`analysis_800_v4/extraction_audit_report_llm.json`
  - 主要改进：paren_mode 语义错判显著下降，|t|/p 误差分位数显著下降
  - 仍存在：少量 PDF 文本层缺失小数点导致 coef 缩放错误（影响 se-mode 的 |t| 复原）

## 1) JF-ready 最大差距（按优先级）

### A. Sampling frame（数据）
1. **定义“总体”**：明确 finance/econ-finance 的范围、年份窗口、来源（SSRN→Crossref/OpenAlex）、排除规则（books/theses）。
2. **扩样本到 ≥3k**：并输出可写进 paper 的“抽样流程图 + 数量表”。
3. **去重与版本映射审计**：把 SSRN→published 的 mapping accuracy 做到可报告（含人工 audit + error bars）。

### B. Measurement validity（测度有效性）
4. **真人双标注 + IRR**：对 cell-level extraction（coef/paren/paren_mode/stars）做 ≥2 annotators 的一致性与误差率报告。
5. **处理“缺失小数点”类 PDF**：给出明确策略（剔除/保守过滤/可复现 OCR），并展示对主结果的稳健性。
6. **误差对主结论的敏感性**：把“抽取误差/覆盖率”纳入 robustness（剔除低质量 paper、加权、边界分析）。

### C. Predictive validity（finance 贡献）
7. **预先指定主回归设计**：risk index → finance outcome（citation/发表/修订/复现/效应收缩/因子 attenuation 等），并明确识别假设。
8. **排除机械混淆**：把 paper length、table count、coverage、year FE 等作为 baseline confounds；提供对照/安慰剂。

### D. Reproducibility（可复现）
9. **环境锁定**：在 `requirements.txt` 之外提供可复现 lock（`pip-tools`/`uv`/`pip freeze` 方案择一）。
10. **一键生成主图表**：脚本化产出 paper exhibits（Table/Figure）并在 `analysis_*/` 中版本化（含 config + run log）。

## 2) 建议执行顺序（最短路径）
1. 先把 **抽取审计（真人双标注 + IRR + 误差率）** 做成可写进 paper 的 measurement section。
2. 同步把 **样本扩到 3k+**，并冻结一个 “main snapshot”。
3. 在冻结快照上固化 **predictive validity 主回归 + robustness**，输出 paper-ready exhibits。

