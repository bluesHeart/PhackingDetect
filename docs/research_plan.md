# Research Plan（下一阶段：OCR→结构化抽取→JF-ready）

本计划面向 `docs/rubric.md` 的 JF-ready 标准，目标是把“测度（measurement）”和“数据（sampling frame）”两条线先做扎实，再推进 predictive validity。

## 0) 我们要下载哪些 PDF（明确清单）

### 主样本（Main corpus，目标 ≥3k PDFs）
- **下载对象**：`corpus_large/ssrn_ids.txt` 中的全部 SSRN id（当前 4524 条）。
- **理由**：
  - 这是用 Crossref 过滤 `container-title: SSRN Electronic Journal` 得到的 SSRN DOI（`10.2139/ssrn.<id>`），可写进 paper 的 sampling frame；
  - 主题词来自 `configs/ssrn_queries_finance_strict.txt`（finance-heavy，减少非金融论文混入）；
  - 4524 是“候选池”，下载后会因缺失 PDF、超长/非论文类型等被过滤，最终希望保留 ≥3000 篇进入主分析。

### 研发/回归测试样本（Dev corpus）
- **下载对象**：可选用 `corpus_800/ssrn_ids.txt`（你也可以从主样本中固定一个 800 子样本）。
- **理由**：用于快速迭代抽取器/过滤规则/审计流程，不必每次跑全量 3k+。

## 1) 在 3090 机器上：下载 → OCR → Markdown

### 1.1 下载 SSRN PDFs（可复现）
```bash
./py scripts/build_ssrn_corpus.py \
  --out-dir corpus_large \
  --ids-file corpus_large/ssrn_ids.txt \
  --download \
  --skip-ssrn-meta \
  --sleep-s 1.0
```
产物：
- `corpus_large/pdfs/ssrn_<id>.pdf`
- `corpus_large/manifest.csv`（记录成功/失败与下载错误）

### 1.2 OCR→Markdown（建议保留“结构化证据”）
**要求（JF 可审计）**：
- 每篇输出：
  - `md`：可读文本（用于复核/审计）
  - 结构化：至少包含 page-level 的 bbox/layout 信息（否则表格对齐会很难做得稳）
- 记录：
  - OCR 工具名 + 版本号 + 模型名 + 参数
  - 失败率与失败原因统计

建议目录约定（示例）：
```
corpus_large_ocr/
  md/ssrn_<id>.md
  layout/ssrn_<id>.json
  logs/...
```

## 2) 抽取器升级（从 PDF-text → OCR-text）
目标：把现有 `scripts/extract_within_paper_metrics.py` 的“表格单元格抽取”升级成：
1) 优先使用 PDF-text（便宜/快）；
2) 若检测到“文本层可疑”（缺小数点/断裂数字/扫描版），回退到 OCR 输出；
3) 输出 provenance：该 paper 使用了哪条路径（pdf-text / ocr / mixed）与对应质量指标。

验收：
- 在 audit 样本上，coef/paren/paren_mode 的误差率显著下降；
- “极端缩放错误”（如 `11207` vs `11.207`）能被识别并处理（剔除/修正/转 OCR）。

## 3) Measurement validity（必须：真人双标注 + IRR）
目标：给 referee 一份可信的“测度准确率/误差率”。

步骤：
- 生成 cell-level audit tasks（已有脚本链）：`scripts/make_extraction_audit_tasks.py`
- 至少 2 位标注者独立标注（coef/paren/paren_mode/stars）
- 评分与 IRR：
  - 误差分布（p50/p90/max）
  - paren_mode match rate（se vs t）
  - IRR（kappa / rank corr 等）

## 4) Reliability（稳定性）
在固定样本上做扰动：
- page budget（`--max-pages-per-paper`）
- 过滤阈值（abs_t 上限、citation-like 规则等）
- OCR vs 非 OCR 路径

输出：稳定性图表（score rank correlation / 分位数稳定性）。

## 5) Predictive validity（finance 贡献）
当 measurement 站稳后再推进：
- SSRN→published 映射（OpenAlex/Crossref）+ 人工 audit accuracy
- outcome：发表/引用/修订/效应收缩/replication proxy 等
- 识别设计与 confound 控制（length、table count、coverage、year FE）

## 6) Paper-ready 复现（工程）
- 固定一个 main snapshot（ids + 版本号）
- 一条命令从 raw corpus 产出所有 exhibits（logs + config capture）

