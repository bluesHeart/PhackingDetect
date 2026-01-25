# Data Collection Plan（JF 采样框架与 OCR 准备）

## 为什么会有“PDF 文本层”问题
很多 finance PDF 的“可选中字符”并不可靠：
- 扫描版/图片版：根本没有真实文本层。
- 字体编码/Unicode 映射异常：负号/小数点/上标等 glyph 在提取时会丢失或被替换（例如 `11.207` → `11207`）。
- 表格布局导致 token 被拆开（例如 `- 2 . 1 0 **`）。

因此，把 PDF **先统一成可复现的 OCR 文本层**（并保留证据/日志）是更容易说服 referee 的路线之一。

## JF-ready 需要收集哪些 PDF（以及为什么）

### 1) 主样本（Main Corpus, ≥3k）
**目标**：做 stylized facts + 主回归（predictive validity）的统计功效与外部效度。

**建议定义**（可写进 paper）：
- 来源：`SSRN Electronic Journal`（Crossref 可检索的 SSRN DOI：`10.2139/ssrn.<id>`）。
- 时间窗：例如 1990–2025（可根据研究问题固定）。
- 主题范围：asset pricing / corporate finance / banking / microstructure / funds / fintech / ESG 等主流 finance 方向。
- 进入主分析前的“经验论文”过滤：至少包含可识别的回归表/检验统计（例如 `t_pairs_seen>0`）。

仓库里已有一个 **≥3k 的候选 id 列表**（4524 篇）：
- `corpus_large/ssrn_ids.txt`
- 这是用 `configs/ssrn_queries_finance_strict.txt` 的 finance 主题词 + Crossref 过滤 `container-title:SSRN Electronic Journal` 拉取的。

### 2) 测度有效性审计样本（Audit Corpus）
**目标**：给 referee 看得懂的“误差率/一致性/鲁棒性”。

建议在主样本上做分层抽样（year/topic/页数/表格密度/是否扫描）：
- cell-level extraction audit（coef/paren/paren_mode/stars）至少双标注 + IRR
- 对比：PDF-text 抽取 vs OCR-text 抽取，报告 error-rate 的下降

### 3) 版本映射审计样本（SSRN→Published）
**目标**：证明 version mapping 的准确率，并支撑 predictive validity（发表/引用/修订等 outcome）。

这部分可以只用 meta/outcome 数据（OpenAlex/Crossref），不一定需要 published PDF；但若能拿到开放获取 PDF，可作为额外验证集。

## 下载与 OCR 的建议工作流（在你的 3090 机器上跑）

### A) 先下载 PDF（SSRN）
在 repo 根目录：
```bash
./py scripts/build_ssrn_corpus.py \\
  --out-dir corpus_large \\
  --ids-file corpus_large/ssrn_ids.txt \\
  --download \\
  --skip-ssrn-meta \\
  --sleep-s 1.0
```
产物：
- `corpus_large/pdfs/ssrn_<id>.pdf`
- `corpus_large/manifest.csv` / `corpus_large/manifest.jsonl`

### B) 分片（方便多进程/多机 OCR）
```bash
mkdir -p ocr_shards
split -d -n l/8 corpus_large/ssrn_ids.txt ocr_shards/ssrn_ids_
ls -1 ocr_shards/ssrn_ids_* | wc -l
```

### C) OCR→Markdown（建议同时保留结构化输出）
不同工具对表格的支持差异很大；为了后续回归表抽取更稳，建议：
- 同时保存 `*.md`（便于阅读/审计）+ `layout.json`/`bbox`（便于定位表格单元格）
- 把 OCR 版本/模型/参数写入日志，保证可复现

