# JF 论文计划：多模态“审计智能体”识别 p-hacking（Finance）

> 目标：把当前原型（多模态读图 + anchors 证据链 + 可复现日志）提升为 **JF 可发表的、可验证的测度与实证论文**。  
> 本文把智能体定位为“**测量工具（measurement device）** + **可审计的风险指数**”，而不是“黑箱判官”。

---

## 0. 现状评估（我们现在有什么、缺什么）

### 已有（可作为论文的 demo / 附录材料）
- 多模态篇内审计脚本：`scripts/p_hacking_agent.py`  
  - 读 PDF 文本 + 页图（vision），输出 anchors（页码 + 可指认短语/表名/注释）。  
  - 每次运行保存全量日志（prompt、响应、raw、错误、耗时）。  
- 可规模化的离线“风险指数”基线（可直接做描述性事实）：  
  - SSRN 语料构建：`scripts/build_ssrn_corpus.py` → `corpus/manifest.csv` + `corpus/pdfs/`  
  - 表格统计量抽取 + caliper/binomial + p-curve 基线：`scripts/extract_within_paper_metrics.py` → `corpus/features.csv` + `corpus/tests/`  
  - 初版 stylized facts：`scripts/generate_stylized_facts.py` → `analysis/stylized_facts.md` + `analysis/title_duplicates.csv` + `analysis/*.png`  
  - test-level 分布事实（更贴近 Brodeur 风格的证据展示）：`scripts/test_level_stylized_facts.py` → `analysis/test_level_stylized_facts.md` + `analysis/fig_test_p_*.png`  
  - 原型去重（避免重复版本双计）：`scripts/dedupe_panel_by_title.py` → `analysis/paper_panel_dedup.csv` + `analysis/paper_dedup_map.csv`  
  - 一键可复现离线 pipeline：`scripts/run_offline_pipeline.py` → `analysis/pipeline_runs/`（记录 config + run.log）  
- 外部效度/版本映射（OpenAlex 原型；把 SSRN 预印本映射到可能的已发表版本）：  
  - OpenAlex works 拉取：`scripts/fetch_openalex_works.py` → `analysis/openalex_works.csv`  
  - 已发表版本匹配（title search）：`scripts/map_published_versions_openalex_search.py` → `analysis/paper_openalex_publication_map_search.csv`  
  - 面板合并：`scripts/build_panel_with_openalex.py` → `analysis/paper_panel_with_openalex.csv`  
  - 预测效度回归（原型）：`scripts/predictive_validity_openalex.py` → `analysis/predictive_validity_openalex*.md`  
  - **版本映射有效性（人工审计）**：`scripts/make_publication_mapping_audit_tasks.py` + `scripts/score_publication_mapping_audit.py` → `analysis/publication_map_audit_report.md`  
- 抽取准确性验证脚手架（便于做 JF 需要的 measurement validity）：`annotations/` + `scripts/make_extraction_audit_tasks.py` + `scripts/score_extraction_audit.py`
- 人工审计标注评分脚本（把 labels 变成可写进论文的表）：`scripts/score_human_audit_labels.py` → `annotations/audit_report.md`
- 方法来源与 DOI 列表（可作为方法来源最小集合）：`p_hacking_agent_methodology.md`
- 逐篇方法精华总结（供我们梳理“模块库”）：`p_hacking_detection_methods.md`

### 关键缺口（没有这些很难上 JF）
1) **大样本、可复现的定量结果**：目前语料仍偏小（原型级），无法回答金融领域“系统性”的问题；需要把语料扩到“能做主表”的量级（通常是数千篇）。  
2) **严格的统计检测模块落地**：已实现 caliper/binomial 与 p-curve 的轻量基线，但仍缺少更严谨的结构检验、multiple-testing 暴露度的系统度量、spec-search 强度的形式化，以及明确的对比基准与不确定性处理。  
3) **可靠性/有效性证明（validation）**：已搭好抽取准确性与半合成实验的脚手架，但还缺“有人标/可复现真值”的结果表（以及 IRR / out-of-sample predictive validity）。  
4) **研究问题的金融学贡献**：需要将风险指数与金融学关切（知识累积、复现、政策与激励）绑定，而不是“通用科研诚信工具”。

---

## 1. JF 风格的研究问题与定位（必须“像金融论文”）

### 1.1 核心研究问题（建议主线）
**RQ1（测度）：** 能否构建一个可审计的、可规模化的 **within-paper p-hacking risk index**，并在金融顶刊论文中稳定测量？  
**RQ2（外部效度）：** 该指数是否预测 **后续知识修正**（效应衰减、复现失败、修订/更正、后续文献推翻）？  
**RQ3（制度与激励）：** 数据/代码披露政策、期刊规范变迁、作者激励（职业阶段、合作者结构等）是否与该指数系统相关？

> 关键：我们不声称“识别真 p-hacking”，而是做 **风险测度** + **可复核证据链** + **预测/解释金融知识演化**。

### 1.2 论文贡献（JF 需要明确“增量”）
- **方法贡献（measurement）：** 面向 PDF 的多模态抽取 + 证据溯源（page + anchor + table cell provenance）。  
- **实证贡献（economics/finance）：** 在金融顶刊的大样本上，量化研究者自由度/显著性通胀风险的分布、趋势与制度关联；并与后续纠错/复现结果建立联系。  
- **可复现贡献（open science / credibility）：** 提供可审计的 pipeline（日志、抽取误差评估、稳健性与消融）。

---

## 2. 研究设计总览（我们要把 LLM 放在什么位置）

### 2.1 推荐架构：LLM 做“证据抽取器”，统计模块做“判别器”
1) **多模态抽取层（LLM/vision + 规则 + 版面识别）**  
   - 定位表格/注释/稳健性页，抽取 `coef / se / t(z) / stars / N / FE / controls / weights` 等。  
   - 输出必须带 provenance：`page`, `bbox`, `table_id`, `row/col`, `anchor_text`。  
2) **可复现统计层（deterministic）**  
   - 基于抽取到的统计量，执行 bunching、p-curve/形状约束、multiple-testing 校准、spec-search 强度等。  
3) **解释层（可选 LLM）**  
   - 把统计层输出翻译成“证据卡片 + 下一步复核建议”，但引用必须可追溯到 provenance。

> JF 更容易接受：**LLM 负责“读 PDF 的脏活”**，而不是直接给“有罪判决”。

---

## 3. 数据：是否需要更多论文？需要什么样的数据集？

### 3.1 主语料（用于定量分析）
**目标：** 覆盖 JF/JFE/RFS（可扩展到 JFQA/Review of Finance）至少 20–30 年的论文 PDF + 元数据。  
**最小要求：**
- 全样本或可解释抽样框架（不能 convenience sample）。  
- 论文级元数据：年份、期刊、领域、作者、JEL/关键词、引用、版本信息（working paper vs published）。  
- 文内结果抽取：主结果表、关键稳健性表、附录。

#### SSRN 作为主要下载源（可行的工程化方案）
如果以 SSRN 工作论文为主语料，建议使用 **Crossref → SSRN DOI（10.2139/ssrn.<id>）→ SSRN 下载** 的可复现路径：
- 通过 Crossref API 按关键词与年份段检索 `container-title: SSRN Electronic Journal`，抽取 SSRN DOI 并解析出 `<id>`；
- 用 `papers.ssrn.com/sol3/Delivery.cfm?abstractid=<id>&download=1` 下载 PDF，并保存哈希与元数据；
- 形成可复现的 corpus manifest（paper_id、url、hash、年份、标题/作者等）。

本仓库已提供对应脚本：
- `scripts/build_ssrn_corpus.py`：收集/下载 SSRN PDF 并生成 `corpus/manifest.csv` + `corpus/manifest.jsonl`
  - 可用 `--queries-file` 固化查询（例如 `configs/ssrn_queries_finance.txt`），便于复现“抽样框架”。
  - 若需要更“金融味”的高精度样本（减少非金融论文混入），可改用 `configs/ssrn_queries_finance_strict.txt`。
- `scripts/extract_within_paper_metrics.py`：在 `corpus/pdfs/` 上抽取可规模化的篇内指标（含阈值附近 t 统计量近似、关键词强度等）

### 3.2 验证数据（用于证明可靠性/有效性；比“更多论文”更重要）
至少需要三类“锚定真值/外部标准”的集合：
1) **Extraction Ground Truth 集**（测量有效性）  
   - 有公开 replication package 或 LaTeX 源的论文：可重算表格或直接从源表格拿真值。  
2) **Human Audit 标注集**（一致性与可解释性）  
   - 随机抽取 N≈200–500 篇（分期刊/年份/领域分层），由 trained RAs 做结构化标注：  
     - 研究者自由度类型（outcome 多、子样本多、变量定义多、窗口多、模型多等）  
     - 是否报告 multiple-testing 校正/预注册/分析计划  
     - 是否存在“阈值边缘集中”“只强调显著”等叙事信号  
   - 这不是标注“是否 p-hacking”，而是标注 **可观察红旗**。  
3) **Outcome/Correction 集**（预测效度）  
   - 收集：后续复现尝试、勘误/更正、后续文献中的效应衰减/推翻证据（可用 citation context + 手工核对）。
   - **可先做的原型 outcome（但不能止步于此）**：OpenAlex citation counts + citation trajectories（必须先完成“预印本→已发表版本”的映射与抽查）。

> 本仓库已提供 Human Audit 的最小脚手架：`annotations/README.md`、`annotations/schema.json`、以及 `scripts/make_audit_tasks.py`（从 `corpus/features.csv` 采样并生成待标注任务 JSON）。
> 同时提供“抽取准确性”的数值标注脚手架：`scripts/make_extraction_audit_tasks.py`（生成 cell-level 片段 PNG + 转录任务）与 `scripts/score_extraction_audit.py`（汇总误差）。
> 如果短期没有 RA，人机结合的折中方案是：先用严格的 deterministic 规则只保留“高置信”已发表版本匹配（并在附录报告阈值与覆盖率）；但 JF 最终仍会期待至少一个独立的人类抽查样本来证明 mapping validity。

### 3.3 数据合规（必须写进论文/附录）
- PDF 获取与解析需遵守期刊版权与访问条款；对外开源时可只发布抽取到的“统计量 + provenance（不含原文图片）”与代码。

---

## 4. 指标体系：我们到底“测量”什么？

### 4.1 需要清晰区分的概念（论文必须写清）
- **p-hacking / researcher degrees of freedom**：通过规格/变量/样本选择把结果推过阈值。  
- **selective reporting**：做了但不报/弱化不利结果。  
- **publication bias**：跨论文层面的选择机制。  
- **multiple testing / data snooping**：大量检验导致假阳性膨胀（即使没有故意 p-hack）。

### 4.2 within-paper risk index 的候选模块（可加权/可分项）
1) **Threshold proximity（阈值边缘性）**  
   - t/z 接近 1.96（或 1.645）、星号阈值边缘、p≈0.05。  
2) **Bunching / missing mass（阈值附近堆积/缺口）**  
   - within-paper 版本：对同一论文中所有 reported tests 做局部窗口比较/置换检验。  
3) **Multiple testing exposure（多重检验暴露度）**  
   - outcome 数、机制指标数、异质性维度数、规格数（能抽取多少算多少）。  
4) **Correction disclosure（校正/预注册披露）**  
   - 是否报告 FDR/FWER/Bonferroni/更高 t 门槛/预分析计划。  
5) **Specification search intensity（规格搜索强度）**  
   - 规格数量、关键系数在规格间波动、显著性翻转、只突出某些列/子样本。  

> 输出建议：一个总分 + 分模块分数 + provenance（便于读者接受）。

---

## 5. 方法：哪些“方法论文”需要补齐（作为模块与引用来源）

### 5.1 必须纳入的经典/可操作方法（建议优先级）
- **阈值附近异常 / 显著性通胀诊断**：Brodeur et al.（2016, 2020）  
- **p-hacking 的可检验限制/形状约束**：Elliott et al.（2022）  
- **金融中的多重检验校准与更高门槛**：Harvey et al.（2015）；Harvey（2017）  
- **发表偏倚结构识别与校正**：Andrews & Kasy（2019）  
- **规格搜索/推断有效性**：Leamer（1978, 1983）

### 5.2 建议新增（让论文更完整、更“顶刊味”）
- **p-curve 系列（重要：给出“分布形状”的可检验预测）**  
  - Simonsohn, Nelson, & Simmons（2014a, 2014b）  
- **多重检验与 data snooping 的严格控制（金融读者很熟悉）**  
  - Benjamini & Hochberg（1995）  
  - Romano & Wolf（2005a, 2005b）  
- **研究者自由度的“机制解释”与可操作定义**  
  - Simmons, Nelson, & Simonsohn（2011）  
- **multiverse/specification curve（把“合理规格集合”形式化）**  
  - Steegen et al.（2016）；Simonsohn et al.（2020）

---

## 6. 可靠性/有效性：怎么证明“智能体识别可靠”？

### 6.1 分层证明框架（建议写成论文第 3–4 节 + 附录）
1) **Measurement validity（抽取准确性）**  
   - 单元格级/统计量级对齐：coef、se、t、stars、N、FE。  
   - 指标：precision/recall、绝对误差、表格覆盖率、provenance 错误率。  
2) **Reliability（稳定性/一致性）**  
   - 同一论文多次运行（不同页预算/不同提示）的分数稳定性。  
   - 与人类审计员红旗标注的一致性（Krippendorff’s alpha / ICC）。  
3) **Convergent validity（收敛效度）**  
   - 风险指数与 bunching、阈值边缘性、规格数量等“可计算红旗”同向相关。  
4) **Discriminant validity（判别效度）**  
   - 排除“篇幅更长/表更多/引用更多”导致的伪相关：加入 controls（页数、表数、样本量等）。  
5) **Predictive validity（预测效度；JF 最看重）**  
   - 预测后续：效应衰减、复现失败、勘误/更正、后续论文推翻/修正。  
   - Out-of-sample：训练/验证按年份切分（避免用未来信息）。  

### 6.2 关键实验：用“已知 p-hacking 机制”的半合成数据做检验
**思路：** 用真实 replication code 生成“多规格结果全集”，再模拟“研究者选择”生成 p-hacked 结果表。  
- 输入：某些论文的公开数据与代码。  
- 生成：  
  - honest version：全披露或预先固定规格；  
  - p-hacked version：从规格集合中挑 p<0.05 的结果（或挑最大 t）。  
- 评价：智能体风险指数能否区分两类（AUC、校准）；并定位到正确的“可疑表格/段落”。

> 仓库里已提供一个“玩具版”sanity-check：`scripts/half_synthetic_experiment.py`（在抽取到的 p 值上机械注入阈值堆积并检验分数响应）。真正可发表版本仍需基于 replication code 的 DGP（上面这段思路）。

> 这一步非常像 JF 的“identification by design”：我们自己构造 DGP 来证明指标的识别力，而不是靠主观案例。

---

## 7. 主实证（JF 风格的结果组织）

### 7.1 描述性事实（stylized facts）
- 顶刊论文的 within-paper risk index 的分布、尾部特征（是否 fat tail）。  
- 不同期刊/领域/年份的差异与趋势。

### 7.2 制度变迁（事件研究）
- 期刊数据/代码披露政策引入前后：风险指数的变化（DID / event study）。  
- 以“可审计性提升”为机制：披露政策应降低阈值边缘性与未校正多重检验暴露度。

### 7.3 预测效度（面向“知识累积”）
- 风险指数是否预测：  
  - 后续文献中的效应衰减（同一关系在后续论文中的估计下降）；  
  - 复现失败/争议（需要手工或半自动标注）。  

---

## 8. 预期的论文结构（方便开始写）
1. Introduction：金融知识可信度问题 + 我们的测度与新事实。  
2. Institutional background：披露政策、replication 文化、顶刊规范。  
3. Data：语料、抽样、标注、外部 outcome。  
4. Method：多模态抽取 + 风险指数构建 + 统计检验模块。  
5. Validation：抽取准确性、稳定性、半合成实验、对比基准。  
6. Results：stylized facts、政策效应、预测效度。  
7. Discussion：边界、伦理、如何使用（审计而非定罪）。  
8. Conclusion。

---

## 9. 计划产出（表/图清单，按 JF 口味）
- **Figure 1**：Pipeline（PDF→provenance→tests→index）。  
- **Table 1**：语料描述（期刊×年份×领域）。  
- **Table 2**：抽取准确性（cell-level / table-level）。  
- **Table 3**：风险指数分布与分解（模块贡献）。  
- **Figure 2**：阈值附近统计量分布（bunching 直观图）。  
- **Table 4**：半合成 p-hacking 实验（AUC/校准）。  
- **Table 5**：披露政策事件研究。  
- **Table 6**：预测效度（效应衰减/复现失败）。  
- **Internet Appendix**：更多稳健性、消融、提示词与日志规范、伦理声明。

---

## 10. 实施路线图（建议 10–16 周的可执行里程碑）

### Phase 0（1 周）：研究问题与预注册草案
- 定稿研究问题、主指标定义、验证设计（避免“我们自己 p-hack”）。  
- 产出：pre-analysis plan v0（内部）。

### Phase 1（2–4 周）：语料与元数据管线
- 抓取/整理 PDF 与 metadata（期刊、年份、作者、DOI、版本）。  
- 产出：corpus manifest（paper_id、来源、版本、hash、可用性）。

### Phase 2（3–6 周）：表格/统计量抽取（最关键工程）
- 表格检测（layout）→ 数字抽取 → 统计量重建（t=coef/se）→ stars 规则学习。  
- provenance 标准化：每个数字对应 page+bbox+table_id+row/col。  
- 产出：structured results dataset（parquet/duckdb）。

### Phase 3（2–4 周）：统计检测模块 + 指标构建
- 实现 bunching、阈值边缘性、multiple-testing 暴露度、校正披露、spec-search 强度。  
- 产出：paper-level index + components。

### Phase 4（2–4 周）：验证与基准对比
- Ground-truth 对齐（replication/LaTeX）+ human audit 标注。  
- 半合成实验（真实数据+代码生成 p-hacked vs honest）。  
- 产出：validation tables/figures（可写进论文）。

### Phase 5（2–4 周）：主实证与写作
- stylized facts、政策事件研究、预测效度。  
- 产出：主回归表、图、附录稳健性。

### Phase 6（1–2 周）：打磨与投稿准备
- 语言、结构、附录、复现包、限制声明、伦理声明。

---

## 11. 风险与对策（提前写好，避免项目卡死）
- **LLM 端点不稳定/不可用**：必须有离线/规则抽取与可替换模型；把 LLM 只用于“读 PDF”而非核心判别。  
- **抽取误差导致测度偏误**：用 ground truth 与 provenance 约束；把不确定性显式纳入（置信度、覆盖率权重）。  
- **“p-hacking”不可直接观测**：用半合成实验 + 外部 outcome 预测效度，避免“真值不存在”的逻辑陷阱。  
- **伦理与声誉风险**：报告必须是“风险诊断”，不对单篇论文做定罪式标签；提供可复核证据链与免责声明。

---

## Appendix A. 建议新增的关键方法论文（含 DOI；后续可扩展）

> 注：以下 DOI 已用 Crossref 校验；若后续版本/期刊不同，以最终引用为准。

### p-curve / researcher degrees of freedom
- Simonsohn, U., Nelson, L. D., & Simmons, J. P. (2014). P-curve: A key to the file-drawer. *Journal of Experimental Psychology: General, 143*(2), 534–547. https://doi.org/10.1037/a0033242  
- Simonsohn, U., Nelson, L. D., & Simmons, J. P. (2014). *p*-Curve and effect size. *Perspectives on Psychological Science, 9*(6). https://doi.org/10.1177/1745691614553988  
- Simmons, J. P., Nelson, L. D., & Simonsohn, U. (2011). False-positive psychology: Undisclosed flexibility in data collection and analysis allows presenting anything as significant. *Psychological Science, 22*(11), 1359–1366. https://doi.org/10.1177/0956797611417632  

### multiverse / specification curve
- Steegen, S., Tuerlinckx, F., Gelman, A., & Vanpaemel, W. (2016). Increasing transparency through a multiverse analysis. *Perspectives on Psychological Science, 11*(5), 702–712. https://doi.org/10.1177/1745691616658637  
- Simonsohn, U., Simmons, J. P., & Nelson, L. D. (2020). Specification curve analysis. *Nature Human Behaviour, 4*(11), 1208–1214. https://doi.org/10.1038/s41562-020-0912-z  

### multiple testing / data snooping
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B, 57*(1), 289–300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x  
- Romano, J. P., & Wolf, M. (2005). Exact and approximate stepdown methods for multiple hypothesis testing. *Journal of the American Statistical Association, 100*(469), 94–108. https://doi.org/10.1198/016214504000000539  
- Romano, J. P., & Wolf, M. (2005). Stepwise multiple testing as formalized data snooping. *Econometrica, 73*(4). https://doi.org/10.1111/j.1468-0262.2005.00615.x  

### broader credibility context（可选）
- Ioannidis, J. P. A. (2005). Why most published research findings are false. *PLoS Medicine, 2*(8), e124. https://doi.org/10.1371/journal.pmed.0020124  

---

## Appendix B. 我们应该如何扩展 Finance 语料（操作建议）
- 先锁定 **JF/JFE/RFS** 的可得 PDF（含 working paper 对照），并建立 paper_id→DOI→版本链。  
- 优先收集 **有 replication package** 的论文作为验证集（即便主样本更大）。  
- 对每篇论文，至少保证抽取：主结果表 + 关键稳健性表（附录/online appendix）。  
