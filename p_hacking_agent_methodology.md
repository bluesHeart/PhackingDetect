# 多模态 p-hacking 风险诊断智能体：方法与来源

本项目的 `scripts/p_hacking_agent.py` 是一个**无需任何 agent 框架**的“篇内（within-paper）p-hacking 风险诊断”脚本。它的核心目标是：以**最小成本**（有限页数的图像阅读 + 轻量文本启发式）最大化发现可疑信号，并把证据落在“页码 + anchors（可指认短语/表名/注释句）”上。

## 智能体实际使用的方法（实现层面）

1) **多模态证据抽取（核心）**  
   - 同时使用 **PyMuPDF 抽取文本**（可能有错/缺失）与 **PDF 单页渲染图片**（更可靠）。  
   - LLM 逐页从图片中抽取：表/图名称、显著性星号惯例、临界结果（靠近阈值）、多重检验风险、稳健性/规格搜索痕迹、选择性强调语言，并输出 anchors。

2) **“性价比”页面选择（尽量找表和稳健性页）**  
   - 先用全文文本的启发式特征（数字密度、Table/Robust/p-value 关键词、星号密度等）给页面打分，再让 LLM（若可用）从摘要信息里挑选少量关键页；若 LLM 选择失败则退回纯启发式选页。  

3) **阈值附近异常的低成本近似（caliper/临界贴合的篇内版本）**  
   - 从全文抽取到的文本中用正则提取 `p=...` / `p<...` 的数值，并统计 `0.05/0.10/0.01` 附近小窗口的左右计数（粗略的阈值附近堆积/缺口信号）。  
   - 更重要的是：让 LLM 在**表格图片**里直接定位是否存在 “t≈1.96 / p≈0.05 / 星号刚好出现” 的结果并给出 anchors。

4) **多重检验/多指标/多子样本的“未校正风险”检查**  
   - 在图片中识别“很多 outcome / 很多列 / 很多异质性 / 很多机制指标”，并检查是否出现 FWER/FDR/更高 t 门槛等校正或事先预注册式约束。  

5) **规格搜索/稳健性压力/选择性报告线索（篇内启发式）**  
   - 重点抓：大量规格切换、样本/控制变量变化导致显著性跳变、只强调显著结果、将零结果移动到附录/在线附录等，并要求 anchors。

6) **离线可规模化测度（JF 主实证用的“可复现统计层”原型）**  
   - 在 `corpus/pdfs/` 上做表格检测与 `(coef,se)→t→p` 重建，并计算：  
     - 0.05/0.10 附近 caliper 统计（含 z 与 two-sided binomial p-value）；  
     - |t|≈1.96/1.645 附近的阈值统计；  
     - 简化版 p-curve 形状统计（显著区间内是否“向 0.05 堆积”）。  
   - 入口：`scripts/extract_within_paper_metrics.py`（输出 `corpus/features.csv` 与 `corpus/tests/`；用于 `analysis/stylized_facts.md`）。  

> 注：该智能体并未在代码里实现“严格的 p-curve 形状检验/完整的发表偏倚结构估计”等重型统计程序；它把这些作为**可复核的后续建议**输出（属于最小成本最大收益策略）。

## 方法来源（作者-年份）

智能体的上述模块主要来自以下方法论文/经典论文的可操作思想：

- **阈值附近堆积 / 显著性通胀、以及基于阈值的诊断**： (Brodeur et al., 2016; Brodeur et al., 2020)  
- **p-curve / p-hacking 的可检验形状限制（作为诊断与后续建议）**： (Elliott et al., 2022)  
- **p-curve 的显著区间形状诊断（用于 p-curve 模块与简化实现参考）**： (Simonsohn et al., 2014a; Simonsohn et al., 2014b)  
- **多重检验与更高显著性门槛（因“因子/结果太多”导致假阳性）**： (Harvey et al., 2015; Harvey, 2017)  
- **发表偏倚识别与校正（用于区分“选择性发表/报告”与 p-hacking）**： (Andrews & Kasy, 2019)  
- **规格搜索/研究者自由度与推断有效性**： (Leamer, 1978; Leamer, 1983)  

## References (APA, with DOI)

- Andrews, I., & Kasy, M. (2019). Identification of and correction for publication bias. *American Economic Review, 109*(8), 2766–2794. https://doi.org/10.1257/aer.20180310
- Brodeur, A., Cook, N., & Heyes, A. (2020). Methods matter: p-hacking and publication bias in causal analysis in economics. *American Economic Review, 110*(11), 3634–3660. https://doi.org/10.1257/aer.20190687
- Brodeur, A., Lé, M., Sangnier, M., & Zylberberg, Y. (2016). Star wars: The empirics strike back. *American Economic Journal: Applied Economics, 8*(1), 1–32. https://doi.org/10.1257/app.20150044
- Elliott, G., Kudrin, N., & Wüthrich, K. (2022). Detecting p-hacking. *Econometrica, 90*(2), 887–906. https://doi.org/10.3982/ecta18583
- Harvey, C. R. (2017). Presidential address: The scientific outlook in financial economics. *The Journal of Finance, 72*(4), 1399–1440. https://doi.org/10.1111/jofi.12530
- Harvey, C. R., Liu, Y., & Zhu, H. (2015). … and the cross-section of expected returns. *Review of Financial Studies, 29*(1), 5–68. https://doi.org/10.1093/rfs/hhv059
- Leamer, E. E. (1983). Let's take the con out of econometrics. *The American Economic Review, 73*(1), 31–43. (No DOI found.)
- Leamer, E. E. (1978). *Specification Searches: Ad Hoc Inference with Nonexperimental Data*. John Wiley & Sons. (No DOI; book.)
- Simonsohn, U., Nelson, L. D., & Simmons, J. P. (2014a). P-curve: A key to the file-drawer. *Journal of Experimental Psychology: General, 143*(2), 534–547. https://doi.org/10.1037/a0033242
- Simonsohn, U., Nelson, L. D., & Simmons, J. P. (2014b). *p*-Curve and effect size. *Perspectives on Psychological Science, 9*(6), 666–681. https://doi.org/10.1177/1745691614553988
