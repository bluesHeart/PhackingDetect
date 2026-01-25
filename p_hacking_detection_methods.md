# 识别 p-hacking 的方法：基于 papers_md 的逐篇精华总结

> 生成时间：2026-01-24 12:31:14
> 说明：本总结由脚本使用 OpenAI-compatible LLM API 对 `papers_md/` 内每篇论文的 Markdown 转写内容逐块阅读后提炼。仅基于文本可见信息；若文内缺失书目信息则在对应条目中标注缺失。

## 方法总览（按类别）

### Bayesian

#### Prior Specification Grouping (Leamer, 1983)

- 检测目标：Influence of subjective researcher 'types' (e.g., Right Winger, Bleeding Heart) on results.
- 核心统计量/检验：Alternative Prior Specifications

### Bayesian Bounds

#### Ellipsoid Bound Theorem (Leamer, 1978)

- 检测目标：The feasible range of posterior means when prior precision is arbitrary or unknown.
- 核心统计量/检验：Feasible ellipsoid centered between prior and sample locations

### Bayesian Econometrics

#### Bayesian Sensitivity Analysis (Leamer, 1983)

- 检测目标：Sensitivity of conclusions to the choice of prior distributions on bias parameters.
- 核心统计量/检验：Generalized Least Squares (GLS) incorporating prior covariance matrix M.

### Bayesian Inference

#### Bayesian Hierarchical Framework (Harvey et al., 2014)

- 检测目标：Incorporates a multiplicity penalty (Ockham's razor) into posterior discovery probabilities.
- 核心统计量/检验：Posterior discovery probability

#### Bayesianized p-value (Harvey, 2017)

- 检测目标：Probability that the null hypothesis is true
- 核心统计量/检验：Posterior odds

#### Conically Uniform Priors (Leamer, 1978)

- 检测目标：Sensitivity to the belief that coefficients have similar size and sign.
- 核心统计量/检验：Information contract curve derived from conical isodensity surfaces.

#### Conjugate Normal-Gamma Prior (Leamer, 1978)

- 检测目标：Information conflict between prior and sample.
- 核心统计量/检验：Posterior distribution parameters

#### Implicit Prior Constraints (Leamer, 1978)

- 检测目标：Inconsistent prior updates when expanding models with data-instigated variables.
- 核心统计量/检验：Variance consistency equations (9.17)

#### Minimum Bayes Factor (MBF) (Harvey, 2017)

- 检测目标：Overstatement of evidence against the null by standard p-values.
Overstated evidence against the null
- 核心统计量/检验：exp(-Z^2/2)
Function of the p-value

#### Student Prior Bayesian Analysis (Leamer, 1978)

- 检测目标：Multimodality when sample and prior information are in conflict.
- 核心统计量/检验：Product of Student-t functions

#### Symmetric-Descending MBF (SD-MBF) (Harvey, 2017)

- 检测目标：False positives when the alternative distribution is symmetric around the null.
- 核心统计量/检验：-e * p * ln(p)

### Bias Correction

#### Frequentist missing data adjustment (Harvey et al., 2014)

- 检测目标：Publication bias from factors that were tested but never published.
- 核心统计量/检验：Multiple testing adjustment with unobserved trials

#### Missing Data Adjustment (Harvey et al., 2014)

- 检测目标：Publication bias and the 'file drawer' problem of unobserved tests.
- 核心统计量/检验：Hurdle adjustment for unobserved factors

### Bias Correction
Inference Correction

#### Median-unbiased estimation (Andrews & Kasy, 2017)

- 检测目标：Estimate inflation due to truncation or selection.
Bias in point estimates and confidence sets.
Bias in point estimates due to selective publication.
- 核心统计量/检验：Inversion of the conditional CDF of published results.
Quantile-unbiased estimators correcting for known selection probabilities.
Quantile-based correction using the inverse of the publication probability.

### Bound Restriction

#### Theoretical Upper Bound Test (Elliott et al., 2021)

- 检测目标：p-hacking distorting p-curve magnitude
- 核心统计量/检验：Comparison of g(p) and derivatives against analytical bounds

### Classification

#### Factor taxonomy (Harvey et al., 2014)

- 检测目标：Redundancy and overlap among hundreds of documented return predictors.
- 核心统计量/检验：Categorization by type (e.g., accounting, macro)

### Cognitive Bias Modeling

#### Access-Biased Memory Model (Leamer, 1978)

- 检测目标：Selective retrieval of prior information/reasons that only support observed data results.
- 核心统计量/检验：A formal model using a discount factor to adjust posterior distributions for selective memory.

### Correction

#### Optimal Quantile-Unbiased Estimation (Andrews & Kasy, 2017)

- 检测目标：Selection bias in multivariate settings with nuisance parameters.
- 核心统计量/检验：Conditioning on sufficient statistics for nuisance parameters.

### Data Meta-Analysis

#### Factor Taxonomy and Census (Harvey et al., 2014)

- 检测目标：Scale of aggregate data mining in the literature.
- 核心统计量/检验：Cumulative count of 313 published factors

### Data Reduction

#### Sufficient Statistic Summarization (Leamer, 1978)

- 检测目标：The extent to which data evidence is fully captured by key parameters.
- 核心统计量/检验：Likelihood function L(p | r, n) for binomial or normal data.

### Data-instigated Modeling
Inference Type
Hypothesis Generation

#### Sherlock Holmes inference (Leamer, 1978)

- 检测目标：Invalidity of traditional statistical inference when hypotheses are constructed post-data.
The fundamental difference between standard statistical inference and models instigated by the data itself.
Distinction between testing pre-specified hypotheses and data-instigated models.
- 核心统计量/检验：Post-data model construction/Hypothesis searching
Discounting the significance of results found through post-hoc data exploration.
Contrast between statistical and evidence-based discovery.

### Decision Theory

#### Bayesian Posterior Mean Estimation (Leamer, 1978)

- 检测目标：Inadmissibility in point estimation.
- 核心统计量/检验：Expected posterior loss minimization

#### Coherence Principle (Betting Odds) (Leamer, 1978)

- 检测目标：Inconsistent subjective probabilities that allow a 'sure loser' (Dutch book) scenario.
- 核心统计量/检验：Linear system of winnings and stakes

#### Control Rule Shrinkage (Leamer, 1978)

- 检测目标：Excessive instrument use due to parameter uncertainty.
- 核心统计量/检验：Shrinkage factor (1 + t_beta^-2)^-1

### Diagnostic

#### Collinearity Measure (c1i) (Leamer, 1978)

- 检测目标：Sensitivity of a parameter to the existence of other parameters.
- 核心统计量/检验：Ratio of feasible interval length to sample interval: sqrt(Chi-square / Zi^2).

#### Incentive Measure (c2) (Leamer, 1978)

- 检测目标：The potential gain in precision from gathering prior information on other parameters.
- 核心统计量/检验：Ratio of conditional variance to unconditional variance.

#### Lindley's Paradox Analysis (Leamer, 1978)

- 检测目标：Conflict between classical p-values and Bayesian evidence in large T.
- 核心统计量/检验：Comparison of z-scores against sample size growth.

#### Sensitivity Analysis (Fragility Analysis) (Leamer, 1983)

- 检测目标：Fragility of inferences due to minor changes in whimsical assumptions.
- 核心统计量/检验：Mapping of assumption regions to inference regions

### Diagnostics

#### Meta-regression Test (Andrews & Kasy, 2017)

- 检测目标：Presence of selection bias.
- 核心统计量/检验：Regression of z-statistics on the inverse of the standard error.

### Distributional Analysis

#### Counterfactual Distribution Analysis (Brodeur et al., 2019)

- 检测目标：The excess or dearth of p-values by comparing observed distributions to a bias-free counterfactual.
- 核心统计量/检验：Number of misallocated tests and the 'two-humped camel' distribution shape.

#### Z-curve Visualization (Brodeur et al., 2019)

- 检测目标：Bunching of test statistics near significance thresholds (e.g., z=1.96).
- 核心统计量/检验：Kernel density (Epanechnikov) of z-statistics.

### Econometric Modeling

#### Probit/Logit Regression Analysis (Brodeur et al., 2019)

- 检测目标：Sensitivity of statistical significance to the specific causal inference method used.
- 核心统计量/检验：Marginal effects of method dummies (IV, DID, RDD) on a significance indicator.

### Error Identification

#### Specification Search Taxonomy (Leamer, 1978)

- 检测目标：Six types of ad-hoc searches: interpretive, testing, proxy, data-selection, simplification, and postdata construction.
- 核心统计量/检验：Categorization of non-experimental ad-hoc inferences

### Estimation

#### Matrix-Weighted Averaging (Leamer, 1978)

- 检测目标：The optimal pooling of information from different samples or priors.
- 核心统计量/检验：Matrix-weighted mean of estimates

#### Misspecification Uncertainty Accounting (Leamer, 1983)

- 检测目标：Bias from left-out variables or non-randomized natural experiments.
- 核心统计量/检验：Var = S + M (Sampling + Misspecification variance)

#### Moment-based (GMM) Estimation (Andrews & Kasy, 2017)

- 检测目标：Publication probabilities without parametric assumptions on true effects.
- 核心统计量/检验：Moment restrictions comparing study pairs with different noise levels.

#### Parametric Step-Function Model (Andrews & Kasy, 2017)

- 检测目标：Relative publication probabilities across significance thresholds.
- 核心统计量/检验：Maximum likelihood estimation of publication weights (beta coefficients).

#### Truncated Maximum Likelihood (Andrews & Kasy, 2017)

- 检测目标：Latent effect distribution parameters.
- 核心统计量/检验：Likelihood function adjusted for the conditional probability of publication.

### Estimation Methodology

#### Matrix Weighted Averages (Leamer, 1978)

- 检测目标：Sensitivity of point estimates to weights assigned to prior vs. data.
- 核心统计量/检验：Combination of prior precision and sample information.

### Excess Mass Estimation

#### Counterfactual Distribution Comparison (Brodeur et al., 2019)

- 检测目标：Quantifies misallocated (missing or surplus) p-values compared to a null.
- 核心统计量/检验：Student-t (df=1) and Cauchy (0, 0.5) as input distributions.

### FDP Control

#### Lehmann and Romano (2005) (Harvey et al., 2014)

- 检测目标：Controls the probability that the realized False Discovery Proportion exceeds a threshold.
- 核心统计量/检验：P(FDP > γ) ≤ α

### FDR Control

#### Benjamini, Hochberg, and Yekutieli (BHY) (Harvey et al., 2014)

- 检测目标：Controls the expected proportion of false discoveries among all rejections.
- 核心统计量/检验：Adjusted p-value threshold under dependency

### First-stage Diagnostics

#### F-statistic Distribution Analysis (Brodeur et al., 2019)

- 检测目标：P-hacking in IV first-stage results, specifically bunching above 10.
- 核心统计量/检验：Histogram and density of first-stage F-statistics.

### Forecasting

#### Forward projection of hurdles (Harvey et al., 2014)

- 检测目标：Future requirements for statistical significance based on factor production rates.
- 核心统计量/检验：Projected minimum t-ratios through 2032

### Fragility Assessment

#### Curve Décolletage (Sensitivity Analysis) (Leamer, 1978)

- 检测目标：Fragility of estimates across a range of prior distributions.
- 核心统计量/检验：Contract curve between prior and data

### Frequentist Inference

#### Multiple Hypothesis Testing Framework (Harvey et al., 2014)

- 检测目标：Spurious significance in a multi-factor search environment.
- 核心统计量/检验：Adjusted p-values and t-ratios

### FWER Control

#### Bonferroni adjustment (Harvey et al., 2014)

- 检测目标：Controls the probability of at least one false discovery.
- 核心统计量/检验：p-value threshold divided by the number of tests

#### Holm's Method (Harvey et al., 2014)

- 检测目标：Sequentially rejects hypotheses to control FWER with more power than Bonferroni.
- 核心统计量/检验：Step-down p-value comparison

### Heuristic

#### Sherlock Holmes Inference (Leamer, 1983)

- 检测目标：Data anomalies that suggest the model horizon needs extension.
- 核心统计量/检验：Study of data anomalies

### Identification

#### Consensus Analysis (Leamer, 1978)

- 检测目标：Whether sample data is sufficient to force observers with different priors to a common posterior.
- 核心统计量/检验：Equivalence of public informativeness and identifiability

#### Identification Problem Assessment (Leamer, 1983)

- 检测目标：Observational equivalence between competing hypotheses in non-experimental data.
- 核心统计量/检验：Comparison of conditional means across different confounding structures.

#### Meta-study Approach (Andrews & Kasy, 2017)

- 检测目标：Publication bias using only published results.
- 核心统计量/检验：Deviations in the distribution of estimates across different standard error levels.

#### Meta-study Variance Approach (Andrews & Kasy, 2017)

- 检测目标：Selectivity by comparing distributions across study precisions.
- 核心统计量/检验：Deviations from the assumption that high-variance studies are noised-up versions of low-variance studies.

### Identification Analysis

#### Likelihood Concentration/Marginalization (Leamer, 1978)

- 检测目标：Singularities in likelihood functions of under-identified models.
- 核心统计量/检验：Comparison of concentrated vs. marginal likelihoods

### Inference

#### Bonferroni-corrected Confidence Sets (Andrews & Kasy, 2017)

- 检测目标：Coverage distortions when selection probabilities are estimated.
- 核心统计量/检验：Adjustment of confidence intervals to account for estimation error in selection models.

### Likelihood Inference

#### Supporting Hyperplanes Method (Leamer, 1978)

- 检测目标：Uncertainty intervals for linear combinations of parameters.
- 核心统计量/检验：Confidence ellipsoids

### Local Discontinuity Test
Statistical Significance Test

#### Caliper Test (Brodeur et al., 2019)

- 检测目标：Bunching within a narrow band (±0.5) around arbitrary significance thresholds.
Bunching or discontinuities of p-values within narrow bands around arbitrary thresholds.
- 核心统计量/检验：Proportion of marginally significant vs. marginally insignificant results.
Proportion of tests in the [1.46, 2.46] z-statistic window.

### Measurement Error Detection

#### Reverse Regression (Leamer, 1978)

- 检测目标：Bounds for coefficients in the presence of measurement error.
- 核心统计量/检验：Inverse of the regression of x on Y

### Meta-analysis

#### Joint p-hacking and Publication Bias Test (Elliott et al., 2021)

- 检测目标：Combined selective reporting and data manipulation
- 核心统计量/检验：Testing the null set of distributions implied by no hacking/bias

### Meta-statistics

#### Interpretive Search Rules (Leamer, 1978)

- 检测目标：Selective reporting based on 'right signs' or significance levels in sequential modeling.
- 核心统计量/检验：Weighted average of points on the feasible ellipsoid

### Metastatistics

#### Metastatistical Discounting (Leamer, 1978)

- 检测目标：Double-counting of evidence when the same data instigates the model and tests it.
- 核心统计量/检验：Formal discounting of data evidence

### Metastatistics
Bayesian Diagnostics
Sensitivity Analysis

#### Information Contract Curve (Leamer, 1978)

- 检测目标：Sensitivity of posterior modes to prior labeling functions.
The path of posterior modes as the relative weight of prior and sample information varies.
Range of estimates produced by varying prior beliefs and specifications.
- 核心统计量/检验：Locus of tangencies between prior and likelihood isodensity surfaces
Locus of matrix-weighted averages between prior and sample means
Locus of posterior means across prior precision matrices.

### Metastatistics / Pooling Search

#### Bayesian Pooling (Shrinkage) (Leamer, 1978)

- 检测目标：Spurious results (clairvoyant paradox) arising from independent priors on similar processes.
- 核心统计量/检验：Matrix-weighted average of OLS estimates and grand mean

### Model Averaging

#### Rotation Invariant Average Regressions (Leamer, 1978)

- 检测目标：Invariance of weighted averages to rotations of the parameter space.
- 核心统计量/检验：Weights based on determinants of sub-matrices of X'X.

### Model Comparison

#### Bayes Factor for Model Selection (Leamer, 1978)

- 检测目标：Overfitting and spurious results in specification searches.
- 核心统计量/检验：B = (ESS0/ESS1)^(T/2) * T^((k0-k1)/2)

### Model Selection

#### Adjusted R-squared Selection (Leamer, 1978)

- 检测目标：Unwarranted preference for models with many explanatory variables.
- 核心统计量/检验：Adjusted R-squared (R-bar squared)

#### Simplification Analysis (Leamer, 1978)

- 检测目标：Loss of prediction/control accuracy when omitting variables.
- 核心统计量/检验：Chi-square compared to (T + k)

#### Specification Search Analysis (Leamer, 1983)

- 检测目标：Bias introduced by searching for significant results through variable selection.
- 核心统计量/检验：Analysis of the 'bias parameters' alpha* and beta* from left-out variables.

#### Theory-based hurdle differentiation (Harvey et al., 2014)

- 检测目标：Over-penalization of theoretically grounded factors vs. empirical mining.
- 核心统计量/检验：Lower t-hurdle for theory-derived factors

### Model Selection Analysis
Specification Search Analysis
Methodology

#### Specimetrics (Leamer, 1978)

- 检测目标：Inferences drawn when the data-generating mechanism is ambiguous or searched.
Processes leading to specific model choices and inferences from ambiguous data-generating mechanisms.
Describes processes of choosing one model specification over another.
- 核心统计量/检验：Process analysis of model specification choices
Identification of inferences properly drawn from searched data sets.
Model choice analysis
Evaluation of model specification processes.

### Model Selection Reporting

#### Overall R-squared (Grand R^2) (Leamer, 1978)

- 检测目标：Over-optimism in reported fit resulting from multiple model comparisons.
- 核心统计量/检验：1 - [ (Sum of (1 - Ri^2)^(-T/2)) / M ]^(-2/T)

### Model Specification

#### Contamination Vector (Experimental Bias) Adjustment (Leamer, 1978)

- 检测目标：Spurious precision caused by ignoring potential omitted variables or measurement errors.
- 核心统计量/检验：beta^c (Contamination vector)

#### Differentiated Hurdles (Harvey et al., 2014)

- 检测目标：Varying credibility between theory-based and empirical factors.
- 核心统计量/检验：Higher t-hurdle for empirical exercises

#### Presimplification Analysis (Leamer, 1978)

- 检测目标：Misspecification in simplified working hypotheses versus more complex world views.
- 核心统计量/检验：Marginalized likelihood

### Model Uncertainty

#### Bayesian Model Averaging (BMA) (Leamer, 1978)

- 检测目标：Underestimation of parameter variance due to model searching.
- 核心统计量/检验：Posterior mean as a mixture of individual model estimates.

### Multiple Comparison

#### Bonferroni and Holm Adjustments (Harvey et al., 2014)

- 检测目标：Family-wise error rate inflation from multiple tests.
- 核心统计量/检验：Size-adjusted p-value thresholds

### Multiple Testing Adjustment

#### Increased t-statistic thresholds (Harvey, 2017)

- 检测目标：False positives in factor discovery
- 核心统计量/检验：t > 3.0

#### t-statistic Thresholding (t > 3.0) (Harvey, 2017)

- 检测目标：False discoveries resulting from extensive factor mining and data dredging.
- 核心统计量/检验：t-statistic > 3.0

### Nonparametric Identification

#### Meta-study-based Identification (Andrews & Kasy, 2017)

- 检测目标：Publication bias using variance-estimate distributions.
- 核心统计量/检验：Deviations from the predicted distribution across studies with different variances.

#### Replication-based Identification (Andrews & Kasy, 2017)

- 检测目标：Selective publication based on initial results.
- 核心统计量/检验：Asymmetry in the joint distribution of initial and replication estimates.

#### Systematic Replication Approach (Andrews & Kasy, 2017)

- 检测目标：Conditional publication probabilities $p(z)$.
Nonparametric conditional publication probabilities.
- 核心统计量/检验：Asymmetries in the joint distribution of initial and replication estimates.
Asymmetry in the joint distribution of original and replication estimates.

### P-value Adjustment

#### BHY Procedure (Harvey et al., 2014)

- 检测目标：False discovery rate under dependency among tests.
- 核心统计量/检验：Benjamini-Hochberg-Yekutieli adjusted p-values

### Parameter Estimation

#### Bayesian Normal-Gamma Updating (Leamer, 1978)

- 检测目标：Conflict between prior information and sample evidence through posterior variance changes.
- 核心统计量/检验：Posterior variance (v**) which increases when sample and prior means conflict.

### Parameter Interpretation

#### Matrix-Weighted Average Bounds (Leamer, 1978)

- 检测目标：Whether multi-parameter posterior estimates lie within the algebraic range of prior and sample locations.
- 核心统计量/检验：Orthotope or parallelotope constraints on transformed coordinates

### Parametric Estimation

#### Step Function Selection Model (Andrews & Kasy, 2017)

- 检测目标：Discontinuities in publication probability at p-value thresholds.
Relative publication probabilities across z-statistic cutoffs.
- 核心统计量/检验：Maximum likelihood estimation of publication probabilities across z-score bins.
Maximum likelihood estimation of $\beta$ parameters for significance brackets.

### Post-hoc Modeling

#### Sherlock Holmes Inference (Data-Instigated) (Leamer, 1978)

- 检测目标：Hypotheses generated from data patterns rather than prior theory.
- 核心统计量/检验：Bayes factors for data-instigated hypotheses

### Predictive Analysis

#### Forward Projection of Cutoffs (Harvey et al., 2014)

- 检测目标：Future data mining risks if factor production continues.
- 核心统计量/检验：Projected minimum t-ratios (e.g., through 2032)

### Prior Probability Assessment

#### Economic plausibility weighting (Harvey, 2017)

- 检测目标：Spurious results from unlikely hypotheses
- 核心统计量/检验：Prior odds

### Prior Specification

#### Exchangeable Normal Process Prior (Leamer, 1978)

- 检测目标：Overestimated knowledge of future systems in constant-parameter models.
- 核心统计量/检验：Precision matrix (singular if mean info is weak)

#### Stable Estimation (Diffuse Priors) (Leamer, 1978)

- 检测目标：Situations where sample information dominates prior information.
- 核心统计量/检验：Comparison of posteriors derived from improper noninformative vs. proper priors.

### Probability Adjustment

#### Bayesianized p-values (Harvey, 2017)

- 检测目标：The posterior probability that the null hypothesis is true given the data.
- 核心统计量/检验：MBF * Prior Odds / (1 + MBF * Prior Odds)

### Publication Bias Detection

#### Meta-analysis of t-statistic distributions (Harvey, 2017)

- 检测目标：Missing non-significant results and file-drawer effects
- 核心统计量/检验：Distribution of reported t-statistics

#### t-statistic Distribution Meta-analysis (Harvey, 2017)

- 检测目标：Selective submission and publication of only 'significant' results.
- 核心统计量/检验：Frequencies of reported t-statistics across specific ranges

### Regularization

#### Ridge Regression (Leamer, 1978)

- 检测目标：Unstable estimates caused by multicollinearity in nonexperimental data.
- 核心统计量/检验：cI penalty term added to (X'X) matrix

### Resampling Method

#### Simulation Framework (Harvey et al., 2014)

- 检测目标：Sampling and estimation uncertainty in benchmark cutoffs.
- 核心统计量/检验：Median benchmark t-ratios

### Research Protocol

#### Registered Reports (Harvey, 2017)

- 检测目标：Selective reporting and p-hacking by fixing the design before data collection.
- 核心统计量/检验：Pre-analysis peer review

### Researcher Bias

#### Revealed Prior Analysis (Leamer, 1978)

- 检测目标：The implicit prior beliefs of a researcher evidenced by their model choices.
- 核心统计量/检验：Information contract curve

### Robustness Check

#### Orthant Preservation Analysis (Leamer, 1978)

- 检测目标：Sign reversals when imposing linear constraints.
- 核心统计量/检验：t-statistic bounds for constrained estimates

### Robustness Test

#### Extreme Bounds Analysis (Leamer, 1983)

- 检测目标：Maximum and minimum possible estimates given a set of doubtful variables.
- 核心统计量/检验：Extreme estimates (Minimum/Maximum)

### Robustness Testing

#### Global/Local Sensitivity Analysis (Leamer, 1978)

- 检测目标：Fragility of posterior results to changes in the researcher's prior information or specification choices.
- 核心统计量/检验：Mapping from prior to posterior distribution

### Robustness Testing
Robustness Check

#### Sensitivity Analysis (Leamer, 1978)

- 检测目标：Probabilistic assumptions that crucially determine the nature of inferences.
- 核心统计量/检验：Comparison of outcomes across minor probabilistic changes
Variation in posterior distributions under minor changes in prior density.

### Sampling Theory

#### Noninformative Stopping Rules Analysis (Leamer, 1978)

- 检测目标：False concerns about bias resulting from data-dependent sampling or search rules.
- 核心统计量/检验：Likelihood Principle

### Sensitivity Analysis

#### Extreme Bounds Analysis (Conceptual) (Leamer, 1983)

- 检测目标：Fragility of econometric inferences to changes in model specification.
- 核心统计量/检验：The range of estimated coefficients across different sets of prior assumptions.

#### Global Sensitivity Analysis (Leamer, 1978)

- 检测目标：The mapping between classes of prior distributions and classes of posterior locations.
- 核心统计量/检验：Correspondence mapping between prior and posterior covariance classes

#### Rectangle Test (Leamer, 1978)

- 检测目标：Collinearity-induced misinterpretation where estimates fall outside the prior-sample range.
- 核心统计量/检验：Check if the contract curve lies in the rectangular solid with diagonal [b*, b].

### Sequential Testing

#### Pretest Estimation Correction (Leamer, 1978)

- 检测目标：Bias introduced by choosing models based on preliminary significance tests.
- 核心统计量/检验：Mean squared error of pretest estimators

### Shape Restriction

#### Complete monotonicity test (Elliott et al., 2021)

- 检测目标：p-hacking in t-tests (specifically specification search)
- 核心统计量/检验：(-1)^k * g^{(k)}(p) >= 0 for all k

#### Non-increasingness test (Elliott et al., 2021)

- 检测目标：General p-hacking and publication bias
- 核心统计量/检验：Checking if the p-value density g(p) is non-increasing

### Shrinkage Estimation

#### Stein-James Estimator (Leamer, 1978)

- 检测目标：Inadmissibility and inefficiency of OLS in models with three or more parameters.
- 核心统计量/检验：Shrinkage factor (1 - (k-2)sigma^2 / Y'Y)

### Significance Testing
Benchmark Modeling
Time-series Adjustment

#### Historical significance cutoffs (Harvey et al., 2014)

- 检测目标：The erosion of statistical power over time as more factors are tested.
Significance inflation over time due to factor production.
Detects how significance hurdles must evolve as the number of tested factors increases.
- 核心统计量/检验：Time-varying t-ratio threshold
Time-varying t-ratio hurdles
Projected t-ratio hurdles (e.g., t > 3.0)

### Social Process Analysis

#### Advocacy Ability Scoring (Leamer, 1978)

- 检测目标：Bias in professional opinion caused by the rhetorical skill of the researcher rather than data evidence.
- 核心统计量/检验：Comparison of briefs with identical facts but different advocacy strengths.

### Specification Search

#### Error-Components Model (Leamer, 1978)

- 检测目标：Unobserved cross-sectional or temporal heterogeneity in the regression constant.
- 核心统计量/检验：Additive variance components (alpha_i, gamma_t, epsilon_it)

#### Hyperbolically Uniform Search (Leamer, 1978)

- 检测目标：Ad hoc variable omission based on coordinate systems.
- 核心统计量/检验：Sequential t-statistics

#### Interpretive Search Analysis (Leamer, 1978)

- 检测目标：Selection of models based on the plausibility of coefficient signs.
- 核心统计量/检验：Coefficient signs and prior-data consistency

#### Interpretive Searches (Leamer, 1978)

- 检测目标：Search for specifications that yield interpretable or desired coefficients.
- 核心统计量/检验：Comparison of posterior distributions across specifications.

#### Principal Component Regression (Interpretive) (Leamer, 1978)

- 检测目标：Implicit prior information in variable reduction.
- 核心统计量/检验：Eigenvalues of the design matrix X'X

#### Proxy search (Leamer, 1978)

- 检测目标：Best observable link to unobservable theoretical constructs.
- 核心统计量/检验：Comparison of ESS across proxy specifications

#### Sherlock Holmes Inference (Post-data Model Construction) (Leamer, 1978)

- 检测目标：Over-confidence in hypotheses that were only formulated after seeing the data.
- 核心统计量/检验：Formal discounting of evidence

#### Theil's R-bar-squared Criterion (Leamer, 1978)

- 检测目标：Whether omitting a variable improves the model's estimated variance.
- 核心统计量/检验：t-statistic > 1

#### Time-Varying Parameter Model (Leamer, 1978)

- 检测目标：Structural shifts that render older data points less relevant for current inference.
- 核心统计量/检验：Kalman filter recursive relationships

### Specification Search Analysis

#### SEARCH (Extreme Hypotheses) (Leamer, 1978)

- 检测目标：The sensitivity of focus coefficients to the inclusion of 'doubtful' control variables.
- 核心统计量/检验：Extreme values of coefficients over all possible linear constraints

### Specification Search Detection

#### Marginal Likelihood Comparison (Leamer, 1978)

- 检测目标：Models that perform well only at specific parameter points.
- 核心统计量/检验：Weighted-average likelihood (index) vs. Maximum Likelihood.

### Specification Search Evaluation

#### Pretest Estimator Analysis (Leamer, 1978)

- 检测目标：Bias and Mean Squared Error (MSE) inflation caused by choosing models based on t-test results.
- 核心统计量/检验：MSE comparison of pretest vs. OLS estimators

### Specification Search Strategy

#### Sequential t-statistic Omission (Leamer, 1978)

- 检测目标：Stability of coefficient signs during variable deletion.
- 核心统计量/检验：Sequence of sequentially computed t-statistics.

### Specification Test

#### Latent Selection Specification Test (Andrews & Kasy, 2017)

- 检测目标：Dependence of publication on true effect size $\Theta^*$.
- 核心统计量/检验：Test of $H_0: \gamma_p = 0$ in a model nesting the baseline selection rule.

### Statistical Correction

#### Discount Factor for Memory Access (Leamer, 1978)

- 检测目标：Overconfidence resulting from remembering only experiments similar to current findings.
- 核心统计量/检验：d = [1 - (N+1)f_beta(p | k+1, N+2)]^(T-s).

### Statistical Inference

#### Discrete Proportions Difference Test (Elliott et al., 2021)

- 检测目标：Violations of monotonicity and bounds in binned data
- 核心统计量/检验：H0: A * pi_{-J} <= b (using differencing matrices)

### Statistical Inference
Frequentist Inference

#### Multiple testing framework (Harvey et al., 2014)

- 检测目标：False discoveries resulting from extensive data mining in factor research.
Data mining and selection bias in factor discovery.
- 核心统计量/检验：Adjusted p-values and t-ratio hurdles
Adjusted t-ratio benchmarks

### Statistical Learning

#### Bayesian Inference (Three-Phase) (Leamer, 1978)

- 检测目标：Updates uncertainty about parameters by combining prior beliefs with sample data.
- 核心统计量/检验：Bayes' Rule [f(θ|x) = f(x|θ)f(θ)/f(x)]

### Statistical Modeling

#### Truncated Exponential Distribution (Harvey et al., 2014)

- 检测目标：Publication bias and missing/unpublished test results.
- 核心统计量/检验：Estimated total trials (M) and population mean (lambda)

### Statistical Rigor

#### Adjustment for multiple tests (Harvey, 2017)

- 检测目标：Data mining and specification searching
- 核心统计量/检验：Corrected p-value thresholds

### Statistical Significance Likelihood

#### Probit Regression Analysis (Brodeur et al., 2019)

- 检测目标：Differential probability of reporting significant results by research method.
- 核心统计量/检验：Marginal effects on a dummy for statistical significance (p < 0.05).

### Testing Correction

#### Sample Size-Dependent Significance Levels (Leamer, 1978)

- 检测目标：False significance in large samples due to fixed alpha levels.
- 核心统计量/检验：Alpha as a decreasing function of sample size T.

### Theoretical Framework

#### Metastatistics / Specimetrics (Leamer, 1978)

- 检测目标：How researcher motives and specification searches influence the choice of models and data.
- 核心统计量/检验：Analysis of the gap between ideal statistical theory and actual practice.

#### Metastatistics/Specimetrics (Leamer, 1978)

- 检测目标：Influence of researcher motives/opinions on model choice.
- 核心统计量/检验：Analysis of the mapping from priors to posteriors.

### Theoretical Framework
Theory of Inference
Research Methodology

#### Metastatistics (Leamer, 1978)

- 检测目标：How researcher motives and opinions influence the choice of model and data.
Influence of researcher motives/opinions on model choice.
Influence of researcher motives and opinions on model and data choice.
- 核心统计量/检验：Analysis of researcher behavior and decision-making
Analysis of the gap between econometric theory and practice.
Mapping from priors to posteriors
Analysis of the gap between ideal and actual inference.

### Variable Selection

#### Contamination Parameter Analysis (Leamer, 1978)

- 检测目标：The influence of left-out variables on the commitment to a regression result.
- 核心统计量/检验：Subjective priors on a contamination parameter beta^c.

#### Proxy Search Evaluation (Leamer, 1978)

- 检测目标：Searching for proxy variables to achieve significant results.
- 核心统计量/检验：Change in R-squared or coefficient stability

### 未分类

#### Analysis of z-statistics distribution (Brodeur et al., 2013)


#### Bayes Factor Comparison (Leamer, 1978)


#### Bayesian decision-theoretic simplification (Leamer, 1978)


#### Bayesian inference (Leamer, 1978)


#### Bayesian Pooling of Samples (Leamer, 1978)


#### Benford's Law test on digits (Brodeur et al., 2013)


#### Benford’s Law tests on coefficients and standard errors (Brodeur et al., 2013)


#### Benjamini, Hochberg and Yekutieli (BHY) Adjustment (Harvey et al., 2014)


#### Benjamini-Hochberg-Yekutieli (BHY) procedure (Harvey et al., 2014)


#### Binomial test (Elliott et al., 2021)


#### Bonferroni's Adjustment (Harvey et al., 2014)


#### Bonferroni-corrected confidence intervals (Andrews & Kasy, 2017)


#### Calculation of misallocation residuals (Brodeur et al., 2013)


#### Caliper tests (Brodeur et al., 2019)


#### Causally constrained conditional prediction (Leamer, 1978)


#### Comparison with theoretical distributions (Brodeur et al., 2013)


#### Conditional chi-squared test (Cox-Shi) (Elliott et al., 2021)


#### Control-simplification analysis (Leamer, 1978)


#### Counterfactual distribution comparison (Brodeur et al., 2013)


#### Curve Décolletage (Dickey's Curve) (Leamer, 1978)


#### Data-selection search (Leamer, 1978)


#### De-rounding of reported statistics (Brodeur et al., 2013)


#### Decomposition by article characteristics (Brodeur et al., 2013)


#### Density discontinuity test (Elliott et al., 2021)


#### Direct Regression Estimate (Leamer, 1978)


#### Distributional symmetry testing (Andrews & Kasy, 2017)


#### Double-Student Posterior Distribution (Leamer, 1978)


#### Errors-in-Variables Bound (Leamer, 1978)


#### Factor Analysis (Leamer, 1978)


#### False Discovery Proportion (FDP) control (Harvey et al., 2014)


#### False Discovery Rate (FDR) control (Harvey et al., 2014)


#### False discovery rate calibration (Harvey, 2017)


#### Family-wise Error Rate (FWER) control (Harvey et al., 2014)


#### Fisher's test (Elliott et al., 2021)


#### Generalized Least Squares (Leamer, 1978)


#### GMM-based parameter estimation (Harvey et al., 2014)


#### Holm adjustment (Harvey et al., 2014)


#### Holm's Adjustment (Harvey et al., 2014)


#### Hypothesis-testing search (Leamer, 1978)


#### Identification of p-value camel shape (Brodeur et al., 2013)


#### Instrumental Variables (Leamer, 1978)


#### Instrumental Variables Estimator (Leamer, 1978)


#### Interpretive search (Leamer, 1978)


#### Inverse-weighting schemes (Brodeur et al., 2019)


#### Kernel density estimation (Brodeur et al., 2013); (Brodeur et al., 2019)


#### Least Concave Majorant (LCM) test (Elliott et al., 2021)


#### Lindley’s conditional prediction analysis (Leamer, 1978)


#### Marginal Likelihood Integration (Leamer, 1978)


#### Matrix-weighted Average (Leamer, 1978)


#### Maximum Likelihood Estimation (Leamer, 1978)


#### Meta-analysis (Harvey, 2017)


#### Meta-regression analysis (Andrews & Kasy, 2017)


#### Minimum Bayes Factor (Harvey, 2017)


#### Multiple test adjustment (Harvey, 2017)


#### Non-parametric estimation of selection and inflation (Brodeur et al., 2013)


#### Non-parametric PAVA estimation (Brodeur et al., 2013)


#### Nonparametric identification via meta-studies (Andrews & Kasy, 2017)


#### Nonparametric identification via replication studies (Andrews & Kasy, 2017)


#### Normal-Wishart Multivariate Sampling (Leamer, 1978)


#### Out-of-sample validation (Harvey et al., 2014)


#### P-curve analysis (Brodeur et al., 2019); (Harvey, 2017)


#### Parametric estimation of selection and inflation (Brodeur et al., 2013)


#### Parametric selection function modeling (Brodeur et al., 2013)


#### Postdata model construction (Leamer, 1978)


#### Proxy Variable Search (Leamer, 1978)


#### Quantile-unbiased estimation (Andrews & Kasy, 2017)


#### Regression analysis for threshold discontinuities (Brodeur et al., 2013)


#### Reverse Regression Estimate (Leamer, 1978)


#### Seemingly Unrelated Regressions (Leamer, 1978)


#### Simplification search (Leamer, 1978)


#### Simplification searches (Leamer, 1978)


#### Simulation-based missing data adjustment (Harvey et al., 2014)


#### Standardized beta coefficients (Leamer, 1978)


#### Structural Likelihood Maximization (Leamer, 1978)


#### Structural modeling of t-statistic distributions (Harvey et al., 2014)


#### Student-Gamma Prior (Leamer, 1978)


#### Sub-sample heterogeneity analysis (Brodeur et al., 2013)


#### Sweeping out the Means (Leamer, 1978)


#### T-statistic threshold adjustment (Harvey, 2017)


#### Truncated likelihood approach (Andrews & Kasy, 2017)


#### Upper bounds test (Elliott et al., 2021)


#### Weighted distribution analysis (Brodeur et al., 2013)


#### Z-statistics distribution analysis (Brodeur et al., 2013)


## 逐篇精华（按文献）

### ... and the Cross-Section of Expected Returns

- 引用：(Harvey, Liu, & Zhu, 2014)
- 作者：Campbell R. Harvey, Yan Liu, Heqing Zhu
- 年份：2014
- 载体/系列：National Bureau of Economic Research Working Paper 20592
- 链接：http://www.nber.org/papers/w20592
- 关键词：Data mining, Multiple testing, Asset pricing, Factor models, Publication bias, p-hacking, False discovery rate, Factor zoo, P-hacking, T-statistic, data mining, multiple testing, publication bias, factor zoo, cross-section of returns, false discovery rate, Multiple Testing, Data Mining, Asset Pricing, False Discovery Rate, Factor Zoo

**核心要点**
- Standard t-ratios of 2.0 are insufficient due to extensive data mining in factor research.
- Newly discovered factors should clear a higher hurdle, typically a t-ratio greater than 3.0.
- Most claimed research findings in financial economics are likely false due to multiple testing.
- The framework provides historical significance cutoffs that evolve as more factors are tested.
- The method accounts for correlation among tests and missing data from unpublished null results.
- Traditional t-statistic thresholds are insufficient due to data mining and the factor zoo.
- A higher hurdle of at least t-ratio > 3.0 is recommended for current factor research.
- Many published findings in financial economics are likely false discoveries under multiple testing adjustments.
- Significance cutoffs have increased over time as the number of tested factors has grown.
- Effective adjustments must account for test correlations and the unobservability of failed factor trials.
- A standard t-ratio of 2.0 is no longer sufficient for claiming a new factor is significant.
- A newly discovered factor should clear a much higher hurdle, typically a t-ratio greater than 3.0.
- Significance thresholds must increase over time as the 'factor zoo' continues to expand.
- The framework accounts for correlations between tests and the 'file drawer' problem of unpublished results.
- Theoretical foundations justify a lower statistical hurdle compared to purely empirical discoveries.
- The standard t-ratio threshold of 2.0 is inadequate due to extensive data mining.
- A newly discovered factor now requires a t-ratio greater than 3.0 to be significant.
- Most claimed research findings in financial economics are likely false.
- Significant hurdles must rise over time as more factors are tested.
- Accounting for correlation among tests and unobserved trials is essential.
- Theory-derived factors should have lower hurdles than purely empirical ones.
- Conventional t-ratio hurdles of 2.0 are insufficient due to extensive factor data mining.
- A newly discovered factor should meet a t-ratio hurdle higher than 3.0 to be significant.
- Many claimed research findings in financial economics are likely false due to multiple testing.
- Publication bias creates a 'missing data' problem where only significant results are observed.
- Appropriate significance cutoffs must increase over time as more factors are tested.
- The BHY adjustment is preferred over Bonferroni when tests are correlated.
- Conventional t-ratios (t > 2.0) are insufficient due to extensive data mining of over 300 factors.
- A newly discovered factor today should clear a higher hurdle, typically a t-ratio greater than 3.0.
- Multiple testing frameworks must account for correlations between factors and unobserved (missing) tests.
- Most claimed research findings in financial economics are likely false due to selection bias and p-hacking.
- Theoretical factors should potentially face lower hurdles than purely empirical, data-mined factors.
- Significance thresholds should increase over time as the 'factor production' in the literature continues.

**该文献提供/强调的方法**
- Bayesian Hierarchical Framework（Bayesian Inference）
- Benjamini, Hochberg and Yekutieli (BHY) Adjustment
- Benjamini, Hochberg, and Yekutieli (BHY)（FDR Control）
- Benjamini-Hochberg-Yekutieli (BHY) procedure
- BHY Procedure（P-value Adjustment）
- Bonferroni adjustment（FWER Control）
- Bonferroni and Holm Adjustments（Multiple Comparison）
- Bonferroni's Adjustment
- Differentiated Hurdles（Model Specification）
- Factor taxonomy（Classification）
- Factor Taxonomy and Census（Data Meta-Analysis）
- False Discovery Proportion (FDP) control
- False Discovery Rate (FDR) control
- Family-wise Error Rate (FWER) control
- Forward Projection of Cutoffs（Predictive Analysis）
- Forward projection of hurdles（Forecasting）
- Frequentist missing data adjustment（Bias Correction）
- GMM-based parameter estimation
- Historical significance cutoffs（Significance Testing
Benchmark Modeling
Time-series Adjustment）
- Holm adjustment
- Holm's Adjustment
- Holm's Method（FWER Control）
- Lehmann and Romano (2005)（FDP Control）
- Missing Data Adjustment（Bias Correction）
- Multiple Hypothesis Testing Framework（Frequentist Inference）
- Multiple testing framework（Statistical Inference
Frequentist Inference）
- Out-of-sample validation
- Simulation Framework（Resampling Method）
- Simulation-based missing data adjustment
- Structural modeling of t-statistic distributions
- Theory-based hurdle differentiation（Model Selection）
- Truncated Exponential Distribution（Statistical Modeling）

### Detecting p-hacking

- 引用：Elliott et al. (2021)
- 作者：Graham Elliott, Nikolay Kudrin, Kaspar Wüthrich
- 年份：2021
- 载体/系列：Working Paper / Journal Article
- 关键词：p-values, p-curve, complete monotonicity, publication bias, p-hacking, t-tests, specification search

**核心要点**
- Under no p-hacking, p-curves are generally non-increasing and continuous for most statistical tests.
- For t-tests, p-curves exhibit complete monotonicity and specific upper bounds on density and derivatives.
- Proposed tests can detect p-hacking even when it fails to induce an increasing p-curve.
- Data rounding often creates spurious evidence of p-hacking; de-rounding is essential for valid inference.
- The tests serve as joint tests for p-hacking and publication bias in multiple study settings.
- p-curves are theoretically non-increasing and continuous under the null of no p-hacking.
- For t-tests, p-curves are completely monotone, providing higher-order testable restrictions.
- Analytical upper bounds on p-curve derivatives can detect hacking that monotonicity tests miss.
- The proposed tests are more powerful than standard Simonsohn p-curve tests.
- The framework allows joint testing of publication bias and p-hacking.
- Results hold for both one-sided and two-sided t-tests under general effect distributions.

**该文献提供/强调的方法**
- Binomial test
- Complete monotonicity test（Shape Restriction）
- Conditional chi-squared test (Cox-Shi)
- Density discontinuity test
- Discrete Proportions Difference Test（Statistical Inference）
- Fisher's test
- Joint p-hacking and Publication Bias Test（Meta-analysis）
- Least Concave Majorant (LCM) test
- Non-increasingness test（Shape Restriction）
- Theoretical Upper Bound Test（Bound Restriction）
- Upper bounds test

### Let's Take the Con out of Econometrics

- 引用：Leamer (1983)
- 作者：Edward E. Leamer
- 年份：1983
- 载体/系列：The American Economic Review
- 链接：http://links.jstor.org/sici?sici=0002-8282%28198303%2973%3A1%3C31%3ALTTCOO%3E2.0.CO%3B2-R
- 关键词：Fragility, Sensitivity Analysis, Data Mining, Specification Search, Misspecification, Prior Distributions, specification search, sensitivity analysis, extreme bounds, fragility, nonexperimental data, prior beliefs

**核心要点**
- Traditional econometric inference is invalidated by specification searches and 'data mining'.
- Results are often 'fragile', meaning they can be reversed by minor changes in model assumptions.
- The 'mapping' from assumptions to inferences is the only credible message in non-experimental data.
- Objectivity is a myth; sampling and prior distributions are opinions, not facts.
- Researchers should perform systematic sensitivity analyses rather than reporting a single 'best' model.
- Non-experimental data requires a fixed misspecification uncertainty (M) that doesn't vanish with sample size.
- Econometric results are often fragile and highly dependent on whimsical modeling assumptions.
- Randomization only ensures unbiasedness on average; it does not guarantee adequate mixing in any single sample.
- The 'con' is the pretense that non-experimental data analysis mirrors controlled scientific experiments.
- Applied econometrics is frequently a 'specification search' rather than a test of pre-specified hypotheses.
- We must acknowledge that bias parameters (from omitted variables or measurement error) are rarely exactly zero.
- Researchers should report the sensitivity of their findings to changes in the model's underlying assumptions.

**该文献提供/强调的方法**
- Bayesian Sensitivity Analysis（Bayesian Econometrics）
- Extreme Bounds Analysis（Robustness Test）
- Extreme Bounds Analysis (Conceptual)（Sensitivity Analysis）
- Identification Problem Assessment（Identification）
- Misspecification Uncertainty Accounting（Estimation）
- Prior Specification Grouping（Bayesian）
- Sensitivity Analysis (Fragility Analysis)（Diagnostic）
- Sherlock Holmes Inference（Heuristic）
- Specification Search Analysis（Model Selection）

### Methods Matter: P-Hacking and Causal Inference in Economics and Finance

- 引用：Brodeur et al. (2019)
- 作者：Abel Brodeur, Nikolai Cook, Anthony Heyes
- 年份：2019
- 载体/系列：Working Paper (University of Ottawa)
- 关键词：p-hacking, publication bias, causal inference, instrumental variables, difference-in-differences, credibility revolution, P-hacking, Publication bias, Causal inference, Instrumental variables, Difference-in-differences, Credibility revolution

**核心要点**
- P-hacking and selective reporting are significantly more prevalent in IV and DID studies than in RCT or RDD.
- The distribution of z-statistics for IV and DID exhibits a 'two-humped camel' shape with missing tests before z=1.65.
- IV papers are about 15% more likely to report results significant at the 5% level compared to RCTs.
- Approximately 15-20% of marginally significant results in IV papers are likely misleading or misallocated.
- P-hacking in IV research also occurs in the first stage, with F-statistics bunching just above the threshold of 10.
- RCT and RDD methods appear more 'trustworthy' with smoother, monotonically falling p-value distributions.
- IV and DID methods show significantly higher levels of p-hacking and publication bias than RCT and RDD.
- Instrumental Variables (IV) research is approximately 15% more likely to report misleading marginally significant results.
- The distribution of test statistics for IV and DID follows a 'two-humped camel shape' indicating selective reporting.
- RCT and RDD methods are found to be the most trustworthy for causal inference.
- Bunching at significance thresholds is 7 to 10% higher for IV than for RCT and RDD.
- Instrumental Variables (IV) and Difference-in-Differences (DID) show significant evidence of p-hacking.
- Randomized Control Trials (RCT) and Regression Discontinuity Design (RDD) are found to be more trustworthy.
- IV papers report approximately 55% more marginally significant (one-star) results than RCT papers.
- The distribution of published test statistics exhibits a 'two-humped camel shape' due to selective reporting.
- Approximately 15% of marginally significant results in IV papers are potentially misleading.
- Reliability of statistical claims is highly sensitive to the econometric method employed.

**该文献提供/强调的方法**
- Caliper Test（Local Discontinuity Test
Statistical Significance Test）
- Caliper tests
- Counterfactual Distribution Analysis（Distributional Analysis）
- Counterfactual Distribution Comparison（Excess Mass Estimation）
- F-statistic Distribution Analysis（First-stage Diagnostics）
- Inverse-weighting schemes
- Kernel density estimation
- P-curve analysis
- Probit Regression Analysis（Statistical Significance Likelihood）
- Probit/Logit Regression Analysis（Econometric Modeling）
- Z-curve Visualization（Distributional Analysis）

### Identification of and correction for publication bias

- 引用：Andrews and Kasy (2017)
- 作者：Isaiah Andrews, Maximilian Kasy
- 年份：2017
- 载体/系列：Working Paper (Harvard/MIT)
- 关键词：Publication bias, Replication, Meta-studies, Identification, Selective publication, P-hacking, PUBLICATION BIAS, REPLICATION, META-STUDIES, IDENTIFICATION, SELECTIVE REPORTING, SELECTIVE PUBLICATION, GMM, UNBIASED ESTIMATION

**核心要点**
- Selective publication based on statistical significance leads to biased estimates and distorted inference.
- Publication probabilities are nonparametrically identified up to scale using replication data or meta-study variance variation.
- Significant results at the 5% level are estimated to be over 30 times more likely to be published than insignificant ones.
- The paper provides median-unbiased estimators and confidence sets that correct for known selection probabilities.
- Asymmetries in the joint distribution of initial and replication estimates provide evidence of selective publication.
- Publication probabilities can be identified nonparametrically using replication or meta-study data.
- Significant results are 30 to 100 times more likely to be published than insignificant ones in some fields.
- Correcting for bias significantly reduces the number of findings considered statistically significant.
- Replication and meta-study approaches yield consistent estimates of the degree of selection.
- Point estimates and confidence intervals can be adjusted once publication probabilities are known.
- Publication bias leads to severe inflation of reported effect sizes and distorted inference.
- Identifies publication probability as a function of results using replication or meta-study data.
- Significant results (p < 0.05) are estimated to be over 30 times more likely to be published than insignificant ones.
- Publication bias is identified up to a constant scale factor.
- Proposes median-unbiased estimators that remain valid for scalar parameters under selection.
- Non-linearity in meta-regressions makes their slopes unreliable for bias correction.
- Minimum wage and de-worming literatures show evidence of selective publication based on significance.
- Finds that results significant at the 5% level are 30 times more likely to be published than insignificant ones in some literatures.
- Proposes estimators that remain valid even when the selection rule is estimated from the data.
- Demonstrates that selective publication leads to severe bias in published estimates and confidence sets.
- Provides a framework for Bayesian inference where the impact of selection depends on the choice of prior.
- Shows that selectivity can sometimes be rationalized if journals aim to minimize policy-related losses.

**该文献提供/强调的方法**
- Bonferroni-corrected confidence intervals
- Bonferroni-corrected Confidence Sets（Inference）
- Distributional symmetry testing
- Latent Selection Specification Test（Specification Test）
- Median-unbiased estimation（Bias Correction
Inference Correction）
- Meta-regression analysis
- Meta-regression Test（Diagnostics）
- Meta-study Approach（Identification）
- Meta-study Variance Approach（Identification）
- Meta-study-based Identification（Nonparametric Identification）
- Moment-based (GMM) Estimation（Estimation）
- Nonparametric identification via meta-studies
- Nonparametric identification via replication studies
- Optimal Quantile-Unbiased Estimation（Correction）
- Parametric Step-Function Model（Estimation）
- Quantile-unbiased estimation
- Replication-based Identification（Nonparametric Identification）
- Step Function Selection Model（Parametric Estimation）
- Systematic Replication Approach（Nonparametric Identification）
- Truncated likelihood approach
- Truncated Maximum Likelihood（Estimation）

### Presidential Address: The Scientific Outlook in Financial Economics

- 引用：Harvey (2017)
- 作者：Campbell R. Harvey
- 年份：2017
- 载体/系列：Journal of Finance
- 链接：https://doi.org/10.1111/jofi.12530
- 关键词：p-hacking, publication bias, Bayesian inference, false discovery rate, multiple testing, p-values, Minimum Bayes Factor, Bayesianized p-values, financial economics, Bayes factor, false positives

**核心要点**
- Standard p-value interpretation is often flawed and leads to an embarrassing number of false positives.
- Publication bias and p-hacking are driven by incentives to produce significant results for top journals.
- A t-statistic greater than 3.0 is necessary but often insufficient for significance when effects are rare.
- Prior beliefs must be incorporated using the minimum Bayes factor to evaluate hypothesis plausibility.
- The expected fraction of false discoveries depends heavily on the ex ante odds of a true effect.
- Standard p-values are often misinterpreted and fail to account for multiple testing and data mining.
- Raising significance thresholds to t > 3.0 is a necessary first step but still may not be sufficient.
- The Minimum Bayes Factor provides a bridge between frequentist p-values and posterior probabilities.
- Publication bias is systemic, driven by journal incentives for high-impact, 'significant' results.
- Economic plausibility and prior beliefs must be formally incorporated into statistical inference.
- The field needs a cultural shift toward transparency, replication, and the publication of negative results.
- Competition for journal space incentivizes p-hacking and selective reporting.
- Conventional p-value thresholds are insufficient in the presence of multiple testing.
- Publication bias is evident in the lack of published results with t-statistics below 2.0.
- Minimum Bayes Factors provide a simple, transparent alternative to standard p-values.
- Thresholds for significance should depend on the economic plausibility of the hypothesis.
- The financial economics field requires a more robust and transparent research culture.

**该文献提供/强调的方法**
- Adjustment for multiple tests（Statistical Rigor）
- Bayesianized p-value（Bayesian Inference）
- Bayesianized p-values（Probability Adjustment）
- Economic plausibility weighting（Prior Probability Assessment）
- False discovery rate calibration
- Increased t-statistic thresholds（Multiple Testing Adjustment）
- Meta-analysis
- Meta-analysis of t-statistic distributions（Publication Bias Detection）
- Minimum Bayes Factor
- Minimum Bayes Factor (MBF)（Bayesian Inference）
- Multiple test adjustment
- P-curve analysis
- Registered Reports（Research Protocol）
- Symmetric-Descending MBF (SD-MBF)（Bayesian Inference）
- t-statistic Distribution Meta-analysis（Publication Bias Detection）
- T-statistic threshold adjustment
- t-statistic Thresholding (t > 3.0)（Multiple Testing Adjustment）

### Specification Searches: Ad Hoc Inference with Nonexperimental Data

- 引用：(Leamer, 1978)
- 作者：Edward E. Leamer
- 年份：1978
- 载体/系列：John Wiley & Sons
- 关键词：Specification searches, Metastatistics, Data mining, Nonexperimental inference, Bayesian econometrics, Specimetrics, Specification Searches, Bayesian Inference, Subjective Probability, Coherence, Sufficient Statistics, Nonexperimental Data, Adjusted R-squared, Model Selection, Curve décolletage, Hypothesis Testing, Lindley's Paradox, Model Averaging, specification search, metastatistics, specimetrics, pretest estimator, multicollinearity, sensitivity analysis, Information Contract Curve, Pretesting, Principal Component Regression, Specification Search, Multicollinearity, Sensitivity Analysis, Bayesian inference, Global sensitivity analysis, Collinearity, Doubtful variables, Identifiability, simplification search, decision theory, parsimony, prediction, control, Proxy Variables, Reverse Regression, Simplification Analysis, Errors-in-Variables, specification searches, errors-in-variables, proxy variables, instrumental variables, nonspherical disturbances, outliers, data selection, Sherlock Holmes Inference, Bayesian Pooling, Data-instigated Models, Time-varying Parameters, Sherlock Holmes inference, Stopping rules, Specification search, Contamination vector, Implicit priors, Access-Biased Memory, Nonexperimental data, Revealed priors, Data-grubbing

**核心要点**
- Traditional statistical theory is invalidated by specification searches because it assumes the model is fixed and given.
- The Axiom of Correct Specification is routinely rejected in practice by researchers who perform multiple regression trials.
- Six distinct varieties of specification searches are identified, each driven by different researcher motives and judgments.
- Bayesian methods provide a flexible framework to make specification searches legitimate by incorporating prior information.
- Data-instigated modeling (Sherlock Holmes inference) is necessary in non-experimental science but complicates statistical discrimination.
- Metastatistics bridges the gap between 'priestly' statistical theory and 'sinner' data analysis practice.
- Real learning involves three distinct phases: summarization, interpretation, and decision.
- The 'sin' of specification searching is often unavoidable or desirable, but it is a sin not to know why you are searching.
- Subjective probabilities (degrees of belief) must obey the probability axioms to ensure coherence in decision making.
- Standard frequency definitions of probability fail to address non-repetitive or individual events.
- Sherlock Holmes inference (postdata model construction) differs fundamentally from traditional statistical inference.
- Distinguishes between ideal statistical theory and 'metastatistics' reflecting actual researcher behavior.
- Proposes that inference depends on both a summarization phase (data) and an interpretation phase (prior).
- Conflict between prior and sample evidence can increase uncertainty, represented by larger posterior variance.
- Argues that 'unavoided sins' in data analysis should be analyzed rather than ignored by theorists.
- Introduces 'Sherlock Holmes' inference as a fundamentally different process from standard statistical testing.
- Critiques noninformative priors, noting they can lead to undesirable outcomes like rejecting all complex models.
- Metastatistics analyzes how researcher motives influence choice of model and data.
- Traditional statistical theory is often irrelevant to nonexperimental econometric practice.
- Adjusted R-squared increases by omitting a variable if its t-statistic squared is less than one.
- Bayesian inference is formally equivalent to pooling sample info with non-sample info.
- Standard conjugate priors treat prior info like a previous sample, preventing multimodality.
- Student priors allow the posterior to reflect conflict between data and prior opinions.
- Classical hypothesis testing at fixed significance levels is flawed because large samples lead to the rejection of almost any null hypothesis.
- Meaningful hypothesis testing requires the significance level to be a decreasing function of the sample size.
- The 'curve décolletage' geometrically represents the locus of posterior modes preferred jointly by the prior and the data.
- Bayesian weighted likelihoods (marginal likelihoods) provide a superior alternative to classical likelihood ratios by averaging over parameter spaces.
- The researcher is obligated to report the mapping of various priors into posteriors to avoid making arbitrary model choices for the reader.
- Fixed significance levels (e.g., 0.05) lead to excessive rejection of null hypotheses as sample size grows.
- Specification searches require penalizing models based on both complexity (parameters) and sample size.
- Total uncertainty about parameters must include the variability of estimates across different model specifications.
- Researcher motives and 'sins' of data mining should be formally analyzed via metastatistics.
- The marginal likelihood is a better indicator of model performance than the maximum likelihood or R-squared.
- A 'grand R-squared' can be used to report the overall effectiveness of a multi-model research program.
- Metastatistics analyzes how researcher motives and opinions influence model and data choice.
- Specification searches involve mining data to find 'acceptable' results, often biasing inferences.
- The 'Grand R-squared' penalizes searches by incorporating the performance of all considered models.
- Classical pretesting often leads to inadmissible estimators and uncontrolled bias.
- Multicollinearity creates an incentive to use prior information to constrain unstable parameters.
- Bayesian sensitivity analysis is the formal tool to police intuitive 'ad hockery' in data mining.
- Interpretive searches are covert methods of introducing uncertain prior information into data analysis.
- The information contract curve identifies the set of all possible posterior modes for a given isodensity structure.
- Principal component regression is equivalent to a Bayesian search with a spherical prior.
- Pretesting is often misleading as it implements prior information without explicit specification or level justification.
- Orthant preservation ensures that omitting the least significant variables will not flip the signs of remaining coefficients.
- Data analysis involves three phases: summarization (sufficient statistics), learning (Bayes' rule), and decision making (loss functions).
- Collinearity is fundamentally a problem of interpreting multidimensional evidence rather than just weak data.
- Traditional variable selection often implicitly assumes specific non-elliptical prior structures.
- The posterior mode can be viewed as a weighted average of restricted estimates with weights related to F-statistics.
- Information contract curves visualize the sensitivity of estimates to the trade-off between prior and data.
- A coefficient suffers from collinearity if its true contract curve deviates from the diagonalized path.
- Conical priors better represent beliefs about similar coefficient magnitudes than standard elliptical priors.
- Traditional econometrics ignores how researcher motives influence model selection, a field termed 'metastatistics'.
- Collinearity is fundamentally a problem of how sample evidence interacts with personal prior information.
- Data evidence in multiparameter models cannot be reliably interpreted in a parameter-by-parameter fashion.
- Fragility in estimates often arises from choosing between 'doubtful' variables without exhaustive searching.
- A model is identified if and only if the experiment leads to a consensus among observers with different priors.
- Bayesian sensitivity analysis can provide bounds for coefficients even when the prior is not fully specified.
- Simplification is a decision problem distinct from the inferential process.
- The 'Man is simple' hypothesis justifies parsimony based on human cognitive limits.
- Statistical significance must be distinguished from economic significance in model choice.
- Classical tests implicitly assume specific processes for generating explanatory variables.
- Causal constraints prevent the distortion of structural parameters during simplification.
- Econometric practice involves 'unavoidable sins' of model searching that theory must address.
- A simplification necessarily decreases prediction accuracy unless the restriction holds exactly.
- In errors-in-variables, the true coefficient is bracketed by direct and reverse regressions.
- Statistical inference differs fundamentally from 'Sherlock Holmes' or data-instigated inference.
- Classical significance tests for simplification fail by ignoring the sample size factor T^-1.
- Metastatistics analyzes how data-generating mechanisms are influenced by researcher choice.
- Metastatistics analyzes how researcher motives and model choice influence statistical inferences.
- Parameter estimates in errors-in-variables models are bounded between direct and reverse regressions.
- The instrumental variables estimate is the ML estimate only if it falls within the regression bounds.
- Inclusion of error-ridden proxies often yields estimates biased toward the omitted-variable case.
- Marginalization of likelihood functions is essential to avoid inconsistent joint-mode estimates in incidental parameter models.
- Maximum likelihood estimates in errors-in-variables models are bounded by direct and reverse regressions.
- Data-selection searches overstate the precision of evidence by using data to pick the model.
- Generalized least squares acts as a matrix-weighted average of OLS and first-difference estimates.
- A poor proxy variable can be worse than none without proper prior information.
- Pooling evidence across equations requires both correlated errors and specific prior information.
- Traditional theory assumes the model is given, but nonexperimental inference requires 'specimetrics' to describe specification choice.
- Sherlock Holmes inference involves weaving evidence into a story, which is essential for science but invalidates classical p-values.
- Pooling evidence across 'similar' processes is necessary to avoid concluding rare events are certainties (clairvoyant paradox).
- Data-instigated models risk double-counting evidence; researchers should formally discount findings when models are found via searching.
- Structural change justifies discounting older observations more effectively than simple heteroscedastic weights.
- Metastatistics analyzes how a researcher's motives and opinions influence the choice of model and data.
- Evidence from data-instigated models (Sherlock Holmes inference) must be formally discounted to avoid double-counting.
- Stopping rules that depend only on data are noninformative for Bayesian inference; they do not invalidate the likelihood.
- A contamination vector (beta^c) should be added to models to represent the quality of the experimental control.
- Posterior uncertainty has a lower bound determined by the prior variance of experimental bias, regardless of sample size.
- Adding new variables is only legitimate if the new priors are consistent with the implicit priors of the simplified model.
- High R-squared values can be misleading indicators of model validity in the presence of misspecification.
- Specification searches are unavoidable in nonexperimental science but require formal 'metastatistical' rules.
- Explaining results post-hoc often involves selective memory that requires a statistical discount factor.
- The order in which variables are added to a model reveals researcher priors and affects data interpretation.
- Inferences should be pushed away from values favored by selective memory to minimize Bayes risk.
- Psychological biases like overconfidence and the 'law of small numbers' lead to systematic errors in judgment.
- Scientific consensus often precedes certainty and can be driven by advocacy rather than information.
- Traditional theory fails because it assumes a fixed model, ignoring specification searches.
- Nonexperimental inference is heavily influenced by the researcher's choice of specifications.
- The 'sin' in practice is not searching for models, but failing to report why and how.
- Bayesian methods help bridge the gap between statistical theory and empirical practice.
- Distinguishes between ideal statistical inference and actual 'metastatistics'.
- Identifies specification searching as a primary source of overconfidence.
- Proposes that researchers should confess the motives behind model selection.
- Argues that data-instigated models require different logic than confirmatory ones.
- Introduces the 'priesthood vs. sinners' gap in econometric practice.
- Suggests sensitivity analysis to reveal how results depend on assumptions.

**该文献提供/强调的方法**
- Access-Biased Memory Model（Cognitive Bias Modeling）
- Adjusted R-squared Selection（Model Selection）
- Advocacy Ability Scoring（Social Process Analysis）
- Bayes Factor Comparison
- Bayes Factor for Model Selection（Model Comparison）
- Bayesian decision-theoretic simplification
- Bayesian inference
- Bayesian Inference (Three-Phase)（Statistical Learning）
- Bayesian Model Averaging (BMA)（Model Uncertainty）
- Bayesian Normal-Gamma Updating（Parameter Estimation）
- Bayesian Pooling (Shrinkage)（Metastatistics / Pooling Search）
- Bayesian Pooling of Samples
- Bayesian Posterior Mean Estimation（Decision Theory）
- Causally constrained conditional prediction
- Coherence Principle (Betting Odds)（Decision Theory）
- Collinearity Measure (c1i)（Diagnostic）
- Conically Uniform Priors（Bayesian Inference）
- Conjugate Normal-Gamma Prior（Bayesian Inference）
- Consensus Analysis（Identification）
- Contamination Parameter Analysis（Variable Selection）
- Contamination Vector (Experimental Bias) Adjustment（Model Specification）
- Control Rule Shrinkage（Decision Theory）
- Control-simplification analysis
- Curve Décolletage (Dickey's Curve)
- Curve Décolletage (Sensitivity Analysis)（Fragility Assessment）
- Data-selection search
- Direct Regression Estimate
- Discount Factor for Memory Access（Statistical Correction）
- Double-Student Posterior Distribution
- Ellipsoid Bound Theorem（Bayesian Bounds）
- Error-Components Model（Specification Search）
- Errors-in-Variables Bound
- Exchangeable Normal Process Prior（Prior Specification）
- Factor Analysis
- Generalized Least Squares
- Global Sensitivity Analysis（Sensitivity Analysis）
- Global/Local Sensitivity Analysis（Robustness Testing）
- Hyperbolically Uniform Search（Specification Search）
- Hypothesis-testing search
- Implicit Prior Constraints（Bayesian Inference）
- Incentive Measure (c2)（Diagnostic）
- Information Contract Curve（Metastatistics
Bayesian Diagnostics
Sensitivity Analysis）
- Instrumental Variables
- Instrumental Variables Estimator
- Interpretive search
- Interpretive Search Analysis（Specification Search）
- Interpretive Search Rules（Meta-statistics）
- Interpretive Searches（Specification Search）
- Likelihood Concentration/Marginalization（Identification Analysis）
- Lindley's Paradox Analysis（Diagnostic）
- Lindley’s conditional prediction analysis
- Marginal Likelihood Comparison（Specification Search Detection）
- Marginal Likelihood Integration
- Matrix Weighted Averages（Estimation Methodology）
- Matrix-weighted Average
- Matrix-Weighted Average Bounds（Parameter Interpretation）
- Matrix-Weighted Averaging（Estimation）
- Maximum Likelihood Estimation
- Metastatistical Discounting（Metastatistics）
- Metastatistics（Theoretical Framework
Theory of Inference
Research Methodology）
- Metastatistics / Specimetrics（Theoretical Framework）
- Metastatistics/Specimetrics（Theoretical Framework）
- Noninformative Stopping Rules Analysis（Sampling Theory）
- Normal-Wishart Multivariate Sampling
- Orthant Preservation Analysis（Robustness Check）
- Overall R-squared (Grand R^2)（Model Selection Reporting）
- Postdata model construction
- Presimplification Analysis（Model Specification）
- Pretest Estimation Correction（Sequential Testing）
- Pretest Estimator Analysis（Specification Search Evaluation）
- Principal Component Regression (Interpretive)（Specification Search）
- Proxy search（Specification Search）
- Proxy Search Evaluation（Variable Selection）
- Proxy Variable Search
- Rectangle Test（Sensitivity Analysis）
- Revealed Prior Analysis（Researcher Bias）
- Reverse Regression（Measurement Error Detection）
- Reverse Regression Estimate
- Ridge Regression（Regularization）
- Rotation Invariant Average Regressions（Model Averaging）
- Sample Size-Dependent Significance Levels（Testing Correction）
- SEARCH (Extreme Hypotheses)（Specification Search Analysis）
- Seemingly Unrelated Regressions
- Sensitivity Analysis（Robustness Testing
Robustness Check）
- Sequential t-statistic Omission（Specification Search Strategy）
- Sherlock Holmes inference（Data-instigated Modeling
Inference Type
Hypothesis Generation）
- Sherlock Holmes Inference (Data-Instigated)（Post-hoc Modeling）
- Sherlock Holmes Inference (Post-data Model Construction)（Specification Search）
- Simplification Analysis（Model Selection）
- Simplification search
- Simplification searches
- Specification Search Taxonomy（Error Identification）
- Specimetrics（Model Selection Analysis
Specification Search Analysis
Methodology）
- Stable Estimation (Diffuse Priors)（Prior Specification）
- Standardized beta coefficients
- Stein-James Estimator（Shrinkage Estimation）
- Structural Likelihood Maximization
- Student Prior Bayesian Analysis（Bayesian Inference）
- Student-Gamma Prior
- Sufficient Statistic Summarization（Data Reduction）
- Supporting Hyperplanes Method（Likelihood Inference）
- Sweeping out the Means
- Theil's R-bar-squared Criterion（Specification Search）
- Time-Varying Parameter Model（Specification Search）

### Star Wars: The Empirics Strike Back

- 引用：Brodeur et al. (2013)
- 作者：Abel Brodeur, Mathias Lé, Marc Sangnier, Yanos Zylberberg
- 年份：2013
- 关键词：inflation bias, publication bias, p-hacking, z-statistics, research incentives, economics journals, Benford's Law, selection bias

**核心要点**
- The distribution of p-values in top economics journals exhibits a 'camel shape' indicating significant inflation bias.
- A valley exists between p-values of 0.10 and 0.25, while a corresponding bump appears just below 0.05.
- Approximately 10% to 20% of marginally significant tests are estimated to be the result of inflation or specification searching.
- Inflation is more pronounced among younger, non-tenured authors and in single-authored papers due to publication incentives.
- The use of 'eye-catchers' like stars is highly correlated with the observed distortion in test statistics.
- Economic journals exhibit a 'camel shape' p-value distribution with a valley between 0.10 and 0.25.
- A significant 'bump' in tests occurs just below the 0.05 significance threshold.
- Researchers inflate results by choosing specifications that turn insignificant tests into significant ones.
- Inflation accounts for 10% to 20% of marginally rejected tests in top journals.
- Selection bias and researcher inflation are distinct mechanisms distorting the empirical literature.

**该文献提供/强调的方法**
- Analysis of z-statistics distribution
- Benford's Law test on digits
- Benford’s Law tests on coefficients and standard errors
- Calculation of misallocation residuals
- Comparison with theoretical distributions
- Counterfactual distribution comparison
- De-rounding of reported statistics
- Decomposition by article characteristics
- Identification of p-value camel shape
- Kernel density estimation
- Non-parametric estimation of selection and inflation
- Non-parametric PAVA estimation
- Parametric estimation of selection and inflation
- Parametric selection function modeling
- Regression analysis for threshold discontinuities
- Sub-sample heterogeneity analysis
- Weighted distribution analysis
- Z-statistics distribution analysis

## 参考文献（APA）

- Harvey, C. R., Liu, Y., & Zhu, H. (2014). ... and the Cross-Section of Expected Returns (Working Paper No. 20592). National Bureau of Economic Research.
- Elliott, G., Kudrin, N., & Wüthrich, K. (2021). Detecting p-hacking. Manuscript, University of California, San Diego.
- Leamer, E. E. (1983). Let's Take the Con out of Econometrics. The American Economic Review, 73(1), 31-43.
- Brodeur, A., Cook, N., & Heyes, A. (2019). Methods Matter: P-Hacking and Causal Inference in Economics and Finance. University of Ottawa, Department of Economics Working Paper.
- Andrews, I., & Kasy, M. (2017). Identification of and correction for publication bias. MIT and Harvard University Working Paper.
- Harvey, C. R. (2017). Presidential Address: The Scientific Outlook in Financial Economics. The Journal of Finance, 72(4), 1399-1440.
- Leamer, E. E. (1978). Specification Searches: Ad Hoc Inference with Nonexperimental Data. John Wiley & Sons.
- Brodeur, A., Lé, M., Sangnier, M., & Zylberberg, Y. (2013). Star Wars: The Empirics Strike Back. IZA Discussion Paper No. 7268.
