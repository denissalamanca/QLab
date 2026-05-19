# **AI-First Quant Lab: AFML Architecture & Product Requirements Document (PRD)**

## **1\. System Overview & Core Philosophy**

**Product:** Fully Autonomous Quantitative Trading System implementing the Lopez de Prado (AFML) "Meta-Labeling" framework, governed by a Human-in-the-Loop (HITL) Control Plane.

**Target Environments:** ICMarkets EU (Proprietary $100K, ESMA Regulated) & FTMO (Prop Firm Evaluation & Funded Accounts).

**Asset Universe:** 14 highly liquid assets (FX, Indices, Metals, Crypto) utilizing multi-year Dukascopy asynchronous tick data (.parquet).

**Trading Frequency:** Mid-Frequency (Holding periods: Minutes to Days), explicitly avoiding latency-arbitrage/HFT to maximize risk-adjusted capacity over $100K capital.

### **1.1. The AFML "Two Brains" Paradigm**

This lab rejects the standard ML approach of predicting exact future returns. Instead, we divide the problem:

* **Brain 1 (Primary Model):** A high-recall, structurally simple algorithmic layer that flags potential market anomalies or setups. Its baseline Sharpe ratio is irrelevant; its only job is to achieve high recall (not miss true positive opportunities).  
* **Brain 2 (Meta-Model):** A complex, non-linear machine learning classifier that acts as a precision filter. It assesses Brain 1's setups alongside complex microstructure features and predicts the *probability of success*, dictating position sizing (or vetoing the false positives).

### **1.2. Global Algorithmic Constraints (Anti-Bias & Anti-Overfitting)**

To prevent coding agents from taking lazy shortcuts or overfitting the 6-year dataset, the following global constraints apply to all phases:

* **Zero Hard-Coded Assumptions:** Absolutely no hard-coded parameters (e.g., window\_size \= 14, stop\_loss \= 20\_pips) are permitted. Every parameter, threshold, and multiplier must be dynamically derived from the underlying data using econometrics, market geometry, or robust statistical distributions (e.g., rolling exponentially weighted moments).  
* **Stable Optimality over Local Optima:** When agents conduct parameter sweeps, they are explicitly forbidden from selecting the hyperparameter combination that yields the absolute peak performance if it is an isolated "spike." Algorithms must be programmed to identify and select parameters from a **stable plateau** (a region where small changes in the parameter do not drastically degrade performance) to mathematically guarantee out-of-sample robustness and prevent curve-fitting.

## **2\. Global Success Criteria (KPIs)**

1. **Probability of Backtest Overfitting (PBO):** No strategy may be submitted to the CEO Control Plane if its computed PBO exceeds 5%.  
2. **Autonomous Execution with HITL Governance:** 24/5 operations where AI handles 100% of pipeline math, but 100% of capital deployments require cryptographic CEO sign-off.  
3. **Path-Dependent Risk Compliance:** Drawdown limits (5% Daily / 10% Max for FTMO) are deterministically managed at the *labeling phase* via Triple-Barrier constraints.

## **3\. Phase 0: Core Infrastructure & Agent Orchestration**

**Objective:** Architect the asynchronous messaging and state-management bus for the specialized AFML Agent pipeline.

* **Agent Roster:**  
  * **Agent 1:** Data Structuring & Stationarity Engineer  
  * **Agent 2:** Primary Alpha & Labeler (Brain 1\)  
  * **Agent 3:** Microstructure Feature Engineer  
  * **Agent 4:** Feature Selection & Orthogonalization Auditor  
  * **Agent 5:** Meta-Learning Architect (Brain 2\)  
  * **Agent 6:** Validation & Overfit Auditor  
  * **Agent 7:** Probabilistic Execution Trader  
  * **Agent 8:** Structural Break & Drift Monitor  
* **Technical Specs:**  
  * **Event-Driven Architecture:** Implement a robust distributed message broker capable of high-throughput serialization of complex ML artifacts without memory leaks or race conditions.  
  * **The Historical Alpha Registry (Hypothesis Database):** The state manager must maintain an immutable, centralized database logging *every single* alpha hypothesis tested by the lab. It must log hyperparameter search spaces, random seeds, and execution timestamps to cryptographically prove the total number of trials for the Deflated Sharpe Ratio calculation.  
* **DoD:** System sustains a load test of all 8 agents communicating non-blocking JSON/Binary payloads, successfully routing priority override commands from the CEO Control Plane in \< 50ms.

## **4\. Phase 1: Structural Data Engineering (Agent 1\)**

**Objective:** Ingest asynchronous ticks, form Information Bars, and enforce mathematical stationarity without destroying market memory.

* **Inputs:** Raw .parquet tick files.  
* **Outputs:** Information Bars, Fractionally Differentiated Price Series, Stationarity Validation Reports.  
* **Technical Specs:**  
  * **Information-Driven Sampling:** Agent 1 must compute Time Bars, Tick Bars (TIB), and Tick Run Bars (TRB). The agent must deterministically select the dataset by calculating the Jarque-Bera test statistic for the return distributions of each. The bar type that yields the lowest Jarque-Bera statistic (the distribution closest to normality) becomes the base dataset.  
  * **Fractional Differentiation (FFD):** Standard .diff() integer differentiation is strictly forbidden. Agent 1 *must* implement the **Fixed-Width Window Fractional Differencing (FFD)** method. Expanding windows are forbidden as they leak memory states. Calculate the optimal ![][image1]\-value that passes the Augmented Dickey-Fuller (ADF) test while retaining maximum Pearson correlation.  
* **DoD (Anti-Lazy Constraints & Truncation Hash Test):** \* Agent 1 must programmatically log a comprehensive parameter sweep for bar sampling thresholds, mathematically proving its final selection minimizes the Jarque-Bera statistic. The FFD output must pass the ADF test (p \< 0.05) strictly using a fixed-width window.  
  * **Truncation Hash Test:** Agent 1 must compute the FFD series on a full dataset (![][image2] to ![][image3]) and a truncated dataset (![][image2] to ![][image4]). The cryptographic hash of the overlapping rows (![][image2] to ![][image4]) from both runs *must match exactly* to mathematically prove zero future data leakage.

## **5\. Phase 2: Primary Signals & Event Labeling \- Brain 1 (Agent 2\)**

**Objective:** Generate candidate trade setups (events) and label their outcomes as an explicitly high-recall event-filter.

* **Inputs:** Information Bars, Historical Alpha Registry.  
* **Outputs:** Event Timestamps (![][image5]), Triple-Barrier Labeled Datasets.  
* **Technical Specs:**  
  * **Dual-Path Event Sampling:** Agent 2 must independently generate "Events" (timestamps ![][image5]) utilizing both a **Symmetric CUSUM** structural filter and native primary alpha entry logic. It must submit all resulting signal vectors to the Alpha Registry for independent downstream evaluation.  
  * **Orthogonality Check:** Reject any generated signal vector demonstrating a pairwise correlation ![][image6] with existing live/paper strategies in the Alpha Registry.  
  * **The Triple-Barrier Method:** Static stop-losses/take-profits (e.g., fixed pips) are strictly forbidden. Barriers *must* be dynamically calculated using a rolling exponentially weighted moving standard deviation (EWM Volatility) of returns.  
* **DoD (Anti-Lazy Constraints):** Agent 2 must log a minimum baseline number of orthogonal hypotheses in the Alpha Registry per research epoch. The Brain 1 model must exhibit a mathematically proven **Recall \> 0.70** and generate a **minimum of 500 distinct events/signals** across the historical dataset to guarantee sufficient statistical power for downstream CPCV. Precision and standalone Sharpe ratio are explicitly ignored.

## **6\. Phase 3: Microstructure Feature Engineering (Agent 3\)**

**Objective:** Extract highly predictive, stationary mathematical features strictly sampled at the event timestamps (![][image5]).

* **Inputs:** Information Bars, FFD Price Series, Event Timestamps (![][image5]).  
* **Outputs:** Dense Feature Matrices (features\_{asset}.parquet).  
* **Technical Specs:**  
  * **Event-Driven Computation:** Features must *only* be computed at the specific timestamps (![][image5]).  
  * **AFML Feature Suite:** Calculate AFML-prescribed institutional metrics including Kyle's Lambda, Amihud's Lambda, Hasbrouck Flow, Corwin-Schultz Bid-Ask spread estimators, and Lempel-Ziv complexity/Shannon entropy measurements.  
  * **Strict Causal Windows:** Look-ahead bias is fatal. All rolling window calculations must strictly exclude the current bar index ![][image5] from the calculation of the feature at ![][image5].  
* **DoD (Anti-Lazy Constraints & Truncation Hash Test):** \* Agent 3 must generate a high-dimensional baseline (e.g., 50+ computationally distinct metrics). Agent 3 is graded on the survival rate of its features during Phase 4\. All generated features must pass independent ADF stationarity checks with zero NaN or infinite values.  
  * **Truncation Hash Test:** As in Phase 1, all calculated rolling microstructure features must pass the cryptographic hash consistency test between a full and truncated data pass to prove absolute causality.

## **7\. Phase 4: Feature Selection & Orthogonalization (Agent 4\)**

**Objective:** Eradicate collinearity and noise by isolating only features that possess true predictive power, preventing the "substitution effect."

* **Inputs:** Dense Feature Matrices, Triple-Barrier Labels.  
* **Outputs:** Reduced & Orthogonalized Feature Matrices.  
* **Technical Specs:**  
  * **Optimal Number of Clusters (ONC):** Agent 4 *must* use the ONC algorithm to group collinear features, dynamically finding the optimal number of clusters by maximizing the clustering silhouette scores.  
  * **MDA over SFI:** Mean Decrease Impurity (MDI / Gini importance) is explicitly banned due to in-sample continuous variable bias. Agent 4 must apply **Clustered Mean Decrease Accuracy (MDA)** evaluated strictly through a Purged K-Fold Cross-Validation loop to prune non-predictive clusters. **Single Feature Importance (SFI)** must only be utilized as a secondary step if Clustered MDA fails to reduce the feature space dimensionality by at least 20%, in which case SFI is used to rank and prune within the surviving clusters.  
* **DoD (Anti-Lazy Constraints):** Agent 4 must output a mathematically reduced matrix where the retained orthogonal clusters demonstrate a statistically significant out-of-sample predictive power against Brain 1 labels. Evaluation via MDI/Gini triggers an automatic failure of the phase.

## **8\. Phase 5: Meta-Labeling & Probabilistic Modeling \- Brain 2 (Agent 5\)**

**Objective:** Train non-linear machine learning classifiers on the clean feature set, strictly enforcing Non-IID sampling.

* **Inputs:** Reduced Feature Matrices, Triple-Barrier Labels.  
* **Outputs:** Meta-Model Artifacts (Pickle/ONNX), Probability Calibrators.  
* **Technical Specs:**  
  * **Sample Uniqueness & Sequential Bootstrapping (CRITICAL):** Agent 5 must calculate the *Sample Uniqueness* array for every label based on overlapping trade durations. It must utilize *Sequential Bootstrapping* to draw training samples, severely down-weighting concurrent/clustered trades.  
  * **Meta-Model Selection Tournament:** Agent 5 must implement a **Sequentially Bootstrapped Random Forest (SBRF)** as the mandatory primary model. The agent must concurrently train a Gradient Boosting (XGBoost) model with identical sequential sample weights. The model that yields the lowest Negative Log-Loss on the purged out-of-sample fold is deterministically selected for deployment.  
  * **Probability Calibration:** Agent 5 must apply both Platt Scaling and Isotonic Regression to the outputs of the winning model. It must deterministically select the calibration method that minimizes the Brier Score of the predicted probabilities.  
* **DoD (Anti-Lazy Constraints & Index Intersection Test):** \* The meta-model must outperform a naive baseline. Evaluation must strictly rely on **Precision**, **F1-Score**, and **Brier Score / Negative Log-Loss**. Reliance on raw Accuracy or AUC-ROC as the primary metric is forbidden.  
  * **Index Intersection Test:** Agent 5 must execute an automated script verifying the integrity of the Purged and Embargoed splits. It must mathematically prove that ![][image7]. Any chronological overlap of a single millisecond results in an automatic quarantine.

## **9\. Phase 6: Rigorous AFML Validation (Agent 6\)**

**Objective:** Mathematically audit the ensembles to detect chronological leakage and multiple-testing overfitting.

* **Inputs:** Combined Model Outputs, Alpha Registry.  
* **Outputs:** Validation Dossiers, Approval/Rejection flags.  
* **Technical Specs:**  
  * **Combinatorially Purged Cross-Validation (CPCV):** Generate thousands of synthetic out-of-sample backtest paths utilizing purged and embargoed combinatorial splits.  
  * **Familywise Error Rate (FWER) & DSR:** Agent 6 must query the Alpha Registry to retrieve the true number of historical trials. It must compute the FWER and calculate the Deflated Sharpe Ratio (DSR) using the exact formula (Bailey and Lopez de Prado 2014\) to penalize the expected maximum Sharpe ratio based on the variance of all failed trials.  
* **DoD (Anti-Lazy Constraints & Target Shuffling Test):** \* Strategies with a PBO \> 5% or a DSR \< 1.0 are autonomously quarantined. Agent 6 must log the exact FWER penalty applied to prove the Multiple Testing problem was mitigated.  
  * **Target Shuffling / Lag Test:** Agent 6 must intentionally lag the target labels backward in time and retrain the evaluation. If the model retains significant predictive power on a nonsensical/lagged timeline, it mathematically proves feature data leakage exists, and the strategy is permanently rejected.

## **10\. Phase 7: Probabilistic Bet Sizing & Execution (Agent 7\)**

**Objective:** Execute CEO-approved models using dynamic, probability-weighted bet sizing, bound by concurrent exposure.

* **Inputs:** Live Ticks, CEO Tokens, Brain 2 Probabilities, Portfolio State.  
* **Outputs:** API Order Dispatches, Execution Logs.  
* **Technical Specs:**  
  * **AFML Bet Sizing (CRITICAL):** Arbitrary Kelly fractions are banned. Agent 7 must size bets by converting the predicted probability ![][image8] into a standardized Z-score, passing it through a **Standard Normal Cumulative Distribution Function (CDF)**. If the empirical distribution of historical probabilities fails a normality test (e.g., Shapiro-Wilk p \< 0.05), Agent 7 must dynamically switch to a **Mixture of Gaussians CDF**.  
  * **Concurrent Sizing Averaging:** The final bet size must be dynamically averaged against the maximum concurrent long vs. short positions expected in the market to guarantee absolute compliance with ESMA/FTMO margin/leverage limits.  
* **DoD:** Agent 7 accurately executes continuous bet sizes without ever breaching global portfolio concurrent exposure limits.

## **11\. Phase 8: Continuous MLOps & Structural Breaks (Agent 8\)**

**Objective:** Monitor the live market environment for paradigm shifts and degrade models before capital is lost.

* **Inputs:** Live Trade Data, Feature Streams.  
* **Outputs:** Retrain Triggers, Quarantine Flags.  
* **Technical Specs:**  
  * **Structural Break Detection:** Agent 8 must run the **Generalized Supremum Augmented Dickey-Fuller (GSADF)** test as the primary structural break monitor to autonomously detect explosive root behavior. The **Chow-Type Dickey-Fuller** test must be used strictly as a secondary confirmation if GSADF signals a regime shift.  
  * **Feature Drift:** Track the live Shapley (SHAP) values of the Meta-Model to flag Concept Drift.  
* **DoD:** The agent successfully triggers a model halt and alerts the CEO Dashboard immediately upon mathematical detection of a structural market break (via GSADF/Chow) or IC decay.

## **12\. Phase 9: The Control Plane (CEO Dashboard & Governance)**

**Objective:** Provide total visibility, manual override capability, and cryptographic sign-off queues across all phases.

* **Inputs:** Telemetry feeds, PBO/DSR Dossiers, Structural Break Alerts.  
* **Outputs:** Signed execution tokens, System Overrides.  
* **Technical Specs:**  
  * **The AFML Factory View:** Track Agent 1's memory-retention charts, Agent 4's ONC silhouette scores, and Agent 5's Sequential Bootstrapping uniqueness distributions.  
  * **Institutional Audit Queue:** Review Agent 6's CPCV backtest distributions, PBO metrics, FWER penalties, and Deflated Sharpe Ratios before manual sign-off.  
  * **Live Operations & Drift View:** Real-time monitoring of Brain 2's live probability generation vs. actual trade outcomes, and live GSADF charts.  
  * **Deterministic Failsafes:** Global Flatten, Brain Subjugation, and strategy-specific sandboxing.  
* **DoD:** The Control Plane functions as a zero-latency centralized command hub, ensuring the AI pipeline remains mathematically rigorous, regulatory-compliant, and strictly subordinate to CEO directives.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAaCAYAAABhJqYYAAAA10lEQVR4XmNgGEZATk5ug7y8/H8o/ocujw4YZWRkhKCKr6FLYgCYYqAtMehyGEBBQcEfpFhRUVEeXQ4OgAokofRZIF6ELg8GQOt8gZJHgfgvEBtB6RR0dSxAwbVAvAUmAGT/AjkBWRFM4gW6BJD/E10MJvEfXQKbGLLEQ3QxUMQgi4EAE1TxYpiAlpYWG1SsCIg1UcIZKjEPxgeG7y0g/zuQyQykX8rKytrCFQMlHUCCQBMeAel3UAP+yUM8bgRXiAQYgYpdgFHMCeKAohoYc2boikYB7QEAk7ZAvN8rxWcAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAaCAYAAAC3g3x9AAABR0lEQVR4Xu2Tu0oDQRSGE9OKILjNslfYxhuIVkbQUhQEHyB5Bq30LSx8ANsoNlr4GDaCtoLYi7EQvKDfyCycPTMmrlhY5IPD2fn/s39mZkmjMeJPyLJsPU3TjxrV1hkVkiQ5ZuiZOsjzfJW+TN2al/H26Iv0FXrPaGEYTumMCgy9erSv3Xj0ntYcGDr1aN5ArudcaxUYWKM1pRYEwbgN7EvdwNHvtDYUgjZt4IX2fgVBlyaQ3SxprzZcwbYNO9NeCX6buV1mZrTnwPChDdzRnmUM/wZ/n36vTQd7d+brVj5UCd61XLPTE7l2EIEOURTNa4/1A60lNUlzUCDH7GrPXs+s1Iz4Rj1RferRPr9Q70VRTJRz3wVS01L7Mby44Qsc+t8eBB/hSixbBB6JdX3MseM4nrPPHX5gQc/UhnsNCNuiJrU34h/yCYoHZXLVDGmgAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACMAAAAZCAYAAAC7OJeSAAAB00lEQVR4Xu2VzStEYRTGUYhslNnM173zkcUsKLOzUSxYSJLCBmWBBUo2VlbKYrJSkrJgIQtlxz8gWSN7LCx8ZEoJg9/L3DpzzBB3zELz1Ome85xz3vPc99733pKSIor4BWzbbrcs6/UH1qzXyBtY/N7n8/kV92gGR6PRSsWfyDjvYMBTFu59F7Lwm5rLG7xeb3UwGGzT/BdijjWXN/C+tHAplZzH46lJi0lK3gDhZ5r7UyAikRYzo3MFBXdei4gUdktYpvMFBY+t2+wKonZ0ToL8pObC4XA9/aP0t+ocfD89c7FYrELncoKFjoyYUCjUoHMGLDhIfsRSJ9DUw/UYn0NRh39n/Hg8Xo5/6dTh7zv+t0i/K59OkYYWQ3wtY0QvwzVh23I9dqiPeFrW5oQLMRk9xAkGD3A9kDlEdsKvyNoMkByiaIKmdUcM3Dg2rL/ADqh5VnGGGHoXsFn4K5mD62DWnqx1DQakVKx3ZpGhvdihzAUCgS7iJVnrGiz4ouJzFa/xUjdy3ZBizAHApmStK5hHl36UtsMxIEY4ZnzuPkL+wfh+v7/KEr8Q/FPHdw374ztyg91hSWzeyeHfYqvYhfwORSKRANyu2RH6txy+iCL+Fd4AaBmK3j3+t7sAAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABsAAAAaCAYAAABGiCfwAAABqElEQVR4Xu2UO0gDQRCGE6MINoqCSF6XS4Jo6zWKr0YbWwuxEC2sbLSys7G1ERQFKxvBR2lrbS8EY2cVxMoXCoagfit7MIynuUQbIT/87Mz/Z2d2NstFIg38SziOMwXfwzKVSuV0jdCgwIkttOG67hjrELyx2hIcSKfTI6ynRmNLTNcIDQqUM5lMn9I+p5Ca1c+0VhM49aHWfmh2rrXQYKJxrSUSiS7b7Fbqnue1oN1L7deg4LRpxkGOtPfnoNGlaZbNZnu196dgmll7hQfaM+D/ncRz+V0H8YTyhuGqfmzfgkJ7ttmi9gyEb/gqrBh5gUYLrCXWFeEFwy+kdR+cfNc8FK2zpyjSKPmzyINRrRnejtYM9B7ya5kHwZyoWrNtrmiTCfeJn4Sum11AV2q+UYEv8BE+mCKwDN+CrswH/jpcs/GXZnz6HKnViiY+wqN+wnTLTHls4oBmxXg83ia1mkDxOXMDfk685dhXy1rw9Xw+30pe8fO6kEwmOylyZWIm6iEuETbbfJ6838TmAPBObK0fFB5kSo8wKvVcLteNN8OXp13qDTQQCh+hgIBfhocLUgAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAcCAYAAAC6YTVCAAABAElEQVR4XmNgGAWYQF5eXhOIp6CL4wVADUeA+D+6OF4A1PCHHE3/FRQUbqGL4wUgTXJyctPRxTEAUKERUGE70IZNIE1A3A3kt6GrwwqAipuB+B66OE4AtIUDqOEnECehy4GAlJSUCJBiQhEEarIHOU1GRkYFRQIKgHKbZWVlldEF95AT1F+RNQFtzkSWxwqgofYdxNbS0mIDsmtAbGAopgGxElZXAAV/AfEfFRUVUaCiFcbGxqxQ8fNAW8OxagIBUVFRHgzPMoA1/gbiyejieAHIFkVFRTNgsHOhy+ECjEDndQKdzQdio0viBECbrgD9eRBdHC8AOk0e5F908VEABAA1tzilF1Fh0wAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAZCAYAAAB3oa15AAACQUlEQVR4Xu2VO2sUURiGd8VLIYaAmC32fpHFRRBZUQQlWIc0ERIr/4CVhWhhkcpKxEYRCXZBtLARFAu1EzRgp2K6iBKSqMQLSUww8fnwzGZ4s3Nms5puHniZ2fe7nPPNziWVSkhI2DKKxeJYqVSa5LhUKBReajyKWq3WQ801tIaeo6uas+Ww4QoLz6ChXC63n+ODTjdC3hSDv8rn81XOH7lBhjSvRbVa7aOgpH630OqMLaq+89Lqh6nX63vchpcCj/PH5tF3NJS6DsHTaLFcLh/VWDfQ62nUAFzVI+qHyWaze90ArXrOH7oBrodzN8BfnSVxAU2kYq6UD91AyF/m1nqivg93K1q/eRtO45FQcBnNoXMai8MzwFc29EH9CNJ2R1Czin5osCMymcxuij+x6EW7NzUehWeAz+in+u3gVjvp+iyw/qDGN8t2mpyl2XeOVzSoeAb4gubU9+Ee6tdoFh3Q+KagwbN2G1M8A8yjd+rHQc0F1/OFxmLho7KPwm/ofbPZ3KHxdngGMP+u+mGIH+ZfvmUvlMDjvBbVMxKSb6NfaEBjcVAz1m4x83gV9op3Q35Pu7z7IW/ADRB/+/HwHCRx3F53NOnXeCdQe4Iev3mLZMK+DuXWWuNf3hV4rHkP7w21hwLPhnQD3Ay8DVB4jIQVFm9orFuKf1+B9gCOo2XbsKRsw/8ontW9dbX2AbPjSqPR2Kl5Ldj8CEmLqKyxf8E+PFyQYfpfovdxjXtIk38K3aH2fKVSKWhCQkJCQsJ/4Q8KPqqlXESv5wAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,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>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAaCAYAAAC+aNwHAAAA60lEQVR4Xu2QPQrCUBCEo0S0sbE1f4RAxNIIongGj2IjeAEvoZ1nsPIqHkJEFLVQdEZegg5I0tjlg+W9zOzu26xllXzh+/4I8cyJhdZ9EQRBw/O8NZNxTpMkqUG2Hcdp43tu9K7WZURRVEfSDbFTj0B/sEkYhp56b9B9zARMslKPpL8Cv6PeGzTYMoEjq0fMLyxVz0DCiUmqEzY1zVvqZaQjuq7bx9lDDHGf4NxTj+O4qTWf2KbBgctkQKto0k+wmIFZ0Ea9QqQL5OjqFQKFZ46Pa1W9QpjXZ6rngqKLef2IuCLukG3NKyn5Gy+i0kdCGNYfHwAAAABJRU5ErkJggg==>