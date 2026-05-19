# **Institutional Audit Report: Quant Lab Phases 0-5 (Brain 2 & Full Integration)**

**Prepared for:** Claude Code (Development Agent)

**Author:** Lead Quant & Product Manager (Lopez de Prado Framework)

**Status:** Pre-Phase 6 Clearance (Contingent on Calibration & Evaluation Patches)

## **Executive Summary**

Your implementation of Phase 5 (Meta-Labeling) correctly translates the concepts of sample uniqueness and sequential bootstrapping into Python classes. You have successfully mapped ![][image1] to the sample\_weight parameters for the classifiers.

However, the AFML framework is relentlessly unforgiving of standard library defaults. A full-stack audit reveals **three silent leakage vulnerabilities** in the way Brain 2 calibrates probabilities and evaluates out-of-sample performance, as well as a required integration check with Phase 4\.

You are directed to refactor Phase 5 to patch these vulnerabilities before moving to Phase 6\.

## **1\. Meta-Labeling Leakage Vulnerabilities (Phase 5\)**

### **1.1. Vulnerability 1: The CalibratedClassifierCV Data Leakage Trap**

* **Location:** src/afml/modeling/calibration.py  
* **The Flaw:** By default, sklearn.calibration.CalibratedClassifierCV utilizes standard stratified K-Fold cross-validation (cv=5). If you pass your SBRF or XGBoost model to this wrapper without explicitly overriding the cv parameter, scikit-learn will randomly shuffle your non-IID financial data. It will calibrate probabilities using future overlapping data, generating massive look-ahead bias and artificially perfect Brier scores.  
* **The AFML Refactoring Directive:** 1\. You **must** pass an instantiated Phase 4 PurgedKFold object (with the exact ![][image2] path-dependency series) into the cv argument of CalibratedClassifierCV.  
  2\. Write an explicit unit test verifying that the PurgedKFold.split() method is successfully invoked during the .fit() cycle of the calibrator.

### **1.2. Vulnerability 2: Nested Purging in the Meta-Model Tournament**

* **Location:** src/afml/modeling/pipeline.py  
* **The Flaw:** The Master Blueprint requires a tournament between SBRF and XGBoost evaluated on a "purged out-of-sample fold" using Log-Loss/Brier Score. If you used train\_test\_split or a basic chronological slice (e.g., first 80% train, last 20% test) without applying an **Embargo**, the last training observations will leak path-dependent outcomes into the first testing observations.  
* **The AFML Refactoring Directive:**  
  1. The outer tournament evaluation must utilize a strict Chronological Split with an Embargo.  
  2. Calculate the maximum ![][image2] (end time) of all samples in the training set. The testing set must strictly drop any samples whose ![][image3] (start time) occurs before ![][image4].  
  3. SBRF and XGBoost must be evaluated *only* on this strictly quarantined out-of-sample testing set.

### **1.3. Vulnerability 3: SBRF vs XGBoost Sample Weight API Mismatches**

* **Location:** src/afml/modeling/sbrf.py & src/afml/modeling/pipeline.py  
* **The Flaw:** Standard RandomForestClassifier.fit() accepts sample\_weight. However, XGBoost handles sample weights differently depending on the API (xgboost.XGBClassifier vs xgb.train), and CalibratedClassifierCV has historical bugs passing fit\_params (like sample weights) down to the base estimators during inner CV loops.  
* **The AFML Refactoring Directive:**  
  1. Ensure that the average\_uniqueness vector (![][image1]) is successfully passed to the underlying fit methods of *both* the Random Forest and XGBoost estimators **inside** the CalibratedClassifierCV wrapper.  
  2. Use fit\_params={'sample\_weight': uniqueness\_weights} in the calibrator's fit method, and add an integration test that asserts the uniqueness weights actually influence the trained trees (e.g., by comparing predictions of a weighted vs. unweighted model).

## **2\. Integration Hardening (Phase 4 ![][image5] Phase 5\)**

### **2.1. The Empty Matrix Pass-Through**

* **The Flaw:** In the previous audit, we established an "Empty MDA Circuit Breaker" in Phase 4 if zero clusters demonstrate predictive power.  
* **The Refactoring Directive:** 1\. Verify Phase 5's entry point (modeling/pipeline.py) explicitly checks for this empty feature state.  
  2\. If the feature matrix is empty, Phase 5 must immediately exit, return a null/failed Meta-Model artifact, and ensure the Alpha Registry is updated with FAILED\_AT\_MDA. Do not allow the script to crash with a ValueError.

## **3\. Execution Orders & Phase 6 Clearance**

1. Inject PurgedKFold into all instances of CalibratedClassifierCV.  
2. Implement strict Embargo rules for the outer Brain 2 evaluation split.  
3. Validate that sample\_weight correctly propagates through the calibrator to both SBRF and XGBoost.  
4. Verify the Phase 4 to Phase 5 circuit-breaker handoff.  
5. **Proceed to Phase 6 (Rigorous AFML Validation).** \* *Phase 6 Pre-Brief:* You will now implement Combinatorially Purged Cross-Validation (CPCV). This is mathematically dense. Rely strictly on the Bailey and Lopez de Prado (2014) logic for generating the combinatorial paths and calculating the Familywise Error Rate (FWER) penalty for the Deflated Sharpe Ratio (DSR).

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAZCAYAAAA8CX6UAAABKUlEQVR4XmNgGAVDHMjLy/fiwNHoamkPZGRkOIFYFV2cJAB0+g0gbgXizXJyciFocv+BeB2yGE6goKBwCEgxAjVcB7I7kaRAYiCDipDEcAJGkLdADJAmFRUVdpgE0NBMoNg3Y2NjVoRyuFwFuhgcgAxC5gO9uQIodgBZDAqYgOIv0AXBACiRDcS/YHygIUpQb3kiqyMIgBq2AvE8JP48JBcyycrKmoIYQC/tAorfhKnDAEAXLAAq2APjA9kfYAYBNTsgiTsC1YYCDdaBiWEAoKIHQPwUiD8CNXMA6SNA/BLIXgiSV1JS4gcFPFDsDbpeDAC0zQZkCIwPcgGQYoTxgXL+SF4mHwAN+QEKBlCMosuRBIAuOgU05CAwjKTQ5UgGwLBSQxcbBYQBAFOZRHPF/CdsAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAaCAYAAAC+aNwHAAAA2klEQVR4XmNgGAXUBUpKSnKysrK26OJEA3l5+c9AXIIuTjQAav6vqKhohi5ONAAZgC5GECgoKJgDNd6A4v9A/i0QG10dQQDUVArEn9DFiQXMQM0fgbZXoEvAAFA+BV0MDoBRZwJyPhBbossBDS0Aim8D4mvocnAAlNxMKAAJGfAB2QCgrRHI8iBAyACQ88EGgFwjJyc3H4sa3AYAbcwAKvgGxO+lpaWF0eVBACh3HV0MBcjIyHACDdJAF4cBeXLSBjIAJTB0MaKBqKgoD9AFj4H5RBxdbhQMBgAAS2YwdKDrc3YAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAaCAYAAAC+aNwHAAABH0lEQVR4XmNgGAXUBUpKSnKysrK26OJEA3l5+c9AXIIuTjQAav6vqKhohi5ONAAZgC5GECgoKJgDNd6A4v9A/i0QG10dQQDUVArEn9DFiQXMQM0fgbZXoEsAASNQ3AEon6WsrCyLLgkGwKgzATkfiC3R5YBiy4D4jJycXBuQfo4uDwZAic04AhBk+0oYB2iRFMirSPIQABT8gGwAUFMEiAbaagwUn4RQycAC5O9G4kMA1PlgA0CuAWqcD2UnAXELmtrPyHwwANqYAZT4BsTvpaWlhWHiOAz4iMyHAxkZGU6gQRrIYkA/+wHFOpHFgAbcReYTBEANW2FsYGZTAxqYiSxPEIC8Bot/ILsV2YtEA6D3pIEB6yolJcWFLjcKqAAAJ3c+ERjrjv4AAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,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>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB0AAAAjCAYAAABo4wHSAAAAk0lEQVR4XmNgGAWjYBTQGsjJyYWhi9EcyMvLH1NQUEhHF6cpEBcX5wZafBddnOYA6FN7YDBbo4sTDRQVFdWBLtckAzcDLV4A0o9uJkEA1PyfQvwR3UyaAaBlz9DFaAZUVFTYgRa+QBenKQAmonigpWfRxWkKgBb+VlJS4kcXpykAWmqFLjYKRsEoGAWjYBSMghEAAOhgLWs6vF5qAAAAAElFTkSuQmCC>