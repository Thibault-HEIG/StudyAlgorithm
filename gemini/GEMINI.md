# Project Context: Unified Student Performance Predictor

## Objective
Build a predictive algorithm to forecast student academic performance (grades) based on sleep, study time, and secondary lifestyle variables. 

## The Challenge
The project relies on 8 distinct, heterogeneous CSV datasets. They contain conflicting schemas, varying target variable scales (e.g., GPA out of 4.0, GPA out of 10.0, percentages out of 100), and different feature representations.

## Technical Environment
* **OS:** macOS
* **IDE:** VS Code
* **Stack:** Python (Pandas, Scikit-learn eventually), SQLite

## AI Assistant Directives (Learning Philosophy)
1.  **Explain the "Why".** Focus on the mathematical and logical reasons behind data transformations (e.g., why Min-Max scaling is used, why median imputation is safer for skewed data) before showing the syntax.
2.  **Be objective and brief.** Do not use filler. Point out flaws in data logic immediately.
3.  **Challenge assumptions.** Do not validate a merge or mathematical operation by default if it introduces data leakage or statistical noise.
