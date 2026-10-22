# Customer Satisfaction Prediction — Olist E-commerce (Capstone)

Binary classification of order-level customer satisfaction for the Olist Brazilian
marketplace, scored **at the point of delivery** so operations can intervene before a
review is ever left.

**Stack:** Python · Pandas · scikit-learn · LightGBM · XGBoost · SHAP · LIME

---

## Problem

Reviews arrive only after an order closes, and only a minority of customers leave one —
so Olist cannot use reviews to save the order they describe. The goal was a model that
flags at-risk orders while they are still in transit, using delivery timing, seller
history and order attributes as predictors.

## Dataset

- Olist Brazilian E-commerce public dataset (Kaggle), Sep 2016 – Aug 2018
- **9 relational tables** merged into **~95,823 order-level rows**
- Target: `satisfied` (binary, derived from review score) — **79% / 21%** class split
- Grain handling: items and payments aggregated to order level before joining to avoid
  row multiplication and target leakage

## Approach

1. **Structural cleaning** — nine tables collapsed to one row per `order_id`, primary item
   selection, payment aggregation, timestamp and type fixes.
2. **Leakage-safe feature engineering** — delivery delay, absolute delivery time, freight
   ratio, seller history, product category, customer state; any post-review signal removed.
3. **EDA & baseline** — logistic regression benchmark, class-imbalance handling.
4. **Model comparison** — Logistic Regression, Random Forest, LightGBM, XGBoost with
   stratified cross-validation and a bias–variance check.
5. **Tuning & threshold selection** — recall-oriented threshold on the minority class.
6. **Explainability** — SHAP (global + local) and LIME for business-readable drivers.

## Results

| Metric | Value |
|---|---|
| ROC-AUC (test) | **0.715** |
| Recall on dissatisfied orders (tuned threshold) | **72.5%** |
| Dataset size | ~95,823 orders |
| Class balance | 79% satisfied / 21% dissatisfied |

**Key drivers:** delivery delay, absolute delivery time and seller history.

**Recommendation:** a live "at-risk order" queue for in-transit orders trending late,
plus seller-level monitoring for sellers repeatedly associated with dissatisfaction.

## Files

| File | Description |
|---|---|
| `data_cleaning.ipynb` | Notebook 1 — merge 9 tables, build the order-level dataset |
| `model_comparison_and_optimisation.ipynb` | Notebook 3 — model comparison, tuning, SHAP/LIME |
| `final_report.pdf` | Full capstone report (problem, scope, risks, results, recommendations) |

> The cleaned dataset (`olist_clean.csv`, ~42 MB) is not committed. Rebuild it by running
> Notebook 1 against the Kaggle dataset `olistbr/brazilian-ecommerce`.

---

**Author:** Rajat Singh · [LinkedIn](https://www.linkedin.com/in/rajatsingh-dataanalyst) · [Portfolio](https://github.com/Rajat-Singh-Source/data-analytics-portfolio) · [Profile](https://github.com/Rajat-Singh-Source)

