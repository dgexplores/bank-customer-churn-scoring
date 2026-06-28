# Bank Customer Churn Predictive Risk Scoring: Methodology, Modeling, and Retention Interventions

**Author:** Senior Machine Learning Engineer  
**Date:** June 2026  

---

## 1. Abstract
This paper details the development of a high-fidelity predictive risk-scoring system designed to preemptively identify retail bank customers at risk of churning. Using a dataset of 10,000 customers from a European bank (exhibiting a 20.37% baseline churn rate), we built a robust machine learning pipeline integrating domain-driven feature engineering, standard-scaled preprocessing, and class-imbalanced learning. We trained and compared Logistic Regression, Decision Trees, Random Forests, and Gradient Boosting models under 5-fold Stratified Cross-Validation. 

Our champion model, **Gradient Boosting**, achieved a **test ROC-AUC of 0.8677** (surpassing the target of 0.85). Rather than utilizing a default 0.5 classification threshold, we tuned the decision threshold to **0.6531** to optimize the F1-score, yielding a balanced F1 of **63.77%** and a test recall of **63.14%** (capturing 63% of actual churners while maintaining 64% precision). Explainability analysis via Gini Importance and SHAP (SHapley Additive exPlanations) revealed that **Customer Age** and the **Number of Bank Products** are the primary drivers of customer exits.

---

## 2. Exploratory Data Analysis (EDA)
An initial analysis of the bank portfolio dataset indicates a significant class imbalance:
* **Total Portfolio**: 10,000 customers
* **Retained (Class 0)**: 7,963 customers (79.63%)
* **Exited (Class 1)**: 2,037 customers (20.37%)

This 80/20 imbalance implies that a naive "always retain" classifier would achieve 80% accuracy but 0% recall, highlighting the need to optimize for ROC-AUC and F1-score rather than raw accuracy.

### Key Demographic and Financial Segment Findings:
1. **Geography**: German customers display a **32.4%** churn rate, whereas French and Spanish customers both display a significantly lower churn rate of **16.1%** and **16.7%** respectively. This suggests regional friction, competitor incentives, or service dissatisfaction in the German market.
2. **Gender**: Female customers churn at a rate of **25.1%**, compared to **16.5%** for Male customers, presenting a gender-specific stickiness gap.
3. **Age**: Active churners are significantly older on average. The median age of churned customers is **45 years**, compared to **36 years** for retained customers. 
4. **Number of Products**: Churn rate behaves non-linearly with product holdings:
   - **1 Product**: 27.7% churn rate.
   - **2 Products**: 7.6% churn rate (the customer "sweet spot").
   - **3 Products**: 82.7% churn rate (severe service risk).
   - **4 Products**: 93.3% churn rate (virtual guarantee of exit).
5. **Activity**: Active members exhibit a churn rate of **14.2%**, whereas inactive members exit at a rate of **26.9%**.

---

## 3. Methodology & Feature Engineering
To eliminate training-serving skew, we designed a unified preprocessing and feature engineering pipeline implemented in `churn_utils.py` and joblib-pickled.

### 3.1. Preprocessing ColumnTransformer
* **Non-informative columns dropped**: `Year`, `CustomerId`, `Surname`.
* **One-Hot Encoding**: Applied to `Geography` and `Gender`, dropping the first level to avoid multicollinearity. Set `handle_unknown="ignore"` to handle unseen values gracefully in production.
* **Standard Scaling**: All numeric features (raw and engineered) are standard-scaled (mean=0, variance=1) to ensure optimal convergence for gradient-based and distance-based estimators.

### 3.2. Derived Features
Seven domain-informed features were engineered:
1. **BalanceSalaryRatio** ($Balance / (EstimatedSalary + 1)$): Measures asset concentration relative to income. High ratios indicate customers with significant wealth parked at the bank who might seek higher-yield investments elsewhere.
2. **ProductDensity** ($NumOfProducts / (Tenure + 1)$): Standardizes product adoption over time. High product density early in tenure may signal service overload.
3. **EngagementProduct** ($IsActiveMember \times NumOfProducts$): Measures active product depth. High values represent the stickiest customers.
4. **AgeTenureInteraction** ($Age \times Tenure$): Captures loyalty progression adjusted for life-stage.
5. **TenureByAge** ($Tenure / (Age + 1)$): Represents the fraction of a customer's life spent with the bank.
6. **CreditPerAge** ($CreditScore / (Age + 1)$): Normalizes credit-worthiness by age.
7. **ZeroBalance** ($1$ if $Balance == 0$ else $0$): Flags inactive/empty accounts.

---

## 4. Model Comparison & Evaluation
We compared four key estimators, correcting for class imbalance by utilizing class-weight balancing (or sample weights for Gradient Boosting):

| Model | CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7685 | 69.75% | 37.34% | 71.74% | 49.12% | 0.7783 |
| **Decision Tree** | 0.7590 | 75.65% | 43.65% | 67.57% | 53.04% | 0.7470 |
| **Random Forest** | 0.8463 | 86.25% | 79.20% | 43.98% | 56.56% | 0.8502 |
| **Gradient Boosting** | **0.8612** | **80.35%** | **51.15%** | **76.41%** | **61.28%** | **0.8677** |

### Discussion:
* **Gradient Boosting** emerged as the champion model with a test **ROC-AUC of 0.8677**, demonstrating superior ability to distinguish churners from non-churners.
* **Random Forest** achieved the highest default accuracy (86.25%) and precision (79.20%), but suffered from low default recall (43.98%), leaving more than half of the churners undetected.
* **Logistic Regression** and **Decision Trees** served as interpretability baselines but lagged in ROC-AUC.

---

## 5. Tuned Decision Threshold Analysis
For business retention campaigns, raw accuracy is secondary to finding a balance between **precision** (avoiding wasting retention budgets on customers who won't churn) and **recall** (capturing as many actual churners as possible). 

Using the Precision-Recall curve on the Gradient Boosting probabilities, we tuned the decision threshold to maximize the F1-score. The optimal threshold was calculated at **0.6531**.

### Classification Metrics at Tuned Threshold (0.6531):
* **Accuracy**: 85.40% (improved from 80.35% default)
* **Precision**: 64.41% (improved from 51.15% default)
* **Recall**: 63.14% (down from 76.41% default, but more stable)
* **F1-Score**: 63.77% (improved from 61.28% default)

By raising the threshold from 0.5 to 0.6531, we significantly filtered out false positives (increasing precision by 13%), which ensures retention campaign budgets are directed to customers who are highly likely to exit.

---

## 6. Model Explainability
To unlock the black-box nature of the Gradient Boosting model, we analyzed global Gini feature importances and run local SHAP TreeExplainer simulations.

### 6.1. Top 5 Global Features:
1. **Age** (40.0% Importance): Age is the most dominant factor. Older customers have a much higher likelihood of churn, suggesting that the bank's digital transition or product offerings may not align with older demographics.
2. **NumOfProducts** (25.8% Importance): Having multiple products has a drastic, non-linear impact. Having exactly 2 products acts as a buffer against churn, while having 3 or 4 products represents extreme risk.
3. **Balance** (6.5% Importance): Higher balances are correlated with churn, suggesting wealthier customers are searching for better yield or are sensitive to fees.
4. **EngagementProduct** (6.0% Importance): Demonstrates that active members who hold multiple products remain sticky, highlighting the importance of engaging customers rather than just selling products.
5. **Geography_Germany** (5.3% Importance): Flags regional churn issues in Germany.

### 6.2. SHAP Analysis
SHAP values show that:
* Being located in Germany pushes risk scores upward.
* Being an inactive member pushes risk scores upward.
* Higher ages contribute positively to risk scores.
* Holding 3 or 4 products has a huge positive impact on risk, while holding 2 products pulls risk down significantly.

---

## 7. Business Recommendations & Conclusions
Based on the predictive model insights, we recommend three concrete retention initiatives:

### 1. The "Two-Product Sweet Spot" Campaign
* **Insight**: Customers with 2 products have the lowest churn rate (7.6%).
* **Action**: Create cross-sell incentives targeted at single-product customers to move them to exactly two products (e.g. cross-selling a credit card to an active checking account user). Conversely, avoid aggressively pushing a 3rd or 4th product without a personalized service call.

### 2. German Market Diagnostic & Retention
* **Insight**: German customers churn at double the baseline rate (32.4%).
* **Action**: Deploy a regional taskforce in Germany to analyze competitor pricing, local fee structures, and service quality. Implement local deposit promotions and high-touch account managers for high-value German clients.

### 3. Active Engagement Rewards
* **Insight**: Active members are twice as sticky as inactive members.
* **Action**: Launch mobile app engagement campaigns. Offer small financial rewards (e.g. waiver of monthly account fee) to inactive customers who log in at least 4 times a month or execute 5 debit card transactions.

---
